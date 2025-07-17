import torch.optim as optim
from pytorch_lightning import LightningModule
from pytorch_lightning.callbacks import Callback
from omegaconf import DictConfig, ListConfig, OmegaConf
import torch
import torch.nn as nn
from typing import Union
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import logging
import numpy as np
import ray
from ray import tune
from ray import ray_constants
from copy import deepcopy
from pytorch_lightning import Trainer
from torch_ema import ExponentialMovingAverage
from typing import List
from tqdm import tqdm
from scipy.stats import pearsonr
import libauc.losses
import libauc.optimizers

from src.data import RealDatasetCollection, SyntheticDatasetCollection
from src.models.utils import grad_reverse, BRTreatmentOutcomeHead, AlphaRise, bce
from src.models.survival_losses import pc_hazard_loss

logger = logging.getLogger(__name__)
ray_constants.FUNCTION_SIZE_ERROR_THRESHOLD = 10**8  # ~ 100Mb


def train_eval_factual(args: dict, train_f: Dataset, val_f: Dataset, orig_hparams: DictConfig, input_size: int, model_cls,
                       tuning_criterion='survival', **kwargs):
    """
    Globally defined method, used for ray tuning
    :param args: Hyperparameter configuration
    :param train_f: Factual train dataset
    :param val_f: Factual val dataset
    :param orig_hparams: DictConfig of original hyperparameters
    :param input_size: Input size of model, infuences concrete hyperparameter configuration
    :param model_cls: class of model
    :param kwargs: Other args
    """
    OmegaConf.register_new_resolver("sum", lambda x, y: x + y, replace=True)
    new_params = deepcopy(orig_hparams)
    model_cls.set_hparams(new_params.model, args, input_size, model_cls.model_type)
    if model_cls.model_type == 'decoder':
        # Passing encoder takes too much memory
        encoder_r_size = new_params.model.encoder.br_size if 'br_size' in new_params.model.encoder \
            else new_params.model.encoder.seq_hidden_units  # Using either br_size or Memory adapter
        model = model_cls(new_params, encoder_r_size=encoder_r_size, **kwargs).double()
    else:
        model = model_cls(new_params, **kwargs).double()

    train_loader = DataLoader(train_f, shuffle=True, batch_size=new_params.model[model_cls.model_type]['batch_size'],
                              drop_last=True)
    trainer = Trainer(gpus=eval(str(new_params.exp.gpus))[:1],
                      logger=None,
                      max_epochs=new_params.exp.max_epochs,
                      progress_bar_refresh_rate=0,
                      gradient_clip_val=new_params.model[model_cls.model_type]['max_grad_norm']
                      if 'max_grad_norm' in new_params.model[model_cls.model_type] else None,
                      callbacks=[AlphaRise(rate=new_params.exp.alpha_rate)])
    trainer.fit(model, train_dataloader=train_loader)

    # For survival models, we don't use traditional metrics
    val_loss = model.get_validation_loss(val_f)
    tune.report(val_loss=val_loss)


class TimeVaryingCausalModel(LightningModule):
    """
    Abstract class for models, estimating counterfactual outcomes over time
    """

    model_type = None  # Will be defined in subclasses
    possible_model_types = None  # Will be defined in subclasses
    tuning_criterion = 'survival'

    def __init__(self,
                 args: DictConfig,
                 dataset_collection: Union[RealDatasetCollection, SyntheticDatasetCollection] = None,
                 autoregressive: bool = None,
                 has_vitals: bool = None,
                 bce_weights: np.array = None,
                 **kwargs):
        """
        Args:
            args: DictConfig of model hyperparameters
            dataset_collection: Dataset collection
            autoregressive: Flag of including previous outcomes to modelling
            has_vitals: Flag of vitals in dataset
            bce_weights: Re-weight BCE if used
            **kwargs: Other arguments
        """
        super().__init__()
        self.dataset_collection = dataset_collection
        if dataset_collection is not None:
            self.autoregressive = self.dataset_collection.autoregressive
            self.has_vitals = self.dataset_collection.has_vitals
            self.bce_weights = None  # Will be calculated, when calling preparing data
        else:
            self.autoregressive = autoregressive
            self.has_vitals = has_vitals
            self.bce_weights = bce_weights

        # General datasets parameters
        self.dim_treatments = args.model.dim_treatments
        self.dim_vitals = args.model.dim_vitals
        self.dim_static_features = args.model.dim_static_features
        self.dim_outcome = args.model.dim_outcomes
        
        self.input_size = None  # Will be defined in subclasses

        self.save_hyperparameters(args)  # Will be logged to mlflow

    def _get_optimizer(self, param_optimizer: list):
        no_decay = ['bias', 'layer_norm']
        sub_args = self.hparams.model[self.model_type]
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                'weight_decay': sub_args['optimizer']['weight_decay'],
            },
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
        ]
        lr = sub_args['optimizer']['learning_rate']
        optimizer_cls = sub_args['optimizer']['optimizer_cls']
        if optimizer_cls.lower() == 'adamw':
            optimizer = optim.AdamW(optimizer_grouped_parameters, lr=lr)
        elif optimizer_cls.lower() == 'adam':
            optimizer = optim.Adam(optimizer_grouped_parameters, lr=lr)
        elif optimizer_cls.lower() == 'sgd':
            optimizer = optim.SGD(optimizer_grouped_parameters, lr=lr,
                                  momentum=sub_args['optimizer']['momentum'])
        else:
            raise NotImplementedError()

        return optimizer

    def _get_lr_schedulers(self, optimizer):
        if not isinstance(optimizer, list):
            lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
            return [optimizer], [lr_scheduler]
        else:
            lr_schedulers = []
            for opt in optimizer:
                lr_schedulers.append(optim.lr_scheduler.ExponentialLR(opt, gamma=0.99))
            return optimizer, lr_schedulers

    def configure_optimizers(self):
        optimizer = self._get_optimizer(list(self.named_parameters()))
        if self.hparams.model[self.model_type]['optimizer']['lr_scheduler']:
            return self._get_lr_schedulers(optimizer)
        return optimizer

    def train_dataloader(self) -> DataLoader:
        sub_args = self.hparams.model[self.model_type]
        return DataLoader(self.dataset_collection.train_f, shuffle=True, batch_size=sub_args['batch_size'], drop_last=True)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.dataset_collection.val_f, batch_size=self.hparams.dataset.val_batch_size)

    def get_predictions(self, dataset: Dataset) -> np.array:
        raise NotImplementedError()

    def get_propensity_scores(self, dataset: Dataset) -> np.array:
        raise NotImplementedError()

    def get_representations(self, dataset: Dataset) -> np.array:
        raise NotImplementedError()

    def get_autoregressive_predictions(self, dataset: Dataset) -> np.array:
        logger.info(f'Autoregressive Prediction for {dataset.subset_name}.')
        if self.model_type == 'decoder':  # CRNDecoder / EDCTDecoder / RMSN Decoder

            predicted_outputs = np.zeros((len(dataset), self.hparams.dataset.projection_horizon, self.dim_outcome))
            for t in range(self.hparams.dataset.projection_horizon):
                logger.info(f't = {t + 2}')

                outputs_scaled = self.get_predictions(dataset)
                predicted_outputs[:, t] = outputs_scaled[:, t]

                if t < (self.hparams.dataset.projection_horizon - 1):
                    dataset.data['prev_outputs'][:, t + 1, :] = outputs_scaled[:, t, :]
        else:
            raise NotImplementedError()

        return predicted_outputs

    def get_validation_loss(self, dataset: Dataset):
        """Calculate validation loss for survival models"""
        logger.info(f'Validation loss calculation for {dataset.subset_name}.')
        dataloader = DataLoader(dataset, batch_size=self.hparams.dataset.val_batch_size, shuffle=False)
        
        total_loss = 0
        n_batches = 0
        
        self.eval()
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                _, _, binary_pred, binary_treatment_pred = self(batch)
                
                # Calculate binary classification loss
                event_indicator = batch['outputs'][:, :, 1]
                binary_labels = (event_indicator.max(dim=1)[0] > 0).float()
                
                # Get last timepoint prediction for binary classification
                if binary_pred.dim() == 3:
                    binary_pred_last = binary_pred[:, -1, 0]
                elif binary_pred.dim() == 2 and binary_pred.shape[1] > 1:
                    binary_pred_last = binary_pred[:, -1]
                else:
                    binary_pred_last = binary_pred.squeeze(-1)
                    
                binary_bce_loss = F.binary_cross_entropy_with_logits(
                    binary_pred_last, 
                    binary_labels,
                    reduction='mean'
                )
                
                # Also calculate treatment survival loss
                if not hasattr(self, 'sliding_windows'):
                    self.sliding_windows = self.dataset_collection.sliding_windows if self.dataset_collection else False
                
                if not self.sliding_windows:
                    event_time_bucket_full = batch['outputs'][:, :, 0]
                    event_indicator_full = batch['outputs'][:, :, 1]
                    event_time_bucket_last = event_time_bucket_full[:, -1].long()
                    event_indicator_last = event_indicator_full[:, -1]
                else:
                    event_time_bucket_last = batch['event_time_bucket']
                    event_indicator_last = batch['event_indicator']
                
                binary_treatment_survival_loss = pc_hazard_loss(
                    binary_treatment_pred,
                    event_time_bucket_last,
                    event_indicator_last,
                    sequence_lengths=batch['sequence_lengths'] if 'sequence_lengths' in batch else None,
                    last_timepoint_only=self.hparams.dataset.get('last_timepoint_only', True),
                    reduction='mean',
                    focal_gamma=self.hparams.dataset.get('focal_gamma', 0.0)
                )
                
                # Combine losses
                outcome_combined = (binary_bce_loss + binary_treatment_survival_loss) / 2.0
                
                total_loss += outcome_combined.item()
                n_batches += 1
        
        return total_loss / n_batches if n_batches > 0 else float('inf')

    @staticmethod
    def set_hparams(model_args: DictConfig, new_args: dict, input_size: int, model_type: str):
        raise NotImplementedError()

    def finetune(self, resources_per_trial: dict):
        """
        Hyperparameter tuning with ray[tune]
        """
        self.prepare_data()
        sub_args = self.hparams.model[self.model_type]
        logger.info(f"Running hyperparameters selection with {sub_args['tune_range']} trials")
        ray.init(num_gpus=len(eval(str(self.hparams.exp.gpus))), num_cpus=4, include_dashboard=False,
                 _redis_max_memory=ray_constants.FUNCTION_SIZE_ERROR_THRESHOLD)

        hparams_grid = {k: tune.choice(v) for k, v in sub_args['hparams_grid'].items()}
        analysis = tune.run(tune.with_parameters(train_eval_factual,
                                                 input_size=self.input_size,
                                                 model_cls=self.__class__,
                                                 tuning_criterion=self.tuning_criterion,
                                                 train_f=deepcopy(self.dataset_collection.train_f),
                                                 val_f=deepcopy(self.dataset_collection.val_f),
                                                 orig_hparams=self.hparams,
                                                 autoregressive=self.autoregressive,
                                                 has_vitals=self.has_vitals,
                                                 bce_weights=self.bce_weights,
                                                 projection_horizon=self.projection_horizon
                                                 if hasattr(self, 'projection_horizon') else None),
                            resources_per_trial=resources_per_trial,
                            metric="val_loss",
                            mode="min",
                            config=hparams_grid,
                            num_samples=sub_args['tune_range'],
                            name=f"{self.__class__.__name__}{self.model_type}",
                            max_failures=3)
        ray.shutdown()

        logger.info(f"Best hyperparameters found: {analysis.best_config}.")
        logger.info("Resetting current hyperparameters to best values.")
        self.set_hparams(self.hparams.model, analysis.best_config, self.input_size, self.model_type)

        self.__init__(self.hparams,
                      dataset_collection=self.dataset_collection,
                      encoder=self.encoder if hasattr(self, 'encoder') else None,
                      propensity_treatment=self.propensity_treatment if hasattr(self, 'propensity_treatment') else None,
                      propensity_history=self.propensity_history if hasattr(self, 'propensity_history') else None)
        return self

    def visualize(self, dataset: Dataset, index=0, artifacts_path=None):
        pass

    def bce_loss(self, treatment_pred, current_treatments, kind='predict'):
        mode = self.hparams.dataset.treatment_mode
        bce_weights = torch.tensor(self.bce_weights).type_as(current_treatments) if self.hparams.exp.bce_weight else None

        if kind == 'predict':
            bce_loss = bce(treatment_pred, current_treatments, mode, bce_weights)
        elif kind == 'confuse':
            if mode == 'continuous':
                # For continuous treatments, confuse means predict zero (no treatment)
                zero_treatments = torch.zeros_like(current_treatments)
                bce_loss = bce(treatment_pred, zero_treatments, mode)
            else:
                uniform_treatments = torch.ones_like(current_treatments)
                if mode == 'multiclass':
                    uniform_treatments *= 1 / current_treatments.shape[-1]
                elif mode == 'multilabel':
                    uniform_treatments *= 0.5
                bce_loss = bce(treatment_pred, uniform_treatments, mode)
        else:
            raise NotImplementedError()
        return bce_loss

    def on_fit_start(self) -> None:  # Issue with logging not yet existing parameters in MlFlow
        if self.trainer.logger is not None:
            self.trainer.logger.filter_submodels = list(self.possible_model_types - {self.model_type})

    def on_fit_end(self) -> None:  # Issue with logging not yet existing parameters in MlFlow
        if self.trainer.logger is not None:
            self.trainer.logger.filter_submodels = list(self.possible_model_types)


class BRCausalModel(TimeVaryingCausalModel):
    """
    Abstract class for models, estimating counterfactual outcomes over time with balanced representations
    """

    model_type = None  # Will be defined in subclasses
    possible_model_types = None   # Will be defined in subclasses
    tuning_criterion = 'survival'

    def __init__(self, args: DictConfig,
                 dataset_collection: Union[RealDatasetCollection, SyntheticDatasetCollection] = None,
                 autoregressive: bool = None,
                 has_vitals: bool = None,
                 bce_weights: np.array = None,
                 **kwargs):
        """
        Args:
            args: DictConfig of model hyperparameters
            dataset_collection: Dataset collection
            autoregressive: Flag of including previous outcomes to modelling
            has_vitals: Flag of vitals in dataset
            bce_weights: Re-weight BCE if used
            **kwargs: Other arguments
        """
        super().__init__(args, dataset_collection, autoregressive, has_vitals, bce_weights)

        # Balancing representation training parameters
        self.balancing = args.exp.balancing
        self.alpha = args.exp.alpha  # Used for gradient-reversal
        self.update_alpha = args.exp.update_alpha
        
        # GradNorm task weights (for balancing binary BCE loss and treatment survival loss)
        if hasattr(args.dataset, 'use_gradnorm') and args.dataset.use_gradnorm:
            # Initialize task weights as learnable parameters
            self.log_task_weights = nn.Parameter(torch.zeros(2))  # [binary_weight, treatment_survival_weight]
            # Store initial task weights (will be set after first loss calculation)
            self.register_buffer('initial_task_losses', torch.zeros(2))
        
        # Initialize APLoss for binary classification head
        # Will be initialized in on_train_start when we have access to the dataloader
        self.ap_loss = None
        # Read from config, default to False if not specified
        self.use_ap_loss = getattr(args.model, 'use_ap_loss', False)

    def on_train_start(self):
        """Initialize APLoss with the correct dataset length"""
        if self.use_ap_loss and self.ap_loss is None:
            # Get data length from the training dataset
            data_length = len(self.dataset_collection.train_f)
            logger.info(f"Initializing APLoss with data_len={data_length}")
            # Initialize with default margin and gamma, can be configured later
            self.ap_loss = libauc.losses.APLoss(data_len=data_length, margin=1.0, gamma=0.9)

    def configure_optimizers(self):
        if self.balancing == 'grad_reverse' and not self.hparams.exp.weights_ema:  # one optimizer
            # Get all parameters including GradNorm task weights if enabled
            all_params = list(self.named_parameters())
            
            # If using APLoss, use SOAP optimizer
            if self.use_ap_loss:
                lr = self.hparams.model[self.model_type]['optimizer']['learning_rate']
                optimizer = libauc.optimizers.SOAP(
                    [p for _, p in all_params],
                    lr=lr,
                    mode='adam'
                )
            else:
                optimizer = self._get_optimizer(all_params)

            if self.hparams.model[self.model_type]['optimizer']['lr_scheduler']:
                return self._get_lr_schedulers(optimizer)

            return optimizer

        else:  # two optimizers - simultaneous gradient descent update
            treatment_head_params = \
                ['br_treatment_outcome_head.' + s for s in self.br_treatment_outcome_head.treatment_head_params]
            treatment_head_params = \
                [k for k in dict(self.named_parameters()) for param in treatment_head_params if k.startswith(param)]
            
            # Exclude GradNorm task weights from the split (they go with non-treatment params)
            non_treatment_head_params = [k for k in dict(self.named_parameters()) 
                                       if k not in treatment_head_params]

            assert len(treatment_head_params + non_treatment_head_params) == len(list(self.named_parameters()))

            treatment_head_params = [(k, v) for k, v in dict(self.named_parameters()).items() if k in treatment_head_params]
            non_treatment_head_params = [(k, v) for k, v in dict(self.named_parameters()).items()
                                         if k in non_treatment_head_params]

            if self.hparams.exp.weights_ema:
                self.ema_treatment = ExponentialMovingAverage([par[1] for par in treatment_head_params],
                                                              decay=self.hparams.exp.beta)
                self.ema_non_treatment = ExponentialMovingAverage([par[1] for par in non_treatment_head_params],
                                                                  decay=self.hparams.exp.beta)

            treatment_head_optimizer = self._get_optimizer(treatment_head_params)
            
            # Use SOAP optimizer for non-treatment head params if using APLoss
            if self.use_ap_loss:
                lr = self.hparams.model[self.model_type]['optimizer']['learning_rate']
                non_treatment_head_optimizer = libauc.optimizers.SOAP(
                    [p for _, p in non_treatment_head_params],
                    lr=lr,
                    mode='adam'
                )
            else:
                non_treatment_head_optimizer = self._get_optimizer(non_treatment_head_params)

            if self.hparams.model[self.model_type]['optimizer']['lr_scheduler']:
                return self._get_lr_schedulers([non_treatment_head_optimizer, treatment_head_optimizer])

            return [non_treatment_head_optimizer, treatment_head_optimizer]

    def optimizer_step(self, epoch: int = None, batch_idx: int = None, optimizer=None, optimizer_idx: int = None, *args,
                       **kwargs) -> None:
        super().optimizer_step(epoch, batch_idx, optimizer, optimizer_idx, *args, **kwargs)
        if self.hparams.exp.weights_ema and optimizer_idx == 0:
            self.ema_non_treatment.update()
        elif self.hparams.exp.weights_ema and optimizer_idx == 1:
            self.ema_treatment.update()

    def _calculate_bce_weights(self) -> None:
        if self.hparams.dataset.treatment_mode == 'multiclass':
            current_treatments = self.dataset_collection.train_f.data['current_treatments']
            current_treatments = current_treatments.reshape(-1, current_treatments.shape[-1])
            current_treatments = current_treatments[self.dataset_collection.train_f.data['active_entries'].flatten().astype(bool)]
            current_treatments = np.argmax(current_treatments, axis=1)

            self.bce_weights = len(current_treatments) / np.bincount(current_treatments) / len(np.bincount(current_treatments))
        else:
            raise NotImplementedError()

    def on_fit_start(self) -> None:  # Issue with logging not yet existing parameters in MlFlow
        if self.trainer.logger is not None:
            self.trainer.logger.filter_submodels = ['decoder'] if self.model_type == 'encoder' else ['encoder']

    def on_fit_end(self) -> None:  # Issue with logging not yet existing parameters in MlFlow
        if self.trainer.logger is not None:
            self.trainer.logger.filter_submodels = ['encoder', 'decoder']

    def training_step(self, batch, batch_ind, optimizer_idx=0):
        for par in self.parameters():
            par.requires_grad = True

        if optimizer_idx == 0:  # grad reversal or domain confusion representation update
            if self.hparams.exp.weights_ema:
                with self.ema_treatment.average_parameters():
                    treatment_pred, _, binary_pred, binary_treatment_pred = self(batch)
            else:
                treatment_pred, _, binary_pred, binary_treatment_pred = self(batch)

            # Create binary labels: 1 if patient ever experiences event, 0 if censored throughout
            # Both sliding and non-sliding windows now use the same format:
            # outputs[:, :, 0] = event_time_bucket
            # outputs[:, :, 1] = event_indicator
            event_indicator = batch['outputs'][:, :, 1]
            binary_labels = (event_indicator.max(dim=1)[0] > 0).float()
            
            # Binary classification loss (only at last timepoint since it's seq2one)
            # binary_pred shape: (batch_size, seq_len, 1) or (batch_size, 1) after seq2one
            # If binary_pred has sequence dimension, take the last timepoint
            if binary_pred.dim() == 3:
                # Shape: (batch_size, seq_len, 1) -> take last timepoint and squeeze
                binary_pred_last = binary_pred[:, -1, 0]  # Shape: (batch_size,)
            elif binary_pred.dim() == 2 and binary_pred.shape[1] > 1:
                # Shape: (batch_size, seq_len) -> take last timepoint
                binary_pred_last = binary_pred[:, -1]
            else:
                # Shape: (batch_size, 1) -> squeeze
                binary_pred_last = binary_pred.squeeze(-1)
            
            # Use APLoss for binary classification if enabled
            if self.use_ap_loss and self.ap_loss is not None:
                # Check if there are any positive samples in this batch
                n_pos = binary_labels.sum().item()
                if n_pos > 0:
                    # APLoss expects predictions after sigmoid
                    binary_pred_sigmoid = torch.sigmoid(binary_pred_last)
                    # Create batch indices for APLoss
                    batch_size = binary_pred_sigmoid.shape[0]
                    batch_idx = torch.arange(batch_size, device=binary_pred_sigmoid.device)
                    # Calculate APLoss
                    binary_bce_loss = self.ap_loss(binary_pred_sigmoid, binary_labels, batch_idx)
                else:
                    # Fallback to BCE for batches with no positive samples
                    binary_bce_loss = F.binary_cross_entropy_with_logits(
                        binary_pred_last, 
                        binary_labels,
                        reduction='mean'
                    )
            else:
                # Fallback to standard BCE loss
                binary_bce_loss = F.binary_cross_entropy_with_logits(
                    binary_pred_last, 
                    binary_labels,
                    reduction='mean'
                )
            if self.balancing == 'grad_reverse':
                bce_loss = self.bce_loss(treatment_pred, batch['current_treatments'].double(), kind='predict')
            elif self.balancing == 'domain_confusion':
                bce_loss = self.bce_loss(treatment_pred, batch['current_treatments'].double(), kind='confuse')
                bce_loss = self.br_treatment_outcome_head.alpha * bce_loss
            else:
                raise NotImplementedError()

            # Masking for treatment prediction loss
            # Note: PC-Hazard loss already returns a scalar and handles its own masking
            bce_loss = (batch['active_entries'].squeeze(-1) * bce_loss).sum() / batch['active_entries'].sum()
            
            # Fourth head loss: PC-Hazard survival loss with treatment
            # binary_treatment_pred has shape (batch_size, seq_len, n_buckets)
            # Extract event_time_bucket and event_indicator from outputs
            if not hasattr(self, 'sliding_windows'):
                self.sliding_windows = self.dataset_collection.sliding_windows if self.dataset_collection else False
            
            if not self.sliding_windows:
                # For non-sliding windows, extract from outputs
                event_time_bucket_full = batch['outputs'][:, :, 0]  # Shape: (batch_size, seq_len)
                event_indicator_full = batch['outputs'][:, :, 1]    # Shape: (batch_size, seq_len)
                
                # Get last timestep values (since we use last_timepoint_only=True)
                event_time_bucket_last = event_time_bucket_full[:, -1].long()
                event_indicator_last = event_indicator_full[:, -1]
            else:
                # For sliding windows, use direct values
                event_time_bucket_last = batch['event_time_bucket']
                event_indicator_last = batch['event_indicator']
            
            # Apply PC-Hazard loss to fourth head
            binary_treatment_survival_loss = pc_hazard_loss(
                binary_treatment_pred,
                event_time_bucket_last,
                event_indicator_last,
                sequence_lengths=batch['sequence_lengths'] if 'sequence_lengths' in batch else None,
                last_timepoint_only=self.hparams.dataset.get('last_timepoint_only', True),
                reduction='mean',
                focal_gamma=self.hparams.dataset.get('focal_gamma', 0.0)
            )
            
            
            # Combine outcome losses
            if hasattr(self.hparams.dataset, 'use_gradnorm') and self.hparams.dataset.use_gradnorm:
                # Use GradNorm weights for binary and treatment survival losses
                task_weights = F.softmax(self.log_task_weights, dim=0)
                outcome_combined = task_weights[0] * binary_bce_loss + task_weights[1] * binary_treatment_survival_loss
                
                # Store individual task losses for GradNorm
                self.last_binary_loss = binary_bce_loss.detach()
                self.last_treatment_survival_loss = binary_treatment_survival_loss.detach()
            else:
                # Equal weights for both heads
                outcome_combined = (binary_bce_loss + binary_treatment_survival_loss) / 2.0
            
            # Total loss includes treatment loss separately
            loss = bce_loss + outcome_combined

            self.log(f'{self.model_type}_train_loss', loss, on_epoch=True, on_step=False, prog_bar=False, sync_dist=True)
            self.log(f'{self.model_type}_train_bce_loss', bce_loss, on_epoch=True, on_step=False, sync_dist=True)
            
            # Log individual losses
            self.log(f'{self.model_type}_train_binary_bce_loss', binary_bce_loss, on_epoch=True, on_step=False, sync_dist=True)
            self.log(f'{self.model_type}_train_binary_treatment_survival_loss', binary_treatment_survival_loss, on_epoch=True, on_step=False, sync_dist=True)
            self.log(f'{self.model_type}_train_outcome_combined_loss', outcome_combined, on_epoch=True, on_step=False, sync_dist=True)
            self.log(f'{self.model_type}_alpha', self.br_treatment_outcome_head.alpha, on_epoch=True, on_step=False,
                     sync_dist=True)
            
            # Log GradNorm task weights if enabled
            if hasattr(self.hparams.dataset, 'use_gradnorm') and self.hparams.dataset.use_gradnorm:
                task_weights = F.softmax(self.log_task_weights, dim=0)
                self.log(f'{self.model_type}_gradnorm_binary_weight', task_weights[0], on_epoch=True, on_step=False, sync_dist=True)
                self.log(f'{self.model_type}_gradnorm_treatment_survival_weight', task_weights[1], on_epoch=True, on_step=False, sync_dist=True)
            
            # Calculate and log treatment Pearson R for monitoring
            with torch.no_grad():
                mask_treatments = batch['active_entries'].squeeze(-1).cpu().numpy().astype(bool)
                treatment_true = batch['current_treatments'].double().cpu().numpy()
                treatment_pred_np = treatment_pred.detach().cpu().numpy()
                
                treatment_correlations = []
                for dim in range(treatment_true.shape[-1]):
                    t_true = treatment_true[:, :, dim][mask_treatments].flatten()
                    t_pred = treatment_pred_np[:, :, dim][mask_treatments].flatten()
                    
                    # Remove any NaN values
                    valid_mask = ~(np.isnan(t_true) | np.isnan(t_pred))
                    t_true = t_true[valid_mask]
                    t_pred = t_pred[valid_mask]
                    
                    if len(t_true) > 1:
                        # Check if either array is constant (no variance)
                        if np.std(t_true) > 1e-8 and np.std(t_pred) > 1e-8:
                            from scipy.stats import pearsonr
                            r, _ = pearsonr(t_true, t_pred)
                            treatment_correlations.append(r)
                        else:
                            # If either array is constant, correlation is undefined
                            # Skip this dimension or use NaN
                            treatment_correlations.append(np.nan)
                
                avg_treatment_pearson_r = np.nanmean(treatment_correlations) if treatment_correlations else np.nan
                self.log(f'{self.model_type}_train_treatment_pearson_r', avg_treatment_pearson_r, on_epoch=True, on_step=False, sync_dist=True)

            return loss

        elif optimizer_idx == 1:  # domain classifier update
            if self.hparams.exp.weights_ema:
                with self.ema_non_treatment.average_parameters():
                    treatment_pred, _, _, _ = self(batch, detach_treatment=True)
            else:
                treatment_pred, _, _, _ = self(batch, detach_treatment=True)

            bce_loss = self.bce_loss(treatment_pred, batch['current_treatments'].double(), kind='predict')
            if self.balancing == 'domain_confusion':
                bce_loss = self.br_treatment_outcome_head.alpha * bce_loss

            # Masking for shorter sequences
            # Attention! Averaging across all the active entries (= sequence masks) for full batch
            bce_loss = (batch['active_entries'].squeeze(-1) * bce_loss).sum() / batch['active_entries'].sum()
            self.log(f'{self.model_type}_train_bce_loss_cl', bce_loss, on_epoch=True, on_step=False, sync_dist=True)

            return bce_loss

    def test_step(self, batch, batch_ind, **kwargs):
        if self.hparams.exp.weights_ema:
            with self.ema_non_treatment.average_parameters():
                with self.ema_treatment.average_parameters():
                    treatment_pred, _, binary_pred, binary_treatment_pred = self(batch)
        else:
            treatment_pred, _, binary_pred, binary_treatment_pred = self(batch)

        if self.balancing == 'grad_reverse':
            bce_loss = self.bce_loss(treatment_pred, batch['current_treatments'].double(), kind='predict')
        elif self.balancing == 'domain_confusion':
            bce_loss = self.bce_loss(treatment_pred, batch['current_treatments'].double(), kind='confuse')

        # Calculate binary classification loss
        event_indicator = batch['outputs'][:, :, 1]
        binary_labels = (event_indicator.max(dim=1)[0] > 0).float()
        
        # Get last timepoint prediction for binary classification
        if binary_pred.dim() == 3:
            binary_pred_last = binary_pred[:, -1, 0]
        elif binary_pred.dim() == 2 and binary_pred.shape[1] > 1:
            binary_pred_last = binary_pred[:, -1]
        else:
            binary_pred_last = binary_pred.squeeze(-1)
            
        binary_bce_loss = F.binary_cross_entropy_with_logits(
            binary_pred_last, 
            binary_labels,
            reduction='mean'
        )

        # Masking for treatment prediction loss
        bce_loss = (batch['active_entries'].squeeze(-1) * bce_loss).sum() / batch['active_entries'].sum()
        loss = bce_loss + binary_bce_loss

        # For survival models, we don't calculate Pearson correlation on hazard predictions
        # as they are not directly comparable to the event_time_bucket/event_indicator labels

        # Calculate Pearson correlation coefficient for treatments (average across dimensions)
        mask_treatments = batch['active_entries'].squeeze(-1).cpu().numpy().astype(bool)
        treatment_true = batch['current_treatments'].double().cpu().numpy()
        treatment_pred_np = treatment_pred.cpu().numpy()
        
        treatment_correlations = []
        for dim in range(treatment_true.shape[-1]):
            t_true = treatment_true[:, :, dim][mask_treatments].flatten()
            t_pred = treatment_pred_np[:, :, dim][mask_treatments].flatten()
            
            # Remove any NaN values
            valid_mask = ~(np.isnan(t_true) | np.isnan(t_pred))
            t_true = t_true[valid_mask]
            t_pred = t_pred[valid_mask]
            
            if len(t_true) > 1:
                # Check if either array is constant (no variance)
                if np.std(t_true) > 1e-8 and np.std(t_pred) > 1e-8:
                    r, _ = pearsonr(t_true, t_pred)
                    treatment_correlations.append(r)
                else:
                    # If either array is constant, correlation is undefined
                    treatment_correlations.append(np.nan)
        
        avg_treatment_pearson_r = np.nanmean(treatment_correlations) if treatment_correlations else np.nan

        subset_name = self.test_dataloader().dataset.subset_name
        self.log(f'{self.model_type}_{subset_name}_loss', loss, on_epoch=True, on_step=False, sync_dist=True)
        self.log(f'{self.model_type}_{subset_name}_bce_loss', bce_loss, on_epoch=True, on_step=False, sync_dist=True)
        
        # Log binary loss
        self.log(f'{self.model_type}_{subset_name}_binary_bce_loss', binary_bce_loss, on_epoch=True, on_step=False, sync_dist=True)
        self.log(f'{self.model_type}_{subset_name}_treatment_pearson_r', avg_treatment_pearson_r, on_epoch=True, on_step=False, sync_dist=True)

    def validation_step(self, batch, batch_idx):
        """Validation step for survival models"""
        # Get predictions
        treatment_pred, _, binary_pred, binary_treatment_pred = self(batch)
        
        # Calculate binary classification loss
        # Get disease labels from outputs
        event_indicator = batch['outputs'][:, :, 1]  # event_indicator column
        binary_labels = (event_indicator.max(dim=1)[0] > 0).float()
        
        # Get last timepoint prediction for binary classification
        if binary_pred.dim() == 3:
            binary_pred_last = binary_pred[:, -1, 0]
        elif binary_pred.dim() == 2 and binary_pred.shape[1] > 1:
            binary_pred_last = binary_pred[:, -1]
        else:
            binary_pred_last = binary_pred.squeeze(-1)
            
        binary_bce_loss = F.binary_cross_entropy_with_logits(
            binary_pred_last, 
            binary_labels,
            reduction='mean'
        )
        
        # Fourth head loss: PC-Hazard survival loss with treatment
        if not hasattr(self, 'sliding_windows'):
            self.sliding_windows = self.dataset_collection.sliding_windows if self.dataset_collection else False
        
        if not self.sliding_windows:
            # For non-sliding windows, extract from outputs
            event_time_bucket_full = batch['outputs'][:, :, 0]
            event_indicator_full = batch['outputs'][:, :, 1]
            event_time_bucket_last = event_time_bucket_full[:, -1].long()
            event_indicator_last = event_indicator_full[:, -1]
        else:
            # For sliding windows, use direct values
            event_time_bucket_last = batch['event_time_bucket']
            event_indicator_last = batch['event_indicator']
        
        # Apply PC-Hazard loss to fourth head
        binary_treatment_survival_loss = pc_hazard_loss(
            binary_treatment_pred,
            event_time_bucket_last,
            event_indicator_last,
            sequence_lengths=batch['sequence_lengths'] if 'sequence_lengths' in batch else None,
            last_timepoint_only=self.hparams.dataset.get('last_timepoint_only', True),
            reduction='mean',
            focal_gamma=self.hparams.dataset.get('focal_gamma', 0.0)
        )
        
        # Equal weights for all three heads
        outcome_combined = (binary_bce_loss + binary_treatment_survival_loss) / 2.0
        
        
        # Log validation losses
        self.log('val_loss', outcome_combined, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('val_binary_bce_loss', binary_bce_loss, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log('val_binary_treatment_survival_loss', binary_treatment_survival_loss, on_epoch=True, prog_bar=False, sync_dist=True)
        
        # Store batch data for epoch end metrics if last batch
        # This ensures on_validation_epoch_end has data to work with
        if batch_idx == 0:  # Only store first batch to save memory
            self.validation_step_outputs = []
        
        # Horizon-specific metrics will be calculated in on_validation_epoch_end
        return outcome_combined
    
    def calculate_binary_metrics(self, dataset, dataset_name='val'):
        """Calculate AUC, AUPRC, and prevalence for binary 'ever disease' prediction
        
        Args:
            dataset: Dataset to evaluate
            dataset_name: Name for logging ('val' or 'test')
            
        Returns:
            Dict of binary classification metrics
        """
        from sklearn.metrics import roc_auc_score, average_precision_score
        
        # Create dataloader
        dataloader = DataLoader(dataset, batch_size=self.hparams.dataset.val_batch_size, shuffle=False)
        
        
        all_binary_preds = []
        all_binary_treatment_preds = []
        all_binary_labels = []
        
        self.eval()
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                # Move batch to device
                for key in batch:
                    if isinstance(batch[key], torch.Tensor):
                        batch[key] = batch[key].to(self.device)
                
                # Get predictions
                _, _, binary_pred, binary_treatment_pred = self(batch)
                
                # Create binary labels: 1 if patient ever experiences event
                # Both sliding and non-sliding windows now use the same format:
                # outputs[:, :, 0] = event_time_bucket
                # outputs[:, :, 1] = event_indicator
                event_indicator = batch['outputs'][:, :, 1]
                binary_labels = (event_indicator.max(dim=1)[0] > 0).float()
                
                # Convert logits to probabilities - always use last timestep
                if binary_pred.dim() == 3:
                    # Shape: (batch_size, seq_len, 1) -> take last timepoint
                    binary_probs = torch.sigmoid(binary_pred[:, -1, 0])
                elif binary_pred.dim() == 2 and binary_pred.shape[1] > 1:
                    # Shape: (batch_size, seq_len) -> take last timepoint
                    binary_probs = torch.sigmoid(binary_pred[:, -1])
                else:
                    # Shape: (batch_size, 1) -> squeeze
                    binary_probs = torch.sigmoid(binary_pred.squeeze(-1))
                
                # Same for binary_treatment_pred - convert hazards to risk scores
                if binary_treatment_pred.dim() == 3:
                    # Get all hazards at last timestep
                    hazards_last = binary_treatment_pred[:, -1, :]  # Shape: (batch_size, n_buckets)
                    # Calculate cumulative hazard
                    cum_hazard = hazards_last.sum(dim=1)
                    # Convert to survival probability
                    survival_prob = torch.exp(-cum_hazard)
                    # Get risk score (probability of disease)
                    binary_treatment_probs = 1 - survival_prob
                else:
                    # Should not happen with current architecture
                    # Fallback: assume single hazard value
                    binary_treatment_probs = 1 - torch.exp(-binary_treatment_pred[:, -1])
                
                all_binary_preds.append(binary_probs.cpu())
                all_binary_treatment_preds.append(binary_treatment_probs.cpu())
                all_binary_labels.append(binary_labels.cpu())
                
        
        # Concatenate all batches
        all_binary_preds = torch.cat(all_binary_preds).numpy()
        all_binary_treatment_preds = torch.cat(all_binary_treatment_preds).numpy()
        all_binary_labels = torch.cat(all_binary_labels).numpy()
        
        # Calculate metrics
        metrics = {}
        
        # Third head metrics (binary without treatment)
        # AUC-ROC
        try:
            auc_roc = roc_auc_score(all_binary_labels, all_binary_preds)
            metrics[f'{dataset_name}_binary_auc_roc'] = auc_roc
        except:
            metrics[f'{dataset_name}_binary_auc_roc'] = np.nan
        
        # AUC-PR
        try:
            auc_pr = average_precision_score(all_binary_labels, all_binary_preds)
            metrics[f'{dataset_name}_binary_auc_pr'] = auc_pr
        except:
            metrics[f'{dataset_name}_binary_auc_pr'] = np.nan
        
        # Fourth head metrics (binary with treatment) - similar to third head
        # AUC-ROC
        try:
            auc_roc_treatment = roc_auc_score(all_binary_labels, all_binary_treatment_preds)
            metrics[f'{dataset_name}_binary_treatment_auc_roc'] = auc_roc_treatment
        except:
            metrics[f'{dataset_name}_binary_treatment_auc_roc'] = np.nan
        
        # AUC-PR
        try:
            auc_pr_treatment = average_precision_score(all_binary_labels, all_binary_treatment_preds)
            metrics[f'{dataset_name}_binary_treatment_auc_pr'] = auc_pr_treatment
        except:
            metrics[f'{dataset_name}_binary_treatment_auc_pr'] = np.nan
        
        # Prevalence and counts
        prevalence = np.mean(all_binary_labels)
        metrics[f'{dataset_name}_binary_prevalence'] = prevalence
        
        # Case/control counts
        n_cases = np.sum(all_binary_labels)
        n_controls = len(all_binary_labels) - n_cases
        metrics[f'{dataset_name}_binary_n_cases'] = int(n_cases)
        metrics[f'{dataset_name}_binary_n_controls'] = int(n_controls)
        
        # Log metrics only if we're in a training/validation context
        try:
            for name, value in metrics.items():
                # Skip logging list/dict values (like bucket_metrics)
                if not isinstance(value, (list, dict)):
                    self.log(name, value, on_epoch=True, sync_dist=True)
        except (AssertionError, AttributeError):
            # We're outside the training loop, so logging is not available
            pass
        
        
        return metrics
    
    def calculate_bucket_specific_metrics(self, dataset, dataset_name='val'):
        """Calculate bucket-specific metrics for the fourth head AND binary classification head
        
        This evaluates both heads on the SAME bucket-specific labels:
        1. Fourth head: Survival model with treatment information
        2. Third head: Binary classification (evaluated on same bucket-specific labels)
        
        Both models predict: "Will patient have event in this specific time bucket?"
        This enables fair comparison of treatment-aware vs treatment-agnostic models.
        
        Args:
            dataset: Dataset to evaluate
            dataset_name: Name for logging ('val' or 'test')
            
        Returns:
            Dict of bucket-specific metrics
        """
        from sklearn.metrics import roc_auc_score, average_precision_score
        import pandas as pd
        import os
        
        # Create dataloader
        dataloader = DataLoader(dataset, batch_size=self.hparams.dataset.val_batch_size, shuffle=False)
        
        # Get number of buckets from model
        n_buckets = self.num_hazard_buckets if hasattr(self, 'num_hazard_buckets') else 2
        
        # Collect predictions and derive bucket-specific labels
        # Create lists for each bucket dynamically
        all_bucket_labels = [[] for _ in range(n_buckets)]
        all_bucket_preds = [[] for _ in range(n_buckets)]
        all_binary_preds = []  # Binary classification predictions (third head)
        all_person_ids = []
        all_original_labels = []
        
        self.eval()
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                # Move batch to device
                for key in batch:
                    if isinstance(batch[key], torch.Tensor):
                        batch[key] = batch[key].to(self.device)
                
                # Get predictions from both third and fourth heads
                _, _, binary_pred, binary_treatment_pred = self(batch)
                
                # Get event information
                event_indicator = batch['outputs'][:, :, 1]
                event_time_bucket = batch['outputs'][:, :, 0]
                
                batch_size = event_indicator.shape[0]
                
                # Get person IDs if available
                if 'person_ids' in batch:
                    person_ids = batch['person_ids']
                else:
                    # Create sequential IDs if not available
                    start_idx = len(all_person_ids) * batch_size if all_person_ids else 0
                    person_ids = torch.arange(start_idx, start_idx + batch_size)
                
                # Derive bucket-specific labels for each sample
                bucket_labels = [torch.zeros(batch_size) for _ in range(n_buckets)]
                original_labels = (event_indicator.max(dim=1)[0] > 0).float()
                
                for i in range(batch_size):
                    # Check if patient ever had event
                    if event_indicator[i].max() > 0:
                        # Find first event
                        event_indices = torch.where(event_indicator[i] > 0)[0]
                        if len(event_indices) > 0:
                            first_event_idx = event_indices[0]
                            event_bucket = int(event_time_bucket[i, first_event_idx].item())
                            
                            # Set label for the bucket where event occurred
                            if event_bucket < n_buckets:
                                bucket_labels[event_bucket][i] = 1
                                # Exclude from later buckets (set to -1)
                                for b in range(event_bucket + 1, n_buckets):
                                    bucket_labels[b][i] = -1
                            else:
                                # Event in censored bucket (beyond our prediction range)
                                # All buckets remain 0
                                pass
                    else:
                        # Censored patient - 0 for all buckets
                        # Labels already initialized to 0
                        pass
                
                # Get predictions for each bucket (last timestep)
                # binary_treatment_pred shape: (batch_size, seq_len, n_buckets)
                if binary_treatment_pred.dim() == 3:
                    # Get hazards at last timestep
                    hazards_last = binary_treatment_pred[:, -1, :]  # Shape: (batch_size, n_buckets)
                    
                    # Calculate risk for each bucket
                    bucket_probs = []
                    for b in range(n_buckets):
                        # Risk up to bucket b = 1 - exp(-sum(hazards[0:b+1]))
                        cum_hazard = hazards_last[:, :b+1].sum(dim=1)
                        bucket_risk = 1 - torch.exp(-cum_hazard)
                        bucket_probs.append(bucket_risk)
                else:
                    # Should not happen with current architecture
                    raise ValueError(f"Unexpected prediction shape: {binary_treatment_pred.shape}")
                
                # Extract binary predictions (third head) at last timestep
                if binary_pred.dim() == 3:
                    binary_probs = torch.sigmoid(binary_pred[:, -1, 0])  # Shape: (batch_size,)
                elif binary_pred.dim() == 2 and binary_pred.shape[1] > 1:
                    binary_probs = torch.sigmoid(binary_pred[:, -1])
                else:
                    binary_probs = torch.sigmoid(binary_pred.squeeze(-1))
                
                # Collect results for all buckets
                for b in range(n_buckets):
                    all_bucket_labels[b].append(bucket_labels[b].cpu())
                    all_bucket_preds[b].append(bucket_probs[b].cpu())
                all_binary_preds.append(binary_probs.cpu())
                all_person_ids.append(person_ids.cpu())
                all_original_labels.append(original_labels.cpu())
        
        # Concatenate all batches
        for b in range(n_buckets):
            all_bucket_labels[b] = torch.cat(all_bucket_labels[b]).numpy()
            all_bucket_preds[b] = torch.cat(all_bucket_preds[b]).numpy()
        all_binary_preds = torch.cat(all_binary_preds).numpy()
        all_person_ids = torch.cat(all_person_ids).numpy()
        all_original_labels = torch.cat(all_original_labels).numpy()
        
        
        metrics = {}
        
        # Calculate metrics for each bucket
        for b in range(n_buckets):
            # Get labels and predictions for this bucket
            bucket_labels = all_bucket_labels[b]
            bucket_preds = all_bucket_preds[b]
            
            # For buckets after the first, only evaluate samples at risk
            if b > 0:
                # Only evaluate samples that weren't excluded (didn't have event in earlier buckets)
                at_risk_mask = bucket_labels != -1
                bucket_labels_at_risk = bucket_labels[at_risk_mask]
                bucket_preds_at_risk = bucket_preds[at_risk_mask]
                n_at_risk = len(bucket_labels_at_risk)
                n_excluded = int((bucket_labels == -1).sum())
            else:
                # For bucket 0, all samples are at risk
                bucket_labels_at_risk = bucket_labels
                bucket_preds_at_risk = bucket_preds
                n_at_risk = len(bucket_labels)
                n_excluded = 0
            
            # Count events and controls
            n_events = int((bucket_labels_at_risk == 1).sum())
            n_controls = int((bucket_labels_at_risk == 0).sum())
            
            # Calculate metrics if we have both events and controls
            if n_events > 0 and n_controls > 0:
                try:
                    # Fourth head metrics (survival)
                    auc_roc = roc_auc_score(bucket_labels_at_risk, bucket_preds_at_risk)
                    auc_pr = average_precision_score(bucket_labels_at_risk, bucket_preds_at_risk)
                    
                    # Third head metrics (binary classification) on same patient group
                    # Use the SAME bucket-specific labels as survival for fair comparison
                    if b > 0:
                        # Only evaluate samples that weren't excluded
                        binary_preds_at_risk = all_binary_preds[at_risk_mask]
                    else:
                        binary_preds_at_risk = all_binary_preds
                    
                    # Use bucket_labels_at_risk (same as survival) instead of original labels
                    binary_auc_roc = roc_auc_score(bucket_labels_at_risk, binary_preds_at_risk)
                    binary_auc_pr = average_precision_score(bucket_labels_at_risk, binary_preds_at_risk)
                except:
                    auc_roc = np.nan
                    auc_pr = np.nan
                    binary_auc_roc = np.nan
                    binary_auc_pr = np.nan
            else:
                auc_roc = np.nan
                auc_pr = np.nan
                binary_auc_roc = np.nan
                binary_auc_pr = np.nan
            
            # Store metrics with bucket number
            metrics[f'{dataset_name}_bucket{b}_auc_roc'] = auc_roc
            metrics[f'{dataset_name}_bucket{b}_auc_pr'] = auc_pr
            metrics[f'{dataset_name}_bucket{b}_binary_auc_roc'] = binary_auc_roc
            metrics[f'{dataset_name}_bucket{b}_binary_auc_pr'] = binary_auc_pr
            metrics[f'{dataset_name}_bucket{b}_n_events'] = n_events
            metrics[f'{dataset_name}_bucket{b}_n_controls'] = n_controls
            metrics[f'{dataset_name}_bucket{b}_n_at_risk'] = n_at_risk
            if b > 0:  # Only track exclusions for buckets after the first
                metrics[f'{dataset_name}_bucket{b}_n_excluded'] = n_excluded
            metrics[f'{dataset_name}_bucket{b}_prevalence'] = n_events / n_at_risk if n_at_risk > 0 else 0
        
        # Log metrics if we're in a training/validation context
        try:
            for name, value in metrics.items():
                self.log(name, value, on_epoch=True, sync_dist=True)
        except (AssertionError, AttributeError):
            # We're outside the training loop, so logging is not available
            pass
        
        return metrics
    
    def calculate_horizon_metrics(self, dataset, dataset_name='val'):
        """Calculate AUC and AUPRC for each horizon in the dataset
        
        Args:
            dataset: Dataset to evaluate
            dataset_name: Name for logging ('val' or 'test')
            
        Returns:
            Dict of metrics by horizon
        """
        # This method was designed for the hazard prediction head which we removed
        # Return empty dict for now
        return {}
            
        from sklearn.metrics import roc_auc_score, average_precision_score
        
        # Create dataloader
        dataloader = DataLoader(dataset, batch_size=self.hparams.dataset.val_batch_size, shuffle=False)
        
        all_hazard_preds = []
        all_event_indicators = []
        all_event_buckets = []
        all_horizon_days = []
        
        self.eval()
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                # Move batch to device
                for key in batch:
                    if isinstance(batch[key], torch.Tensor):
                        batch[key] = batch[key].to(self.device)
                
                # Get predictions
                _, outcome_pred, _, _, _ = self(batch)
                
                # Extract last timestep predictions
                batch_size = outcome_pred.shape[0]
                seq_lengths = batch['sequence_lengths']
                
                last_timestep_preds = []
                for i in range(batch_size):
                    last_idx = seq_lengths[i] - 1
                    last_timestep_preds.append(outcome_pred[i, last_idx, :])
                
                hazard_preds = torch.stack(last_timestep_preds)
                hazard_preds = F.softplus(hazard_preds)
                
                all_hazard_preds.append(hazard_preds.cpu())
                all_event_indicators.append(batch['event_indicator'].cpu())
                all_event_buckets.append(batch['event_time_bucket'].cpu())
                all_horizon_days.append(batch['horizon_days'].cpu())
        
        # Concatenate all batches
        all_hazard_preds = torch.cat(all_hazard_preds, dim=0).numpy()
        all_event_indicators = torch.cat(all_event_indicators, dim=0).numpy()
        all_event_buckets = torch.cat(all_event_buckets, dim=0).numpy()
        all_horizon_days = torch.cat(all_horizon_days, dim=0).numpy()
        
        # Calculate survival probabilities
        cumulative_hazards = np.cumsum(all_hazard_preds, axis=1)
        survival_probs = np.exp(-cumulative_hazards)
        
        # Build horizon to bucket mapping from config - horizons are required
        horizon_to_bucket = {}
        if hasattr(self.hparams.dataset, 'horizons'):
            for idx, h in enumerate(self.hparams.dataset.horizons):
                if len(h) == 2:
                    # Old format: (name, days)
                    name, days = h
                    horizon_to_bucket[days] = idx
                else:
                    # New format: (name, start, end)
                    name, start, end = h
                    # For bucket-based approach, we don't use horizon_to_bucket
                    # since all windows predict all buckets
                    pass
        else:
            raise ValueError("horizons must be defined in dataset config")
        
        unique_horizons = sorted(np.unique(all_horizon_days))
        horizon_metrics = {}
        
        for horizon in unique_horizons:
            horizon_mask = all_horizon_days == horizon
            if not np.any(horizon_mask):
                continue
                
            bucket_idx = horizon_to_bucket.get(int(horizon), -1)
            if bucket_idx == -1:
                continue
            
            # Get predictions and labels
            horizon_events = all_event_indicators[horizon_mask]
            horizon_event_buckets = all_event_buckets[horizon_mask]
            
            # Risk scores
            # With 2 buckets, survival_probs has shape (n_samples, 2)
            # bucket_idx 0 or 1 directly indexes into survival_probs
            horizon_risk_scores = 1 - survival_probs[horizon_mask, bucket_idx]
            
            # Binary labels
            horizon_binary_labels = (horizon_events == 1) & (horizon_event_buckets <= bucket_idx)
            
            n_events = horizon_binary_labels.sum()
            n_controls = len(horizon_binary_labels) - n_events
            
            if n_events > 0 and n_controls > 0:
                try:
                    auc_roc = roc_auc_score(horizon_binary_labels, horizon_risk_scores)
                    auc_pr = average_precision_score(horizon_binary_labels, horizon_risk_scores)
                    
                    # Log metrics if in validation mode
                    if dataset_name == 'val':
                        self.log(f'AUROC_{int(horizon)}d', auc_roc, on_epoch=True, prog_bar=False, sync_dist=True)
                        self.log(f'AUPRC_{int(horizon)}d', auc_pr, on_epoch=True, prog_bar=False, sync_dist=True)
                    
                    # Calculate prevalence
                    total_samples = n_events + n_controls
                    prevalence = (n_events / total_samples) * 100 if total_samples > 0 else 0
                    
                    horizon_metrics[int(horizon)] = {
                        'auc_roc': auc_roc,
                        'auc_pr': auc_pr,
                        'n_events': int(n_events),
                        'n_controls': int(n_controls),
                        'prevalence': prevalence
                    }
                except Exception as e:
                    logger.warning(f"Could not calculate metrics for horizon {horizon}: {e}")
        
        return horizon_metrics
    
    def calculate_bucket_metrics(self, dataset, dataset_name='val'):
        """Calculate metrics for each bucket in bucket-based approach"""
        # This method was designed for the hazard prediction head which we removed
        # Return empty dict for now
        return {}
        
        # Check if using new bucket format (3 elements) or old format (2 elements)
        if len(horizons[0]) == 2:
            # Convert old format to bucket format
            bucket_horizons = []
            for i, (name, days) in enumerate(horizons):
                if i == 0:
                    bucket_horizons.append([name, 1, days])
                else:
                    prev_end = bucket_horizons[-1][2]
                    bucket_horizons.append([name, prev_end + 1, days])
            horizons = bucket_horizons
        
        from sklearn.metrics import roc_auc_score, average_precision_score
        
        # Create dataloader
        dataloader = DataLoader(dataset, batch_size=self.hparams.dataset.val_batch_size, shuffle=False)
        
        # Collect all predictions and labels
        all_hazard_preds = []
        all_event_indicators = []
        all_event_buckets = []
        
        self.eval()
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                # Move batch to device
                for key in batch:
                    if isinstance(batch[key], torch.Tensor):
                        batch[key] = batch[key].to(self.device)
                
                # Get predictions
                _, outcome_pred, _, _, _ = self(batch)
                
                # Extract hazard predictions (after softplus)
                hazard_pred = outcome_pred  # Already positive after softplus
                
                # Get last timestep predictions
                if hasattr(self.hparams.dataset, 'last_timepoint_only') and self.hparams.dataset.last_timepoint_only:
                    if 'sequence_lengths' in batch:
                        seq_lengths = batch['sequence_lengths'].long() - 1
                    else:
                        seq_lengths = torch.full((hazard_pred.shape[0],), hazard_pred.shape[1] - 1, device=hazard_pred.device)
                    
                    batch_indices = torch.arange(hazard_pred.shape[0], device=hazard_pred.device)
                    hazard_pred_last = hazard_pred[batch_indices, seq_lengths]
                else:
                    hazard_pred_last = hazard_pred[:, -1]
                
                all_hazard_preds.append(hazard_pred_last.cpu().numpy())
                
                # Extract last timestep labels to match predictions
                if hasattr(self.hparams.dataset, 'last_timepoint_only') and self.hparams.dataset.last_timepoint_only:
                    # Get last timestep event info for each patient
                    batch_size = hazard_pred.shape[0]
                    if 'sequence_lengths' in batch:
                        seq_lengths = batch['sequence_lengths'].long() - 1
                    else:
                        seq_lengths = torch.full((batch_size,), hazard_pred.shape[1] - 1, device=hazard_pred.device)
                    
                    batch_indices = torch.arange(batch_size, device=hazard_pred.device)
                    
                    # Extract last timestep from event_indicator and event_time_bucket
                    event_indicator_full = batch['outputs'][:, :, 1]  # event_indicator is second column
                    event_bucket_full = batch['outputs'][:, :, 0]     # event_time_bucket is first column
                    
                    event_indicator_last = event_indicator_full[batch_indices, seq_lengths]
                    event_bucket_last = event_bucket_full[batch_indices, seq_lengths]
                    
                    all_event_indicators.append(event_indicator_last.cpu().numpy())
                    all_event_buckets.append(event_bucket_last.cpu().numpy())
                else:
                    # For seq2seq mode (not currently used)
                    all_event_indicators.append(batch['event_indicator'].cpu().numpy())
                    all_event_buckets.append(batch['event_time_bucket'].cpu().numpy())
        
        # Concatenate all batches
        if len(all_hazard_preds) == 0:
            logger.warning("No predictions collected for bucket metrics")
            return {}
            
        all_hazard_preds = np.concatenate(all_hazard_preds, axis=0)
        all_event_indicators = np.concatenate(all_event_indicators, axis=0)
        all_event_buckets = np.concatenate(all_event_buckets, axis=0)
        
        
        # Calculate survival probabilities
        cumulative_hazards = np.cumsum(all_hazard_preds, axis=1)
        survival_probs = np.exp(-cumulative_hazards)
        
        bucket_metrics = {}
        
        # Analyze each bucket
        for bucket_idx, h in enumerate(horizons):
            name, start, end = h
            
            # Get all windows where event occurs in this bucket
            event_in_bucket = (all_event_indicators == 1) & (all_event_buckets == bucket_idx)
            # Get all windows that are not censored before this bucket
            not_censored_before = all_event_buckets >= bucket_idx
            
            if not np.any(not_censored_before):
                continue
            
            # Risk scores for this bucket
            # With 2 buckets, survival_probs has shape (n_samples, 2)
            # bucket_idx 0 or 1 directly indexes into survival_probs
            risk_scores = 1 - survival_probs[:, bucket_idx]
            
            # Binary labels: event in this bucket vs no event in this bucket
            binary_labels = event_in_bucket.astype(float)
            
            # Filter to windows that could have events
            valid_mask = not_censored_before
            binary_labels_valid = binary_labels[valid_mask]
            risk_scores_valid = risk_scores[valid_mask]
            
            n_events = binary_labels_valid.sum()
            n_no_events = len(binary_labels_valid) - n_events
            
            
            if n_events > 0 and n_no_events > 0:
                try:
                    auc_roc = roc_auc_score(binary_labels_valid, risk_scores_valid)
                    auc_pr = average_precision_score(binary_labels_valid, risk_scores_valid)
                    prevalence = 100 * n_events / len(binary_labels_valid)
                    
                    bucket_name = f"{name} ({start}-{end}d)"
                    bucket_metrics[bucket_name] = {
                        'auc_roc': auc_roc,
                        'auc_pr': auc_pr,
                        'n_events': int(n_events),
                        'n_controls': int(n_no_events),
                        'prevalence': prevalence
                    }
                    logger.info(f"Added metrics for bucket {bucket_name}")
                except Exception as e:
                    logger.warning(f"Could not compute metrics for bucket {bucket_idx}: {e}")
            else:
                pass
        
        return bucket_metrics
    
    def calculate_patient_level_bucket_metrics(self, dataset, dataset_name='val'):
        """Calculate AUC and AUPRC at PATIENT level (comparable to binary classification)
        
        This aggregates predictions across all timesteps to answer:
        "Will this patient ever have disease in bucket X?"
        
        Args:
            dataset: Dataset to evaluate
            dataset_name: Name for logging ('val' or 'test')
        """
        # This method was designed for the hazard prediction head which we removed
        # Return empty dict for now
        return {}
        horizons = self.hparams.dataset.horizons
        if not horizons:
            return {}
        
        from sklearn.metrics import roc_auc_score, average_precision_score
        
        # Create dataloader
        dataloader = DataLoader(dataset, batch_size=self.hparams.dataset.val_batch_size, shuffle=False)
        
        # Collect predictions for each patient
        patient_predictions = []  # Max risk across timesteps for each patient
        patient_labels = []       # Whether patient ever has event in bucket
        
        self.eval()
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                # Move batch to device
                for key in batch:
                    if isinstance(batch[key], torch.Tensor):
                        batch[key] = batch[key].to(self.device)
                
                # Get predictions
                _, outcome_pred, _, _, _ = self(batch)
                hazard_pred = outcome_pred  # Already positive after softplus
                
                # Get batch info
                batch_size = hazard_pred.shape[0]
                seq_lengths = batch['sequence_lengths'].long()
                
                # Extract labels
                event_indicator = batch['outputs'][:, :, 1]
                event_bucket = batch['outputs'][:, :, 0]
                
                # For each patient
                for i in range(batch_size):
                    seq_len = seq_lengths[i]
                    
                    # Get all hazard predictions for this patient
                    patient_hazards = hazard_pred[i, :seq_len].cpu().numpy()
                    
                    # Calculate survival probabilities for all timesteps
                    cumulative_hazards = np.cumsum(patient_hazards, axis=1)
                    survival_probs = np.exp(-cumulative_hazards)
                    
                    # For bucket 1: take MAX risk across all timesteps
                    # This represents the highest risk prediction for this patient
                    bucket_idx = 1  # eventual bucket
                    risks = 1 - survival_probs[:, bucket_idx]
                    max_risk = risks.max()
                    
                    # Label: does patient EVER have event in bucket 1?
                    patient_events = event_indicator[i, :seq_len].cpu().numpy()
                    patient_buckets = event_bucket[i, :seq_len].cpu().numpy()
                    has_event_in_bucket = np.any((patient_events == 1) & (patient_buckets == bucket_idx))
                    
                    patient_predictions.append(max_risk)
                    patient_labels.append(float(has_event_in_bucket))
        
        # Convert to arrays
        patient_predictions = np.array(patient_predictions)
        patient_labels = np.array(patient_labels)
        
        # Calculate metrics
        n_events = patient_labels.sum()
        n_controls = len(patient_labels) - n_events
        
        metrics = {}
        
        # Always log the case/control counts
        logger.info(f"\n{dataset_name.title()} Patient-Level Bucket Metrics (Comparable to Binary):")
        logger.info(f"  Cases: {int(n_events)}")
        logger.info(f"  Controls: {int(n_controls)}")
        
        if n_events > 0 and n_controls > 0:
            try:
                auc_roc = roc_auc_score(patient_labels, patient_predictions)
                auc_pr = average_precision_score(patient_labels, patient_predictions)
                prevalence = n_events / len(patient_labels)
                
                metrics[f'{dataset_name}_patient_bucket_auc_roc'] = auc_roc
                metrics[f'{dataset_name}_patient_bucket_auc_pr'] = auc_pr
                metrics[f'{dataset_name}_patient_bucket_prevalence'] = prevalence
                metrics[f'{dataset_name}_patient_bucket_n_events'] = int(n_events)
                metrics[f'{dataset_name}_patient_bucket_n_controls'] = int(n_controls)
                
                logger.info(f"  AUROC: {auc_roc:.4f}")
                logger.info(f"  AUPRC: {auc_pr:.4f}")
                logger.info(f"  Prevalence: {100*prevalence:.2f}%")
            except Exception as e:
                logger.warning(f"Could not compute patient-level bucket metrics: {e}")
        
        return metrics
    
    def calculate_treatment_pearson_r(self, dataset, dataset_name='val'):
        """Calculate Pearson R for treatment predictions across all active timepoints
        
        Args:
            dataset: Dataset to evaluate
            dataset_name: Name for logging ('val' or 'test')
            
        Returns:
            Pearson R correlation coefficient
        """
        from scipy.stats import pearsonr
        
        # Create dataloader
        dataloader = DataLoader(dataset, batch_size=self.hparams.dataset.val_batch_size, shuffle=False)
        
        all_treatment_preds = []
        all_treatment_true = []
        all_active_masks = []
        
        self.eval()
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                # Move batch to device
                for key in batch:
                    if isinstance(batch[key], torch.Tensor):
                        batch[key] = batch[key].to(self.device)
                
                # Get predictions
                treatment_pred, _, _, _ = self(batch)
                
                # Collect predictions, true values, and active masks
                all_treatment_preds.append(treatment_pred.cpu())
                all_treatment_true.append(batch['current_treatments'].cpu())
                all_active_masks.append(batch['active_entries'].squeeze(-1).cpu())
        
        # Concatenate all batches
        all_treatment_preds = torch.cat(all_treatment_preds, dim=0)
        all_treatment_true = torch.cat(all_treatment_true, dim=0)
        all_active_masks = torch.cat(all_active_masks, dim=0)
        
        # Flatten and apply mask to get only active timepoints
        treatment_preds_flat = all_treatment_preds[all_active_masks > 0].numpy()
        treatment_true_flat = all_treatment_true[all_active_masks > 0].numpy()
        
        # Calculate Pearson R for each treatment dimension and average
        n_treatments = treatment_preds_flat.shape[1]
        pearson_rs = []
        
        for i in range(n_treatments):
            # Check if either array is constant (no variance)
            if np.std(treatment_true_flat[:, i]) > 1e-8 and np.std(treatment_preds_flat[:, i]) > 1e-8:
                r, _ = pearsonr(treatment_true_flat[:, i], treatment_preds_flat[:, i])
                if not np.isnan(r):
                    pearson_rs.append(r)
        
        # Return average Pearson R across all treatments
        avg_pearson_r = np.mean(pearson_rs) if pearson_rs else 0.0
        return avg_pearson_r
    
    def export_predictions(self, dataset, dataset_name='val'):
        """Export binary predictions with person IDs for subgroup analysis
        
        Args:
            dataset: Dataset to evaluate
            dataset_name: Name for export file ('val' or 'test')
            
        Returns:
            DataFrame with person_id, y_true, y_prob
        """
        import pandas as pd
        import os
        import pickle
        
        # Load patient splits from saved pickle file
        from pathlib import Path
        # Get the root directory of the project
        root_dir = Path(__file__).parent.parent.parent
        splits_path = root_dir / 'outputs' / 'patient_splits_survival.pkl'
        
        if not splits_path.exists():
            logger.error(f"Patient splits file not found at {splits_path}")
            logger.info("Using fallback method - assuming sequential order matches dataset order")
            # Fallback: just use indices if splits file doesn't exist
            person_ids_for_split = list(range(len(dataset)))
        else:
            with open(splits_path, 'rb') as f:
                patient_splits = pickle.load(f)
            
            # Get the person IDs for this dataset split
            if dataset_name == 'val':
                person_ids_for_split = patient_splits['val']
            elif dataset_name == 'test':
                person_ids_for_split = patient_splits['test']
            else:
                raise ValueError(f"Unknown dataset_name: {dataset_name}")
        
        # Create dataloader
        dataloader = DataLoader(dataset, batch_size=self.hparams.dataset.val_batch_size, shuffle=False)
        
        all_person_ids = []
        all_y_true = []
        all_y_prob = []
        
        patient_idx = 0
        
        self.eval()
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                # Move batch to device
                for key in batch:
                    if isinstance(batch[key], torch.Tensor):
                        batch[key] = batch[key].to(self.device)
                
                # Get predictions from binary classification head (3rd head)
                _, _, binary_pred, _ = self(batch)
                
                # Extract binary predictions at last timestep
                if binary_pred.dim() == 3:
                    binary_logits = binary_pred[:, -1, 0]  # Shape: (batch_size,)
                elif binary_pred.dim() == 2 and binary_pred.shape[1] > 1:
                    binary_logits = binary_pred[:, -1]
                else:
                    binary_logits = binary_pred.squeeze(-1)
                
                # Convert to probabilities
                binary_probs = torch.sigmoid(binary_logits)
                
                # Get true labels (ever disease)
                event_indicator = batch['outputs'][:, :, 1]
                binary_labels = (event_indicator.max(dim=1)[0] > 0).float()
                
                # Get person IDs for this batch
                batch_size = binary_probs.shape[0]
                batch_person_ids = person_ids_for_split[patient_idx:patient_idx + batch_size]
                patient_idx += batch_size
                
                all_person_ids.extend(batch_person_ids)
                all_y_true.extend(binary_labels.cpu().numpy())
                all_y_prob.extend(binary_probs.cpu().numpy())
        
        # Create DataFrame
        df_predictions = pd.DataFrame({
            'person_id': all_person_ids,
            'y_true': all_y_true,
            'y_prob': all_y_prob
        })
        
        # Save to file
        output_dir = root_dir / 'outputs'
        output_dir.mkdir(exist_ok=True)
        output_path = output_dir / f'predictions_{dataset_name}.csv'
        df_predictions.to_csv(output_path, index=False)
        logger.info(f"Exported {len(df_predictions)} predictions to {output_path}")
        
        return df_predictions
    
    def export_survival_predictions(self, dataset, dataset_name='val'):
        """Export survival hazard predictions from 4th head with person IDs for analysis
        
        Args:
            dataset: Dataset to evaluate
            dataset_name: Name for export file ('val' or 'test')
            
        Returns:
            DataFrame with person_id, hazard predictions for each bucket, true labels
        """
        import pandas as pd
        import os
        import pickle
        
        # Load patient splits from saved pickle file
        from pathlib import Path
        # Get the root directory of the project
        root_dir = Path(__file__).parent.parent.parent
        splits_path = root_dir / 'outputs' / 'patient_splits_survival.pkl'
        
        if not splits_path.exists():
            logger.error(f"Patient splits file not found at {splits_path}")
            logger.info("Using fallback method - assuming sequential order matches dataset order")
            # Fallback: just use indices if splits file doesn't exist
            person_ids_for_split = list(range(len(dataset)))
        else:
            with open(splits_path, 'rb') as f:
                patient_splits = pickle.load(f)
            
            # Get the person IDs for this dataset split
            if dataset_name == 'val':
                person_ids_for_split = patient_splits['val']
            elif dataset_name == 'test':
                person_ids_for_split = patient_splits['test']
            else:
                raise ValueError(f"Unknown dataset_name: {dataset_name}")
        
        # Create dataloader
        dataloader = DataLoader(dataset, batch_size=self.hparams.dataset.val_batch_size, shuffle=False)
        
        # Get number of hazard buckets
        n_buckets = self.num_hazard_buckets if hasattr(self, 'num_hazard_buckets') else 2
        
        all_person_ids = []
        all_hazard_preds = {f'hazard_bucket_{i}': [] for i in range(n_buckets)}
        all_event_times = []
        all_event_indicators = []
        all_y_true = []  # Binary ever-disease label
        
        patient_idx = 0
        
        self.eval()
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                # Move batch to device
                for key in batch:
                    if isinstance(batch[key], torch.Tensor):
                        batch[key] = batch[key].to(self.device)
                
                # Get predictions from survival head (4th head)
                _, _, binary_pred, binary_treatment_pred = self(batch)
                
                # Extract survival hazard predictions at last timestep
                # binary_treatment_pred shape: (batch_size, seq_len, n_buckets)
                if binary_treatment_pred.dim() == 3:
                    # For seq2one models, we want the last timestep
                    if hasattr(self.hparams.dataset, 'last_timepoint_only') and self.hparams.dataset.last_timepoint_only:
                        if 'sequence_lengths' in batch:
                            seq_lengths = batch['sequence_lengths']
                            hazard_preds_last = []
                            for i in range(binary_treatment_pred.shape[0]):
                                seq_len = seq_lengths[i]
                                hazard_preds_last.append(binary_treatment_pred[i, seq_len-1])
                            hazard_preds_last = torch.stack(hazard_preds_last)
                        else:
                            hazard_preds_last = binary_treatment_pred[:, -1]  # Shape: (batch_size, n_buckets)
                    else:
                        # For seq2seq, take the last timestep
                        hazard_preds_last = binary_treatment_pred[:, -1]
                elif binary_treatment_pred.dim() == 2:
                    # Already reduced to (batch_size, n_buckets)
                    hazard_preds_last = binary_treatment_pred
                else:
                    raise ValueError(f"Unexpected prediction shape: {binary_treatment_pred.shape}")
                
                # Get true labels
                if not hasattr(self, 'sliding_windows'):
                    self.sliding_windows = self.dataset_collection.sliding_windows if self.dataset_collection else False
                
                if not self.sliding_windows:
                    # For non-sliding windows, extract from outputs
                    event_time_bucket_full = batch['outputs'][:, :, 0]
                    event_indicator_full = batch['outputs'][:, :, 1]
                    event_time_bucket_last = event_time_bucket_full[:, -1].long()
                    event_indicator_last = event_indicator_full[:, -1]
                    # Binary label: ever disease
                    binary_labels = (event_indicator_full.max(dim=1)[0] > 0).float()
                else:
                    # For sliding windows, use direct values
                    event_time_bucket_last = batch['event_time_bucket']
                    event_indicator_last = batch['event_indicator']
                    # For sliding windows, need to check if patient ever has event
                    binary_labels = (event_indicator_last > 0).float()
                
                # Get person IDs for this batch
                batch_size = hazard_preds_last.shape[0]
                batch_person_ids = person_ids_for_split[patient_idx:patient_idx + batch_size]
                patient_idx += batch_size
                
                # Store predictions
                all_person_ids.extend(batch_person_ids)
                all_event_times.extend(event_time_bucket_last.cpu().numpy())
                all_event_indicators.extend(event_indicator_last.cpu().numpy())
                all_y_true.extend(binary_labels.cpu().numpy())
                
                # Store hazard predictions for each bucket
                for i in range(n_buckets):
                    all_hazard_preds[f'hazard_bucket_{i}'].extend(hazard_preds_last[:, i].cpu().numpy())
        
        # Create DataFrame
        df_data = {
            'person_id': all_person_ids,
            'event_time_bucket': all_event_times,
            'event_indicator': all_event_indicators,
            'y_true': all_y_true  # Binary ever-disease label
        }
        
        # Add hazard predictions for each bucket
        for bucket_key in all_hazard_preds:
            df_data[bucket_key] = all_hazard_preds[bucket_key]
        
        df_survival_predictions = pd.DataFrame(df_data)
        
        # Save to file
        output_dir = root_dir / 'outputs'
        output_dir.mkdir(exist_ok=True)
        output_path = output_dir / f'survival_predictions_{dataset_name}.csv'
        df_survival_predictions.to_csv(output_path, index=False)
        logger.info(f"Exported {len(df_survival_predictions)} survival predictions to {output_path}")
        
        return df_survival_predictions
    
    def on_validation_epoch_end(self):
        """Calculate and display metrics at end of validation epoch"""
        # Calculate bucket metrics for bucket-based approach
        bucket_metrics = self.calculate_bucket_metrics(self.dataset_collection.val_f, 'val')
        
        # Calculate binary classification metrics
        binary_metrics = self.calculate_binary_metrics(self.dataset_collection.val_f, 'val')
        
        # Calculate treatment Pearson R
        treatment_pearson_r = self.calculate_treatment_pearson_r(self.dataset_collection.val_f, 'val')
        logger.info(f"\nValidation Treatment Pearson R (avg across treatments): {treatment_pearson_r:.4f}")
        
        # Display binary classification metrics
        logger.info(f"\nValidation Binary Classification Metrics (Ever Disease):")
        logger.info(f"  AUROC: {binary_metrics.get('val_binary_auc_roc', np.nan):.4f}")
        logger.info(f"  AUPRC: {binary_metrics.get('val_binary_auc_pr', np.nan):.4f}")
        logger.info(f"  Cases: {binary_metrics.get('val_binary_n_cases', 0)}")
        logger.info(f"  Controls: {binary_metrics.get('val_binary_n_controls', 0)}")
        logger.info(f"  Prevalence: {binary_metrics.get('val_binary_prevalence', np.nan):.2%}")
        
        # Display bucket-specific metrics
        if bucket_metrics:
            logger.info("\nValidation Metrics by Bucket:")
            logger.info("Bucket         | AUC-ROC | AUC-PR | Prevalence | Events/No-Events")
            logger.info("---------------|---------|--------|------------|------------------")
            
            for bucket_name in sorted(bucket_metrics.keys()):
                metrics = bucket_metrics[bucket_name]
                logger.info(f"{bucket_name:<14} | {metrics['auc_roc']:>7.4f} | {metrics['auc_pr']:>6.4f} | "
                          f"{metrics['prevalence']:>9.1f}% | {metrics['n_events']:>3}/{metrics['n_controls']:<3}")
        else:
            pass
        
        # Export predictions for both binary and survival heads
        self.export_predictions(self.dataset_collection.val_f, 'val')
        self.export_survival_predictions(self.dataset_collection.val_f, 'val')
    
    def test_step(self, batch, batch_idx):
        """Test step for survival models - identical to validation step"""
        # Get predictions
        treatment_pred, _, binary_pred, binary_treatment_pred = self(batch)
        
        # Calculate binary classification loss
        # Get disease labels from outputs
        event_indicator = batch['outputs'][:, :, 1]  # event_indicator column
        binary_labels = (event_indicator.max(dim=1)[0] > 0).float()
        
        # Get last timepoint prediction for binary classification
        if binary_pred.dim() == 3:
            binary_pred_last = binary_pred[:, -1, 0]
        elif binary_pred.dim() == 2 and binary_pred.shape[1] > 1:
            binary_pred_last = binary_pred[:, -1]
        else:
            binary_pred_last = binary_pred.squeeze(-1)
            
        binary_bce_loss = F.binary_cross_entropy_with_logits(
            binary_pred_last, 
            binary_labels,
            reduction='mean'
        )
        
        # Fourth head loss: PC-Hazard survival loss with treatment
        if not hasattr(self, 'sliding_windows'):
            self.sliding_windows = self.dataset_collection.sliding_windows if self.dataset_collection else False
        
        if not self.sliding_windows:
            # For non-sliding windows, extract from outputs
            event_time_bucket_full = batch['outputs'][:, :, 0]
            event_indicator_full = batch['outputs'][:, :, 1]
            event_time_bucket_last = event_time_bucket_full[:, -1].long()
            event_indicator_last = event_indicator_full[:, -1]
        else:
            # For sliding windows, use direct values
            event_time_bucket_last = batch['event_time_bucket']
            event_indicator_last = batch['event_indicator']
        
        # Apply PC-Hazard loss to fourth head
        binary_treatment_survival_loss = pc_hazard_loss(
            binary_treatment_pred,
            event_time_bucket_last,
            event_indicator_last,
            sequence_lengths=batch['sequence_lengths'] if 'sequence_lengths' in batch else None,
            last_timepoint_only=self.hparams.dataset.get('last_timepoint_only', True),
            reduction='mean',
            focal_gamma=self.hparams.dataset.get('focal_gamma', 0.0)
        )
        
        # Equal weights for both heads
        outcome_combined = (binary_bce_loss + binary_treatment_survival_loss) / 2.0
        
        # Log test losses
        self.log('test_loss', outcome_combined, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('test_binary_bce_loss', binary_bce_loss, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log('test_binary_treatment_survival_loss', binary_treatment_survival_loss, on_epoch=True, prog_bar=False, sync_dist=True)
        
        # Horizon-specific metrics will be calculated in on_test_epoch_end
        return outcome_combined
    
    def on_test_epoch_end(self):
        """Calculate and display metrics at end of test epoch"""
        logger.info("\nCalculating test metrics...")
        
        if not hasattr(self.dataset_collection, 'test_f') or self.dataset_collection.test_f is None:
            logger.warning("No test dataset available")
            return
            
        # Calculate bucket metrics for bucket-based approach
        bucket_metrics = self.calculate_bucket_metrics(self.dataset_collection.test_f, 'test')
        
        # Calculate patient-level bucket metrics (comparable to binary)
        patient_bucket_metrics = self.calculate_patient_level_bucket_metrics(self.dataset_collection.test_f, 'test')
        
        # Calculate treatment Pearson R
        treatment_pearson_r = self.calculate_treatment_pearson_r(self.dataset_collection.test_f, 'test')
        logger.info(f"\nTest Treatment Pearson R (avg across treatments): {treatment_pearson_r:.4f}")
        
        # Calculate binary classification metrics
        binary_metrics = self.calculate_binary_metrics(self.dataset_collection.test_f, 'test')
        
        # Display binary classification metrics
        logger.info(f"\nTest Binary Classification Metrics (Ever Disease):")
        logger.info(f"  AUROC: {binary_metrics.get('test_binary_auc_roc', np.nan):.4f}")
        logger.info(f"  AUPRC: {binary_metrics.get('test_binary_auc_pr', np.nan):.4f}")
        logger.info(f"  Prevalence: {binary_metrics.get('test_binary_prevalence', np.nan):.2%}")
        
        # Display bucket-specific metrics
        if bucket_metrics:
            logger.info("\nTest Metrics by Bucket:")
            logger.info("Bucket         | AUC-ROC | AUC-PR | Prevalence | Events/No-Events")
            logger.info("---------------|---------|--------|------------|------------------")
            
            for bucket_name in sorted(bucket_metrics.keys()):
                metrics = bucket_metrics[bucket_name]
                logger.info(f"{bucket_name:<14} | {metrics['auc_roc']:>7.4f} | {metrics['auc_pr']:>6.4f} | "
                          f"{metrics['prevalence']:>9.1f}% | {metrics['n_events']:>3}/{metrics['n_controls']:<3}")
        else:
            pass
        
        # Export predictions for both binary and survival heads
        self.export_predictions(self.dataset_collection.test_f, 'test')
        self.export_survival_predictions(self.dataset_collection.test_f, 'test')
        
        # Skip old horizon metrics
        horizon_metrics = None  # self.calculate_horizon_metrics(self.dataset_collection.test_f, 'test')
        
        if horizon_metrics:
            # Print header
            logger.info("\nTest Metrics by Horizon:")
            logger.info("Horizon | AUC-ROC | AUC-PR | Prevalence | Events/Controls")
            logger.info("--------|---------|--------|------------|----------------")
            
            # Print metrics in table format
            for horizon in sorted(horizon_metrics.keys()):
                metrics = horizon_metrics[horizon]
                logger.info(f"{horizon:>6}d | {metrics['auc_roc']:>7.4f} | {metrics['auc_pr']:>6.4f} | "
                          f"{metrics['prevalence']:>9.1f}% | {metrics['n_events']:>3}/{metrics['n_controls']:<3}")
        else:
            logger.warning("No horizon metrics were calculated for test set")

    def predict_step(self, batch, batch_idx, dataset_idx=None):
        """
        Generates normalised output predictions
        """
        if self.hparams.exp.weights_ema:
            with self.ema_non_treatment.average_parameters():
                _, br, binary_pred, _ = self(batch)
        else:
            _, br, binary_pred, _ = self(batch)
        
        return binary_pred.cpu(), br.cpu()

    def get_representations(self, dataset: Dataset) -> np.array:
        logger.info(f'Balanced representations inference for {dataset.subset_name}.')
        # Creating Dataloader
        data_loader = DataLoader(dataset, batch_size=self.hparams.dataset.val_batch_size, shuffle=False)
        _, br = [torch.cat(arrs) for arrs in zip(*self.trainer.predict(self, data_loader))]
        return br.numpy()

    def get_predictions(self, dataset: Dataset) -> np.array:
        logger.info(f'Predictions for {dataset.subset_name}.')
        # Creating Dataloader
        data_loader = DataLoader(dataset, batch_size=self.hparams.dataset.val_batch_size, shuffle=False)
        binary_pred, _ = [torch.cat(arrs) for arrs in zip(*self.trainer.predict(self, data_loader))]
        return binary_pred.numpy()


class LossBreakdownCallback(Callback):
    """Callback to print loss breakdown at the end of each epoch"""
    
    def on_train_epoch_end(self, trainer, pl_module):
        # Get the logged metrics
        metrics = trainer.callback_metrics
        model_type = pl_module.model_type
        
        # Prepare loss breakdown message
        loss_parts = []
        
        # Total loss
        if f'{model_type}_train_loss' in metrics:
            total_loss = metrics[f'{model_type}_train_loss'].item()
            loss_parts.append(f"Total Loss: {total_loss:.4f}")
        
        # Treatment prediction loss
        if f'{model_type}_train_bce_loss' in metrics:
            bce_loss = metrics[f'{model_type}_train_bce_loss'].item()
            loss_parts.append(f"Treatment BCE (confuse): {bce_loss:.8f}")
        
        # Domain classifier loss (actual treatment prediction)
        if f'{model_type}_train_bce_loss_cl' in metrics:
            bce_loss_cl = metrics[f'{model_type}_train_bce_loss_cl'].item()
            loss_parts.append(f"Treatment BCE (classify): {bce_loss_cl:.6f}")
        
        # Treatment Pearson R (how well are treatments predicted)
        if f'{model_type}_train_treatment_pearson_r' in metrics:
            pearson_r = metrics[f'{model_type}_train_treatment_pearson_r'].item()
            loss_parts.append(f"Treatment Pearson R: {pearson_r:.4f}")
        
        # Binary BCE loss
        if f'{model_type}_train_binary_bce_loss' in metrics:
            binary_bce_loss = metrics[f'{model_type}_train_binary_bce_loss'].item()
            loss_parts.append(f"Binary BCE Loss: {binary_bce_loss:.4f}")
        
        # Combined outcome loss
        if f'{model_type}_train_outcome_combined_loss' in metrics:
            outcome_combined = metrics[f'{model_type}_train_outcome_combined_loss'].item()
            loss_parts.append(f"Outcome Combined Loss: {outcome_combined:.4f}")
        
        # GradNorm weights if enabled
        if f'{model_type}_gradnorm_binary_weight' in metrics and f'{model_type}_gradnorm_treatment_survival_weight' in metrics:
            binary_weight = metrics[f'{model_type}_gradnorm_binary_weight'].item()
            treatment_survival_weight = metrics[f'{model_type}_gradnorm_treatment_survival_weight'].item()
            loss_parts.append(f"GradNorm Weights - Binary: {binary_weight:.3f}, Treatment Survival: {treatment_survival_weight:.3f}")
        
        # Add validation metrics if available
        val_parts = []
        
        # Total validation loss
        if 'val_loss' in metrics:
            val_loss = metrics['val_loss'].item()
            val_parts.append(f"Total Val Loss: {val_loss:.4f}")
        


class GradNormCallback(Callback):
    """Callback to apply GradNorm for balancing multitask losses in survival models
    
    Reference: "GradNorm: Gradient Normalization for Adaptive Loss Balancing in Deep Multitask Networks"
    """
    
    def __init__(self, alpha=1.5, update_every_n_steps=1):
        self.alpha = alpha
        self.update_every_n_steps = update_every_n_steps
        self.initial_losses = None
        self.step_count = 0
        
    def on_train_start(self, trainer, pl_module):
        """Initialize initial loss values on first epoch"""
        if hasattr(pl_module, 'log_task_weights'):
            logger.info(f"GradNorm enabled with alpha={self.alpha}")
            # Will initialize after first batch
            self.initial_losses = None
    
    def on_after_backward(self, trainer, pl_module):
        """Apply GradNorm after gradients are computed"""
        if not hasattr(pl_module, 'log_task_weights'):
            return
            
        # Only update every n steps
        self.step_count += 1
        if self.step_count % self.update_every_n_steps != 0:
            return
            
        # Skip if we don't have the last losses stored
        if not hasattr(pl_module, 'last_binary_loss') or not hasattr(pl_module, 'last_treatment_survival_loss'):
            return
            
        # Get current task losses
        task_losses = torch.stack([pl_module.last_binary_loss.to(pl_module.device), 
                                  pl_module.last_treatment_survival_loss.to(pl_module.device)])
        
        # Initialize initial losses on first batch
        if self.initial_losses is None:
            self.initial_losses = task_losses.clone().detach()
            pl_module.initial_task_losses.copy_(self.initial_losses)
            logger.info(f"GradNorm initialized with initial losses: binary={self.initial_losses[0]:.4f}, treatment_survival={self.initial_losses[1]:.4f}")
            return
        
        # Get current task weights
        with torch.no_grad():
            task_weights = F.softmax(pl_module.log_task_weights, dim=0)
        
        # Compute relative losses (L_i(t) / L_i(0))
        relative_losses = task_losses / (self.initial_losses + 1e-8)
        
        # Compute average relative loss
        avg_relative_loss = relative_losses.mean()
        
        # Compute relative inverse training rates
        relative_inverse_rates = relative_losses / (avg_relative_loss + 1e-8)
        
        # Compute desired gradient magnitudes
        desired_grad_norms = (relative_inverse_rates ** self.alpha).detach()
        
        # Get gradient norms w.r.t task weights
        if pl_module.log_task_weights.grad is not None:
            # Get gradients of weighted loss w.r.t. last shared layer
            last_shared_layer = None
            for name, param in pl_module.named_parameters():
                if 'br_treatment_outcome_head.linear1' in name and param.requires_grad:
                    last_shared_layer = param
                    break
            
            if last_shared_layer is not None and last_shared_layer.grad is not None:
                # Compute gradient norms for each task
                grad_norms = []
                for i in range(2):
                    # Get gradient of task i loss w.r.t. shared parameters
                    task_grad = torch.autograd.grad(
                        task_losses[i] * task_weights[i],
                        last_shared_layer,
                        retain_graph=True,
                        create_graph=True
                    )[0]
                    grad_norms.append(torch.norm(task_grad))
                
                grad_norms = torch.stack(grad_norms)
                
                # Normalize gradient norms
                mean_grad_norm = grad_norms.mean().detach()
                grad_norms = grad_norms / (mean_grad_norm + 1e-8)
                
                # Compute GradNorm loss
                gradnorm_loss = torch.abs(grad_norms - desired_grad_norms).sum()
                
                # Compute gradients of GradNorm loss w.r.t. task weights
                task_weight_grads = torch.autograd.grad(
                    gradnorm_loss,
                    pl_module.log_task_weights,
                    retain_graph=True
                )[0]
                
                # Update task weight gradients
                if pl_module.log_task_weights.grad is None:
                    pl_module.log_task_weights.grad = task_weight_grads
                else:
                    pl_module.log_task_weights.grad += task_weight_grads
    
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        """Log GradNorm statistics"""
        if hasattr(pl_module, 'log_task_weights') and self.step_count % 100 == 0:
            with torch.no_grad():
                task_weights = F.softmax(pl_module.log_task_weights, dim=0)
                pl_module.log('gradnorm_task0_weight', task_weights[0], on_step=True, prog_bar=False)
                pl_module.log('gradnorm_task1_weight', task_weights[1], on_step=True, prog_bar=False)
