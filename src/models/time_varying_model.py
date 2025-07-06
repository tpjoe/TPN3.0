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

from src.data import RealDatasetCollection, SyntheticDatasetCollection
from src.models.utils import grad_reverse, BRTreatmentOutcomeHead, AlphaRise, bce

logger = logging.getLogger(__name__)
ray_constants.FUNCTION_SIZE_ERROR_THRESHOLD = 10**8  # ~ 100Mb


def train_eval_factual(args: dict, train_f: Dataset, val_f: Dataset, orig_hparams: DictConfig, input_size: int, model_cls,
                       tuning_criterion='rmse', **kwargs):
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

    if tuning_criterion == 'rmse':
        val_rmse_orig, val_rmse_all = model.get_normalised_masked_rmse(val_f)
        tune.report(val_rmse_orig=val_rmse_orig, val_rmse_all=val_rmse_all)
    elif tuning_criterion == 'bce':
        val_bce_orig, val_bce_all = model.get_masked_bce(val_f)
        tune.report(val_bce_orig=val_bce_orig, val_bce_all=val_bce_all)
    else:
        raise NotImplementedError()


class TimeVaryingCausalModel(LightningModule):
    """
    Abstract class for models, estimating counterfactual outcomes over time
    """

    model_type = None  # Will be defined in subclasses
    possible_model_types = None  # Will be defined in subclasses
    tuning_criterion = None

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
            print(self.bce_weights)

        # General datasets parameters
        self.dim_treatments = args.model.dim_treatments
        self.dim_vitals = args.model.dim_vitals
        self.dim_static_features = args.model.dim_static_features
        self.dim_outcome = args.model.dim_outcomes
        # Store original list for multitask models
        if isinstance(self.dim_outcome, (list, ListConfig)):
            self.dim_outcome_list = list(self.dim_outcome)
            # Initialize GradNorm parameters for multitask learning if enabled
            self.num_tasks = len(self.dim_outcome_list)
            use_gradnorm = getattr(args.dataset, 'use_gradnorm', False) if 'dataset' in args else False
            if self.num_tasks > 1 and use_gradnorm:
                # Task weights for GradNorm (learnable)
                self.register_parameter('task_weights', nn.Parameter(torch.ones(self.num_tasks)))
                # Initial task losses for relative training rate
                self.register_buffer('initial_task_losses', torch.zeros(self.num_tasks))
                self.register_buffer('task_loss_history', torch.zeros(self.num_tasks))
                self.gradnorm_alpha = getattr(args.dataset, 'gradnorm_alpha', 1.5) if 'dataset' in args else 1.5
                self.training_steps = 0
        else:
            self.dim_outcome_list = None
        
        # Debug output
        # print(f"TimeVaryingModel init - dim_outcome: {self.dim_outcome}, type: {type(self.dim_outcome)}")
        # print(f"TimeVaryingModel init - dim_outcome_list: {self.dim_outcome_list}")

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

    def get_masked_bce(self, dataset: Dataset):
        logger.info(f'BCE calculation for {dataset.subset_name}.')
        treatment_pred = torch.tensor(self.get_propensity_scores(dataset))
        current_treatments = torch.tensor(dataset.data['current_treatments'])

        bce = (self.bce_loss(treatment_pred, current_treatments, kind='predict')).unsqueeze(-1).numpy()
        bce = bce * dataset.data['active_entries']

        # Calculation like in original paper (Masked-Averaging over datapoints (& outputs) and then non-masked time axis)
        bce_orig = bce.sum(0).sum(-1) / dataset.data['active_entries'].sum(0).sum(-1)
        bce_orig = bce_orig.mean()

        # Masked averaging over all dimensions at once
        bce_all = bce.sum() / dataset.data['active_entries'].sum()

        return bce_orig, bce_all

    def get_binary_classification_metrics(self, dataset: Dataset, min_seq_length: int = None, outcome_idx: int = None):
        """
        Calculate binary classification metrics: AUC-ROC and AUC-PR using last prediction per patient.
        
        Args:
            dataset: Dataset to evaluate
            min_seq_length: If provided, only include patients with at least this many timesteps
            outcome_idx: If provided, only evaluate this outcome index (for multi-outcome models)
        """
        from sklearn.metrics import roc_auc_score, average_precision_score
        
        logger.info(f'Binary classification metrics calculation for {dataset.subset_name}.')
        
        # Get predictions (logits)
        outputs_logits = self.get_predictions(dataset)
        # Extract specific outcome if specified
        if outcome_idx is not None:
            # Get dimension info from config
            # For multitask models with combined head, we know the structure from dataset
            if hasattr(self.dataset_collection, 'outcome_columns') and len(self.dataset_collection.outcome_columns) > 1:
                # Reconstruct dimensions from config - all outcomes have dim 1 in our case
                dim_list = [1] * len(self.dataset_collection.outcome_columns)
                dim_start = sum(dim_list[:outcome_idx])
                dim_end = dim_start + dim_list[outcome_idx]
                # Extract the specific outcome
            else:
                # Single outcome case
                dim_start = 0
                # Handle both list and scalar cases
                if isinstance(self.dim_outcome, (list, ListConfig)):
                    dim_end = sum(self.dim_outcome)  # Sum all dimensions
                else:
                    dim_end = self.dim_outcome
            
            # Only slice if we have multiple output dimensions
            if outputs_logits.shape[-1] > 1:
                outputs_logits = outputs_logits[:, :, dim_start:dim_end]
        
        # Apply sigmoid to get probabilities
        outputs_probs = 1 / (1 + np.exp(-outputs_logits))
        
        # Get last valid prediction per patient
        n_patients, n_timesteps, n_outputs = outputs_probs.shape
        y_true_last = []
        y_probs_last = []
        patient_indices_used = []
        
        # Get sequence lengths if filtering is requested
        if min_seq_length is not None and 'sequence_lengths' in dataset.data:
            sequence_lengths = dataset.data['sequence_lengths']
        else:
            sequence_lengths = None
        
        for i in range(n_patients):
            # Skip patients with insufficient sequence length
            if sequence_lengths is not None and sequence_lengths[i] < min_seq_length:
                continue
                
            # Find the last active timestep for this patient
            active_mask = dataset.data['active_entries'][i, :, 0].astype(bool)
            if active_mask.any():
                last_active_idx = np.where(active_mask)[0][-1]
                if outcome_idx is not None:
                    # Extract specific outcome - outputs_probs is already sliced, but dataset.data['outputs'] is not
                    y_true_last.append(dataset.data['outputs'][i, last_active_idx, outcome_idx])
                    y_probs_last.append(outputs_probs[i, last_active_idx, 0])  # Already sliced above
                else:
                    y_true_last.append(dataset.data['outputs'][i, last_active_idx, 0])
                    y_probs_last.append(outputs_probs[i, last_active_idx, 0])
                patient_indices_used.append(i)
        
        y_true_last = np.array(y_true_last)
        y_probs_last = np.array(y_probs_last)
        
        # Calculate metrics
        if len(np.unique(y_true_last)) > 1:
            auc_roc = roc_auc_score(y_true_last, y_probs_last)
            auc_pr = average_precision_score(y_true_last, y_probs_last)
        else:
            auc_roc = np.nan
            auc_pr = np.nan
        
        logger.info(f'Using {len(y_true_last)} patients\' last predictions for AUC calculation')
        if min_seq_length is not None:
            logger.info(f'  (filtered to patients with at least {min_seq_length} timesteps)')
        
        return auc_roc, auc_pr

    def evaluate_one_seq_per_patient_binary(self, dataset: Dataset, projection_horizon: int = 5, use_ground_truth_feedback: bool = False):
        """
        Evaluate binary classification where each patient contributes exactly one sequence per n-step.
        Assumes dataset has been prepared with create_one_seq_per_patient_for_n_step().
        
        Args:
            dataset: Dataset with n_step_datasets attribute
            projection_horizon: Maximum prediction horizon
            use_ground_truth_feedback: If True, use ground truth outcomes as feedback for autoregressive prediction
        """
        from sklearn.metrics import roc_auc_score, average_precision_score
        from torch.utils.data import DataLoader
        
        logger.info(f'Binary classification metrics (one seq per patient) on {dataset.subset_name}')
        if use_ground_truth_feedback:
            logger.info('Using ground truth feedback for autoregressive prediction')
            
        # Check if dataset has been exploded
        if hasattr(dataset, 'exploded') and dataset.exploded:
            logger.warning('Dataset has been exploded - results may not be comparable to teacher forcing')
        
        # Don't take early returns - let the main loop handle all cases
        # This ensures we properly evaluate 0-step when projection_horizon=0
        
        # First create the one-seq-per-patient datasets if not already done
        if not hasattr(dataset, 'n_step_datasets'):
            dataset.create_one_seq_per_patient_for_n_step(projection_horizon)
        
        auc_rocs = []
        auc_prs = []
        
        # For each n-step dataset
        logger.info(f'Evaluating {len(dataset.n_step_datasets)} n-step datasets: {list(dataset.n_step_datasets.keys())}')
        for n_step, n_step_data in dataset.n_step_datasets.items():
            # n_step is already the step number (1, 2, 3, etc.)
            
            # Get predictions using the history portion of the data
            n_patients = len(n_step_data['sequence_lengths'])
            
            if use_ground_truth_feedback and n_step > 1:
                # For multi-step prediction with ground truth feedback, we need to predict step by step
                # This is more complex and requires iterative prediction
                logger.info(f'Using ground truth feedback for {n_step}-step prediction')
                all_preds = []
                all_labels = []
                
                # Process each patient individually for autoregressive prediction
                self.eval()
                with torch.no_grad():
                    for idx in range(n_patients):
                        history_length = int(n_step_data['sequence_lengths'][idx])
                        if history_length > 0:
                            # Get the original patient index and full sequence data
                            patient_idx = n_step_data['original_patient_idx'][idx]
                            true_future = n_step_data['true_future_outputs']
                            
                            # Create a working copy of the data for this patient
                            # We'll expand this as we make predictions
                            from copy import deepcopy
                            working_data = {}
                            
                            # Start with the history portion
                            for key in ['prev_treatments', 'current_treatments', 'static_features', 
                                       'prev_outputs', 'outputs', 'active_entries']:
                                if key in n_step_data:
                                    # Create arrays with room for the full sequence (history + n_step)
                                    if key == 'static_features':
                                        working_data[key] = n_step_data[key][idx:idx+1]
                                    else:
                                        # Create zero-padded arrays
                                        orig_shape = n_step_data[key][idx:idx+1].shape
                                        new_shape = list(orig_shape)
                                        new_shape[1] = history_length + n_step  # Extend sequence length
                                        working_data[key] = np.zeros(new_shape)
                                        # Copy the history portion
                                        working_data[key][:, :history_length] = n_step_data[key][idx:idx+1, :history_length]
                            
                            if 'vitals' in n_step_data:
                                orig_shape = n_step_data['vitals'][idx:idx+1].shape
                                new_shape = list(orig_shape)
                                new_shape[1] = history_length + n_step
                                working_data['vitals'] = np.zeros(new_shape)
                                working_data['vitals'][:, :history_length] = n_step_data['vitals'][idx:idx+1, :history_length]
                            
                            # Perform iterative prediction with ground truth feedback
                            for step in range(n_step):
                                # Update sequence length for this step
                                current_length = history_length + step
                                working_data['sequence_lengths'] = np.array([current_length])
                                
                                # Convert to tensor batch
                                batch_data = {}
                                for key, value in working_data.items():
                                    batch_data[key] = torch.tensor(value).double()
                                
                                # Make prediction
                                model_output = self(batch_data)
                                if isinstance(model_output, tuple):
                                    outcome_pred = model_output[1]
                                else:
                                    outcome_pred = model_output
                                
                                # Get prediction at the current last timestep
                                pred_logit = outcome_pred[0, current_length-1, 0].cpu().numpy()
                                
                                # If this is the final step, save the prediction
                                if step == n_step - 1:
                                    all_preds.append(pred_logit)
                                    # Get the true label at the final timestep
                                    true_label = true_future[idx, history_length + n_step - 1, 0]
                                    all_labels.append(true_label)
                                
                                # Feed ground truth for next iteration (if not the last step)
                                if step < n_step - 1:
                                    # Get ground truth at the current prediction timestep
                                    true_outcome = true_future[idx, history_length + step, 0]
                                    
                                    # Update the data with ground truth
                                    # The outcome at time t becomes prev_output at time t+1
                                    working_data['prev_outputs'][0, current_length, 0] = true_outcome
                                    working_data['outputs'][0, current_length, 0] = true_outcome
                                    working_data['active_entries'][0, current_length, 0] = 1.0
                                    
                                    # For treatments and vitals, we need to get them from the original full sequence
                                    # Get the original patient data if available
                                    if hasattr(dataset, 'data') and 'vitals' in dataset.data:
                                        # Get vitals from original dataset at the correct timestep
                                        orig_patient_idx = patient_idx
                                        future_timestep = history_length + step
                                        if 'vitals' in working_data and future_timestep < dataset.data['vitals'].shape[1]:
                                            working_data['vitals'][0, current_length] = dataset.data['vitals'][orig_patient_idx, future_timestep]
                                    
                                    # Similarly for treatments
                                    if hasattr(dataset, 'data') and 'current_treatments' in dataset.data:
                                        orig_patient_idx = patient_idx
                                        future_timestep = history_length + step
                                        if future_timestep < dataset.data['current_treatments'].shape[1]:
                                            working_data['current_treatments'][0, current_length] = dataset.data['current_treatments'][orig_patient_idx, future_timestep]
                                            # prev_treatments at t+1 is current_treatments at t
                                            if step > 0:
                                                working_data['prev_treatments'][0, current_length] = dataset.data['current_treatments'][orig_patient_idx, future_timestep-1]
                
                predictions = None  # Not used in this path
            else:
                # Original implementation for 1-step or no ground truth feedback
                predictions = []
                
                data_loader = DataLoader(range(n_patients), batch_size=256, shuffle=False)
                self.eval()
                
                with torch.no_grad():
                    for patient_indices in data_loader:
                        batch_data = {}
                        for key in ['prev_treatments', 'current_treatments', 'static_features', 
                                   'prev_outputs', 'outputs', 'active_entries', 'sequence_lengths']:
                            if key in n_step_data:
                                batch_data[key] = torch.tensor(n_step_data[key][patient_indices]).double()
                        
                        if 'vitals' in n_step_data:
                            batch_data['vitals'] = torch.tensor(n_step_data['vitals'][patient_indices]).double()
                        
                        # Get predictions
                        model_output = self(batch_data)
                        if isinstance(model_output, tuple):
                            outcome_pred = model_output[1]  # Use outcome predictions
                        else:
                            outcome_pred = model_output
                        predictions.append(outcome_pred.cpu().numpy())
                
                predictions = np.concatenate(predictions, axis=0)  # Shape: (n_patients, max_seq_length, 1)
            
            # Process predictions based on which path we took
            if use_ground_truth_feedback and n_step > 1:
                # Already collected all_preds and all_labels in the ground truth feedback path
                pass
            else:
                # Original path - collect predictions and labels
                # Get the true future outcomes
                true_future = n_step_data['true_future_outputs']
                valid_indices = dataset.valid_patient_indices
                
                # Collect predictions and labels
                all_preds = []
                all_labels = []
                patient_idx_list = []
                history_lengths_list = []
                
                for idx, patient_idx in enumerate(valid_indices):
                    history_length = int(n_step_data['sequence_lengths'][idx])
                    if history_length > 0:
                        # Get predictions for the last n_step timesteps
                        # Model predicts at each timestep, we want prediction from last history timestep
                        pred_logit = predictions[idx, history_length-1, 0]  # Prediction from last history timestep
                        
                        # Get true labels for the n_step future timesteps
                        if n_step == 0:
                            # For 0-step, we're predicting at the last timestep (no future)
                            # The label is at history_length-1 (same timestep as prediction)
                            true_label = true_future[idx, history_length-1, 0]
                            all_labels.append(true_label)
                        else:
                            true_label = true_future[idx, history_length:history_length+n_step, 0]
                            # For n-step predictions, use the last timestep (most distant prediction)
                            all_labels.append(true_label[-1])
                        
                        all_preds.append(pred_logit)
                        patient_idx_list.append(patient_idx)
                        history_lengths_list.append(history_length)
            
            # Convert to arrays and calculate metrics
            all_preds = np.array(all_preds)
            all_labels = np.array(all_labels)
        
            # Apply sigmoid to get probabilities
            all_probs = 1 / (1 + np.exp(-all_preds))
            
            # Remove NaN values
            valid_mask = ~(np.isnan(all_probs) | np.isnan(all_labels))
            all_probs = all_probs[valid_mask]
            all_labels = all_labels[valid_mask]
            
            if len(np.unique(all_labels)) > 1 and len(all_labels) > 1:
                auc_roc = roc_auc_score(all_labels, all_probs)
                auc_pr = average_precision_score(all_labels, all_probs)
            else:
                auc_roc = np.nan
                auc_pr = np.nan
                
            auc_rocs.append(auc_roc)
            auc_prs.append(auc_pr)
            
            if n_step == 0:
                logger.info(f'{n_step}-step (full sequence): {np.sum(valid_mask)} patients, AUC-ROC: {auc_roc:.3f}, AUC-PR: {auc_pr:.3f}')
            else:
                logger.info(f'{n_step}-step: {np.sum(valid_mask)} patients, AUC-ROC: {auc_roc:.3f}, AUC-PR: {auc_pr:.3f}')
        
        return auc_rocs, auc_prs

    def get_normalised_masked_rmse(self, dataset: Dataset, one_step_counterfactual=False, outcome_idx: int = None):
        logger.info(f'RMSE calculation for {dataset.subset_name}.')
        outputs_scaled = self.get_predictions(dataset)
        
        # Extract specific outcome if specified
        if outcome_idx is not None:
            # Get dimension info from config
            if isinstance(self.dim_outcome, (list, ListConfig)):
                dim_start = sum(self.dim_outcome[:outcome_idx])
                dim_end = dim_start + self.dim_outcome[outcome_idx]
            else:
                # Single outcome case - this shouldn't happen if outcome_idx is provided
                dim_start = 0
                dim_end = self.dim_outcome
            outputs_scaled = outputs_scaled[:, :, dim_start:dim_end]
            
            # Also extract corresponding ground truth
            outputs_true = dataset.data['outputs'][:, :, dim_start:dim_end]
            outputs_unscaled_true = dataset.data['unscaled_outputs'][:, :, dim_start:dim_end] if 'unscaled_outputs' in dataset.data else None
            active_entries = dataset.data['active_entries'][:, :, dim_start:dim_end]
            
            # Get unscale setting for this outcome
            if hasattr(self.hparams.exp.unscale_rmse, 'get'):
                # Dictionary format
                outcome_name = self.dataset_collection.outcome_columns[outcome_idx]
                unscale = self.hparams.exp.unscale_rmse.get(outcome_name, False)
            else:
                unscale = self.hparams.exp.unscale_rmse
        else:
            unscale = self.hparams.exp.unscale_rmse
            outputs_true = dataset.data['outputs']
            outputs_unscaled_true = dataset.data['unscaled_outputs'] if 'unscaled_outputs' in dataset.data else None
            active_entries = dataset.data['active_entries']
            
        percentage = self.hparams.exp.percentage_rmse

        if unscale:
            output_stds, output_means = dataset.scaling_params['output_stds'], dataset.scaling_params['output_means']
            
            # Handle dictionary format when we have multiple outcomes
            if isinstance(output_stds, dict) and outcome_idx is not None:
                outcome_name = self.dataset_collection.outcome_columns[outcome_idx]
                std = output_stds[outcome_name]
                mean = output_means[outcome_name]
                outputs_unscaled = outputs_scaled * std + mean
            elif isinstance(output_stds, dict):
                # Multiple outcomes but no specific index - this shouldn't happen in practice
                raise ValueError("Multiple outcomes but no outcome_idx specified for unscaling")
            else:
                # Single outcome case
                outputs_unscaled = outputs_scaled * output_stds + output_means

            # Batch-wise masked-MSE calculation is tricky, thus calculating for full dataset at once
            mse = ((outputs_unscaled - outputs_unscaled_true) ** 2) * active_entries
        else:
            # Batch-wise masked-MSE calculation is tricky, thus calculating for full dataset at once
            mse = ((outputs_scaled - outputs_true) ** 2) * active_entries

        # Calculation like in original paper (Masked-Averaging over datapoints (& outputs) and then non-masked time axis)
        mse_orig = mse.sum(0).sum(-1) / active_entries.sum(0).sum(-1)
        mse_orig = mse_orig.mean()
        rmse_normalised_orig = np.sqrt(mse_orig) / dataset.norm_const

        # Masked averaging over all dimensions at once
        mse_all = mse.sum() / active_entries.sum()
        rmse_normalised_all = np.sqrt(mse_all) / dataset.norm_const

        if percentage:
            rmse_normalised_orig *= 100.0
            rmse_normalised_all *= 100.0

        if one_step_counterfactual:
            # Only considering last active entry with actual counterfactuals
            num_samples, time_dim, output_dim = dataset.data['active_entries'].shape
            last_entries = dataset.data['active_entries'] - np.concatenate([dataset.data['active_entries'][:, 1:, :],
                                                                            np.zeros((num_samples, 1, output_dim))], axis=1)
            if unscale:
                mse_last = ((outputs_unscaled - dataset.data['unscaled_outputs']) ** 2) * last_entries
            else:
                mse_last = ((outputs_scaled - dataset.data['outputs']) ** 2) * last_entries

            mse_last = mse_last.sum() / last_entries.sum()
            rmse_normalised_last = np.sqrt(mse_last) / dataset.norm_const

            if percentage:
                rmse_normalised_last *= 100.0

            return rmse_normalised_orig, rmse_normalised_all, rmse_normalised_last

        return rmse_normalised_orig, rmse_normalised_all

    def get_normalised_n_step_rmses(self, dataset: Dataset, datasets_mc: List[Dataset] = None):
        logger.info(f'RMSE calculation for {dataset.subset_name}.')
        assert self.model_type == 'decoder' or self.model_type == 'multi' or self.model_type == 'g_net' or \
               self.model_type == 'msm_regressor'
        assert hasattr(dataset, 'data_processed_seq')

        unscale = self.hparams.exp.unscale_rmse
        percentage = self.hparams.exp.percentage_rmse
        outputs_scaled = self.get_autoregressive_predictions(dataset if datasets_mc is None else datasets_mc)

        if unscale:
            output_stds, output_means = dataset.scaling_params['output_stds'], dataset.scaling_params['output_means']
            outputs_unscaled = outputs_scaled * output_stds + output_means

            mse = ((outputs_unscaled - dataset.data_processed_seq['unscaled_outputs']) ** 2) \
                * dataset.data_processed_seq['active_entries']
        else:
            mse = ((outputs_scaled - dataset.data_processed_seq['outputs']) ** 2) * dataset.data_processed_seq['active_entries']

        nan_idx = np.unique(np.where(np.isnan(dataset.data_processed_seq['outputs']))[0])
        not_nan = np.array([i for i in range(outputs_scaled.shape[0]) if i not in nan_idx])

        # Calculation like in original paper (Masked-Averaging over datapoints (& outputs) and then non-masked time axis)
        mse_orig = mse[not_nan].sum(0).sum(-1) / dataset.data_processed_seq['active_entries'][not_nan].sum(0).sum(-1)
        rmses_normalised_orig = np.sqrt(mse_orig) / dataset.norm_const

        if percentage:
            rmses_normalised_orig *= 100.0

        # Calculate Pearson correlations for each timestep
        mask = dataset.data_processed_seq['active_entries'][not_nan].astype(bool)
        pearson_rs = []
        for t in range(outputs_scaled.shape[1]):
            mask_t = mask[:, t, :].flatten()
            if unscale:
                y_true = dataset.data_processed_seq['unscaled_outputs'][not_nan, t, :].flatten()[mask_t]
                y_pred = outputs_unscaled[not_nan, t, :].flatten()[mask_t]
            else:
                y_true = dataset.data_processed_seq['outputs'][not_nan, t, :].flatten()[mask_t]
                y_pred = outputs_scaled[not_nan, t, :].flatten()[mask_t]
            
            if len(y_true) > 1:  # Need at least 2 points for correlation
                r, _ = pearsonr(y_true, y_pred)
                pearson_rs.append(r)
            else:
                pearson_rs.append(np.nan)
        
        return rmses_normalised_orig, np.array(pearson_rs)

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
                            metric=f"val_{self.tuning_criterion}_all",
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
    tuning_criterion = 'rmse'

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

    def configure_optimizers(self):
        if self.balancing == 'grad_reverse' and not self.hparams.exp.weights_ema:  # one optimizer
            optimizer = self._get_optimizer(list(self.named_parameters()))

            if self.hparams.model[self.model_type]['optimizer']['lr_scheduler']:
                return self._get_lr_schedulers(optimizer)

            return optimizer

        else:  # two optimizers - simultaneous gradient descent update
            treatment_head_params = \
                ['br_treatment_outcome_head.' + s for s in self.br_treatment_outcome_head.treatment_head_params]
            treatment_head_params = \
                [k for k in dict(self.named_parameters()) for param in treatment_head_params if k.startswith(param)]
            non_treatment_head_params = [k for k in dict(self.named_parameters()) if k not in treatment_head_params]

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
                    treatment_pred, outcome_pred, _ = self(batch)
            else:
                treatment_pred, outcome_pred, _ = self(batch)

            # Handle multiple outcomes
            if isinstance(outcome_pred, dict):
                # Multiple outcomes - calculate loss for each
                outcome_losses = {}
                task_losses = []  # For GradNorm
                
                # Split batch outputs by outcome
                start_idx = 0
                # Get dimension list
                dim_list = self.dim_outcome_list if self.dim_outcome_list is not None else [self.dim_outcome]
                
                for i, (outcome_name, (otype, dim)) in enumerate(zip(self.dataset_collection.outcome_columns, 
                                                     zip(self.dataset_collection.outcome_types, 
                                                         dim_list))):
                    end_idx = start_idx + dim
                    outcome_target = batch['outputs'][:, :, start_idx:end_idx]
                    
                    if otype == 'binary':
                        # Use BCE loss for binary outcomes
                        loss = F.binary_cross_entropy_with_logits(outcome_pred[outcome_name], outcome_target, reduce=False)
                    else:
                        # Use MSE loss for continuous outcomes
                        loss = F.mse_loss(outcome_pred[outcome_name], outcome_target, reduce=False)
                    
                    outcome_losses[outcome_name] = loss
                    
                    # Mask and average the loss
                    masked_loss = (batch['active_entries'] * loss).sum() / batch['active_entries'].sum()
                    task_losses.append(masked_loss)
                    
                    start_idx = end_idx
                
                # Apply GradNorm weighting
                if hasattr(self, 'task_weights') and self.training:
                    # Store task losses for GradNorm callback
                    self.task_losses_step = task_losses
                    
                    # Normalize task weights
                    normalized_weights = F.softmax(self.task_weights, dim=0)
                    
                    # Calculate weighted loss
                    total_outcome_loss = sum(w * loss for w, loss in zip(normalized_weights, task_losses))
                else:
                    # Use manual weights if not using GradNorm
                    total_outcome_loss = 0
                    for outcome_name, loss in outcome_losses.items():
                        if hasattr(self.hparams.dataset, 'task_weights') and outcome_name in self.hparams.dataset.task_weights:
                            weight = self.hparams.dataset.task_weights[outcome_name]
                        else:
                            weight = 1.0
                        total_outcome_loss = total_outcome_loss + weight * loss
                
                outcome_loss = total_outcome_loss
            else:
                # Single outcome - original behavior
                if hasattr(self.dataset_collection, 'outcome_type') and self.dataset_collection.outcome_type == 'binary':
                    outcome_loss = F.binary_cross_entropy_with_logits(outcome_pred, batch['outputs'], reduce=False)
                else:
                    outcome_loss = F.mse_loss(outcome_pred, batch['outputs'], reduce=False)
            if self.balancing == 'grad_reverse':
                bce_loss = self.bce_loss(treatment_pred, batch['current_treatments'].double(), kind='predict')
            elif self.balancing == 'domain_confusion':
                bce_loss = self.bce_loss(treatment_pred, batch['current_treatments'].double(), kind='confuse')
                bce_loss = self.br_treatment_outcome_head.alpha * bce_loss
            else:
                raise NotImplementedError()

            # Masking for shorter sequences
            # Attention! Averaging across all the active entries (= sequence masks) for full batch
            bce_loss = (batch['active_entries'].squeeze(-1) * bce_loss).sum() / batch['active_entries'].sum()
            outcome_loss = (batch['active_entries'] * outcome_loss).sum() / batch['active_entries'].sum()

            loss = bce_loss + outcome_loss

            self.log(f'{self.model_type}_train_loss', loss, on_epoch=True, on_step=False, prog_bar=False, sync_dist=True)
            self.log(f'{self.model_type}_train_bce_loss', bce_loss, on_epoch=True, on_step=False, sync_dist=True)
            
            # Log outcome losses
            if isinstance(outcome_pred, dict):
                # Log individual outcome losses
                for outcome_name, (otype, loss_val) in zip(self.dataset_collection.outcome_columns,
                                                          zip(self.dataset_collection.outcome_types, 
                                                              outcome_losses.values())):
                    masked_loss = (batch['active_entries'] * loss_val).sum() / batch['active_entries'].sum()
                    weight = self.hparams.dataset.task_weights.get(outcome_name, 1.0) if hasattr(self.hparams.dataset, 'task_weights') else 1.0
                    weighted_loss = weight * masked_loss
                    
                    if otype == 'binary':
                        self.log(f'{self.model_type}_train_{outcome_name}_bce_loss', masked_loss, 
                                on_epoch=True, on_step=False, prog_bar=False, sync_dist=True)
                        self.log(f'{self.model_type}_train_{outcome_name}_weighted_loss', weighted_loss, 
                                on_epoch=True, on_step=False, sync_dist=True)
                    else:
                        self.log(f'{self.model_type}_train_{outcome_name}_mse_loss', masked_loss, 
                                on_epoch=True, on_step=False, prog_bar=False, sync_dist=True)
                        self.log(f'{self.model_type}_train_{outcome_name}_weighted_loss', weighted_loss, 
                                on_epoch=True, on_step=False, sync_dist=True)
                self.log(f'{self.model_type}_train_total_outcome_loss', outcome_loss, 
                        on_epoch=True, on_step=False, sync_dist=True)
            else:
                # Single outcome - original behavior
                if hasattr(self.dataset_collection, 'outcome_type') and self.dataset_collection.outcome_type == 'binary':
                    self.log(f'{self.model_type}_train_outcome_bce_loss', outcome_loss, on_epoch=True, on_step=False, sync_dist=True)
                else:
                    self.log(f'{self.model_type}_train_mse_loss', outcome_loss, on_epoch=True, on_step=False, sync_dist=True)
            self.log(f'{self.model_type}_alpha', self.br_treatment_outcome_head.alpha, on_epoch=True, on_step=False,
                     sync_dist=True)

            return loss

        elif optimizer_idx == 1:  # domain classifier update
            if self.hparams.exp.weights_ema:
                with self.ema_non_treatment.average_parameters():
                    treatment_pred, _, _ = self(batch, detach_treatment=True)
            else:
                treatment_pred, _, _ = self(batch, detach_treatment=True)

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
                    treatment_pred, outcome_pred, _ = self(batch)
        else:
            treatment_pred, outcome_pred, _ = self(batch)

        if self.balancing == 'grad_reverse':
            bce_loss = self.bce_loss(treatment_pred, batch['current_treatments'].double(), kind='predict')
        elif self.balancing == 'domain_confusion':
            bce_loss = self.bce_loss(treatment_pred, batch['current_treatments'].double(), kind='confuse')

        # Handle multiple outcomes
        if isinstance(outcome_pred, dict):
            # Multiple outcomes - calculate loss for each
            outcome_losses = {}
            total_outcome_loss = 0
            
            # Split batch outputs by outcome
            start_idx = 0
            for outcome_name, (otype, dim) in zip(self.dataset_collection.outcome_columns, 
                                                 zip(self.dataset_collection.outcome_types, 
                                                     self.dim_outcome)):
                end_idx = start_idx + dim
                outcome_target = batch['outputs'][:, :, start_idx:end_idx]
                
                if otype == 'binary':
                    # Use BCE loss for binary outcomes
                    loss = F.binary_cross_entropy_with_logits(outcome_pred[outcome_name], outcome_target, reduce=False)
                else:
                    # Use MSE loss for continuous outcomes
                    loss = F.mse_loss(outcome_pred[outcome_name], outcome_target, reduce=False)
                
                outcome_losses[outcome_name] = loss
                total_outcome_loss = total_outcome_loss + loss
                start_idx = end_idx
            
            outcome_loss = total_outcome_loss
        else:
            # Single outcome - original behavior
            if hasattr(self.dataset_collection, 'outcome_type') and self.dataset_collection.outcome_type == 'binary':
                outcome_loss = F.binary_cross_entropy_with_logits(outcome_pred, batch['outputs'], reduce=False)
            else:
                outcome_loss = F.mse_loss(outcome_pred, batch['outputs'], reduce=False)

        # Masking for shorter sequences
        # Attention! Averaging across all the active entries (= sequence masks) for full batch
        bce_loss = (batch['active_entries'].squeeze(-1) * bce_loss).sum() / batch['active_entries'].sum()
        outcome_loss = (batch['active_entries'] * outcome_loss).sum() / batch['active_entries'].sum()
        loss = bce_loss + outcome_loss

        # Calculate Pearson correlation coefficient for outcomes
        if isinstance(outcome_pred, dict):
            # Multiple outcomes - calculate correlation for continuous ones
            pearson_r = {}
            mask = batch['active_entries'].cpu().numpy().astype(bool)
            
            start_idx = 0
            # Get dimension list
            dim_list = self.dim_outcome_list if self.dim_outcome_list is not None else [self.dim_outcome]
            
            for outcome_name, (otype, dim) in zip(self.dataset_collection.outcome_columns,
                                                 zip(self.dataset_collection.outcome_types,
                                                     dim_list)):
                end_idx = start_idx + dim
                if otype == 'binary':
                    pearson_r[outcome_name] = np.nan  # Not meaningful for binary
                else:
                    y_true = batch['outputs'][:, :, start_idx:end_idx].cpu().numpy()[mask].flatten()
                    y_pred = outcome_pred[outcome_name].cpu().numpy()[mask].flatten()
                    pearson_r[outcome_name], _ = pearsonr(y_true, y_pred)
                start_idx = end_idx
        else:
            # Single outcome - original behavior
            if hasattr(self.dataset_collection, 'outcome_type') and self.dataset_collection.outcome_type == 'binary':
                pearson_r = np.nan
            else:
                mask = batch['active_entries'].cpu().numpy().astype(bool)
                # For single outcome with multiple output dimensions, we need to handle the mask properly
                if batch['outputs'].shape[-1] > 1:
                    # Multiple output dimensions - need to expand mask
                    mask_expanded = np.broadcast_to(mask, batch['outputs'].shape)
                    y_true = batch['outputs'].cpu().numpy()[mask_expanded].flatten()
                    y_pred = outcome_pred.cpu().numpy()[mask_expanded].flatten()
                else:
                    y_true = batch['outputs'].cpu().numpy()[mask].flatten()
                    y_pred = outcome_pred.cpu().numpy()[mask].flatten()
                pearson_r, _ = pearsonr(y_true, y_pred)

        # Calculate Pearson correlation coefficient for treatments (average across dimensions)
        mask_treatments = batch['active_entries'].squeeze(-1).cpu().numpy().astype(bool)
        treatment_true = batch['current_treatments'].double().cpu().numpy()
        treatment_pred_np = treatment_pred.cpu().numpy()
        
        # Simple save before calculating correlations
        if self.trainer.testing:
            import pandas as pd
            from pathlib import Path
            
            subset_name = self.test_dataloader().dataset.subset_name
            output_dir = Path(__file__).parent.parent.parent / "outputs" / "treatment_predictions"
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Save raw arrays for debugging
            np.save(output_dir / f"{subset_name}_treatment_true.npy", treatment_true)
            np.save(output_dir / f"{subset_name}_treatment_pred.npy", treatment_pred_np)
            np.save(output_dir / f"{subset_name}_treatment_mask.npy", mask_treatments)
            logger.info(f"Saved {subset_name} treatment arrays to {output_dir}")
        
        treatment_correlations = []
        for dim in range(treatment_true.shape[-1]):
            t_true = treatment_true[:, :, dim][mask_treatments].flatten()
            t_pred = treatment_pred_np[:, :, dim][mask_treatments].flatten()
            
            # Remove any NaN values
            valid_mask = ~(np.isnan(t_true) | np.isnan(t_pred))
            t_true = t_true[valid_mask]
            t_pred = t_pred[valid_mask]
            
            if len(t_true) > 1:
                r, _ = pearsonr(t_true, t_pred)
                treatment_correlations.append(r)
        
        avg_treatment_pearson_r = np.nanmean(treatment_correlations) if treatment_correlations else np.nan

        subset_name = self.test_dataloader().dataset.subset_name
        self.log(f'{self.model_type}_{subset_name}_loss', loss, on_epoch=True, on_step=False, sync_dist=True)
        self.log(f'{self.model_type}_{subset_name}_bce_loss', bce_loss, on_epoch=True, on_step=False, sync_dist=True)
        
        # Log outcome losses and metrics
        if isinstance(outcome_pred, dict):
            # Multiple outcomes - log each separately
            for outcome_name, (otype, loss_val) in zip(self.dataset_collection.outcome_columns,
                                                      zip(self.dataset_collection.outcome_types,
                                                          outcome_losses.values())):
                masked_loss = (batch['active_entries'] * loss_val).sum() / batch['active_entries'].sum()
                if otype == 'binary':
                    self.log(f'{self.model_type}_{subset_name}_{outcome_name}_bce_loss', masked_loss,
                            on_epoch=True, on_step=False, sync_dist=True)
                else:
                    self.log(f'{self.model_type}_{subset_name}_{outcome_name}_mse_loss', masked_loss,
                            on_epoch=True, on_step=False, sync_dist=True)
                    self.log(f'{self.model_type}_{subset_name}_{outcome_name}_pearson_r', pearson_r[outcome_name],
                            on_epoch=True, on_step=False, sync_dist=True)
            self.log(f'{self.model_type}_{subset_name}_total_outcome_loss', outcome_loss,
                    on_epoch=True, on_step=False, sync_dist=True)
        else:
            # Single outcome - original behavior
            if hasattr(self.dataset_collection, 'outcome_type') and self.dataset_collection.outcome_type == 'binary':
                self.log(f'{self.model_type}_{subset_name}_outcome_bce_loss', outcome_loss, on_epoch=True, on_step=False, sync_dist=True)
            else:
                self.log(f'{self.model_type}_{subset_name}_mse_loss', outcome_loss, on_epoch=True, on_step=False, sync_dist=True)
            self.log(f'{self.model_type}_{subset_name}_pearson_r', pearson_r, on_epoch=True, on_step=False, sync_dist=True)
        self.log(f'{self.model_type}_{subset_name}_treatment_pearson_r', avg_treatment_pearson_r, on_epoch=True, on_step=False, sync_dist=True)

    def validation_step(self, batch, batch_idx):
        """Validation step to compute metrics during training"""
        from sklearn.metrics import roc_auc_score, average_precision_score
        
        # Get predictions
        treatment_pred, outcome_pred, _ = self(batch)
        
        # Calculate losses (same as training but without gradients)
        if isinstance(outcome_pred, dict):
            # Multiple outcomes
            outcome_losses = {}
            val_metrics = {}
            total_loss = 0
            
            start_idx = 0
            # Get dimension list
            dim_list = self.dim_outcome_list if self.dim_outcome_list is not None else [self.dim_outcome]
            
            for outcome_name, (otype, dim) in zip(self.dataset_collection.outcome_columns,
                                                 zip(self.dataset_collection.outcome_types,
                                                     dim_list)):
                end_idx = start_idx + dim
                outcome_target = batch['outputs'][:, :, start_idx:end_idx]
                
                if otype == 'binary':
                    loss = F.binary_cross_entropy_with_logits(outcome_pred[outcome_name], outcome_target, reduce=False)
                    
                    # Calculate AUC metrics for binary outcomes
                    mask = batch['active_entries'].cpu().numpy().astype(bool).flatten()
                    y_true = outcome_target.cpu().numpy().flatten()[mask]
                    y_pred_logits = outcome_pred[outcome_name].detach().cpu().numpy().flatten()[mask]
                    y_pred_probs = 1 / (1 + np.exp(-y_pred_logits))
                    
                    if len(np.unique(y_true)) > 1:  # Need both classes for AUC
                        auc_roc = roc_auc_score(y_true, y_pred_probs)
                        auc_pr = average_precision_score(y_true, y_pred_probs)
                        val_metrics[f'val_{outcome_name}_auc_roc'] = auc_roc
                        val_metrics[f'val_{outcome_name}_auc_pr'] = auc_pr
                else:
                    loss = F.mse_loss(outcome_pred[outcome_name], outcome_target, reduce=False)
                    
                    # Calculate R for continuous outcomes
                    mask = batch['active_entries'].cpu().numpy().astype(bool)
                    y_true = outcome_target.cpu().numpy()[mask].flatten()
                    y_pred = outcome_pred[outcome_name].detach().cpu().numpy()[mask].flatten()
                    if len(y_true) > 1:  # Need at least 2 points for correlation
                        pearson_r, _ = pearsonr(y_true, y_pred)
                        val_metrics[f'val_{outcome_name}_r'] = pearson_r
                
                outcome_losses[outcome_name] = loss
                masked_loss = (batch['active_entries'] * loss).sum() / batch['active_entries'].sum()
                val_metrics[f'val_{outcome_name}_loss'] = masked_loss.item() if isinstance(masked_loss, torch.Tensor) else masked_loss
                total_loss += masked_loss
                
                start_idx = end_idx
            
            # Log all metrics
            for metric_name, metric_value in val_metrics.items():
                # Convert to CPU if it's a tensor
                if isinstance(metric_value, torch.Tensor):
                    metric_value = metric_value.cpu().item()
                if not np.isnan(metric_value):  # Only log valid metrics
                    self.log(metric_name, metric_value, on_epoch=True, prog_bar=False, sync_dist=True)
            
            # Log total validation loss
            total_loss_value = total_loss.item() if isinstance(total_loss, torch.Tensor) else total_loss
            self.log('val_loss', total_loss_value, on_epoch=True, prog_bar=True, sync_dist=True)
        else:
            # Single outcome - original behavior
            if hasattr(self.dataset_collection, 'outcome_type') and self.dataset_collection.outcome_type == 'binary':
                loss = F.binary_cross_entropy_with_logits(outcome_pred, batch['outputs'], reduce=False)
                
                # Calculate AUC metrics
                mask = batch['active_entries'].cpu().numpy().astype(bool).flatten()
                y_true = batch['outputs'].cpu().numpy().flatten()[mask]
                y_pred_logits = outcome_pred.detach().cpu().numpy().flatten()[mask]
                y_pred_probs = 1 / (1 + np.exp(-y_pred_logits))
                
                if len(np.unique(y_true)) > 1:
                    auc_roc = roc_auc_score(y_true, y_pred_probs)
                    auc_pr = average_precision_score(y_true, y_pred_probs)
                    self.log('val_auc_roc', auc_roc, on_epoch=True, prog_bar=False, sync_dist=True)
                    self.log('val_auc_pr', auc_pr, on_epoch=True, prog_bar=False, sync_dist=True)
            else:
                loss = F.mse_loss(outcome_pred, batch['outputs'], reduce=False)
                
                # Calculate R
                mask = batch['active_entries'].cpu().numpy().astype(bool)
                if batch['outputs'].shape[-1] > 1:
                    mask_expanded = np.broadcast_to(mask, batch['outputs'].shape)
                    y_true = batch['outputs'].cpu().numpy()[mask_expanded].flatten()
                    y_pred = outcome_pred.detach().cpu().numpy()[mask_expanded].flatten()
                else:
                    y_true = batch['outputs'].cpu().numpy()[mask].flatten()
                    y_pred = outcome_pred.detach().cpu().numpy()[mask].flatten()
                
                if len(y_true) > 1:
                    pearson_r, _ = pearsonr(y_true, y_pred)
                    self.log('val_r', pearson_r, on_epoch=True, prog_bar=False, sync_dist=True)
            
            # Masked loss
            outcome_loss = (batch['active_entries'] * loss).sum() / batch['active_entries'].sum()
            outcome_loss_value = outcome_loss.item() if isinstance(outcome_loss, torch.Tensor) else outcome_loss
            self.log('val_loss', outcome_loss_value, on_epoch=True, prog_bar=True, sync_dist=True)

    def predict_step(self, batch, batch_idx, dataset_idx=None):
        """
        Generates normalised output predictions
        """
        if self.hparams.exp.weights_ema:
            with self.ema_non_treatment.average_parameters():
                _, outcome_pred, br = self(batch)
        else:
            _, outcome_pred, br = self(batch)
        
        # Handle dict outcomes
        if isinstance(outcome_pred, dict):
            # Concatenate predictions in the same order as outcome_columns
            outcome_tensors = []
            for outcome_name in self.dataset_collection.outcome_columns:
                outcome_tensors.append(outcome_pred[outcome_name])
            outcome_pred = torch.cat(outcome_tensors, dim=-1)
        
        return outcome_pred.cpu(), br.cpu()

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
        outcome_pred, _ = [torch.cat(arrs) for arrs in zip(*self.trainer.predict(self, data_loader))]
        return outcome_pred.numpy()


class GradNormCallback(Callback):
    """Callback to apply GradNorm for balancing multitask losses"""
    
    def __init__(self, alpha=1.5, update_every_n_steps=1):
        self.alpha = alpha
        self.update_every_n_steps = update_every_n_steps
        self.initial_losses = None
        
    def on_after_backward(self, trainer, pl_module):
        # Only apply if we have multiple tasks with learnable weights
        if not hasattr(pl_module, 'task_weights') or not hasattr(pl_module, 'task_losses_step'):
            return
            
        # Skip first few batches to let initial losses stabilize
        if trainer.global_step < 10:
            return
            
        # Only update every N steps
        if trainer.global_step % self.update_every_n_steps != 0:
            return
        
        # Get task losses from the stored values
        task_losses = pl_module.task_losses_step
        if task_losses is None or len(task_losses) != pl_module.num_tasks:
            return
            
        # Initialize initial losses
        if self.initial_losses is None:
            with torch.no_grad():
                self.initial_losses = torch.stack(task_losses).detach().clone()
                return
        
        # Get shared parameters (exclude final task-specific heads)
        shared_params = []
        for name, param in pl_module.named_parameters():
            if 'outcome_head' not in name and 'task_weights' not in name and param.requires_grad and param.grad is not None:
                shared_params.append(param)
        
        if not shared_params:
            return
            
        # Compute gradient norms for each task using existing gradients
        with torch.no_grad():
            # Get current task weights
            weights = F.softmax(pl_module.task_weights, dim=0)
            
            # Approximate gradient norms from loss magnitudes and learning progress
            current_losses = torch.stack(task_losses).detach()
            loss_ratios = current_losses / self.initial_losses.clamp(min=1e-8)
            
            # Inverse training rates (tasks that improved less get higher weight)
            inverse_rates = loss_ratios / loss_ratios.mean()
            target_weights = inverse_rates ** self.alpha
            target_weights = target_weights / target_weights.sum() * pl_module.num_tasks
            
            # Update task weights using gradient descent
            weight_grad = 2.0 * (weights - target_weights.detach())
            pl_module.task_weights.data -= 0.025 * weight_grad  # Small learning rate
            
            # Renormalize to ensure they sum to num_tasks
            pl_module.task_weights.data = pl_module.num_tasks * F.softmax(pl_module.task_weights, dim=0)
    
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        # This is now empty - we do everything in on_after_backward
        pass
    
    def on_train_epoch_end(self, trainer, pl_module):
        # Log task weights
        if hasattr(pl_module, 'task_weights'):
            normalized_weights = F.softmax(pl_module.task_weights, dim=0)
            for i, (name, weight) in enumerate(zip(pl_module.dataset_collection.outcome_columns, normalized_weights)):
                pl_module.log(f'gradnorm_weight_{name}', weight.item(), on_epoch=True)


class LossBreakdownCallback(Callback):
    """Callback to print loss breakdown at the end of each epoch"""
    
    def on_train_epoch_end(self, trainer, pl_module):
        # Get the logged metrics
        metrics = trainer.callback_metrics
        model_type = pl_module.model_type
        
        # Debug: print all available metrics
        # print(f"Available metrics: {list(metrics.keys())}")
        
        # Prepare loss breakdown message
        loss_parts = []
        
        # Total loss
        if f'{model_type}_train_loss' in metrics:
            total_loss = metrics[f'{model_type}_train_loss'].item()
            loss_parts.append(f"Total Loss: {total_loss:.4f}")
        
        # Treatment prediction loss
        if f'{model_type}_train_bce_loss' in metrics:
            bce_loss = metrics[f'{model_type}_train_bce_loss'].item()
            loss_parts.append(f"Treatment BCE: {bce_loss:.4f}")
        
        # Individual outcome losses
        if hasattr(pl_module.dataset_collection, 'outcome_columns'):
            loss_parts.append("Outcome Losses:")
            for i, outcome_name in enumerate(pl_module.dataset_collection.outcome_columns):
                # Check for different loss types
                if f'{model_type}_train_{outcome_name}_mse_loss' in metrics:
                    loss_val = metrics[f'{model_type}_train_{outcome_name}_mse_loss'].item()
                    weight_str = ""
                    
                    # Show GradNorm weight if available
                    if hasattr(pl_module, 'task_weights'):
                        normalized_weights = F.softmax(pl_module.task_weights, dim=0)
                        weight_str = f" (gradnorm_weight={normalized_weights[i].item():.3f})"
                    elif hasattr(pl_module.hparams.dataset, 'task_weights') and outcome_name in pl_module.hparams.dataset.task_weights:
                        weight = pl_module.hparams.dataset.task_weights[outcome_name]
                        weight_str = f" (manual_weight={weight})"
                    
                    loss_parts.append(f"  - {outcome_name} (MSE): {loss_val:.4f}{weight_str}")
                elif f'{model_type}_train_{outcome_name}_bce_loss' in metrics:
                    loss_val = metrics[f'{model_type}_train_{outcome_name}_bce_loss'].item()
                    weight_str = ""
                    
                    # Show GradNorm weight if available
                    if hasattr(pl_module, 'task_weights'):
                        normalized_weights = F.softmax(pl_module.task_weights, dim=0)
                        weight_str = f" (gradnorm_weight={normalized_weights[i].item():.3f})"
                    elif hasattr(pl_module.hparams.dataset, 'task_weights') and outcome_name in pl_module.hparams.dataset.task_weights:
                        weight = pl_module.hparams.dataset.task_weights[outcome_name]
                        weight_str = f" (manual_weight={weight})"
                    
                    loss_parts.append(f"  - {outcome_name} (BCE): {loss_val:.4f}{weight_str}")
        
        # Add validation metrics if available
        val_parts = []
        
        # Total validation loss
        if 'val_loss' in metrics:
            val_loss = metrics['val_loss'].item()
            val_parts.append(f"Total Val Loss: {val_loss:.4f}")
        
        # Individual outcome validation metrics
        if hasattr(pl_module.dataset_collection, 'outcome_columns'):
            for outcome_name in pl_module.dataset_collection.outcome_columns:
                outcome_metrics = []
                
                # Check for validation loss
                if f'val_{outcome_name}_loss' in metrics:
                    loss_val = metrics[f'val_{outcome_name}_loss'].item()
                    outcome_metrics.append(f"loss={loss_val:.4f}")
                
                # Check for R (continuous outcomes)
                if f'val_{outcome_name}_r' in metrics:
                    r_val = metrics[f'val_{outcome_name}_r'].item()
                    outcome_metrics.append(f"R={r_val:.4f}")
                
                # Check for AUC metrics (binary outcomes)
                if f'val_{outcome_name}_auc_roc' in metrics:
                    auc_roc = metrics[f'val_{outcome_name}_auc_roc'].item()
                    outcome_metrics.append(f"AUC-ROC={auc_roc:.4f}")
                
                if f'val_{outcome_name}_auc_pr' in metrics:
                    auc_pr = metrics[f'val_{outcome_name}_auc_pr'].item()
                    outcome_metrics.append(f"AUC-PR={auc_pr:.4f}")
                
                if outcome_metrics:
                    val_parts.append(f"  - {outcome_name}: {', '.join(outcome_metrics)}")
        
        # For single outcome models
        if 'val_r' in metrics:
            r_val = metrics['val_r'].item()
            val_parts.append(f"Val R: {r_val:.4f}")
        
        if 'val_auc_roc' in metrics:
            auc_roc = metrics['val_auc_roc'].item()
            val_parts.append(f"Val AUC-ROC: {auc_roc:.4f}")
        
        if 'val_auc_pr' in metrics:
            auc_pr = metrics['val_auc_pr'].item()
            val_parts.append(f"Val AUC-PR: {auc_pr:.4f}")
        
        # Print the breakdown
        if loss_parts or val_parts:
            print(f"\nEpoch {trainer.current_epoch} Metrics:")
            
            if loss_parts:
                print("  Training:")
                for part in loss_parts:
                    print(f"    {part}")
            
            if val_parts:
                print("  Validation:")
                for part in val_parts:
                    print(f"    {part}")
            
            print("")  # Empty line for readability
