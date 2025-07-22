from pytorch_lightning import LightningModule
from omegaconf import DictConfig, ListConfig
import torch
from torch import nn
from omegaconf.errors import MissingMandatoryValue
import torch.nn.functional as F
from hydra.utils import instantiate
from torch.utils.data import DataLoader, Dataset, Subset
import logging
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from typing import Union
from functools import partial
import seaborn as sns
from sklearn.manifold import TSNE

from src.models.edct import EDCT
from src.models.utils_transformer import TransformerMultiInputBlock, LayerNorm
from src.data import RealDatasetCollection, SyntheticDatasetCollection
from src.models.utils import BRTreatmentOutcomeHead

logger = logging.getLogger(__name__)


class CT(EDCT):
    """
    Pytorch-Lightning implementation of Causal Transformer (CT)
    """

    model_type = 'multi'  # multi-input model
    possible_model_types = {'multi'}

    def __init__(self, args: DictConfig,
                 dataset_collection: Union[RealDatasetCollection, SyntheticDatasetCollection] = None,
                 autoregressive: bool = None,
                 has_vitals: bool = None,
                 projection_horizon: int = None,
                 bce_weights: np.array = None, **kwargs):
        """
        Args:
            args: DictConfig of model hyperparameters
            dataset_collection: Dataset collection
            autoregressive: Flag of including previous outcomes to modelling
            has_vitals: Flag of vitals in dataset
            projection_horizon: Range of tau-step-ahead prediction (tau = projection_horizon + 1)
            bce_weights: Re-weight BCE if used
            **kwargs: Other arguments
        """
        super().__init__(args, dataset_collection, autoregressive, has_vitals, bce_weights)

        if self.dataset_collection is not None:
            self.projection_horizon = self.dataset_collection.projection_horizon
        else:
            self.projection_horizon = projection_horizon

        # Used in hparam tuning
        # Handle list dim_outcome
        dim_outcome_max = sum(self.dim_outcome) if isinstance(self.dim_outcome, (list, ListConfig)) else self.dim_outcome
        self.input_size = max(self.dim_treatments, self.dim_static_features, self.dim_vitals, dim_outcome_max)
        logger.info(f'Max input size of {self.model_type}: {self.input_size}')
        assert self.autoregressive  # prev_outcomes are obligatory

        self.basic_block_cls = TransformerMultiInputBlock
        self._init_specific(args.model.multi, dataset_collection)
        self.save_hyperparameters(args)

    def _init_specific(self, sub_args: DictConfig, dataset_collection=None):
        """
        Initialization of specific sub-network (only multi)
        Args:
            sub_args: sub-network hyperparameters
            dataset_collection: Dataset collection for getting num_buckets
        """
        try:
            super(CT, self)._init_specific(sub_args, dataset_collection)

            if self.seq_hidden_units is None or self.br_size is None or self.fc_hidden_units is None \
                    or self.dropout_rate is None:
                raise MissingMandatoryValue()

            self.treatments_input_transformation = nn.Linear(self.dim_treatments, self.seq_hidden_units)
            self.vitals_input_transformation = \
                nn.Linear(self.dim_vitals, self.seq_hidden_units) if self.has_vitals else None
            
            # Handle multiple outcomes properly
            total_dim_outcome = sum(self.dim_outcome) if isinstance(self.dim_outcome, (list, ListConfig)) else self.dim_outcome
            self.outputs_input_transformation = nn.Linear(total_dim_outcome, self.seq_hidden_units)
            self.static_input_transformation = nn.Linear(self.dim_static_features, self.seq_hidden_units)

            self.n_inputs = 3 if self.has_vitals else 2  # prev_outcomes and prev_treatments

            self.transformer_blocks = nn.ModuleList(
                [self.basic_block_cls(self.seq_hidden_units, self.num_heads, self.head_size, self.seq_hidden_units * 4,
                                      self.dropout_rate,
                                      self.dropout_rate if sub_args.attn_dropout else 0.0,
                                      self_positional_encoding_k=self.self_positional_encoding_k,
                                      self_positional_encoding_v=self.self_positional_encoding_v,
                                      n_inputs=self.n_inputs,
                                      disable_cross_attention=sub_args.disable_cross_attention,
                                      isolate_subnetwork=sub_args.isolate_subnetwork) for _ in range(self.num_layer)])

            # Don't create single head here - let EDCT parent class handle multiple heads

            # self.last_layer_norm = LayerNorm(self.seq_hidden_units)
        except MissingMandatoryValue:
            logger.warning(f"{self.model_type} not fully initialised - some mandatory args are missing! "
                           f"(It's ok, if one will perform hyperparameters search afterward).")

    def prepare_data(self) -> None:
        if self.dataset_collection is not None and not self.dataset_collection.processed_data_multi:
            self.dataset_collection.process_data_multi()
        if self.bce_weights is None and self.hparams.exp.bce_weight:
            self._calculate_bce_weights()

    def forward(self, batch, detach_treatment=False):
        fixed_split = batch['future_past_split'] if 'future_past_split' in batch else None

        if self.training and self.hparams.model.multi.augment_with_masked_vitals and self.has_vitals:
            # Augmenting original batch with vitals-masked copy
            assert fixed_split is None  # Only for training data
            fixed_split = torch.empty((2 * len(batch['active_entries']),)).type_as(batch['active_entries'])
            for i, seq_len in enumerate(batch['active_entries'].sum(1).int()):
                fixed_split[i] = seq_len  # Original batch
                fixed_split[len(batch['active_entries']) + i] = torch.randint(0, int(seq_len) + 1, (1,)).item()  # Augmented batch

            for (k, v) in batch.items():
                batch[k] = torch.cat((v, v), dim=0)

        prev_treatments = batch['prev_treatments']
        vitals = batch['vitals'] if self.has_vitals else None
        prev_outputs = batch['prev_outputs']
        static_features = batch['static_features']
        curr_treatments = batch['current_treatments']
        active_entries = batch['active_entries']

        br = self.build_br(prev_treatments, vitals, prev_outputs, static_features, active_entries, fixed_split)
        
        treatment_pred = self.br_treatment_outcome_head.build_treatment(br, detach_treatment)
        outcome_pred = self.br_treatment_outcome_head.build_outcome(br, curr_treatments)
        bucket_pred = self.br_treatment_outcome_head.build_bucket(br, curr_treatments)
        
        # Split predictions if multiple outcomes
        if hasattr(self, 'dim_outcome_list') and self.dim_outcome_list is not None:
            outcome_pred_dict = {}
            bucket_pred_dict = {}
            start_idx = 0
            for outcome_name, dim in zip(self.dataset_collection.outcome_columns, self.dim_outcome_list):
                end_idx = start_idx + dim
                outcome_pred_dict[outcome_name] = outcome_pred[..., start_idx:end_idx]
                # Bucket predictions should not be split - they represent time buckets, not outcomes
                # Each outcome gets the full bucket predictions
                bucket_pred_dict[outcome_name] = bucket_pred
                start_idx = end_idx
            outcome_pred = outcome_pred_dict
            bucket_pred = bucket_pred_dict

        return treatment_pred, outcome_pred, br, bucket_pred

    def build_br(self, prev_treatments, vitals, prev_outputs, static_features, active_entries, fixed_split):

        active_entries_treat_outcomes = torch.clone(active_entries)
        active_entries_vitals = torch.clone(active_entries)

        if fixed_split is not None and self.has_vitals:  # Test sequence data / Train augmented data
            for i in range(len(active_entries)):

                # Masking vitals in range [fixed_split: ]
                active_entries_vitals[i, int(fixed_split[i]):, :] = 0.0
                vitals[i, int(fixed_split[i]):] = 0.0

        x_t = self.treatments_input_transformation(prev_treatments)
        x_o = self.outputs_input_transformation(prev_outputs)
        x_v = self.vitals_input_transformation(vitals) if self.has_vitals else None
        x_s = self.static_input_transformation(static_features.unsqueeze(1))  # .expand(-1, x_t.size(1), -1)

        # if active_encoder_br is None and encoder_r is None:  # Only self-attention
        for block in self.transformer_blocks:

            if self.self_positional_encoding is not None:
                x_t = x_t + self.self_positional_encoding(x_t)
                x_o = x_o + self.self_positional_encoding(x_o)
                x_v = x_v + self.self_positional_encoding(x_v) if self.has_vitals else None

            if self.has_vitals:
                x_t, x_o, x_v = block((x_t, x_o, x_v), x_s, active_entries_treat_outcomes, active_entries_vitals)
            else:
                x_t, x_o = block((x_t, x_o), x_s, active_entries_treat_outcomes)

        if not self.has_vitals:
            x = (x_o + x_t) / 2
        else:
            if fixed_split is not None:  # Test seq data
                x = torch.empty_like(x_o)
                for i in range(len(active_entries)):
                    # Masking vitals in range [fixed_split: ]
                    x[i, :int(fixed_split[i])] = \
                        (x_o[i, :int(fixed_split[i])] + x_t[i, :int(fixed_split[i])] + x_v[i, :int(fixed_split[i])]) / 3
                    x[i, int(fixed_split[i]):] = (x_o[i, int(fixed_split[i]):] + x_t[i, int(fixed_split[i]):]) / 2
            else:  # Train data always has vitals
                x = (x_o + x_t + x_v) / 3

        output = self.output_dropout(x)
        br = self.br_treatment_outcome_head.build_br(output)
        return br

    def get_autoregressive_predictions(self, dataset: Dataset) -> np.array:
        logger.info(f'Autoregressive Prediction for {dataset.subset_name}.')

        predicted_outputs = np.zeros((len(dataset), self.hparams.dataset.projection_horizon, self.dim_outcome))

        for t in range(self.hparams.dataset.projection_horizon + 1):
            logger.info(f't = {t + 1}')
            outputs_scaled, _ = self.get_predictions(dataset)

            for i in range(len(dataset)):
                split = int(dataset.data['future_past_split'][i])
                if t < self.hparams.dataset.projection_horizon:
                    # Handle multiple outcomes with different types
                    if hasattr(self.dataset_collection, 'outcome_types'):
                        # Multiple outcomes case
                        prev_output_values = []
                        start_idx = 0
                        for j, (otype, dim) in enumerate(zip(self.dataset_collection.outcome_types, self.hparams.model.dim_outcomes)):
                            end_idx = start_idx + dim
                            if otype == 'binary':
                                # Apply sigmoid to get probabilities for binary outcomes
                                probs = 1 / (1 + np.exp(-outputs_scaled[i, split - 1 + t, start_idx:end_idx]))
                                prev_output_values.append(probs)
                            else:
                                # Keep continuous values as is
                                prev_output_values.append(outputs_scaled[i, split - 1 + t, start_idx:end_idx])
                            start_idx = end_idx
                        dataset.data['prev_outputs'][i, split + t, :] = np.concatenate(prev_output_values)
                    elif hasattr(self.dataset_collection, 'outcome_type') and self.dataset_collection.outcome_type == 'binary':
                        # Single binary outcome - original code
                        probs = 1 / (1 + np.exp(-outputs_scaled[i, split - 1 + t, :]))
                        dataset.data['prev_outputs'][i, split + t, :] = probs
                    else:
                        # Single continuous outcome - original code
                        dataset.data['prev_outputs'][i, split + t, :] = outputs_scaled[i, split - 1 + t, :]
                if t > 0:
                    predicted_outputs[i, t - 1, :] = outputs_scaled[i, split - 1 + t, :]

        return predicted_outputs

    def visualize(self, dataset: Dataset, index=0, artifacts_path=None):
        """
        Vizualizes attention scores
        :param dataset: dataset
        :param index: index of an instance
        :param artifacts_path: Path for saving
        """
        fig_keys = ['self_attention_o', 'self_attention_t', 'cross_attention_ot', 'cross_attention_to']
        if self.has_vitals:
            fig_keys += ['cross_attention_vo', 'cross_attention_ov', 'cross_attention_vt', 'cross_attention_tv',
                         'self_attention_v']
        self._visualize(fig_keys, dataset, index, artifacts_path)
