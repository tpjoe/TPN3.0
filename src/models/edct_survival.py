"""
EDCT model for survival analysis, inheriting from survival-specific base classes
"""
from pytorch_lightning import LightningModule
from omegaconf import DictConfig, ListConfig
import torch
from torch import nn
import torch.nn.functional as F
from omegaconf.errors import MissingMandatoryValue
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

from src.models.time_varying_model_survival import BRCausalModel
from src.models.utils_transformer import TransformerMultiInputBlock, LayerNorm, RelativePositionalEncoding, AbsolutePositionalEncoding
from src.data import RealDatasetCollection, SyntheticDatasetCollection
from src.models.utils import BRTreatmentOutcomeHead

logger = logging.getLogger(__name__)


class EDCTSurvival(BRCausalModel):
    """
    EDCT model adapted for survival analysis
    """

    model_type = None  # Will be defined in subclasses
    possible_model_types = {'encoder', 'decoder', 'multi'}

    def __init__(self, args: DictConfig,
                 dataset_collection: Union[RealDatasetCollection, SyntheticDatasetCollection] = None,
                 autoregressive: bool = None,
                 has_vitals: bool = None,
                 projection_horizon: int = None,
                 bce_weights: np.array = None, **kwargs):
        super().__init__(args, dataset_collection, autoregressive, has_vitals, bce_weights)

        if self.dataset_collection is not None:
            self.projection_horizon = self.dataset_collection.projection_horizon
        else:
            self.projection_horizon = projection_horizon

        logger.info(f'EDCT Survival - Projection horizon: {self.projection_horizon}')

        # Used in hparam tuning
        # Handle list dim_outcome
        dim_outcome_max = sum(self.dim_outcome) if isinstance(self.dim_outcome, (list, ListConfig)) else self.dim_outcome
        self.input_size = max(self.dim_treatments, self.dim_static_features, self.dim_vitals, dim_outcome_max)
        logger.info(f'Max input size of {self.model_type}: {self.input_size}')
        assert self.autoregressive  # prev_outcomes are obligatory
        
    def _init_specific(self, sub_args: DictConfig):
        """
        Initialization of specific sub-network parameters
        Args:
            sub_args: sub-network hyperparameters  
        """
        try:
            self.seq_hidden_units = sub_args.seq_hidden_units
            self.num_heads = sub_args.num_heads
            self.head_size = sub_args.seq_hidden_units // sub_args.num_heads
            self.num_layer = sub_args.num_layer
            self.dropout_rate = sub_args.dropout_rate
            self.fc_hidden_units = sub_args.fc_hidden_units
            self.br_size = sub_args.br_size
            
            if self.seq_hidden_units is None or self.br_size is None or self.fc_hidden_units is None or self.dropout_rate is None:
                raise MissingMandatoryValue()
                
            self.input_transformation = nn.Linear(self.input_size, self.seq_hidden_units) if self.input_size else None
            
            # Init of positional encodings
            self.self_positional_encoding = self.self_positional_encoding_k = self.self_positional_encoding_v = None
            if sub_args.self_positional_encoding.absolute:
                self.self_positional_encoding = AbsolutePositionalEncoding(
                    self.max_seq_length, self.seq_hidden_units,
                    sub_args.self_positional_encoding.trainable
                )
            else:
                # Relative positional encoding is shared across heads
                self.self_positional_encoding_k = RelativePositionalEncoding(
                    sub_args.self_positional_encoding.max_relative_position, self.head_size,
                    sub_args.self_positional_encoding.trainable
                )
                self.self_positional_encoding_v = RelativePositionalEncoding(
                    sub_args.self_positional_encoding.max_relative_position, self.head_size,
                    sub_args.self_positional_encoding.trainable
                )
                
            self.cross_positional_encoding = self.cross_positional_encoding_k = self.cross_positional_encoding_v = None
            if 'cross_positional_encoding' in sub_args and sub_args.cross_positional_encoding.absolute:
                self.cross_positional_encoding = AbsolutePositionalEncoding(
                    self.max_seq_length, self.seq_hidden_units,
                    sub_args.cross_positional_encoding.trainable
                )
            elif 'cross_positional_encoding' in sub_args:
                # Relative positional encoding is shared across heads
                self.cross_positional_encoding_k = RelativePositionalEncoding(
                    sub_args.cross_positional_encoding.max_relative_position, self.head_size,
                    sub_args.cross_positional_encoding.trainable, cross_attn=True
                )
                self.cross_positional_encoding_v = RelativePositionalEncoding(
                    sub_args.cross_positional_encoding.max_relative_position, self.head_size,
                    sub_args.cross_positional_encoding.trainable, cross_attn=True
                )
                
            self.basic_block_cls = TransformerMultiInputBlock if self.model_type == 'multi' else nn.Module
            
            self.transformer_blocks = [
                self.basic_block_cls(
                    self.seq_hidden_units, self.num_heads, self.head_size,
                    self.seq_hidden_units * 4, self.dropout_rate, self.dropout_rate,
                    self_positional_encoding_k=self.self_positional_encoding_k,
                    self_positional_encoding_v=self.self_positional_encoding_v,
                    cross_positional_encoding_k=self.cross_positional_encoding_k,
                    cross_positional_encoding_v=self.cross_positional_encoding_v
                ) for _ in range(self.num_layer)
            ]
            self.transformer_blocks = nn.ModuleList(self.transformer_blocks)
            self.output_dropout = nn.Dropout(self.dropout_rate)
            
            # Use single head with combined dimensions for multitask
            if isinstance(self.dim_outcome, (list, ListConfig)):
                total_dim_outcome = sum(self.dim_outcome)
            else:
                total_dim_outcome = self.dim_outcome
                
            self.br_treatment_outcome_head = BRTreatmentOutcomeHead(
                self.seq_hidden_units, self.br_size,
                self.fc_hidden_units, self.dim_treatments, total_dim_outcome,
                self.alpha, self.update_alpha, self.balancing
            )
        except MissingMandatoryValue:
            logger.warning(f"{self.model_type} not fully initialised - some mandatory args are missing! "
                          f"(It's ok, if one will perform hyperparameters search afterward).")

    @staticmethod
    def set_hparams(model_args: DictConfig, new_args: dict, input_size: int, model_type: str):
        """
        Used for hyperparameter tuning
        """
        sub_args = model_args[model_type]
        sub_args.seq_hidden_units = int(input_size * new_args['hidden_units_multiplier'])
        sub_args.fc_hidden_units = int(sub_args.seq_hidden_units * new_args['fc_hidden_units_multiplier'])
        sub_args.dropout_rate = new_args['dropout_rate']

    def get_optimizer_parameters(self, n_parameters: int):
        if self.hparams.model[self.model_type]['optimizer']['separate_lr']:
            if self.model_type == 'encoder':
                backbone_params = self.backbone.parameters()
                br_output_params = self.br_output_proj.parameters()
                lr = self.hparams.model[self.model_type]['optimizer']['lr_transformer']
                return [{'params': list(backbone_params), 'lr': lr},
                        {'params': list(br_output_params), 'lr': lr}]
            else:
                raise NotImplementedError()
                
    def build_br(self, prev_treatments, vitals_or_prev_outputs, static_features, active_entries, encoder_br=None,
                 active_encoder_br=None):
        x = torch.cat((prev_treatments, vitals_or_prev_outputs), dim=-1)
        x = torch.cat((x, static_features.unsqueeze(1).expand(-1, x.size(1), -1)), dim=-1)
        x = self.input_transformation(x)
        
        if active_encoder_br is None and encoder_br is None:  # Only self-attention
            for block in self.transformer_blocks:
                if self.self_positional_encoding is not None:
                    x = x + self.self_positional_encoding(x)
                x = block(x, active_entries)
        else:  # Both self-attention and cross-attention
            assert x.shape[-1] == encoder_br.shape[-1]
            for block in self.transformer_blocks:
                if self.cross_positional_encoding is not None:
                    encoder_br = encoder_br + self.cross_positional_encoding(encoder_br)
                if self.self_positional_encoding is not None:
                    x = x + self.self_positional_encoding(x)
                x = block(x, encoder_br, active_entries, active_encoder_br)
                
        output = self.output_dropout(x)
        br = self.br_treatment_outcome_head.build_br(output)
        return br