import torch
from torch import nn
import numpy as np
from torch.autograd import Function
import torch.nn.functional as F
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.loggers import MLFlowLogger
from pytorch_lightning.utilities import rank_zero_only
from copy import deepcopy
from typing import List
from torch import Tensor

def grad_reverse(x, scale=1.0):

    class ReverseGrad(Function):
        """
        Gradient reversal layer
        """

        @staticmethod
        def forward(ctx, x):
            return x

        @staticmethod
        def backward(ctx, grad_output):
            return scale * grad_output.neg()

    return ReverseGrad.apply(x)


class FilteringMlFlowLogger(MLFlowLogger):
    def __init__(self, filter_submodels: List[str] = None, **kwargs):
        super().__init__(**kwargs)
        self.filter_submodels = filter_submodels

    @rank_zero_only
    def log_hyperparams(self, params) -> None:
        params = deepcopy(params)
        [params.model.pop(filter_submodel) for filter_submodel in self.filter_submodels if filter_submodel in params.model]
        super().log_hyperparams(params)


def bce(treatment_pred, current_treatments, mode, weights=None):
    if mode == 'multiclass':
        return F.cross_entropy(treatment_pred.permute(0, 2, 1), current_treatments.permute(0, 2, 1), reduction='none', weight=weights)
    elif mode == 'multilabel':
        return F.binary_cross_entropy_with_logits(treatment_pred, current_treatments, reduction='none', weight=weights).mean(dim=-1)
    elif mode == 'continuous':
        # For continuous treatments, use MSE loss
        return F.mse_loss(treatment_pred, current_treatments, reduction='none').mean(dim=-1)
    else:
        raise NotImplementedError()


class BRTreatmentOutcomeHead(nn.Module):
    """Used by CRN, EDCT, MultiInputTransformer"""

    def __init__(self, seq_hidden_units, br_size, fc_hidden_units, dim_treatments, dim_outcome, alpha=0.0, update_alpha=True,
                 balancing='grad_reverse', num_buckets=None):
        super().__init__()

        self.seq_hidden_units = seq_hidden_units
        self.br_size = br_size
        self.fc_hidden_units = fc_hidden_units
        self.dim_treatments = dim_treatments
        self.dim_outcome = dim_outcome
        self.alpha = alpha if not update_alpha else 0.0
        self.alpha_max = alpha
        self.balancing = balancing
        self.num_buckets = num_buckets

        self.linear1 = nn.Linear(self.seq_hidden_units, self.br_size)
        self.elu1 = nn.ELU()

        self.linear2 = nn.Linear(self.br_size, self.fc_hidden_units)
        self.elu2 = nn.ELU()
        self.linear3 = nn.Linear(self.fc_hidden_units, self.dim_treatments)

        self.linear4 = nn.Linear(self.br_size + self.dim_treatments, self.fc_hidden_units)
        self.elu3 = nn.ELU()
        self.linear5 = nn.Linear(self.fc_hidden_units, self.dim_outcome)

        # Add bucket prediction head if num_buckets is specified
        if self.num_buckets is not None and self.num_buckets > 0:
            self.linear6 = nn.Linear(self.br_size + self.dim_treatments, self.fc_hidden_units)
            self.elu4 = nn.ELU()
            self.linear7 = nn.Linear(self.fc_hidden_units, self.num_buckets)

        self.treatment_head_params = ['linear2', 'linear3']

    def build_treatment(self, br, detached=False):
        if detached:
            br = br.detach()

        if self.balancing == 'grad_reverse':
            br = grad_reverse(br, self.alpha)

        br = self.elu2(self.linear2(br))
        treatment = self.linear3(br)  # Softmax is encapsulated into F.cross_entropy()
        return treatment

    def build_outcome(self, br, current_treatment):
        x = torch.cat((br, current_treatment), dim=-1)
        x = self.elu3(self.linear4(x))
        outcome = self.linear5(x)
        return outcome

    def build_br(self, seq_output):
        br = self.elu1(self.linear1(seq_output))
        return br
    
    def build_bucket(self, br, current_treatment):
        """Build bucket predictions for horizon-based classification"""
        if self.num_buckets is None:
            return None
        x = torch.cat((br, current_treatment), dim=-1)
        x = self.elu4(self.linear6(x))
        bucket_pred = self.linear7(x)
        return bucket_pred


class ROutcomeVitalsHead(nn.Module):
    """Used by G-Net"""

    def __init__(self, seq_hidden_units, r_size, fc_hidden_units, dim_outcome, dim_vitals, num_comp, comp_sizes):
        super().__init__()

        self.seq_hidden_units = seq_hidden_units
        self.r_size = r_size
        self.fc_hidden_units = fc_hidden_units
        self.dim_outcome = dim_outcome
        self.dim_vitals = dim_vitals
        self.num_comp = num_comp
        self.comp_sizes = comp_sizes

        self.linear1 = nn.Linear(self.seq_hidden_units, self.r_size)
        self.elu1 = nn.ELU()

        # Conditional distribution networks init
        self.cond_nets = []
        add_input_dim = 0
        for comp in range(self.num_comp):
            linear2 = nn.Linear(self.r_size + add_input_dim, self.fc_hidden_units)
            elu2 = nn.ELU()
            linear3 = nn.Linear(self.fc_hidden_units, self.comp_sizes[comp])
            self.cond_nets.append(nn.Sequential(linear2, elu2, linear3))

            add_input_dim += self.comp_sizes[comp]

        self.cond_nets = nn.ModuleList(self.cond_nets)

    def build_r(self, seq_output):
        r = self.elu1(self.linear1(seq_output))
        return r

    def build_outcome_vitals(self, r):
        vitals_outcome_pred = []
        for cond_net in self.cond_nets:
            out = cond_net(r)
            r = torch.cat((out, r), dim=-1)
            vitals_outcome_pred.append(out)
        return torch.cat(vitals_outcome_pred, dim=-1)


class AlphaRise(Callback):
    """
    Exponential alpha rise
    """
    def __init__(self, rate='exp'):
        self.rate = rate

    def on_epoch_end(self, trainer, pl_module) -> None:
        if pl_module.hparams.exp.update_alpha:
            assert hasattr(pl_module, 'br_treatment_outcome_head')
            p = float(pl_module.current_epoch + 1) / float(pl_module.hparams.exp.max_epochs)
            if self.rate == 'lin':
                pl_module.br_treatment_outcome_head.alpha = p * pl_module.br_treatment_outcome_head.alpha_max
            elif self.rate == 'exp':
                pl_module.br_treatment_outcome_head.alpha = \
                    (2. / (1. + np.exp(-10. * p)) - 1.0) * pl_module.br_treatment_outcome_head.alpha_max
            else:
                raise NotImplementedError()


def clip_normalize_stabilized_weights(stabilized_weights, active_entries, multiple_horizons=False):
    """
    Used by RMSNs
    """
    active_entries = active_entries.astype(bool)
    stabilized_weights[~np.squeeze(active_entries)] = np.nan
    sw_tilde = np.clip(stabilized_weights, np.nanquantile(stabilized_weights, 0.01), np.nanquantile(stabilized_weights, 0.99))
    if multiple_horizons:
        sw_tilde = sw_tilde / np.nanmean(sw_tilde, axis=0, keepdims=True)
    else:
        sw_tilde = sw_tilde / np.nanmean(sw_tilde)

    sw_tilde[~np.squeeze(active_entries)] = 0.0
    return sw_tilde


"""
PC-Hazard loss functions for survival analysis
Based on SurvTRACE implementation
"""



def pad_col(haz, where='start'):
    """Add a column of zeros to hazard predictions
    
    Args:
        haz: hazard predictions of shape (batch_size, n_intervals)
        where: 'start' or 'end' - where to add the zero column
    
    Returns:
        padded hazard predictions of shape (batch_size, n_intervals + 1)
    """
    batch_size = haz.shape[0]
    device = haz.device
    dtype = haz.dtype
    
    zero_col = torch.zeros(batch_size, 1, dtype=dtype, device=device)
    
    if where == 'start':
        return torch.cat([zero_col, haz], dim=1)
    else:
        return torch.cat([haz, zero_col], dim=1)


def log_softplus(input, threshold=-15.):
    """Equivalent to 'F.softplus(input).log()', but for 'input < threshold',
    we return 'input', as this is approximately the same.
    
    Args:
        input: Input tensor
        threshold: Threshold for when to just return input (default: -15)
    
    Returns:
        log(softplus(input))
    """
    output = input.clone()
    above = input >= threshold
    output[above] = F.softplus(input[above]).log()
    return output


def hazard_to_survival_prob(hazard_pred):
    """Convert hazard predictions to survival probabilities
    
    Args:
        hazard_pred: Predicted hazards of shape (batch_size, n_buckets) or (batch_size,)
                    Raw predictions from model (before softplus)
    
    Returns:
        Survival probabilities for each time bucket of shape (batch_size, n_buckets)
    """
    # Ensure positive hazards
    hazard = F.softplus(hazard_pred)
    
    # Handle both single bucket and multi-bucket cases
    if hazard.dim() == 1:
        hazard = hazard.unsqueeze(1)
    
    # Calculate cumulative hazard: H(t) = Σ h(t)
    cum_hazard = hazard.cumsum(dim=1)
    
    # Calculate survival probability: S(t) = exp(-H(t))
    survival_prob = torch.exp(-cum_hazard)
    
    return survival_prob


def hazard_to_event_prob(hazard_pred):
    """Convert hazard predictions to event probabilities
    
    Args:
        hazard_pred: Predicted hazards of shape (batch_size, n_buckets)
                    Raw predictions from model (before softplus)
    
    Returns:
        Event probability by end of each time bucket
    """
    survival_prob = hazard_to_survival_prob(hazard_pred)
    # Event probability: P(event by time t) = 1 - S(t)
    event_prob = 1 - survival_prob
    return event_prob


def pc_hazard_loss(hazard_pred, event_time_bucket, event_indicator, reduction='none', focal_gamma=0.0):
    """PC-Hazard loss for survival analysis with optional focal loss modulation
    
    Args:
        hazard_pred: Predicted hazards of shape (batch_size, n_buckets)
                    These should already be positive (after softplus)
        event_time_bucket: Event/censoring time as bucket index (batch_size,)
                          This indicates WHEN the event/censoring occurs in future time
        event_indicator: Binary event indicator (batch_size,)
                        1 = event occurred, 0 = censored
        reduction: 'none', 'mean', or 'sum'
        focal_gamma: Focal loss gamma parameter (0 = no focal loss, higher = more focus on hard samples)
                    Recommended values: 0.5, 1.0, 2.0 for imbalanced data
    
    Returns:
        PC-Hazard loss
    """
    # Debug shape issue
    # if hazard_pred.shape[1] != 2:
    #     print(f"WARNING: hazard_pred has shape {hazard_pred.shape}, expected (batch_size, 2)")
    #     exit()
    
    _, n_buckets = hazard_pred.shape
    
    # Ensure inputs are the right type
    event_time_bucket = event_time_bucket.long()
    event_indicator = event_indicator.float()
    
    hazard_for_loss = F.softplus(hazard_pred)
    
    # Add zero padding at the beginning (for t=0 where hazard=0)
    hazard_padded = pad_col(hazard_for_loss, where='end')
    # Calculate cumulative hazard up to event/censoring time
    cum_hazard = hazard_padded.cumsum(1)
    
    # Gather cumulative hazard at event/censoring time
    # After padding, bucket index i corresponds to position i in padded array
    idx_durations = event_time_bucket.view(-1, 1)
    sum_hazard = cum_hazard.gather(1, idx_durations).view(-1)
    
    # Get hazard at event time (from unpadded hazard)
    idx_durations_clamped = torch.clamp(event_time_bucket, 0, n_buckets - 1)
    hazard_at_event = hazard_for_loss.gather(1, idx_durations_clamped.view(-1, 1)).view(-1)
    
    # Calculate log hazard for events (with numerical stability)
    log_hazard_at_event = torch.log(hazard_at_event + 1e-7)
    
    # PC-Hazard loss components:
    # 1. For events (event_indicator=1): -log(hazard at event time)
    # 2. For all (events and censored): cumulative hazard up to event/censoring time
    loss_per_patient = -event_indicator * log_hazard_at_event + sum_hazard
    
    # Apply focal loss modulation if gamma > 0
    if focal_gamma > 0:
        # Calculate survival probability at event/censoring time
        # S(t) = exp(-cumulative_hazard)
        survival_prob = torch.exp(-sum_hazard)
        
        # Define confidence q based on sample type
        # For events: confidence = 1 - survival_prob (want low survival)
        # For censored: confidence = survival_prob (want high survival)
        q = torch.where(
            event_indicator > 0,
            1 - survival_prob,  # Events: confidence when predicting death correctly
            survival_prob       # Censored: confidence when predicting survival correctly
        )
        
        # Apply focal modulation: (1 - q)^gamma
        # This down-weights easy samples (high q) and emphasizes hard samples (low q)
        focal_weight = (1 - q).pow(focal_gamma)
        loss_per_patient = focal_weight * loss_per_patient
    
    # Apply reduction
    if reduction == 'none':
        return loss_per_patient
    elif reduction == 'mean':
        return loss_per_patient.mean()
    elif reduction == 'sum':
        return loss_per_patient.sum()
    else:
        raise ValueError(f"Invalid reduction: {reduction}")


def save_checkpoint_with_ema(model, checkpoint_path, trainer=None, **kwargs):
    """
    Save checkpoint with EMA weights if available.
    
    Args:
        model: The model to save
        checkpoint_path: Path to save checkpoint
        trainer: Optional trainer object for additional state
        **kwargs: Additional items to save in checkpoint
    """
    checkpoint = {
        'state_dict': model.state_dict(),
        **kwargs
    }
    
    # Add trainer state if available
    if trainer is not None:
        checkpoint['epoch'] = trainer.current_epoch
        checkpoint['global_step'] = trainer.global_step
    
    # Check if model has EMA
    if hasattr(model, 'ema_treatment') and hasattr(model, 'ema_non_treatment'):
        # Save EMA states
        checkpoint['ema_treatment_state'] = model.ema_treatment.state_dict()
        checkpoint['ema_non_treatment_state'] = model.ema_non_treatment.state_dict()
        
        # Also save a version with EMA weights applied
        with model.ema_non_treatment.average_parameters():
            with model.ema_treatment.average_parameters():
                checkpoint['ema_state_dict'] = model.state_dict()
    
    torch.save(checkpoint, checkpoint_path)
    return checkpoint


def load_checkpoint_with_ema(model, checkpoint_path, use_ema=True):
    """
    Load checkpoint and optionally use EMA weights.
    
    Args:
        model: Model to load weights into
        checkpoint_path: Path to checkpoint file
        use_ema: If True and available, use EMA weights
    
    Returns:
        The loaded checkpoint dict
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    # Determine which state dict to use
    if use_ema and 'ema_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['ema_state_dict'])
        print(f"Loaded checkpoint with EMA weights from {checkpoint_path}")
        
        # Also restore EMA objects if they exist
        if hasattr(model, 'ema_treatment') and 'ema_treatment_state' in checkpoint:
            model.ema_treatment.load_state_dict(checkpoint['ema_treatment_state'])
        if hasattr(model, 'ema_non_treatment') and 'ema_non_treatment_state' in checkpoint:
            model.ema_non_treatment.load_state_dict(checkpoint['ema_non_treatment_state'])
    else:
        # Load regular state dict
        if 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            # Old style checkpoint - just the state dict
            model.load_state_dict(checkpoint)
        
        if use_ema and 'ema_state_dict' not in checkpoint:
            print("Warning: EMA weights requested but not found in checkpoint")
    
    return checkpoint


