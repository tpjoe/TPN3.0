"""
PC-Hazard loss functions for survival analysis
Based on SurvTRACE implementation
"""
import torch
import torch.nn.functional as F
from torch import Tensor


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


def pc_hazard_loss(hazard_pred, event_time_bucket, event_indicator, 
                   sequence_lengths=None, last_timepoint_only=True,
                   reduction='mean', focal_gamma=0.0):
    """PC-Hazard loss for survival analysis with optional focal loss modulation
    
    Args:
        hazard_pred: Predicted hazards of shape (batch_size, seq_len, n_buckets)
                    These should already be positive (after softplus)
        event_time_bucket: Event/censoring time as bucket index (batch_size,)
                          This indicates WHEN the event/censoring occurs in future time
        event_indicator: Binary event indicator (batch_size,)
                        1 = event occurred, 0 = censored
        sequence_lengths: Actual sequence lengths for each patient (batch_size,)
                         If None, will use all timesteps
        last_timepoint_only: If True, only use last timestep for each sequence
        reduction: 'none', 'mean', or 'sum'
        focal_gamma: Focal loss gamma parameter (0 = no focal loss, higher = more focus on hard samples)
                    Recommended values: 0.5, 1.0, 2.0 for imbalanced data
    
    Returns:
        PC-Hazard loss
    """
    batch_size, seq_len, n_buckets = hazard_pred.shape
    
    # Ensure inputs are the right type
    event_time_bucket = event_time_bucket.long()
    event_indicator = event_indicator.float()
    
    if last_timepoint_only:
        # Extract predictions from the last valid timestep for each patient
        if sequence_lengths is not None:
            # Use actual sequence lengths
            last_indices = sequence_lengths.long() - 1
        else:
            # Assume all sequences are full length
            last_indices = torch.full((batch_size,), seq_len - 1, device=hazard_pred.device)
        
        # Gather predictions from last timestep
        # Create index tensor for gathering
        batch_indices = torch.arange(batch_size, device=hazard_pred.device)
        hazard_pred_last = hazard_pred[batch_indices, last_indices]  # Shape: (batch_size, n_buckets)
        
        # Now we have one prediction per patient
        hazard_for_loss = hazard_pred_last
    else:
        # For seq2seq, we would need to handle all timesteps
        # This would require event_time_bucket and event_indicator to be 
        # (batch_size, seq_len) which they currently aren't
        raise NotImplementedError("seq2seq mode requires per-timestep event information")
    
    # Add zero padding at the beginning (for t=0 where hazard=0)
    hazard_padded = pad_col(hazard_for_loss, where='start')
    
    # Calculate cumulative hazard up to event/censoring time
    cum_hazard = hazard_padded.cumsum(1)
    
    # Gather cumulative hazard at event/censoring time
    # After padding, bucket index i corresponds to position i in padded array
    idx_durations = event_time_bucket.view(-1, 1)
    sum_hazard = cum_hazard.gather(1, idx_durations).view(-1)
    
    # Get hazard at event time (from unpadded hazard)
    # Clamp to ensure we don't go out of bounds
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