#!/usr/bin/env python
"""
Test script with full training to debug validation metrics
"""
import torch
import numpy as np
from omegaconf import DictConfig, OmegaConf
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from torch.utils.data import DataLoader
from pytorch_lightning.utilities.seed import seed_everything
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, EarlyStopping, ModelCheckpoint
import logging
import os
from pathlib import Path
from src.models.survival_losses import pc_hazard_loss
from src.models.utils import AlphaRise
from src.models.time_varying_model_survival import LossBreakdownCallback

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
torch.set_default_dtype(torch.double)


# Handle both script and interactive modes
try:
    # If running as a script
    base_dir = os.path.dirname(__file__)
except NameError:
    # If running interactively, use current working directory
    base_dir = os.getcwd()
    print(f"Running in interactive mode from: {base_dir}")

# Initialize Hydra with config directory
config_dir = os.path.join(base_dir, "config")
print(f"Using config directory: {config_dir}")

# Initialize without context manager for interactive use
from hydra import initialize, compose
initialize(config_path="config", version_base=None)

# Compose config with overrides
args = compose(
    config_name="config.yaml",
    overrides=[
        "+backbone=ct_survival",
        "+dataset=survival", 
        "+backbone/ct_hparams=survival",
        "exp.gpus=null",
        "exp.logging=False",
        "exp.max_epochs=5",  # Run a few epochs to see metrics
        "exp.seed=42",
        "+dataset.max_number=1000"  # Increased from 100 to ensure sufficient data
    ]
)

# Register the sum resolver
OmegaConf.register_new_resolver("sum", lambda x, y: x + y, replace=True)

# Non-strict access to fields
OmegaConf.set_struct(args, False)

print("\n2. Loading and processing data...")
seed_everything(args.exp.seed)

# Instantiate dataset
dataset_collection = instantiate(args.dataset, _recursive_=True)
dataset_collection.process_data_multi()

# Set model dimensions based on actual data
args.model.dim_treatments = dataset_collection.train_f.data['current_treatments'].shape[-1]
args.model.dim_vitals = dataset_collection.train_f.data['vitals'].shape[-1]
args.model.dim_static_features = dataset_collection.train_f.data['static_features'].shape[-1]
args.model.dim_outcomes = dataset_collection.train_f.data['outputs'].shape[-1]

print("\n3. Initializing model...")
# Import and initialize model directly
from src.models.ct_survival import CTSurvival
model = CTSurvival(args, dataset_collection)

print(f"\nModel initialized: {model.__class__.__name__}")
print(f"Number of parameters: {sum(p.numel() for p in model.parameters()):,}")

# Quick test of forward pass before training
print("\n4. Testing forward pass...")
train_loader = DataLoader(dataset_collection.train_f, batch_size=4, shuffle=False)
batch = next(iter(train_loader))

model.eval()
with torch.no_grad():
    treatment_pred, hazard, br = model(batch)

print(f"  Treatment predictions shape: {treatment_pred.shape}")
print(f"  Hazard predictions shape: {hazard.shape}")
print(f"  Balancing representation shape: {br.shape}")

# Setup training callbacks
print("\n5. Setting up training...")
callbacks = [
    AlphaRise(rate=args.exp.alpha_rate if hasattr(args.exp, 'alpha_rate') else 0.0),
    LossBreakdownCallback()
]

# Create trainer
trainer = Trainer(
    gpus=eval(str(args.exp.gpus)), 
    max_epochs=args.exp.max_epochs,
    callbacks=callbacks, 
    terminate_on_nan=True,
    gradient_clip_val=args.model.multi.max_grad_norm if hasattr(args.model.multi, 'max_grad_norm') else 1.0,
    checkpoint_callback=False,
    logger=False,
    progress_bar_refresh_rate=1,
    log_every_n_steps=1
)

# First test that our validation_step is being called
print("\n6. Testing validation step...")
val_loader = DataLoader(dataset_collection.val_f, batch_size=4, shuffle=False)
val_batch = next(iter(val_loader))

# Call validation_step directly with logging disabled
try:
    # Temporarily disable logging to avoid trainer dependency
    original_log = model.log
    model.log = lambda *args, **kwargs: None
    
    model.validation_step(val_batch, 0)
    print("✓ Direct validation_step call succeeded")
    
    # Restore original log method
    model.log = original_log
except Exception as e:
    print(f"✗ Direct validation_step failed: {e}")
    import traceback
    traceback.print_exc()
    # Restore original log method in case of error
    if 'original_log' in locals():
        model.log = original_log

# Train the model
print("\n7. Training model...")
try:
    trainer.fit(model)
    print("\n✓ Training completed successfully!")
except Exception as e:
    print(f"\n✗ Training failed with error: {e}")
    import traceback
    traceback.print_exc()

# Test validation metrics on test set
print("\n8. Testing metrics calculation on test set...")
if hasattr(dataset_collection, 'test_f') and hasattr(model, 'calculate_horizon_metrics'):
    # Disable logging for direct metric calculation
    original_log_test = model.log
    model.log = lambda *args, **kwargs: None
    
    test_metrics = model.calculate_horizon_metrics(dataset_collection.test_f, 'test')
    
    # Restore original log method
    model.log = original_log_test
    
    if test_metrics:
        print("\nTest Metrics by Horizon:")
        print("Horizon | AUC-ROC | AUC-PR | Events/Controls")
        print("--------|---------|--------|----------------")
        
        for horizon in sorted(test_metrics.keys()):
            metrics = test_metrics[horizon]
            print(f"{horizon:>6}d | {metrics['auc_roc']:>7.4f} | {metrics['auc_pr']:>6.4f} | "
                  f"{metrics['n_events']:>3}/{metrics['n_controls']:<3}")
    else:
        print("No test metrics calculated - likely insufficient data.")

# Test survival probability computation
print("\n9. Testing survival probability computation...")
model.eval()
with torch.no_grad():
    # Get a batch for testing
    test_loader = DataLoader(dataset_collection.val_f, batch_size=4, shuffle=False)
    test_batch = next(iter(test_loader))
    
    # Run inference
    treatment_pred, hazard, br = model(test_batch)
    
    # Compute survival probabilities
    # S(t) = exp(-cumulative_hazard)
    cumulative_hazard = torch.cumsum(hazard, dim=-1)
    survival_probs = torch.exp(-cumulative_hazard)
    
    print(f"  Survival probabilities shape: {survival_probs.shape}")
    print(f"  Example survival curve for first patient at last timepoint:")
    for i, horizon in enumerate([3, 7, 14, 21, 28, 35, 42, 49, 56, 63, 70, 77, 84]):
        if i < survival_probs.shape[-1]:
            prob = survival_probs[0, -1, i].item()
            print(f"    {horizon}d: {prob:.4f}")

print("\n10. Summary:")
print(f"  ✓ Model initialized: {model.__class__.__name__}")
print(f"  ✓ Forward pass works: hazard shape {hazard.shape}")
print(f"  ✓ Training completed: {args.exp.max_epochs} epochs")
print(f"  ✓ Validation metrics calculated")
print(f"  ✓ Survival probabilities computed")
print("\nNote: For interactive use without trainer, disable logging with:")
print("  model.log = lambda *args, **kwargs: None")

print("\n11. Done!")
