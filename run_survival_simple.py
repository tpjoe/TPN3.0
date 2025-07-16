#!/usr/bin/env python3
"""
Run survival training with simplified hazard decoder (BR -> bucket 1 directly)
"""
import subprocess
import sys

# Run the training with simplified CT Survival model
cmd = [
    sys.executable, "-m", "runnables.train_multi_survival",
    "+backbone=ct_survival",
    "+dataset=survival",
    "+backbone/ct_hparams=survival",
    "exp.gpus=null",  # Use CPU for local testing
    "exp.logging=False",
    "exp.seed=42",
    "exp.max_epochs=2",
    "dataset.sliding_windows=False",
    "dataset.binary_loss_weight=0.5"
]

print("Running command:")
print(" ".join(cmd))
subprocess.run(cmd)