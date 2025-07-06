#!/usr/bin/env python
"""
Simple script to train CT model on synthetic neonatal data
"""
import sys
import subprocess

def main():
    cmd = [
        sys.executable, "-m", "runnables.train_multi",
        "+backbone=ct",
        "+dataset=synthetic_neonatal",
        "+backbone/ct_hparams=synthetic_neonatal",
        "exp.gpus=null",  # Use CPU
        "exp.logging=False",  # Disable MLflow logging
        "exp.seed=42",
        "exp.max_epochs=70"
    ]
    
    print("Running command:", " ".join(cmd))
    subprocess.run(cmd)

if __name__ == "__main__":
    main()