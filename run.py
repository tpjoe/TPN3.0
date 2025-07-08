import sys
import subprocess

cmd = [
    sys.executable, "-m", "runnables.train_multi",
    "+backbone=ct",
    "+dataset=synthetic_neonatal",
    "+backbone/ct_hparams=synthetic_neonatal",
    "exp.gpus=null",
    "exp.logging=False",
    "exp.seed=42",
    "exp.max_epochs=1000"
]

print("Running command:", " ".join(cmd))

subprocess.run(cmd)