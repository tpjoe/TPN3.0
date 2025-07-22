import sys
import subprocess
import os
from pathlib import Path
from multiprocessing import Pool, cpu_count
import time

# Configuration
n_runs = 5  # Number of runs per disease
diseases = ['bpd', 'nec', 'rop', 'max_chole_TPNEHR', 'ivh', 'pvl', 'rds', 'anemia', 'sepsis', 'jaundice', 'pulm_hem', 'pulm_htn', 'pda', 'death']
base_seed = 42
max_parallel = min(cpu_count(), 4)  # Max number of parallel processes (default: min of CPU count or 4)

# Auto-detect project root or accept as argument
if len(sys.argv) > 1 and sys.argv[1].startswith('--root='):
    project_root = sys.argv[1].split('=')[1]
    sys.argv.pop(1)  # Remove from argv so it doesn't interfere with other args
else:
    # Try to auto-detect project root by looking for key files/directories
    current_dir = Path(__file__).parent
    
    # Look for markers that indicate project root
    while current_dir != current_dir.parent:
        if (current_dir / 'src' / '__init__.py').exists() and \
           (current_dir / 'config').exists() and \
           (current_dir / 'runnables').exists():
            project_root = str(current_dir)
            break
        current_dir = current_dir.parent
    else:
        # If not found, assume current directory
        project_root = os.getcwd()
        print(f"Warning: Could not auto-detect project root. Using current directory: {project_root}")

# Add project root to Python path so imports work correctly
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Set environment variable for child processes
os.environ['TPN3_PROJECT_ROOT'] = project_root

print(f"Project root: {project_root}")
print(f"Running {n_runs} runs for diseases: {diseases}")

# Loop through diseases and runs
for disease in diseases:
    for run_no in range(n_runs):
        seed = base_seed + run_no
        
        cmd = [
            sys.executable, "-m", "runnables.train_multi",
            "+backbone=ct",
            "+dataset=synthetic_neonatal",
            "+backbone/ct_hparams=synthetic_neonatal",
            "exp.gpus=null",
            "exp.logging=False",
            f"exp.seed={seed}",
            "exp.max_epochs=1",
            f"dataset.target_disease={disease}",
            f"+exp.run_number={run_no}",
            "hydra.run.dir=.",
            "hydra.output_subdir=null",
            "hydra.job.chdir=false"
        ]
        
        print(f"\n{'='*60}")
        print(f"Running experiment: Disease={disease}, Run={run_no}, Seed={seed}")
        print(f"{'='*60}")
        print("Command:", " ".join(cmd))
        
        # Run subprocess in the project root directory
        subprocess.run(cmd, cwd=project_root)
        
print(f"\nAll experiments completed!")