import sys
import subprocess
import os
from pathlib import Path
from multiprocessing import Pool, cpu_count

# Configuration
n_runs = 5  # Number of runs per disease
diseases = ['bpd', 'nec', 'rop', 'max_chole_TPNEHR', 'ivh', 'pvl', 'rds', 'anemia', 'sepsis', 'jaundice', 'pulm_hem', 'pulm_htn', 'pda', 'death']
base_seed = 42
max_parallel = 4  # Number of parallel processes
gpu_list = [0, 1, 0, 2, None]  # GPU assignment for each run (None = no GPU/CPU only)

# Auto-detect project root
current_dir = Path(__file__).parent
while current_dir != current_dir.parent:
    if (current_dir / 'src' / '__init__.py').exists() and \
       (current_dir / 'config').exists() and \
       (current_dir / 'runnables').exists():
        project_root = str(current_dir)
        break
    current_dir = current_dir.parent
else:
    project_root = os.getcwd()

# Add project root to Python path
if project_root not in sys.path:
    sys.path.insert(0, project_root)

os.environ['TPN3_PROJECT_ROOT'] = project_root

def run_experiment(args):
    """Run a single experiment"""
    disease, run_no = args
    seed = base_seed + run_no
    
    # Get GPU assignment for this run
    gpu = gpu_list[run_no % len(gpu_list)]
    gpu_arg = "exp.gpus=null" if gpu is None else f"exp.gpus=[{gpu}]"
    
    cmd = [
        sys.executable, "-m", "runnables.train_multi",
        "+backbone=ct",
        "+dataset=synthetic_neonatal",
        "+backbone/ct_hparams=synthetic_neonatal",
        gpu_arg,
        "exp.logging=False",
        f"exp.seed={seed}",
        "exp.max_epochs=1000",
        f"dataset.target_disease={disease}",
        f"+exp.run_number={run_no}",
        "hydra.run.dir=.",
        "hydra.output_subdir=null",
        "hydra.job.chdir=false"
    ]
    
    gpu_str = "CPU" if gpu is None else f"GPU {gpu}"
    print(f"Starting: {disease} run {run_no} on {gpu_str}")
    
    # Suppress all output by redirecting to DEVNULL
    result = subprocess.run(cmd, cwd=project_root, 
                          stdout=subprocess.DEVNULL, 
                          stderr=subprocess.DEVNULL)
    return disease, run_no, result.returncode

# Process one disease at a time
for disease in diseases:
    print(f"\n{'='*60}")
    print(f"Processing disease: {disease}")
    print(f"{'='*60}")
    
    # Create experiment list for this disease only
    experiments = [(disease, run_no) for run_no in range(n_runs)]
    
    print(f"Running {n_runs} experiments in parallel for {disease}")
    
    # Run in parallel for this disease
    with Pool(processes=max_parallel) as pool:
        results = pool.map(run_experiment, experiments)
    
    # Summary for this disease
    successful = sum(1 for _, _, code in results if code == 0)
    print(f"Completed {disease}: {successful}/{n_runs} successful")

print(f"\nAll diseases completed!")