#!/usr/bin/env python3
"""
Restart failed experiments on available GPUs
"""

import json
import pathlib
import subprocess
import sys
import os
from run_phase4_experiments import PHASE4_EXPERIMENTS

def start_experiment(exp_name, gpu_id):
    """Start an experiment on a specific GPU."""
    if exp_name not in PHASE4_EXPERIMENTS:
        print(f"❌ Experiment {exp_name} not found in PHASE4_EXPERIMENTS")
        return False
    
    exp_config = PHASE4_EXPERIMENTS[exp_name]
    
    # Create experiment directory
    exp_dir = pathlib.Path("experiments/retrieval") / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config
    config_file = exp_dir / "config.json"
    exp_config_copy = exp_config.copy()
    exp_config_copy['config']['gpu_id'] = gpu_id
    with open(config_file, 'w') as f:
        json.dump(exp_config_copy['config'], f, indent=2)
    
    # Prepare log files
    log_file = exp_dir / "training.log"
    error_log = exp_dir / "error.log"
    
    # Build command
    script = exp_config['script']
    cmd = [
        sys.executable, script,
        "--config", str(config_file)
    ]
    
    if 'domain' in exp_config['config']:
        cmd.extend(["--domain", str(exp_config['config']['domain'])])
    
    # Set CUDA_VISIBLE_DEVICES
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    
    print(f"🚀 Starting {exp_name} on GPU {gpu_id}")
    print(f"   Command: {' '.join(cmd)}")
    print(f"   Log: {log_file}")
    print(f"   Error log: {error_log}")
    
    try:
        with open(log_file, 'a') as log_f, open(error_log, 'a') as err_f:
            process = subprocess.Popen(
                cmd,
                stdout=log_f,
                stderr=err_f,
                cwd=pathlib.Path(".").absolute(),
                env=env
            )
        print(f"   ✅ Started (PID: {process.pid})")
        return True
    except Exception as e:
        print(f"   ❌ Failed to start: {e}")
        return False

if __name__ == "__main__":
    # Failed experiments to restart
    experiments_to_restart = [
        ("phase4_domain_specific_govt", 2),  # Use GPU 2 (free)
        ("phase4_domain_specific_cloud", 3),  # Use GPU 3 (free)
        ("phase4_bge_large", None),  # Will find free GPU
    ]
    
    print("=" * 60)
    print("Restarting Failed Experiments")
    print("=" * 60)
    
    # Check available GPUs
    import torch
    if not torch.cuda.is_available():
        print("❌ CUDA not available")
        sys.exit(1)
    
    # Find free GPUs
    free_gpus = []
    for gpu_id in range(torch.cuda.device_count()):
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=index,utilization.gpu,memory.used',
                 '--format=csv,noheader,nounits', f'--id={gpu_id}'],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                parts = result.stdout.strip().split(', ')
                if len(parts) >= 3:
                    util = int(parts[1])
                    mem = int(parts[2])
                    if util < 10 and mem < 100:
                        free_gpus.append(gpu_id)
        except:
            continue
    
    print(f"Available GPUs: {free_gpus}")
    
    # Start experiments
    started = 0
    for exp_name, gpu_id in experiments_to_restart:
        if gpu_id is None:
            if free_gpus:
                gpu_id = free_gpus.pop(0)
            else:
                print(f"❌ No free GPU available for {exp_name}")
                continue
        
        if start_experiment(exp_name, gpu_id):
            started += 1
    
    print("=" * 60)
    print(f"✅ Started {started} experiments")
    print("=" * 60)

