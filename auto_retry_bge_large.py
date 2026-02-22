#!/usr/bin/env python3
"""
Auto-retry phase4_bge_large experiment when GPU becomes available
"""

import json
import pathlib
import subprocess
import sys
import os
import time
import torch

def check_gpu_free(gpu_id, threshold_util=10, threshold_mem=100):
    """Check if GPU is free."""
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
                return util < threshold_util and mem < threshold_mem
    except:
        pass
    return False

def start_experiment(gpu_id):
    """Start phase4_bge_large experiment on specified GPU."""
    from run_phase4_experiments import PHASE4_EXPERIMENTS
    
    exp_name = "phase4_bge_large"
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
    
    # Set CUDA_VISIBLE_DEVICES
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    
    print(f"🚀 Starting {exp_name} on GPU {gpu_id}")
    print(f"   Command: {' '.join(cmd)}")
    
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
        return process.pid
    except Exception as e:
        print(f"   ❌ Failed to start: {e}")
        return None

if __name__ == "__main__":
    print("=" * 60)
    print("Auto-Retry: phase4_bge_large")
    print("=" * 60)
    print("Waiting for GPU 2 to become available...")
    
    # Wait for GPU 2 to free up
    max_wait = 3600  # 1 hour max wait
    check_interval = 30  # Check every 30 seconds
    waited = 0
    
    while waited < max_wait:
        if check_gpu_free(2):
            print(f"\n✅ GPU 2 is now free! Starting experiment...")
            pid = start_experiment(2)
            if pid:
                print(f"\n✅ Successfully started experiment (PID: {pid})")
                print("Monitor with: tail -f experiments/retrieval/phase4_bge_large/training.log")
                sys.exit(0)
            else:
                print("\n❌ Failed to start experiment")
                sys.exit(1)
        
        print(f"   Still waiting... (waited {waited}s, GPU 2 still in use)")
        time.sleep(check_interval)
        waited += check_interval
    
    print(f"\n⏰ Timeout: GPU 2 did not become free within {max_wait}s")
    print("Please check manually and retry later")
    sys.exit(1)

