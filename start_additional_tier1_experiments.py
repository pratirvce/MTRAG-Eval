#!/usr/bin/env python3
"""
Start additional Tier 1 experiments on free GPUs
Focuses on high-priority experiments that haven't been run or need re-running
"""

import os
import sys
import subprocess
import pathlib
import json
import time
import logging
from typing import List, Tuple, Optional

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

def get_free_gpus() -> List[int]:
    """Get list of free GPUs (utilization < 10%)"""
    try:
        import torch
        if not torch.cuda.is_available():
            return []
        
        free_gpus = []
        for i in range(torch.cuda.device_count()):
            # Check if GPU is being used by checking processes
            # Simple heuristic: if nvidia-smi shows low utilization, it's free
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=index,utilization.gpu', '--format=csv,noheader,nounits'],
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                for line in lines:
                    parts = line.split(', ')
                    if len(parts) == 2:
                        gpu_idx = int(parts[0])
                        util = int(parts[1])
                        if util < 10:  # Less than 10% utilization
                            free_gpus.append(gpu_idx)
        return sorted(free_gpus)
    except Exception as e:
        logger.warning(f"Could not detect free GPUs: {e}")
        # Assume all GPUs are available
        return list(range(6))

def check_experiment_status(exp_name: str) -> Tuple[bool, str]:
    """Check if experiment needs to be run"""
    exp_dir = pathlib.Path(f"experiments/retrieval/{exp_name}")
    
    if not exp_dir.exists():
        return True, "not_started"
    
    results_file = exp_dir / "results.json"
    if not results_file.exists():
        return True, "in_progress"
    
    try:
        with open(results_file) as f:
            data = json.load(f)
        
        # Check if it's a valid completed result
        if 'average' in data:
            avg = data['average']
            if avg.get('nDCG@10', 0) > 0:
                return False, "completed"
            else:
                return True, "failed_needs_rerun"
        elif 'status' in data:
            if data['status'] == 'training_completed' and 'nDCG@10' not in str(data):
                return True, "needs_evaluation"
            elif data['status'] == 'failed':
                return True, "failed_needs_rerun"
        
        return False, "completed"
    except Exception as e:
        logger.warning(f"Error checking {exp_name}: {e}")
        return True, "needs_check"

def start_experiment(exp_name: str, script: str, gpu: int, output_dir: str, resume: bool = True):
    """Start an experiment on a specific GPU"""
    logger.info(f"Starting {exp_name} on GPU {gpu}")
    
    # Activate venv if it exists
    venv_python = pathlib.Path("venv/bin/python")
    if venv_python.exists():
        python_cmd = str(venv_python)
    else:
        python_cmd = "python3"
    
    cmd = [
        python_cmd,
        script,
        "--experiment_name", exp_name,
        "--gpu", str(gpu),
        "--output_dir", output_dir
    ]
    
    if resume:
        cmd.append("--resume")
    else:
        cmd.append("--no-resume")
    
    # Run in background
    log_file = pathlib.Path(f"experiments/retrieval/{exp_name}/training.log")
    log_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(log_file, 'w') as log:
        process = subprocess.Popen(
            cmd,
            stdout=log,
            stderr=subprocess.STDOUT,
            cwd=pathlib.Path.cwd()
        )
    
    logger.info(f"Started {exp_name} on GPU {gpu} (PID: {process.pid})")
    return process.pid

def main():
    """Main function to start additional Tier 1 experiments"""
    
    # Get free GPUs
    free_gpus = get_free_gpus()
    logger.info(f"Free GPUs: {free_gpus}")
    
    if not free_gpus:
        logger.warning("No free GPUs available. Waiting for GPUs to free up...")
        return
    
    # Define experiments to run
    experiments = [
        {
            "name": "tier1_cross_encoder_evaluation",
            "script": "evaluate_cross_encoder_tier1.py",
            "priority": 1,
            "description": "Evaluate fine-tuned cross-encoder (training already done)"
        },
        {
            "name": "tier1_cross_attention_rerun",
            "script": "train_cross_attention_tier1.py",
            "priority": 2,
            "description": "Re-run cross-attention (previous run failed with 0.0)"
        },
        {
            "name": "tier1_llm_query_expansion",
            "script": "train_llm_query_expansion_tier1.py",
            "priority": 3,
            "description": "LLM-based multi-query expansion"
        },
        {
            "name": "tier1_multistage_2stage",
            "script": "train_multistage_tier1.py",
            "priority": 4,
            "description": "2-stage multi-stage retrieval"
        }
    ]
    
    # Check which experiments need to run
    experiments_to_start = []
    for exp in experiments:
        exp_name = exp["name"]
        should_run, status = check_experiment_status(exp_name)
        
        if should_run:
            logger.info(f"{exp_name}: {status} - will start")
            experiments_to_start.append((exp, status))
        else:
            logger.info(f"{exp_name}: {status} - skipping")
    
    # Sort by priority
    experiments_to_start.sort(key=lambda x: x[0]["priority"])
    
    # Start experiments on free GPUs
    started = []
    for (exp, status), gpu in zip(experiments_to_start, free_gpus):
        exp_name = exp["name"]
        script = exp["script"]
        output_dir = f"experiments/retrieval/{exp_name}"
        
        # Check if script exists
        script_path = pathlib.Path(script)
        if not script_path.exists():
            logger.warning(f"Script {script} not found. Skipping {exp_name}.")
            continue
        
        # For failed experiments, don't resume
        resume = status != "failed_needs_rerun"
        
        try:
            pid = start_experiment(exp_name, script, gpu, output_dir, resume=resume)
            started.append({
                "name": exp_name,
                "gpu": gpu,
                "pid": pid,
                "status": status
            })
            logger.info(f"✅ Started {exp_name} on GPU {gpu}")
        except Exception as e:
            logger.error(f"Failed to start {exp_name}: {e}")
    
    # Summary
    logger.info(f"\n=== Started {len(started)} experiments ===")
    for exp_info in started:
        logger.info(f"  - {exp_info['name']} on GPU {exp_info['gpu']} (PID: {exp_info['pid']})")
    
    if started:
        logger.info(f"\nMonitor logs in: experiments/retrieval/*/training.log")

if __name__ == "__main__":
    main()

