#!/usr/bin/env python3
"""
Start all novel and fixed experiments in parallel on free GPUs
"""

import subprocess
import pathlib
import json
import time
import logging
import os

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

def get_available_gpus():
    """Get list of available GPU IDs"""
    result = subprocess.run(['nvidia-smi', '--query-gpu=index,utilization.gpu,memory.used,memory.total', '--format=csv,noheader,nounits'], 
                          capture_output=True, text=True)
    
    available_gpus = []
    for line in result.stdout.strip().split('\n'):
        if ',' in line:
            parts = line.split(', ')
            if len(parts) >= 4:
                gpu_idx = int(parts[0])
                util = int(parts[1])
                mem_used = int(parts[2])
                mem_total = int(parts[3])
                mem_pct = (mem_used / mem_total) * 100
                
                if util < 10 and mem_pct < 10:
                    available_gpus.append(gpu_idx)
    
    return available_gpus

def is_experiment_running(exp_name):
    """Check if experiment is already running"""
    result = subprocess.run(['pgrep', '-f', exp_name], capture_output=True, text=True)
    return result.returncode == 0

def is_experiment_completed(exp_name):
    """Check if experiment is already completed with valid results"""
    results_file = pathlib.Path(f"experiments/retrieval/{exp_name}/results.json")
    if results_file.exists():
        try:
            with open(results_file) as f:
                data = json.load(f)
                score = data.get('average', {}).get('nDCG@10', 0)
                if score > 0:
                    return True
        except:
            pass
    return False

def start_experiment(exp_name, script, gpu_id, output_dir, resume=True):
    """Start an experiment in the background"""
    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / "training.log"
    
    # Use venv python directly
    if pathlib.Path("venv/bin/python3").exists():
        python_cmd = "venv/bin/python3"
    else:
        python_cmd = "python3"
    
    cmd = f"cd {pathlib.Path.cwd()} && {python_cmd} {script} --experiment_name {exp_name} --gpu {gpu_id} --output_dir {output_dir} {'--resume' if resume else '--no-resume'}"
    
    logger.info(f"Starting {exp_name} on GPU {gpu_id}")
    
    with open(log_file, 'a') as f:
        f.write(f"\n{'='*80}\n")
        f.write(f"Started: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"GPU: {gpu_id}\n")
        f.write(f"Command: {cmd}\n")
        f.write(f"{'='*80}\n")
    
    process = subprocess.Popen(
        cmd,
        shell=True,
        stdout=open(log_file, 'a'),
        stderr=subprocess.STDOUT,
        cwd=pathlib.Path.cwd()
    )
    
    return process

def main():
    # All experiments to start
    experiments = [
        # Novel experiments
        {
            "name": "tier1_enhanced_contrastive_hardnegatives",
            "script": "train_enhanced_contrastive_tier1.py",
            "priority": 1,
            "description": "Enhanced Contrastive Learning (Novel)"
        },
        {
            "name": "tier1_qdit_transformer",
            "script": "train_qdit_transformer_tier1.py",
            "priority": 2,
            "description": "QDIT Transformer (Novel)"
        },
        # Fixed failed experiments
        {
            "name": "tier1_pseudo_relevance_feedback",
            "script": "train_pseudo_relevance_feedback_tier1.py",
            "priority": 3,
            "description": "Pseudo-Relevance Feedback (Fixed)"
        },
        {
            "name": "tier1_learning_to_rank_listwise",
            "script": "train_learning_to_rank_tier1.py",
            "priority": 4,
            "description": "Learning-to-Rank (Fixed)"
        },
        {
            "name": "tier1_cross_attention_query_document",
            "script": "train_cross_attention_tier1.py",
            "priority": 5,
            "description": "Cross-Attention Query-Document (Fixed)"
        }
    ]
    
    # Get available GPUs
    available_gpus = get_available_gpus()
    logger.info(f"Available GPUs: {available_gpus}")
    
    if not available_gpus:
        logger.error("No GPUs available!")
        return
    
    # Filter experiments: skip if running or completed
    experiments_to_start = []
    for exp in experiments:
        exp_name = exp["name"]
        
        if is_experiment_running(exp_name):
            logger.info(f"Skipping {exp_name} - already running")
            continue
        
        if is_experiment_completed(exp_name):
            logger.info(f"Skipping {exp_name} - already completed")
            continue
        
        experiments_to_start.append(exp)
    
    # Sort by priority
    experiments_to_start.sort(key=lambda x: x["priority"])
    
    # Start experiments on available GPUs
    processes = []
    gpu_assignments = {}
    
    for i, exp in enumerate(experiments_to_start[:len(available_gpus)]):
        gpu_id = available_gpus[i]
        output_dir = pathlib.Path("experiments/retrieval") / exp["name"]
        
        process = start_experiment(
            exp["name"],
            exp["script"],
            gpu_id,
            output_dir,
            resume=True
        )
        
        processes.append((exp["name"], process, gpu_id, exp["description"]))
        gpu_assignments[exp["name"]] = gpu_id
        logger.info(f"✅ Started {exp['name']} on GPU {gpu_id} (PID: {process.pid})")
        
        # Small delay between starts
        time.sleep(2)
    
    # Save assignments
    status_file = pathlib.Path("all_experiment_assignments.json")
    with open(status_file, 'w') as f:
        json.dump(gpu_assignments, f, indent=2)
    
    logger.info(f"\n{'='*80}")
    logger.info(f"Started {len(processes)} experiments:")
    for name, proc, gpu, desc in processes:
        logger.info(f"  • {name[:50]:50} GPU {gpu} (PID: {proc.pid}) - {desc}")
    logger.info(f"{'='*80}\n")
    
    logger.info("Experiments are running in the background with resume support.")
    logger.info("Check logs in experiments/retrieval/*/training.log")

if __name__ == "__main__":
    main()

