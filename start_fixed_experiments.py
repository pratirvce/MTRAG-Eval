#!/usr/bin/env python3
"""
Start fixed experiments and new high-priority experiments
Runs in parallel with resume support
"""

import subprocess
import pathlib
import json
import time
import logging
from typing import List, Dict

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

def get_available_gpus() -> List[int]:
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

def start_experiment(exp_name: str, script: str, gpu_id: int, output_dir: pathlib.Path, resume: bool = True) -> subprocess.Popen:
    """Start an experiment in the background"""
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / "training.log"
    
    # Activate venv if it exists
    venv_activate = ""
    if (pathlib.Path("venv/bin/activate").exists()):
        venv_activate = "source venv/bin/activate && "
    
    cmd = f"cd {pathlib.Path.cwd()} && {venv_activate}python3 {script} --experiment_name {exp_name} --gpu {gpu_id} --output_dir {output_dir} {'--resume' if resume else '--no-resume'}"
    
    logger.info(f"Starting {exp_name} on GPU {gpu_id}")
    logger.info(f"Command: {cmd}")
    
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
    # Experiments to start (fixed + new high-priority)
    experiments = [
        {
            "name": "tier1_learning_to_rank_listwise",
            "script": "train_learning_to_rank_tier1.py",
            "priority": 1,
            "description": "Learning-to-Rank (FIXED)"
        },
        {
            "name": "tier1_pseudo_relevance_feedback",
            "script": "train_pseudo_relevance_feedback_tier1.py",
            "priority": 2,
            "description": "Pseudo-Relevance Feedback (FIXED)"
        },
        {
            "name": "tier1_cross_attention_query_document",
            "script": "train_cross_attention_tier1.py",
            "priority": 3,
            "description": "Cross-Attention Query-Document (FIXED GPU)"
        },
        {
            "name": "tier1_iterative_refinement_improved",
            "script": "train_iterative_refinement_improved_tier1.py",
            "priority": 4,
            "description": "Iterative Refinement Retrieval"
        },
        {
            "name": "tier1_multistage_3stage_finetuned",
            "script": "train_multistage_3stage_finetuned_tier1.py",
            "priority": 5,
            "description": "3-Stage Multi-Stage Retrieval"
        }
    ]
    
    # Get available GPUs
    available_gpus = get_available_gpus()
    logger.info(f"Available GPUs: {available_gpus}")
    
    if not available_gpus:
        logger.error("No GPUs available!")
        return
    
    # Sort experiments by priority
    experiments.sort(key=lambda x: x["priority"])
    
    # Start experiments on available GPUs
    processes = []
    gpu_assignments = {}
    
    for i, exp in enumerate(experiments[:len(available_gpus)]):
        gpu_id = available_gpus[i]
        output_dir = pathlib.Path("experiments/retrieval") / exp["name"]
        
        # Check if already completed
        results_file = output_dir / "results.json"
        if results_file.exists():
            try:
                with open(results_file) as f:
                    data = json.load(f)
                    if 'average' in data and data['average'].get('nDCG@10', 0) > 0:
                        logger.info(f"Skipping {exp['name']} - already completed with score {data['average']['nDCG@10']:.5f}")
                        continue
            except:
                pass
        
        process = start_experiment(
            exp["name"],
            exp["script"],
            gpu_id,
            output_dir,
            resume=True
        )
        
        processes.append((exp["name"], process, gpu_id))
        gpu_assignments[exp["name"]] = gpu_id
        logger.info(f"✅ Started {exp['name']} on GPU {gpu_id} (PID: {process.pid})")
        
        # Small delay between starts
        time.sleep(2)
    
    # Save assignments
    status_file = pathlib.Path("experiment_assignments.json")
    with open(status_file, 'w') as f:
        json.dump(gpu_assignments, f, indent=2)
    
    logger.info(f"\n{'='*80}")
    logger.info(f"Started {len(processes)} experiments:")
    for name, proc, gpu in processes:
        logger.info(f"  • {name} on GPU {gpu} (PID: {proc.pid})")
    logger.info(f"{'='*80}\n")
    
    logger.info("Experiments are running in the background. Check logs in experiments/retrieval/*/training.log")

if __name__ == "__main__":
    main()

