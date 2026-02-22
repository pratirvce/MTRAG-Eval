#!/usr/bin/env python3
"""
Start experiments on all available GPUs
Distributes pending experiments across GPUs 2-5
"""

import sys
import pathlib
from experiment_runner import ExperimentRunner

def start_experiments_on_all_gpus():
    """Start pending experiments on all available GPUs"""
    runner = ExperimentRunner()
    runner.update_status()
    
    # Get pending experiments
    status = runner.get_status_summary()
    pending = [
        name for name, info in status['experiments'].items()
        if info['status'] == 'pending'
    ]
    
    if not pending:
        print("No pending experiments to start")
        return
    
    print(f"Found {len(pending)} pending experiments")
    
    # Get available GPUs (2-5, since 0-1 are in use)
    available_gpus = [2, 3, 4, 5]
    
    # Get script paths
    def get_script_path(experiment_name: str) -> str:
        if "ensemble" in experiment_name:
            return "train_ensemble.py"
        elif "domain_specific" in experiment_name and "hard" in experiment_name:
            return "train_combined_techniques.py"
        elif "reranking" in experiment_name:
            return "train_reranking.py"
        elif "query_expansion" in experiment_name or "query_rewriting" in experiment_name:
            return "train_query_expansion.py"
        elif "hybrid" in experiment_name:
            return "train_hybrid_learned.py"
        elif "domain_specific" in experiment_name:
            return "train_domain_specific_bge.py"
        else:
            return "train_advanced_bge.py"
    
    # Start experiments, distributing across GPUs
    started = 0
    gpu_index = 0
    
    for exp_name in pending:
        if gpu_index >= len(available_gpus):
            print(f"\nAll GPUs are now in use. {len(pending) - started} experiments still pending.")
            break
        
        gpu_id = available_gpus[gpu_index]
        script_path = get_script_path(exp_name)
        
        # Get config path
        exp_dir = pathlib.Path("experiments/retrieval") / exp_name
        config_path = exp_dir / "config.json"
        
        if not config_path.exists():
            print(f"⚠️  Config not found for {exp_name}, skipping")
            continue
        
        print(f"Starting {exp_name} on GPU {gpu_id}...")
        
        if runner.start_experiment(exp_name, script_path, gpu_id):
            started += 1
            gpu_index += 1
            print(f"✅ Started {exp_name} on GPU {gpu_id}")
        else:
            print(f"❌ Failed to start {exp_name}")
    
    print(f"\n✅ Started {started} experiments across GPUs {available_gpus[:gpu_index]}")
    
    # Show final status
    runner.update_status()
    status = runner.get_status_summary()
    print(f"\nCurrent Status:")
    print(f"  Running: {status['running']}")
    print(f"  Pending: {status['pending']}")
    print(f"  Completed: {status['completed']}")

if __name__ == "__main__":
    start_experiments_on_all_gpus()

