#!/usr/bin/env python3
"""
Monitor new high-impact experiments
"""

import subprocess
import json
import pathlib
import time
from datetime import datetime

def check_gpu_status():
    """Check GPU status excluding 0 and 1"""
    result = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.used,memory.total,utilization.gpu', '--format=csv,noheader,nounits'], 
                          capture_output=True, text=True)
    
    gpus = {}
    for line in result.stdout.strip().split('\n'):
        parts = line.split(', ')
        if len(parts) >= 4:
            gpu_id = int(parts[0])
            if gpu_id in [0, 1]:
                continue
            mem_used = int(parts[1])
            mem_total = int(parts[2])
            util = int(parts[3])
            mem_free = mem_total - mem_used
            gpus[gpu_id] = {
                'mem_free_mb': mem_free,
                'mem_free_gb': mem_free / 1024,
                'util': util,
                'status': 'AVAILABLE' if mem_free > 2000 and util < 50 else 'BUSY'
            }
    return gpus

def check_experiment_status(exp_name):
    """Check if experiment is running, completed, or pending"""
    exp_dir = pathlib.Path(f"experiments/retrieval/{exp_name}")
    
    if not exp_dir.exists():
        return 'PENDING', None, None
    
    results_file = exp_dir / "results.json"
    if results_file.exists():
        try:
            with open(results_file) as f:
                results = json.load(f)
                if 'average' in results and 'nDCG@10' in results['average']:
                    score = results['average']['nDCG@10']
                    return 'COMPLETED', score, None
                elif results.get('status') == 'failed':
                    return 'FAILED', None, results.get('error', 'Unknown error')
        except:
            pass
    
    result = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
    if exp_name in result.stdout:
        for line in result.stdout.split('\n'):
            if exp_name in line and 'train_' in line:
                parts = line.split()
                if len(parts) > 1:
                    pid = parts[1]
                    gpu = None
                    if '--gpu' in line:
                        idx = line.find('--gpu')
                        gpu_str = line[idx:idx+10].split()[1] if idx >= 0 else None
                        if gpu_str and gpu_str.isdigit():
                            gpu = int(gpu_str)
                    return 'RUNNING', None, {'pid': pid, 'gpu': gpu}
    
    return 'STARTED', None, None

def main():
    new_experiments = [
        'tier1_colbert_multivector',
        'tier1_ensemble_meta_learner',
        'tier1_generative_query_expansion',
        'tier1_multistage_llm_scoring',
        'tier1_iterative_query_expansion',
        'tier1_bge_v2_large_asymmetric'
    ]
    
    print("=" * 100)
    print(f"MONITORING NEW EXPERIMENTS - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 100)
    
    # Check GPU status
    print("\n📊 GPU STATUS (Excluding GPU 0 and 1):")
    print("-" * 100)
    gpus = check_gpu_status()
    for gpu_id in sorted(gpus.keys()):
        gpu = gpus[gpu_id]
        status_icon = "✅" if gpu['status'] == 'AVAILABLE' else "🔄"
        print(f"{status_icon} GPU {gpu_id}: {gpu['status']:12} | Free: {gpu['mem_free_gb']:5.1f}GB | Util: {gpu['util']:3d}%")
    
    # Check experiment status
    print("\n🔬 NEW HIGH-IMPACT EXPERIMENTS STATUS:")
    print("-" * 100)
    
    completed = 0
    running = 0
    pending = 0
    
    for exp_name in new_experiments:
        status, score, info = check_experiment_status(exp_name)
        
        if status == 'COMPLETED':
            print(f"✅ {exp_name:45} | COMPLETED | nDCG@10: {score:.4f}")
            completed += 1
        elif status == 'RUNNING':
            gpu_info = info.get('gpu', '?') if info else '?'
            pid_info = info.get('pid', '?') if info else '?'
            print(f"🔄 {exp_name:45} | RUNNING   | GPU: {gpu_info}, PID: {pid_info}")
            running += 1
        elif status == 'FAILED':
            error = info if isinstance(info, str) else 'Unknown'
            print(f"❌ {exp_name:45} | FAILED    | Error: {error[:50]}")
        elif status == 'STARTED':
            print(f"⏸️  {exp_name:45} | STARTED   | (No results yet)")
        else:
            print(f"⏳ {exp_name:45} | PENDING   | (Waiting for GPU)")
            pending += 1
    
    print("\n" + "=" * 100)
    print(f"Summary: {completed} completed, {running} running, {pending} pending")
    print("=" * 100)

if __name__ == "__main__":
    main()

