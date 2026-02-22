#!/bin/bash
# Resume hybrid experiments on free GPUs

echo "Checking GPU availability and resuming hybrid experiments..."

# Source virtual environment
source venv/bin/activate 2>/dev/null || echo "⚠️  Virtual environment not found, continuing..."

# Check which GPUs are in use by Phase 6 experiments
USED_GPUS=$(ps aux | grep -E "train_cross_encoder|train_multistage|train_llm_query" | grep -v grep | grep -oP "gpu_id \K[0-9]+" | sort -u)

echo "GPUs in use by Phase 6: $USED_GPUS"

# Get all available GPUs (0-5)
ALL_GPUS=(0 1 2 3 4 5)
FREE_GPUS=()

for gpu in "${ALL_GPUS[@]}"; do
    if [[ ! " $USED_GPUS " =~ " $gpu " ]]; then
        FREE_GPUS+=($gpu)
    fi
done

echo "Free GPUs: ${FREE_GPUS[@]}"

if [ ${#FREE_GPUS[@]} -eq 0 ]; then
    echo "❌ No free GPUs available"
    exit 1
fi

# Load paused experiments
if [ ! -f "hybrid_experiments_paused.json" ]; then
    echo "❌ No paused experiments found (hybrid_experiments_paused.json not found)"
    exit 1
fi

# Resume experiments on free GPUs
python3 << 'PYTHON_SCRIPT'
import json
import subprocess
import os
import sys

# Load paused experiments
with open("hybrid_experiments_paused.json", "r") as f:
    data = json.load(f)

# Get free GPUs
used_gpus = set()
for line in os.popen("ps aux | grep -E 'train_cross_encoder|train_multistage|train_llm_query' | grep -v grep | grep -oP 'gpu_id \\K[0-9]+'").read().strip().split('\n'):
    if line:
        used_gpus.add(int(line))

all_gpus = set(range(6))
free_gpus = sorted(list(all_gpus - used_gpus))

print(f"Free GPUs: {free_gpus}")
print(f"Paused experiments: {len(data['experiments'])}\n")

if not free_gpus:
    print("❌ No free GPUs available")
    sys.exit(1)

# Resume experiments on free GPUs
gpu_index = 0
resumed_count = 0

for exp in data['experiments']:
    if gpu_index >= len(free_gpus):
        print(f"\n⚠️  No more free GPUs. {len(data['experiments']) - resumed_count} experiments remaining.")
        break
    
    exp_name = exp['exp_name']
    config = exp['config']
    gpu_id = free_gpus[gpu_index]
    
    # Check if experiment already completed
    results_file = f"experiments/retrieval/{exp_name}/results.json"
    if os.path.exists(results_file):
        print(f"⏭️  Skipping {exp_name} (already completed)")
        continue
    
    # Check if config exists
    if not os.path.exists(config):
        print(f"⚠️  Config not found: {config}, skipping {exp_name}")
        continue
    
    # Create log file
    log_file = f"experiments/retrieval/{exp_name}/training.log"
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    # Build command
    cmd = [
        "python", "train_hybrid_learned.py",
        "--config", config,
        "--gpu_id", str(gpu_id)
    ]
    
    print(f"🚀 Resuming {exp_name} on GPU {gpu_id}...")
    print(f"   Command: {' '.join(cmd)}")
    
    # Start process in background
    try:
        with open(log_file, "a") as log:
            log.write(f"\n{'='*60}\n")
            log.write(f"Resumed at: {os.popen('date').read().strip()}\n")
            log.write(f"GPU: {gpu_id}\n")
            log.write(f"{'='*60}\n\n")
            
            process = subprocess.Popen(
                cmd,
                stdout=log,
                stderr=subprocess.STDOUT,
                cwd=os.getcwd(),
                preexec_fn=os.setsid
            )
        print(f"   ✅ Started with PID: {process.pid}\n")
        resumed_count += 1
        gpu_index += 1
    except Exception as e:
        print(f"   ❌ Failed to start: {e}\n")

print(f"\n✅ Resumed {resumed_count} hybrid experiments on free GPUs")
print(f"📊 Free GPUs used: {free_gpus[:gpu_index]}")
print(f"\nMonitor with: tail -f experiments/retrieval/phase5_hybrid_*/training.log")
PYTHON_SCRIPT

