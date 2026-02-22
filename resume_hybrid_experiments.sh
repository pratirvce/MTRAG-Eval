#!/bin/bash
# Script to resume paused hybrid experiments

RESUME_FILE="hybrid_experiments_paused.json"

if [ ! -f "$RESUME_FILE" ]; then
    echo "❌ No paused experiments found ($RESUME_FILE not found)"
    exit 1
fi

echo "Resuming hybrid experiments from $RESUME_FILE..."

# Source virtual environment
source venv/bin/activate 2>/dev/null || echo "⚠️  Virtual environment not found, continuing..."

# Parse JSON and restart experiments
python3 << 'PYTHON_SCRIPT'
import json
import subprocess
import os
import sys

with open("hybrid_experiments_paused.json", "r") as f:
    data = json.load(f)

print(f"Found {len(data['experiments'])} paused experiments")
print(f"Paused at: {data['paused_at']}\n")

for exp in data['experiments']:
    exp_name = exp['exp_name']
    config = exp['config']
    gpu_id = exp['gpu_id']
    
    # Check if experiment already completed
    results_file = f"experiments/retrieval/{exp_name}/results.json"
    if os.path.exists(results_file):
        print(f"⏭️  Skipping {exp_name} (already completed)")
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
            process = subprocess.Popen(
                cmd,
                stdout=log,
                stderr=subprocess.STDOUT,
                cwd=os.getcwd(),
                preexec_fn=os.setsid
            )
        print(f"   ✅ Started with PID: {process.pid}\n")
    except Exception as e:
        print(f"   ❌ Failed to start: {e}\n")

print("\n✅ All experiments resumed")
print("Monitor with: tail -f experiments/retrieval/phase5_hybrid_*/training.log")
PYTHON_SCRIPT

