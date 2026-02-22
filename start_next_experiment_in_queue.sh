#!/bin/bash
# Script to automatically start the next experiment in the queue when a GPU becomes available
# This script monitors GPU availability and starts experiments in priority order

cd /home/prevanka/prati/su-mt-rag/mt-rag-benchmark

# Activate virtual environment
source venv/bin/activate

echo "=========================================="
echo "Experiment Queue Manager"
echo "=========================================="
echo "Checking for available GPUs and next experiments in queue..."
echo ""

# Function to check if a GPU is free (memory usage < 500 MB and utilization < 10%)
check_gpu_free() {
    local gpu_id=$1
    local mem_used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits --id=$gpu_id)
    local utilization=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits --id=$gpu_id)
    
    if [ "$mem_used" -lt 500 ] && [ "$utilization" -lt 10 ]; then
        return 0  # GPU is free
    else
        return 1  # GPU is in use
    fi
}

# Function to check if an experiment is already running
is_experiment_running() {
    local exp_name=$1
    if ps aux | grep -E "train_.*$exp_name" | grep -v grep > /dev/null; then
        return 0  # Experiment is running
    else
        return 1  # Experiment is not running
    fi
}

# Check all GPUs (0-5)
available_gpus=()
for gpu_id in {0..5}; do
    if check_gpu_free $gpu_id; then
        available_gpus+=($gpu_id)
        echo "✅ GPU $gpu_id is FREE (available for new experiments)"
    else
        echo "⏳ GPU $gpu_id is IN USE"
    fi
done

echo ""
echo "Available GPUs: ${available_gpus[@]}"
echo ""

# Priority queue of experiments (in order)
# Format: "experiment_name|script_name|config_path|expected_time"

declare -a QUEUE=(
    "hierarchical_multigranularity|train_hierarchical_multigranularity_retrieval.py|experiments/retrieval/phase8_hierarchical_multigranularity/config.json|12-16 hours"
    "contrastive_learning|train_contrastive_conversation_doc.py|experiments/retrieval/phase8_contrastive_learning/config.json|12-16 hours"
)

# Check queue and start experiments
gpu_idx=0
for queue_item in "${QUEUE[@]}"; do
    IFS='|' read -r exp_name script_name config_path expected_time <<< "$queue_item"
    
    # Check if experiment is already running
    if is_experiment_running "$exp_name"; then
        echo "⏸️  $exp_name is already running. Skipping."
        continue
    fi
    
    # Check if config file exists (experiment is implemented)
    if [ ! -f "$config_path" ]; then
        echo "⚠️  $exp_name is not yet implemented (config not found: $config_path)"
        echo "   Status: Needs implementation before running"
        continue
    fi
    
    # Check if we have available GPUs
    if [ $gpu_idx -ge ${#available_gpus[@]} ]; then
        echo "⏸️  No more available GPUs. Remaining experiments will wait."
        break
    fi
    
    gpu_id=${available_gpus[$gpu_idx]}
    
    echo ""
    echo "🚀 Starting: $exp_name"
    echo "   Script: $script_name"
    echo "   Config: $config_path"
    echo "   GPU: $gpu_id"
    echo "   Expected Time: $expected_time"
    
    # Check if script exists
    if [ ! -f "$script_name" ]; then
        echo "   ❌ Script not found: $script_name"
        echo "   Status: Needs implementation"
        continue
    fi
    
    # Create log directory if it doesn't exist
    log_dir=$(dirname "$config_path")
    mkdir -p "$log_dir"
    log_file="$log_dir/training.log"
    
    # Start experiment in background
    echo "   Starting experiment in background..."
    nohup python "$script_name" --config "$config_path" --gpu_id "$gpu_id" > "$log_file" 2>&1 &
    new_pid=$!
    
    echo "   ✅ Started with PID: $new_pid"
    echo "   📝 Log file: $log_file"
    echo "   Monitor with: tail -f $log_file"
    
    gpu_idx=$((gpu_idx + 1))
    
    # Wait a bit before starting next experiment
    sleep 5
done

echo ""
echo "=========================================="
echo "Queue Processing Complete"
echo "=========================================="
echo ""
echo "To monitor all running experiments:"
echo "  ps aux | grep 'python train_' | grep -v grep"
echo ""
echo "To check GPU usage:"
echo "  nvidia-smi"
echo ""
echo "To view experiment queue:"
echo "  cat EXPERIMENT_QUEUE.md"
echo ""

