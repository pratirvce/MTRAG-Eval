#!/bin/bash

# Script to start pending experiments on free GPUs
# Automatically detects free GPUs and starts pending experiments

cd /home/prevanka/prati/su-mt-rag/mt-rag-benchmark
source venv/bin/activate

echo "=== Starting Pending Experiments on Free GPUs ==="
echo ""

# Function to check if GPU is free (memory < 5% and utilization < 20%)
check_gpu_free() {
    local gpu_id=$1
    local mem_info=$(nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits | awk -F', ' -v gpu=$gpu_id '$1==gpu {print $2, $3, $4}')
    if [ -z "$mem_info" ]; then
        return 1
    fi
    local mem_used=$(echo $mem_info | awk '{print $1}')
    local mem_total=$(echo $mem_info | awk '{print $2}')
    local util=$(echo $mem_info | awk '{print $3}')
    local mem_percent=$(echo "scale=2; $mem_used * 100 / $mem_total" | bc)
    
    if (( $(echo "$mem_percent < 5" | bc -l) )) && (( $(echo "$util < 20" | bc -l) )); then
        return 0
    else
        return 1
    fi
}

# Function to check if experiment is already running
is_experiment_running() {
    local exp_name=$1
    ps aux | grep -E "$exp_name" | grep -v grep | grep python > /dev/null
    return $?
}

# Function to check if experiment is complete
is_experiment_complete() {
    local config_path=$1
    local exp_dir=$(dirname "$config_path")
    [ -f "$exp_dir/results.json" ]
    return $?
}

# List of pending experiments with priorities
declare -a experiments=(
    "phase6_cross_encoder_finetuned_ensemble:train_cross_encoder_finetuned.py:experiments/retrieval/phase6_cross_encoder_finetuned_ensemble/config.json:1"
    "phase6_multistage_2stage:train_multistage_retrieval.py:experiments/retrieval/phase6_multistage_2stage/config.json:2"
)

# Find free GPUs
free_gpus=()
for gpu_id in 0 1 2 3 4 5; do
    if check_gpu_free $gpu_id; then
        free_gpus+=($gpu_id)
        echo "✅ GPU $gpu_id is free"
    fi
done

if [ ${#free_gpus[@]} -eq 0 ]; then
    echo "❌ No free GPUs available"
    exit 0
fi

echo ""
echo "Free GPUs: ${free_gpus[*]}"
echo ""

# Start experiments on free GPUs
started=0
gpu_idx=0

for exp_info in "${experiments[@]}"; do
    IFS=':' read -r exp_name script_name config_path priority <<< "$exp_info"
    
    # Check if experiment is already running
    if is_experiment_running "$exp_name"; then
        echo "⏭️  Skipping $exp_name (already running)"
        continue
    fi
    
    # Check if experiment is complete
    if is_experiment_complete "$config_path"; then
        echo "✅ Skipping $exp_name (already complete)"
        continue
    fi
    
    # Check if we have free GPUs
    if [ $gpu_idx -ge ${#free_gpus[@]} ]; then
        echo "⚠️  No more free GPUs. Remaining experiments will wait."
        break
    fi
    
    gpu_id=${free_gpus[$gpu_idx]}
    
    # Check if config exists
    if [ ! -f "$config_path" ]; then
        echo "⚠️  Config not found: $config_path. Skipping."
        continue
    fi
    
    log_file="$(dirname $config_path)/training.log"
    
    echo "🚀 Starting $exp_name on GPU $gpu_id..."
    
    # Kill any existing process for this experiment
    pids=$(ps aux | grep "$script_name.*$config_path" | grep -v grep | awk '{print $2}')
    for pid in $pids; do
        echo "  Killing existing process $pid"
        kill $pid 2>/dev/null
    done
    
    # Start the experiment
    nohup python "$script_name" --config "$config_path" --gpu_id "$gpu_id" --resume > "$log_file" 2>&1 &
    new_pid=$!
    
    echo "  ✅ Started with PID: $new_pid on GPU $gpu_id"
    echo "  📝 Log: $log_file"
    echo ""
    
    started=$((started + 1))
    gpu_idx=$((gpu_idx + 1))
    
    # Small delay between starts
    sleep 2
done

echo ""
echo "✅ Started $started experiment(s)"
echo ""
echo "To monitor:"
echo "  watch -n 1 nvidia-smi"
echo "  tail -f experiments/retrieval/phase6_*/training.log"
