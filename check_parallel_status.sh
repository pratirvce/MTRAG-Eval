#!/bin/bash
# Check status of parallel experiments

cd "$(dirname "$0")"

echo "=== Parallel Experiments Status ==="
echo ""

# Check parallel status file
if [ -f "experiments/retrieval/phase4_parallel_status.json" ]; then
    python3 run_phase4_parallel.py --status 2>/dev/null
else
    echo "No parallel experiments running yet."
fi

echo ""
echo "=== GPU Usage ==="
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader | \
    awk -F', ' '{printf "GPU %s: %s%% util, %s / %s MB\n", $1, $2, $3, $4}'

echo ""
echo "=== Running Processes ==="
if [ -f "experiments/retrieval/phase4_parallel_runner.pid" ]; then
    PID=$(cat experiments/retrieval/phase4_parallel_runner.pid)
    if ps -p $PID > /dev/null 2>&1; then
        echo "✅ Parallel runner is running (PID: $PID)"
        ps -p $PID -o pid,etime,cmd
    else
        echo "❌ Parallel runner is not running (stale PID file)"
    fi
else
    echo "No parallel runner PID file found"
fi

echo ""
echo "=== Active Training Processes ==="
ps aux | grep -E "train_advanced_bge|train_domain_specific_bge|train_improved_bge" | grep -v grep | \
    awk '{printf "PID %s: %s %s %s %s\n", $2, $11, $12, $13, $14}'


