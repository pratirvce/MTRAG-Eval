#!/bin/bash
# Start Phase 4 experiments in parallel across multiple GPUs

cd "$(dirname "$0")"

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Create necessary directories
mkdir -p experiments/retrieval/logs

echo "=== Starting Phase 4 Parallel Experiments ==="
echo ""
echo "Available GPUs will be automatically detected"
echo "Experiments will run in parallel across multiple GPUs"
echo ""

# Check GPU availability
python3 -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'GPU Count: {torch.cuda.device_count() if torch.cuda.is_available() else 0}')" 2>/dev/null

echo ""
echo "Starting parallel experiment runner..."
echo "Logs: experiments/retrieval/logs/phase4_parallel_*.log"
echo "Status: experiments/retrieval/phase4_parallel_status.json"
echo ""

# Run in background with nohup
nohup python run_phase4_parallel.py --experiment all --max_parallel 5 > experiments/retrieval/logs/parallel_runner_$(date +%Y%m%d_%H%M%S).out 2>&1 &

PID=$!
echo "Background process started (PID: $PID)"
echo "PID saved to: experiments/retrieval/phase4_parallel_runner.pid"
echo "$PID" > experiments/retrieval/phase4_parallel_runner.pid

sleep 2
if ps -p $PID > /dev/null 2>&1; then
    echo "✅ Parallel experiments are running!"
    echo ""
    echo "Monitor progress with:"
    echo "  python run_phase4_parallel.py --status"
    echo "  tail -f experiments/retrieval/logs/phase4_parallel_*.log"
    echo ""
    echo "To stop, run:"
    echo "  kill $PID"
else
    echo "❌ Process failed to start. Check logs for errors."
fi


