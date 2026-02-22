#!/bin/bash
# Start all experiments in background

cd "$(dirname "$0")"

# Activate virtual environment
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Create necessary directories
mkdir -p experiments/logs
mkdir -p models

# Start experiments
echo "Starting experiments..."
echo "This will run all phases sequentially in the background."
echo ""

nohup python run_experiments.py > experiments/logs/experiments_$(date +%Y%m%d_%H%M%S).log 2>&1 &

EXPERIMENT_PID=$!
echo "Experiments started with PID: $EXPERIMENT_PID"
echo $EXPERIMENT_PID > experiments/logs/current_experiment.pid

echo ""
echo "To monitor: tail -f experiments/logs/experiments_*.log"
echo "To check status: ps -p $EXPERIMENT_PID"
echo "To stop: kill $EXPERIMENT_PID"
echo ""

