#!/bin/bash
# Background experiment runner
# This script runs all experiments in the background and logs everything

cd "$(dirname "$0")"

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Create logs directory
mkdir -p experiments/logs

# Log file with timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="experiments/logs/experiments_${TIMESTAMP}.log"

echo "Starting experiments at $(date)" | tee -a "$LOG_FILE"
echo "Log file: $LOG_FILE"
echo ""

# Run experiments in background, redirecting all output
nohup python run_experiments.py >> "$LOG_FILE" 2>&1 &

# Get PID
EXPERIMENT_PID=$!
echo "Experiments started with PID: $EXPERIMENT_PID"
echo "PID: $EXPERIMENT_PID" >> "$LOG_FILE"
echo ""

# Save PID to file for easy tracking
echo $EXPERIMENT_PID > experiments/logs/current_experiment.pid

echo "To monitor progress:"
echo "  tail -f $LOG_FILE"
echo ""
echo "To check if still running:"
echo "  ps -p $EXPERIMENT_PID"
echo ""
echo "To stop experiments:"
echo "  kill $EXPERIMENT_PID"
echo ""

# Show initial log
echo "--- Initial log output ---"
tail -n 20 "$LOG_FILE"

