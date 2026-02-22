#!/bin/bash
# Start the auto fixed experiments runner in the background

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Create log directory if it doesn't exist
mkdir -p logs

# Start the script in the background
nohup python3 auto_start_fixed_experiments.py > logs/auto_fixed_experiments.log 2>&1 &

# Get the PID
PID=$!

# Save PID to file
echo $PID > logs/auto_fixed_experiments.pid

echo "Auto Fixed Experiments Runner started with PID: $PID"
echo "Log file: logs/auto_fixed_experiments.log"
echo "PID file: logs/auto_fixed_experiments.pid"
echo ""
echo "To stop: kill \$(cat logs/auto_fixed_experiments.pid)"
echo "To view logs: tail -f logs/auto_fixed_experiments.log"

