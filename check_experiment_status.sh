#!/bin/bash
# Check status of running experiments

cd "$(dirname "$0")"

PID_FILE="experiments/logs/current_experiment.pid"

if [ ! -f "$PID_FILE" ]; then
    echo "No experiment PID file found. Experiments may not be running."
    exit 1
fi

PID=$(cat "$PID_FILE")

if ps -p $PID > /dev/null 2>&1; then
    echo "✅ Experiments are running (PID: $PID)"
    echo ""
    echo "Latest log entries:"
    echo "---"
    tail -n 20 experiments/logs/experiments_*.log 2>/dev/null | tail -n 10
    echo "---"
    echo ""
    echo "To see full log: tail -f experiments/logs/experiments_*.log"
else
    echo "❌ Experiments are not running (PID: $PID not found)"
    echo ""
    echo "Checking for completed results..."
    
    if [ -f "experiments/retrieval/experiment_summary.json" ]; then
        echo "✅ Found results summary!"
        python summarize_results.py
    else
        echo "No results found yet."
    fi
fi

