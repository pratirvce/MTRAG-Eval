#!/bin/bash
# Script to start the auto-runner in the background

cd "$(dirname "$0")"

# Check if already running
if pgrep -f "auto_start_fixed_experiments.py" > /dev/null; then
    echo "⚠️  Auto-runner is already running"
    ps aux | grep "auto_start_fixed_experiments.py" | grep -v grep
    exit 1
fi

# Start in background with nohup
echo "🚀 Starting auto-runner in background..."
nohup python3 auto_start_fixed_experiments.py > auto_runner.log 2>&1 &
PID=$!

echo "✅ Auto-runner started (PID: $PID)"
echo "📝 Logs: auto_runner.log"
echo "📊 Status file: auto_fixed_experiments_status.json"
echo ""
echo "To stop: kill $PID"
echo "To view logs: tail -f auto_runner.log"

