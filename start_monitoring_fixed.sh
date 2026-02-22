#!/bin/bash
# Start automated monitoring for fixed cross-attention experiments
# This script runs the monitor in the background and logs output

cd "$(dirname "$0")"

MONITOR_LOG="fixed_experiments_monitor.log"
MONITOR_PID_FILE="fixed_experiments_monitor.pid"

# Check if already running
if [ -f "$MONITOR_PID_FILE" ]; then
    OLD_PID=$(cat "$MONITOR_PID_FILE")
    if ps -p $OLD_PID > /dev/null 2>&1; then
        echo "⚠️  Monitoring is already running (PID: $OLD_PID)"
        echo "   To stop: kill $OLD_PID"
        exit 1
    else
        echo "Removing stale PID file..."
        rm -f "$MONITOR_PID_FILE"
    fi
fi

# Start monitoring in background
echo "🚀 Starting automated monitoring..."
nohup python3 monitor_fixed_experiments.py > "$MONITOR_LOG" 2>&1 &
MONITOR_PID=$!

echo $MONITOR_PID > "$MONITOR_PID_FILE"
echo "✅ Monitoring started (PID: $MONITOR_PID)"
echo "   Log file: $MONITOR_LOG"
echo "   Status file: fixed_experiments_status.json"
echo "   Status report: fixed_experiments_status.txt"
echo ""
echo "To view logs: tail -f $MONITOR_LOG"
echo "To stop: kill $MONITOR_PID"
echo "To check status: python3 monitor_fixed_experiments.py --once"

