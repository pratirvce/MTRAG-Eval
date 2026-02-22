#!/bin/bash
# Start experiment monitoring in background

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Source virtual environment if it exists
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
fi

# Start monitoring in background
nohup python monitor_experiments.py --check-interval 60 > experiment_monitor.log 2>&1 &
MONITOR_PID=$!

echo "✅ Experiment monitor started"
echo "   PID: $MONITOR_PID"
echo "   Log: experiment_monitor.log"
echo "   Status file: experiment_status.json"
echo "   Failure notifications: experiment_failures.json"
echo ""
echo "To stop monitoring:"
echo "   kill $MONITOR_PID"
echo ""
echo "To check status:"
echo "   python monitor_experiments.py --once"
echo ""
echo "To view monitor log:"
echo "   tail -f experiment_monitor.log"

# Save PID for later
echo $MONITOR_PID > experiment_monitor.pid

