#!/bin/bash

# Start experiment monitoring with email notifications
# Checks every 5 minutes (300 seconds) for failures

cd /home/prevanka/prati/su-mt-rag/mt-rag-benchmark
source venv/bin/activate

echo "Starting experiment monitoring with email notifications..."
echo "Email: pratirvce@gmail.com"
echo "Check interval: 5 minutes (300 seconds)"
echo ""

# Kill existing monitor if running
pids=$(ps aux | grep "monitor_experiments.py" | grep -v grep | awk '{print $2}')
for pid in $pids; do
    echo "Stopping existing monitor (PID: $pid)"
    kill $pid 2>/dev/null
done

sleep 2

# Start new monitor in background
nohup python monitor_experiments.py --check-interval 300 > monitor.log 2>&1 &
MONITOR_PID=$!

echo "✅ Monitor started with PID: $MONITOR_PID"
echo "📝 Log file: monitor.log"
echo ""
echo "To check status:"
echo "  tail -f monitor.log"
echo ""
echo "To stop monitoring:"
echo "  kill $MONITOR_PID"
echo ""
echo "Email notifications will be sent to: pratirvce@gmail.com"
echo "when experiments fail."

