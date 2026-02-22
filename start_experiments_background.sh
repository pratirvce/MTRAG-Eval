#!/bin/bash
# Start experiments in background and set up monitoring

cd /home/prevanka/prati/su-mt-rag/mt-rag-benchmark

# Activate virtual environment
source venv/bin/activate

echo "=========================================="
echo "Starting Phase 5 Experiments"
echo "=========================================="

# Step 1: Setup experiments (if not already done)
if [ ! -f "experiment_status.json" ] || [ ! -s "experiment_status.json" ]; then
    echo "Setting up experiments..."
    python setup_phase5_experiments.py
fi

# Step 2: Start all pending experiments
echo "Starting experiments in parallel..."
python manage_experiments.py start

# Step 3: Start monitoring in background
echo "Starting experiment monitor..."
nohup python monitor_experiments.py --interval 300 > monitor.log 2>&1 &
MONITOR_PID=$!

echo "Monitor PID: $MONITOR_PID"
echo $MONITOR_PID > monitor.pid

echo "=========================================="
echo "Experiments started!"
echo "=========================================="
echo ""
echo "Monitor Status:"
echo "  - Monitor PID: $MONITOR_PID"
echo "  - Monitor log: monitor.log"
echo "  - Notifications: experiment_notifications.log"
echo ""
echo "Commands:"
echo "  Check status: python manage_experiments.py status"
echo "  View monitor log: tail -f monitor.log"
echo "  View notifications: tail -f experiment_notifications.log"
echo "  Stop monitor: kill \$(cat monitor.pid)"
echo "  Generate report: python monitor_experiments.py --report"
echo ""

