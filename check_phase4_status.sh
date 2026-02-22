#!/bin/bash
# Quick status checker for Phase 4 experiments

cd "$(dirname "$0")"

echo "Phase 4 Experiments Status"
echo "=========================="
echo ""

# Check if status file exists
if [ -f "experiments/retrieval/phase4_status.json" ]; then
    python3 -c "
import json
from datetime import datetime

with open('experiments/retrieval/phase4_status.json', 'r') as f:
    status = json.load(f)

print('Started:', status.get('started_at', 'N/A'))
print('Last Updated:', status.get('last_updated', 'N/A'))
print('Current:', status.get('current_experiment', 'None'))
print()
print('Experiments:')
print('-' * 80)

experiments = status.get('experiments', {})
for exp_name, exp_data in experiments.items():
    status_val = exp_data.get('status', 'unknown')
    started = exp_data.get('started_at', 'N/A')
    if started != 'N/A' and len(started) > 19:
        started = started[:19]
    
    icon = '✅' if status_val == 'completed' else '🔄' if status_val == 'running' else '❌' if status_val == 'failed' else '⏳'
    print(f'{icon} {exp_name:<40} | {status_val:<12} | {started}')
"
else
    echo "No status file found. Experiments may not have started yet."
fi

echo ""
echo "Recent logs:"
ls -lt experiments/retrieval/logs/*.log 2>/dev/null | head -5

echo ""
echo "Running processes:"
if [ -f "experiments/retrieval/phase4_runner.pid" ]; then
    PID=$(cat experiments/retrieval/phase4_runner.pid)
    if ps -p $PID > /dev/null 2>&1; then
        echo "✅ Background runner is running (PID: $PID)"
        ps -p $PID -o pid,etime,cmd
    else
        echo "❌ Background runner is not running (stale PID file)"
    fi
else
    echo "No PID file found"
fi

