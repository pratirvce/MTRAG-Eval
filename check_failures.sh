#!/bin/bash
# Quick script to check for experiment failures and print notification

cd "$(dirname "$0")"

# Run one-time check
python monitor_experiments.py --once

# Check if failures file exists and has failures
if [ -f "experiment_failures.json" ]; then
    FAILED_COUNT=$(python3 -c "import json; f=open('experiment_failures.json'); d=json.load(f); print(d.get('total_failed', 0))" 2>/dev/null || echo "0")
    
    if [ "$FAILED_COUNT" -gt 0 ]; then
        echo ""
        echo "⚠️  $FAILED_COUNT experiment(s) failed! See experiment_failures.json for details"
        exit 1
    fi
fi

exit 0

