#!/bin/bash
# Start Tier 1 Experiments
# This script starts all Tier 1 experiments in parallel with email notifications

cd /home/prevanka/prati/su-mt-rag/mt-rag-benchmark

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Set email password if provided (for email notifications)
# You can set this via: export EMAIL_PASSWORD="your-app-password"
# For Gmail, you need to use an App Password, not your regular password

# Run the Tier 1 experiments master script
# This will:
# 1. Start all experiments in parallel on available GPUs
# 2. Monitor for failures and send email notifications
# 3. Support resume from checkpoints
# 4. Run in background with nohup

nohup python run_tier1_experiments.py --max-parallel 6 > tier1_experiments.log 2>&1 &

echo "Tier 1 experiments started in background!"
echo "PID: $!"
echo "Log file: tier1_experiments.log"
echo ""
echo "To monitor progress:"
echo "  tail -f tier1_experiments.log"
echo ""
echo "To check status:"
echo "  python run_tier1_experiments.py --status"
echo ""
echo "To stop all experiments:"
echo "  pkill -f run_tier1_experiments.py"

