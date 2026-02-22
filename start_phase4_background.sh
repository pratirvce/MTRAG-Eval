#!/bin/bash
# Start Phase 4 experiments in the background

cd "$(dirname "$0")"

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Create necessary directories
mkdir -p experiments/retrieval/logs

echo "Starting Phase 4 experiments in the background..."
echo "Logs will be saved to: experiments/retrieval/logs/"
echo "Status will be tracked in: experiments/retrieval/phase4_status.json"
echo ""
echo "To check status, run: ./check_phase4_status.sh"
echo "Or: python run_phase4_background.py --status"
echo ""

# Run in background with nohup
nohup python run_phase4_background.py --experiment all > experiments/retrieval/logs/phase4_runner_$(date +%Y%m%d_%H%M%S).out 2>&1 &

PID=$!
echo "Background process started (PID: $PID)"
echo "PID saved to: experiments/retrieval/phase4_runner.pid"

# Wait a moment and check if it's still running
sleep 2
if ps -p $PID > /dev/null 2>&1; then
    echo "✅ Experiments are running!"
    echo ""
    echo "Monitor progress with:"
    echo "  ./check_phase4_status.sh"
    echo "  tail -f experiments/retrieval/logs/phase4_background_*.log"
    echo ""
    echo "To stop, run:"
    echo "  kill $PID"
else
    echo "❌ Process failed to start. Check logs for errors."
fi

