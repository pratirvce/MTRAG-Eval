#!/bin/bash
# Script to pause hybrid experiments gracefully

echo "Pausing hybrid experiments..."

# Find all hybrid experiment processes
PIDS=$(ps aux | grep "train_hybrid_learned.py" | grep -v grep | awk '{print $2}')

if [ -z "$PIDS" ]; then
    echo "No hybrid experiments running"
    exit 0
fi

# Save process info for later resume
RESUME_FILE="hybrid_experiments_paused.json"
echo "{" > "$RESUME_FILE"
echo "  \"paused_at\": \"$(date -Iseconds)\"," >> "$RESUME_FILE"
echo "  \"experiments\": [" >> "$RESUME_FILE"

FIRST=true
for PID in $PIDS; do
    # Get process details
    CMD=$(ps -p $PID -o cmd= 2>/dev/null)
    if [ -z "$CMD" ]; then
        continue
    fi
    
    # Extract config path from command
    CONFIG=$(echo "$CMD" | grep -oP '--config \K[^\s]+' || echo "")
    EXP_NAME=$(echo "$CONFIG" | grep -oP 'phase5_hybrid_\K[^/]+' || echo "")
    GPU_ID=$(echo "$CMD" | grep -oP '--gpu_id \K[0-9]+' || echo "")
    
    if [ -n "$CONFIG" ] && [ -n "$EXP_NAME" ]; then
        if [ "$FIRST" = false ]; then
            echo "," >> "$RESUME_FILE"
        fi
        echo "    {" >> "$RESUME_FILE"
        echo "      \"pid\": $PID," >> "$RESUME_FILE"
        echo "      \"exp_name\": \"$EXP_NAME\"," >> "$RESUME_FILE"
        echo "      \"config\": \"$CONFIG\"," >> "$RESUME_FILE"
        echo "      \"gpu_id\": $GPU_ID," >> "$RESUME_FILE"
        echo "      \"cmd\": \"$CMD\"" >> "$RESUME_FILE"
        echo "    }" >> "$RESUME_FILE"
        FIRST=false
        
        echo "  Stopping $EXP_NAME (PID: $PID, GPU: $GPU_ID)..."
        # Send SIGTERM for graceful shutdown
        kill -TERM $PID 2>/dev/null
    fi
done

echo "  ]" >> "$RESUME_FILE"
echo "}" >> "$RESUME_FILE"

# Wait for processes to terminate gracefully
echo "Waiting for graceful shutdown..."
sleep 5

# Force kill any remaining processes
for PID in $PIDS; do
    if kill -0 $PID 2>/dev/null; then
        echo "  Force killing PID $PID..."
        kill -9 $PID 2>/dev/null
    fi
done

echo "✅ Hybrid experiments paused"
echo "📝 Resume info saved to: $RESUME_FILE"
echo ""
echo "To resume later, run: ./resume_hybrid_experiments.sh"

