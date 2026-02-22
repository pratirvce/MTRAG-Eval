#!/bin/bash
# Script to stop the duplicate original process (PID 1311837)

echo "═══════════════════════════════════════════════════════════════"
echo "        Stopping Duplicate Process: PID 1311837"
echo "═══════════════════════════════════════════════════════════════"
echo ""

# Check if process exists
if ! ps -p 1311837 > /dev/null 2>&1; then
    echo "❌ Process 1311837 is not running"
    exit 1
fi

echo "✅ Process found. Details:"
ps -p 1311837 -o pid,etime,cmd --no-headers
echo ""

# Confirm
read -p "Stop this process? (yes/no): " confirm
if [[ ! "$confirm" =~ ^[Yy][Ee][Ss]$ ]]; then
    echo "❌ Cancelled"
    exit 0
fi

echo ""
echo "🛑 Stopping process gracefully..."
kill -SIGTERM 1311837

echo "⏳ Waiting 30 seconds for cleanup..."
sleep 30

# Check if still running
if ps -p 1311837 > /dev/null 2>&1; then
    echo "⚠️  Process still running. Force stopping..."
    kill -9 1311837
    sleep 2
fi

# Verify stopped
if ps -p 1311837 > /dev/null 2>&1; then
    echo "❌ Failed to stop process"
    exit 1
else
    echo "✅ Process stopped successfully"
fi

echo ""
echo "📊 GPU Status:"
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader | head -1

echo ""
echo "✅ Done! GPU 0 should now be free."
