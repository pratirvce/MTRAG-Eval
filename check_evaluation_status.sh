#!/bin/bash
# Script to check evaluation status and show results when complete

cd /home/prevanka/prati/su-mt-rag/mt-rag-benchmark

echo "Checking evaluation status..."
echo "================================"

# Check if process is running
if ps aux | grep -E "python.*evaluate_finetuned_bge" | grep -v grep > /dev/null; then
    echo "✅ Evaluation is still running..."
    echo ""
    echo "Latest progress:"
    tail -5 evaluation.log 2>/dev/null | grep -E "Batches:|Processing Domain|Results for" || tail -3 evaluation.log
    echo ""
    echo "To monitor in real-time, run: tail -f evaluation.log"
else
    echo "✅ Evaluation appears to be complete!"
    echo ""
    echo "================================"
    echo "FINAL RESULTS:"
    echo "================================"
    tail -50 evaluation.log | grep -A 100 "Finished All Domains\|Results for\|Your Fine-Tuned\|Individual Domain" || tail -30 evaluation.log
fi

