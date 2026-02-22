#!/bin/bash
# Script to stop old experiment versions after fixed versions complete
# Queue this to run after fixed experiments complete

echo "Stopping old experiment versions..."

# Old versions to stop (only if fixed versions have completed successfully)
OLD_EXPERIMENTS=(
    "best_paper_temporal_memory"  # Old version, fixed: best_paper_temporal_memory_fixed
    "best_paper_rl_adaptive_retrieval"  # Old version, fixed: best_paper_rl_adaptive_retrieval_fixed
)

for exp_name in "${OLD_EXPERIMENTS[@]}"; do
    # Check if fixed version exists and has results
    if [ "$exp_name" = "best_paper_temporal_memory" ]; then
        fixed_name="best_paper_temporal_memory_fixed"
    elif [ "$exp_name" = "best_paper_rl_adaptive_retrieval" ]; then
        fixed_name="best_paper_rl_adaptive_retrieval_fixed"
    fi
    
    # Check if fixed version has completed
    if [ -f "experiments/retrieval/$fixed_name/results.json" ]; then
        echo "✅ Fixed version $fixed_name completed, stopping old version $exp_name..."
        
        # Find and kill the process
        pids=$(pgrep -f "train_.*_tier1.py.*$exp_name" | grep -v "$fixed_name")
        if [ -n "$pids" ]; then
            for pid in $pids; do
                echo "  Stopping PID $pid..."
                kill $pid 2>/dev/null
            done
        else
            echo "  No running process found for $exp_name"
        fi
    else
        echo "⏳ Fixed version $fixed_name not completed yet, keeping old version running"
    fi
done

echo "Done!"
