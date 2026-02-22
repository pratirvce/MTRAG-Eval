#!/bin/bash
# Pause Tier 2 experiments (hybrid) and start all Tier 1 experiments

echo "=========================================="
echo "Pausing Tier 2 Experiments (Hybrid)"
echo "=========================================="

# Find and pause hybrid experiments
HYBRID_PIDS=$(ps aux | grep "train_hybrid_learned.py" | grep -v grep | awk '{print $2}')

if [ -z "$HYBRID_PIDS" ]; then
    echo "No hybrid experiments running"
else
    echo "Found hybrid experiments: $HYBRID_PIDS"
    
    # Save experiment info for later resume
    RESUME_FILE="tier2_experiments_paused.json"
    echo "{" > "$RESUME_FILE"
    echo "  \"paused_at\": \"$(date -Iseconds)\"," >> "$RESUME_FILE"
    echo "  \"experiments\": [" >> "$RESUME_FILE"
    
    FIRST=true
    for PID in $HYBRID_PIDS; do
        CMD=$(ps -p $PID -o cmd= 2>/dev/null)
        if [ -z "$CMD" ]; then
            continue
        fi
        
        CONFIG=$(echo "$CMD" | grep -oP '--config \K[^\s]+' || echo "")
        EXP_NAME=$(echo "$CONFIG" | sed 's|.*/phase5_hybrid_\([^/]*\)/.*|\1|' || echo "")
        GPU_ID=$(echo "$CMD" | grep -oP '--gpu_id \K[0-9]+' || echo "")
        
        if [ -n "$CONFIG" ] && [ -n "$EXP_NAME" ]; then
            if [ "$FIRST" = false ]; then
                echo "," >> "$RESUME_FILE"
            fi
            echo "    {" >> "$RESUME_FILE"
            echo "      \"pid\": $PID," >> "$RESUME_FILE"
            echo "      \"exp_name\": \"phase5_hybrid_$EXP_NAME\"," >> "$RESUME_FILE"
            echo "      \"config\": \"$CONFIG\"," >> "$RESUME_FILE"
            echo "      \"gpu_id\": $GPU_ID," >> "$RESUME_FILE"
            echo "      \"cmd\": \"$CMD\"" >> "$RESUME_FILE"
            echo "    }" >> "$RESUME_FILE"
            FIRST=false
            
            echo "  Pausing phase5_hybrid_$EXP_NAME (PID: $PID, GPU: $GPU_ID)..."
            kill -TERM $PID 2>/dev/null
        fi
    done
    
    echo "  ]" >> "$RESUME_FILE"
    echo "}" >> "$RESUME_FILE"
    
    # Wait for graceful shutdown
    echo "Waiting for graceful shutdown..."
    sleep 5
    
    # Force kill any remaining
    for PID in $HYBRID_PIDS; do
        if kill -0 $PID 2>/dev/null; then
            echo "  Force killing PID $PID..."
            kill -9 $PID 2>/dev/null
        fi
    done
    
    echo "✅ Tier 2 experiments paused"
    echo "📝 Resume info saved to: $RESUME_FILE"
fi

echo ""
echo "=========================================="
echo "Starting Tier 1 Experiments"
echo "=========================================="

# Source virtual environment
source venv/bin/activate 2>/dev/null || echo "⚠️  Virtual environment not found"

# Check which Tier 1 experiments are running
CROSS_ENCODER_RUNNING=$(ps aux | grep "train_cross_encoder_finetuned.py" | grep -v grep | grep python | wc -l)
MULTISTAGE_RUNNING=$(ps aux | grep "train_multistage_retrieval.py" | grep -v grep | grep python | wc -l)
LLM_EXPANSION_RUNNING=$(ps aux | grep "train_llm_query_expansion.py" | grep -v grep | grep python | wc -l)

# Get available GPUs
AVAILABLE_GPUS=()
for gpu in 0 1 2 3 4 5; do
    # Check if GPU is free (low utilization)
    UTIL=$(nvidia-smi -i $gpu --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | tr -d ' ')
    if [ -n "$UTIL" ] && [ "$UTIL" -lt 20 ]; then
        AVAILABLE_GPUS+=($gpu)
    fi
done

echo "Available GPUs: ${AVAILABLE_GPUS[@]}"
echo ""

# 1. Cross-encoder fine-tuning (Priority #1)
if [ "$CROSS_ENCODER_RUNNING" -eq 0 ]; then
    if [ ${#AVAILABLE_GPUS[@]} -gt 0 ]; then
        GPU_ID=${AVAILABLE_GPUS[0]}
        echo "🚀 Starting cross-encoder fine-tuning on GPU $GPU_ID..."
        nohup python train_cross_encoder_finetuned.py \
            --config experiments/retrieval/phase6_cross_encoder_finetuned_ensemble/config.json \
            --gpu_id $GPU_ID \
            --resume > experiments/retrieval/phase6_cross_encoder_finetuned_ensemble/training.log 2>&1 &
        echo "   ✅ Started with PID: $!"
        AVAILABLE_GPUS=("${AVAILABLE_GPUS[@]:1}")  # Remove used GPU
    else
        echo "⚠️  No available GPU for cross-encoder"
    fi
else
    echo "✅ Cross-encoder already running"
fi

# 2. Multi-stage retrieval (Priority #2)
if [ "$MULTISTAGE_RUNNING" -eq 0 ]; then
    if [ ${#AVAILABLE_GPUS[@]} -gt 0 ]; then
        GPU_ID=${AVAILABLE_GPUS[0]}
        echo "🚀 Starting multi-stage retrieval on GPU $GPU_ID..."
        nohup python train_multistage_retrieval.py \
            --config experiments/retrieval/phase6_multistage_2stage/config.json \
            --gpu_id $GPU_ID \
            --resume > experiments/retrieval/phase6_multistage_2stage/training.log 2>&1 &
        echo "   ✅ Started with PID: $!"
        AVAILABLE_GPUS=("${AVAILABLE_GPUS[@]:1}")
    else
        echo "⚠️  No available GPU for multi-stage"
    fi
else
    echo "✅ Multi-stage already running"
fi

# 3. LLM query expansion (Priority #3)
if [ "$LLM_EXPANSION_RUNNING" -eq 0 ]; then
    if [ ${#AVAILABLE_GPUS[@]} -gt 0 ]; then
        GPU_ID=${AVAILABLE_GPUS[0]}
        echo "🚀 Starting LLM query expansion on GPU $GPU_ID..."
        nohup python train_llm_query_expansion.py \
            --config experiments/retrieval/phase6_llm_query_expansion_gpt4_multi/config.json \
            --gpu_id $GPU_ID \
            --resume > experiments/retrieval/phase6_llm_query_expansion_gpt4_multi/training.log 2>&1 &
        echo "   ✅ Started with PID: $!"
        AVAILABLE_GPUS=("${AVAILABLE_GPUS[@]:1}")
    else
        echo "⚠️  No available GPU for LLM expansion"
    fi
else
    echo "✅ LLM expansion already running"
fi

echo ""
echo "=========================================="
echo "Summary"
echo "=========================================="
echo "Tier 2 (Hybrid): Paused"
echo "Tier 1 (Priority):"
echo "  - Cross-encoder: $([ "$CROSS_ENCODER_RUNNING" -gt 0 ] && echo "✅ Running" || echo "🚀 Starting")"
echo "  - Multi-stage: $([ "$MULTISTAGE_RUNNING" -gt 0 ] && echo "✅ Running" || echo "🚀 Starting")"
echo "  - LLM expansion: $([ "$LLM_EXPANSION_RUNNING" -gt 0 ] && echo "✅ Running" || echo "🚀 Starting")"
echo ""
echo "Monitor with:"
echo "  tail -f experiments/retrieval/phase6_*/training.log"
echo "  nvidia-smi"

