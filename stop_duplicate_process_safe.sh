#!/bin/bash
# Safe script to stop duplicate process with comprehensive checks

set -e  # Exit on error

ORIGINAL_PID="1311837"
PARALLEL_PID="1850801"
EXPERIMENT_NAME="phase4_hard_negatives_cosine"
LOG_DIR="experiments/retrieval/${EXPERIMENT_NAME}"
BACKUP_DIR="experiments/retrieval/${EXPERIMENT_NAME}/process_analysis_$(date +%Y%m%d_%H%M%S)"

echo "═══════════════════════════════════════════════════════════════"
echo "     SAFE DUPLICATE PROCESS STOPPER - PRE-FLIGHT CHECKS"
echo "═══════════════════════════════════════════════════════════════"
echo ""

# Function to log messages
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

# Function to check if process exists
check_process() {
    local pid=$1
    if ps -p "$pid" > /dev/null 2>&1; then
        return 0
    else
        return 1
    fi
}

# Function to get process info
get_process_info() {
    local pid=$1
    if check_process "$pid"; then
        ps -p "$pid" -o pid,etime,cmd --no-headers 2>/dev/null || echo "Process info unavailable"
    else
        echo "Process not found"
    fi
}

# Function to get GPU info for a PID
get_gpu_for_pid() {
    local pid=$1
    nvidia-smi --query-compute-apps=pid,gpu_uuid --format=csv,noheader,nounits 2>/dev/null | \
        grep "^$pid" | head -1 | awk -F', ' '{print $2}' || echo "unknown"
}

# Function to check checkpoints
check_checkpoints() {
    local exp_name=$1
    local checkpoint_count=0
    local checkpoint_paths=(
        "./models/${exp_name}-checkpoints"
        "${LOG_DIR}/models/${exp_name}-checkpoints"
    )
    
    for cp_path in "${checkpoint_paths[@]}"; do
        if [ -d "$cp_path" ]; then
            count=$(find "$cp_path" -type d -name "checkpoint-*" 2>/dev/null | wc -l)
            checkpoint_count=$((checkpoint_count + count))
        fi
    done
    
    echo "$checkpoint_count"
}

# Function to check if process is part of parallel system
check_parallel_system() {
    local pid=$1
    # Check if parent is run_phase4_parallel.py or if it's in the parallel status file
    local ppid=$(ps -p "$pid" -o ppid= 2>/dev/null | xargs)
    if [ -n "$ppid" ]; then
        local parent_cmd=$(ps -p "$ppid" -o cmd= 2>/dev/null || echo "")
        if echo "$parent_cmd" | grep -q "run_phase4_parallel"; then
            return 0
        fi
    fi
    return 1
}

# Function to analyze log files
analyze_logs() {
    local pid=$1
    local exp_name=$2
    
    echo "  Log Analysis:"
    
    # Check error log
    if [ -f "${LOG_DIR}/error.log" ]; then
        error_size=$(stat -f%z "${LOG_DIR}/error.log" 2>/dev/null || stat -c%s "${LOG_DIR}/error.log" 2>/dev/null || echo "0")
        if [ "$error_size" -gt 1000 ]; then
            echo "    ⚠️  Error log has content: ${error_size} bytes"
            echo "    📄 Last 3 error lines:"
            tail -3 "${LOG_DIR}/error.log" 2>/dev/null | sed 's/^/      /' || echo "      (could not read)"
        else
            echo "    ✅ Error log is small/empty"
        fi
    fi
    
    # Check training log
    if [ -f "${LOG_DIR}/training.log" ]; then
        log_size=$(stat -f%z "${LOG_DIR}/training.log" 2>/dev/null || stat -c%s "${LOG_DIR}/training.log" 2>/dev/null || echo "0")
        if [ "$log_size" -gt 0 ]; then
            echo "    📄 Training log size: ${log_size} bytes"
            echo "    📄 Last activity:"
            tail -2 "${LOG_DIR}/training.log" 2>/dev/null | grep -v "^$" | sed 's/^/      /' || echo "      (no recent activity)"
        else
            echo "    ⚠️  Training log is empty"
        fi
    fi
}

# ============================================================================
# PRE-FLIGHT CHECKS
# ============================================================================

log "Starting pre-flight checks..."

# Check 1: Verify both processes exist
echo ""
echo "CHECK 1: Process Existence"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if ! check_process "$ORIGINAL_PID"; then
    log "❌ ERROR: Original process ($ORIGINAL_PID) is not running!"
    exit 1
fi
log "✅ Original process ($ORIGINAL_PID) is running"

if ! check_process "$PARALLEL_PID"; then
    log "❌ ERROR: Parallel process ($PARALLEL_PID) is not running!"
    log "   Cannot stop original if parallel is not running."
    exit 1
fi
log "✅ Parallel process ($PARALLEL_PID) is running"

# Check 2: Verify they're the same experiment
echo ""
echo "CHECK 2: Experiment Verification"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

original_cmd=$(ps -p "$ORIGINAL_PID" -o cmd= 2>/dev/null)
parallel_cmd=$(ps -p "$PARALLEL_PID" -o cmd= 2>/dev/null)

if ! echo "$original_cmd" | grep -q "$EXPERIMENT_NAME"; then
    log "❌ ERROR: Original process is not running $EXPERIMENT_NAME!"
    exit 1
fi

if ! echo "$parallel_cmd" | grep -q "$EXPERIMENT_NAME"; then
    log "❌ ERROR: Parallel process is not running $EXPERIMENT_NAME!"
    exit 1
fi

log "✅ Both processes are running $EXPERIMENT_NAME"

# Check 3: Verify config files match
echo ""
echo "CHECK 3: Configuration Verification"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

config_file="${LOG_DIR}/config.json"
if [ ! -f "$config_file" ]; then
    log "⚠️  WARNING: Config file not found: $config_file"
else
    log "✅ Config file exists: $config_file"
    # Both processes use the same config file, so they're identical
fi

# Check 4: Checkpoint status
echo ""
echo "CHECK 4: Checkpoint Status"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

checkpoint_count=$(check_checkpoints "$EXPERIMENT_NAME")
if [ "$checkpoint_count" -gt 0 ]; then
    log "⚠️  WARNING: Found $checkpoint_count checkpoint(s)! Progress exists."
    log "   Stopping will lose progress after latest checkpoint."
    read -p "   Continue anyway? (yes/no): " checkpoint_confirm
    if [[ ! "$checkpoint_confirm" =~ ^[Yy][Ee][Ss]$ ]]; then
        log "❌ Cancelled by user (checkpoints exist)"
        exit 0
    fi
else
    log "✅ No checkpoints found (still in mining phase)"
    log "   ⚠️  All mining progress will be lost (expected)"
fi

# Check 5: Parallel system membership
echo ""
echo "CHECK 5: Process Management Status"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if check_parallel_system "$PARALLEL_PID"; then
    log "✅ Parallel process is part of managed system"
else
    log "⚠️  WARNING: Parallel process may not be part of managed system"
fi

if check_parallel_system "$ORIGINAL_PID"; then
    log "⚠️  WARNING: Original process appears to be part of managed system too"
else
    log "✅ Original process is NOT part of managed system (safe to stop)"
fi

# Check 6: GPU utilization and memory
echo ""
echo "CHECK 6: GPU Resource Analysis"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

original_gpu=$(nvidia-smi --query-compute-apps=pid,gpu_uuid --format=csv,noheader,nounits 2>/dev/null | \
    grep "^$ORIGINAL_PID" | head -1 | awk -F', ' '{print $2}')

parallel_gpu=$(nvidia-smi --query-compute-apps=pid,gpu_uuid --format=csv,noheader,nounits 2>/dev/null | \
    grep "^$PARALLEL_PID" | head -1 | awk -F', ' '{print $2}')

# Get GPU index from UUID (simplified)
original_gpu_idx=$(nvidia-smi --list-gpus | grep -n "$original_gpu" | cut -d: -f1 | head -1)
parallel_gpu_idx=$(nvidia-smi --list-gpus | grep -n "$parallel_gpu" | cut -d: -f1 | head -1)

if [ -n "$original_gpu_idx" ]; then
    original_gpu_idx=$((original_gpu_idx - 1))
    original_gpu_info=$(nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader,nounits | \
        awk -F', ' -v idx="$original_gpu_idx" '$1==idx {print $2","$3}')
    log "  Original (PID $ORIGINAL_PID): GPU $original_gpu_idx - $original_gpu_info"
fi

if [ -n "$parallel_gpu_idx" ]; then
    parallel_gpu_idx=$((parallel_gpu_idx - 1))
    parallel_gpu_info=$(nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader,nounits | \
        awk -F', ' -v idx="$parallel_gpu_idx" '$1==idx {print $2","$3}')
    log "  Parallel (PID $PARALLEL_PID): GPU $parallel_gpu_idx - $parallel_gpu_info"
fi

# Check 7: Process runtime comparison
echo ""
echo "CHECK 7: Runtime Comparison"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

original_etime=$(ps -p "$ORIGINAL_PID" -o etime= 2>/dev/null | xargs)
parallel_etime=$(ps -p "$PARALLEL_PID" -o etime= 2>/dev/null | xargs)

log "  Original (PID $ORIGINAL_PID): Runtime: $original_etime"
log "  Parallel (PID $PARALLEL_PID): Runtime: $parallel_etime"

# Check 8: Log file analysis
echo ""
echo "CHECK 8: Log File Analysis"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
log "Analyzing logs for $EXPERIMENT_NAME..."
analyze_logs "$ORIGINAL_PID" "$EXPERIMENT_NAME"

# ============================================================================
# SUMMARY AND CONFIRMATION
# ============================================================================

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "                    SUMMARY & RECOMMENDATION"
echo "═══════════════════════════════════════════════════════════════"
echo ""

echo "Process to STOP:"
echo "  PID: $ORIGINAL_PID"
get_process_info "$ORIGINAL_PID" | sed 's/^/  /'
echo ""

echo "Process to KEEP:"
echo "  PID: $PARALLEL_PID"
get_process_info "$PARALLEL_PID" | sed 's/^/  /'
echo ""

echo "Impact:"
echo "  ✅ Will free GPU $original_gpu_idx (if applicable)"
echo "  ✅ Parallel process will continue running"
echo "  ⚠️  Mining progress for original process will be lost"
echo ""

# Create backup/record
mkdir -p "$BACKUP_DIR"
echo "Creating backup record in: $BACKUP_DIR"
{
    echo "Stop Record - $(date)"
    echo "Original PID: $ORIGINAL_PID"
    echo "Parallel PID: $PARALLEL_PID"
    echo "Original Process Info:"
    get_process_info "$ORIGINAL_PID"
    echo ""
    echo "Parallel Process Info:"
    get_process_info "$PARALLEL_PID"
    echo ""
    echo "Checkpoints found: $checkpoint_count"
} > "$BACKUP_DIR/stop_record.txt"

# Final confirmation
echo ""
read -p "⚠️  CONFIRM: Stop process $ORIGINAL_PID? (type 'STOP' to confirm): " final_confirm

if [ "$final_confirm" != "STOP" ]; then
    log "❌ Cancelled by user"
    exit 0
fi

# ============================================================================
# STOP PROCESS
# ============================================================================

echo ""
log "🛑 Stopping process $ORIGINAL_PID..."

# Graceful stop
log "Sending SIGTERM (graceful shutdown)..."
kill -SIGTERM "$ORIGINAL_PID" 2>/dev/null || {
    log "❌ ERROR: Failed to send SIGTERM"
    exit 1
}

# Wait for cleanup
log "Waiting 30 seconds for cleanup..."
for i in {30..1}; do
    if ! check_process "$ORIGINAL_PID"; then
        log "✅ Process stopped gracefully"
        echo "✅ SUCCESS: Process stopped successfully!"
        nvidia-smi --query-gpu=index,utilization.gpu --format=csv,noheader | head -1
        exit 0
    fi
    sleep 1
    echo -n "."
done
echo ""

# Force stop if still running
if check_process "$ORIGINAL_PID"; then
    log "⚠️  Process still running. Force stopping with SIGKILL..."
    kill -9 "$ORIGINAL_PID" 2>/dev/null || {
        log "❌ ERROR: Failed to force stop"
        exit 1
    }
    sleep 2
    
    if check_process "$ORIGINAL_PID"; then
        log "❌ ERROR: Process still running after force stop!"
        exit 1
    fi
fi

# Verify parallel process still running
if ! check_process "$PARALLEL_PID"; then
    log "⚠️  WARNING: Parallel process is also stopped! This may be unexpected."
else
    log "✅ Parallel process is still running (good)"
fi

# Final status
echo ""
log "✅ SUCCESS: Process $ORIGINAL_PID stopped"
echo ""
echo "📊 Current GPU Status:"
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader

echo ""
log "✅ Done! Check backup record: $BACKUP_DIR/stop_record.txt"
