# Phase 4 Experiments - Monitoring Guide

## ✅ Experiments Started!

Your Phase 4 experiments are now running in the background. This guide shows you how to monitor their progress.

## 📊 Quick Status Check

### Method 1: Status Script (Recommended)
```bash
./check_phase4_status.sh
```

### Method 2: Python Status Check
```bash
python run_phase4_background.py --status
```

### Method 3: Check Status File Directly
```bash
cat experiments/retrieval/phase4_status.json | python -m json.tool
```

## 📝 Log Files

### Main Background Runner Log
```bash
# View latest log
tail -f experiments/retrieval/logs/phase4_background_*.log

# View specific log (replace timestamp)
tail -f experiments/retrieval/logs/phase4_background_20251127_173831.log
```

### Individual Experiment Logs
Each experiment has its own training log:
```bash
# Current experiment
tail -f experiments/retrieval/phase4_hard_negatives_cosine/training.log

# Specific experiment
tail -f experiments/retrieval/{experiment_name}/training.log
```

### Runner Output Log
```bash
tail -f experiments/retrieval/logs/phase4_runner_*.out
```

## 🔍 Monitoring Commands

### Check if Background Process is Running
```bash
# Check PID file
cat experiments/retrieval/phase4_runner.pid

# Check process status
ps -p $(cat experiments/retrieval/phase4_runner.pid)

# See all Python processes
ps aux | grep "run_phase4_background"
```

### Check GPU Usage (if using GPU)
```bash
watch -n 1 nvidia-smi
```

### Check Experiment Progress
```bash
# See which experiments are running
python run_phase4_background.py --status

# Check latest log entries
tail -50 experiments/retrieval/logs/phase4_background_*.log
```

## 📈 Understanding Status

Status values:
- **⏳ pending**: Not started yet
- **🔄 running**: Currently training
- **✅ completed**: Finished successfully
- **❌ failed**: Error occurred

## 🛑 Stopping Experiments

### Graceful Stop (Recommended)
```bash
# Get PID
PID=$(cat experiments/retrieval/phase4_runner.pid)

# Send SIGTERM (allows cleanup)
kill $PID
```

### Force Stop
```bash
# If graceful stop doesn't work
kill -9 $(cat experiments/retrieval/phase4_runner.pid)
```

### Stop All Python Training Processes
```bash
pkill -f "train_advanced_bge.py"
pkill -f "train_domain_specific_bge.py"
```

## 🔄 Resuming After Interruption

If experiments are interrupted, you can resume:
```bash
# Status is automatically saved, just restart
./start_phase4_background.sh

# Or manually resume
python run_phase4_background.py --experiment all --resume
```

Already completed experiments will be skipped automatically.

## 📊 Expected Timeline

| Experiment | Estimated Time | Status |
|------------|---------------|--------|
| phase4_hard_negatives_cosine | 2-4 hours | 🔄 Running |
| phase4_hard_negatives_triplet | 2-4 hours | ⏳ Pending |
| phase4_hard_negatives_5neg | 2-4 hours | ⏳ Pending |
| phase4_bge_large | 4-6 hours | ⏳ Pending |
| phase4_domain_specific_* | 4-8 hours total | ⏳ Pending |

**Total Estimated Time**: 12-16 hours

## 🎯 What to Watch For

### Good Signs ✅
- Training loss decreasing
- Validation metrics improving
- Process running continuously
- No error messages in logs

### Warning Signs ⚠️
- Training loss not decreasing
- Validation metrics degrading (overfitting)
- Out of memory errors
- Process stopped unexpectedly

### Checkpoints
- Models save checkpoints during training
- Best model saved based on validation metrics
- Checkpoint locations: `./models/{experiment_name}-checkpoints/`

## 📁 Important Files

- **Status**: `experiments/retrieval/phase4_status.json`
- **PID**: `experiments/retrieval/phase4_runner.pid`
- **Logs**: `experiments/retrieval/logs/phase4_background_*.log`
- **Experiment Logs**: `experiments/retrieval/{experiment_name}/training.log`
- **Models**: `./models/{experiment_name}/`

## 🔔 Notifications (Optional)

You can set up email notifications when experiments complete:
```bash
# Add to your .bashrc or run manually
alias check-experiments='./check_phase4_status.sh && echo "Check complete at $(date)"'
```

## 📊 Quick Status Summary

Run this to get a quick overview:
```bash
echo "=== Phase 4 Experiments Status ===" && \
python run_phase4_background.py --status && \
echo "" && \
echo "=== Recent Log Activity ===" && \
tail -10 experiments/retrieval/logs/phase4_background_*.log
```

## 🆘 Troubleshooting

### Process Not Running
```bash
# Check if PID file exists but process is dead
if [ -f experiments/retrieval/phase4_runner.pid ]; then
    PID=$(cat experiments/retrieval/phase4_runner.pid)
    if ! ps -p $PID > /dev/null; then
        echo "Stale PID file, removing..."
        rm experiments/retrieval/phase4_runner.pid
        # Restart experiments
        ./start_phase4_background.sh
    fi
fi
```

### Check Logs for Errors
```bash
# Check for errors in background log
grep -i error experiments/retrieval/logs/phase4_background_*.log

# Check experiment-specific errors
find experiments/retrieval -name "error.log" -exec tail -20 {} \;
```

### Out of Memory
```bash
# Check memory usage
free -h
nvidia-smi  # For GPU memory
```

## 📝 Next Steps

1. **Monitor**: Use `./check_phase4_status.sh` regularly
2. **Wait**: Experiments run sequentially (12-16 hours total)
3. **Evaluate**: After completion, run evaluation:
   ```bash
   python evaluate_advanced_models.py --model_path ./models/phase4_hard_negatives_cosine
   ```
4. **Compare**: Compare results with baseline and previous phases

---

**Current Status**: Check with `./check_phase4_status.sh`

