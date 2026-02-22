# Resume Experiment Usage Guide

## Overview

This guide explains how to pause and resume experiments to free GPUs for other users.

## Current Checkpoint Status

Based on the latest check, **no checkpoints exist yet** for the running experiments:
- `phase4_hard_negatives_cosine`: No checkpoints
- `phase4_hard_negatives_triplet`: No checkpoints
- `phase4_bge_large`: No checkpoints
- `phase4_hard_negatives_5neg`: No checkpoints

**Reason**: Experiments are likely still in the **hard negative mining phase**, which happens before training starts. Checkpoints are only saved every 1000 training steps, so you need to wait for training to begin and reach at least step 1000.

---

## How to Check for Checkpoints

```bash
# Check if checkpoints exist for an experiment
find experiments/retrieval/phase4_*/models/*-checkpoints/ -name "checkpoint-*" -type d 2>/dev/null

# Or check specific experiment
ls -la ./models/phase4_hard_negatives_cosine-checkpoints/checkpoint-* 2>/dev/null
```

---

## Pause and Resume Workflow

### Step 1: Check for Checkpoints (Required Before Pausing)

```bash
EXPERIMENT="phase4_hard_negatives_cosine"
CHECKPOINT_COUNT=$(find ./models/${EXPERIMENT}-checkpoints/ -name "checkpoint-*" -type d 2>/dev/null | wc -l)

if [ $CHECKPOINT_COUNT -eq 0 ]; then
    echo "⚠️  No checkpoints found. Pausing will lose hard negative mining progress."
    echo "   Hard negative mining will restart from beginning when resumed."
else
    echo "✅ Found $CHECKPOINT_COUNT checkpoint(s). Safe to pause."
fi
```

### Step 2: Pause Experiment (Graceful Stop)

```bash
# Find the process ID
EXPERIMENT="phase4_hard_negatives_cosine"
PID=$(ps aux | grep "$EXPERIMENT" | grep train_advanced_bge | grep -v grep | awk '{print $2}')

if [ -z "$PID" ]; then
    echo "Experiment not running"
    exit 1
fi

# Graceful stop (allows cleanup)
echo "Stopping process $PID gracefully..."
kill -SIGTERM $PID

# Wait 30 seconds for cleanup
sleep 30

# If still running, force kill
if ps -p $PID > /dev/null; then
    echo "Force killing process..."
    kill -9 $PID
fi

# Verify GPU freed
nvidia-smi
```

### Step 3: Resume Experiment

#### Option A: Auto-detect Latest Checkpoint

```bash
python resume_experiment.py --experiment phase4_hard_negatives_cosine --gpu_id 0
```

#### Option B: Specify Checkpoint Path

```bash
CHECKPOINT="./models/phase4_hard_negatives_cosine-checkpoints/checkpoint-2000"
python resume_experiment.py --experiment phase4_hard_negatives_cosine \
    --checkpoint "$CHECKPOINT" --gpu_id 0
```

---

## Resume Script Details

### Features

1. **Auto-detection**: Automatically finds the latest checkpoint if not specified
2. **Checkpoint Info**: Displays checkpoint state (steps, epochs completed)
3. **Error Handling**: Clear error messages if checkpoint not found
4. **GPU Assignment**: Supports `--gpu_id` to specify which GPU to use

### Usage

```bash
python resume_experiment.py --help
```

### Arguments

- `--experiment` (required): Experiment name (e.g., `phase4_hard_negatives_cosine`)
- `--checkpoint` (optional): Specific checkpoint path (auto-detects latest if not provided)
- `--gpu_id` (optional): GPU ID to use for training

---

## What Happens During Resume

1. **Checkpoint Loading**: The model is loaded from the specified checkpoint
2. **State Recovery**: Checkpoint state (steps, epochs) is read from `trainer_state.json`
3. **Training Continuation**: Training continues from where it left off
4. **Hard Negative Mining**: Re-mines hard negatives (faster with resumed model than initial mining)
5. **Checkpoint Saving**: New checkpoints continue to be saved every 1000 steps

---

## Important Notes

### ⚠️ Hard Negative Mining Phase

If you pause during **hard negative mining** (before training starts):
- ❌ **Progress is lost** - mining will restart from beginning
- ✅ **No checkpoint needed** - mining doesn't require checkpoints
- ✅ **Can pause/resume anytime** - just restarts mining

### ✅ Training Phase

If you pause during **training** (after mining completes):
- ✅ **Progress is saved** if checkpoint exists (step 1000+)
- ✅ **Can resume** from latest checkpoint
- ✅ **Training continues** from where it left off

### 🔍 How to Check Current Phase

```bash
# Check experiment log
tail -f experiments/retrieval/phase4_hard_negatives_cosine/training.log

# Look for:
# - "Mining hard negatives..." = Still in mining phase
# - "Starting model fine-tuning..." = Training started
# - "Step X/XXX" = Training in progress (checkpoints may exist)
```

---

## Example: Complete Pause/Resume Workflow

```bash
#!/bin/bash
EXPERIMENT="phase4_hard_negatives_cosine"
GPU_ID=0

# 1. Check for checkpoints
echo "Checking for checkpoints..."
CHECKPOINTS=$(find ./models/${EXPERIMENT}-checkpoints/ -name "checkpoint-*" -type d 2>/dev/null)
if [ -z "$CHECKPOINTS" ]; then
    echo "⚠️  Warning: No checkpoints found. Will lose mining progress if paused."
    read -p "Continue with pause? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# 2. Pause
echo "Pausing experiment..."
PID=$(ps aux | grep "$EXPERIMENT" | grep train_advanced_bge | grep -v grep | awk '{print $2}')
if [ -n "$PID" ]; then
    kill -SIGTERM $PID
    sleep 30
    if ps -p $PID > /dev/null; then
        kill -9 $PID
    fi
    echo "✅ Experiment paused"
else
    echo "Experiment not running"
fi

# 3. Wait (GPU freed for other users)
echo "GPU freed. Waiting..."
sleep 3600  # Wait 1 hour (example)

# 4. Resume
echo "Resuming experiment..."
python resume_experiment.py --experiment "$EXPERIMENT" --gpu_id $GPU_ID
```

---

## Troubleshooting

### Error: "No checkpoint found"

**Cause**: Training hasn't reached step 1000 yet, or checkpoint directory doesn't exist.

**Solution**: 
- Wait for training to reach step 1000 (first checkpoint)
- Or accept that hard negative mining will restart

### Error: "Checkpoint path does not exist"

**Cause**: Invalid checkpoint path specified.

**Solution**: 
- Use auto-detection: `python resume_experiment.py --experiment XXX` (without `--checkpoint`)
- Or verify path: `ls -la ./models/XXX-checkpoints/`

### Resume starts from beginning

**Cause**: Checkpoint doesn't contain training state, or model.fit() restarts.

**Solution**: This is expected if resuming from a checkpoint saved before training starts. The model will continue training normally.

---

## Implementation Details

### Modified Files

1. **`train_advanced_bge.py`**:
   - Added support for `resume_from_checkpoint` config parameter
   - Loads model from checkpoint if specified
   - Displays checkpoint state information
   - Continues training from checkpoint

2. **`resume_experiment.py`** (new):
   - Auto-detects latest checkpoint
   - Loads checkpoint state
   - Modifies config to resume
   - Calls `run_advanced_training()` with resume config

### Checkpoint Structure

```
./models/{experiment_name}-checkpoints/
  └── checkpoint-{step}/
      ├── model.safetensors (or pytorch_model.bin)
      ├── config.json
      ├── trainer_state.json  (contains: global_step, epoch, etc.)
      └── ... (other model files)
```

---

## Summary

✅ **Resume script implemented** - `resume_experiment.py`  
✅ **Training script modified** - `train_advanced_bge.py` supports resume  
⚠️ **Checkpoints not available yet** - Experiments still in mining phase  
✅ **Safe to pause/resume** - Will restart mining, then continue training  

**Next Steps**:
1. Wait for experiments to reach step 1000 (first checkpoint)
2. Use `resume_experiment.py` to resume after pausing
3. Monitor checkpoint creation: `watch -n 60 'find ./models/*-checkpoints -name checkpoint-* | wc -l'`

