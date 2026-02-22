# Plan: Pausing and Resuming Experiments to Free GPUs

## Overview
This document outlines the strategy for safely pausing running experiments to free GPUs for other users, and resuming them later without losing progress.

## Current Checkpoint System

### ✅ What's Already in Place:
1. **Automatic Checkpoints**: All experiments save checkpoints every 1000 training steps
   - Location: `experiments/retrieval/{experiment_name}/models/{exp_name}-checkpoints/checkpoint-{step}/`
   - Includes: model weights, optimizer state, training state
   - Config: `checkpoint_save_steps=1000` in config files

2. **Signal Handling**: The parallel runner handles SIGTERM/SIGINT gracefully
   - Location: `run_phase4_parallel.py` (lines 380-381)

### ⚠️ Current Limitations:
1. **No Built-in Resume**: SentenceTransformer's `model.fit()` doesn't have a direct `resume_from_checkpoint` parameter
2. **Hard Negative Mining Phase**: If stopped during mining (before training starts), no checkpoint exists yet
3. **Manual Resume Required**: Need custom script to load checkpoint and continue training

---

## Pause/Resume Strategy

### **OPTION 1: Graceful Stop (Recommended)**

#### When to Use:
- ✅ During active training (after hard negative mining completes)
- ✅ When checkpoints exist (at least 1 checkpoint saved)
- ✅ You have time to let current training step complete

#### Steps to Pause:

1. **Identify Running Experiments**:
   ```bash
   ps aux | grep train_advanced_bge | grep -v grep
   nvidia-smi  # Check GPU usage
   ```

2. **Find Latest Checkpoint** (for each experiment):
   ```bash
   # Check if checkpoints exist
   ls -lt experiments/retrieval/phase4_*/models/*-checkpoints/checkpoint-* 2>/dev/null | head -5
   ```

3. **Gracefully Stop Process** (SIGTERM - allows cleanup):
   ```bash
   # Find PID
   PID=$(ps aux | grep "phase4_hard_negatives_cosine" | grep train_advanced_bge | grep -v grep | awk '{print $2}')
   
   # Send SIGTERM (graceful)
   kill -SIGTERM $PID
   
   # Wait 30 seconds for cleanup
   sleep 30
   
   # If still running, force kill
   if ps -p $PID > /dev/null; then
       kill -9 $PID
   fi
   ```

4. **Verify Checkpoint Saved**:
   ```bash
   # Check latest checkpoint timestamp
   find experiments/retrieval/phase4_*/models/*-checkpoints/ -name "checkpoint-*" -type d | sort | tail -1
   ```

#### Steps to Resume:

1. **Create Resume Script** (see `resume_experiment.py` below)
2. **Identify Checkpoint**:
   ```bash
   LATEST_CHECKPOINT=$(find experiments/retrieval/phase4_*/models/*-checkpoints/ -name "checkpoint-*" -type d | sort -V | tail -1)
   echo $LATEST_CHECKPOINT
   ```

3. **Modify Config to Resume**:
   - Load checkpoint path in training script
   - Adjust starting epoch/step based on checkpoint

4. **Resume Training**:
   ```bash
   python resume_experiment.py --experiment phase4_hard_negatives_cosine --checkpoint $LATEST_CHECKPOINT --gpu_id 0
   ```

---

### **OPTION 2: Force Stop (Use with Caution)**

#### When to Use:
- ⚠️ Only if you MUST free GPU immediately
- ⚠️ During hard negative mining (before any checkpoints)
- ⚠️ No other option available

#### Risks:
- ❌ Loss of progress in hard negative mining phase (must restart mining)
- ❌ Potential data corruption if killed mid-write
- ❌ Training must restart from beginning

#### Steps:
```bash
# Force kill (immediate)
kill -9 $PID

# Note: Hard negative mining will restart from beginning when resumed
```

---

### **OPTION 3: Suspend Process (Advanced)**

#### When to Use:
- ✅ Need to pause for short time (< 1 hour)
- ✅ Want to preserve exact state (including in-memory data)
- ⚠️ Only works if process accepts SIGSTOP/SIGCONT

#### Steps:
```bash
# Suspend (freezes process)
kill -SIGSTOP $PID

# Resume later
kill -SIGCONT $PID
```

#### ⚠️ Warnings:
- GPU memory still allocated (doesn't free GPU)
- System swap may occur
- Not recommended for long pauses

---

## Implementation: Resume Script

### File: `resume_experiment.py`

```python
#!/usr/bin/env python3
"""
Resume a paused experiment from the latest checkpoint.
"""
import argparse
import json
import pathlib
import logging
from sentence_transformers import SentenceTransformer
from train_advanced_bge import run_advanced_training

logging.basicConfig(level=logging.INFO)

def find_latest_checkpoint(experiment_dir):
    """Find the latest checkpoint for an experiment."""
    checkpoint_dir = experiment_dir / "models"
    if not checkpoint_dir.exists():
        return None
    
    # Find all checkpoint directories
    checkpoints = list(checkpoint_dir.glob("*-checkpoints/checkpoint-*"))
    if not checkpoints:
        return None
    
    # Sort by checkpoint number (extract from path)
    def get_step(cp):
        try:
            return int(cp.name.split("-")[-1])
        except:
            return 0
    
    latest = sorted(checkpoints, key=get_step, reverse=True)[0]
    return latest

def load_checkpoint_info(checkpoint_path):
    """Load training state from checkpoint."""
    state_file = checkpoint_path / "trainer_state.json"
    if not state_file.exists():
        return None
    
    with open(state_file) as f:
        state = json.load(f)
    
    return {
        "global_step": state.get("global_step", 0),
        "epoch": state.get("epoch", 0.0),
        "best_model_checkpoint": state.get("best_model_checkpoint")
    }

def resume_training(experiment_name, checkpoint_path=None, gpu_id=None):
    """Resume training from checkpoint."""
    exp_dir = pathlib.Path(f"experiments/retrieval/{experiment_name}")
    config_file = exp_dir / "config.json"
    
    if not config_file.exists():
        raise FileNotFoundError(f"Config not found: {config_file}")
    
    with open(config_file) as f:
        config = json.load(f)
    
    # Find latest checkpoint if not specified
    if checkpoint_path is None:
        checkpoint_path = find_latest_checkpoint(exp_dir)
        if checkpoint_path is None:
            raise ValueError(f"No checkpoint found for {experiment_name}")
    
    checkpoint_path = pathlib.Path(checkpoint_path)
    logging.info(f"📂 Resuming from checkpoint: {checkpoint_path}")
    
    # Load checkpoint info
    checkpoint_info = load_checkpoint_info(checkpoint_path)
    if checkpoint_info:
        logging.info(f"   Step: {checkpoint_info['global_step']}, Epoch: {checkpoint_info['epoch']}")
    
    # Modify config to resume from checkpoint
    config['resume_from_checkpoint'] = str(checkpoint_path)
    if gpu_id is not None:
        config['gpu_id'] = gpu_id
    
    # Update model path to checkpoint
    # Note: This requires modifying train_advanced_bge.py to support resume
    config['base_model'] = str(checkpoint_path)
    
    logging.info(f"🔄 Resuming {experiment_name}...")
    run_advanced_training(config)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", required=True, help="Experiment name")
    parser.add_argument("--checkpoint", help="Checkpoint path (optional, will find latest)")
    parser.add_argument("--gpu_id", type=int, help="GPU ID to use")
    args = parser.parse_args()
    
    resume_training(args.experiment, args.checkpoint, args.gpu_id)
```

---

## Required Code Changes

### Modify `train_advanced_bge.py`:

Add checkpoint resume support:

```python
def run_advanced_training(config: Dict):
    # ... existing code ...
    
    # Check for resume from checkpoint
    resume_from = config.get('resume_from_checkpoint')
    if resume_from and pathlib.Path(resume_from).exists():
        logging.info(f"🔄 Resuming from checkpoint: {resume_from}")
        model = SentenceTransformer(resume_from, device=device)
        # Load training state and adjust epochs/steps if needed
    else:
        model = SentenceTransformer(base_model_name, device=device)
    
    # ... rest of training code ...
```

---

## Pause/Resume Workflow (Quick Reference)

### To Free GPUs NOW:

```bash
# 1. List running experiments
./check_parallel_status.sh

# 2. For each experiment to pause:
EXPERIMENT="phase4_hard_negatives_cosine"
PID=$(ps aux | grep "$EXPERIMENT" | grep train_advanced_bge | grep -v grep | awk '{print $2}')

# 3. Check for checkpoints
ls -lt experiments/retrieval/$EXPERIMENT/models/*-checkpoints/checkpoint-* 2>/dev/null | head -1

# 4. Gracefully stop (if checkpoint exists)
if [ -n "$PID" ]; then
    kill -SIGTERM $PID
    echo "Waiting 30s for cleanup..."
    sleep 30
    if ps -p $PID > /dev/null; then
        kill -9 $PID
    fi
fi

# 5. Verify GPU freed
nvidia-smi
```

### To Resume Later:

```bash
# 1. Find latest checkpoint
EXPERIMENT="phase4_hard_negatives_cosine"
CHECKPOINT=$(find experiments/retrieval/$EXPERIMENT/models/*-checkpoints/ -name "checkpoint-*" -type d | sort -V | tail -1)

# 2. Resume
python resume_experiment.py --experiment $EXPERIMENT --checkpoint $CHECKPOINT --gpu_id 0
```

---

## Risk Assessment

### ✅ Safe to Pause:
- After at least 1 checkpoint saved (1000+ steps)
- During training (not mining)
- When you can wait 30s for graceful shutdown

### ⚠️ Risky to Pause:
- During hard negative mining (before training)
- Before first checkpoint (first 1000 steps)
- Force kill (kill -9) without graceful stop

### ❌ Will Lose Progress:
- Hard negative mining (must restart)
- First 1000 steps if no checkpoint yet
- Force kill during checkpoint save

---

## Recommendations

1. **Best Practice**: Wait for at least 1 checkpoint before pausing
2. **Use SIGTERM**: Always try graceful stop first (30s timeout)
3. **Document State**: Save experiment state before pausing:
   ```bash
   # Save current status
   ./check_parallel_status.sh > pause_state_$(date +%Y%m%d_%H%M%S).txt
   ```
4. **Resume ASAP**: Don't leave paused for days (checkpoint paths may change)
5. **Test Resume**: After code changes, test resume on a small experiment first

---

## Alternative: Implement Checkpoint Resume in Training Script

For a more robust solution, modify `train_advanced_bge.py` to:
1. Check for existing checkpoints at startup
2. Automatically resume if checkpoint found
3. Skip completed epochs/steps
4. Continue from last checkpoint

This requires modifying the training loop to:
- Load from checkpoint if exists
- Skip already-completed steps
- Adjust epoch counter
- Handle hard negative mining state

---

## Summary

**Current Status**: 
- ✅ Checkpoints are saved automatically (every 1000 steps)
- ⚠️ Resume requires manual script (not yet implemented)
- ⚠️ Hard negative mining progress not saved (must restart)

**To Pause**: Use `kill -SIGTERM` after first checkpoint
**To Resume**: Need to implement `resume_experiment.py` (outlined above)
**Risk Level**: Low if paused after checkpoint, High if paused during mining

**Next Steps**:
1. Implement `resume_experiment.py`
2. Modify `train_advanced_bge.py` to support resume
3. Test on small experiment first
4. Document checkpoint locations for each experiment

