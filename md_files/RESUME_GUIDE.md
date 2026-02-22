# Guide to Stop and Resume Experiments

## Overview

This guide explains how to:
1. **Stop all experiments** and **release GPU memory** while saving resume information
2. **Resume experiments** from their last checkpoint later

## Key Files

- `stop_and_save_resume_info.py` - Stops all processes and saves resume info
- `resume_from_checkpoint.py` - Resumes stopped experiments from checkpoints
- `stopped_experiments_resume_info.json` - Stores resume information for stopped experiments

## Stopping Experiments and Releasing GPU Memory

### Option 1: Use the Stop Script (Recommended)

```bash
cd /home/prevanka/prati/su-mt-rag/mt-rag-benchmark
python3 stop_and_save_resume_info.py
```

This script will:
1. Find all running/paused training processes
2. Stop them (gracefully with SIGTERM, then SIGKILL if needed)
3. Release GPU memory
4. Save resume information to `stopped_experiments_resume_info.json`

### Option 2: Manual Stop (if script fails)

If some processes don't stop with the script, you can manually kill them:

```bash
# Find all training processes
ps aux | grep -E "python.*train.*tier1|python.*task_a" | grep -v grep

# Kill all training processes (forceful)
ps aux | grep -E "python.*train.*tier1|python.*task_a" | grep -v grep | awk '{print $2}' | xargs kill -9
```

## Resuming Experiments

### Option 1: Automatic Resume (Recommended)

```bash
cd /home/prevanka/prati/su-mt-rag/mt-rag-benchmark
python3 resume_from_checkpoint.py
```

This will resume all stopped experiments from their last checkpoint.

### Option 2: Resume Specific GPU

```bash
# Resume only experiments on GPU 0
python3 resume_from_checkpoint.py --gpu 0
```

### Option 3: List Stopped Experiments

```bash
# See what experiments can be resumed
python3 resume_from_checkpoint.py --list
```

### Option 4: Dry Run (Preview)

```bash
# See what would be resumed without actually resuming
python3 resume_from_checkpoint.py --dry-run
```

### Option 5: Manual Resume

If automatic resume doesn't work, you can manually restart experiments with the `--resume` flag:

```bash
python3 train_neural_ndcg_tier1.py \
    --experiment_name tier1_neural_ndcg \
    --gpu 0 \
    --output_dir experiments/retrieval/tier1_neural_ndcg \
    --resume
```

**Note:** Most training scripts support `--resume` flag which automatically loads the latest checkpoint from the output directory.

## How Checkpoint Resuming Works

1. **Checkpoint Saving**: Training scripts automatically save checkpoints during training (typically every N steps or epochs)
2. **Checkpoint Location**: Checkpoints are saved in the experiment's output directory (e.g., `experiments/retrieval/tier1_neural_ndcg/checkpoint-XXX/`)
3. **Automatic Detection**: When using `--resume`, scripts automatically find and load the latest checkpoint
4. **State Preservation**: Checkpoints include:
   - Model weights
   - Optimizer state
   - Training step/epoch number
   - Learning rate schedule state

## Resume Information File

The `stopped_experiments_resume_info.json` file contains:
- Experiment name
- GPU assignment
- Script path
- Output directory
- Checkpoint directory (latest checkpoint found)
- Original command
- Stop timestamp

## Important Notes

1. **Memory Release**: Stopping processes (not just pausing) is necessary to fully release GPU memory
2. **Checkpoint Frequency**: Ensure your training scripts save checkpoints regularly (check `checkpoint_save_steps` parameter)
3. **Data Loss**: Stopping processes may lose progress since the last checkpoint. Ensure frequent checkpointing for important experiments
4. **Resume Compatibility**: Resume works only if:
   - The training script supports `--resume` flag
   - Checkpoints exist in the output directory
   - The script hasn't changed significantly since stopping

## Troubleshooting

### Process Won't Stop

If a process won't stop gracefully:
```bash
# Force kill specific PID
kill -9 <PID>

# Or kill all training processes
ps aux | grep -E "python.*train" | grep -v grep | awk '{print $2}' | xargs kill -9
```

### Resume Not Working

1. Check if checkpoint directory exists:
   ```bash
   ls -la experiments/retrieval/<experiment_name>/checkpoint-*/
   ```

2. Verify the script supports `--resume`:
   ```bash
   python3 <script> --help | grep resume
   ```

3. Check training logs for checkpoint saves:
   ```bash
   tail -f experiments/retrieval/<experiment_name>/training.log
   ```

### GPU Memory Not Released

1. Wait a few seconds after stopping processes
2. Check if processes are truly gone:
   ```bash
   nvidia-smi
   ps aux | grep python
   ```
3. If memory is still allocated, there may be zombie processes. Try rebooting if necessary (last resort).

## Example Workflow

```bash
# 1. Stop all experiments and release GPU memory
python3 stop_and_save_resume_info.py

# 2. Verify GPUs are free
nvidia-smi

# 3. (Optional) Do other work that needs GPUs

# 4. Later, resume all stopped experiments
python3 resume_from_checkpoint.py

# 5. Monitor resumed experiments
tail -f experiments/retrieval/*/training.log
```

