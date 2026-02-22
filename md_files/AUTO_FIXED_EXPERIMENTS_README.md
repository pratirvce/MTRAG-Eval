# Auto Fixed Experiments Runner

This script automatically starts fixed experiments and new/pending experiments when GPUs become available.

## Overview

The `auto_start_fixed_experiments.py` script monitors GPU availability and automatically starts:
1. **Fixed Experiments** - Experiments that had bugs fixed and need to be re-run
2. **Pending Experiments** - New experiments that haven't been started yet

## Features

- ✅ **Automatic GPU Detection** - Monitors GPU utilization and memory to find free GPUs
- ✅ **Priority-Based Queue** - Starts experiments in priority order (fixed experiments first)
- ✅ **Resume Support** - Automatically resumes experiments from checkpoints if available
- ✅ **Status Tracking** - Tracks experiment status (pending, running, completed, failed)
- ✅ **Retry Logic** - Automatically retries failed experiments (up to 3 times)
- ✅ **Process Monitoring** - Monitors running processes and updates status
- ✅ **Persistent Status** - Saves status to JSON file for recovery

## Fixed Experiments (Priority 1)

These experiments have fixes applied and are ready to re-run:

1. **`tier1_enhanced_contrastive_hardnegatives_fixed`**
   - Fix: Dtype conversion before F.normalize()
   - Script: `train_enhanced_contrastive_tier1.py`

2. **`tier1_learning_to_rank_listwise_fixed`**
   - Fix: Evaluator reuse fixed (removed None retriever creation)
   - Script: `train_learning_to_rank_tier1.py`

3. **`best_paper_hierarchical_routing_fixed`**
   - Fix: Indentation bug fixed in run_hierarchical_routing_evaluation
   - Script: `train_hierarchical_routing_tier1.py`

## Pending Experiments

These experiments are queued to start when GPUs become available:

1. **`tier1_cross_attention_query_document_fixed`** (Priority 2)
2. **`tier1_adversarial_curriculum`** (Priority 3)
3. **`tier1_graph_aware_retrieval`** (Priority 3)
4. **`tier1_llm_distillation`** (Priority 4)
5. **`tier1_multitask_retrieval`** (Priority 4)

## Usage

### Start the Auto Runner

```bash
# Option 1: Use the convenience script
./start_auto_fixed_experiments.sh

# Option 2: Run directly
python3 auto_start_fixed_experiments.py
```

### Monitor Status

```bash
# View live logs
tail -f logs/auto_fixed_experiments.log

# Check status file
cat auto_fixed_experiments_status.json | python3 -m json.tool
```

### Stop the Auto Runner

```bash
# If using the convenience script
kill $(cat logs/auto_fixed_experiments.pid)

# Or find and kill manually
ps aux | grep auto_start_fixed_experiments.py
kill <PID>
```

## Configuration

You can modify these settings in the script:

```python
self.check_interval = 300  # Check every 5 minutes
self.min_gpu_utilization = 10  # GPU is free if utilization < 10%
self.min_free_memory_mb = 2000  # At least 2GB free memory
self.max_retries = 3  # Maximum retry attempts for failed experiments
```

## How It Works

1. **GPU Monitoring**: Every 5 minutes (configurable), the script checks GPU utilization and memory
2. **Experiment Queue**: Builds a queue of experiments that need to run, sorted by priority
3. **GPU Assignment**: Assigns free GPUs to experiments in priority order
4. **Process Management**: Starts experiments in the background and tracks their PIDs
5. **Status Updates**: Continuously monitors running experiments and updates their status
6. **Automatic Retry**: If an experiment fails, it's automatically retried (up to 3 times)

## Status File

The script maintains a status file (`auto_fixed_experiments_status.json`) that tracks:
- Experiment status (pending, running, completed, failed)
- GPU assignment
- Process ID (PID)
- Start time
- Error messages
- Retry count

## Example Output

```
2025-12-18 23:30:00 - INFO - ============================================================
2025-12-18 23:30:00 - INFO - Auto Fixed Experiments Runner Started
2025-12-18 23:30:00 - INFO - ============================================================
2025-12-18 23:30:00 - INFO - Monitoring 3 fixed experiments and 5 pending experiments
2025-12-18 23:30:00 - INFO - Check interval: 300 seconds
2025-12-18 23:30:00 - INFO - GPU free threshold: <10% utilization, >2000MB free memory
2025-12-18 23:30:00 - INFO - ============================================================

--- Iteration 1 ---
2025-12-18 23:30:00 - INFO - Time: 2025-12-18 23:30:00
2025-12-18 23:30:01 - INFO - Free GPUs: [5]
2025-12-18 23:30:01 - INFO - Experiments to run: 3
2025-12-18 23:30:01 - INFO - Starting tier1_enhanced_contrastive_hardnegatives_fixed on GPU 5...
2025-12-18 23:30:02 - INFO - ✅ Started tier1_enhanced_contrastive_hardnegatives_fixed on GPU 5 (PID: 12345)
```

## Troubleshooting

### Script not starting experiments

1. Check GPU availability:
   ```bash
   nvidia-smi
   ```

2. Check if experiments are already running:
   ```bash
   ps aux | grep train_.*tier1
   ```

3. Check experiment status:
   ```bash
   cat auto_fixed_experiments_status.json | python3 -m json.tool
   ```

### Experiments failing immediately

1. Check the experiment log:
   ```bash
   tail -50 experiments/retrieval/<experiment_name>/training.log
   ```

2. Verify the script exists:
   ```bash
   ls -la train_*tier1.py
   ```

3. Check Python environment:
   ```bash
   which python3
   python3 --version
   ```

### GPU not detected as free

- Adjust `min_gpu_utilization` and `min_free_memory_mb` in the script
- Check if other processes are using the GPU:
  ```bash
  nvidia-smi
  ```

## Notes

- The script runs continuously until stopped (Ctrl+C or kill signal)
- Status is saved periodically and on shutdown
- Experiments are started with `--resume` flag if checkpoints exist
- Failed experiments are automatically retried up to 3 times
- The script prioritizes fixed experiments over pending ones

