# Parallel Experiments Guide - Multi-GPU Execution

This guide explains how to run experiments in parallel across multiple GPUs for faster execution.

## 🚀 Features

1. **Automatic GPU Detection**: Automatically detects free GPUs
2. **Parallel Execution**: Runs multiple experiments simultaneously
3. **GPU Assignment**: Assigns each experiment to a specific GPU
4. **Load Balancing**: Distributes experiments across available GPUs
5. **Domain-Specific Parallel**: Run all 4 domains simultaneously on separate GPUs

## 📊 GPU Availability

Current setup:
- **Total GPUs**: 6 (NVIDIA RTX 3090)
- **Currently in use**: GPU 0 (Phase 4 experiment)
- **Free GPUs**: 5 (GPUs 1-5)

## 🎯 Usage Options

### Option 1: Run All Phase 4 Experiments in Parallel

This will distribute all Phase 4 experiments across available GPUs:

```bash
./start_parallel_experiments.sh
```

Or manually:
```bash
python run_phase4_parallel.py --experiment all --max_parallel 5
```

**What happens:**
- Detects free GPUs (GPUs 1-5)
- Starts up to 5 experiments in parallel
- Each experiment runs on a separate GPU
- Automatically starts next experiment when one completes

**Expected speedup**: 3-5x faster than sequential execution

### Option 2: Run Domain-Specific Experiments in Parallel

Run all 4 domain-specific models simultaneously, each on its own GPU:

```bash
./start_domain_parallel.sh
```

Or manually:
```bash
python run_domain_specific_parallel.py
```

**What happens:**
- clapnq → GPU 1
- fiqa → GPU 2
- govt → GPU 3
- cloud → GPU 4

**Expected speedup**: 4x faster (all domains train simultaneously)

### Option 3: Run Single Experiment on Specific GPU

Modify config to specify GPU:

```python
{
  "gpu_id": 2,  # Use GPU 2
  ...
}
```

Or set environment variable:
```bash
CUDA_VISIBLE_DEVICES=2 python train_advanced_bge.py --config config.json
```

## 📋 Monitoring Parallel Experiments

### Check Status
```bash
python run_phase4_parallel.py --status
```

This shows:
- Which experiments are running
- Which GPU each experiment is using
- Completion status

### View Logs
```bash
# Main parallel runner log
tail -f experiments/retrieval/logs/phase4_parallel_*.log

# Specific experiment log
tail -f experiments/retrieval/phase4_hard_negatives_cosine/training.log
```

### Check GPU Usage
```bash
watch -n 1 nvidia-smi
```

## 🔧 Configuration

### Maximum Parallel Experiments

Default: 5 (uses 5 GPUs simultaneously)

Change with:
```bash
python run_phase4_parallel.py --max_parallel 3  # Use only 3 GPUs
```

### GPU Selection

The system automatically selects free GPUs. To manually specify:

1. Edit experiment config:
```json
{
  "gpu_id": 1,  # Use GPU 1
  ...
}
```

2. Or modify `run_phase4_parallel.py` to specify GPU IDs

## ⚡ Performance Improvements

### Sequential Execution (Current)
- 1 experiment at a time
- Total time: ~100-150 hours for all experiments

### Parallel Execution (5 GPUs)
- 5 experiments simultaneously
- Total time: ~20-30 hours for all experiments
- **Speedup: 4-5x faster**

### Domain-Specific Parallel
- All 4 domains simultaneously
- Total time: ~2-4 hours (one training per domain)
- **Speedup: 4x faster**

## 📊 Expected Timeline Comparison

| Method | Time for 7 Experiments | Speedup |
|--------|------------------------|---------|
| Sequential | 100-150 hours (4-6 days) | 1x |
| Parallel (5 GPUs) | 20-30 hours (1 day) | 4-5x |
| Domain-Specific Parallel | 2-4 hours | 4x |

## 🎯 Recommended Workflow

1. **First**: Stop current sequential experiment (if needed)
2. **Then**: Start parallel experiments:
   ```bash
   ./start_parallel_experiments.sh
   ```
3. **Or**: Start domain-specific parallel:
   ```bash
   ./start_domain_parallel.sh
   ```

## ⚠️ Important Notes

1. **GPU Memory**: Each experiment uses ~1.6 GB GPU memory
   - 5 parallel experiments = ~8 GB total (well within limits)

2. **CPU Usage**: Multiple experiments will increase CPU usage
   - Monitor system resources

3. **Disk I/O**: Multiple experiments writing logs simultaneously
   - Ensure sufficient disk space and I/O capacity

4. **Current Experiment**: The running Phase 4 experiment on GPU 0 will continue
   - Parallel experiments will use GPUs 1-5

## 🔍 Troubleshooting

### No GPUs Available
```bash
# Check GPU status
nvidia-smi

# Check if processes are using GPUs
nvidia-smi pmon
```

### Out of Memory
- Reduce `max_parallel` to use fewer GPUs
- Or reduce batch sizes in experiment configs

### Experiments Not Starting
- Check logs: `experiments/retrieval/logs/phase4_parallel_*.log`
- Verify GPUs are actually free
- Check GPU driver/CUDA installation

## 📝 Status Files

- **Status**: `experiments/retrieval/phase4_parallel_status.json`
- **PID**: `experiments/retrieval/phase4_parallel_runner.pid`
- **Logs**: `experiments/retrieval/logs/phase4_parallel_*.log`

## 🎓 Example: Running Domain-Specific in Parallel

```bash
# This will start all 4 domains simultaneously
python run_domain_specific_parallel.py

# Output:
# Starting clapnq training on GPU 1
# Starting fiqa training on GPU 2
# Starting govt training on GPU 3
# Starting cloud training on GPU 4
# Waiting for all domain experiments to complete...
# ✅ All domain-specific experiments completed!
```

**Time**: ~2-4 hours (vs ~8-12 hours sequential)

---

**Ready to speed up?** Use `./start_parallel_experiments.sh` or `./start_domain_parallel.sh`!


