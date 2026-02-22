# Quick Start: Parallel Multi-GPU Experiments

## ✅ Implementation Complete!

All parallel execution features have been implemented:

1. ✅ **Multi-GPU experiment scheduler** - Distributes experiments across GPUs
2. ✅ **GPU device assignment** - Each experiment uses a specific GPU
3. ✅ **Parallel domain-specific runner** - All 4 domains simultaneously
4. ✅ **Automatic GPU detection** - Finds free GPUs automatically

## 🚀 Quick Start

### Option 1: Run All Phase 4 Experiments in Parallel (Recommended)

This distributes all experiments across 5 free GPUs:

```bash
./start_parallel_experiments.sh
```

**What it does:**
- Detects free GPUs (1-5, since 0 is in use)
- Starts up to 5 experiments simultaneously
- Each runs on a separate GPU
- **Expected time: 20-30 hours** (vs 100-150 hours sequential)

### Option 2: Run Domain-Specific Models in Parallel

Run all 4 domain-specific models simultaneously:

```bash
./start_domain_parallel.sh
```

**What it does:**
- clapnq → GPU 1
- fiqa → GPU 2  
- govt → GPU 3
- cloud → GPU 4
- **Expected time: 2-4 hours** (vs 8-12 hours sequential)

## 📊 Current Status

**Free GPUs Available:** 5 (GPUs 1-5)
**Current Usage:** GPU 0 running Phase 4 experiment
**Can Run:** 5 experiments in parallel immediately

## 🔍 Monitor Progress

```bash
# Check parallel experiment status
./check_parallel_status.sh

# Or use Python status
python run_phase4_parallel.py --status

# Watch GPU usage
watch -n 1 nvidia-smi
```

## ⚡ Expected Speedups

| Scenario | Sequential Time | Parallel Time | Speedup |
|----------|----------------|---------------|---------|
| All Phase 4 (7 experiments) | 100-150 hours | 20-30 hours | **4-5x** |
| Domain-specific (4 domains) | 8-12 hours | 2-4 hours | **4x** |

## 📝 Files Created

- `run_phase4_parallel.py` - Main parallel experiment runner
- `run_domain_specific_parallel.py` - Domain-specific parallel runner
- `train_advanced_bge_multigpu.py` - Multi-GPU training script
- `start_parallel_experiments.sh` - Startup script
- `start_domain_parallel.sh` - Domain parallel startup
- `check_parallel_status.sh` - Status checker
- `PARALLEL_EXPERIMENTS_GUIDE.md` - Detailed guide

## 🎯 Recommended Next Steps

Since you have 5 free GPUs, you can:

1. **Start parallel Phase 4 experiments** (recommended):
   ```bash
   ./start_parallel_experiments.sh
   ```
   This will run remaining Phase 4 experiments in parallel.

2. **Or start domain-specific parallel** (faster, smaller scope):
   ```bash
   ./start_domain_parallel.sh
   ```
   This trains all 4 domain models simultaneously.

3. **Both can run together**:
   - Current experiment continues on GPU 0
   - Domain-specific on GPUs 1-4
   - Leaves GPU 5 free for other tasks

## ⚠️ Note

The current sequential experiment on GPU 0 will continue. The parallel system will use GPUs 1-5. Both can run simultaneously without conflicts!

---

**Ready to speed up?** Run `./start_parallel_experiments.sh` or `./start_domain_parallel.sh`!


