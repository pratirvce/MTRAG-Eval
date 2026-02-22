# Experiment Status

## ✅ Experiments Started Successfully!

**Start Time:** $(date)

**Status:** Running in background

## Current Status

The experiments are running through all 3 phases:

### Phase 1: Quick Wins (In Progress)
1. ✅ phase1_baseline - Training...
2. ⏳ phase1_epochs3 - Pending
3. ⏳ phase1_epochs5 - Pending

### Phase 2: Training Improvements (Pending)
4. ⏳ phase2_lr1e5
5. ⏳ phase2_lr5e5
6. ⏳ phase2_cosine_loss
7. ⏳ phase2_augmentation

### Phase 3: Retrieval Strategy (Pending)
8. ⏳ phase3_hybrid
9. ⏳ phase3_reranking
10. ⏳ phase3_hybrid_reranking

## Monitor Progress

### Check Status
```bash
./check_experiment_status.sh
```

### View Live Logs
```bash
tail -f experiments/logs/experiments_*.log
```

### Check Process
```bash
ps aux | grep run_experiments
```

### View Latest Progress
```bash
tail -50 experiments/logs/experiments_*.log | grep -E "(INFO|Experiment|Results)"
```

## Expected Timeline

- **Phase 1**: ~2-4 hours (currently running)
- **Phase 2**: ~4-6 hours
- **Phase 3**: ~6-8 hours
- **Total**: ~12-18 hours

## Results Location

Results will be saved to:
- `experiments/retrieval/{experiment_name}/results.json`
- `experiments/retrieval/experiment_summary.json` (final summary)

## Stop Experiments (if needed)

```bash
pkill -f run_experiments.py
```

## Restart (if needed)

```bash
./start_experiments.sh
```

