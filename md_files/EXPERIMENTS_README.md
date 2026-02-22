# Retrieval Improvement Experiments

This directory contains scripts to systematically improve retrieval performance through multiple phases of experiments.

## Quick Start

### Start All Experiments
```bash
./start_experiments.sh
```

Or manually:
```bash
source venv/bin/activate
nohup python run_experiments.py > experiments/logs/experiments_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

### Check Status
```bash
./check_experiment_status.sh
```

### View Results
```bash
python summarize_results.py
```

## Experiment Phases

### Phase 1: Quick Wins ✅
1. **phase1_baseline** - Original configuration (baseline)
2. **phase1_epochs3** - 3 epochs, batch 32, validation, proper splits
3. **phase1_epochs5** - 5 epochs, batch 32, validation, proper splits

**Expected improvements:** +10-20% over baseline

### Phase 2: Training Improvements ✅
4. **phase2_lr1e5** - Learning rate 1e-5
5. **phase2_lr5e5** - Learning rate 5e-5  
6. **phase2_cosine_loss** - CosineSimilarityLoss
7. **phase2_augmentation** - With data augmentation

**Expected improvements:** +10-15% additional

### Phase 3: Retrieval Strategy ✅
8. **phase3_hybrid** - Hybrid retrieval (dense + BM25)
9. **phase3_reranking** - Dense + reranking with cross-encoder
10. **phase3_hybrid_reranking** - Hybrid + reranking

**Expected improvements:** +15-25% additional

## File Structure

```
mt-rag-benchmark/
├── train_improved_bge.py          # Improved training script
├── evaluate_improved_bge.py       # Improved evaluation script
├── run_experiments.py             # Main experiment runner
├── summarize_results.py           # Results summarizer
├── start_experiments.sh           # Start script
├── check_experiment_status.sh     # Status checker
├── experiments/
│   ├── retrieval/                 # Experiment results
│   │   ├── {experiment_name}/
│   │   │   ├── config.json
│   │   │   ├── training.log
│   │   │   ├── evaluation.log
│   │   │   └── results.json
│   │   └── experiment_summary.json
│   └── logs/                      # Experiment logs
├── models/                        # Trained models
│   └── {experiment_name}/
└── EXPERIMENTS_README.md          # This file
```

## Monitoring Experiments

### View Live Logs
```bash
tail -f experiments/logs/experiments_*.log
```

### Check if Running
```bash
ps -p $(cat experiments/logs/current_experiment.pid)
```

### Stop Experiments
```bash
kill $(cat experiments/logs/current_experiment.pid)
```

## Results

Results are automatically saved to:
- Individual: `experiments/retrieval/{experiment_name}/results.json`
- Summary: `experiments/retrieval/experiment_summary.json`

Each result file contains:
- Domain-specific metrics (Recall@5, Recall@10, nDCG@5, nDCG@10)
- Average metrics across all domains
- Comparison with baseline

## Expected Timeline

- **Phase 1**: ~2-4 hours (3 experiments × ~1 hour each)
- **Phase 2**: ~4-6 hours (4 experiments × ~1-1.5 hours each)
- **Phase 3**: ~6-8 hours (3 experiments × ~2-3 hours each)

**Total**: ~12-18 hours for all experiments

## Troubleshooting

### Out of Memory
- Reduce batch size in config
- Use gradient accumulation

### Training Fails
- Check `experiments/retrieval/{experiment}/training.log`
- Verify data files exist

### Evaluation Fails
- Check `experiments/retrieval/{experiment}/evaluation.log`
- Verify model was saved correctly

### BM25 Not Available
- Hybrid retrieval will automatically fall back to dense
- Install rank-bm25: `pip install rank-bm25`

## Next Steps After Experiments

1. Review results: `python summarize_results.py`
2. Identify best configuration
3. Train final model with best config
4. Evaluate on test set
5. Compare with baseline

