# Experiment Tracker

This document tracks all retrieval improvement experiments.

## How to Run Experiments

### Option 1: Run in Background (Recommended)
```bash
./run_experiments_background.sh
```

This will:
- Start all experiments in the background
- Log everything to `experiments/logs/experiments_TIMESTAMP.log`
- Save PID for tracking

### Option 2: Run Directly
```bash
python run_experiments.py
```

### Monitor Progress
```bash
# Watch log file
tail -f experiments/logs/experiments_*.log

# Check if running
ps -p $(cat experiments/logs/current_experiment.pid)

# Stop experiments
kill $(cat experiments/logs/current_experiment.pid)
```

## Experiment Phases

### Phase 1: Quick Wins
1. **phase1_baseline** - Original baseline (1 epoch, batch 16, no validation)
2. **phase1_epochs3** - 3 epochs, batch 32, with validation, proper train/val split
3. **phase1_epochs5** - 5 epochs, batch 32, with validation, proper train/val split

### Phase 2: Training Improvements
4. **phase2_lr1e5** - Learning rate 1e-5
5. **phase2_lr5e5** - Learning rate 5e-5
6. **phase2_cosine_loss** - CosineSimilarityLoss instead of MultipleNegativesRankingLoss
7. **phase2_augmentation** - With data augmentation

### Phase 3: Retrieval Strategy
8. **phase3_hybrid** - Hybrid retrieval (dense + BM25)
9. **phase3_reranking** - Dense retrieval + reranking
10. **phase3_hybrid_reranking** - Hybrid + reranking

## Results Location

- **Configs**: `experiments/retrieval/{experiment_name}/config.json`
- **Training logs**: `experiments/retrieval/{experiment_name}/training.log`
- **Evaluation logs**: `experiments/retrieval/{experiment_name}/evaluation.log`
- **Results**: `experiments/retrieval/{experiment_name}/results.json`
- **Summary**: `experiments/retrieval/experiment_summary.json`
- **Models**: `./models/{experiment_name}/`

## Results Summary

Results will be automatically recorded in `experiments/retrieval/experiment_summary.json` after all experiments complete.

