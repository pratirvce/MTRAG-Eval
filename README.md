# MTRAG SemEval 2026 Team Repository

Fork of IBM's Multi-Turn Retrieval-Augmented Generation (MTRAG) benchmark.
This repository contains our team's experiments for the **SemEval 2026 Multi-Turn RAG** competition.

## Top 3 Best Models (by nDCG@10)

| Rank | Experiment | nDCG@10 | Recall@10 | Method |
|------|-----------|---------|-----------|--------|
| 1 | `best_paper_large_model_finetuning` | **0.5101** | 0.6221 | BGE-large fine-tuned with advanced training |
| 2 | `ensemble_best_performing` | **0.4576** | 0.5564 | Ensemble of best-performing models |
| 3 | `contrastive_learning_finetuning` | **0.4576** | 0.5564 | Contrastive learning fine-tuning |

**Baseline comparison:** Paper baseline nDCG@10 = 0.3000, SOTA (Elser) = 0.5400

## Structure

- `experiments/retrieval/` - Experiment configs and results (config.json + results.json per experiment)
- `scripts/` - Evaluation and helper scripts from IBM
- `configs/` - Model configuration files
- `md_files/` - Comprehensive experiment documentation and analysis
- Root `*.py` / `*.sh` files - Training, evaluation, and experiment management scripts

## Excluded from Git (too large for GitHub)

The following are excluded via `.gitignore` and not tracked:
- `models/` - Trained model weights (33GB+)
- `corpora/` - Dataset corpus files
- `human/`, `synthetic/` - Original MTRAG datasets
- `data_splits/` - Train/val/test splits
- `checkpoints/` - Training checkpoints
- `submission/` - Submission files

## Setup

```bash
git clone https://github.com/pratirvce/MTRAG-Eval.git
cd MTRAG-Eval
pip install -r experiments/retrieval/requirements.txt
```

## Experiment Results

See `md_files/EXPERIMENT_RESULTS_COMPLETE_SUMMARY.md` for detailed results across all experiments.
See `md_files/ALL_EXPERIMENTS_RESULTS_TABLE.md` for the complete results table.
