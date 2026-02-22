# Phase 5 Experiments - Robust Experiment System

This document describes the robust experiment system with checkpoint/resume, parallel execution, and pause/restart capabilities.

## Features

✅ **Checkpoint/Resume**: Experiments automatically save checkpoints and can resume from where they left off if killed  
✅ **Parallel Execution**: Run multiple experiments simultaneously across all available GPUs  
✅ **Pause/Resume**: Pause experiments at any time and resume later  
✅ **Status Tracking**: Track experiment status, progress, and results  
✅ **Automatic Recovery**: Automatically detect and recover from crashes  

## Quick Start

### 1. Setup Experiments

```bash
# Register all Phase 5 experiments
python setup_phase5_experiments.py
```

This will create 20 experiments:
- 2 Ensemble methods
- 2 Combined techniques (domain-specific + hard negatives)
- 4 Reranking experiments
- 3 Query expansion experiments
- 9 Hybrid retrieval experiments (3 models × 3 alpha values)

### 2. Start Experiments

```bash
# Start all pending experiments in parallel
python manage_experiments.py start

# Or use the runner directly
python experiment_runner.py --action run_parallel
```

### 3. Monitor Status

```bash
# Check status of all experiments
python manage_experiments.py status

# Or use the runner directly
python experiment_runner.py --action status
```

### 4. Pause/Resume

```bash
# Pause an experiment
python manage_experiments.py pause phase5_ensemble_domain_specific

# Resume an experiment
python manage_experiments.py resume phase5_ensemble_domain_specific

# Pause all running experiments
python manage_experiments.py pause_all

# Resume all paused experiments
python manage_experiments.py resume_all
```

### 5. Stop an Experiment

```bash
# Stop an experiment gracefully
python manage_experiments.py stop phase5_ensemble_domain_specific
```

## Experiment Types

### 1. Ensemble Methods

**Script**: `train_ensemble.py`

Combines multiple models using:
- **RRF (Reciprocal Rank Fusion)**: Combines ranked lists using RRF formula
- **Weighted Average**: Combines scores with learned weights

**Experiments**:
- `phase5_ensemble_domain_specific`: RRF of best domain-specific models
- `phase5_ensemble_weighted`: Weighted average of multiple models

### 2. Combined Techniques

**Script**: `train_combined_techniques.py`

Trains domain-specific models with hard negative mining.

**Experiments**:
- `phase5_domain_specific_clapnq_hard_negatives`
- `phase5_domain_specific_govt_hard_negatives`

**Features**:
- Starts from pre-trained multi-domain model
- Mines hard negatives during training
- Supports checkpoint/resume

### 3. Advanced Reranking

**Script**: `train_reranking.py`

Uses cross-encoder models to rerank top-K results from dense retrieval.

**Experiments**:
- `phase5_reranking_clapnq`: Rerank with ClapNQ domain-specific model
- `phase5_reranking_govt`: Rerank with Govt domain-specific model
- `phase5_reranking_cloud`: Rerank with Cloud domain-specific model
- `phase5_reranking_multi_domain`: Rerank with multi-domain model

**Configuration**:
- Initial retrieval: Top 100 results
- Reranking: Top 20 results
- Cross-encoder: `cross-encoder/ms-marco-MiniLM-L-12-v2`

### 4. Query Expansion

**Script**: `train_query_expansion.py`

Expands queries before retrieval to improve recall.

**Experiments**:
- `phase5_query_expansion_clapnq`: Query expansion with ClapNQ model
- `phase5_query_expansion_govt`: Query expansion with Govt model
- `phase5_query_expansion_multi`: Query expansion with multi-domain model

**Note**: Currently uses simple synonym expansion. Can be enhanced with LLM-based expansion.

### 5. Hybrid Retrieval with Learned Weights

**Script**: `train_hybrid_learned.py`

Combines BM25 and dense retrieval with learned weights.

**Experiments**:
- `phase5_hybrid_clapnq_alpha{0.3,0.5,0.7}`: Test different alpha values
- `phase5_hybrid_govt_alpha{0.3,0.5,0.7}`: Test different alpha values
- `phase5_hybrid_multi_alpha{0.3,0.5,0.7}`: Test different alpha values

**Alpha**: Weight for dense retrieval (1-alpha for BM25)

## Experiment Runner Details

### Status File

Experiments are tracked in `experiment_status.json`:
```json
{
  "experiment_name": {
    "experiment_name": "phase5_ensemble_domain_specific",
    "status": "running",
    "gpu_id": 1,
    "pid": 12345,
    "start_time": "2025-12-13T14:00:00",
    "checkpoint_path": "./models/...",
    "config_path": "experiments/retrieval/.../config.json",
    "log_path": "experiments/retrieval/.../training.log"
  }
}
```

### Checkpoint System

- Checkpoints are saved automatically during training
- Location: `{output_path}-checkpoints/`
- Resume: Experiments automatically resume from latest checkpoint if killed
- Manual resume: Use `--resume_from_checkpoint` flag

### GPU Management

- Automatically detects available GPUs
- Distributes experiments across GPUs
- Monitors GPU utilization
- Can manually assign GPU with `--gpu_id`

### Process Management

- Uses `psutil` for process monitoring
- Graceful shutdown with SIGTERM
- Automatic recovery from crashes
- Status updates every time you check

## Advanced Usage

### Manual Experiment Registration

```bash
python experiment_runner.py \
    --action register \
    --experiment my_experiment \
    --config experiments/retrieval/my_experiment/config.json
```

### Manual Experiment Start

```bash
python experiment_runner.py \
    --action start \
    --experiment my_experiment \
    --script train_ensemble.py \
    --gpu_id 0
```

### Update Status

```bash
# Update status of all experiments
python manage_experiments.py update

# Or directly
python experiment_runner.py --action update
```

## Troubleshooting

### Experiment Not Starting

1. Check GPU availability:
   ```bash
   nvidia-smi
   ```

2. Check experiment status:
   ```bash
   python manage_experiments.py status
   ```

3. Check logs:
   ```bash
   tail -f experiments/retrieval/<experiment_name>/training.log
   ```

### Experiment Stuck

1. Pause and resume:
   ```bash
   python manage_experiments.py pause <name>
   python manage_experiments.py resume <name>
   ```

2. Stop and restart:
   ```bash
   python manage_experiments.py stop <name>
   python manage_experiments.py start
   ```

### Checkpoint Issues

1. Check checkpoint directory exists:
   ```bash
   ls -la models/<model_name>-checkpoints/
   ```

2. Manually resume from checkpoint:
   ```bash
   python train_combined_techniques.py \
       --config experiments/retrieval/.../config.json \
       --resume_from_checkpoint models/.../checkpoint-1000
   ```

## Expected Results

Based on literature and similar experiments:

| Experiment Type | Expected Improvement |
|----------------|---------------------|
| Ensemble (RRF) | +3-8% Recall@10 |
| Ensemble (Weighted) | +2-6% Recall@10 |
| Domain-Specific + Hard Negatives | +5-12% Recall@10 |
| Reranking | +5-12% Recall@10 |
| Query Expansion | +3-8% Recall@10 |
| Hybrid (Learned) | +2-7% Recall@10 |

## Next Steps

1. **Monitor Progress**: Use `manage_experiments.py status` regularly
2. **Review Results**: Check `experiments/retrieval/<name>/results.json` when complete
3. **Compare**: Compare against baseline and Phase 4 results
4. **Iterate**: Use best performing techniques for final submission

## Files Created

- `experiment_runner.py`: Core experiment runner with checkpoint/resume
- `train_ensemble.py`: Ensemble method implementation
- `train_combined_techniques.py`: Domain-specific + hard negatives
- `train_reranking.py`: Cross-encoder reranking
- `train_query_expansion.py`: Query expansion
- `train_hybrid_learned.py`: Hybrid retrieval with learned weights
- `setup_phase5_experiments.py`: Setup script for all experiments
- `manage_experiments.py`: Convenient management script
- `experiment_status.json`: Status tracking file

## Notes

- All experiments support checkpoint/resume automatically
- Experiments run in parallel across available GPUs
- Status is saved automatically
- Logs are saved to `experiments/retrieval/<name>/training.log`
- Results are saved to `experiments/retrieval/<name>/results.json`

