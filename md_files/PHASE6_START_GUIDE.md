# Phase 6 ACL Experiments - Quick Start Guide

## 🚀 Starting Experiments

### Option 1: Use Experiment Manager (Recommended)

```bash
# Start all experiments in parallel
python manage_phase6_experiments.py --config phase6_experiments.json

# Check status
python manage_phase6_experiments.py --config phase6_experiments.json --status

# Stop a specific experiment
python manage_phase6_experiments.py --config phase6_experiments.json --stop phase6_cross_encoder_finetuned_ensemble

# Stop all experiments
python manage_phase6_experiments.py --config phase6_experiments.json --stop-all
```

### Option 2: Run Individual Experiments

```bash
# Cross-encoder fine-tuning (CRITICAL - Highest Priority)
python train_cross_encoder_finetuned.py \
    --config experiments/retrieval/phase6_cross_encoder_finetuned_ensemble/config.json \
    --gpu_id 0 \
    --resume

# Multi-stage retrieval
python train_multistage_retrieval.py \
    --config experiments/retrieval/phase6_multistage_2stage/config.json \
    --gpu_id 1 \
    --resume

# LLM query expansion
python train_llm_query_expansion.py \
    --config experiments/retrieval/phase6_llm_query_expansion_gpt4_multi/config.json \
    --gpu_id 2 \
    --resume
```

## 📊 Monitoring Experiments

### Check GPU Usage
```bash
watch -n 1 nvidia-smi
```

### Check Experiment Logs
```bash
# Cross-encoder
tail -f experiments/retrieval/phase6_cross_encoder_finetuned_ensemble/training.log

# Multi-stage
tail -f experiments/retrieval/phase6_multistage_2stage/training.log

# LLM expansion
tail -f experiments/retrieval/phase6_llm_query_expansion_gpt4_multi/training.log
```

### Check Experiment Status
```bash
python manage_phase6_experiments.py --config phase6_experiments.json --status
```

## ⏸️ Pausing and Resuming

### Pause an Experiment
Press `Ctrl+C` in the terminal running the experiment, or:
```bash
python manage_phase6_experiments.py --config phase6_experiments.json --stop <experiment_name>
```

### Resume an Experiment
Simply restart with `--resume` flag (enabled by default):
```bash
python train_cross_encoder_finetuned.py \
    --config experiments/retrieval/phase6_cross_encoder_finetuned_ensemble/config.json \
    --gpu_id 0 \
    --resume
```

The script will automatically:
- Load checkpoint if available
- Continue from last saved epoch/step
- Skip already completed domains (for evaluation scripts)

## 🎯 Experiment Priorities

1. **phase6_cross_encoder_finetuned_ensemble** (Priority 1)
   - Highest impact on nDCG
   - Expected: 0.49-0.52 nDCG@10
   - Time: 3-5 days

2. **phase6_multistage_2stage** (Priority 2)
   - Novel contribution
   - Expected: 0.48-0.51 nDCG@10
   - Time: 2-4 days

3. **phase6_llm_query_expansion_gpt4_multi** (Priority 3)
   - Modern approach
   - Expected: 0.48-0.51 nDCG@10
   - Time: 2-3 days

## 📁 Checkpoint Locations

- **Cross-encoder**: `models/phase6_cross_encoder_finetuned_ensemble/checkpoints/`
- **Multi-stage**: `experiments/retrieval/phase6_multistage_2stage/checkpoints/`
- **LLM expansion**: `experiments/retrieval/phase6_llm_query_expansion_gpt4_multi/checkpoints/`

## 🔧 Troubleshooting

### Out of Memory (OOM)
- Reduce `batch_size` in config.json
- Use smaller model (e.g., L-6 instead of L-12)

### Process Not Resuming
- Check checkpoint files exist
- Verify checkpoint_info.json is valid
- Try `--no-resume` to start fresh

### GPU Not Available
- Check GPU availability: `nvidia-smi`
- Verify CUDA_VISIBLE_DEVICES is set correctly
- Check if other processes are using GPU

## 📈 Expected Results

After all experiments complete:
- **Best Case**: 0.55 nDCG@10 (beats Elser's 0.54)
- **Realistic**: 0.52 nDCG@10 (competitive, top 3)
- **Conservative**: 0.50 nDCG@10 (strong improvement)

## 🎓 Next Steps

1. Wait for cross-encoder training to complete
2. Run multi-stage evaluation using fine-tuned cross-encoder
3. Combine all techniques in final ensemble
4. Analyze results and prepare for paper submission

