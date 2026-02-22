# Quick Start: Phase 4 Advanced Experiments

This guide helps you quickly start running advanced experiments to improve retrieval scores.

## Prerequisites

1. ✅ Phase 1 baseline model trained (`./models/phase1_epochs5`)
2. ✅ GPU available (recommended)
3. ✅ Data splits prepared (`data_splits/retrieval_tasks/`)

## 🚀 Quick Start (Recommended Order)

### Step 1: Hard Negative Mining Experiment (Highest Impact)

This is the most impactful improvement:
```bash
python train_advanced_bge.py \
  --experiment_name phase4_hard_negatives_cosine \
  --num_hard_negatives 3 \
  --loss_function CosineSimilarityLoss \
  --epochs 5 \
  --batch_size 32 \
  --learning_rate 2e-5 \
  --use_validation \
  --use_data_splits
```

**Time**: ~2-4 hours  
**Expected**: Recall@10: 0.43-0.48 (vs 0.38 baseline)

### Step 2: Domain-Specific Fine-Tuning (For Best Domain Performance)

Train models optimized for each domain:
```bash
# Train all domains
python train_domain_specific_bge.py \
  --domain all \
  --epochs 7 \
  --learning_rate 1e-5 \
  --use_pretrained_multi_domain \
  --pretrained_multi_domain_path ./models/phase1_epochs5
```

**Time**: ~1-2 hours per domain (4-8 hours total)  
**Expected**: Domain-specific improvements of +3-10%

### Step 3: Evaluate and Compare

```bash
# Evaluate hard negatives model
python evaluate_advanced_models.py \
  --model_path ./models/phase4_hard_negatives_cosine \
  --output results/phase4_hard_negatives_results.json

# Evaluate domain-specific models
python evaluate_advanced_models.py --domain_specific
```

## 📊 Run All Experiments Automatically

```bash
# Run all Phase 4 experiments (takes ~12-16 hours)
python run_phase4_experiments.py --experiment all

# Or run specific experiment
python run_phase4_experiments.py --experiment phase4_hard_negatives_cosine
```

## 🎯 Recommended Sequence for Best Results

1. **First**: Hard negatives with CosineSimilarityLoss
2. **Second**: Domain-specific models (especially for worst-performing domains like FiQA)
3. **Third**: Try BGE-large if you have GPU memory (8GB+)
4. **Fourth**: Ensemble or combine approaches

## 💡 Tips

- **Start Small**: Run one experiment first to ensure everything works
- **Monitor Training**: Check `experiments/retrieval/{exp_name}/training.log`
- **Use Validation**: Always enable `--use_validation` to catch overfitting
- **Save Checkpoints**: Models are saved automatically during training

## 🔍 Monitoring Progress

```bash
# Watch training log
tail -f experiments/retrieval/phase4_hard_negatives_cosine/training.log

# Check GPU usage
nvidia-smi -l 1
```

## 📈 Expected Timeline

| Experiment | Time | Priority |
|------------|------|----------|
| Hard Negatives (Cosine) | 2-4h | ⭐⭐⭐ Highest |
| Domain-Specific (All) | 4-8h | ⭐⭐ High |
| Hard Negatives (Triplet) | 2-4h | ⭐ Medium |
| BGE-Large | 4-6h | ⭐ Medium |

## 🎓 Next Steps

After Phase 4 experiments complete:
1. Compare all results
2. Identify best model per domain
3. Consider ensemble methods
4. Move to Phase 5: Query expansion and reranking

## ❓ Troubleshooting

### Out of Memory
```bash
# Reduce batch size
--batch_size 16

# Reduce hard negatives
--num_hard_negatives 2
```

### Training Too Slow
- Hard negative mining only happens once (at start)
- Consider reducing `num_hard_negatives` for faster training

### No Improvement
- Check validation metrics are improving
- Try different learning rates (1e-5, 2e-5, 5e-5)
- Ensure you're using proper train/val/test splits

For detailed information, see [PHASE4_ADVANCED_EXPERIMENTS.md](PHASE4_ADVANCED_EXPERIMENTS.md)

