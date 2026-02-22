# Phase 4: Advanced Retrieval Experiments

This document describes advanced experiments designed to significantly improve retrieval performance beyond the baseline.

## 🎯 Goal

Improve retrieval scores from current baseline:
- **Baseline BGE (Last Turn)**: Recall@10 = 0.38, nDCG@10 = 0.30
- **Target**: Recall@10 > 0.50, nDCG@10 > 0.40

## 🚀 Advanced Techniques

### 1. Hard Negative Mining

**Problem**: Random negatives in batches are too easy. The model doesn't learn fine-grained distinctions.

**Solution**: Mine "hard negatives" - passages that are semantically similar to the query but are not relevant. This forces the model to learn subtle differences.

**Expected Improvement**: +5-15% Recall@10

**Implementation**: `train_advanced_bge.py`

### 2. Domain-Specific Fine-Tuning

**Problem**: One-size-fits-all model may not optimize for each domain's specific characteristics.

**Solution**: Train separate models for each domain, starting from a multi-domain pre-trained model.

**Expected Improvement**: +3-10% per domain

**Implementation**: `train_domain_specific_bge.py`

### 3. Larger Base Models

**Problem**: BGE-base (110M params) may be insufficient for complex retrieval tasks.

**Solution**: Use BGE-large (335M params) which has stronger representation capabilities.

**Expected Improvement**: +5-12% overall

**Trade-off**: Slower inference, higher memory requirements

### 4. Advanced Loss Functions

**Current**: MultipleNegativesRankingLoss (only positives)

**Alternatives**:
- **CosineSimilarityLoss**: Explicit positive/negative pairs with cosine similarity
- **TripletLoss**: Explicit triplets (query, positive, negative) with margin

**Expected Improvement**: +2-7% depending on data characteristics

## 📋 Experiment Configurations

### Experiment 1: Hard Negatives with CosineSimilarityLoss
```bash
python train_advanced_bge.py \
  --experiment_name phase4_hard_negatives_cosine \
  --num_hard_negatives 3 \
  --loss_function CosineSimilarityLoss \
  --epochs 5 \
  --batch_size 32 \
  --learning_rate 2e-5
```

### Experiment 2: Hard Negatives with TripletLoss
```bash
python train_advanced_bge.py \
  --experiment_name phase4_hard_negatives_triplet \
  --num_hard_negatives 3 \
  --loss_function TripletLoss \
  --epochs 5 \
  --batch_size 32 \
  --learning_rate 2e-5
```

### Experiment 3: More Hard Negatives (5 per positive)
```bash
python train_advanced_bge.py \
  --experiment_name phase4_hard_negatives_5neg \
  --num_hard_negatives 5 \
  --loss_function CosineSimilarityLoss \
  --epochs 5 \
  --batch_size 24 \
  --learning_rate 2e-5
```

### Experiment 4: BGE-Large Model
```bash
python train_advanced_bge.py \
  --experiment_name phase4_bge_large \
  --base_model BAAI/bge-large-en-v1.5 \
  --num_hard_negatives 3 \
  --epochs 3 \
  --batch_size 16 \
  --learning_rate 1e-5
```

### Experiment 5: Domain-Specific Models
```bash
# Train all domain-specific models
python train_domain_specific_bge.py \
  --domain all \
  --epochs 7 \
  --learning_rate 1e-5 \
  --use_pretrained_multi_domain \
  --pretrained_multi_domain_path ./models/phase1_epochs5

# Or train individual domain
python train_domain_specific_bge.py \
  --domain fiqa \
  --epochs 7 \
  --learning_rate 1e-5 \
  --use_pretrained_multi_domain
```

## 🏃 Running All Experiments

### Option 1: Run All Phase 4 Experiments
```bash
python run_phase4_experiments.py --experiment all
```

### Option 2: Run Specific Experiment
```bash
python run_phase4_experiments.py --experiment phase4_hard_negatives_cosine
```

### Option 3: List Available Experiments
```bash
python run_phase4_experiments.py --list
```

## 📊 Evaluation

### Evaluate Single Model
```bash
python evaluate_advanced_models.py \
  --model_path ./models/phase4_hard_negatives_cosine \
  --output results/phase4_hard_negatives_cosine.json
```

### Evaluate Domain-Specific Models
```bash
python evaluate_advanced_models.py --domain_specific
```

## 🔄 Workflow

1. **Baseline**: Ensure Phase 1 baseline is trained (`phase1_epochs5`)
2. **Hard Negatives**: Run hard negative mining experiments
3. **Domain-Specific**: Train domain-specific models starting from baseline
4. **Larger Models**: Try BGE-large if resources allow
5. **Evaluate**: Compare all results

## 📈 Expected Results

Based on research and similar experiments:

| Experiment | Expected Recall@10 | Expected nDCG@10 |
|------------|-------------------|------------------|
| Baseline (BGE-base) | 0.38 | 0.30 |
| Hard Negatives + Cosine | 0.43-0.48 | 0.35-0.40 |
| Hard Negatives + Triplet | 0.42-0.47 | 0.34-0.39 |
| Domain-Specific | 0.45-0.52 | 0.37-0.43 |
| BGE-Large + Hard Neg | 0.48-0.55 | 0.40-0.45 |
| **Best Combination** | **0.50-0.58** | **0.42-0.48** |

## ⚠️ Important Notes

1. **Hard Negative Mining**: First run uses base model to mine negatives. This takes time but significantly improves training quality.

2. **Domain-Specific**: Train on all domains first, then fine-tune per domain. This leverages multi-domain knowledge.

3. **Memory**: BGE-large requires ~6GB GPU memory. Reduce batch size if OOM.

4. **Training Time**: 
   - Hard negatives: ~2-4 hours (including mining time)
   - Domain-specific: ~1-2 hours per domain
   - BGE-large: ~4-6 hours

5. **Best Practice**: Always use validation during training to prevent overfitting.

## 🎓 Next Steps After Phase 4

1. **Ensemble Methods**: Combine multiple models' predictions
2. **Query Expansion**: Use LLMs to expand queries with context
3. **Reranking**: Fine-tune cross-encoders for reranking
4. **Hybrid Retrieval**: Combine dense + sparse + domain-specific models
5. **Multi-Stage Training**: Progressive fine-tuning on increasingly hard examples

## 📝 Results Tracking

Results are saved to:
- Training logs: `experiments/retrieval/{experiment_name}/training.log`
- Configs: `experiments/retrieval/{experiment_name}/config.json`
- Evaluation: Run `evaluate_advanced_models.py` after training

Compare results with baseline:
```bash
python evaluate_advanced_models.py --model_path ./models/phase1_epochs5
python evaluate_advanced_models.py --model_path ./models/phase4_hard_negatives_cosine
```

## 🐛 Troubleshooting

### Out of Memory
- Reduce batch size: `--batch_size 16`
- Reduce hard negatives: `--num_hard_negatives 2`
- Use gradient accumulation (modify script)

### Slow Hard Negative Mining
- Hard negative mining happens once at the start
- Consider caching mined negatives for future runs
- Reduce corpus size for mining (sample subset)

### Model Not Improving
- Check if validation loss is decreasing
- Try different learning rates (1e-5, 2e-5, 5e-5)
- Increase epochs
- Check data quality

## 📚 References

- Hard Negative Mining: [Xiong et al., 2020](https://arxiv.org/abs/2010.06467)
- Domain Adaptation: [Gururangan et al., 2020](https://aclanthology.org/2020.emnlp-main.744/)
- BGE Models: [BGE Repository](https://github.com/FlagOpen/FlagEmbedding)

