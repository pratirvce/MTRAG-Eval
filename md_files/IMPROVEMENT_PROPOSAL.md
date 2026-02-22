# 🚀 Experiment Improvement Proposal: Targeting nDCG@10 > 0.60

## 📊 Current Best Performance Analysis

### Top Performers:
1. **best_paper_large_model_finetuning**: 0.5101 nDCG@10
   - Model: BGE-large-en-v1.5
   - Epochs: 3
   - Batch size: 8
   - Learning rate: 5e-5 (default)
   - Loss: MultipleNegativesRankingLoss

2. **tier1_contrastive_learning**: 0.4576 nDCG@10
   - Model: BGE-base-en-v1.5
   - Epochs: 3
   - Batch size: 16
   - Learning rate: 2e-5

3. **phase5_query_expansion_govt**: 0.4515 nDCG@10
   - Query expansion on govt domain

4. **best_paper_adversarial_curriculum**: 0.4464 nDCG@10
   - Adversarial curriculum learning

5. **phase5_ensemble_domain_specific**: 0.4434 nDCG@10
   - Domain-specific ensemble

---

## 🎯 Strategy to Reach nDCG@10 > 0.60

### Gap Analysis:
- Current best: 0.5101
- Target: > 0.60
- Gap: ~0.09 (17.6% improvement needed)

### Key Insights:
1. **Large models work**: BGE-large (0.5101) > BGE-base (0.4576) = +11.5%
2. **Training duration matters**: Only 3 epochs used, could increase
3. **Ensemble helps**: Multiple methods at 0.45+ can be combined
4. **Query expansion helps**: +0.03-0.04 improvement
5. **Adversarial/curriculum learning**: Consistent improvements

---

## 🔬 Proposed Experiments

### Experiment 1: Ultra-Large Model with Extended Training
**Expected: 0.55-0.60 nDCG@10**

**Combines:**
- BGE-large-en-v1.5 (or BGE-v2-large if available)
- Extended training: 5-7 epochs (vs current 3)
- Optimized learning rate schedule: cosine with warmup
- Larger effective batch size: batch_size=4, gradient_accumulation=4 (effective=16)
- Temperature-scaled contrastive loss

**Hyperparameter Changes:**
```python
{
    "model_path": "BAAI/bge-large-en-v1.5",  # or bge-v2-large
    "epochs": 7,  # Increased from 3
    "batch_size": 4,  # Reduced to allow gradient accumulation
    "gradient_accumulation_steps": 4,  # Effective batch = 16
    "learning_rate": 1e-5,  # Slightly lower for stability
    "warmup_steps": 500,  # Add warmup
    "lr_scheduler": "cosine",  # Better than linear
    "loss_scale": 25.0,  # Increased from 20.0
    "weight_decay": 0.01,  # Add regularization
    "max_grad_norm": 1.0
}
```

**Rationale:**
- More epochs = better convergence
- Gradient accumulation = stable training with larger effective batch
- Cosine LR = better final performance
- Warmup = stable early training
- Higher loss scale = sharper decision boundaries

---

### Experiment 2: Large Model + Adversarial Curriculum + Query Expansion
**Expected: 0.58-0.65 nDCG@10**

**Combines:**
- Large model finetuning (0.5101)
- Adversarial curriculum learning (0.4464)
- Query expansion (0.4515)
- Multi-stage retrieval with cross-encoder reranking

**Pipeline:**
1. Stage 1: Dense retrieval with BGE-large (top-100)
2. Stage 2: Query expansion + sparse retrieval (BM25, top-100)
3. Stage 3: Hybrid fusion (learned weights)
4. Stage 4: Cross-encoder reranking (top-50 → top-10)
5. Stage 5: Adversarial hard negative mining during training

**Hyperparameters:**
```python
{
    "dense_model": "BAAI/bge-large-en-v1.5",
    "dense_epochs": 5,
    "dense_batch_size": 4,
    "dense_gradient_accumulation": 4,
    "cross_encoder_model": "cross-encoder/ms-marco-MiniLM-L-12-v2",
    "cross_encoder_epochs": 3,
    "cross_encoder_batch_size": 16,
    "query_expansion": "embedding_based",  # Use embeddings for expansion
    "fusion_method": "learned_adaptive",  # Domain-aware fusion
    "adversarial_ratio": 0.3,  # 30% hard negatives
    "curriculum_schedule": "exponential"  # Start easy, get harder
}
```

---

### Experiment 3: Enhanced Ensemble of Best Methods
**Expected: 0.60-0.68 nDCG@10**

**Combines:**
- Large model finetuning (0.5101)
- Contrastive learning variant (0.4576)
- Query expansion (0.4515)
- Adversarial curriculum (0.4464)
- Domain-specific models (0.4434)

**Ensemble Strategy:**
- Train 5 specialized models:
  1. Large model (all domains)
  2. Domain-specific large models (one per domain)
  3. Contrastive learning with hard negatives
  4. Adversarial curriculum model
  5. Query-expanded model

- Ensemble fusion:
  - Weighted average based on domain
  - Learned meta-weights per domain
  - Cross-encoder for final reranking

**Hyperparameters:**
```python
{
    "models": [
        {"type": "large_all", "epochs": 7, "batch_size": 4, "grad_accum": 4},
        {"type": "large_domain_specific", "epochs": 5, "batch_size": 8},
        {"type": "contrastive_hardneg", "epochs": 5, "hard_neg_ratio": 0.4},
        {"type": "adversarial_curriculum", "epochs": 5, "adv_ratio": 0.3},
        {"type": "query_expanded", "epochs": 5, "expansion_method": "embedding"}
    ],
    "ensemble_method": "learned_weighted",
    "meta_learner": "domain_adaptive",
    "final_reranker": "cross_encoder"
}
```

---

### Experiment 4: Optimized Hybrid with Advanced Training
**Expected: 0.62-0.70 nDCG@10**

**Improves existing:** `task_a_optimized_hybrid_reranking`

**Key Improvements:**
1. **Better base model training:**
   - BGE-large with 7 epochs
   - Temperature-scaled loss (temp=0.05)
   - Hard negative mining (top-1000, select hardest)
   - Gradient accumulation for stability

2. **Advanced cross-encoder:**
   - Fine-tune on MTRAG with 5 epochs
   - Use larger model: `cross-encoder/ms-marco-MiniLM-L-6-v2` → `cross-encoder/ms-marco-electra-base`
   - Batch size: 32 with gradient accumulation

3. **Smarter fusion:**
   - Domain-specific fusion weights
   - Query-type adaptation (short vs long queries)
   - Confidence-based weighting

4. **Better query expansion:**
   - Embedding-based expansion (find similar queries)
   - Domain-specific expansion vocabularies
   - Multi-step expansion (expand → retrieve → expand again)

**Hyperparameters:**
```python
{
    "dense_model": {
        "base": "BAAI/bge-large-en-v1.5",
        "epochs": 7,
        "batch_size": 4,
        "gradient_accumulation": 4,
        "learning_rate": 1e-5,
        "warmup_steps": 500,
        "lr_scheduler": "cosine",
        "loss_scale": 25.0,
        "temperature": 0.05,  # Sharper contrastive learning
        "hard_negatives": {
            "top_k": 1000,
            "select_hardest": 100,
            "mining_strategy": "semantic_similarity"
        }
    },
    "cross_encoder": {
        "base": "cross-encoder/ms-marco-electra-base",
        "epochs": 5,
        "batch_size": 16,
        "gradient_accumulation": 2,
        "learning_rate": 2e-5
    },
    "fusion": {
        "method": "learned_domain_adaptive",
        "hidden_dim": 256,
        "domain_embedding_dim": 64,
        "query_feature_dim": 32
    },
    "query_expansion": {
        "method": "embedding_based",
        "expansion_ratio": 1.5,  # 50% more terms
        "domain_specific": True,
        "multi_step": True
    }
}
```

---

### Experiment 5: Multi-Stage Hierarchical with Large Models
**Expected: 0.60-0.68 nDCG@10**

**Combines:**
- Large model at each stage
- Hierarchical retrieval (coarse → fine)
- Cross-encoder reranking
- Learned routing between stages

**Pipeline:**
1. **Stage 1 (Coarse)**: BGE-large, retrieve top-500
2. **Stage 2 (Medium)**: Domain-specific BGE-large, retrieve top-100
3. **Stage 3 (Fine)**: Cross-encoder, rerank top-50 → top-10
4. **Routing**: Learned which queries need which stages

**Hyperparameters:**
```python
{
    "stage1": {
        "model": "BAAI/bge-large-en-v1.5",
        "top_k": 500,
        "epochs": 5
    },
    "stage2": {
        "model": "domain_specific_bge_large",
        "top_k": 100,
        "epochs": 5
    },
    "stage3": {
        "model": "cross-encoder/ms-marco-electra-base",
        "top_k": 10,
        "epochs": 3
    },
    "routing": {
        "method": "learned_classifier",
        "features": ["query_length", "domain", "complexity_score"]
    }
}
```

---

## 🔧 Hyperparameter Optimization Strategy

### For Large Model Training:

**Current (0.5101):**
- Epochs: 3
- Batch size: 8
- LR: 5e-5 (default)
- Warmup: 0
- LR schedule: linear

**Optimized (Target: 0.55-0.60):**
- Epochs: 7 (more training)
- Batch size: 4 + grad_accum=4 (effective=16, more stable)
- LR: 1e-5 (lower for stability with more epochs)
- Warmup: 500 steps (10% of training)
- LR schedule: cosine (better convergence)
- Loss scale: 25.0 (sharper boundaries)
- Weight decay: 0.01 (regularization)
- Temperature: 0.05 (contrastive sharpness)

### For Cross-Encoder:

**Current:**
- Epochs: 3
- Batch size: 16
- LR: 2e-5

**Optimized:**
- Epochs: 5
- Batch size: 16 + grad_accum=2 (effective=32)
- LR: 2e-5 (keep)
- Use larger model: electra-base instead of MiniLM

### For Ensemble:

**Current:**
- Simple averaging

**Optimized:**
- Learned weights per domain
- Meta-learner for adaptive weighting
- Cross-encoder final reranking

---

## 📈 Expected Performance Gains

| Experiment | Base Components | Expected nDCG@10 | Improvement |
|------------|----------------|------------------|-------------|
| Current Best | Large model (3 epochs) | 0.5101 | Baseline |
| Exp 1 | Large model (7 epochs, optimized) | 0.55-0.60 | +8-18% |
| Exp 2 | Large + Adversarial + Query Expansion | 0.58-0.65 | +14-27% |
| Exp 3 | Enhanced Ensemble | 0.60-0.68 | +18-33% |
| Exp 4 | Optimized Hybrid (improved) | 0.62-0.70 | +22-37% |
| Exp 5 | Multi-Stage Hierarchical | 0.60-0.68 | +18-33% |

---

## 🎯 Recommended Priority Order

1. **Experiment 4** (Optimized Hybrid): Highest expected gain, builds on existing work
2. **Experiment 1** (Ultra-Large Extended Training): Simplest, likely +0.04-0.09 gain
3. **Experiment 2** (Large + Adversarial + Query): Good combination of proven methods
4. **Experiment 3** (Enhanced Ensemble): Highest ceiling but most complex
5. **Experiment 5** (Multi-Stage Hierarchical): Good if others don't reach target

---

## 🛠️ Implementation Notes

### Key Techniques to Implement:

1. **Temperature-Scaled Contrastive Loss:**
   ```python
   # Instead of standard cosine similarity
   similarity = cosine_sim(query_emb, doc_emb) / temperature
   # Lower temperature (0.05) = sharper boundaries
   ```

2. **Hard Negative Mining:**
   ```python
   # Retrieve top-1000, select hardest negatives
   hard_negatives = select_hardest_negatives(
       candidates=top_1000,
       query=query_emb,
       num_select=100,
       strategy="semantic_similarity"
   )
   ```

3. **Gradient Accumulation:**
   ```python
   # Allows larger effective batch size
   for i, batch in enumerate(dataloader):
       loss = model(batch) / accumulation_steps
       loss.backward()
       if (i + 1) % accumulation_steps == 0:
           optimizer.step()
           optimizer.zero_grad()
   ```

4. **Cosine LR Schedule with Warmup:**
   ```python
   scheduler = CosineAnnealingLR(
       optimizer,
       T_max=total_steps,
       eta_min=1e-7
   )
   # Warmup for first 10% of steps
   ```

5. **Domain-Adaptive Fusion:**
   ```python
   # Learn different fusion weights per domain
   domain_weights = fusion_model(domain_id, query_features)
   fused_score = domain_weights[0] * dense + domain_weights[1] * sparse
   ```

---

## ✅ Next Steps

1. Implement Experiment 1 (simplest, high impact)
2. Implement Experiment 4 (builds on existing optimized hybrid)
3. Evaluate and iterate based on results
4. If needed, implement Experiment 3 (ensemble) for maximum performance

---

## 📝 Notes

- All experiments should be Task A compliant (retrieval only, no text generation)
- Query expansion must use embeddings/thesaurus, not LLM generation
- Cross-encoders are allowed (they score, not generate)
- Ensemble methods are allowed (they combine scores, not generate)

