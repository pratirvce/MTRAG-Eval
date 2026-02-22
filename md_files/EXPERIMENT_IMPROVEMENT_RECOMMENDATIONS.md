# Experiment Improvement Recommendations for Low nDCG@10 Scores

**Generated:** 2025-12-20  
**Target:** Improve experiments with nDCG@10 < 0.20 to reach > 0.40

---

## Executive Summary

This document provides specific, actionable recommendations to improve nDCG@10 scores for underperforming experiments. The best performing experiment (`best_paper_large_model_finetuning`) achieves **0.5101 nDCG@10**, while many experiments are stuck at **0.1796** or lower, suggesting training or implementation issues.

---

## Critical Issues Identified

### 1. **Experiments Scoring Exactly 0.1796** (Multiple experiments)
**Affected:** `tier1_causal_inference`, `tier1_learned_indices`, `tier1_differentiable_retrieval`, `tier1_curriculum_contrastive`, `tier1_mixture_experts`, `tier1_multi_turn_state_tracking`, `tier1_uncertainty_aware`, `tier1_foundation_distillation`

**Problem:** All these experiments have identical scores (0.1796 nDCG@10, 0.2420 Recall@10), which strongly suggests:
- They're not actually training (using the same baseline model)
- Training is failing silently
- Evaluation is using the wrong model
- Checkpoint/resume logic is broken

**Immediate Fixes:**
1. **Verify training is actually happening:**
   - Check training logs for loss values decreasing
   - Verify model weights are being updated (compare before/after training)
   - Ensure `model.train()` is called and gradients are computed

2. **Fix checkpoint/resume logic:**
   - Many experiments may be loading checkpoints that skip training
   - Add explicit `--no-resume` flag for fresh training
   - Verify saved models are actually being used in evaluation

3. **Ensure proper model saving:**
   - Save model after each domain training
   - Verify evaluation uses the trained model, not the base model
   - Add logging to show which model is being evaluated

---

## Category-Specific Improvements

### 2. **Multitask Retrieval Experiments** (nDCG@10: 0.0185-0.0261)
**Affected:** `tier1_multitask_retrieval`, `best_paper_multitask_retrieval`

**Problems:**
- Extremely low scores suggest complete failure
- Likely not learning task-specific representations
- Multi-task loss may be poorly balanced

**Improvements:**
1. **Better Loss Balancing:**
   ```python
   # Current: Simple weighted sum
   # Improved: Dynamic task weighting (Uncertainty Weighting)
   loss = Σ(1/(2*σ²_i) * L_i + log(σ²_i))
   ```

2. **Task-Specific Heads:**
   - Use separate projection layers for each task
   - Share only the base encoder
   - Add task-specific fine-tuning

3. **Curriculum Learning:**
   - Start with easier tasks, gradually add harder ones
   - Use task difficulty sampling

4. **Increase Training:**
   - Current: 3 epochs → Increase to 10-15 epochs
   - Use larger batch size with gradient accumulation
   - Add warmup steps (10% of total steps)

---

### 3. **Graph-Aware Retrieval** (nDCG@10: 0.1738)
**Affected:** `tier1_graph_aware_retrieval`, `best_paper_graph_aware_retrieval`

**Problems:**
- Graph construction may be too sparse or noisy
- Graph neural network may not be learning effectively
- Graph features may not be properly integrated

**Improvements:**
1. **Better Graph Construction:**
   - Use multiple similarity thresholds (not just top-k)
   - Add semantic edges (not just similarity-based)
   - Include temporal/conversation context in edges

2. **Improved GNN Architecture:**
   - Use Graph Attention Network (GAT) instead of simple GCN
   - Add residual connections
   - Use multi-hop reasoning (2-3 layers)

3. **Graph-Enhanced Features:**
   - Combine graph embeddings with dense embeddings
   - Use graph structure for hard negative mining
   - Add graph-based query expansion

4. **Training Improvements:**
   - Pre-train graph encoder separately
   - Use contrastive loss with graph-augmented negatives
   - Increase epochs from 3 to 5-7

---

### 4. **Temporal Memory Experiments** (nDCG@10: 0.1484)
**Affected:** `best_paper_temporal_memory`, `best_paper_temporal_memory_fixed`

**Problems:**
- Memory mechanism may not be capturing relevant history
- Attention over memory may be too weak
- Memory updates may be too aggressive/too conservative

**Improvements:**
1. **Better Memory Architecture:**
   - Use key-value memory (not just embeddings)
   - Add memory gating (forget/update gates)
   - Implement memory compression for long conversations

2. **Improved Attention:**
   - Use multi-head attention over memory
   - Add positional encoding for temporal order
   - Weight recent memories more heavily

3. **Memory Update Strategy:**
   - Use learnable update rules
   - Add memory decay for old information
   - Implement memory retrieval with relevance scoring

4. **Training:**
   - Use longer conversation sequences
   - Add curriculum learning (short → long conversations)
   - Increase batch size with gradient checkpointing

---

### 5. **Causal Inference** (nDCG@10: 0.1796 - likely not training)
**Affected:** `tier1_causal_inference`

**Problems:**
- Score identical to baseline suggests no training
- Causal structure may not be properly modeled
- Confounders may not be correctly identified

**Improvements:**
1. **Verify Training:**
   - Check if causal model is actually being trained
   - Verify gradients are flowing through causal layers
   - Add explicit logging for causal effect estimates

2. **Better Causal Modeling:**
   - Use do-calculus for intervention modeling
   - Add backdoor adjustment for confounders
   - Implement instrumental variables if needed

3. **Causal-Aware Loss:**
   - Add causal regularization term
   - Use counterfactual examples in training
   - Optimize for causal effect, not just correlation

4. **Architecture:**
   - Separate confounder encoder
   - Add causal attention mechanism
   - Use structural causal model (SCM)

---

### 6. **Learning-to-Rank** (nDCG@10: 0.1807)
**Affected:** `tier1_learning_to_rank_listwise_fixed`

**Problems:**
- LTR model may be too simple (XGBoost/LightGBM)
- Features may not be discriminative enough
- Listwise loss may not be optimized correctly

**Improvements:**
1. **Better Features:**
   - Add more interaction features (query-doc similarity variants)
   - Include BM25 scores as features
   - Add cross-encoder scores as features
   - Use neural features from multiple models

2. **Neural LTR:**
   - Replace XGBoost with neural ranking model
   - Use ListNet or ListMLE loss
   - Add attention over document list

3. **Training Improvements:**
   - Use larger candidate sets (top 200 → top 500)
   - Add hard negative mining for LTR
   - Use curriculum learning (easy → hard queries)

4. **Ensemble:**
   - Combine multiple LTR models
   - Use different feature sets per model
   - Weight models by query type

---

### 7. **Hard Negatives Experiments** (nDCG@10: 0.1456)
**Affected:** `phase4_hard_negatives_5neg`

**Problems:**
- Too many hard negatives may hurt training
- Hard negatives may not be hard enough
- Negative sampling strategy may be suboptimal

**Improvements:**
1. **Better Hard Negative Mining:**
   - Use dynamic hard negative mining (update during training)
   - Mine from in-batch negatives (more efficient)
   - Use BM25 + dense retrieval for hard negatives
   - Add adversarial hard negatives

2. **Negative Ratio:**
   - Current: 5 negatives per query
   - Optimal: 1-3 hard negatives + in-batch negatives
   - Use curriculum: start with easy negatives, increase difficulty

3. **Loss Function:**
   - Use margin-based loss (not just contrastive)
   - Add temperature scaling for hard negatives
   - Use focal loss to focus on hard examples

4. **Training:**
   - Increase batch size (more in-batch negatives)
   - Use longer training (5-7 epochs)
   - Add warmup for hard negative mining

---

### 8. **Cross-Attention Experiments** (nDCG@10: 0.0000 - BROKEN)
**Affected:** `tier1_cross_attention_rerun`, `phase8_cross_attention_query_document`

**Problems:**
- Zero scores indicate complete failure
- Likely evaluation error or model not saving
- Cross-attention may not be implemented correctly

**Immediate Fixes:**
1. **Debug Evaluation:**
   - Check if model is being loaded correctly
   - Verify retrieval is actually running
   - Check for errors in evaluation code

2. **Fix Cross-Attention:**
   - Verify attention mechanism is working
   - Check if gradients are flowing
   - Ensure query-document interaction is computed

3. **Model Saving:**
   - Verify model is saved after training
   - Check model loading in evaluation
   - Add explicit model path logging

---

## Universal Improvements (Apply to All Low Performers)

### 1. **Model Scaling**
- **Current:** BGE-base (110M parameters)
- **Upgrade to:** BGE-large (335M) or BGE-v2-large
- **Expected gain:** +0.05-0.10 nDCG@10

### 2. **Training Duration**
- **Current:** 3 epochs
- **Increase to:** 5-10 epochs with early stopping
- **Expected gain:** +0.02-0.05 nDCG@10

### 3. **Learning Rate Schedule**
- **Current:** Fixed or linear decay
- **Improve to:** Cosine annealing with warm restarts
- **Expected gain:** +0.01-0.03 nDCG@10

### 4. **Batch Size & Gradient Accumulation**
- **Current:** Small batches (2-4) due to memory
- **Improve:** Larger effective batch size via gradient accumulation
- **Target:** Effective batch size of 32-64
- **Expected gain:** +0.02-0.04 nDCG@10

### 5. **Better Negative Sampling**
- **Current:** Random or simple hard negatives
- **Improve to:**
  - In-batch negatives (free, effective)
  - BM25 hard negatives
  - Dynamic hard negative mining
  - Adversarial negatives
- **Expected gain:** +0.03-0.06 nDCG@10

### 6. **Loss Function Improvements**
- **Current:** Basic contrastive loss
- **Improve to:**
  - Temperature-scaled contrastive loss
  - Margin-based loss
  - Focal loss for hard examples
  - Direct nDCG optimization (NeuralNDCG)
- **Expected gain:** +0.02-0.05 nDCG@10

### 7. **Data Augmentation**
- **Add:**
  - Query paraphrasing
  - Document summarization
  - Back-translation
  - Synonym replacement
- **Expected gain:** +0.01-0.03 nDCG@10

### 8. **Ensemble Methods**
- **Combine:**
  - Multiple model checkpoints
  - Different architectures
  - Different training strategies
- **Expected gain:** +0.03-0.08 nDCG@10

### 9. **Query Expansion**
- **Add:**
  - Pseudo-relevance feedback
  - LLM-based expansion
  - Multi-query generation
- **Expected gain:** +0.02-0.05 nDCG@10

### 10. **Multi-Stage Retrieval**
- **Implement:**
  - Stage 1: Dense retrieval (top 100)
  - Stage 2: Cross-encoder reranking (top 10)
  - Stage 3: LLM-based scoring (optional)
- **Expected gain:** +0.05-0.15 nDCG@10

---

## Priority Action Plan

### Phase 1: Critical Fixes (Immediate)
1. **Fix experiments scoring 0.1796** - Verify training is happening
2. **Fix cross-attention experiments** - Debug zero scores
3. **Fix multitask experiments** - Debug extremely low scores

### Phase 2: Quick Wins (1-2 days)
1. **Model scaling** - Upgrade to BGE-large
2. **Training duration** - Increase to 5-7 epochs
3. **Better negatives** - Add in-batch + BM25 hard negatives
4. **Loss improvements** - Temperature scaling + margin

### Phase 3: Architecture Improvements (3-5 days)
1. **Graph-aware** - Better GNN architecture
2. **Temporal memory** - Improved memory mechanism
3. **Learning-to-rank** - Neural LTR model
4. **Causal inference** - Proper causal modeling

### Phase 4: Advanced Techniques (1 week)
1. **Multi-stage retrieval** - Dense + reranking pipeline
2. **Ensemble methods** - Combine multiple models
3. **Query expansion** - LLM-based expansion
4. **Direct optimization** - NeuralNDCG for nDCG@10

---

## Expected Results

| Experiment Category | Current nDCG@10 | Target nDCG@10 | Improvement Strategy |
|---------------------|----------------|----------------|---------------------|
| Multitask | 0.0185 | 0.35-0.40 | Fix training + better loss balancing |
| Graph-aware | 0.1738 | 0.35-0.40 | Better GNN + graph construction |
| Temporal memory | 0.1484 | 0.30-0.35 | Improved memory architecture |
| Causal inference | 0.1796 | 0.35-0.40 | Fix training + proper causal modeling |
| Learning-to-rank | 0.1807 | 0.30-0.35 | Neural LTR + better features |
| Hard negatives | 0.1456 | 0.30-0.35 | Better mining + loss function |
| Cross-attention | 0.0000 | 0.25-0.30 | Fix implementation + evaluation |

**Overall Goal:** Bring all experiments above **0.30 nDCG@10**, with best performers reaching **0.45-0.50 nDCG@10**.

---

## Implementation Notes

1. **Start with verification:** Before improving, verify that experiments are actually training
2. **Incremental improvements:** Apply one improvement at a time to measure impact
3. **A/B testing:** Compare improved versions with baselines
4. **Monitoring:** Track training loss, validation metrics, and final scores
5. **Documentation:** Document what works and what doesn't for future reference

---

## References

- Best performing: `best_paper_large_model_finetuning` (0.5101 nDCG@10)
- Baseline: `phase1_baseline` (0.4576 nDCG@10)
- Key techniques: Large models, contrastive learning, query expansion, ensemble methods

