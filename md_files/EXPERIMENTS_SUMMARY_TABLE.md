# Phase 4 Experiments Summary Table

## Complete Experiment Overview

| # | Experiment Name | Category | Status | Purpose & Rationale | Key Configuration | Expected Results | Actual Results |
|---|----------------|----------|--------|---------------------|-------------------|------------------|----------------|
| 1 | `phase4_domain_specific_fiqa` | Domain-Specific | ✅ **Completed** | **Why:** FiQA domain had the worst baseline performance (R@10=0.22, nDCG@10=0.17). Domain-specific fine-tuning can capture finance-specific terminology and patterns that a general model misses. **Rationale:** Starting from a multi-domain pre-trained model and further specializing it should improve recall by learning domain-specific semantic relationships. | • Base: BGE-base-en-v1.5<br>• Epochs: 7<br>• Batch: 32<br>• LR: 1e-5<br>• Starts from: phase1_epochs5 (multi-domain model) | +3-10% Recall@10 improvement over baseline | ✅ Training completed<br>• Final Loss: 0.65<br>• Runtime: ~30 minutes<br>• Model saved |
| 2 | `phase4_domain_specific_clapnq` | Domain-Specific | ✅ **Completed** | **Why:** ClapNQ is Wikipedia-based but may benefit from domain specialization. Even though it's already the best performing domain (R@10=0.40), further optimization can push performance even higher. **Rationale:** Wikipedia queries often require precise entity and fact retrieval - domain-specific tuning helps with these nuances. | • Base: BGE-base-en-v1.5<br>• Epochs: 7<br>• Batch: 32<br>• LR: 1e-5<br>• Starts from: phase1_epochs5 | +2-8% improvement over current best | ✅ Training completed<br>• Final Loss: 0.65<br>• Runtime: ~1.5 hours<br>• Model saved |
| 3 | `phase4_domain_specific_govt` | Domain-Specific | ✅ **Completed** | **Why:** Government documents have specific language, terminology, and structure. Govt domain showed good performance (R@10=0.39) but domain-specific fine-tuning can improve understanding of bureaucratic language, regulations, and public policy context. **Rationale:** Government queries often need exact matches for regulations, procedures, and official terminology. | • Base: BGE-base-en-v1.5<br>• Epochs: 7<br>• Batch: 32<br>• LR: 1e-5<br>• Starts from: phase1_epochs5 | +3-8% improvement | ✅ Training completed<br>• Final Loss: 0.6506<br>• Runtime: ~50 minutes<br>• Epoch 7/7 finished |
| 4 | `phase4_domain_specific_cloud` | Domain-Specific | ✅ **Completed** | **Why:** Cloud/technical documentation has specialized technical terminology, code snippets, and API references. Cloud domain was below baseline (R@10=0.31), suggesting the general model struggles with technical jargon. **Rationale:** Technical documentation retrieval benefits from understanding code, APIs, and technical concepts - domain-specific models learn these patterns better. | • Base: BGE-base-en-v1.5<br>• Epochs: 7<br>• Batch: 32<br>• LR: 1e-5<br>• Starts from: phase1_epochs5 | +5-12% improvement | ✅ Training completed<br>• Final Loss: 0.7461<br>• Runtime: ~1.7 hours<br>• Epoch 7/7 finished |
| 5 | `phase4_hard_negatives_cosine` | Hard Negative Mining | 🔄 **Running** | **Why:** Random negatives in batches are too easy - the model doesn't learn fine-grained distinctions between similar passages. Hard negative mining forces the model to distinguish between semantically similar but irrelevant passages. **Rationale:** This technique addresses the core challenge of retrieval: not just finding relevant docs, but correctly rejecting similar-but-wrong ones. **Expected Impact:** Major improvement in precision and recall. | • Base: BGE-base-en-v1.5<br>• Epochs: 5<br>• Batch: 32<br>• LR: 2e-5<br>• Loss: CosineSimilarityLoss<br>• Hard Negatives: 3 per positive | +5-15% Recall@10 improvement | 🔄 Running on GPU 0 & 4<br>• Runtime: 90+ hours (GPU 0), 14+ hours (GPU 4)<br>• Status: Long-running hard negative mining |
| 6 | `phase4_hard_negatives_triplet` | Hard Negative Mining | 🔄 **Running** | **Why:** TripletLoss explicitly models (query, positive, negative) triplets with a margin, providing stronger supervision than batch negatives. **Rationale:** Explicit negative supervision helps the model learn better boundaries between relevant and irrelevant passages. Different loss functions may work better for different data distributions. | • Base: BGE-base-en-v1.5<br>• Epochs: 5<br>• Batch: 32<br>• LR: 2e-5<br>• Loss: TripletLoss<br>• Hard Negatives: 3 per positive | +5-12% improvement | 🔄 Running on GPU 1<br>• Runtime: ~15 hours<br>• Status: Active training |
| 7 | `phase4_hard_negatives_5neg` | Hard Negative Mining | 🔄 **Running** | **Why:** More hard negatives (5 vs 3) provide richer negative supervision and force the model to handle more challenging scenarios. **Rationale:** Increasing the number of hard negatives increases training difficulty but should lead to more robust models that are better at rejecting similar-but-irrelevant passages. | • Base: BGE-base-en-v1.5<br>• Epochs: 5<br>• Batch: 24 (smaller due to more negatives)<br>• LR: 2e-5<br>• Loss: CosineSimilarityLoss<br>• Hard Negatives: 5 per positive | +7-15% improvement | 🔄 Running on GPU 5<br>• Runtime: ~15 hours<br>• Status: Active training |
| 8 | `phase4_bge_large` | Model Architecture | 🔄 **Running** | **Why:** BGE-base (110M params) may be insufficient for complex retrieval tasks. BGE-large (335M params) has stronger representation capabilities and can capture more nuanced semantic relationships. **Rationale:** Larger models typically perform better on complex tasks, but with trade-offs in speed and memory. This tests whether model scale alone improves retrieval. | • Base: BGE-large-en-v1.5 (335M params)<br>• Epochs: 3 (fewer due to size)<br>• Batch: 16 (smaller due to memory)<br>• LR: 1e-5 (lower for stability)<br>• Hard Negatives: 3 | +5-12% overall improvement | 🔄 Running on GPU 2<br>• Runtime: ~47 minutes<br>• Status: ✅ Successfully started after auto-retry<br>• Previous failure: Cache permission issue (fixed) |
| 9 | `phase4_domain_specific_all` | Domain-Specific | ❌ **Failed** | **Why:** This was intended to train all domain-specific models at once, but the configuration used an invalid domain value "all". **Rationale:** N/A - this was a configuration error that has been replaced by individual domain experiments (experiments 1-4). | • N/A - Config error | N/A | ❌ Failed due to invalid domain "all"<br>• Resolution: Split into individual domain experiments (1-4)<br>• All individual experiments completed successfully |

---

## Experiment Categories & Strategy

### 1. Domain-Specific Fine-Tuning (Experiments 1-4)
**Strategy:** Train specialized models for each domain starting from a multi-domain pre-trained base.

**Why This Approach:**
- **One-size-fits-all limitation:** A general model may not optimize for domain-specific characteristics
- **Domain expertise:** Each domain (finance, government, cloud, Wikipedia) has unique terminology, structure, and query patterns
- **Transfer learning:** Starting from a multi-domain model provides a strong foundation, then specializing improves domain-specific performance
- **Expected gains:** 3-10% improvement per domain

**Results:**
- ✅ All 4 domain-specific experiments completed successfully
- ✅ All reached epoch 7/7 with final losses between 0.65-0.75
- ⏳ Evaluation pending to measure actual retrieval improvements

---

### 2. Hard Negative Mining (Experiments 5-7)
**Strategy:** Mine "hard negatives" - passages that are semantically similar to queries but are not relevant.

**Why This Approach:**
- **Problem with random negatives:** Easy negatives don't teach the model fine-grained distinctions
- **Hard negatives force learning:** Model must learn to distinguish between similar-but-wrong passages
- **Multiple strategies:** Testing different loss functions (CosineSimilarityLoss vs TripletLoss) and different numbers of negatives (3 vs 5)
- **Expected gains:** 5-15% improvement depending on configuration

**Current Status:**
- 🔄 3 experiments running in parallel
- ⏳ Long-running process due to hard negative mining complexity
- ⏳ Estimated completion: 10-50 more hours depending on experiment

---

### 3. Larger Model Architecture (Experiment 8)
**Strategy:** Use BGE-large (335M params) instead of BGE-base (110M params).

**Why This Approach:**
- **Capacity limitation:** Base model may lack capacity for complex semantic relationships
- **Scale benefits:** Larger models typically capture more nuanced patterns
- **Trade-off analysis:** Test whether the performance gain justifies increased computational cost
- **Expected gains:** 5-12% improvement

**Current Status:**
- 🔄 Running on GPU 2
- ✅ Successfully recovered from initial cache permission failure
- ⏳ Early stage (~47 minutes runtime)

---

## Performance Targets

**Baseline Performance (Phase 1):**
- Recall@10: 0.38
- nDCG@10: 0.30

**Target Performance (Phase 4):**
- Recall@10: > 0.50 (targeting +32% improvement)
- nDCG@10: > 0.40 (targeting +33% improvement)

**Domain-Specific Baseline:**
- **ClapNQ:** R@10=0.40, nDCG@10=0.30 (best)
- **Govt:** R@10=0.39, nDCG@10=0.29 (good)
- **Cloud:** R@10=0.31, nDCG@10=0.23 (below baseline)
- **FiQA:** R@10=0.22, nDCG@10=0.17 (worst)

---

## Execution Summary

| Status | Count | Percentage |
|--------|-------|------------|
| ✅ Completed | 4 | 44% |
| 🔄 Running | 5 | 56% |
| ❌ Failed | 1 | 11% (replaced by individual experiments) |

**GPU Utilization:**
- 5 GPUs actively training
- 1 GPU free (GPU 3)
- All experiments running in parallel where possible

**Next Steps:**
1. ⏳ Wait for hard negative mining experiments to complete (10-50 hours)
2. ⏳ Wait for BGE-large experiment to complete (~3-5 hours)
3. ⏳ Evaluate all completed models on test set
4. ⏳ Compare results and identify best performing approaches
5. ⏳ Integrate best models into production pipeline

---

## Key Insights & Rationale

### Why Hard Negative Mining?
Hard negative mining addresses the core challenge in retrieval: **distinguishing between relevant and similar-but-irrelevant passages**. Random negatives are too easy and don't push the model to learn fine-grained semantic distinctions. By explicitly mining hard negatives (semantically similar but incorrect), we force the model to learn better decision boundaries.

### Why Domain-Specific Models?
Different domains have different:
- **Terminology:** Finance (FiQA) vs. Technical (Cloud) vs. Government (Govt)
- **Query patterns:** Wikipedia queries vs. API documentation queries
- **Relevance criteria:** What makes a passage relevant differs by domain

A specialized model for each domain can capture these nuances better than a general model.

### Why Multiple Loss Functions?
Different loss functions model the learning objective differently:
- **CosineSimilarityLoss:** Explicit positive/negative pairs with cosine similarity
- **TripletLoss:** Explicit triplets (query, positive, negative) with margin
- **MultipleNegativesRankingLoss:** Uses batch negatives (baseline)

Testing multiple approaches helps identify which works best for our specific data distribution.

### Why Larger Models?
Scale matters in deep learning. BGE-large (335M) vs BGE-base (110M) provides:
- More parameters to capture complex patterns
- Better representation of nuanced semantic relationships
- Potentially better performance, with computational trade-offs

---

*Last Updated: 2025-11-28*
*Status: 4/9 experiments completed, 5/9 running, 0 pending failures*

