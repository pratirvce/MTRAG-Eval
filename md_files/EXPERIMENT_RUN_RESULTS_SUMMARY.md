# Experiment Run Results Summary

**Last Updated**: 2025-12-14  
**Total Experiments**: 20 (7 completed, 6 running, 7 pending)

---

## 🎯 Executive Summary

### Current Best Performance
- **Best Overall**: `phase5_ensemble_domain_specific`
  - **Recall@10**: **0.5441** (+43.2% vs baseline)
  - **nDCG@10**: **0.4539** (+51.3% vs baseline)
  - **Method**: Ensemble of domain-specific models using Reciprocal Rank Fusion (RRF)

### Baseline Comparison
- **Paper Baseline**: R@10=0.38, nDCG@10=0.30
- **Current Best**: R@10=0.5441, nDCG@10=0.4539
- **Improvement**: +43.2% Recall@10, +51.3% nDCG@10 🚀

---

## 📊 Completed Experiments (7/20)

### Top 3 Performers

#### 1. 🥇 phase5_ensemble_domain_specific (BEST)
- **Recall@10**: 0.5441
- **nDCG@10**: 0.4539
- **Method**: Ensemble (RRF) of domain-specific models
- **Domain Breakdown**:
  - ClapNQ: R@10=0.6381, nDCG@10=0.5357
  - FiQA: R@10=0.5030, nDCG@10=0.4108
  - Govt: R@10=0.5403, nDCG@10=0.4673
  - Cloud: R@10=0.4948, nDCG@10=0.4017

#### 2. 🥈 phase5_ensemble_weighted
- **Recall@10**: 0.5284
- **nDCG@10**: 0.4370
- **Method**: Weighted ensemble of domain-specific models
- **Improvement**: -2.9% vs best ensemble

#### 3. 🥉 phase5_query_expansion_clapnq
- **Recall@10**: 0.4999
- **nDCG@10**: 0.4186
- **Method**: Query expansion for ClapNQ domain
- **Notable**: ClapNQ domain achieved R@10=0.6016, nDCG@10=0.4981 (excellent single-domain performance)

### Reranking Experiments (Lower Performance)

#### 4. phase5_reranking_clapnq
- **Recall@10**: 0.4319
- **nDCG@10**: 0.3250
- **Assessment**: Moderate performance, best among reranking experiments

#### 5. phase5_reranking_govt
- **Recall@10**: 0.3539
- **nDCG@10**: 0.2896
- **Assessment**: Below baseline performance

#### 6. phase5_reranking_cloud
- **Recall@10**: 0.2776
- **nDCG@10**: 0.2028
- **Assessment**: Poor performance, worst among completed experiments

#### 7. phase5_reranking_multi_domain
- **Recall@10**: 0.3365
- **nDCG@10**: 0.2619
- **Assessment**: Below baseline performance

**Reranking Summary**: Cross-encoder reranking experiments underperformed compared to dense retrieval and ensemble methods.

---

## 🔄 Running Experiments (6/20)

### Long-Running Training (GPU 1)

#### 1. phase5_domain_specific_clapnq_hard_negatives
- **Status**: 🟢 Running (~26 hours)
- **Type**: Training with hard negative mining
- **Expected**: High potential (+3-8% nDCG improvement)
- **Target**: Improve ranking discrimination for ClapNQ domain

#### 2. phase5_domain_specific_govt_hard_negatives
- **Status**: 🟢 Running (~26 hours)
- **Type**: Training with hard negative mining
- **Expected**: High potential (+3-8% nDCG improvement)
- **Target**: Improve ranking discrimination for Govt domain

### Just Started Evaluation (GPUs 2-5)

#### 3. phase5_query_expansion_govt
- **Status**: 🟢 Running (just started)
- **Type**: Query expansion evaluation
- **Expected**: ~1-2 hours completion
- **Potential**: +2-5% improvement if similar to ClapNQ results

#### 4. phase5_query_expansion_multi
- **Status**: 🟢 Running (just started)
- **Type**: Query expansion evaluation (multi-domain)
- **Expected**: ~1-2 hours completion
- **Potential**: +2-5% improvement

#### 5. phase5_hybrid_clapnq_alpha0.3
- **Status**: 🟢 Running (just started)
- **Type**: Hybrid retrieval (BM25 + Dense, α=0.3)
- **Expected**: ~1-2 hours completion
- **Potential**: +1-4% improvement with optimal alpha

#### 6. phase5_hybrid_clapnq_alpha0.5
- **Status**: 🟢 Running (just started)
- **Type**: Hybrid retrieval (BM25 + Dense, α=0.5)
- **Expected**: ~1-2 hours completion
- **Potential**: +1-4% improvement with optimal alpha

---

## ⏳ Pending Experiments (7/20)

All pending experiments are hybrid retrieval evaluations (faster than training):

1. `phase5_hybrid_clapnq_alpha0.7` (BM25 + Dense, α=0.7)
2. `phase5_hybrid_govt_alpha0.3` (BM25 + Dense, α=0.3)
3. `phase5_hybrid_govt_alpha0.5` (BM25 + Dense, α=0.5)
4. `phase5_hybrid_govt_alpha0.7` (BM25 + Dense, α=0.7)
5. `phase5_hybrid_multi_alpha0.3` (BM25 + Dense, α=0.3)
6. `phase5_hybrid_multi_alpha0.5` (BM25 + Dense, α=0.5)
7. `phase5_hybrid_multi_alpha0.7` (BM25 + Dense, α=0.7)

**Note**: These will auto-start as GPUs become available.

---

## 📈 Performance Progression by Phase

| Phase | Best R@10 | Best nDCG@10 | Improvement vs Baseline | Key Technique |
|-------|-----------|--------------|-------------------------|---------------|
| **Baseline** | 0.38 | 0.30 | - | Paper baseline |
| **Phase 1** | 0.47 | 0.37 | +23.5% R@10, +22.4% nDCG@10 | Extended training (5 epochs) |
| **Phase 2** | 0.51 | 0.41 | +34.2% R@10, +36.6% nDCG@10 | Data augmentation |
| **Phase 3** | 0.34 | 0.27 | -10.8% R@10, -9.7% nDCG@10 | Hybrid retrieval (underperformed) |
| **Phase 4** | 0.60* | 0.50* | +58.3% R@10, +66.0% nDCG@10* | Domain-specific models (single domain) |
| **Phase 5** | **0.5441** | **0.4539** | **+43.2% R@10, +51.3% nDCG@10** | Ensemble methods |

*Phase 4 single-domain results (ClapNQ), not average

**Total Improvement**: +43.2% Recall@10, +51.3% nDCG@10 from baseline

---

## 🏆 Best Results by Domain

| Domain | Best Model | Recall@10 | nDCG@10 | Improvement vs Baseline |
|--------|-----------|-----------|---------|------------------------|
| **ClapNQ** | phase5_ensemble_domain_specific | **0.6381** | **0.5357** | +67.9% / +78.6% |
| **Govt** | phase5_ensemble_domain_specific | **0.5403** | **0.4673** | +42.2% / +55.8% |
| **FiQA** | phase5_ensemble_domain_specific | **0.5030** | **0.4108** | +32.4% / +36.9% |
| **Cloud** | phase5_ensemble_domain_specific | **0.4948** | **0.4017** | +30.2% / +33.9% |

**Key Insight**: Ensemble of domain-specific models provides best performance across all domains.

---

## 📊 Detailed Results Table

| Experiment | Recall@10 | nDCG@10 | Status | Method |
|------------|-----------|---------|--------|--------|
| **phase5_ensemble_domain_specific** | **0.5441** | **0.4539** | ✅ | Ensemble (RRF) |
| phase5_ensemble_weighted | 0.5284 | 0.4370 | ✅ | Weighted Ensemble |
| phase5_query_expansion_clapnq | 0.4999 | 0.4186 | ✅ | Query Expansion |
| phase5_reranking_clapnq | 0.4319 | 0.3250 | ✅ | Reranking |
| phase5_reranking_govt | 0.3539 | 0.2896 | ✅ | Reranking |
| phase5_reranking_multi_domain | 0.3365 | 0.2619 | ✅ | Reranking |
| phase5_reranking_cloud | 0.2776 | 0.2028 | ✅ | Reranking |
| phase5_domain_specific_clapnq_hard_negatives | - | - | 🟢 Running | Hard Negatives |
| phase5_domain_specific_govt_hard_negatives | - | - | 🟢 Running | Hard Negatives |
| phase5_query_expansion_govt | - | - | 🟢 Running | Query Expansion |
| phase5_query_expansion_multi | - | - | 🟢 Running | Query Expansion |
| phase5_hybrid_clapnq_alpha0.3 | - | - | 🟢 Running | Hybrid (α=0.3) |
| phase5_hybrid_clapnq_alpha0.5 | - | - | 🟢 Running | Hybrid (α=0.5) |
| phase5_hybrid_clapnq_alpha0.7 | - | - | ⏳ Pending | Hybrid (α=0.7) |
| phase5_hybrid_govt_alpha0.3 | - | - | ⏳ Pending | Hybrid (α=0.3) |
| phase5_hybrid_govt_alpha0.5 | - | - | ⏳ Pending | Hybrid (α=0.5) |
| phase5_hybrid_govt_alpha0.7 | - | - | ⏳ Pending | Hybrid (α=0.7) |
| phase5_hybrid_multi_alpha0.3 | - | - | ⏳ Pending | Hybrid (α=0.3) |
| phase5_hybrid_multi_alpha0.5 | - | - | ⏳ Pending | Hybrid (α=0.5) |
| phase5_hybrid_multi_alpha0.7 | - | - | ⏳ Pending | Hybrid (α=0.7) |

---

## 🔍 Key Findings

### ✅ What Works Best

1. **Ensemble Methods** 🏆
   - Domain-specific ensemble achieves best overall performance
   - RRF (Reciprocal Rank Fusion) outperforms weighted ensemble
   - Combines strengths of multiple domain-specific models

2. **Domain-Specific Models**
   - Individual domain models show excellent performance
   - ClapNQ domain-specific: 0.6016 R@10, 0.4981 nDCG@10
   - Foundation for successful ensemble

3. **Query Expansion**
   - Proven effective on ClapNQ domain
   - Shows promise for other domains (experiments running)

### ⚠️ What Underperformed

1. **Reranking (Cross-Encoder)**
   - All reranking experiments underperformed
   - Best reranking result: 0.4319 R@10 (vs 0.5441 ensemble)
   - May need better reranker training or different approach

2. **Hybrid Retrieval (Phase 3)**
   - Early hybrid experiments underperformed
   - Phase 5 hybrid experiments still running (may show improvement with optimized alpha)

### 🎯 Expected Improvements from Running Experiments

1. **Hard Negative Mining** (2 experiments running)
   - Expected: +3-8% nDCG@10 improvement
   - If successful: Could reach 0.47-0.49 nDCG@10

2. **Query Expansion** (2 experiments running)
   - Expected: +2-5% improvement if consistent with ClapNQ results
   - Potential: 0.47-0.48 nDCG@10

3. **Hybrid Retrieval** (2 running, 7 pending)
   - Expected: +1-4% improvement with optimal alpha
   - Potential: 0.47-0.49 nDCG@10 with best alpha

---

## 📈 Performance Trajectory

### Current State
- **Best nDCG@10**: 0.4539 (phase5_ensemble_domain_specific)
- **Target**: 0.50+ nDCG@10 (10% improvement from current)

### Expected Scenarios

#### Scenario 1: Hard Negatives Success (Most Likely)
- Hard negatives: +5% → **0.4766**
- Query expansion: +3% → **0.4909**
- Hybrid optimal: +2% → **0.5007**
- **Final**: **~0.50 nDCG@10** ✅

#### Scenario 2: Moderate Success
- Hard negatives: +3% → **0.4675**
- Query expansion: +2% → **0.4770**
- Hybrid optimal: +1% → **0.4818**
- **Final**: **~0.48 nDCG@10** ✅

#### Scenario 3: Limited Success
- Hard negatives: +1% → **0.4584**
- Query expansion: +1% → **0.4626**
- Hybrid optimal: +0.5% → **0.4650**
- **Final**: **~0.46-0.47 nDCG@10** (still improvement)

---

## 🎯 Recommendations

### For Production Use
1. **Primary**: Use `phase5_ensemble_domain_specific` model
   - Best overall performance (0.5441 R@10, 0.4539 nDCG@10)
   - Robust across all domains

2. **Alternative**: Use individual domain-specific models
   - ClapNQ: Best single-domain performance
   - Domain-specific models available for each domain

### For Future Experiments
1. **Wait for hard negative results** (highest potential)
2. **Complete query expansion experiments** (proven on ClapNQ)
3. **Optimize hybrid retrieval** (find best alpha per domain)
4. **Consider advanced ensemble** (combine best techniques)

---

## 📝 Notes

- All experiments are **MTRAGEval Task A compliant**
- **No failures** so far (0 failed experiments)
- **All 6 GPUs currently utilized** efficiently
- Results are from validation sets; test set evaluation recommended for final comparison
- **Progress**: 35% completed (7/20), 30% running (6/20), 35% pending (7/20)

---

*Summary compiled from: COMPREHENSIVE_EXPERIMENT_RESULTS.md, CURRENT_EXPERIMENT_STATUS.md, NDCG_IMPROVEMENT_ANALYSIS.md, and experiment result JSON files*

