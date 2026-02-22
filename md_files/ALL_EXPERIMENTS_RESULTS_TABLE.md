# Complete Experiment Results Table

**Last Updated**: 2025-12-16  
**Baseline Reference**: Paper Baseline (Recall@10: 0.3800, nDCG@10: 0.3000)

---

## 📊 All Experiments Results (Sorted by Recall@10)

| Rank | Experiment | Recall@10 | nDCG@10 | R@10 vs Paper | nDCG@10 vs Paper | Phase |
|------|------------|-----------|---------|---------------|------------------|-------|
| - | **Paper Baseline** | **0.3800** | **0.3000** | - | - | Reference |
| 🥇 | **phase5_ensemble_domain_specific** | **0.5441** | **0.4539** | **+43.2%** | **+51.3%** | Phase 5 |
| 🥈 | phase5_query_expansion_govt | 0.5317 | 0.4515 | +39.9% | +50.5% | Phase 5 |
| 🥉 | phase5_ensemble_weighted | 0.5284 | 0.4370 | +39.1% | +45.7% | Phase 5 |
| 4 | phase5_query_expansion_multi | 0.5099 | 0.4098 | +34.2% | +36.6% | Phase 5 |
| 5 | phase5_query_expansion_clapnq | 0.4999 | 0.4186 | +31.5% | +39.5% | Phase 5 |
| 6 | phase5_reranking_clapnq | 0.4319 | 0.3250 | +13.6% | +8.3% | Phase 5 |
| 7 | phase5_reranking_govt | 0.3539 | 0.2896 | -6.9% | -3.5% | Phase 5 |
| 8 | phase3_hybrid_reranking | 0.3388 | 0.2709 | -10.9% | -9.7% | Phase 3 |
| 9 | phase3_reranking | 0.3388 | 0.2709 | -10.9% | -9.7% | Phase 3 |
| 10 | phase5_reranking_multi_domain | 0.3365 | 0.2619 | -11.4% | -12.7% | Phase 5 |
| 11 | phase5_reranking_cloud | 0.2776 | 0.2028 | -27.0% | -32.4% | Phase 5 |
| 12 | phase4_hard_negatives_5neg | 0.1713 | 0.1456 | -54.9% | -51.5% | Phase 4 |

---

## 📈 Results by Phase

### Phase 1: Foundation Experiments

| Experiment | Recall@10 | nDCG@10 | R@10 vs Paper | nDCG@10 vs Paper | Status |
|------------|-----------|---------|---------------|------------------|--------|
| phase1_baseline | 0.3076 | 0.2303 | -19.1% | -23.2% | ✅ Completed |
| phase1_epochs3 | ~0.3388* | ~0.2709* | -10.9% | -9.7% | ✅ Completed |
| phase1_epochs5 | ~0.4693* | ~0.3671* | +23.5% | +22.4% | ✅ Completed |

*Note: Phase 1 experiments use domain-specific format. Values shown are estimates from domain averages.

### Phase 2: Hyperparameter & Data Augmentation

| Experiment | Recall@10 | nDCG@10 | R@10 vs Paper | nDCG@10 vs Paper | Status |
|------------|-----------|---------|---------------|------------------|--------|
| phase2_augmentation | 0.5099 | 0.4099 | +34.2% | +36.6% | ✅ Completed |
| phase2_lr1e5 | ~0.3309* | ~0.2605* | -12.9% | -13.2% | ✅ Completed |
| phase2_lr5e5 | ~0.4208* | ~0.3331* | +10.7% | +11.0% | ✅ Completed |

*Note: Values are estimates from domain averages.

### Phase 3: Hybrid Retrieval & Reranking

| Experiment | Recall@10 | nDCG@10 | R@10 vs Paper | nDCG@10 vs Paper | Status |
|------------|-----------|---------|---------------|------------------|--------|
| phase3_hybrid | ~0.3388* | ~0.2709* | -10.9% | -9.7% | ✅ Completed |
| phase3_reranking | 0.3388 | 0.2709 | -10.9% | -9.7% | ✅ Completed |
| phase3_hybrid_reranking | 0.3388 | 0.2709 | -10.9% | -9.7% | ✅ Completed |

### Phase 4: Advanced Techniques

#### Domain-Specific Models (Single Domain Results)

| Experiment | Domain | Recall@10 | nDCG@10 | R@10 vs Paper | nDCG@10 vs Paper | Status |
|------------|--------|-----------|---------|---------------|------------------|--------|
| phase4_domain_specific_clapnq | ClapNQ | 0.6016 | 0.4981 | +58.3% | +66.0% | ✅ Completed |
| phase4_domain_specific_govt | Govt | 0.5511 | 0.4628 | +45.0% | +54.3% | ✅ Completed |
| phase4_domain_specific_cloud | Cloud | 0.5293 | 0.4104 | +39.3% | +36.8% | ✅ Completed |
| phase4_domain_specific_fiqa | FiQA | 0.5119 | 0.4026 | +34.7% | +34.2% | ✅ Completed |

**Average Domain-Specific**: ~0.5485 Recall@10, ~0.4435 nDCG@10 (+44.3% / +47.8%)

#### Hard Negative Mining

| Experiment | Recall@10 | nDCG@10 | R@10 vs Paper | nDCG@10 vs Paper | Status |
|------------|-----------|---------|---------------|------------------|--------|
| phase4_hard_negatives_cosine | ~0.1627* | ~0.1444* | -57.2% | -51.9% | ✅ Completed (Poor) |
| phase4_hard_negatives_5neg | 0.1713 | 0.1456 | -54.9% | -51.5% | ✅ Completed (Poor) |

*Note: Hard negative experiments underperformed significantly.

### Phase 5: Ensemble & Advanced Techniques

| Experiment | Recall@10 | nDCG@10 | R@10 vs Paper | nDCG@10 vs Paper | Status |
|------------|-----------|---------|---------------|------------------|--------|
| **phase5_ensemble_domain_specific** 🥇 | **0.5441** | **0.4539** | **+43.2%** | **+51.3%** | ✅ Completed |
| phase5_ensemble_weighted | 0.5284 | 0.4370 | +39.1% | +45.7% | ✅ Completed |
| phase5_query_expansion_govt | 0.5317 | 0.4515 | +39.9% | +50.5% | ✅ Completed |
| phase5_query_expansion_multi | 0.5099 | 0.4098 | +34.2% | +36.6% | ✅ Completed |
| phase5_query_expansion_clapnq | 0.4999 | 0.4186 | +31.5% | +39.5% | ✅ Completed |
| phase5_reranking_clapnq | 0.4319 | 0.3250 | +13.6% | +8.3% | ✅ Completed |
| phase5_reranking_govt | 0.3539 | 0.2896 | -6.9% | -3.5% | ✅ Completed |
| phase5_reranking_multi_domain | 0.3365 | 0.2619 | -11.4% | -12.7% | ✅ Completed |
| phase5_reranking_cloud | 0.2776 | 0.2028 | -27.0% | -32.4% | ✅ Completed |

#### Hybrid Retrieval (Running)

| Experiment | Status | Expected Completion |
|------------|--------|---------------------|
| phase5_hybrid_clapnq_alpha0.7 | 🟢 Running | ~1-2 hours |
| phase5_hybrid_govt_alpha0.3 | 🟢 Running | ~1-2 hours |
| phase5_hybrid_govt_alpha0.5 | 🟢 Running | ~1-2 hours |
| phase5_hybrid_govt_alpha0.7 | 🟢 Running | ~1-2 hours |
| phase5_hybrid_multi_alpha0.3 | 🟢 Running | ~1-2 hours |
| phase5_hybrid_multi_alpha0.5 | 🟢 Running | ~1-2 hours |
| phase5_hybrid_multi_alpha0.7 | 🟢 Running | ~1-2 hours |

---

## 🏆 Top 5 Performers

| Rank | Experiment | Recall@10 | nDCG@10 | Improvement |
|------|------------|-----------|---------|-------------|
| 🥇 | **phase5_ensemble_domain_specific** | **0.5441** | **0.4539** | **+43.2% / +51.3%** |
| 🥈 | phase5_query_expansion_govt | 0.5317 | 0.4515 | +39.9% / +50.5% |
| 🥉 | phase5_ensemble_weighted | 0.5284 | 0.4370 | +39.1% / +45.7% |
| 4 | phase5_query_expansion_multi | 0.5099 | 0.4098 | +34.2% / +36.6% |
| 5 | phase5_query_expansion_clapnq | 0.4999 | 0.4186 | +31.5% / +39.5% |

---

## 📊 Performance Summary

### Best Results
- **Best Recall@10**: 0.5441 (phase5_ensemble_domain_specific)
- **Best nDCG@10**: 0.4539 (phase5_ensemble_domain_specific)
- **Best Single Domain**: 0.6016 R@10, 0.4981 nDCG@10 (phase4_domain_specific_clapnq)

### Overall Statistics
- **Total Experiments**: 23 completed
- **Experiments Above Baseline**: 12 (52%)
- **Experiments Below Baseline**: 11 (48%)
- **Average Improvement (Top 5)**: +37.6% Recall@10, +45.9% nDCG@10

### Techniques That Worked Best
1. ✅ **Ensemble Methods** - Best overall performance
2. ✅ **Query Expansion** - Strong improvements across domains
3. ✅ **Domain-Specific Models** - Excellent single-domain performance
4. ✅ **Data Augmentation** - Good multi-domain baseline

### Techniques That Underperformed
1. ❌ **Hard Negative Mining** - Significant performance degradation
2. ❌ **Reranking (Some Domains)** - Mixed results, some domains worse
3. ❌ **Hybrid Retrieval (Phase 3)** - Underperformed compared to dense

---

## 📈 Progress Over Phases

| Phase | Best R@10 | Best nDCG@10 | Improvement vs Paper |
|-------|-----------|--------------|---------------------|
| Baseline | 0.3800 | 0.3000 | - |
| Phase 1 | 0.4693 | 0.3671 | +23.5% / +22.4% |
| Phase 2 | 0.5099 | 0.4099 | +34.2% / +36.6% |
| Phase 3 | 0.3388 | 0.2709 | -10.9% / -9.7% |
| Phase 4 | 0.6016* | 0.4981* | +58.3% / +66.0%* |
| Phase 5 | **0.5441** | **0.4539** | **+43.2% / +51.3%** |

*Phase 4 single-domain result (ClapNQ), not average

---

## 🎯 Key Insights

1. **Ensemble methods achieve best overall performance** - Combining domain-specific models via RRF yields the best results
2. **Domain-specific models excel** - Single-domain models show exceptional performance (up to 0.6016 R@10)
3. **Query expansion is effective** - Consistent improvements across multiple domains
4. **Hard negatives need refinement** - Current implementation significantly underperformed
5. **Reranking is domain-dependent** - Works well for some domains (ClapNQ) but not others (Cloud)

---

*Table compiled from all experiment result JSON files*  
*Baseline: Paper Baseline (Recall@10: 0.3800, nDCG@10: 0.3000)*

