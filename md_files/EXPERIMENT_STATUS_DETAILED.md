# Detailed Experiment Status Report

**Generated**: 2025-12-17  
**Current Best**: 0.4576 nDCG@10 (Contrastive Learning)

---

## ✅ Completed Experiments (10)

### Top Performers:
1. **tier1_contrastive_learning**: **0.4576 nDCG@10** 🏆 **BEST**
2. **tier1_cross_encoder_evaluation**: **0.2716 nDCG@10**
3. **tier2_multistage_3stage**: **0.2480 nDCG@10**
4. **tier1_hierarchical_multigranularity**: **0.2289 nDCG@10**
5. **tier1_iterative_refinement_improved**: **0.2231 nDCG@10**
6. **tier2_ensemble_advanced**: **0.2192 nDCG@10**

### Completed but Issues:
- **tier1_cross_attention_query_document**: 0.0 nDCG@10 (failed)
- **tier1_cross_attention_rerun**: 0.0 nDCG@10 (failed)
- **tier1_cross_encoder_finetuned**: Training completed, no evaluation results
- **tier1_cross_encoder_large**: Completed but no valid results

---

## 🟢 Currently Running Experiments (4)

### 1. **tier1_cross_encoder_domain_specific**
- **Status**: 🟢 RUNNING
- **Issue**: ⚠️ CUDA error - "invalid device ordinal"
- **Error**: GPU device mismatch
- **Action Needed**: Fix GPU assignment or device configuration

### 2. **tier1_ensemble_best_methods**
- **Status**: 🟢 RUNNING
- **Issue**: ⚠️ ZeroDivisionError - division by zero in evaluation
- **Error**: No scores returned from evaluation (empty results)
- **Action Needed**: Fix ensemble logic to handle empty results

### 3. **tier1_llm_query_expansion**
- **Status**: 🟢 RUNNING
- **Issue**: ⚠️ ImportError - cannot import `run_llm_query_expansion`
- **Error**: Function name mismatch in wrapper script
- **Action Needed**: Fix import or function name in `train_llm_query_expansion.py`

### 4. **tier1_multistage_2stage**
- **Status**: 🟢 RUNNING
- **Issue**: ⚠️ ImportError - cannot import `run_multistage_retrieval`
- **Error**: Function name mismatch in wrapper script
- **Action Needed**: Fix import or function name in `train_multistage_retrieval.py`

---

## ⚠️ Issues Summary

### Critical Issues:
1. **CUDA Device Error** (tier1_cross_encoder_domain_specific)
   - Invalid device ordinal
   - Need to fix GPU assignment

2. **Import Errors** (2 experiments)
   - LLM query expansion wrapper
   - Multi-stage wrapper
   - Need to fix function names/imports

3. **Evaluation Errors** (tier1_ensemble_best_methods)
   - ZeroDivisionError in evaluation
   - Need to handle empty results

### Failed Experiments:
- Cross-Attention experiments (both returned 0.0)
- Need investigation and re-implementation

---

## 📊 Performance Summary

| Experiment | nDCG@10 | Status | Notes |
|------------|---------|--------|-------|
| tier1_contrastive_learning | **0.4576** | ✅ | **BEST** |
| tier1_cross_encoder_evaluation | 0.2716 | ✅ | Good |
| tier2_multistage_3stage | 0.2480 | ✅ | Moderate |
| tier1_hierarchical_multigranularity | 0.2289 | ✅ | Moderate |
| tier1_iterative_refinement_improved | 0.2231 | ✅ | Low |
| tier2_ensemble_advanced | 0.2192 | ✅ | Low |
| tier1_cross_attention_rerun | 0.0 | ⚠️ | Failed |
| tier1_cross_attention_query_document | 0.0 | ⚠️ | Failed |

---

## 🔧 Recommended Actions

### Immediate Fixes:
1. **Fix GPU device error** in domain-specific cross-encoder
2. **Fix import errors** in LLM expansion and multi-stage wrappers
3. **Fix ensemble evaluation** to handle empty results

### Re-run Needed:
1. Cross-Attention experiments (both failed with 0.0)
2. Cross-Encoder Large (completed but no results)

### Monitor:
1. Domain-specific cross-encoder (after fixing GPU issue)
2. Ensemble best methods (after fixing evaluation)

---

## 🎯 Current Progress

- **Completed**: 10 experiments
- **Running**: 4 experiments (with issues)
- **Best Score**: 0.4576 nDCG@10
- **Gap to Elser (0.54)**: -0.0824 (need +18% improvement)
- **Gap to Tier 1 Target (0.55)**: -0.0924 (need +20% improvement)

---

*Last Updated: 2025-12-17*
