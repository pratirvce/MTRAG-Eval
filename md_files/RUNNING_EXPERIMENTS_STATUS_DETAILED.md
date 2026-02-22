# Running Experiments Status - Detailed Report
**Last Updated:** 2025-12-18 20:36

---

## 🔄 ACTIVE EXPERIMENTS (1)

### 1. **tier1_cross_attention_rerun_fixed_v2** ✅ GOOD PROGRESS
- **GPU:** 2 (75% utilization, 5.3 GB memory)
- **Status:** Running
- **Progress:** ~50% complete (2/4 domains done)
- **Completed Domains:**
  - ✅ **clapnq**: Recall@10=0.3716, nDCG@10=0.2765
  - ✅ **fiqa**: Recall@10=0.2768, nDCG@10=0.2298
- **Current Domain:** govt (processing)
- **Remaining:** cloud
- **Note:** Fix is working! Getting non-zero results ✅

---

## ❌ COMPLETED WITH ERRORS/ISSUES (5)

### 2. **tier1_learning_to_rank_listwise**
- **Status:** ❌ Failed
- **Error:** `ValueError: Model/Technique has not been provided!`
- **Issue:** Missing model/technique configuration in evaluator
- **Results:** Zero (due to error)

### 3. **tier1_enhanced_contrastive_hardnegatives**
- **Status:** ❌ Failed
- **Error:** `KeyError: 'last_hidden_state'`
- **Issue:** Model output format mismatch - expecting 'last_hidden_state' key
- **Results:** Zero (due to error)

### 4. **best_paper_meta_learning**
- **Status:** ✅ Completed (but zero results)
- **Results:** Recall@10=0.0000, nDCG@10=0.0000
- **Issue:** Completed but produced zero results - needs investigation
- **Last Activity:** 2025-12-18 20:35:37

### 5. **best_paper_hierarchical_routing**
- **Status:** ✅ Completed (but zero results)
- **Results:** Recall@10=0.0000, nDCG@10=0.0000
- **Issue:** Completed but produced zero results - needs investigation
- **Last Activity:** 2025-12-18 20:35:38

### 6. **best_paper_rl_adaptive_retrieval**
- **Status:** ✅ Completed (but zero results)
- **Results:** Recall@10=0.0000, nDCG@10=0.0000
- **Issue:** Completed but produced zero results - needs investigation
- **Last Activity:** 2025-12-18 20:35:38

---

## 📊 GPU Status

| GPU | Utilization | Memory | Status | Experiment |
|-----|-------------|--------|--------|------------|
| 0 | 100% | 6.8 GB | 🔴 BUSY | Unknown process |
| 1 | 0% | 264 MB | 🟢 FREE | Available |
| 2 | 75% | 5.3 GB | 🔴 BUSY | tier1_cross_attention_rerun_fixed_v2 |
| 3 | 100% | 14.9 GB | 🔴 BUSY | Unknown process |
| 4 | 100% | 5.6 GB | 🔴 BUSY | Unknown process |
| 5 | 100% | 5.5 GB | 🔴 BUSY | Unknown process |

**Available GPUs:** 1

---

## ✅ GOOD NEWS

**tier1_cross_attention_rerun_fixed_v2** is working correctly with the fix:
- Getting **non-zero results** (Recall@10=0.37 for clapnq, 0.28 for fiqa)
- Progress is steady
- Should complete in ~1-2 hours

---

## ⚠️ ISSUES TO ADDRESS

1. **tier1_learning_to_rank_listwise**: Missing model configuration
2. **tier1_enhanced_contrastive_hardnegatives**: Model output format issue
3. **best_paper_meta_learning**: Zero results - needs investigation
4. **best_paper_hierarchical_routing**: Zero results - needs investigation
5. **best_paper_rl_adaptive_retrieval**: Zero results - needs investigation

---

## 📝 Next Steps

1. **Monitor** `tier1_cross_attention_rerun_fixed_v2` until completion
2. **Investigate** why best_paper experiments produced zero results
3. **Fix** errors in tier1_learning_to_rank_listwise and tier1_enhanced_contrastive_hardnegatives
4. **Start new experiments** on GPU 1 when ready

---

*Generated: 2025-12-18 20:36*

