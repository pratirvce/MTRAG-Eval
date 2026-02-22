# Experiment Status Summary
**Last Updated:** 2025-12-18 20:15

---

## 🔄 Currently Running Experiments (4)

### 1. **tier1_cross_attention_rerun_fixed_v2** (NEW - with fix)
- **Status:** 🔄 Running
- **GPU:** 2 (100% util, 4.5 GB / 24 GB)
- **PID:** 3008822
- **Progress:** Encoding corpus for clapnq domain (~3% complete)
- **Fix Applied:** ✅ Using original query embedding for cosine similarity
- **Expected:** Non-zero results

### 2. **best_paper_hierarchical_routing**
- **Status:** 🔄 Running
- **GPU:** 4 (100% util, 5.6 GB / 24 GB)
- **PID:** 3002894
- **Progress:** Training in progress (batch 391/391 completed for current phase)
- **Last Activity:** Encoding Batch 2/4

### 3. **best_paper_meta_learning**
- **Status:** 🔄 Running
- **GPU:** 3 (100% util, 12.6 GB / 24 GB)
- **PID:** 3003617
- **Progress:** Training in progress
- **Last Activity:** Epoch 1.0 completed, train_loss: 1.54

### 4. **best_paper_rl_adaptive_retrieval**
- **Status:** 🔄 Running
- **GPU:** 5 (100% util, 5.5 GB / 24 GB)
- **PID:** 2983620
- **Progress:** Training in progress
- **Last Activity:** Encoding Batch 1/4

---

## ⚠️ Completed with Issues (2)

### 1. **phase8_cross_attention_query_document_fixed**
- **Status:** ✅ Complete (but zero results)
- **Results:** Recall@10: 0.0000, nDCG@10: 0.0000
- **Issue:** Started before fix was applied
- **Action Needed:** Re-run with fix applied

### 2. **best_paper_temporal_memory**
- **Status:** ✅ Complete
- **Results:** Recall@10: 0.0006, nDCG@10: 0.0005
- **Issue:** Extremely low scores (near zero)
- **Action Needed:** Investigate implementation

---

## ✅ Recently Completed (Successful)

### 1. **tier1_contrastive_learning** ⭐ Best Performance
- **Results:** 
  - Recall@10: **0.5331**
  - nDCG@10: **0.4576**
- **Status:** Excellent performance

### 2. **best_paper_llm_distillation**
- **Results:**
  - Recall@10: 0.2668
  - nDCG@10: 0.2027
- **Status:** Good performance

### 3. **best_paper_graph_aware_retrieval**
- **Results:**
  - Recall@10: 0.2485
  - nDCG@10: 0.1830
- **Status:** Good performance

---

## 📊 Overall Statistics

- **Total Experiments:** 82
- **✅ Complete:** 15
- **🔄 Running:** 4
- **⏳ Pending:** 65
- **⚠️ Issues:** 2

---

## 🎯 GPU Status

| GPU | Status | Utilization | Memory | Experiment |
|-----|--------|-------------|--------|------------|
| 0 | 🟢 FREE | 0% | 1.5 GB / 24 GB | - |
| 1 | 🟢 FREE | 0% | 264 MB / 24 GB | - |
| 2 | 🔴 BUSY | 100% | 4.5 GB / 24 GB | tier1_cross_attention_rerun_fixed_v2 |
| 3 | 🔴 BUSY | 100% | 12.6 GB / 24 GB | best_paper_meta_learning |
| 4 | 🔴 BUSY | 100% | 5.6 GB / 24 GB | best_paper_hierarchical_routing |
| 5 | 🔴 BUSY | 100% | 5.5 GB / 24 GB | best_paper_rl_adaptive_retrieval |

**Available GPUs:** 0, 1 (can start 2 more experiments)

---

## 🔧 Cross-Attention Fix Status

### Fixed Experiments:
1. ✅ **tier1_cross_attention_rerun_fixed_v2** - Running with fix
2. ⚠️ **phase8_cross_attention_query_document_fixed** - Completed before fix (needs re-run)

### Fix Applied:
- Using original query embedding for cosine similarity (not attention output)
- Smart attention weight detection
- Debug logging enabled

---

## 📝 Next Steps

1. **Monitor** `tier1_cross_attention_rerun_fixed_v2` to verify fix works
2. **Re-run** `phase8_cross_attention_query_document_fixed` with fix applied
3. **Investigate** `best_paper_temporal_memory` low scores
4. **Start new experiments** on available GPUs (0, 1)

---

*Generated: 2025-12-18 20:15*
