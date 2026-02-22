# Running Experiments Status Report

**Generated:** 2025-12-18 20:06  
**Total Running:** 6 experiments

---

## 🔄 Currently Running Experiments

### 1. **phase8_cross_attention_query_document_fixed**
- **Status:** 🟢 Running
- **PID:** 2999647
- **GPU:** 1 (100% utilization, 5.3 GB memory)
- **CPU:** 316%
- **Started:** 2025-12-18 19:31
- **Progress:** 
  - clapnq: 🔄 In Progress (Encoding corpus: 100% complete, batch 566/566)
  - fiqa: 🔄 In Progress
  - govt: 🔄 In Progress
- **Last Activity:** Completed corpus encoding for clapnq domain
- **Estimated Time Remaining:** ~1-2 hours (3 domains remaining)

### 2. **tier1_cross_attention_rerun_fixed** ⚠️
- **Status:** ✅ Just Completed (but with ZERO results!)
- **PID:** 2999721 (process finished)
- **GPU:** 2 (now free)
- **Results:** 
  - Recall@10: 0.0000
  - nDCG@10: 0.0000
- **Issue:** Experiment completed but produced zero results - the fix may not have worked correctly
- **Action Needed:** Investigate why results are still zero

### 3. **best_paper_hierarchical_routing**
- **Status:** 🟢 Running
- **PID:** 3002894
- **GPU:** 4 (100% utilization, 5.6 GB memory)
- **CPU:** 319%
- **Started:** 2025-12-18 19:42
- **Progress:** Encoding corpus (batch 55/391 for clapnq domain)
- **Last Activity:** Encoding corpus documents at ~1.01s/it
- **Estimated Time Remaining:** ~4-6 days

### 4. **best_paper_meta_learning**
- **Status:** 🟢 Running
- **PID:** 3003617
- **GPU:** 3 (100% utilization, 11.9 GB memory)
- **CPU:** 239%
- **Started:** 2025-12-18 19:45
- **Progress:** Loading corpus for clapnq domain
- **Last Activity:** Evaluating meta-learned model, processing clapnq domain
- **Estimated Time Remaining:** ~5-7 days

### 5. **best_paper_rl_adaptive_retrieval**
- **Status:** 🟢 Running
- **PID:** 2983620
- **GPU:** 5 (100% utilization, 5.5 GB memory)
- **CPU:** 339%
- **Started:** Earlier (PID suggests it's been running longer)
- **Progress:** Unknown (check logs)
- **Estimated Time Remaining:** ~5-7 days

### 6. **tier1_learning_to_rank_listwise**
- **Status:** 🟢 Running
- **PID:** 3001949 (from earlier check) / 3006204 (new process)
- **GPU:** 0 (100% utilization, 6.7 GB memory)
- **CPU:** 287%
- **Progress:** Encoding corpus (batch 356/391 for clapnq domain, ~91% complete)
- **Last Activity:** Encoding corpus at ~1.41s/it
- **Estimated Time Remaining:** ~4-5 days

---

## 📊 GPU Utilization Summary

| GPU | Utilization | Memory Used | Memory Total | Experiment |
|-----|-------------|-------------|--------------|------------|
| 0 | 100% | 6.7 GB | 24.6 GB | tier1_learning_to_rank_listwise |
| 1 | 100% | 5.3 GB | 24.6 GB | phase8_cross_attention_query_document_fixed |
| 2 | 0% | 0.3 GB | 24.6 GB | ✅ **FREE** (tier1_cross_attention_rerun_fixed completed) |
| 3 | 100% | 11.9 GB | 24.6 GB | best_paper_meta_learning |
| 4 | 100% | 5.6 GB | 24.6 GB | best_paper_hierarchical_routing |
| 5 | 100% | 5.5 GB | 24.6 GB | best_paper_rl_adaptive_retrieval |

**Available GPUs:** 1 (GPU 2)

---

## ⚠️ Issues Detected

### Critical Issue: tier1_cross_attention_rerun_fixed
- **Problem:** Experiment completed but produced **all-zero results** (Recall@10: 0.0000, nDCG@10: 0.0000)
- **Possible Causes:**
  1. The cross-attention fix may not have been applied correctly
  2. There may be another bug in the score computation
  3. The safety fallback to cosine similarity may not be working
- **Action Required:** 
  - Check the log file: `experiments/retrieval/tier1_cross_attention_rerun_fixed/training.log`
  - Verify the fix was applied correctly
  - Check if scores are being computed but not saved properly

---

## 📈 Progress Summary

| Experiment | Domain Progress | Overall Status |
|------------|----------------|----------------|
| phase8_cross_attention_query_document_fixed | clapnq: 100% encoded, fiqa/govt/cloud: pending | ~25% complete |
| tier1_cross_attention_rerun_fixed | All domains: completed (but zero results) | ❌ Failed |
| best_paper_hierarchical_routing | clapnq: ~14% encoded | ~3% complete |
| best_paper_meta_learning | clapnq: loading corpus | ~1% complete |
| best_paper_rl_adaptive_retrieval | Unknown | Running |
| tier1_learning_to_rank_listwise | clapnq: ~91% encoded | ~23% complete |

---

## 🔍 Quick Commands

```bash
# Check all running experiments
ps aux | grep "train_.*tier1.py" | grep -v grep

# Check GPU status
nvidia-smi

# View specific experiment logs
tail -f experiments/retrieval/phase8_cross_attention_query_document_fixed/training.log
tail -f experiments/retrieval/best_paper_hierarchical_routing/training.log
tail -f experiments/retrieval/best_paper_meta_learning/training.log

# Check fixed experiments status
python3 monitor_fixed_experiments.py --once

# Check comprehensive status
python3 check_all_runs_status.py
```

---

## 📝 Notes

1. **tier1_cross_attention_rerun_fixed** completed but with zero results - this needs immediate investigation
2. **phase8_cross_attention_query_document_fixed** is progressing well - should complete in ~1-2 hours
3. **GPU 2 is now free** - can be used for new experiments
4. All other experiments are running normally and making progress

---

*Last Updated: 2025-12-18 20:06*
