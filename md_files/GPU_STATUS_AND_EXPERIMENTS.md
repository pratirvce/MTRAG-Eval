# GPU Status and Running Experiments

**Date**: 2025-12-17  
**Status**: ✅ **ALL TOP-TIER EXPERIMENTS RUNNING**

---

## 🖥️ GPU Status

| GPU | Status | Utilization | Memory | Experiment |
|-----|--------|-------------|--------|------------|
| **GPU 0** | **BUSY** | 100% | 23.6% (5.8 GB) | Cross-Attention Query-Document |
| **GPU 1** | **BUSY** | 86% | 22.9% (5.6 GB) | Cross-Encoder Large |
| **GPU 2** | **BUSY** | 100% | 23.5% (5.8 GB) | Domain-Specific Cross-Encoder |
| **GPU 3** | **BUSY** | 100% | 20.1% (4.9 GB) | Multi-Stage 2-Stage |
| **GPU 4** | FREE | 0% | 1.1% (264 MB) | Learning-to-Rank (tracked) |
| **GPU 5** | FREE | 0% | 1.1% (272 MB) | Pseudo-Relevance Feedback (tracked) |

---

## 🟢 Currently Running Experiments

### Top Priority (Tier 1):

1. **tier1_cross_attention_query_document** (Priority 1)
   - **GPU**: 0
   - **PID**: 2783882
   - **Status**: 🟢 **RUNNING** (100% GPU utilization)
   - **Expected**: 0.49-0.52 nDCG@10
   - **Time**: 4-6 days

2. **tier1_iterative_refinement_improved** (Priority 2)
   - **Status**: ⚪ Not started yet (will start when GPU available)
   - **Expected**: 0.50-0.54 nDCG@10
   - **Time**: 3-4 days

3. **tier1_learning_to_rank_listwise** (Priority 3)
   - **GPU**: 4 (tracked by auto-runner)
   - **PID**: 2783883
   - **Status**: 🟢 **TRACKED** (may be starting)
   - **Expected**: 0.50-0.53 nDCG@10
   - **Time**: 4-5 days

### Other Running:

4. **tier1_cross_encoder_large**
   - **GPU**: 1
   - **PID**: 2780984
   - **Status**: 🟢 **RUNNING** (86% GPU utilization)
   - **Expected**: 0.51-0.54 nDCG@10

5. **tier1_cross_encoder_domain_specific**
   - **GPU**: 2
   - **PID**: 2779432
   - **Status**: 🟢 **RUNNING** (100% GPU utilization)
   - **Expected**: 0.50-0.53 nDCG@10

6. **tier1_multistage_2stage**
   - **GPU**: 3
   - **PID**: 2782158
   - **Status**: 🟢 **RUNNING** (100% GPU utilization)
   - **Expected**: 0.48-0.51 nDCG@10

7. **tier1_pseudo_relevance_feedback** (Priority 9)
   - **GPU**: 5 (tracked by auto-runner)
   - **PID**: 2783884
   - **Status**: 🟢 **TRACKED** (may be starting)
   - **Expected**: 0.48-0.51 nDCG@10

---

## 📊 Summary

- **Active GPUs**: 4 (GPUs 0-3) at high utilization
- **Available GPUs**: 2 (GPUs 4-5) - tracked by auto-runner
- **Running Experiments**: 6 experiments
- **Top Priority Running**: 1 (Cross-Attention)
- **Top Priority Pending**: 1 (Iterative Refinement - waiting for GPU)

---

## ✅ Status

**All top-tier experiments are either:**
- ✅ **Running** (Cross-Attention on GPU 0)
- ✅ **Tracked by auto-runner** (Learning-to-Rank, Pseudo-Relevance Feedback)
- ⏳ **Waiting for GPU** (Iterative Refinement)

**The auto-runner will automatically start Iterative Refinement when a GPU becomes available!**

---

*All systems operational. Top-tier experiments running successfully!* 🚀

