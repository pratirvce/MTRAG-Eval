# Running Experiments Status Report

**Generated:** 2025-12-18 22:45  
**Total Experiments:** 89  
**Running:** 6  
**In Progress (with partial completion):** 1

---

## 🔄 Currently Running Experiments (6)

### 1. `best_paper_temporal_memory`
- **Status:** 🔄 Running
- **GPU:** 3
- **PID:** 3038375
- **CPU Usage:** 171%
- **Memory:** 0.2% (2.5 GB)
- **Runtime:** ~57 minutes
- **Started:** 2025-12-18 22:45
- **Description:** Temporal Memory Networks for conversation context
- **Note:** ⚠️ GPU conflict - sharing GPU 3 with `best_paper_meta_learning_fixed`

---

### 2. `tier1_cross_encoder_domain_specific`
- **Status:** 🔄 Running
- **GPU:** 4
- **PID:** 3028756
- **CPU Usage:** 238%
- **Memory:** 0.1% (1.8 GB)
- **Runtime:** ~277 minutes (~4.6 hours)
- **Started:** 2025-12-18 21:23
- **Description:** Cross-encoder reranking with domain-specific models

---

### 3. `best_paper_rl_adaptive_retrieval_fixed`
- **Status:** 🔄 Running
- **GPU:** 2
- **PID:** 3032761
- **CPU Usage:** 276%
- **Memory:** 0.1% (1.5 GB)
- **Runtime:** ~263 minutes (~4.4 hours)
- **Started:** 2025-12-18 21:44
- **Description:** Reinforcement Learning for Adaptive Retrieval (Fixed version)
- **Fix Applied:** Heuristic-based strategy selection (replacing untrained RL agent)

---

### 4. `best_paper_meta_learning_fixed`
- **Status:** 🔄 Running
- **GPU:** 3
- **PID:** 3033035
- **CPU Usage:** 156%
- **Memory:** 0.5% (5.9 GB)
- **Runtime:** ~149 minutes (~2.5 hours)
- **Started:** 2025-12-18 21:44
- **Description:** Cross-Domain Transfer Learning with Meta-Learning (MAML) (Fixed version)
- **Fix Applied:** Increased training examples (500) and epochs (3 for meta-training, 2 for adaptation)
- **Note:** ⚠️ GPU conflict - sharing GPU 3 with `best_paper_temporal_memory`

---

### 5. `best_paper_hierarchical_routing_fixed`
- **Status:** 🔄 Running
- **GPU:** 1
- **PID:** 3032592
- **CPU Usage:** 240%
- **Memory:** 0.1% (1.5 GB)
- **Runtime:** ~229 minutes (~3.8 hours)
- **Started:** 2025-12-18 21:44
- **Description:** Hierarchical Multi-Stage with Learned Routing (Fixed version)
- **Fix Applied:** Heuristic-based routing strategy (replacing untrained router) + fixed indentation bug

---

### 6. `best_paper_rl_adaptive_retrieval` (OLD VERSION)
- **Status:** 🔄 Running
- **GPU:** 5
- **PID:** 2983620
- **CPU Usage:** 329%
- **Memory:** 0.1% (1.8 GB)
- **Runtime:** ~1099 minutes (~18.3 hours)
- **Started:** 2025-12-17 17:45
- **Description:** Reinforcement Learning for Adaptive Retrieval (Original version - should be stopped)
- **Note:** ⚠️ **OLD VERSION** - Should be stopped after fixed version completes

---

## 📊 In Progress Experiments (with partial completion)

### `tier1_cross_attention_query_document`
- **Status:** 🔄 In Progress (25% complete)
- **Progress:**
  - ✅ **cloud:** Complete
  - 🔄 **govt:** In Progress
  - 🔄 **clapnq:** In Progress
  - 🔄 **fiqa:** In Progress
- **Last Update:** 2025-12-18 03:19
- **Description:** Cross-attention between query and document embeddings

---

## 🖥️ GPU Utilization

| GPU | Model | Utilization | Memory Used | Memory Total | Status |
|-----|-------|-------------|-------------|--------------|--------|
| 0 | RTX 3090 | 100% | 6.7 GB | 24 GB | 🔄 In Use |
| 1 | RTX 3090 | 100% | 5.5 GB | 24 GB | 🔄 In Use (Hierarchical Routing) |
| 2 | RTX 3090 | 100% | 5.4 GB | 24 GB | 🔄 In Use (RL Adaptive Fixed) |
| 3 | RTX 3090 | 100% | 10.1 GB | 24 GB | ⚠️ **CONFLICT** (Temporal Memory + Meta-Learning) |
| 4 | RTX 3090 | 100% | 10.0 GB | 24 GB | 🔄 In Use (Cross-Encoder) |
| 5 | RTX 3090 | 100% | 5.4 GB | 24 GB | 🔄 In Use (RL Adaptive OLD) |

**Note:** All GPUs are at 100% utilization, indicating experiments are actively running.

---

## ⚠️ Issues & Recommendations

### 1. GPU Conflict on GPU 3
- **Issue:** Both `best_paper_temporal_memory` and `best_paper_meta_learning_fixed` are running on GPU 3
- **Impact:** Both processes may be competing for GPU resources, potentially slowing down both
- **Recommendation:** 
  - Monitor if both are making progress
  - Consider stopping one and moving it to a different GPU if available
  - Or wait for one to complete

### 2. Old Version Still Running
- **Issue:** `best_paper_rl_adaptive_retrieval` (old version) is still running on GPU 5
- **Impact:** Wasting GPU resources on an outdated experiment
- **Recommendation:**
  - Stop after `best_paper_rl_adaptive_retrieval_fixed` completes
  - Use the script: `./stop_old_experiments.sh`

### 3. Long-Running Experiment
- **Issue:** `best_paper_rl_adaptive_retrieval` has been running for ~18.3 hours
- **Impact:** May be stuck or taking very long
- **Recommendation:**
  - Check logs for errors
  - Verify it's making progress

---

## 📈 Expected Completion Times

Based on current runtime and typical experiment durations:

| Experiment | Expected Completion | Notes |
|------------|---------------------|-------|
| `best_paper_temporal_memory` | ~2-4 hours | Just started |
| `tier1_cross_encoder_domain_specific` | ~1-2 hours | Already running 4.6 hours |
| `best_paper_rl_adaptive_retrieval_fixed` | ~1-2 hours | Already running 4.4 hours |
| `best_paper_meta_learning_fixed` | ~1-2 hours | Already running 2.5 hours |
| `best_paper_hierarchical_routing_fixed` | ~1-2 hours | Already running 3.8 hours |
| `best_paper_rl_adaptive_retrieval` (OLD) | Unknown | Running 18.3 hours - may be stuck |

---

## ✅ Completed Experiments (Recent)

- `tier1_cross_attention_fixed_v3` - ✅ Complete (2025-12-18 22:59)
- `tier1_cross_attention_rerun_fixed_v2` - ✅ Complete (2025-12-18 20:48)
- `best_paper_meta_learning_fixed` - ✅ Complete (previous run)
- `tier1_iterative_refinement_improved` - ✅ Complete (2025-12-17 09:40)
- `tier2_multistage_3stage` - ✅ Complete (2025-12-17 09:17)

---

## 🎯 Next Steps

1. **Monitor GPU 3 conflict** - Check if both experiments are making progress
2. **Stop old RL Adaptive** - After fixed version completes
3. **Check long-running experiment** - Verify `best_paper_rl_adaptive_retrieval` is not stuck
4. **Wait for fixed experiments** - All fixed versions should complete soon
5. **Re-run Hierarchical Routing** - After the current fixed version completes (if needed)

---

## 📝 Summary

- **6 experiments currently running** across 6 GPUs
- **All GPUs at 100% utilization** - experiments are actively processing
- **1 GPU conflict** (GPU 3) - two experiments sharing resources
- **1 old version** still running - should be stopped
- **1 experiment in progress** with partial completion (25%)

**Overall Status:** ✅ All systems running, but some optimization needed (GPU conflict, old version)

---

*Last Updated: 2025-12-18 22:45*

