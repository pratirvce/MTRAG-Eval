# Next Experiments to Start

**Last Updated:** 2025-12-18 19:40

## Current GPU Status

| GPU | Utilization | Status | Current Use |
|-----|-------------|--------|-------------|
| 0 | 0% | ✅ **FREE** | Available |
| 1 | 100% | 🔄 Running | phase8_cross_attention_query_document_fixed |
| 2 | 69% | 🔄 Running | tier1_cross_attention_rerun_fixed |
| 3 | 0% | ✅ **FREE** | Available |
| 4 | 0% | ✅ **FREE** | Available |
| 5 | 100% | 🔄 Running | Other experiment |

**Available GPUs: 0, 3, 4 (3 GPUs free)**

---

## 🎯 High Priority Experiments (Best Paper Roadmap)

### ✅ Already Completed

- ✅ **best_paper_graph_aware_retrieval** - Complete
- ✅ **best_paper_large_model_finetuning** - Complete
- ✅ **best_paper_learned_rrf** - Complete
- ✅ **best_paper_adversarial_curriculum** - Complete
- ✅ **best_paper_llm_distillation** - Complete
- ✅ **best_paper_multitask_retrieval** - Complete

### Priority 1: Remaining High-Impact Experiments ⭐⭐⭐⭐⭐

1. **best_paper_temporal_memory**
   - **Script:** `train_temporal_memory_tier1.py` ✅ Exists
   - **Expected nDCG@10:** 0.58-0.62
   - **Time Estimate:** 4-6 days
   - **Description:** Temporal Memory Networks (First memory-augmented retrieval)
   - **Status:** ⏳ Pending
   - **Priority:** 🔴 **CRITICAL**

2. **best_paper_meta_learning**
   - **Script:** `train_meta_learning_tier1.py`
   - **Expected nDCG@10:** 0.57-0.61
   - **Time Estimate:** 5-7 days
   - **Description:** Cross-Domain Transfer Learning with Meta-Learning (MAML)
   - **Status:** ⏳ Pending
   - **Priority:** 🔴 **CRITICAL**

3. **best_paper_hierarchical_routing**
   - **Script:** `train_hierarchical_routing_tier1.py`
   - **Expected nDCG@10:** 0.60-0.64
   - **Time Estimate:** 4-6 days
   - **Description:** Hierarchical Multi-Stage with Learned Routing
   - **Status:** ⏳ Pending
   - **Priority:** 🔴 **CRITICAL**

---

## 🔧 Tier 1 Experiments (Lower Priority but Important)

### Priority 3: Additional Novel Methods

6. **tier1_cross_encoder_finetuned**
   - **Script:** `train_cross_encoder_finetuned_tier1.py`
   - **Expected nDCG@10:** 0.49-0.52
   - **Time Estimate:** 3-5 days
   - **Status:** ⏳ Pending
   - **Priority:** 🟡 **MEDIUM**

7. **tier1_hierarchical_multigranularity**
   - **Script:** `train_hierarchical_multigranularity_tier1.py`
   - **Expected nDCG@10:** 0.49-0.52
   - **Time Estimate:** 3-4 days
   - **Status:** ⏳ Pending
   - **Priority:** 🟡 **MEDIUM**

8. **tier1_iterative_refinement_improved**
   - **Script:** `train_iterative_refinement_improved_tier1.py`
   - **Expected nDCG@10:** 0.50-0.54
   - **Time Estimate:** 3-4 days
   - **Status:** ⏳ Pending
   - **Priority:** 🟡 **MEDIUM**

9. **tier1_contrastive_learning**
   - **Script:** `train_contrastive_learning_tier1.py`
   - **Expected nDCG@10:** 0.49-0.52
   - **Time Estimate:** 5-7 days
   - **Status:** ⏳ Pending
   - **Priority:** 🟡 **MEDIUM**

---

## 📋 Recommended Next Steps

### Immediate Actions (Start Now with 3 Free GPUs)

**Option A: Start Top 3 Remaining Priority 1 Experiments** (Recommended)
```bash
# GPU 0: Temporal Memory
venv/bin/python3 train_temporal_memory_tier1.py \
  --experiment_name best_paper_temporal_memory \
  --gpu 0 \
  --output_dir experiments/retrieval/best_paper_temporal_memory \
  --resume > experiments/retrieval/best_paper_temporal_memory/training.log 2>&1 &

# GPU 3: Meta Learning
venv/bin/python3 train_meta_learning_tier1.py \
  --experiment_name best_paper_meta_learning \
  --gpu 3 \
  --output_dir experiments/retrieval/best_paper_meta_learning \
  --resume > experiments/retrieval/best_paper_meta_learning/training.log 2>&1 &

# GPU 4: Hierarchical Routing
venv/bin/python3 train_hierarchical_routing_tier1.py \
  --experiment_name best_paper_hierarchical_routing \
  --gpu 4 \
  --output_dir experiments/retrieval/best_paper_hierarchical_routing \
  --resume > experiments/retrieval/best_paper_hierarchical_routing/training.log 2>&1 &
```

**Option B: Use Auto-Runner** (Automated)
```bash
# The auto-runner will automatically start experiments when GPUs are free
python3 auto_start_top_experiments.py --check-interval 300 --max-parallel 6
```

**Option C: Start Tier 1 Experiments** (If best_paper scripts not ready)
```bash
# GPU 0: Cross-Encoder Finetuned
venv/bin/python3 train_cross_encoder_finetuned_tier1.py \
  --experiment_name tier1_cross_encoder_finetuned \
  --gpu 0 \
  --output_dir experiments/retrieval/tier1_cross_encoder_finetuned \
  --resume > experiments/retrieval/tier1_cross_encoder_finetuned/training.log 2>&1 &

# GPU 3: Hierarchical Multigranularity
venv/bin/python3 train_hierarchical_multigranularity_tier1.py \
  --experiment_name tier1_hierarchical_multigranularity \
  --gpu 3 \
  --output_dir experiments/retrieval/tier1_hierarchical_multigranularity \
  --resume > experiments/retrieval/tier1_hierarchical_multigranularity/training.log 2>&1 &

# GPU 4: Contrastive Learning
venv/bin/python3 train_contrastive_learning_tier1.py \
  --experiment_name tier1_contrastive_learning \
  --gpu 4 \
  --output_dir experiments/retrieval/tier1_contrastive_learning \
  --resume > experiments/retrieval/tier1_contrastive_learning/training.log 2>&1 &
```

---

## ⚠️ Important Notes

1. **Check Script Availability:** Before starting, verify that the training scripts exist:
   ```bash
   ls -la train_graph_aware_tier1.py train_rl_adaptive_tier1.py train_temporal_memory_tier1.py
   ```

2. **Monitor Running Experiments:** The fixed cross-attention experiments are still running and should complete in ~2 hours. Don't start experiments that might conflict.

3. **Resume Capability:** All scripts support `--resume` flag to continue from checkpoints if interrupted.

4. **Auto-Runner:** Consider using `auto_start_top_experiments.py` which automatically:
   - Detects free GPUs
   - Starts experiments by priority
   - Monitors for failures
   - Sends email notifications
   - Retries failed experiments

---

## 📊 Expected Timeline

If starting all Priority 1 experiments now:
- **Graph-Aware:** 4-6 days
- **RL Adaptive:** 5-7 days  
- **Temporal Memory:** 4-6 days

**Total:** ~1 week for all Priority 1 experiments to complete

---

## 🔍 Quick Status Check Commands

```bash
# Check GPU status
nvidia-smi

# Check experiment status
python3 check_all_runs_status.py | grep -E "(best_paper_|tier1_)"

# Check if scripts exist
ls -la train_*_tier1.py | grep -E "(graph|rl|temporal|large|learned)"

# Monitor running experiments
tail -f experiments/retrieval/*/training.log
```

---

## 🎯 Summary

**Recommended Next 3 Experiments:**
1. ✅ **best_paper_temporal_memory** (GPU 0) - Script exists ✅
2. ✅ **best_paper_meta_learning** (GPU 3) - Check if script exists
3. ✅ **best_paper_hierarchical_routing** (GPU 4) - Check if script exists

These are the remaining high-priority experiments for the best paper submission. If their scripts don't exist, start the Tier 1 experiments instead.

