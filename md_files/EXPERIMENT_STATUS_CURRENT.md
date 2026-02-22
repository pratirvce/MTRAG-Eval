# Current Experiment Status

**Last Updated**: 2025-12-17  
**Total Experiments**: 7 running + completed experiments

---

## 🟢 **Currently Running Experiments**

### **Tier 1 Experiments (High Priority)**

#### 1. **tier1_cross_encoder_finetuned** (Priority 1)
- **Status**: ✅ Training Completed, Evaluation Pending
- **GPU**: Was on GPU 0
- **Model**: Saved to `models/tier1_cross_encoder_finetuned`
- **Expected**: 0.49-0.52 nDCG@10
- **Next**: Needs evaluation on test set

#### 2. **tier1_cross_attention_query_document** (Priority 2)
- **Status**: 🟢 Running
- **GPU**: 0
- **Started**: 08:37
- **Expected**: 0.49-0.52 nDCG@10
- **Time**: 4-6 days
- **Log**: `experiments/retrieval/tier1_cross_attention_query_document/training.log`

#### 3. **tier1_hierarchical_multigranularity** (Priority 3)
- **Status**: 🟢 Running
- **GPU**: 1
- **Started**: 08:37
- **Expected**: 0.49-0.52 nDCG@10
- **Time**: 3-4 days
- **Log**: `experiments/retrieval/tier1_hierarchical_multigranularity/training.log`

#### 4. **tier1_iterative_refinement_improved** (Priority 4)
- **Status**: 🟢 Running
- **GPU**: 2
- **Started**: 08:37
- **Expected**: 0.50-0.54 nDCG@10
- **Time**: 3-4 days
- **Log**: `experiments/retrieval/tier1_iterative_refinement_improved/training.log`

#### 5. **tier1_contrastive_learning** (Priority 5)
- **Status**: 🟢 Running
- **GPU**: 3
- **Started**: 08:39
- **Expected**: 0.49-0.52 nDCG@10
- **Time**: 5-7 days (includes training phase)
- **Log**: `experiments/retrieval/tier1_contrastive_learning/training.log`
- **Phase**: Training (Epoch 1/3)

### **Tier 2 Experiments (Additional)**

#### 6. **tier2_multistage_3stage** (Multi-Stage Retrieval)
- **Status**: 🟢 Running
- **GPU**: 4
- **Started**: 08:42
- **Expected**: 0.48-0.53 nDCG@10
- **Time**: 2-4 days
- **Log**: `experiments/retrieval/tier2_multistage_3stage/training.log`

#### 7. **tier2_ensemble_advanced** (Advanced Ensemble)
- **Status**: 🟢 Running
- **GPU**: 5
- **Started**: 08:41
- **Expected**: 0.46-0.50 nDCG@10
- **Time**: 1-2 days
- **Log**: `experiments/retrieval/tier2_ensemble_advanced/training.log`

---

## 📊 **GPU Utilization**

| GPU | Experiment | Utilization | Memory Used | Status |
|-----|------------|-------------|-------------|--------|
| 0 | Cross-Attention (Tier 1) | Check below | Check below | 🟢 Running |
| 1 | Hierarchical Multi-Granularity (Tier 1) | Check below | Check below | 🟢 Running |
| 2 | Iterative Refinement (Tier 1) | Check below | Check below | 🟢 Running |
| 3 | Contrastive Learning (Tier 1) | Check below | Check below | 🟢 Running |
| 4 | Multi-Stage Retrieval (Tier 2) | Check below | Check below | 🟢 Running |
| 5 | Advanced Ensemble (Tier 2) | Check below | Check below | 🟢 Running |

*Run `nvidia-smi` for current GPU stats*

---

## ✅ **Completed Experiments (Best Results)**

| Rank | Experiment | Recall@10 | nDCG@10 | Status |
|------|------------|-----------|---------|--------|
| 🥇 | phase5_query_expansion_govt | 0.5317 | **0.4515** | ✅ Completed |
| 🥈 | phase5_ensemble_domain_specific | 0.5356 | 0.4434 | ✅ Completed |
| 🥉 | phase5_ensemble_weighted | 0.5284 | 0.4370 | ✅ Completed |
| 4 | phase5_query_expansion_clapnq | 0.4999 | 0.4186 | ✅ Completed |
| 5 | phase5_query_expansion_multi | 0.5099 | 0.4098 | ✅ Completed |

**Current Best**: 0.4515 nDCG@10 (Query Expansion Govt)  
**SOTA Target**: 0.5400 nDCG@10  
**Gap**: -0.0885 (needs +8.85% improvement)

---

## 📁 **Monitoring Commands**

### Check Running Processes
```bash
ps aux | grep -E "train_.*_tier|train_multistage|train_ensemble" | grep -v grep
```

### Check GPU Usage
```bash
nvidia-smi
# Or
watch -n 1 nvidia-smi
```

### Check Individual Experiment Logs
```bash
# Cross-Attention
tail -f experiments/retrieval/tier1_cross_attention_query_document/training.log

# Hierarchical
tail -f experiments/retrieval/tier1_hierarchical_multigranularity/training.log

# Iterative Refinement
tail -f experiments/retrieval/tier1_iterative_refinement_improved/training.log

# Contrastive Learning
tail -f experiments/retrieval/tier1_contrastive_learning/training.log

# Multi-Stage
tail -f experiments/retrieval/tier2_multistage_3stage/training.log

# Ensemble
tail -f experiments/retrieval/tier2_ensemble_advanced/training.log
```

### Check Status File
```bash
cat tier1_experiments_status.json | python3 -m json.tool
```

### Check Master Log
```bash
tail -f tier1_experiments.log
```

---

## ⏱️ **Expected Completion Timeline**

- **tier2_ensemble_advanced**: 1-2 days (started 08:41)
- **tier2_multistage_3stage**: 2-4 days (started 08:42)
- **tier1_hierarchical_multigranularity**: 3-4 days (started 08:37)
- **tier1_iterative_refinement_improved**: 3-4 days (started 08:37)
- **tier1_cross_attention_query_document**: 4-6 days (started 08:37)
- **tier1_contrastive_learning**: 5-7 days (started 08:39, includes training)

---

## 🎯 **Summary**

**Running**: 6 experiments (4 Tier 1 + 2 Tier 2)  
**Completed Training**: 1 experiment (cross-encoder, needs evaluation)  
**Completed & Evaluated**: Multiple Phase 1-7 experiments  
**Best Result So Far**: 0.4515 nDCG@10  
**All GPUs**: Fully utilized ✅

**Status**: All systems operational, experiments progressing normally! 🚀
