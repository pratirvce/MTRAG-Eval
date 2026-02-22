# Additional Experiments Running on Available GPUs

**Last Updated**: 2025-12-17 08:41  
**GPUs 4-5**: Running additional high-value experiments

---

## ✅ Additional Experiments Started

### 1. **tier2_multistage_3stage** (Multi-Stage Retrieval) 🟢 RUNNING
- **Status**: ✅ Running
- **GPU**: 4
- **Started**: 08:41
- **Expected**: 0.48-0.53 nDCG@10
- **Time**: 2-4 days
- **Log**: `experiments/retrieval/tier2_multistage_3stage/training.log`
- **Implementation**: 
  - Stage 1: Fast dense retrieval (top 100)
  - Stage 2: Cross-encoder reranking (top 50)
  - Stage 3: Final reranking (top 20)

### 2. **tier2_ensemble_advanced** (Advanced Ensemble) 🟢 RUNNING
- **Status**: ✅ Running
- **GPU**: 5
- **Started**: 08:41
- **Expected**: 0.46-0.50 nDCG@10
- **Time**: 1-2 days
- **Log**: `experiments/retrieval/tier2_ensemble_advanced/training.log`
- **Implementation**: Weighted ensemble of multiple models

---

## 📊 Complete GPU Utilization

```
GPU 0: Cross-Attention (Tier 1, Priority 2) - 94% utilization, 6052 MiB
GPU 1: Hierarchical Multi-Granularity (Tier 1, Priority 3) - 100% utilization, 5196 MiB
GPU 2: Iterative Refinement (Tier 1, Priority 4) - 100% utilization, 5694 MiB
GPU 3: Contrastive Learning (Tier 1, Priority 5) - 100% utilization, 13118 MiB
GPU 4: Multi-Stage Retrieval (Tier 2) - Starting, 523 MiB
GPU 5: Advanced Ensemble (Tier 2) - Running, 2120 MiB
```

**Total**: 6 experiments running, all GPUs utilized!

---

## 📁 Monitoring Commands

### Check Additional Experiments
```bash
# Multi-Stage Retrieval
tail -f experiments/retrieval/tier2_multistage_3stage/training.log

# Advanced Ensemble
tail -f experiments/retrieval/tier2_ensemble_advanced/training.log
```

### Check All Experiments
```bash
# View all running processes
ps aux | grep -E "train_.*_tier|train_multistage|train_ensemble" | grep -v grep

# Check GPU usage
watch -n 1 nvidia-smi
```

---

## 🎯 Summary

**All 6 GPUs are now running experiments!**

### Tier 1 Experiments (High Priority):
- ✅ Cross-Encoder: Training completed
- 🟢 Cross-Attention: Running on GPU 0
- 🟢 Hierarchical Multi-Granularity: Running on GPU 1
- 🟢 Iterative Refinement: Running on GPU 2
- 🟢 Contrastive Learning: Running on GPU 3

### Tier 2 Experiments (Additional):
- 🟢 Multi-Stage Retrieval: Running on GPU 4
- 🟢 Advanced Ensemble: Running on GPU 5

**Maximum GPU utilization achieved!** 🚀

