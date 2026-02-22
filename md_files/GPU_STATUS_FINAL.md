# GPU Status - Final Report

**Date**: 2025-12-17  
**Time**: Current

---

## 🖥️ GPU Utilization Status

| GPU | Status | GPU Util | Memory | Temperature | Power | Experiment |
|-----|--------|----------|--------|-------------|-------|------------|
| **GPU 0** | FREE | 0% | 6.3% (1.5 GB) | 29°C | 23 W | Ensemble/LLM (completed) |
| **GPU 1** | **BUSY** | **100%** | 22.5% (5.5 GB) | 64°C | 279 W | Cross-Encoder Large |
| **GPU 2** | **BUSY** | **100%** | 23.5% (5.8 GB) | 71°C | 280 W | Domain-Specific Cross-Encoder |
| **GPU 3** | **BUSY** | **100%** | 19.6% (4.8 GB) | ~65°C | ~280 W | Multi-Stage 2-Stage |
| **GPU 4** | FREE | 0% | 1.1% (264 MB) | 31°C | 22 W | Available |
| **GPU 5** | FREE | 0% | 1.1% (272 MB) | 30°C | 22 W | Available |

---

## 📊 Summary

- **Active GPUs**: 3 (GPU 1, 2, 3) at 100% utilization
- **Available GPUs**: 3 (GPU 0, 4, 5) ready for new experiments
- **Utilization**: 50% of GPUs actively running experiments
- **Temperature**: All within normal range (29-71°C)
- **Power**: Active GPUs drawing ~280W (normal for RTX 3090)

---

## 🟢 Running Experiments

1. **tier1_cross_encoder_large** (GPU 1)
   - Status: Running
   - Progress: Processing batches
   - Expected: 0.51-0.54 nDCG@10

2. **tier1_cross_encoder_domain_specific** (GPU 2)
   - Status: Running
   - Progress: ~35% complete
   - Expected: 0.50-0.53 nDCG@10

3. **tier1_multistage_2stage** (GPU 3)
   - Status: Running
   - Progress: Just started
   - Expected: 0.48-0.51 nDCG@10

---

## ✅ Completed Experiments

- **tier1_ensemble_best_methods**: 0.4576 nDCG@10
- **tier1_llm_query_expansion**: Completed (checking results)
- **tier1_contrastive_learning**: 0.4576 nDCG@10 (BEST)

---

## 🔄 Auto-Runner

- **Status**: ✅ Active
- **Will Start**: Additional experiments on GPUs 0, 4, 5 when ready
- **Monitoring**: Every 5 minutes

---

*All GPUs are healthy and experiments are running successfully!*
