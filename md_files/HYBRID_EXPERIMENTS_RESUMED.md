# Hybrid Experiments Resumed on Free GPUs ✅

**Resumed**: 2025-12-16  
**Status**: 3 hybrid experiments resumed on GPUs 3, 4, 5

---

## 🔄 Resumed Hybrid Experiments

### 1. phase5_hybrid_govt_alpha0.7
- **GPU**: 3
- **Status**: ✅ Running
- **Config**: `experiments/retrieval/phase5_hybrid_govt_alpha0.7/config.json`
- **Log**: `experiments/retrieval/phase5_hybrid_govt_alpha0.7/training.log`

### 2. phase5_hybrid_multi_alpha0.3
- **GPU**: 4
- **Status**: ✅ Running
- **Config**: `experiments/retrieval/phase5_hybrid_multi_alpha0.3/config.json`
- **Log**: `experiments/retrieval/phase5_hybrid_multi_alpha0.3/training.log`

### 3. phase5_hybrid_multi_alpha0.5
- **GPU**: 5
- **Status**: ✅ Running
- **Config**: `experiments/retrieval/phase5_hybrid_multi_alpha0.5/config.json`
- **Log**: `experiments/retrieval/phase5_hybrid_multi_alpha0.5/training.log`

---

## 📊 Current GPU Allocation

| GPU | Utilization | Memory | Experiment |
|-----|-------------|--------|------------|
| 0 | 97% | 9.8 GB | Cross-encoder fine-tuning (Phase 6) |
| 1 | 100% | 4.8 GB | Multi-stage retrieval (Phase 6) |
| 2 | 100% | 4.6 GB | LLM query expansion (Phase 6) |
| 3 | - | - | Hybrid govt alpha0.7 |
| 4 | - | - | Hybrid multi alpha0.3 |
| 5 | - | - | Hybrid multi alpha0.5 |

---

## 📝 Remaining Hybrid Experiments

The following hybrid experiments are still pending (will resume when more GPUs free up):

1. `phase5_hybrid_clapnq_alpha0.7` (was on GPU 0)
2. `phase5_hybrid_govt_alpha0.3` (was on GPU 1)
3. `phase5_hybrid_govt_alpha0.5` (was on GPU 2)
4. `phase5_hybrid_multi_alpha0.7` (was on GPU 0)

**Note**: These will automatically resume when Phase 6 experiments complete and free up GPUs, or you can manually resume them.

---

## 🔧 Monitor Commands

### Check Hybrid Experiments
```bash
ps aux | grep train_hybrid_learned.py | grep -v grep
```

### Monitor Logs
```bash
tail -f experiments/retrieval/phase5_hybrid_*/training.log
```

### Check GPU Usage
```bash
watch -n 1 nvidia-smi
```

---

## ✅ Summary

- **Phase 6 Priority Experiments**: 3 running (GPUs 0, 1, 2)
- **Hybrid Experiments Resumed**: 3 running (GPUs 3, 4, 5)
- **Total Active Experiments**: 6
- **Free GPUs**: 0 (all GPUs in use)

**All GPUs are now fully utilized!** 🚀

