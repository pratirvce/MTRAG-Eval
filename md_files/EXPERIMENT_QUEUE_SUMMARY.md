# Experiment Queue Summary - Quick Reference

**Last Updated**: 2025-12-16 23:55

---

## 🎯 **Next Experiments to Run (Priority Order)**

### **1. ⭐⭐⭐⭐ Hierarchical Multi-Granularity Retrieval** (NEXT TO START)
- **Status**: ⏸️ Needs implementation
- **Expected**: 0.49-0.52 nDCG@10
- **Time**: 3-4 days to implement, 12-16 hours to run
- **GPU**: Start on GPU 1 (available now) or next free GPU
- **Action**: Implement `train_hierarchical_multigranularity_retrieval.py`

### **2. ⭐⭐⭐⭐ Contrastive Learning** (AFTER #1)
- **Status**: ⏸️ Needs implementation
- **Expected**: 0.49-0.52 nDCG@10
- **Time**: 5-7 days to implement (includes training), 12-16 hours to run
- **GPU**: Start when GPU available (after Hierarchical or in parallel)
- **Action**: Implement `train_contrastive_conversation_doc.py`

### **3. ⭐⭐⭐ LLM Query Rewriting** (OPTIONAL)
- **Status**: ⏸️ Needs implementation
- **Expected**: 0.51-0.54 nDCG@10
- **Time**: 2-3 days to implement, 16-24 hours to run (slower due to API)
- **Requires**: LLM API access (GPT-4)
- **Action**: Implement if time permits and API available

---

## 📊 **Current GPU Status**

| GPU | Status | Experiment | Available In |
|-----|--------|------------|--------------|
| 0 | 🔄 Running | Cross-Encoder Evaluation | ~2-3 hours |
| 1 | 🔄 Running | Unknown | Unknown |
| 2 | 🔄 Running | Cross-Attention Query-Document | ~4-8 hours |
| 3 | 🔄 Running | Iterative Refinement | ~2-4 hours |
| 4 | 🔄 Running | Ensemble Domain-Specific | ~1-2 hours |
| 5 | 🔄 Running | Multi-Stage Fine-Tuned | ~1-2 hours |

---

## 🚀 **Quick Start Commands**

### **Check Queue Status**:
```bash
cat EXPERIMENT_QUEUE.md
```

### **Start Next Experiment Automatically**:
```bash
./start_next_experiment_in_queue.sh
```

### **Check GPU Availability**:
```bash
nvidia-smi
```

### **Monitor Running Experiments**:
```bash
ps aux | grep "python train_" | grep -v grep
```

---

## ✅ **Recommended Action**

**Immediate**: Implement **Hierarchical Multi-Granularity Retrieval** (ready to start when GPU becomes available)

**Next**: Implement **Contrastive Learning** when next GPU becomes available (GPU 4/5 in ~1-2 hours)

---

**Full details**: See `EXPERIMENT_QUEUE.md`

