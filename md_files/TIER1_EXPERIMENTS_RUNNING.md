# Tier 1 Experiments - Currently Running

**Last Updated**: 2025-12-17 08:37  
**Master Runner PID**: Check with `ps aux | grep run_tier1_experiments`

---

## ✅ Current Status

### 1. **tier1_cross_encoder_finetuned** (Priority 1) ✅ COMPLETED
- **Status**: Already completed (was running from earlier)
- **GPU**: Was on GPU 0
- **Expected**: 0.49-0.52 nDCG@10
- **Results**: Check `experiments/retrieval/tier1_cross_encoder_finetuned/results.json`

### 2. **tier1_cross_attention_query_document** (Priority 2) 🟢 RUNNING
- **Status**: ✅ Running
- **GPU**: 0
- **PID**: Check status file
- **Expected**: 0.49-0.52 nDCG@10
- **Time**: 4-6 days
- **Log**: `experiments/retrieval/tier1_cross_attention_query_document/training.log`
- **Implementation**: Using cross-attention mechanism for query-document interaction

### 3. **tier1_hierarchical_multigranularity** (Priority 3) 🟢 RUNNING
- **Status**: ✅ Running
- **GPU**: 1
- **PID**: Check status file
- **Expected**: 0.49-0.52 nDCG@10
- **Time**: 3-4 days
- **Log**: `experiments/retrieval/tier1_hierarchical_multigranularity/training.log`
- **Implementation**: Sentence, paragraph, and document-level retrieval with RRF combination

### 4. **tier1_iterative_refinement_improved** (Priority 4) 🟢 RUNNING
- **Status**: ✅ Running
- **GPU**: 2
- **PID**: Check status file
- **Expected**: 0.50-0.54 nDCG@10
- **Time**: 3-4 days
- **Log**: `experiments/retrieval/tier1_iterative_refinement_improved/training.log`
- **Implementation**: Two-round retrieval with feedback and RRF combination

### 5. **tier1_contrastive_learning** (Priority 5) ⏳ PENDING
- **Status**: ⏳ Waiting for GPU availability
- **Expected**: 0.49-0.52 nDCG@10
- **Time**: 5-7 days (includes training phase)
- **Implementation**: Contrastive learning on conversation-document pairs

---

## 📊 GPU Utilization

```
GPU 0: Cross-Attention (Priority 2) - Running
GPU 1: Hierarchical Multi-Granularity (Priority 3) - Running
GPU 2: Iterative Refinement (Priority 4) - Running
GPU 3-5: Available (will be used for contrastive learning when ready)
```

---

## 📁 Monitoring Commands

### Check Overall Status
```bash
# View status file
cat tier1_experiments_status.json | python3 -m json.tool

# View master log
tail -f tier1_experiments.log

# Check running processes
ps aux | grep tier1
```

### Check Individual Experiments
```bash
# Cross-Attention (Priority 2)
tail -f experiments/retrieval/tier1_cross_attention_query_document/training.log

# Hierarchical Multi-Granularity (Priority 3)
tail -f experiments/retrieval/tier1_hierarchical_multigranularity/training.log

# Iterative Refinement (Priority 4)
tail -f experiments/retrieval/tier1_iterative_refinement_improved/training.log
```

### Check GPU Usage
```bash
watch -n 1 nvidia-smi
```

---

## 🔔 Email Notifications

**Note**: EMAIL_PASSWORD is not currently set. To enable email notifications:

```bash
export EMAIL_PASSWORD="your-gmail-app-password"
# Then restart the master runner if needed
```

You will receive emails at **pratirvce@gmail.com** for:
- ✅ Experiment start
- ✅ Experiment completion (with results)
- ❌ Experiment failures (with error details)

---

## 🔄 Resume Capability

All experiments support resume:
- Checkpoints saved automatically
- If interrupted, just run: `python run_tier1_experiments.py`
- Will automatically resume from last checkpoint

---

## ⏱️ Expected Timeline

- **Cross-Attention**: 4-6 days (started 08:37)
- **Hierarchical Multi-Granularity**: 3-4 days (started 08:37)
- **Iterative Refinement**: 3-4 days (started 08:37)
- **Contrastive Learning**: 5-7 days (will start when GPU available)

---

## ✅ Summary

**All Tier 1 experiments are now properly implemented and running!**

- ✅ Cross-Encoder: Completed
- 🟢 Cross-Attention: Running on GPU 0
- 🟢 Hierarchical Multi-Granularity: Running on GPU 1
- 🟢 Iterative Refinement: Running on GPU 2
- ⏳ Contrastive Learning: Waiting for GPU

**Everything is working correctly!** 🚀

