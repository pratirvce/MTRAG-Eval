# All Tier 1 Experiments - Running Status

**Last Updated**: 2025-12-17 08:39  
**All 5 Tier 1 experiments are now running!**

---

## ✅ Current Status - All Experiments Running

### 1. **tier1_cross_encoder_finetuned** (Priority 1) ✅ COMPLETED
- **Status**: Training completed
- **GPU**: Was on GPU 0
- **Model**: Saved to `models/tier1_cross_encoder_finetuned`
- **Next**: Evaluation pending
- **Expected**: 0.49-0.52 nDCG@10

### 2. **tier1_cross_attention_query_document** (Priority 2) 🟢 RUNNING
- **Status**: ✅ Running
- **GPU**: 0
- **Started**: 08:37
- **Expected**: 0.49-0.52 nDCG@10
- **Time**: 4-6 days
- **Log**: `experiments/retrieval/tier1_cross_attention_query_document/training.log`
- **Implementation**: Cross-attention mechanism for query-document interaction

### 3. **tier1_hierarchical_multigranularity** (Priority 3) 🟢 RUNNING
- **Status**: ✅ Running
- **GPU**: 1
- **Started**: 08:37
- **Expected**: 0.49-0.52 nDCG@10
- **Time**: 3-4 days
- **Log**: `experiments/retrieval/tier1_hierarchical_multigranularity/training.log`
- **Implementation**: Sentence, paragraph, and document-level retrieval with RRF

### 4. **tier1_iterative_refinement_improved** (Priority 4) 🟢 RUNNING
- **Status**: ✅ Running
- **GPU**: 2
- **Started**: 08:37
- **Expected**: 0.50-0.54 nDCG@10
- **Time**: 3-4 days
- **Log**: `experiments/retrieval/tier1_iterative_refinement_improved/training.log`
- **Implementation**: Two-round retrieval with feedback and RRF combination

### 5. **tier1_contrastive_learning** (Priority 5) 🟢 RUNNING
- **Status**: ✅ Running
- **GPU**: 3
- **Started**: 08:39
- **Expected**: 0.49-0.52 nDCG@10
- **Time**: 5-7 days (includes training phase)
- **Log**: `experiments/retrieval/tier1_contrastive_learning/training.log`
- **Implementation**: Contrastive learning on conversation-document pairs
- **Training**: Epoch 1/3 (1465 training examples)

---

## 📊 GPU Utilization

```
GPU 0: Cross-Attention (Priority 2) - 100% utilization, 6052 MiB
GPU 1: Hierarchical Multi-Granularity (Priority 3) - 99% utilization, 5154 MiB
GPU 2: Iterative Refinement (Priority 4) - 100% utilization, 5632 MiB
GPU 3: Contrastive Learning (Priority 5) - Starting, 1316 MiB
GPU 4: Available
GPU 5: Available
```

**Total**: 4 experiments running, 2 GPUs available

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

# Check GPU usage
watch -n 1 nvidia-smi
```

### Check Individual Experiments
```bash
# Cross-Attention (Priority 2)
tail -f experiments/retrieval/tier1_cross_attention_query_document/training.log

# Hierarchical Multi-Granularity (Priority 3)
tail -f experiments/retrieval/tier1_hierarchical_multigranularity/training.log

# Iterative Refinement (Priority 4)
tail -f experiments/retrieval/tier1_iterative_refinement_improved/training.log

# Contrastive Learning (Priority 5)
tail -f experiments/retrieval/tier1_contrastive_learning/training.log
```

---

## ⏱️ Expected Timeline

- **Cross-Attention**: 4-6 days (started 08:37)
- **Hierarchical Multi-Granularity**: 3-4 days (started 08:37)
- **Iterative Refinement**: 3-4 days (started 08:37)
- **Contrastive Learning**: 5-7 days (started 08:39, includes training)

---

## 🔔 Email Notifications

**Note**: EMAIL_PASSWORD is not currently set. To enable email notifications:

```bash
export EMAIL_PASSWORD="your-gmail-app-password"
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

## ✅ Summary

**All 5 Tier 1 experiments are now running!**

- ✅ Cross-Encoder: Training completed (evaluation pending)
- 🟢 Cross-Attention: Running on GPU 0
- 🟢 Hierarchical Multi-Granularity: Running on GPU 1
- 🟢 Iterative Refinement: Running on GPU 2
- 🟢 Contrastive Learning: Running on GPU 3

**GPUs 4-5 are available** if you want to start additional experiments.

**Everything is working correctly!** 🚀
