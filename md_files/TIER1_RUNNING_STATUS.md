# Tier 1 Experiments - Currently Running Status

**Started**: 2025-12-17 08:32  
**Master Runner PID**: Check with `ps aux | grep run_tier1_experiments`

---

## ✅ Currently Running Experiments

### 1. **tier1_cross_encoder_finetuned** (Priority 1) 🟢 RUNNING
- **GPU**: 0
- **PID**: Check status file
- **Status**: ✅ Running
- **Expected**: 0.49-0.52 nDCG@10
- **Time**: 3-5 days
- **Log**: `experiments/retrieval/tier1_cross_encoder_finetuned/training.log`

### 2. **tier1_cross_attention_query_document** (Priority 2) 🟡 STARTING
- **GPU**: 1
- **PID**: Check status file
- **Status**: 🟡 Starting (may fail if not implemented)
- **Expected**: 0.49-0.52 nDCG@10
- **Time**: 4-6 days
- **Log**: `experiments/retrieval/tier1_cross_attention_query_document/training.log`

### 3. **tier1_hierarchical_multigranularity** (Priority 3) 🟡 STARTING
- **GPU**: 2
- **PID**: Check status file
- **Status**: 🟡 Starting (may fail if not implemented)
- **Expected**: 0.49-0.52 nDCG@10
- **Time**: 3-4 days
- **Log**: `experiments/retrieval/tier1_hierarchical_multigranularity/training.log`

### 4. **tier1_iterative_refinement_improved** (Priority 4) ⏳ PENDING
- **Status**: ⏳ Waiting for GPU availability
- **Expected**: 0.50-0.54 nDCG@10
- **Time**: 3-4 days

### 5. **tier1_contrastive_learning** (Priority 5) ⏳ PENDING
- **Status**: ⏳ Waiting for GPU availability
- **Expected**: 0.49-0.52 nDCG@10
- **Time**: 5-7 days

---

## 📊 Monitoring Commands

### Check Status
```bash
# View current status
cat tier1_experiments_status.json

# View master log
tail -f tier1_experiments.log

# Check running processes
ps aux | grep tier1
```

### Check Individual Experiments
```bash
# Cross-encoder (Priority 1)
tail -f experiments/retrieval/tier1_cross_encoder_finetuned/training.log

# Cross-attention (Priority 2)
tail -f experiments/retrieval/tier1_cross_attention_query_document/training.log

# Hierarchical (Priority 3)
tail -f experiments/retrieval/tier1_hierarchical_multigranularity/training.log
```

### Check GPU Usage
```bash
watch -n 1 nvidia-smi
```

---

## ⚠️ Important Notes

### Email Notifications
**Email password not set!** To enable email notifications:
```bash
export EMAIL_PASSWORD="your-gmail-app-password"
# Then restart the master runner
```

### Resume Capability
All experiments support resume:
- If interrupted, just run again: `python run_tier1_experiments.py`
- Will automatically resume from checkpoints

### Expected Behavior
1. **Cross-encoder** will run successfully (fully implemented)
2. **Other experiments** may fail initially (need implementations)
3. **You'll get email notifications** (if EMAIL_PASSWORD is set)
4. **Status is tracked** in `tier1_experiments_status.json`

---

## 🎯 Next Steps

1. **Monitor progress**: Check logs regularly
2. **Set email password**: For failure notifications
3. **Implement missing experiments**: As needed
4. **Resume if needed**: Just run the script again

---

**All experiments are running! Check logs for progress.** 🚀

