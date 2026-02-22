# Auto Experiments Setup - Complete Guide

**Status**: ✅ **ACTIVE** - Automatically starting top experiments when GPUs become available

---

## 🚀 What's Running

The `auto_start_top_experiments.py` script is now running in the background and will:

1. **Monitor GPU availability** every 5 minutes (300 seconds)
2. **Automatically start top-priority experiments** when GPUs become free
3. **Run experiments in parallel** (up to 6 simultaneously)
4. **Resume from checkpoints** if experiments are interrupted
5. **Send email notifications** on:
   - Experiment start
   - Experiment completion
   - Experiment failures/errors

---

## 📋 Top Experiments Queue (Priority Order)

1. **Ensemble Best Methods** (Priority 1)
   - Combines: Contrastive Learning + Cross-Encoder + LLM Expansion + Multi-Stage
   - Expected: 0.48-0.52 nDCG@10
   - Time: 1-2 days
   - Script: `train_ensemble_best_tier1.py`

2. **Cross-Encoder Large Model** (Priority 2)
   - Uses L-24 model for better reranking
   - Expected: 0.51-0.54 nDCG@10
   - Time: 2-4 days
   - Script: `train_cross_encoder_large_tier1.py`

3. **Domain-Specific Cross-Encoder** (Priority 3)
   - One model per domain
   - Expected: 0.50-0.53 nDCG@10
   - Time: 4-6 days
   - Script: `train_cross_encoder_domain_specific_tier1.py`

4. **3-Stage Multi-Stage** (Priority 4)
   - With fine-tuned models
   - Expected: 0.50-0.53 nDCG@10
   - Time: 3-5 days

5. **LLM Expansion Domain-Specific** (Priority 5)
   - Domain-specific prompts
   - Expected: 0.50-0.53 nDCG@10
   - Time: 2-3 days

6. **In-Batch Hard Negatives** (Priority 6)
   - Hard negative mining
   - Expected: 0.48-0.51 nDCG@10
   - Time: 4-6 days

7. **Adaptive Multi-Stage** (Priority 7)
   - Adaptive stage selection
   - Expected: 0.51-0.54 nDCG@10
   - Time: 3-4 days

8. **Pseudo-Relevance Feedback** (Priority 8)
   - With LLM expansion
   - Expected: 0.49-0.52 nDCG@10
   - Time: 2-3 days

---

## 📧 Email Notifications

**Email**: pratirvce@gmail.com

**Notifications sent for**:
- ✅ Experiment started
- ✅ Experiment completed (with results)
- ❌ Experiment failed/errored

**To enable email notifications**:
```bash
export EMAIL_PASSWORD="your-gmail-app-password"
```

**Note**: You need to use a Gmail App Password (not your regular password). See: https://support.google.com/accounts/answer/185833

---

## 🔄 Resume Capability

All experiments support resume:
- **Checkpoints saved** automatically during training
- **Resume on restart** if experiment is interrupted
- **Status tracking** in `auto_experiments_status.json`

**To manually resume**:
```bash
# Experiments will automatically resume from checkpoints
# No manual action needed - the auto-runner handles it
```

---

## 📊 Monitoring

### Check Status
```bash
# View auto-runner log
tail -f auto_experiments.log

# Check running experiments
ps aux | grep auto_start_top_experiments

# Check experiment status
cat auto_experiments_status.json
```

### Check GPU Usage
```bash
nvidia-smi
```

### Check Experiment Logs
```bash
# View logs for specific experiment
tail -f experiments/retrieval/tier1_*/training.log
```

---

## 🛠️ Manual Control

### Stop Auto-Runner
```bash
# Find PID
ps aux | grep auto_start_top_experiments

# Kill process
kill <PID>
```

### Restart Auto-Runner
```bash
cd /home/prevanka/prati/su-mt-rag/mt-rag-benchmark
source venv/bin/activate  # if using venv
nohup python auto_start_top_experiments.py --check-interval 300 --max-parallel 6 > auto_experiments.log 2>&1 &
```

### Adjust Settings
```bash
# Change check interval (default: 300 seconds = 5 minutes)
python auto_start_top_experiments.py --check-interval 600 --max-parallel 6

# Change max parallel experiments (default: 6)
python auto_start_top_experiments.py --check-interval 300 --max-parallel 4
```

---

## 📁 Files Created

- `auto_start_top_experiments.py` - Main auto-runner script
- `auto_experiments.log` - Log file for auto-runner
- `auto_experiments_status.json` - Status tracking file
- `train_ensemble_best_tier1.py` - Ensemble wrapper
- `train_cross_encoder_large_tier1.py` - Large model wrapper
- `train_cross_encoder_domain_specific_tier1.py` - Domain-specific wrapper

---

## ✅ Features

- ✅ **Automatic GPU detection** - Finds free GPUs automatically
- ✅ **Priority-based scheduling** - Starts highest priority experiments first
- ✅ **Parallel execution** - Runs multiple experiments simultaneously
- ✅ **Resume capability** - Automatically resumes from checkpoints
- ✅ **Email notifications** - Alerts on start, completion, and errors
- ✅ **Error handling** - Graceful handling of failures
- ✅ **Status tracking** - Persistent status across restarts

---

## 🎯 Expected Results

If all top 3 experiments succeed:
- **Ensemble**: 0.48-0.52 nDCG@10
- **Cross-Encoder Large**: 0.51-0.54 nDCG@10 ✅ **BEATS ELSER!**
- **Domain-Specific**: 0.50-0.53 nDCG@10

**Combined potential**: **0.52-0.55 nDCG@10** (beats Elser's 0.54!) 🎉

---

*Last Updated: 2025-12-17*
*Status: ✅ ACTIVE*

