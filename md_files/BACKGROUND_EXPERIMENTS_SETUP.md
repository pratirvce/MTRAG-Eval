# Background Experiments Setup - Complete ✅

## 🎉 What's Been Set Up

### 1. Experiments Started ✅
- **20 experiments** registered and ready
- **5 experiments** currently running
- **15 experiments** pending (will start as GPUs become available)

### 2. Monitoring System ✅
- **Monitor running** in background (PID: 2439566)
- Checks status every **5 minutes**
- Automatically detects completions and failures
- Logs notifications to `experiment_notifications.log`

### 3. Result Checking ✅
- Quick command to check all results: `python check_results.py`
- Detailed status: `python manage_experiments.py status`
- Automatic result saving when experiments complete

## 🚀 Quick Commands

### Check What's Running
```bash
# Status overview
python manage_experiments.py status

# Detailed report
python monitor_experiments.py --report

# Check results
python check_results.py
```

### View Notifications
```bash
# See completion/failure notifications
tail -f experiment_notifications.log

# See monitor activity
tail -f monitor.log
```

### Check Specific Experiment
```bash
# Check results for specific experiment
python check_results.py phase5_ensemble_domain_specific

# View training log
tail -f experiments/retrieval/phase5_ensemble_domain_specific/training.log
```

## 📊 Current Status

**Running**: 5 experiments
- `phase5_ensemble_domain_specific`
- `phase5_ensemble_weighted`
- `phase5_domain_specific_clapnq_hard_negatives`
- `phase5_domain_specific_govt_hard_negatives`
- `phase5_reranking_clapnq`

**Pending**: 15 experiments (will start automatically as GPUs free up)

**Monitor**: Active (PID: 2439566)

## 🔔 Notifications

You'll be notified automatically when:

### ✅ Experiment Completes
```
✅ EXPERIMENT COMPLETED: phase5_ensemble_domain_specific
   Recall@10: 0.4873
   nDCG@10: 0.4227
   Results saved to: experiments/retrieval/phase5_ensemble_domain_specific/results.json
```

### ❌ Experiment Fails
```
❌ EXPERIMENT FAILED: phase5_domain_specific_clapnq_hard_negatives
   Error: CUDA out of memory
   Check logs: experiments/retrieval/phase5_domain_specific_clapnq_hard_negatives/
```

## 📁 Important Files

| File | Purpose |
|------|---------|
| `experiment_status.json` | Current status of all experiments |
| `experiment_notifications.log` | Completion/failure notifications |
| `monitor.log` | Monitor activity log |
| `monitor.pid` | Monitor process ID |
| `experiments/retrieval/<name>/results.json` | Experiment results |
| `experiments/retrieval/<name>/training.log` | Training log |
| `experiments/retrieval/<name>/error.log` | Error log (if failed) |

## 🎯 What Happens Next

1. **Experiments Run**: They'll run in parallel across available GPUs
2. **Monitor Checks**: Every 5 minutes, monitor checks status
3. **Notifications**: You'll see notifications when experiments complete or fail
4. **Results Saved**: Results automatically saved to `results.json`
5. **Auto-Resume**: If killed, training experiments resume from checkpoint

## 🔧 Troubleshooting

### Check if Monitor is Running
```bash
ps aux | grep monitor_experiments
cat monitor.pid
```

### Restart Monitor
```bash
# Stop current monitor
kill $(cat monitor.pid)

# Start new monitor
nohup python -u monitor_experiments.py --interval 300 >> monitor.log 2>&1 &
echo $! > monitor.pid
```

### Check Experiment Status
```bash
# Update status
python manage_experiments.py update

# Check status
python manage_experiments.py status
```

### View Recent Notifications
```bash
# Last 20 lines
tail -20 experiment_notifications.log

# Follow in real-time
tail -f experiment_notifications.log
```

## 📚 Documentation

- **Quick Start**: `QUICK_START_PHASE5.md`
- **Full Guide**: `PHASE5_EXPERIMENTS_README.md`
- **Monitoring Guide**: `EXPERIMENT_MONITORING_GUIDE.md`
- **Current Status**: `EXPERIMENTS_RUNNING.md`

## ✨ Features

✅ **Automatic Monitoring**: Checks every 5 minutes  
✅ **Failure Detection**: Notifies immediately on failure  
✅ **Completion Notifications**: Shows results when complete  
✅ **Result Checking**: Easy command to check all results  
✅ **Background Execution**: All runs in background  
✅ **Auto-Resume**: Training experiments resume if killed  
✅ **Parallel Execution**: Uses all available GPUs  

## 🎊 You're All Set!

Experiments are running in the background with automatic monitoring. Just check `experiment_notifications.log` periodically to see when experiments complete or fail!

```bash
# Quick check
tail -20 experiment_notifications.log

# Or check results
python check_results.py
```

