# Experiment Monitoring Guide

## 🔍 Overview

The experiment monitoring system automatically detects when experiments fail and generates notifications with details about what went wrong.

---

## 🚀 Quick Start

### Start Continuous Monitoring (Background)
```bash
./start_experiment_monitor.sh
```

This will:
- Start monitoring in the background
- Check every 60 seconds
- Save status to `experiment_status.json`
- Save failures to `experiment_failures.json`
- Print notifications when failures are detected

### Run One-Time Check
```bash
python monitor_experiments.py --once
```

### Run Continuous Monitoring (Foreground)
```bash
python monitor_experiments.py --check-interval 60
```

---

## 📊 What It Monitors

### Experiment Types:
- ✅ Cross-encoder fine-tuning
- ✅ Multi-stage retrieval
- ✅ LLM query expansion
- ✅ Hybrid retrieval
- ✅ Reranking
- ✅ Ensemble
- ✅ Query expansion

### What It Detects:
1. **Process Stopped**: Experiment process no longer running
2. **No Results File**: Process stopped but no `results.json` created
3. **Errors in Logs**: Error patterns detected in log files
4. **Exit Codes**: Non-zero exit codes

### Error Patterns Detected:
- `Error:`, `Traceback`, `Exception:`
- `IndexError`, `KeyError`, `ValueError`, `TypeError`
- `AttributeError`, `RuntimeError`
- `CUDA error`, `Out of memory`, `OOM`
- `failed`, `Failed`, `FAILED`

---

## 📁 Output Files

### `experiment_status.json`
Current status of all experiments:
```json
{
  "timestamp": "2025-12-16T22:30:00",
  "running": [
    {
      "name": "phase6_cross_encoder_finetuned_ensemble",
      "pid": 12345,
      "gpu_id": 0,
      "status": "running"
    }
  ],
  "failed": [],
  "completed": [],
  "warnings": []
}
```

### `experiment_failures.json`
Detailed failure reports:
```json
{
  "timestamp": "2025-12-16T22:30:00",
  "failed_experiments": [
    {
      "name": "phase6_llm_query_expansion_gpt4_multi",
      "type": "llm_expansion",
      "pid": 12345,
      "gpu_id": 2,
      "config": "experiments/retrieval/.../config.json",
      "status": "stopped",
      "errors": [
        {
          "pattern": "IndexError",
          "line": "IndexError: index 1 is out of bounds...",
          "context": "...",
          "line_number": 273
        }
      ],
      "log_file": "experiments/retrieval/.../training.log"
    }
  ],
  "total_failed": 1
}
```

### `experiment_monitor.log`
Monitor process log (when running in background)

---

## 🔔 Notification Format

When failures are detected, you'll see:

```
================================================================================
⚠️  EXPERIMENT FAILURES DETECTED
================================================================================

🚨 1 experiment(s) failed and need attention:

1. phase6_llm_query_expansion_gpt4_multi (llm_expansion)
   PID: 12345
   GPU: 2
   Status: stopped
   Config: experiments/retrieval/phase6_llm_query_expansion_gpt4_multi/config.json
   Log: experiments/retrieval/phase6_llm_query_expansion_gpt4_multi/training.log
   Errors Found: 1
   First Error: IndexError
   Error Line: IndexError: index 1 is out of bounds for dimension 0 with size 1...

================================================================================

📝 To fix and restart:
   1. Check the log files for detailed error messages
   2. Fix the issues in the code/config
   3. Restart with: python <script> --config <config> --gpu_id <gpu> --resume

📄 Full failure report saved to: experiment_failures.json
================================================================================
```

---

## 🛠️ Usage Examples

### Check Status Once
```bash
python monitor_experiments.py --once
```

### Check Every 30 Seconds
```bash
python monitor_experiments.py --check-interval 30
```

### Run Without Notifications
```bash
python monitor_experiments.py --once --no-notify
```

### Start Background Monitoring
```bash
./start_experiment_monitor.sh
```

### Stop Background Monitoring
```bash
# Get PID
cat experiment_monitor.pid

# Kill process
kill $(cat experiment_monitor.pid)
```

### View Monitor Log
```bash
tail -f experiment_monitor.log
```

---

## 🔧 Integration with Cron

To run checks periodically (e.g., every 5 minutes):

```bash
# Edit crontab
crontab -e

# Add this line (checks every 5 minutes)
*/5 * * * * cd /home/prevanka/prati/su-mt-rag/mt-rag-benchmark && /usr/bin/python3 monitor_experiments.py --once >> experiment_monitor_cron.log 2>&1
```

---

## 📋 Manual Check Commands

### Check All Running Experiments
```bash
ps aux | grep -E "train_cross_encoder|train_multistage|train_llm_query|train_hybrid" | grep -v grep
```

### Check for Errors in Logs
```bash
# Check specific experiment
grep -i "error\|traceback\|exception" experiments/retrieval/<experiment_name>/training.log | tail -20

# Check all experiments
find experiments/retrieval -name "training.log" -exec grep -l "Error\|Traceback" {} \;
```

### Check Experiment Results
```bash
# List completed experiments
find experiments/retrieval -name "results.json" -exec dirname {} \;

# Check if specific experiment completed
ls experiments/retrieval/<experiment_name>/results.json
```

---

## 🎯 Best Practices

1. **Start Monitoring Early**: Start monitoring when you start experiments
2. **Check Regularly**: Review `experiment_failures.json` regularly
3. **Fix Promptly**: Address failures quickly to avoid wasting GPU time
4. **Use Resume**: Most experiments support `--resume` to continue from checkpoints
5. **Check Logs**: Always check the full log file for detailed error context

---

## 🔍 Troubleshooting

### Monitor Not Detecting Experiments
- Check that experiment processes are actually running
- Verify experiment names match the patterns in `monitor_experiments.py`

### False Positives
- Some warnings in logs might not be actual failures
- Check the full log context before restarting
- Verify if results.json exists (experiment might have completed)

### Monitor Process Died
- Check `experiment_monitor.log` for errors
- Restart with `./start_experiment_monitor.sh`

---

## 📝 Example Workflow

1. **Start Experiments**:
   ```bash
   python train_cross_encoder_finetuned.py --config ... --gpu_id 0 &
   ```

2. **Start Monitoring**:
   ```bash
   ./start_experiment_monitor.sh
   ```

3. **Check Status Periodically**:
   ```bash
   python monitor_experiments.py --once
   ```

4. **When Failure Detected**:
   - Check `experiment_failures.json` for details
   - Review log file mentioned in failure report
   - Fix the issue
   - Restart experiment with `--resume`

---

## ✅ Summary

The monitoring system provides:
- ✅ Automatic failure detection
- ✅ Detailed error reports
- ✅ Status tracking
- ✅ Easy restart instructions
- ✅ Background monitoring support

**Keep monitoring running to catch failures early!** 🚀
