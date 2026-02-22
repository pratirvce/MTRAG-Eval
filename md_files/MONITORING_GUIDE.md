# Automated Monitoring Guide for Fixed Experiments

## Overview

Automated monitoring has been set up for the fixed cross-attention experiments. The monitor checks experiment status every 5 minutes and provides real-time updates.

## Monitored Experiments

- `phase8_cross_attention_query_document_fixed`
- `tier1_cross_attention_rerun_fixed`

## Quick Commands

### Check Current Status (One-time)
```bash
cd /home/prevanka/prati/su-mt-rag/mt-rag-benchmark
python3 monitor_fixed_experiments.py --once
```

### View Status Report File
```bash
cat fixed_experiments_status.txt
```

### View JSON Status
```bash
cat fixed_experiments_status.json | jq
```

### View Monitoring Logs
```bash
tail -f fixed_experiments_monitor.log
```

### Start Monitoring (if not running)
```bash
./start_monitoring_fixed.sh
```

### Stop Monitoring
```bash
# Get PID
cat fixed_experiments_monitor.pid

# Kill process
kill $(cat fixed_experiments_monitor.pid)
rm fixed_experiments_monitor.pid
```

### Check if Monitoring is Running
```bash
ps aux | grep monitor_fixed_experiments | grep -v grep
```

## Status Files

- **`fixed_experiments_status.json`** - Machine-readable status (updated every check)
- **`fixed_experiments_status.txt`** - Human-readable status report (updated every check)
- **`fixed_experiments_monitor.log`** - Monitoring process logs
- **`fixed_experiments_monitor.pid`** - PID of monitoring process

## What the Monitor Tracks

1. **Process Status**: Whether experiments are running, stopped, or completed
2. **Progress**: Domain-by-domain completion status
3. **Results**: Final scores (Recall@10, nDCG@10) when available
4. **Errors**: Any errors detected in logs or checkpoints
5. **Activity**: Latest activity from log files

## Monitoring Features

- ✅ Automatic detection of experiment completion
- ✅ Progress tracking per domain
- ✅ Error detection and reporting
- ✅ Final results summary when all experiments complete
- ✅ Continuous monitoring (checks every 5 minutes)
- ✅ Status files updated automatically

## Example Status Output

```
================================================================================
Fixed Cross-Attention Experiments Monitoring
Last Updated: 2025-12-18 19:36:11
================================================================================

📊 phase8_cross_attention_query_document_fixed
--------------------------------------------------------------------------------
  Status: 🟢 RUNNING
  PID: 2999647
  CPU: 305%
  Memory: 0.1%
  Progress: 25.0%
  Domain Progress:
    clapnq: 🔄 In Progress
    fiqa: ⏳ Pending
    govt: ⏳ Pending
    cloud: ⏳ Pending
  Last Activity: Batches: 21%|██| 296/1433 [04:51<14:52, 1.27it/s]
```

## Customization

### Change Check Interval
```bash
# Run with custom interval (e.g., 10 minutes = 600 seconds)
python3 monitor_fixed_experiments.py --interval 600
```

### Run Once and Exit
```bash
python3 monitor_fixed_experiments.py --once
```

## Integration with Main Status Checker

The fixed experiments will also appear in the main status checker:
```bash
python3 check_all_runs_status.py | grep fixed
```

## Troubleshooting

### Monitor Not Running
1. Check if process exists: `ps aux | grep monitor_fixed_experiments`
2. Check log file: `tail fixed_experiments_monitor.log`
3. Restart: `./start_monitoring_fixed.sh`

### Status Not Updating
1. Check if monitor process is running
2. Check log file for errors
3. Verify experiment directories exist

### Experiments Not Detected
1. Verify experiment names match exactly
2. Check that experiment directories exist in `experiments/retrieval/`
3. Check log files for experiment activity

