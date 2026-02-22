# Experiments Currently Running

## ✅ Status: Experiments Started and Monitoring Active

**Last Updated**: 2025-12-13 14:27:57

### Currently Running (5 experiments)
1. 🟢 `phase5_ensemble_domain_specific` - GPU 1, PID: 2437866
2. 🟢 `phase5_ensemble_weighted` - GPU 1, PID: 2437964
3. 🟢 `phase5_domain_specific_clapnq_hard_negatives` - GPU 1, PID: 2438163
4. 🟢 `phase5_domain_specific_govt_hard_negatives` - GPU 1, PID: 2438366
5. 🟢 `phase5_reranking_clapnq` - GPU 1, PID: 2438526

### Pending (15 experiments)
- 3 more reranking experiments
- 3 query expansion experiments
- 9 hybrid retrieval experiments

### Monitor Status
- **Monitor PID**: 2439566
- **Check Interval**: 300 seconds (5 minutes)
- **Monitor Log**: `monitor.log`
- **Notifications**: `experiment_notifications.log`

## 📋 Quick Commands

### Check Status
```bash
# Quick status
python manage_experiments.py status

# Detailed report
python monitor_experiments.py --report

# Check results
python check_results.py
```

### View Logs
```bash
# Monitor log
tail -f monitor.log

# Notifications
tail -f experiment_notifications.log

# Specific experiment log
tail -f experiments/retrieval/<name>/training.log
```

### Monitor Control
```bash
# Check if monitor is running
ps aux | grep monitor_experiments

# Stop monitor
kill $(cat monitor.pid)

# Restart monitor
nohup python -u monitor_experiments.py --interval 300 >> monitor.log 2>&1 &
echo $! > monitor.pid
```

## 🔔 Notifications

The monitor will automatically notify you when:
- ✅ An experiment completes (with results)
- ❌ An experiment fails (with error details)

Notifications are logged to `experiment_notifications.log`

## 📊 Results

Results are automatically saved to:
```
experiments/retrieval/<experiment_name>/results.json
```

Check results with:
```bash
python check_results.py
```

## 🎯 Next Steps

1. **Monitor Progress**: Check status regularly
2. **Watch for Notifications**: Monitor `experiment_notifications.log`
3. **Check Results**: Use `python check_results.py` when experiments complete
4. **Review Logs**: Check training logs if experiments fail

## 📝 Notes

- Experiments run in parallel across available GPUs
- Monitor checks status every 5 minutes
- Failed experiments can be restarted manually
- All experiments support checkpoint/resume

