# Phase 7: Conversation-Aware Retrieval - Implementation Guide

## ✅ Implementation Complete

### Features Implemented:

1. ✅ **Conversation-Aware Contextual Retrieval** with attention mechanism
2. ✅ **Parallel execution** support (via experiment manager)
3. ✅ **Pause/Resume** capability (checkpointing after each domain)
4. ✅ **Error monitoring** (integrated with monitor_experiments.py)
5. ✅ **Graceful shutdown** (SIGTERM/SIGINT handling)

---

## 🚀 Quick Start

### 1. Start the Experiment

```bash
# Start with experiment manager (recommended)
python manage_phase7_experiments.py --config phase7_experiments.json

# Or start directly
python train_conversation_aware_retrieval.py \
    --config experiments/retrieval/phase7_conversation_aware_attention/config.json \
    --gpu_id 0 \
    --resume
```

### 2. Monitor Progress

```bash
# Monitor logs
tail -f experiments/retrieval/phase7_conversation_aware_attention/training.log

# Check status
python manage_phase7_experiments.py --status

# Monitor for errors
python monitor_experiments.py --once
```

### 3. Pause/Resume

```bash
# Pause specific experiment
python manage_phase7_experiments.py --pause phase7_conversation_aware_attention

# Resume specific experiment
python manage_phase7_experiments.py --resume phase7_conversation_aware_attention

# Stop all
python manage_phase7_experiments.py --stop-all
```

---

## 📋 Experiment Details

### Experiment Name
`phase7_conversation_aware_attention`

### What It Does
- Uses **full conversation history** (not just last turn)
- Applies **attention mechanism** to focus on relevant turns
- Encodes conversation context for better retrieval
- Falls back to weighted average if attention unavailable

### Expected Results
- **nDCG@10**: 0.52-0.57 (+15-25% improvement)
- **Recall@10**: Improved coverage

### Time Estimate
- **3-5 days** (depending on corpus size)

---

## 🔧 Configuration

### Config File: `experiments/retrieval/phase7_conversation_aware_attention/config.json`

```json
{
  "experiment_name": "phase7_conversation_aware_attention",
  "model_path": "BAAI/bge-base-en-v1.5",
  "domains": ["clapnq", "fiqa", "govt", "cloud"],
  "use_data_splits": true,
  "use_attention": true,
  "resume": true
}
```

### Parameters:
- `model_path`: Base model for encoding
- `domains`: List of domains to evaluate
- `use_data_splits`: Use train/test splits if available
- `use_attention`: Enable attention mechanism (falls back to weighted average if fails)
- `resume`: Resume from checkpoint

---

## 📊 Checkpointing

### How It Works:
- **Checkpoint saved after each domain** completes
- **Resume automatically** skips completed domains
- **Checkpoint location**: `experiments/retrieval/phase7_conversation_aware_attention/checkpoints/checkpoint.json`

### Checkpoint Format:
```json
{
  "clapnq": {
    "domain": "clapnq",
    "results": {
      "Recall@10": 0.5432,
      "nDCG@10": 0.5210
    },
    "timestamp": "2025-12-16T22:00:00"
  }
}
```

---

## 🔍 Error Monitoring

### Automatic Detection:
- **Process monitoring**: Checks if process is still running
- **Log scanning**: Scans last 50 lines for error patterns
- **Exit code checking**: Detects non-zero exit codes

### Error Patterns Detected:
- `Error:`, `Traceback`, `Exception:`
- `IndexError`, `KeyError`, `ValueError`, `TypeError`
- `AttributeError`, `RuntimeError`
- `CUDA error`, `Out of memory`, `OOM`
- `failed`, `Failed`, `FAILED`

### Notifications:
- Errors logged to `experiment_failures.json`
- Status saved to `experiment_status.json`
- Console output shows detected errors

---

## 🎯 Parallel Execution

### How to Run in Parallel:

1. **Multiple experiments** (if you add more):
   ```bash
   python manage_phase7_experiments.py --max-parallel 3
   ```

2. **Multiple domains** (currently sequential, but can be parallelized):
   - Modify script to process domains in parallel
   - Use multiprocessing or threading

3. **Multiple GPUs**:
   - Manager automatically assigns free GPUs
   - Each experiment runs on separate GPU

---

## 📝 Logs and Output

### Log File:
`experiments/retrieval/phase7_conversation_aware_attention/training.log`

### Results File:
`experiments/retrieval/phase7_conversation_aware_attention/results.json`

### Results Format:
```json
{
  "clapnq": {
    "Recall@1": 0.1234,
    "Recall@3": 0.2345,
    "Recall@5": 0.3456,
    "Recall@10": 0.4567,
    "nDCG@1": 0.2345,
    "nDCG@3": 0.3456,
    "nDCG@5": 0.4567,
    "nDCG@10": 0.5678
  },
  "average": {
    "Recall@10": 0.4500,
    "nDCG@10": 0.5500
  }
}
```

---

## 🛠️ Troubleshooting

### Issue: Attention encoder fails to initialize
**Solution**: Script automatically falls back to weighted average. Check GPU memory.

### Issue: Out of memory
**Solution**: Reduce batch size in script (currently 32) or use smaller model.

### Issue: Process hangs
**Solution**: Use `--pause` to gracefully stop, then check logs.

### Issue: Checkpoint not found
**Solution**: Normal for first run. Checkpoint created after first domain completes.

---

## ✅ Next Steps

1. **Start the experiment**:
   ```bash
   python manage_phase7_experiments.py
   ```

2. **Monitor progress**:
   ```bash
   tail -f experiments/retrieval/phase7_conversation_aware_attention/training.log
   ```

3. **Check for errors**:
   ```bash
   python monitor_experiments.py --once
   ```

4. **After completion**: Review results and compare to baseline

---

## 🎉 Expected Outcome

**Target**: Beat Elser's 0.54 nDCG@10  
**Expected**: 0.52-0.57 nDCG@10  
**Improvement**: +15-25% over current best (0.4539)

**This is the highest priority novel experiment for Task A!** 🚀

