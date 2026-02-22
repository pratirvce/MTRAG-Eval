# Quick Start - Phase 5 Experiments

## 🚀 Get Started in 3 Steps

### Step 1: Setup Experiments
```bash
cd /home/prevanka/prati/su-mt-rag/mt-rag-benchmark
source venv/bin/activate
python setup_phase5_experiments.py
```

This registers 20 experiments:
- 2 Ensemble methods
- 2 Combined techniques  
- 4 Reranking experiments
- 3 Query expansion
- 9 Hybrid retrieval

### Step 2: Start All Experiments
```bash
python manage_experiments.py start
```

This will:
- ✅ Automatically detect available GPUs
- ✅ Distribute experiments across GPUs
- ✅ Start experiments in parallel
- ✅ Enable checkpoint/resume automatically

### Step 3: Monitor Progress
```bash
# Check status
python manage_experiments.py status

# Watch logs
tail -f experiments/retrieval/<experiment_name>/training.log
```

## 🎮 Common Commands

```bash
# Status
python manage_experiments.py status

# Pause an experiment
python manage_experiments.py pause <experiment_name>

# Resume an experiment
python manage_experiments.py resume <experiment_name>

# Stop an experiment
python manage_experiments.py stop <experiment_name>

# Pause all
python manage_experiments.py pause_all

# Resume all
python manage_experiments.py resume_all

# Update status
python manage_experiments.py update
```

## 📊 Check Results

Results are saved to:
```
experiments/retrieval/<experiment_name>/results.json
```

## 🔧 Features

✅ **Auto-Resume**: If killed, experiments resume from checkpoint  
✅ **Parallel**: Runs on all available GPUs simultaneously  
✅ **Pause/Resume**: Pause anytime, resume later  
✅ **Status Tracking**: Always know what's running  

## 📝 Notes

- Experiments automatically save checkpoints every 1000 steps
- Status is saved to `experiment_status.json`
- Logs are in `experiments/retrieval/<name>/training.log`
- All experiments support graceful shutdown (Ctrl+C)

## 🆘 Troubleshooting

**Experiment not starting?**
```bash
# Check GPU availability
nvidia-smi

# Check status
python manage_experiments.py status
```

**Experiment stuck?**
```bash
# Pause and resume
python manage_experiments.py pause <name>
python manage_experiments.py resume <name>
```

**Need to restart?**
```bash
# Stop and start
python manage_experiments.py stop <name>
python manage_experiments.py start
```

## 📚 Full Documentation

See `PHASE5_EXPERIMENTS_README.md` for complete documentation.

