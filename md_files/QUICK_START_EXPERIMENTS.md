# Quick Start: Running Retrieval Improvement Experiments

## ✅ Implementation Complete

All improvements have been implemented and are ready to run:

### Created Files:
1. ✅ `train_improved_bge.py` - Enhanced training script
2. ✅ `evaluate_improved_bge.py` - Enhanced evaluation script  
3. ✅ `run_experiments.py` - Main experiment runner
4. ✅ `summarize_results.py` - Results summarizer
5. ✅ `start_experiments.sh` - Background starter script
6. ✅ `check_experiment_status.sh` - Status checker

## 🚀 Start Experiments

### Option 1: Background (Recommended)
```bash
cd /home/prevanka/prati/su-mt-rag/mt-rag-benchmark
source venv/bin/activate
./start_experiments.sh
```

### Option 2: Manual Background
```bash
cd /home/prevanka/prati/su-mt-rag/mt-rag-benchmark
source venv/bin/activate
nohup python run_experiments.py > experiments/logs/experiments_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

## 📊 Monitor Progress

### Check Status
```bash
./check_experiment_status.sh
```

### View Live Logs
```bash
tail -f experiments/logs/experiments_*.log
```

### View Results (After Completion)
```bash
python summarize_results.py
```

## 📋 Experiments Being Run

### Phase 1: Quick Wins (3 experiments)
1. Baseline (original config)
2. 3 epochs + validation + proper splits
3. 5 epochs + validation + proper splits

### Phase 2: Training Improvements (4 experiments)
4. Learning rate 1e-5
5. Learning rate 5e-5
6. CosineSimilarityLoss
7. Data augmentation

### Phase 3: Retrieval Strategy (3 experiments)
8. Hybrid retrieval (dense + BM25)
9. Reranking with cross-encoder
10. Hybrid + reranking

**Total: 10 experiments**

## ⏱️ Expected Timeline

- **Phase 1**: ~2-4 hours
- **Phase 2**: ~4-6 hours
- **Phase 3**: ~6-8 hours
- **Total**: ~12-18 hours

## 📁 Results Location

- **Results**: `experiments/retrieval/{experiment_name}/results.json`
- **Summary**: `experiments/retrieval/experiment_summary.json`
- **Models**: `./models/{experiment_name}/`
- **Logs**: `experiments/logs/experiments_*.log`

## 🔍 Troubleshooting

### Check if running:
```bash
ps aux | grep run_experiments
```

### View latest log:
```bash
ls -lt experiments/logs/ | head -3
tail -50 experiments/logs/experiments_*.log
```

### Stop experiments:
```bash
pkill -f run_experiments.py
```

### Restart:
```bash
./start_experiments.sh
```

## 📈 Expected Results

After all experiments complete, you should see:
- Best Recall@10: **0.45-0.55** (vs baseline 0.38)
- Best nDCG@10: **0.35-0.45** (vs baseline 0.30)

All results will be automatically recorded and compared with baseline!

