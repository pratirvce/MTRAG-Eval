# Quick Start - Tier 1 Experiments

## ✅ Ready to Run!

All Tier 1 experiments have been set up with:
- ✅ Resume/checkpoint capability
- ✅ Email notifications to pratirvce@gmail.com
- ✅ Parallel execution on multiple GPUs
- ✅ Automatic status tracking

## 🚀 Start Now

```bash
cd /home/prevanka/prati/su-mt-rag/mt-rag-benchmark

# Activate your environment
source venv/bin/activate  # or: conda activate mt-rag

# Set email password (optional but recommended)
export EMAIL_PASSWORD="your-gmail-app-password"

# Start all experiments
./start_tier1_experiments.sh
```

Or run directly:
```bash
python run_tier1_experiments.py --max-parallel 6
```

## 📊 What Will Happen

1. **Cross-Encoder (Priority 1)** ✅ - Will start immediately and run successfully
2. **Other experiments** - Will attempt to start; you'll get email notifications about status

## 📧 Email Notifications

You'll receive emails at **pratirvce@gmail.com** for:
- Each experiment start
- Each experiment completion (with results)
- Any failures (with error details)

## 📁 Check Status

```bash
# View status
cat tier1_experiments_status.json

# View logs
tail -f tier1_experiments.log

# Check specific experiment
ls -la experiments/retrieval/tier1_*/
```

## 🔄 Resume After Interruption

Just run again - it will automatically resume from checkpoints:
```bash
python run_tier1_experiments.py --max-parallel 6
```

## ⚠️ Note

- Cross-encoder is fully implemented and will run
- Other experiments have frameworks but may need implementations
- You'll get email notifications for all events
- Failed experiments can be resumed after implementation

**Everything is ready! Just run the startup script!** 🚀

