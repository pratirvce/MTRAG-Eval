# Tier 1 Experiments Setup and Execution Guide

## Overview

This guide explains how to set up and run all 5 Tier 1 experiments with resume capability and email notifications.

## Prerequisites

1. **Email Setup (for notifications)**:
   ```bash
   # For Gmail, create an App Password:
   # 1. Go to Google Account settings
   # 2. Security → 2-Step Verification → App passwords
   # 3. Generate an app password for "Mail"
   # 4. Set it as environment variable:
   
   export EMAIL_PASSWORD="your-app-password-here"
   ```

2. **GPU Availability**:
   - Ensure you have GPUs available (the script will auto-detect)
   - Currently configured for up to 6 GPUs in parallel

## Experiments Included

1. **Fine-Tuned Cross-Encoder Reranking** (Priority 1) ✅ Ready
   - Script: `train_cross_encoder_finetuned_tier1.py`
   - Expected: 0.49-0.52 nDCG@10
   - Time: 3-5 days

2. **Cross-Attention Query-Document** (Priority 2) ⚠️ Needs Implementation
   - Script: `train_cross_attention_tier1.py`
   - Expected: 0.49-0.52 nDCG@10
   - Time: 4-6 days

3. **Hierarchical Multi-Granularity** (Priority 3) ⚠️ Needs Implementation
   - Script: `train_hierarchical_multigranularity_tier1.py`
   - Expected: 0.49-0.52 nDCG@10
   - Time: 3-4 days

4. **Improved Iterative Refinement** (Priority 4) ⚠️ Needs Improvement
   - Script: `train_iterative_refinement_improved_tier1.py`
   - Expected: 0.50-0.54 nDCG@10
   - Time: 3-4 days

5. **Contrastive Learning** (Priority 5) ⚠️ Needs Implementation
   - Script: `train_contrastive_learning_tier1.py`
   - Expected: 0.49-0.52 nDCG@10
   - Time: 5-7 days

## Quick Start

### Option 1: Use the startup script (Recommended)

```bash
cd /home/prevanka/prati/su-mt-rag/mt-rag-benchmark

# Set email password (optional but recommended)
export EMAIL_PASSWORD="your-app-password"

# Start all experiments
./start_tier1_experiments.sh
```

### Option 2: Run manually

```bash
cd /home/prevanka/prati/su-mt-rag/mt-rag-benchmark

# Set email password
export EMAIL_PASSWORD="your-app-password"

# Run master script
python run_tier1_experiments.py --max-parallel 6
```

## Features

### ✅ Resume Capability
- All experiments support checkpoint/resume
- If interrupted, experiments can be resumed from last checkpoint
- Checkpoints saved automatically

### ✅ Email Notifications
- Sends email to `pratirvce@gmail.com` on:
  - Experiment start
  - Experiment completion (with results)
  - Experiment failure (with error details)

### ✅ Parallel Execution
- Automatically assigns experiments to available GPUs
- Runs up to 6 experiments in parallel (configurable)
- Prioritizes experiments by priority order

### ✅ Status Monitoring
- Status saved to `tier1_experiments_status.json`
- Can check status at any time
- Logs saved to experiment directories

## Monitoring

### Check Status
```bash
# View status file
cat tier1_experiments_status.json

# Check running processes
ps aux | grep tier1

# View logs
tail -f tier1_experiments.log
```

### Check Individual Experiment
```bash
# View experiment log
tail -f experiments/retrieval/tier1_cross_encoder_finetuned/training.log

# Check results
cat experiments/retrieval/tier1_cross_encoder_finetuned/results.json
```

## Stopping Experiments

### Graceful Shutdown
```bash
# Send SIGTERM to master script (saves checkpoints)
pkill -TERM -f run_tier1_experiments.py
```

### Resume After Stop
```bash
# Just run again - it will automatically resume from checkpoints
python run_tier1_experiments.py --max-parallel 6
```

## Troubleshooting

### Email Notifications Not Working
1. Check if `EMAIL_PASSWORD` is set: `echo $EMAIL_PASSWORD`
2. For Gmail, ensure you're using an App Password, not regular password
3. Check logs for email errors: `grep -i email tier1_experiments.log`

### Experiments Not Starting
1. Check GPU availability: `nvidia-smi`
2. Check if scripts are executable: `ls -la train_*_tier1.py`
3. Check logs: `tail -f tier1_experiments.log`

### Resume Not Working
1. Check if checkpoint exists: `ls experiments/retrieval/tier1_*/checkpoint.json`
2. Ensure `--resume` flag is used (default)
3. Check checkpoint file format

## Implementation Status

### ✅ Fully Implemented
- Fine-Tuned Cross-Encoder (uses existing `train_cross_encoder_finetuned.py`)

### ⚠️ Needs Implementation
- Cross-Attention Query-Document (framework ready, needs implementation)
- Hierarchical Multi-Granularity (framework ready, needs implementation)
- Improved Iterative Refinement (needs debugging from previous failure)
- Contrastive Learning (framework ready, needs implementation)

## Next Steps

1. **Start Ready Experiments**: Cross-encoder will start immediately
2. **Implement Missing Experiments**: Complete implementations for priorities 2-5
3. **Monitor Progress**: Check logs and email notifications
4. **Resume if Needed**: Experiments will auto-resume from checkpoints

## Expected Timeline

- **Week 1**: Cross-encoder completes (3-5 days)
- **Week 2**: Other experiments start as implementations are completed
- **Week 3**: All experiments complete, results available

## Contact

For issues or questions, check:
- Log files in `experiments/retrieval/tier1_*/`
- Status file: `tier1_experiments_status.json`
- Master log: `tier1_experiments.log`

