# Current Experiments Status

**Last Updated**: 2025-12-16 23:20

---

## ✅ Running Experiments

### 1. **Cross-Encoder Fine-Tuning** (Priority #1) ⭐⭐⭐⭐⭐
- **GPU**: 3
- **Status**: ✅ **Running** (just started)
- **PID**: Check with `ps aux | grep train_cross_encoder`
- **Current Activity**: Loading data and training
- **Expected**: 0.49-0.52 nDCG@10 (+8-15% improvement)
- **Time**: 3-5 days
- **Log**: `experiments/retrieval/phase6_cross_encoder_finetuned_ensemble/training.log`

### 2. **Conversation-Aware Retrieval** (Phase 7) ⭐⭐⭐⭐⭐
- **GPU**: 1
- **Status**: ✅ **Running** (started 23:10)
- **Current Activity**: Encoding corpus (ClapNQ domain, ~10% complete)
- **Expected**: 0.52-0.57 nDCG@10 (+15-25% improvement)
- **Time**: 3-5 days
- **Log**: `experiments/retrieval/phase7_conversation_aware_attention/training.log`
- **Note**: Using optimized code with corpus cache

### 3. **LLM Query Expansion** (Priority #3) ⭐⭐⭐⭐
- **GPU**: 2
- **Status**: ✅ **Running** (started 22:26)
- **Current Activity**: Processing queries with GPT-4 expansion
- **Expected**: 0.48-0.51 nDCG@10 (+6-12% improvement)
- **Time**: 2-3 days
- **Log**: `experiments/retrieval/phase6_llm_query_expansion_gpt4_multi/training.log`

---

## ✅ Completed Experiments

### 1. **Multi-Stage Retrieval** (Priority #2)
- **Status**: ✅ **COMPLETE**
- **Results**: `experiments/retrieval/phase6_multistage_2stage/results.json`
- **Note**: May be re-run with fine-tuned cross-encoder for better results

---

## 📊 GPU Status

| GPU | Status | Memory Used | Utilization | Experiment |
|-----|--------|-------------|-------------|------------|
| 0 | 🟡 Busy | 17,591 MB (71.6%) | 0% | `run_ablations.py` |
| 1 | 🟢 Active | 4,529 MB (18.4%) | 100% | Conversation-Aware |
| 2 | 🟢 Active | 4,915 MB (20.0%) | 100% | LLM Expansion |
| 3 | 🟢 Active | 264 MB (1.1%) | 0% | Cross-Encoder (just started) |
| 4 | 🟢 **FREE** | 264 MB (1.1%) | 0% | **Available** |
| 5 | 🟢 **FREE** | 272 MB (1.1%) | 0% | **Available** |

---

## 🎯 Next Steps

### Available for Future Experiments:
- **GPU 4**: Free and available
- **GPU 5**: Free and available

### Potential Experiments to Start:
1. **Iterative Refinement Retrieval** (from TASK_A_RETRIEVAL_EXPERIMENTS.md)
   - Impact: +10-18% nDCG@10
   - Time: 2-3 days
   - Status: Not yet implemented

2. **Cross-Attention Query-Document Interaction**
   - Impact: +8-15% nDCG@10
   - Time: 4-6 days
   - Status: Not yet implemented

3. **Hierarchical Multi-Granularity Retrieval**
   - Impact: +8-14% nDCG@10
   - Time: 3-4 days
   - Status: Not yet implemented

---

## 📝 Monitor Commands

```bash
# Check all running experiments
ps aux | grep -E "train_cross_encoder|train_conversation_aware|train_llm_query" | grep -v grep

# Monitor specific logs
tail -f experiments/retrieval/phase6_cross_encoder_finetuned_ensemble/training.log
tail -f experiments/retrieval/phase7_conversation_aware_attention/training.log
tail -f experiments/retrieval/phase6_llm_query_expansion_gpt4_multi/training.log

# Check GPU usage
watch -n 1 nvidia-smi

# Check experiment failures
python monitor_experiments.py --once
```

---

## ✅ Summary

**Running**: 3 experiments (Cross-encoder, Conversation-aware, LLM expansion)  
**Completed**: 1 experiment (Multi-stage)  
**Free GPUs**: 2 (GPU 4, GPU 5)  
**Status**: All priority experiments running! 🚀

