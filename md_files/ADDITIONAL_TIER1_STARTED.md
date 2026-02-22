# Additional Tier 1 Experiments Started

**Date**: 2025-12-17  
**Status**: ✅ Started on Free GPUs

## 🚀 Newly Started Experiments

### 1. **Cross-Encoder Evaluation** (GPU 0)
- **Experiment**: `tier1_cross_encoder_evaluation`
- **Status**: 🟢 RUNNING
- **Description**: Evaluate the fine-tuned cross-encoder model for reranking
- **Expected**: 0.49-0.52 nDCG@10
- **Priority**: ⭐⭐⭐⭐⭐ (Highest)
- **PID**: 2765415
- **Log**: `experiments/retrieval/tier1_cross_encoder_evaluation/training.log`

### 2. **Cross-Attention Rerun** (GPU 1)
- **Experiment**: `tier1_cross_attention_rerun`
- **Status**: 🟢 RUNNING
- **Description**: Re-run cross-attention query-document interaction (previous run failed with 0.0)
- **Expected**: 0.49-0.52 nDCG@10
- **Priority**: ⭐⭐⭐⭐⭐ (Highest)
- **PID**: 2765416
- **Log**: `experiments/retrieval/tier1_cross_attention_rerun/training.log`

### 3. **LLM Query Expansion** (GPU 2)
- **Experiment**: `tier1_llm_query_expansion`
- **Status**: 🟢 RUNNING
- **Description**: LLM-based multi-query expansion for better retrieval
- **Expected**: 0.48-0.51 nDCG@10
- **Priority**: ⭐⭐⭐⭐ (High)
- **PID**: 2765417
- **Log**: `experiments/retrieval/tier1_llm_query_expansion/training.log`

### 4. **Multi-Stage 2-Stage** (GPU 3)
- **Experiment**: `tier1_multistage_2stage`
- **Status**: 🟢 RUNNING
- **Description**: 2-stage multi-stage retrieval (Dense + Cross-encoder reranking)
- **Expected**: 0.48-0.51 nDCG@10
- **Priority**: ⭐⭐⭐⭐ (High)
- **PID**: 2765418
- **Log**: `experiments/retrieval/tier1_multistage_2stage/training.log`

## 📊 GPU Utilization

- **GPU 0**: Cross-Encoder Evaluation
- **GPU 1**: Cross-Attention Rerun
- **GPU 2**: LLM Query Expansion
- **GPU 3**: Multi-Stage 2-Stage
- **GPU 4**: Available (hierarchical experiment may still be running)
- **GPU 5**: Available

## 🎯 Expected Impact

If all experiments succeed:
- **Cross-Encoder**: +10-17% improvement (0.49-0.52 nDCG@10)
- **Cross-Attention**: +10-17% improvement (0.49-0.52 nDCG@10)
- **LLM Expansion**: +6-12% improvement (0.48-0.51 nDCG@10)
- **Multi-Stage**: +6-12% improvement (0.48-0.51 nDCG@10)

**Combined Potential**: If we ensemble the best results, could achieve **0.52-0.56 nDCG@10** (beating Elser's 0.54!)

## 📝 Monitoring

Monitor progress:
```bash
# Check logs
tail -f experiments/retrieval/tier1_*/training.log

# Check GPU usage
nvidia-smi

# Check process status
ps aux | grep tier1_
```

## ✅ Summary

- **4 new experiments started** on free GPUs
- **All high-priority Tier 1 experiments** now running
- **Expected completion**: 3-7 days depending on experiment
- **Resume capability**: All experiments support resume if interrupted

---

*Started by: start_additional_tier1_experiments.py*

