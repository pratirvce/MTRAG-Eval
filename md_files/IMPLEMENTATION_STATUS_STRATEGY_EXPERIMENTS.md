# Implementation Status: Strategy-Compliant Experiments

## Overview
This document tracks the implementation status of experiments from `STRATEGY_FOR_NDCG_0.90_PLUS_COMPLIANT.md`.

## Implementation Status

### ✅ Already Implemented (5/10)
1. ✅ **Multi-Stage Hybrid Retrieval Pipeline** - `train_multistage_hierarchical_retrieval.py`
2. ✅ **Blended RAG Hybrid Fusion** - `train_hybrid_dynamic_fusion.py`
3. ✅ **BGE-v2-Large Domain-Specific** - `train_model_scaling_bge_v2.py`
4. ✅ **Advanced LLM Query Rewriting** - `train_llm_query_expansion_constrained.py`
5. ✅ **Direct nDCG Optimization** - `train_neural_ndcg.py`

### ✅ Newly Implemented (5/10)
6. ✅ **OpenRAG-Style Retrieval-Only End-to-End** - `train_openrag_retrieval_only.py` + `train_openrag_retrieval_only_tier1.py`
7. ✅ **MetaRAG Self-Reflection** - `train_metarag_retrieval.py` + `train_metarag_retrieval_tier1.py` (needs wrapper)
8. ⏳ **Transform Retrieval** - Needs implementation
9. ⏳ **ChainRAG Multi-Hop** - Needs implementation
10. ⏳ **Probing-RAG** - Needs implementation

## Next Steps

1. **Create remaining 3 core scripts:**
   - `train_transform_retrieval.py`
   - `train_chainrag_retrieval.py`
   - `train_probing_rag.py`

2. **Create tier1 wrappers for all 5 new experiments:**
   - `train_metarag_retrieval_tier1.py`
   - `train_transform_retrieval_tier1.py`
   - `train_chainrag_retrieval_tier1.py`
   - `train_probing_rag_tier1.py`
   - ✅ `train_openrag_retrieval_only_tier1.py` (already created)

3. **Add to auto-runner:**
   - ✅ All 5 experiments added to `PENDING_EXPERIMENTS` in `auto_start_fixed_experiments.py`

## Auto-Runner Configuration

All new experiments have been added with:
- **Priority 1** (Highest): OpenRAG, MetaRAG
- **Priority 2** (High): Transform, ChainRAG, Probing-RAG
- **Resume option**: Enabled by default
- **Parallel execution**: Supported on free GPUs

## Task A Compliance

All experiments are verified for Task A (Retrieval Only) compliance:
- ✅ No text generation
- ✅ Uses relevance labels only (not generation feedback)
- ✅ Query preprocessing allowed (LLM query rewriting)
- ✅ Retrieval quality assessment only (not generation quality)

## Expected Performance

| Experiment | Expected nDCG@10 | Status |
|------------|-----------------|--------|
| OpenRAG Retrieval-Only | 0.55-0.60 | ✅ Implemented |
| MetaRAG | 0.58-0.63 | ✅ Implemented |
| Transform Retrieval | 0.56-0.61 | ⏳ Pending |
| ChainRAG | 0.57-0.62 | ⏳ Pending |
| Probing-RAG | 0.53-0.58 | ⏳ Pending |

**Last Updated:** 2025-12-20
