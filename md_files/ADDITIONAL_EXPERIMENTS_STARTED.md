# Additional Experiments Started on Free GPUs

**Date**: 2025-12-17  
**Status**: ✅ **STARTED**

---

## 🚀 Newly Started Experiments

### 1. **tier1_llm_query_expansion**
- **GPU**: 0
- **Status**: 🟢 **STARTED**
- **Description**: LLM-based multi-query expansion
- **Expected**: 0.48-0.51 nDCG@10
- **Time**: 2-3 days
- **Script**: `train_llm_query_expansion_tier1.py`

### 2. **tier1_multistage_2stage**
- **GPU**: 3
- **Status**: 🟢 **STARTED**
- **Description**: 2-stage multi-stage retrieval (Dense + Cross-encoder)
- **Expected**: 0.48-0.51 nDCG@10
- **Time**: 3-5 days
- **Script**: `train_multistage_tier1.py`

---

## 📊 Current GPU Utilization

| GPU | Status | Utilization | Experiment |
|-----|--------|-------------|------------|
| GPU 0 | **BUSY** | Starting | LLM Query Expansion |
| GPU 1 | **BUSY** | 100% | Cross-Encoder Large |
| GPU 2 | **BUSY** | 100% | Domain-Specific Cross-Encoder |
| GPU 3 | **BUSY** | Starting | Multi-Stage 2-Stage |
| GPU 4 | FREE | 0% | Available |
| GPU 5 | FREE | 0% | Available |

---

## ✅ All Running Experiments

1. **tier1_cross_encoder_domain_specific** (GPU 2)
   - Status: Running (~35% complete)

2. **tier1_cross_encoder_large** (GPU 1)
   - Status: Running (processing batches)

3. **tier1_llm_query_expansion** (GPU 0)
   - Status: Just started

4. **tier1_multistage_2stage** (GPU 3)
   - Status: Just started

---

## 📈 Expected Results

If all experiments succeed:
- **Domain-Specific Cross-Encoder**: 0.50-0.53 nDCG@10
- **Cross-Encoder Large**: 0.51-0.54 nDCG@10 ✅ **Could beat Elser!**
- **LLM Query Expansion**: 0.48-0.51 nDCG@10
- **Multi-Stage 2-Stage**: 0.48-0.51 nDCG@10

**Combined Potential**: If we ensemble the best results, could achieve **0.52-0.56 nDCG@10** (beats Elser's 0.54!) 🎉

---

## 🔄 Auto-Runner Status

- **Status**: ✅ **ACTIVE**
- **Monitoring**: Every 5 minutes
- **Will Start**: Additional experiments when GPUs 4-5 become available

---

*All experiments started successfully!*

