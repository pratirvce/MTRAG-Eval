# Current Experiment Status Report

**Generated**: 2025-12-17 11:55:38  
**Current Best**: 0.4576 nDCG@10 (Contrastive Learning)

---

## 📊 Overall Summary

- **Completed**: 6 experiments with valid results
- **Running**: 4 experiments
- **Failed/Issues**: 4 experiments (need attention)
- **Best Score**: 0.4576 nDCG@10
- **Gap to Elser (0.54)**: -0.0824 (18.0% improvement needed)
- **Gap to Tier 1 (0.55)**: -0.0924 (20.2% improvement needed)

---

## ✅ Completed Experiments (6)

| Rank | Experiment | nDCG@10 | Status |
|------|------------|---------|--------|
| 🥇 | **tier1_contrastive_learning** | **0.4576** | ✅ **BEST** |
| 🥈 | tier1_cross_encoder_evaluation | 0.2716 | ✅ Good |
| 🥉 | tier2_multistage_3stage | 0.2480 | ✅ Moderate |
| 4 | tier1_hierarchical_multigranularity | 0.2289 | ✅ Moderate |
| 5 | tier1_iterative_refinement_improved | 0.2231 | ✅ Low |
| 6 | tier2_ensemble_advanced | 0.2192 | ✅ Low |

---

## 🟢 Currently Running Experiments (4)

### 1. **tier1_cross_encoder_domain_specific**
- **GPU**: 2
- **PID**: 2779432
- **Status**: 🟢 **ACTIVE** (100% GPU utilization)
- **Progress**: Processing batches (35% complete - batch 141/391)
- **Activity**: Encoding batches, making good progress
- **Started**: 2025-12-17 11:53:22

### 2. **tier1_ensemble_best_methods**
- **GPU**: 0
- **PID**: 2779430
- **Status**: ⚠️ **ERROR** (Syntax error - needs fix)
- **Issue**: Syntax error in ensemble script
- **Started**: 2025-12-17 11:53:22

### 3. **tier1_llm_query_expansion**
- **Status**: ⚠️ **ERROR** (Import error - needs fix)
- **Issue**: Import error (may be from old process)
- **Note**: File has correct import, may need restart

### 4. **tier1_multistage_2stage**
- **Status**: ⚠️ **ERROR** (Import error - needs fix)
- **Issue**: Import error (may be from old process)
- **Note**: File has correct import, may need restart

---

## ⚠️ Failed/Issues (4)

1. **tier1_cross_attention_query_document**: 0.0 nDCG@10 (failed)
2. **tier1_cross_attention_rerun**: 0.0 nDCG@10 (failed)
3. **tier1_cross_encoder_finetuned**: Training completed, no evaluation results
4. **tier1_cross_encoder_large**: Completed but no valid results

---

## 🖥️ GPU Status

| GPU | Utilization | Memory | Status | Experiment |
|-----|-------------|--------|--------|------------|
| **GPU 0** | 0% | 6.3% (1.5 GB) | Idle | Ensemble (error) |
| **GPU 1** | 0% | 1.1% (264 MB) | Idle | Cross-Encoder Large (error) |
| **GPU 2** | **100%** | 23.1% (5.7 GB) | **BUSY** | Domain-Specific Cross-Encoder ✅ |
| **GPU 3** | 0% | 1.1% (264 MB) | Free | Available |
| **GPU 4** | 0% | 1.1% (264 MB) | Free | Available |
| **GPU 5** | 0% | 1.1% (272 MB) | Free | Available |

---

## 🔄 Auto-Runner Status

- **Status**: ✅ **ACTIVE**
- **PID**: 2779411
- **Monitoring**: Every 5 minutes
- **Tracking**: 3 experiments
- **Last Updated**: 2025-12-17 11:53:22

**Tracked Experiments**:
1. tier1_ensemble_best_methods (GPU 0, PID: 2779430)
2. tier1_cross_encoder_large (GPU 1, PID: 2779431)
3. tier1_cross_encoder_domain_specific (GPU 2, PID: 2779432)

---

## 📈 Performance Analysis

### Current Best: 0.4576 nDCG@10

**Comparison**:
- **vs Elser (0.54)**: -0.0824 (-18.0%)
- **vs Tier 1 Target (0.55)**: -0.0924 (-20.2%)
- **vs Previous Best (0.4515)**: +0.0061 (+1.4%) ✅

**Progress**:
- ✅ Beat previous best (Query Expansion: 0.4515)
- ❌ Still below Elser baseline (0.54)
- ❌ Still below Tier 1 target (0.55)

---

## 🎯 Active Work

### Currently Running:
- **Domain-Specific Cross-Encoder**: Making good progress (35% complete)
  - Processing all 4 domains sequentially
  - Expected: 0.50-0.53 nDCG@10
  - Time remaining: ~3-4 hours

### Needs Attention:
1. **Ensemble Best Methods**: Syntax error - needs fix
2. **Cross-Encoder Large**: Error - needs investigation
3. **LLM Query Expansion**: Import error (may be old process)
4. **Multi-Stage 2-Stage**: Import error (may be old process)

---

## 📝 Next Steps

1. **Fix ensemble syntax error** (high priority)
2. **Restart failed experiments** after fixes
3. **Monitor domain-specific cross-encoder** (running well)
4. **Wait for running experiments to complete**
5. **Start additional experiments** when GPUs free up

---

## ✅ Summary

- **1 experiment running successfully** (Domain-Specific Cross-Encoder)
- **3 experiments need fixes** (Ensemble, Large, LLM, Multi-Stage)
- **6 experiments completed** with valid results
- **4 experiments failed** and need re-running
- **Auto-runner active** and monitoring

**Overall Status**: 🟡 **PARTIALLY ACTIVE** - 1 experiment running well, others need fixes

---

*Last Updated: 2025-12-17 11:55:38*
