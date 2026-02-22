# Experiment Results Summary - Current Status

**Last Updated**: 2025-12-17  
**Baseline Reference**: Paper Baseline (Recall@10: 0.3800, nDCG@10: 0.3000)  
**SOTA Baseline**: Elser with Query Rewrite (nDCG@10: 0.5400)

---

## 🏆 **Best Overall Performance**

### **Top Performer: Phase 5 Query Expansion (Govt)**
- **Recall@10**: **0.5317** (+39.9% vs paper baseline)
- **nDCG@10**: **0.4515** (+50.5% vs paper baseline)
- **Method**: Query expansion on government domain
- **Status**: ✅ Completed

### **Second Best: Phase 5 Ensemble Domain-Specific**
- **Recall@10**: **0.5356** (+40.9% vs paper baseline)
- **nDCG@10**: **0.4434** (+47.8% vs paper baseline)
- **Method**: Ensemble of domain-specific models using Reciprocal Rank Fusion (RRF)
- **Status**: ✅ Completed

**Domain Breakdown:**
- ClapNQ: R@10=0.6173, nDCG@10=0.5266
- FiQA: R@10=0.5149, nDCG@10=0.4137
- Govt: R@10=0.5152, nDCG@10=0.4257
- Cloud: R@10=0.4948, nDCG@10=0.4077

**Comparison to SOTA:**
- Current Best nDCG@10: **0.4515** (Query Expansion Govt)
- SOTA (Elser): **0.5400**
- **Gap**: **-0.0885** (needs +8.85% improvement to match SOTA)
- **Gap to Tier 1 Minimum (0.55)**: **-0.0985** (needs +9.85% improvement)

---

## 📊 **Top 10 Completed Experiments (Ranked by nDCG@10)**

| Rank | Experiment | Recall@10 | nDCG@10 | Improvement vs Baseline | Phase | Status |
|------|------------|-----------|---------|------------------------|-------|--------|
| 🥇 | **phase5_query_expansion_govt** | **0.5317** | **0.4515** | **+39.9% / +50.5%** | Phase 5 | ✅ Completed |
| 🥈 | phase5_ensemble_domain_specific | 0.5356 | 0.4434 | +40.9% / +47.8% | Phase 5 | ✅ Completed |
| 🥉 | phase5_ensemble_weighted | 0.5284 | 0.4370 | +39.1% / +45.7% | Phase 5 | ✅ Completed |
| 4 | phase5_query_expansion_multi | 0.5099 | 0.4098 | +34.2% / +36.6% | Phase 5 | ✅ Completed |
| 5 | phase5_query_expansion_clapnq | 0.4999 | 0.4186 | +31.5% / +39.5% | Phase 5 | ✅ Completed |
| 6 | phase7_conversation_aware_attention | 0.4643 | 0.3657 | +22.2% / +21.9% | Phase 7 | ✅ Completed |
| 7 | phase5_reranking_clapnq | 0.4319 | 0.3250 | +13.6% / +8.3% | Phase 5 | ✅ Completed |
| 8 | phase6_cross_encoder_evaluation | 0.3639 | 0.2978 | -4.2% / -0.7% | Phase 6 | ✅ Completed |
| 9 | phase3_hybrid_reranking | 0.3388 | 0.2709 | -10.9% / -9.7% | Phase 3 | ✅ Completed |
| 10 | phase7_iterative_refinement | 0.2867 | 0.2231 | -24.5% / -25.6% | Phase 7 | ✅ Completed |

---

## 📈 **Performance by Phase**

### **Phase 1: Foundation Experiments**
| Experiment | Recall@10 | nDCG@10 | Status |
|------------|-----------|---------|--------|
| phase1_baseline | 0.3076 | 0.2303 | ✅ Completed |
| phase1_epochs3 | ~0.3388 | ~0.2709 | ✅ Completed |
| phase1_epochs5 | ~0.4693 | ~0.3671 | ✅ Completed |

**Best Phase 1**: Extended training (5 epochs) achieved +23.5% improvement

### **Phase 2: Hyperparameter & Data Augmentation**
| Experiment | Recall@10 | nDCG@10 | Status |
|------------|-----------|---------|--------|
| phase2_augmentation | 0.5099 | 0.4099 | ✅ Completed |
| phase2_lr1e5 | ~0.3309 | ~0.2605 | ✅ Completed |
| phase2_lr5e5 | ~0.4208 | ~0.3331 | ✅ Completed |

**Best Phase 2**: Data augmentation achieved +34.2% improvement

### **Phase 3: Hybrid Retrieval & Reranking**
| Experiment | Recall@10 | nDCG@10 | Status |
|------------|-----------|---------|--------|
| phase3_hybrid | ~0.3388 | ~0.2709 | ✅ Completed |
| phase3_reranking | 0.3388 | 0.2709 | ✅ Completed |
| phase3_hybrid_reranking | 0.3388 | 0.2709 | ✅ Completed |

**Phase 3 Assessment**: Underperformed compared to dense retrieval (-10.9%)

### **Phase 4: Domain-Specific Models**
| Experiment | Domain | Recall@10 | nDCG@10 | Status |
|------------|--------|-----------|---------|--------|
| phase4_domain_specific_clapnq | ClapNQ | **0.6016** | **0.4981** | ✅ Completed |
| phase4_domain_specific_govt | Govt | 0.5511 | 0.4628 | ✅ Completed |
| phase4_domain_specific_cloud | Cloud | 0.5293 | 0.4104 | ✅ Completed |
| phase4_domain_specific_fiqa | FiQA | 0.5119 | 0.4026 | ✅ Completed |
| **Average** | - | **0.5485** | **0.4435** | ✅ Completed |

**Best Phase 4**: Domain-specific models achieved +44.3% average improvement

### **Phase 5: Ensemble & Advanced Techniques**
| Experiment | Recall@10 | nDCG@10 | Status |
|------------|-----------|---------|--------|
| **phase5_ensemble_domain_specific** 🥇 | **0.5356** | **0.4434** | ✅ Completed |
| phase5_ensemble_weighted | 0.5284 | 0.4370 | ✅ Completed |
| phase5_query_expansion_govt | 0.5317 | 0.4515 | ✅ Completed |
| phase5_query_expansion_multi | 0.5099 | 0.4098 | ✅ Completed |
| phase5_query_expansion_clapnq | 0.4999 | 0.4186 | ✅ Completed |
| phase5_reranking_clapnq | 0.4319 | 0.3250 | ✅ Completed |

**Best Phase 5**: Ensemble domain-specific achieved best overall performance

### **Phase 6: Cross-Encoder & Multi-Stage**
| Experiment | Recall@10 | nDCG@10 | Status |
|------------|-----------|---------|--------|
| phase6_cross_encoder_evaluation | 0.3639 | 0.2978 | ✅ Completed |
| phase6_multistage_2stage | - | - | ✅ Completed |

**Phase 6 Assessment**: Cross-encoder needs fine-tuning (currently underperforming)

### **Phase 7: Advanced Retrieval Methods**
| Experiment | Recall@10 | nDCG@10 | Status |
|------------|-----------|---------|--------|
| phase7_conversation_aware_attention | 0.4643 | 0.3657 | ✅ Completed |
| phase7_iterative_refinement | 0.2867 | 0.2231 | ✅ Completed (Failed) |

**Phase 7 Assessment**: Conversation-aware shows promise, iterative refinement needs rework

### **Phase 8: Cross-Attention**
| Experiment | Recall@10 | nDCG@10 | Status |
|------------|-----------|---------|--------|
| phase8_cross_attention_query_document | 0.0 | 0.0 | ❌ Failed (needs re-implementation) |

---

## 🎯 **Key Findings**

### **What Worked Best:**
1. ✅ **Domain-Specific Fine-Tuning** - Best single technique (+44.3% improvement)
2. ✅ **Ensemble Methods** - Best overall result (0.4434 nDCG@10)
3. ✅ **Query Expansion** - Strong improvement (+36.6% to +50.5%)
4. ✅ **Data Augmentation** - Good improvement (+34.2%)
5. ✅ **Extended Training** - Moderate improvement (+23.5%)

### **What Underperformed:**
1. ❌ **Hard Negative Mining** - Significant degradation (-51.5%)
2. ❌ **Iterative Refinement** - Failed implementation (-25.6%)
3. ❌ **Cross-Attention (Phase 8)** - Failed (0.0 results)
4. ❌ **Hybrid Retrieval (Phase 3)** - Underperformed (-9.7%)
5. ❌ **Reranking (some domains)** - Mixed results

---

## 🚀 **Currently Running Experiments**

### **Tier 1 Experiments (High Priority):**
1. **tier1_cross_encoder_finetuned** - Training completed, evaluation pending
2. **tier1_cross_attention_query_document** - Running on GPU 0
3. **tier1_hierarchical_multigranularity** - Running on GPU 1
4. **tier1_iterative_refinement_improved** - Running on GPU 2
5. **tier1_contrastive_learning** - Running on GPU 3

### **Tier 2 Experiments (Additional):**
6. **tier2_multistage_3stage** - Running on GPU 4
7. **tier2_ensemble_advanced** - Running on GPU 5

**Expected Results:**
- Cross-Encoder: 0.49-0.52 nDCG@10
- Cross-Attention: 0.49-0.52 nDCG@10
- Hierarchical: 0.49-0.52 nDCG@10
- Iterative Refinement: 0.50-0.54 nDCG@10 (if fixed)
- Contrastive Learning: 0.49-0.52 nDCG@10

---

## 📊 **Performance Progression**

| Phase | Best nDCG@10 | Improvement vs Baseline | Key Technique |
|-------|--------------|------------------------|---------------|
| Phase 1 | 0.3671 | +22.4% | Extended Training |
| Phase 2 | 0.4099 | +36.6% | Data Augmentation |
| Phase 3 | 0.2709 | -9.7% | Hybrid/Reranking (failed) |
| Phase 4 | 0.4435 | +47.8% | Domain-Specific Models |
| Phase 5 | 0.4515 | +50.5% | Query Expansion (Govt) |
| Phase 6 | 0.2978 | -0.7% | Cross-Encoder (needs fine-tuning) |
| Phase 7 | 0.3657 | +21.9% | Conversation-Aware |
| **Current Best** | **0.4515** | **+50.5%** | **Query Expansion (Govt)** |

---

## 🎯 **Gap Analysis**

### **Current Position:**
- **Your Best**: 0.4515 nDCG@10 (Query Expansion Govt)
- **SOTA (Elser)**: 0.5400 nDCG@10
- **Gap**: -0.0885 (16.4% below SOTA)

### **Tier 1 Conference Targets:**
- **Minimum for Acceptance**: 0.55-0.57 nDCG@10
- **Strong Acceptance**: 0.57-0.60 nDCG@10
- **Top Leaderboard**: 0.60-0.65 nDCG@10

### **Required Improvements:**
- To match SOTA: **+8.85%** improvement needed
- For Tier 1 minimum: **+9.85%** improvement needed
- For strong acceptance: **+11.85%** improvement needed

---

## 💡 **Next Steps**

The currently running Tier 1 experiments are expected to close the gap:
- If all Tier 1 experiments succeed: **0.55-0.59 nDCG@10** (beats SOTA!)
- If top 2 experiments succeed: **0.50-0.53 nDCG@10** (competitive)
- If best single experiment succeeds: **0.49-0.52 nDCG@10** (strong improvement)

**All experiments are running and will complete over the next few days!**

---

*Last Updated: 2025-12-17*

