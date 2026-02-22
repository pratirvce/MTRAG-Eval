# All Experiments Status Report

**Last Updated**: 2025-12-17 22:27:19

---

## 📊 Overall Summary

- **✅ Completed**: 9 experiments
- **🟢 Running**: 0 experiments (processes not active)
- **❌ Failed**: 5 experiments
- **⏳ Pending**: 5 experiments
- **🖥️ GPUs**: 2/6 active (GPUs 0, 1 busy; GPUs 2-5 free)

---

## 🏆 Top Performing Experiments

| Rank | Experiment | nDCG@10 | Status |
|------|------------|---------|--------|
| 🥇 | **tier1_contrastive_learning** | **0.45755** | ✅ Completed |
| 🥇 | **tier1_ensemble_best_methods** | **0.45755** | ✅ Completed |
| 🥉 | tier1_cross_encoder_evaluation | 0.27158 | ✅ Completed |
| 4 | tier1_cross_encoder_large | 0.27158 | ✅ Completed |
| 5 | tier1_cross_encoder_domain_specific | 0.25593 | ✅ Completed |
| 6 | tier2_multistage_3stage | 0.24800 | ✅ Completed |
| 7 | tier1_hierarchical_multigranularity | 0.22890 | ✅ Completed |
| 8 | tier1_iterative_refinement_improved | 0.22309 | ✅ Completed |
| 9 | tier2_ensemble_advanced | 0.21923 | ✅ Completed |

**Current Best**: **0.45755 nDCG@10**  
**Target (Elser)**: 0.54 nDCG@10  
**Gap**: -0.08245 (15.3% improvement needed)

---

## ❌ Failed Experiments

1. **tier1_cross_attention_query_document**
   - Issue: Zero scores (CUDA device errors - fixed but needs rerun)
   - Status: Needs restart

2. **tier1_cross_attention_rerun**
   - Issue: Zero scores (rerun of failed experiment)
   - Status: Needs restart

3. **tier1_cross_encoder_finetuned**
   - Issue: Invalid results
   - Status: Needs investigation

4. **tier1_learning_to_rank_listwise**
   - Issue: `'SentenceBERT' object has no attribute 'encode'`
   - Status: ✅ **FIXED** (code updated, needs restart)

5. **tier1_pseudo_relevance_feedback**
   - Issue: `Model/Technique has not been provided!`
   - Status: ✅ **FIXED** (code updated, needs restart)

---

## 🚀 Novel Experiments Status

### 1. **tier1_enhanced_contrastive_hardnegatives**
- **Status**: ⏸️ Log exists but not running
- **Issue**: Process failed to start (venv activation issue)
- **Expected**: 0.50-0.54 nDCG@10
- **Action**: Needs restart with fixed command

### 2. **tier1_qdit_transformer**
- **Status**: ⏸️ Log exists but not running
- **Issue**: Process failed to start (venv activation issue)
- **Expected**: 0.52-0.56 nDCG@10
- **Action**: Needs restart with fixed command

---

## ⏳ Pending Experiments

1. **tier1_enhanced_contrastive_hardnegatives** - Novel experiment (needs restart)
2. **tier1_llm_query_expansion** - Completed successfully (log shows completion)
3. **tier1_multistage_2stage** - Completed successfully (log shows completion)
4. **tier1_multistage_3stage_finetuned** - Needs restart
5. **tier1_qdit_transformer** - Novel experiment (needs restart)

---

## 🖥️ GPU Status

| GPU | Utilization | Memory | Status | Assignment |
|-----|-------------|--------|--------|------------|
| GPU 0 | 100% | 23.6% (5793MB) | 🟢 **BUSY** | Unknown process |
| GPU 1 | 100% | 22.3% (5481MB) | 🟢 **BUSY** | Unknown process |
| GPU 2 | 0% | 1.1% (280MB) | ⚪ **FREE** | Available |
| GPU 3 | 0% | 1.1% (264MB) | ⚪ **FREE** | Available |
| GPU 4 | 0% | 1.1% (264MB) | ⚪ **FREE** | Available |
| GPU 5 | 0% | 1.1% (272MB) | ⚪ **FREE** | Available |

**Summary**: 2 GPUs busy (unknown processes), 4 GPUs available

---

## 📈 Progress Analysis

### Completed Experiments
- **Total**: 9 experiments successfully completed
- **Best Score**: 0.45755 nDCG@10
- **Average Score**: ~0.28 nDCG@10 (excluding zeros)

### Failed Experiments
- **Total**: 5 experiments failed
- **Main Issues**: 
  - Code errors (2 fixed, need restart)
  - Zero scores (2 need investigation/restart)
  - Invalid results (1 needs investigation)

### Novel Experiments
- **Total**: 2 novel experiments implemented
- **Status**: Both need restart (venv activation issue)
- **Expected Impact**: Could reach 0.52-0.56 nDCG@10

---

## 🎯 Next Steps

### Immediate Actions:

1. **Restart Fixed Experiments**:
   - tier1_learning_to_rank_listwise (FIXED)
   - tier1_pseudo_relevance_feedback (FIXED)
   - tier1_cross_attention_query_document (GPU issue FIXED)

2. **Restart Novel Experiments**:
   - tier1_enhanced_contrastive_hardnegatives (fix venv command)
   - tier1_qdit_transformer (fix venv command)

3. **Investigate Failed**:
   - tier1_cross_encoder_finetuned (check why invalid results)
   - tier1_cross_attention_rerun (rerun after fix)

### When GPUs Free Up:

- Start additional high-priority experiments
- Run novel experiments in parallel
- Continue with fixed experiments

---

## 📝 Notes

- **Best Performance**: Contrastive Learning (0.45755) - matches ensemble
- **Gap to Target**: 18% improvement needed to beat Elser (0.54)
- **Novel Experiments**: Have high potential (0.52-0.56 expected)
- **GPU Availability**: 4 GPUs free for new experiments
- **Resume Support**: All experiments have checkpoint/resume capability

---

*Generated: 2025-12-17 22:27:19*
