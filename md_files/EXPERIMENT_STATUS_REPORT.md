# Experiment Status Report

**Generated:** 2025-12-19 18:01:31

## Executive Summary

- **Running:** 0 experiments (checking...)
- **Completed:** 53 experiments
- **Failed:** 4 experiments (need fixes)
- **Pending:** 48 experiments (not started yet)
- **Auto-Runner:** ✅ Running (PID: 3048317)

---

## 1. Currently Running Experiments

*Check with: `ps aux | grep train_ | grep -v grep`*

---

## 2. Auto-Runner Status

### Status
- ✅ **Running** (PID: 3048317)
- **Check Interval:** 300 seconds (5 minutes)
- **Log File:** `logs/auto_fixed_experiments.log`

### Issues Identified

1. **Most pending experiments are already completed**
   - 5/5 pending experiments monitored by auto-runner are completed
   - Auto-runner should focus on failed experiments instead

2. **tier1_enhanced_contrastive_hardnegatives_fixed keeps failing**
   - Error: Tensor size mismatch (different sizes each run)
   - Auto-runner keeps retrying but it fails repeatedly
   - Needs actual fix before re-running

3. **Some failed experiments not in auto-runner's list**
   - `tier1_cross_attention_query_document` (CUDA device error)
   - `phase2_cosine_loss` (Training configuration error)
   - `tier1_enhanced_contrastive` (Tensor size mismatch)

### Auto-Runner Experiments Status

#### Fixed Experiments (3 total)
| Experiment | Status | Notes |
|------------|--------|-------|
| `tier1_enhanced_contrastive_hardnegatives_fixed` | ❌ FAILED/NEEDS RE-RUN | Tensor size mismatch - needs fix |
| `tier1_learning_to_rank_listwise_fixed` | ✅ COMPLETED | Fix worked |
| `best_paper_hierarchical_routing_fixed` | ✅ COMPLETED | Fix worked |

#### Pending Experiments (5 total)
| Experiment | Status | Notes |
|------------|--------|-------|
| `tier1_cross_attention_query_document_fixed` | ✅ COMPLETED | Already done |
| `tier1_adversarial_curriculum` | ✅ COMPLETED | Already done |
| `tier1_graph_aware_retrieval` | ✅ COMPLETED | Already done |
| `tier1_llm_distillation` | ✅ COMPLETED | Already done |
| `tier1_multitask_retrieval` | ✅ COMPLETED | Already done |

---

## 3. Failed Experiments That Need Fixes & Re-Run

### High Priority (Critical)

#### 1. tier1_enhanced_contrastive_hardnegatives_fixed
- **Error:** Tensor size mismatch - `The size of tensor a (X) must match the size of tensor b (Y) at non-singleton dimension 1`
- **Root Cause:** Query and document embeddings have different sequence lengths
- **Fix Needed:** 
  - Add padding/truncation for variable-length sequences
  - Check embedding extraction logic
  - Ensure consistent tensor dimensions before operations
- **Status:** ❌ Auto-runner keeps retrying but fails repeatedly
- **Script:** `train_enhanced_contrastive_tier1.py`

#### 2. tier1_learning_to_rank_listwise_fixed
- **Error:** Evaluator retriever not set
- **Root Cause:** Evaluator initialization issue
- **Fix Applied:** Removed line that re-initialized evaluator with None
- **Status:** ⚠️ May need additional verification
- **Script:** `train_learning_to_rank_tier1.py`

### Medium Priority

#### 3. tier1_cross_attention_query_document
- **Error:** CUDA device error - `invalid device ordinal`
- **Root Cause:** Device handling issue during model loading
- **Fix Needed:**
  - Verify CUDA availability before model loading
  - Check device assignment logic
  - Ensure proper model initialization sequence
- **Status:** ❌ Not in auto-runner's list
- **Script:** `train_cross_attention_tier1.py`

#### 4. tier1_enhanced_contrastive
- **Error:** Tensor size mismatch - `The size of tensor a (125) must match the size of tensor b (424) at non-singleton dimension 1`
- **Root Cause:** Same as tier1_enhanced_contrastive_hardnegatives_fixed
- **Fix Needed:** Same padding/alignment fix
- **Status:** ❌ Not in auto-runner's list
- **Script:** `train_enhanced_contrastive_tier1.py`

#### 5. tier1_enhanced_contrastive_hardnegatives
- **Error:** `'last_hidden_state'` KeyError
- **Root Cause:** Model output structure doesn't match expected format
- **Fix Needed:** Check model output structure and adjust extraction logic
- **Status:** ❌ Not in auto-runner's list
- **Script:** `train_enhanced_contrastive_tier1.py`

### Low Priority

#### 6. phase2_cosine_loss
- **Error:** Training configuration error in `fit()` method
- **Root Cause:** Issue with training objectives or checkpoint path configuration
- **Fix Needed:** Review training configuration parameters
- **Status:** ❌ Not in auto-runner's list
- **Script:** `train_improved_bge.py`

---

## 4. Completed Experiments (53 total)

✅ **best_paper_adversarial_curriculum**
✅ **best_paper_graph_aware_retrieval**
✅ **best_paper_hierarchical_routing**
✅ **best_paper_hierarchical_routing_fixed**
✅ **best_paper_large_model_finetuning**
✅ **best_paper_learned_rrf**
✅ **best_paper_llm_distillation**
✅ **best_paper_meta_learning**
✅ **best_paper_meta_learning_fixed**
✅ **best_paper_multitask_retrieval**
✅ **best_paper_rl_adaptive_retrieval**
✅ **best_paper_rl_adaptive_retrieval_fixed**
✅ **best_paper_temporal_memory**
✅ **multistage_retrieval**
✅ **phase3_hybrid_reranking**
✅ **phase3_reranking**
✅ **phase4_hard_negatives_5neg**
✅ **phase5_ensemble_domain_specific**
✅ **phase5_ensemble_weighted**
✅ **phase5_query_expansion_clapnq**
✅ **phase5_query_expansion_govt**
✅ **phase5_query_expansion_multi**
✅ **phase5_reranking_clapnq**
✅ **phase5_reranking_cloud**
✅ **phase5_reranking_govt**
✅ **phase5_reranking_multi_domain**
✅ **phase6_cross_encoder_evaluation**
✅ **phase6_llm_query_expansion_gpt4_multi**
✅ **phase6_multistage_2stage**
✅ **phase6_multistage_2stage_finetuned**
✅ **phase7_conversation_aware_attention**
✅ **phase7_iterative_refinement**
✅ **tier1_adversarial_curriculum**
✅ **tier1_contrastive_learning**
✅ **tier1_cross_attention**
✅ **tier1_cross_attention_fixed_v3**
✅ **tier1_cross_attention_query_document_fixed**
✅ **tier1_cross_attention_rerun_fixed_v2**
✅ **tier1_cross_encoder_domain_specific**
✅ **tier1_cross_encoder_evaluation**
✅ **tier1_cross_encoder_large**
✅ **tier1_ensemble_best_methods**
✅ **tier1_graph_aware_retrieval**
✅ **tier1_hierarchical_multigranularity**
✅ **tier1_iterative_refinement_improved**
✅ **tier1_learning_to_rank_listwise**
✅ **tier1_learning_to_rank_listwise_fixed**
✅ **tier1_llm_distillation**
✅ **tier1_multitask_retrieval**
✅ **tier1_pseudo_relevance_feedback**
✅ **tier1_qdit_transformer**
✅ **tier2_ensemble_advanced**
✅ **tier2_multistage_3stage**

---

## 5. Recommendations

### Immediate Actions

1. **Fix tensor size mismatch in tier1_enhanced_contrastive_hardnegatives_fixed**
   - Add padding/truncation logic
   - Ensure consistent embedding dimensions
   - Test with different sequence lengths

2. **Update auto-runner configuration**
   - Remove completed experiments from pending list
   - Add failed experiments that need fixes:
     - `tier1_cross_attention_query_document`
     - `phase2_cosine_loss`
     - `tier1_enhanced_contrastive`
     - `tier1_enhanced_contrastive_hardnegatives`

3. **Stop auto-runner from retrying tier1_enhanced_contrastive_hardnegatives_fixed**
   - It will keep failing until the actual fix is applied
   - Remove from auto-runner list until fix is ready

### Long-term Actions

1. **Improve error detection in auto-runner**
   - Detect repeated failures
   - Stop retrying after N consecutive failures
   - Alert when experiments need manual fixes

2. **Expand auto-runner to monitor all failed experiments**
   - Not just fixed experiments
   - Include experiments with known fixable issues

3. **Create fix tracking system**
   - Track which experiments have fixes applied
   - Track which experiments need fixes
   - Prioritize fixes based on impact

---

## 6. How to Check Status

### Check Running Experiments
```bash
ps aux | grep train_ | grep -v grep
```

### Check GPU Utilization
```bash
nvidia-smi
```

### Check Auto-Runner Logs
```bash
tail -f logs/auto_fixed_experiments.log
```

### Check Experiment Results
```bash
ls -la experiments/retrieval/*/results.json
```

### Check Failed Experiments
```bash
grep -r "Failed:" experiments/retrieval/*/training.log
```

---

## 7. Auto-Runner Configuration

**File:** `auto_start_fixed_experiments.py`

**Current Configuration:**
- Fixed Experiments: 3
- Pending Experiments: 5 (all completed)
- Check Interval: 300 seconds
- Max Parallel: 6 GPUs

**Recommendation:** Update to include more failed experiments and remove completed ones.
