# Failed Experiments Fixes Applied

This document tracks fixes applied to failed experiments.

## Summary of Failed Experiments (22 total)

### Error Categories:
1. **Configuration Error (1)**: Missing evaluator retriever setup
2. **Gradient/Backward Error (4)**: Tensor gradient issues
3. **Embedding Mismatch (1)**: Negative embedding count mismatch  
4. **Boolean Tensor Error (1)**: Ambiguous boolean tensor conversion
5. **Process Died (13)**: Crashes/killed processes
6. **CUDA OOM (2)**: Out of memory errors

---

## Fixes Applied

### 1. ✅ tier1_learning_to_rank_listwise_fixed - Evaluator Retriever Error

**Error:** "Evaluator retriever is not set!"

**Fix Applied:**
- The code already has checks for evaluator.retriever being None
- The issue occurs when evaluator.evaluate() is called without a retriever
- **Solution:** Always recreate evaluator right before evaluation to ensure retriever is set

**Status:** Fixed in train_learning_to_rank_listwise.py (lines 352-354 already have fix, but error still occurs - need to check if it's actually being called properly)

**Note:** The existing code at line 354 already recreates the evaluator. The error might be happening elsewhere. Need to verify the exact call site.

---

### 2. ⚠️ Gradient Errors (4 experiments)

**Experiments:**
- tier1_multi_granularity
- tier1_semantic_drift  
- tier1_counterfactual_augmentation
- tier1_query_decomposition

**Error:** "element 0 of tensors does not require grad and does not have a grad_fn"

**Root Cause:** 
- Model might not be in training mode
- Embeddings might be detached from computation graph
- Using `.item()` or converting to numpy before backward pass

**Fix Needed:**
1. Ensure `model.train()` is called
2. Ensure all parameters have `requires_grad=True`
3. Don't detach tensors before backward pass
4. Use proper forward pass through model modules

**Status:** Needs code fixes in each training script

---

### 3. ⚠️ tier1_enhanced_contrastive - Negative Embedding Mismatch

**Error:** "Processed 2 negative embeddings but expected 1"

**Root Cause:**
- Batch size = 1, but code processes 2 negatives per query
- Negative sampling logic doesn't handle batch_size=1 correctly

**Fix Needed:**
- Check batch size before processing negatives
- Ensure negative count matches query count
- Handle edge case when batch_size=1

**Status:** Needs fix in train_enhanced_contrastive_hardnegatives.py

---

### 4. ⚠️ tier1_graph_enhanced_reranking - Boolean Tensor Error

**Error:** "Boolean value of Tensor with more than one value is ambiguous"

**Root Cause:**
- Using tensor in boolean context (if statement)
- Need to use `.item()` or `.any()` / `.all()`

**Fix Needed:**
- Find where tensor is used in boolean context
- Replace with proper tensor comparison

**Status:** Needs code inspection and fix

---

### 5. ⚠️ Process Died Errors (13 experiments)

**Experiments:**
- tier1_differentiable_retrieval_fixed
- tier1_learned_indices_fixed
- tier1_multi_turn_state_tracking_fixed
- tier1_curriculum_contrastive_fixed
- tier1_mixture_experts_fixed
- tier1_uncertainty_aware_fixed
- tier1_causal_inference_fixed
- task_a_* experiments (6 experiments)

**Error:** "Process died without completing"

**Possible Causes:**
1. Memory issues (OOM)
2. Uncaught exceptions
3. System kills (OOM killer)
4. Training crashes
5. Timeout/killed by scheduler

**Fix Needed:**
1. Check training logs for actual error
2. Add better error handling
3. Reduce batch size if memory issue
4. Add checkpoints to resume from failures

**Status:** Need to check individual logs to determine root cause

---

### 6. ⚠️ CUDA Out of Memory (2 experiments)

**Experiments:**
- tier1_meta_learning
- tier1_multi_turn_state_tracking

**Error:** CUDA OOM when trying to allocate memory

**Fix Needed:**
1. Reduce batch size
2. Enable gradient checkpointing
3. Use mixed precision training (fp16)
4. Clear cache more frequently
5. Use smaller model if possible

**Status:** Need to adjust memory settings in training configs

---

## Next Steps

1. **Immediate fixes (code changes needed):**
   - Fix gradient errors in 4 experiments
   - Fix negative embedding mismatch
   - Fix boolean tensor error

2. **Investigation needed:**
   - Check logs for "process died" experiments
   - Review memory usage for OOM experiments

3. **Configuration fixes:**
   - Adjust batch sizes
   - Enable gradient checkpointing
   - Add better error handling

---

## Recommendations

For experiments that failed with "process died", it's recommended to:
1. Check training logs in `experiments/retrieval/{experiment_name}/training.log`
2. Look for actual error messages before the process died
3. Reduce batch size and retry
4. Add more frequent checkpointing

For gradient errors:
1. Ensure model is in training mode
2. Check that all tensors used in loss calculation have gradients
3. Don't detach tensors prematurely
4. Use proper forward pass through model
