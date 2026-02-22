# Failed Experiments - Fix Summary

I've analyzed the 22 failed experiments. Here's what can be fixed:

## Fixable Issues (8 experiments)

### 1. Configuration Error (1 experiment)
- **tier1_learning_to_rank_listwise_fixed**: Code already has evaluator recreation logic, but error still occurs. The evaluator might be getting reset. Fix: Ensure evaluator is created fresh before each evaluation call.

### 2. Gradient Errors (4 experiments) - FIXABLE
- tier1_multi_granularity
- tier1_semantic_drift  
- tier1_counterfactual_augmentation
- tier1_query_decomposition

**Fix:** Ensure model.train(), check requires_grad, don't detach before backward()

### 3. Embedding Mismatch (1 experiment) - FIXABLE
- tier1_enhanced_contrastive: Fix negative sampling logic for batch_size=1

### 4. Boolean Tensor (1 experiment) - FIXABLE  
- tier1_graph_enhanced_reranking: Replace tensor boolean usage with .item() or .any()

### 5. CUDA OOM (1 experiment) - FIXABLE
- Reduce batch size, enable fp16, add gradient checkpointing

## Needs Investigation (13 experiments)

"Process died" errors need log file inspection to determine root cause. Many may be memory-related crashes.

## Recommendation

Given the complexity, I recommend:
1. Fix the 7 code-level bugs first (gradient, embedding, boolean tensor errors)
2. Investigate "process died" errors by checking logs
3. Address OOM issues by reducing batch sizes

Would you like me to:
A) Fix the code-level bugs now (gradient, embedding mismatch, boolean tensor)?
B) Create a script to analyze logs for "process died" errors?
C) Create fixes for batch size/memory issues?
