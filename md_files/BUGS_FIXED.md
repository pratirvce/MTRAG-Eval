# Bugs Fixed - Experiment Scripts

## Fixed Issues

### 1. ✅ train_adversarial_curriculum.py
**Bug:** `HFValidationError` - Passing model object string instead of path
**Fix:** Save model to temporary path before loading with SentenceBERT wrapper
**Line:** 204
**Change:** Save model to `temp_model_path` before creating SentenceBERT instance

### 2. ✅ train_llm_distillation.py
**Bug:** `HFValidationError` - Same as above
**Fix:** Save model to temporary path before loading with SentenceBERT wrapper
**Line:** 205
**Change:** Save model to `temp_model_path` before creating SentenceBERT instance

### 3. ✅ train_hierarchical_routing.py
**Bug:** `IndexError: index 1 is out of bounds` - Processing queries one at a time
**Fix:** Process queries in batches of 32 instead of individually
**Line:** 169
**Change:** Batch queries before retrieval, then process each query in batch for routing/reranking

### 4. ✅ train_large_model_finetuning.py
**Bug:** `No training examples found` - Wrong corpus file path
**Fix:** Use correct corpus path from `corpora/passage_level/` instead of train directory
**Line:** 48
**Change:** Set `corpus_file` to `corpora/passage_level/{domain}.jsonl` even when using data splits

### 5. ✅ train_learned_rrf.py
**Bug:** `TypeError: Unable to extract query/object scores` - Missing queries in results
**Fix:** Ensure all queries have entries in final_results (even if empty)
**Line:** 250-256
**Change:** Initialize empty dict for all queries, then fill with sorted results

## Testing Recommendations

1. Test each script with a single domain first
2. Verify model loading works correctly
3. Check that batch processing handles edge cases
4. Ensure all queries have results (even if empty)

## Status

All 5 bugs have been fixed. Scripts should now run successfully.

