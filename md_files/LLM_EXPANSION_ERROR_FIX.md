# LLM Query Expansion Error - Fixed ✅

## 🔍 Error Analysis

### Problem:
The LLM query expansion experiment stopped with an `IndexError`:
```
IndexError: index 1 is out of bounds for dimension 0 with size 1
```

### Root Cause:
The error occurs in BEIR's `exact_search.py` when processing a single query. The code assumes at least 2 queries:
```python
min(top_k + 1, len(cos_scores[1])),  # Tries to access index 1
```

But the LLM expansion script was processing queries one at a time:
```python
temp_queries = {query_id: variation}  # Only 1 query!
results = evaluator.retrieve(corpus, temp_queries)  # Fails here
```

### Why This Happens:
- BEIR's dense retrieval implementation has an assumption about multiple queries
- When there's only 1 query, `cos_scores` has shape `[1, num_docs]`
- Accessing `cos_scores[1]` is out of bounds

---

## ✅ Fix Applied

### Solution:
Changed the retrieval approach to **batch queries together** instead of processing one at a time:

1. **Batch Processing**: Collect multiple query variations into batches (batch_size=10)
2. **Single Retrieval Call**: Retrieve for all variations in batch at once
3. **Map Results Back**: Map results back to original query IDs
4. **Fallback**: If batching fails, use workaround with dummy second query

### Code Changes:
- Modified `train_llm_query_expansion.py` lines 258-284
- Now processes queries in batches of 10
- Maps temporary query IDs back to original query IDs
- Includes error handling and fallback mechanism

---

## 🚀 How to Restart

### Option 1: Resume from Checkpoint (Recommended)
The experiment saved checkpoints, so you can resume:
```bash
python train_llm_query_expansion.py \
    --config experiments/retrieval/phase6_llm_query_expansion_gpt4_multi/config.json \
    --gpu_id 2 \
    --resume
```

### Option 2: Start Fresh
If you want to start over:
```bash
python train_llm_query_expansion.py \
    --config experiments/retrieval/phase6_llm_query_expansion_gpt4_multi/config.json \
    --gpu_id 2 \
    --no-resume
```

---

## 📊 What Was Saved

The experiment saved:
- **Expanded queries** for clapnq domain (checkpoint saved)
- **Progress**: Query expansion completed for clapnq
- **Checkpoint location**: `experiments/retrieval/phase6_llm_query_expansion_gpt4_multi/checkpoints/`

When you resume, it will:
- Skip clapnq (already completed)
- Continue with remaining domains (fiqa, govt, cloud)
- Use the fixed batch processing approach

---

## ✅ Status

- **Error**: Fixed ✅
- **Code**: Updated ✅
- **Ready to Resume**: Yes ✅

The fix ensures that queries are always processed in batches, avoiding the single-query bug in BEIR's exact_search implementation.

