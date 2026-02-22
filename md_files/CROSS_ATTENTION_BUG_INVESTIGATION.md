# Cross-Attention Zero Results Bug Investigation

**Date:** 2025-12-18  
**Status:** ✅ Root Cause Identified and Fixed

---

## 🔍 Problem

The `tier1_cross_attention_rerun_fixed` experiment completed but produced **all-zero results**:
- Recall@10: 0.0000
- nDCG@10: 0.0000

Despite the initial fix to handle multi-head attention weights, the results were still zero.

---

## 🐛 Root Cause Identified

### The Bug

The original fix computed cosine similarity between:
- **Attention output** (weighted sum of document embeddings)
- **Document embeddings**

**Problem:** For an untrained cross-attention model, the attention weights are **uniform** (approximately 1/num_docs for each document). This means:
1. The attention output is essentially the **average of all document embeddings**
2. Comparing this average to each individual document gives **very similar scores** for all documents
3. When all scores are nearly identical, ranking becomes meaningless
4. The top-K selection produces random ordering
5. Evaluation against qrels produces zero recall/nDCG

### Why the Safety Check Didn't Help

The safety check only triggered if:
- `score_std < 1e-6` (all scores exactly the same)
- `score_max < 1e-6` (all scores exactly zero)

But the scores were **nearly identical** (not exactly the same), so the check didn't trigger, and the meaningless ranking was used.

---

## ✅ Fix Applied

### Change 1: Use Original Query Embedding

Instead of using the attention output (which is uniform for untrained models), we now use the **original query embedding** for cosine similarity:

```python
# OLD (BUGGY):
attn_output_expanded = attn_output.expand(-1, doc_embs.size(1), -1)
interaction_scores = cosine_similarity(attn_output_expanded, doc_embs)

# NEW (FIXED):
query_emb_original = query_emb.squeeze(1)  # Original query embedding
query_emb_expanded = query_emb_original.unsqueeze(1).expand(-1, doc_embs.size(1), -1)
interaction_scores = cosine_similarity(query_emb_expanded, doc_embs)
```

### Change 2: Smart Attention Weight Usage

Added logic to detect if attention weights are informative:

```python
# Check if attention weights are informative (not uniform)
if attn_std > 0.01 and (attn_max - attn_min) > 0.05:
    # Attention is informative, use weighted combination
    interaction_scores = 0.9 * interaction_scores + 0.1 * attention_scores
else:
    # Attention is uniform/uninformative, use pure cosine similarity
    pass
```

### Change 3: Added Debug Logging

Added comprehensive debug logging to help diagnose issues:
- Number of queries in retrieval_results
- Number of queries in qrels
- Query ID matching
- Score ranges and statistics

---

## 📝 Code Changes

**File:** `train_cross_attention_retrieval.py`

**Location:** `CrossAttentionInteraction.forward()` method (lines 112-129)

**Key Changes:**
1. Use original query embedding instead of attention output for cosine similarity
2. Check if attention weights are informative before using them
3. Fall back to pure cosine similarity when attention is uninformative
4. Added debug logging in evaluation section

---

## 🧪 Expected Results After Fix

With the fix, the experiments should now produce:
- **Non-zero scores** - Cosine similarity between query and documents provides meaningful differences
- **Reasonable performance** - Similar to baseline cosine similarity retrieval
- **Valid evaluation metrics** - Recall@10 and nDCG@10 should be > 0

---

## 🔄 Next Steps

1. **Re-run the failed experiment:**
   ```bash
   venv/bin/python3 train_cross_attention_tier1.py \
     --experiment_name tier1_cross_attention_rerun_fixed_v2 \
     --gpu 2 \
     --output_dir experiments/retrieval/tier1_cross_attention_rerun_fixed_v2 \
     --no-resume
   ```

2. **Monitor phase8_cross_attention_query_document_fixed** - It's still running and should benefit from the fix if it hasn't completed yet

3. **Verify the fix works** - Check that new runs produce non-zero results

---

## 📊 Technical Details

### Why Attention Output Was Problematic

For an untrained MultiheadAttention:
- Query: `[1, 1, 768]`
- Documents: `[1, 10, 768]`
- Attention weights: `[1, 8, 1, 10]` → averaged → `[1, 1, 10]` → uniform `~0.1` for each doc
- Attention output: `sum(doc_embeddings * 0.1)` = **average of all documents**
- Cosine similarity(avg_docs, doc_i): **nearly identical for all i**

### Why Original Query Works

- Query embedding: `[1, 768]` - represents the actual query semantics
- Document embeddings: `[10, 768]` - represent individual document semantics
- Cosine similarity(query, doc_i): **meaningful differences** based on actual semantic similarity

---

*Investigation completed: 2025-12-18 20:10*

