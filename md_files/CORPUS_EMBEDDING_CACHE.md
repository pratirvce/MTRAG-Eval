# Corpus Embedding Cache - Optimization Implemented ✅

## Problem Identified

The original implementation was **re-encoding the corpus inside the query batch loop**, meaning:
- Corpus was encoded **once per query batch** (very inefficient!)
- For 1000 queries with batch_size=32, corpus was encoded **31 times**!
- This wasted hours of computation time

## Solution Implemented

### ✅ Optimizations:

1. **Encode corpus ONCE per domain** (not per query batch)
2. **Cache corpus embeddings** to disk (`.npy` files)
3. **Reuse cached embeddings** if they exist and match
4. **Save time**: Corpus encoding happens once, then reused for all queries

### Cache Location:
```
experiments/retrieval/phase7_conversation_aware_attention/cache/
├── clapnq_corpus_embeddings.npy
├── fiqa_corpus_embeddings.npy
├── govt_corpus_embeddings.npy
└── cloud_corpus_embeddings.npy
```

### How It Works:

1. **First run**: Encodes corpus and saves to cache
2. **Subsequent runs**: Loads from cache (instant!)
3. **Cache validation**: Checks if cache size matches corpus size
4. **Automatic fallback**: Re-encodes if cache is invalid

### Time Savings:

- **Before**: Corpus encoded 31 times (for 1000 queries, batch_size=32)
- **After**: Corpus encoded once, cached, reused
- **Savings**: ~30x faster for corpus encoding step!

### Example:
- **ClapNQ**: 183,408 documents
  - Encoding time: ~30-60 minutes
  - **With cache**: Loads in seconds on subsequent runs!

---

## Usage

### Automatic:
- Cache is created automatically on first run
- Cache is reused automatically on subsequent runs
- No manual intervention needed

### Manual Cache Management:
```bash
# Check cache files
ls -lh experiments/retrieval/phase7_conversation_aware_attention/cache/

# Delete cache to force re-encoding
rm experiments/retrieval/phase7_conversation_aware_attention/cache/*.npy

# Cache is per-domain, so you can delete specific domain cache
rm experiments/retrieval/phase7_conversation_aware_attention/cache/clapnq_corpus_embeddings.npy
```

---

## Cache File Format

- **Format**: NumPy array (`.npy`)
- **Shape**: `[num_documents, embedding_dim]` (e.g., `[183408, 768]`)
- **Size**: ~560MB per 100K documents (768-dim embeddings, float32)

### Example Sizes:
- **ClapNQ**: ~1.1 GB (183K documents)
- **Cloud**: ~350 MB (61K documents)
- **FiQA**: ~300 MB (49K documents)
- **Govt**: ~450 MB (72K documents)

---

## Benefits

1. ✅ **Faster execution**: Corpus encoded once, reused many times
2. ✅ **Resume-friendly**: Cache persists across runs
3. ✅ **GPU-efficient**: Corpus encoding happens once, freeing GPU for queries
4. ✅ **Time-saving**: Saves hours of computation on subsequent runs

---

## Notes

- **Model-dependent**: Cache is specific to the model used
- **Corpus-dependent**: Cache is specific to the corpus version
- **Automatic validation**: Checks cache size matches corpus size
- **Safe to delete**: Cache will be regenerated if missing

---

## Status

✅ **Optimization implemented and active!**

The experiment now:
1. Checks for cached corpus embeddings
2. Loads if available and valid
3. Encodes and caches if not available
4. Reuses cache for all queries in that domain

**This saves significant time, especially on resume!** 🚀

