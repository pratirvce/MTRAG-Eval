# Failed Experiments Analysis

## Summary

Out of 5 experiments started, all 5 failed due to script bugs. Here's a detailed breakdown:

---

## 1. ❌ best_paper_adversarial_curriculum (GPU 1)

**Error:** `HFValidationError: Repo id must use alphanumeric chars...`

**Root Cause:** The script is trying to pass a model object's string representation instead of a model ID/path to HuggingFace Hub. It's passing:
```
'sentence-transformers/SentenceTransformer(...)' 
```
instead of a valid model identifier like `'BAAI/bge-base-en-v1.5'`.

**Fix Needed:** Check `train_adversarial_curriculum_tier1.py` - ensure model loading uses a string path/ID, not a model object.

**Location:** HuggingFace Hub validation when trying to load adapter configs.

---

## 2. ❌ best_paper_hierarchical_routing (GPU 2)

**Error:** `IndexError: index 1 is out of bounds for dimension 0 with size 1`

**Root Cause:** The script processes queries one at a time in a loop:
```python
for qid, query_text in queries.items():
    stage1_results = evaluator.retrieve(corpus, {qid: query_text})
```

But the BEIR dense search code expects multiple queries. When `cos_scores[1]` is accessed, it fails because there's only one query (size 1).

**Fix Needed:** Modify `train_hierarchical_routing.py` to batch queries instead of processing one at a time, or modify the retrieval logic to handle single-query batches.

**Location:** `beir/retrieval/search/dense/exact_search.py:102`

---

## 3. ❌ best_paper_large_model_finetuning (GPU 3)

**Error:** `No training examples found!`

**Root Cause:** The script successfully loads corpus and queries from all domains but finds 0 training examples. This suggests the data loading/example generation logic is broken - it's not converting the loaded data into training examples.

**Fix Needed:** Check `train_large_model_tier1.py` - verify that it's properly converting queries and documents into training examples. The logs show it loaded documents and queries but couldn't create examples.

**Location:** After loading all domain data, before training starts.

---

## 4. ❌ best_paper_learned_rrf (GPU 4)

**Error:** `TypeError: Unable to extract query/object scores.`

**Root Cause:** The `final_results` dictionary format doesn't match what the BEIR evaluator expects. The evaluator expects:
```python
{
  'query_id': {'doc_id': score, ...},
  ...
}
```
But it seems the format might be incorrect or missing required keys.

**Fix Needed:** Check `train_learned_rrf.py` around line 261 - verify the `final_results` format matches BEIR's expected format for evaluation.

**Location:** `train_learned_rrf.py:261` during evaluation.

---

## 5. ❌ best_paper_llm_distillation (GPU 5)

**Error:** `HFValidationError: Repo id must use alphanumeric chars...`

**Root Cause:** Same as #1 - passing model object string representation instead of model ID.

**Fix Needed:** Same as adversarial_curriculum - check model loading in `train_llm_distillation_tier1.py`.

---

## Recommendations

### Immediate Actions:
1. **Fix model loading issues** in adversarial_curriculum and llm_distillation scripts
2. **Fix single-query batch handling** in hierarchical_routing script  
3. **Fix training example generation** in large_model_finetuning script
4. **Fix results format** in learned_rrf script

### Scripts to Check:
- `train_adversarial_curriculum_tier1.py` - model loading
- `train_hierarchical_routing.py` - query batching  
- `train_large_model_tier1.py` - example generation
- `train_learned_rrf.py` - results format
- `train_llm_distillation_tier1.py` - model loading

### Testing Strategy:
1. Fix each script individually
2. Test with a small subset (one domain) before full run
3. Verify results format before evaluation step

---

## Current Status

**Running:** 
- ✅ `tier1_learning_to_rank_listwise` (GPU 0) - Working correctly

**Failed:**
- ❌ All 5 newly started experiments crashed within minutes

**Next Steps:**
1. Fix the script bugs listed above
2. Re-run failed experiments after fixes
3. Consider testing on a single domain first to catch errors early

