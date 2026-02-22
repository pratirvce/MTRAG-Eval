# Task A Compliance: Embedding and Chunking Improvements

## Task A Rules (from MTRAGEval)

### ✅ Allowed:
1. **Retrieval-only systems** - No text generation
2. **Any embedding model** - Can use any pre-trained or fine-tuned model
3. **Any retrieval method** - Dense, sparse, hybrid, reranking
4. **Corpus processing** - Can re-chunk, re-index, preprocess documents
5. **Query processing** - Can rewrite, expand, transform queries

### ❌ NOT Allowed:
1. **Text generation** - Cannot generate answers
2. **Metadata usage** - Cannot use question type, answerability, multi-turn type (only corpus domain available)
3. **Evaluation data access** - Cannot use evaluation queries during training

---

## Compliance Analysis of Embedding & Chunking Improvements

### ✅ FULLY COMPLIANT (Safe for Paper Submission)

#### 1. **Larger Base Models (BGE-v2-large, BGE-large)** ✅
**Status:** ✅ **FULLY ALLOWED**

**Why:**
- Using a different embedding model is explicitly allowed
- No generation involved
- Pure retrieval improvement

**Paper Impact:** ✅ **POSITIVE**
- Shows you used state-of-the-art models
- Demonstrates thorough experimentation
- Expected improvement: +0.05-0.08 nDCG@10

**Recommendation:** ✅ **USE THIS** - High impact, zero risk

---

#### 2. **Asymmetric Query-Document Encoding** ✅
**Status:** ✅ **FULLY ALLOWED**

**Why:**
- Just a different encoding strategy
- Uses built-in BGE model features (prompts)
- No generation, pure retrieval

**Paper Impact:** ✅ **POSITIVE**
- Shows understanding of query-document asymmetry
- Demonstrates proper use of model features
- Expected improvement: +0.03-0.06 nDCG@10

**Recommendation:** ✅ **USE THIS** - Easy to implement, good gains

---

#### 3. **Cross-Encoder Reranking** ✅
**Status:** ✅ **FULLY ALLOWED**

**Why:**
- Reranking is explicitly part of retrieval
- No text generation
- Standard retrieval technique

**Paper Impact:** ✅ **POSITIVE**
- Well-established technique
- Shows multi-stage retrieval pipeline
- Expected improvement: +0.08-0.15 nDCG@10

**Recommendation:** ✅ **USE THIS** - Highest impact improvement

---

#### 4. **Fine-Tuning on MTRAG Training Data** ✅
**Status:** ✅ **FULLY ALLOWED**

**Why:**
- Training on provided training data is allowed
- No generation involved
- Standard practice in retrieval

**Paper Impact:** ✅ **POSITIVE**
- Shows domain adaptation
- Demonstrates proper use of training data
- Expected improvement: +0.03-0.06 nDCG@10

**Recommendation:** ✅ **USE THIS** - Standard practice, good gains

---

#### 5. **Semantic Chunking** ✅
**Status:** ✅ **FULLY ALLOWED** (with caveat)

**Why:**
- Corpus processing is allowed
- Can re-chunk documents for indexing
- No generation involved

**Important Note:**
- The corpus is provided as full documents
- You can re-chunk them during indexing
- The qrels reference specific chunk IDs, but you can create your own chunks
- **However:** You must ensure your chunks can be mapped back to the original document IDs for evaluation

**Paper Impact:** ✅ **POSITIVE**
- Shows understanding of chunking importance
- Demonstrates preprocessing improvements
- Expected improvement: +0.03-0.06 nDCG@10

**Recommendation:** ✅ **USE THIS** - But ensure proper document ID mapping

---

#### 6. **Hierarchical Chunking** ✅
**Status:** ✅ **FULLY ALLOWED**

**Why:**
- Multi-granularity retrieval is allowed
- Can index documents at multiple levels
- No generation involved

**Paper Impact:** ✅ **POSITIVE**
- Novel approach to chunking
- Shows multi-level retrieval understanding
- Expected improvement: +0.03-0.06 nDCG@10

**Recommendation:** ✅ **USE THIS** - Novel and effective

---

#### 7. **Hybrid Dense-Sparse Embeddings** ✅
**Status:** ✅ **FULLY ALLOWED**

**Why:**
- Hybrid retrieval is explicitly allowed
- No generation involved
- Standard retrieval technique

**Paper Impact:** ✅ **POSITIVE**
- Well-established technique
- Shows comprehensive retrieval approach
- Expected improvement: +0.02-0.05 nDCG@10

**Recommendation:** ✅ **USE THIS** - Already implemented in your codebase

---

#### 8. **Multi-Vector Embeddings (ColBERT-style)** ✅
**Status:** ✅ **FULLY ALLOWED**

**Why:**
- Different embedding strategy
- No generation involved
- Token-level matching is still retrieval

**Paper Impact:** ✅ **POSITIVE**
- State-of-the-art technique
- Shows advanced retrieval understanding
- Expected improvement: +0.04-0.08 nDCG@10

**Recommendation:** ✅ **USE THIS** - Advanced but effective

---

### ⚠️ CONDITIONALLY ALLOWED (Need Careful Implementation)

#### 9. **Query-Aware Chunking** ⚠️
**Status:** ⚠️ **CONDITIONALLY ALLOWED**

**Why Allowed:**
- Query processing is allowed
- Can select/re-rank chunks based on query

**Potential Issues:**
- If you re-chunk the corpus per query, this might be seen as "generating" new documents
- However, if you pre-compute multiple chunking strategies and select the best one per query, this should be fine

**Paper Impact:** ⚠️ **NEUTRAL to POSITIVE**
- Novel approach
- But might be seen as too complex
- Expected improvement: +0.03-0.06 nDCG@10

**Recommendation:** ⚠️ **USE WITH CAUTION**
- Pre-compute multiple chunking strategies
- Select best chunks per query (not re-chunk per query)
- Document clearly in paper

---

#### 10. **Adaptive Chunk Sizes** ✅
**Status:** ✅ **ALLOWED** (but limited benefit)

**Why:**
- Can use different chunk sizes per domain
- Corpus processing is allowed

**Note:**
- The evaluation provides corpus domain (ClapNQ, FiQA, Govt, Cloud)
- You can use this to select chunk size
- This is allowed since domain is provided

**Paper Impact:** ✅ **POSITIVE**
- Shows domain-aware processing
- Demonstrates thoughtful preprocessing
- Expected improvement: +0.02-0.04 nDCG@10

**Recommendation:** ✅ **USE THIS** - Safe and effective

---

## 🚫 NOT ALLOWED / PROBLEMATIC

### ❌ **Dynamic Re-chunking Per Query**
**Status:** ❌ **NOT RECOMMENDED**

**Why:**
- Re-chunking the corpus for each query might be seen as creating new documents
- Could be interpreted as "generation" of new passages
- Evaluation expects consistent document IDs

**Recommendation:** ❌ **AVOID THIS**
- Instead, pre-compute multiple chunking strategies
- Select best chunks at retrieval time (not re-chunk)

---

## 📋 Recommended Approach for Paper Submission

### ✅ Safe and High-Impact Improvements:

1. **BGE-v2-large or BGE-large** ✅
   - Zero risk, high impact
   - Expected: +0.05-0.08 nDCG@10

2. **Asymmetric Encoding** ✅
   - Zero risk, easy implementation
   - Expected: +0.03-0.06 nDCG@10

3. **Cross-Encoder Reranking** ✅
   - Zero risk, highest impact
   - Expected: +0.08-0.15 nDCG@10

4. **Semantic Chunking** ✅
   - Low risk, high impact
   - Expected: +0.03-0.06 nDCG@10
   - **Important:** Ensure proper document ID mapping

5. **Fine-Tuning on MTRAG Data** ✅
   - Zero risk, standard practice
   - Expected: +0.03-0.06 nDCG@10

6. **Hierarchical Chunking** ✅
   - Low risk, novel approach
   - Expected: +0.03-0.06 nDCG@10

7. **Hybrid Dense-Sparse** ✅
   - Zero risk, well-established
   - Expected: +0.02-0.05 nDCG@10

**Combined Expected Gain:** +0.27-0.52 nDCG@10  
**From 0.51 baseline → 0.78-1.03 nDCG@10** (theoretical maximum)

---

## 📝 Paper Writing Recommendations

### What to Emphasize:

1. **Retrieval-Only Focus:**
   - Emphasize that all improvements are retrieval-only
   - No text generation involved
   - All techniques are standard in IR literature

2. **Preprocessing Improvements:**
   - Frame chunking improvements as "corpus preprocessing"
   - Show that you're optimizing the indexing strategy
   - Document that you maintain document ID consistency

3. **Multi-Stage Pipeline:**
   - Frame as "hierarchical retrieval pipeline"
   - Show progression: chunking → embedding → retrieval → reranking
   - Emphasize that each stage is retrieval-only

4. **Reproducibility:**
   - Document all chunking strategies clearly
   - Provide code for corpus preprocessing
   - Show that results are reproducible

### What to Avoid:

1. ❌ Don't call it "query-aware document generation"
2. ❌ Don't imply you're creating new documents
3. ❌ Don't use metadata that won't be available at evaluation time

---

## ✅ Final Compliance Checklist

Before submitting, ensure:

- [ ] All techniques are retrieval-only (no text generation)
- [ ] Chunking is done during indexing (not per-query)
- [ ] Document IDs are properly mapped for evaluation
- [ ] No metadata usage (only corpus domain)
- [ ] All improvements are standard IR techniques
- [ ] Code is reproducible and well-documented

---

## 🎯 Recommended Implementation Priority

### Phase 1: Zero-Risk, High-Impact (Week 1)
1. ✅ BGE-v2-large
2. ✅ Asymmetric encoding
3. ✅ Cross-encoder reranking
4. ✅ Fine-tuning on MTRAG data

**Expected:** +0.19-0.35 nDCG@10  
**Risk:** Zero

### Phase 2: Low-Risk, High-Impact (Week 2)
5. ✅ Semantic chunking
6. ✅ Hierarchical chunking
7. ✅ Hybrid dense-sparse

**Expected:** +0.08-0.17 nDCG@10  
**Risk:** Low (ensure proper ID mapping)

### Phase 3: Advanced (Week 3-4)
8. ✅ Multi-vector embeddings (ColBERT)
9. ✅ Adaptive chunk sizes
10. ⚠️ Query-aware chunking (pre-computed only)

**Expected:** +0.09-0.18 nDCG@10  
**Risk:** Low to Medium

---

## 📊 Summary

**All embedding and chunking improvements are COMPLIANT with Task A** as long as:

1. ✅ No text generation
2. ✅ Chunking is done during indexing (not per-query re-chunking)
3. ✅ Document IDs are properly maintained
4. ✅ No metadata usage beyond corpus domain

**Paper Acceptance Risk:** ✅ **LOW**
- All techniques are standard in IR literature
- No generation involved
- Proper preprocessing is expected and encouraged
- Multi-stage retrieval pipelines are common

**Recommendation:** ✅ **PROCEED WITH CONFIDENCE**

These improvements are not only allowed but **expected** for a competitive submission. They demonstrate:
- Thorough experimentation
- Understanding of retrieval fundamentals
- Proper use of state-of-the-art techniques
- Attention to preprocessing details

---

**Last Updated:** 2025-12-20  
**Compliance Status:** ✅ All improvements are Task A compliant  
**Paper Risk:** ✅ Low risk - all techniques are standard IR practices

