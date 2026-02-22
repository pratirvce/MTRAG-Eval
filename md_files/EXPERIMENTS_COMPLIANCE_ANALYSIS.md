# Compliance Analysis: Experiments for nDCG@10 > 0.90

**Reference:** [MTRAGEval Task A - Retrieval Only](https://ibm.github.io/mt-rag-benchmark/MTRAGEval/)  
**Benchmark Repository:** [IBM/mt-rag-benchmark](https://github.com/IBM/mt-rag-benchmark/)

---

## Task A Compliance Rules Summary

### ✅ **ALLOWED:**
1. **Retrieval-only systems** - No text generation
2. **Query processing/rewriting** - LLMs can be used for query expansion/rewriting (preprocessing)
3. **Relevance scoring** - Models can score/rank documents (no text generation)
4. **Multi-stage retrieval** - Dense → Sparse → Reranking pipelines
5. **Ensemble methods** - Combining multiple retrieval models
6. **Training on MTRAG data** - Using training data for fine-tuning
7. **Any embedding model** - Can use any pre-trained or fine-tuned model
8. **Corpus processing** - Can re-chunk, re-index, preprocess documents

### ❌ **NOT ALLOWED:**
1. **Text generation** - Cannot generate answers or responses
2. **Generation feedback** - Cannot use generation quality to optimize retrieval
3. **Metadata usage** - Cannot use question type, answerability, multi-turn type (only corpus domain available)
4. **Generation objectives** - Cannot optimize for generation quality

---

## Compliance Analysis of Proposed Experiments

### ✅ **FULLY COMPLIANT EXPERIMENTS**

#### 1. **Multi-Stage Retrieval with LLM Reranking** ⚠️ **NEEDS MODIFICATION**
**Status:** ⚠️ **CONDITIONALLY COMPLIANT** (with modification)

**Original Proposal:**
- Stage 4: LLM-based reranking with reasoning

**Compliance Issue:**
- If "LLM reranking" means generating text explanations → ❌ **NOT ALLOWED**
- If "LLM reranking" means LLM scores documents (relevance scoring) → ✅ **ALLOWED**

**Compliant Version:**
```python
# ✅ ALLOWED: LLM scores documents (no text generation)
def llm_relevance_scoring(query, documents):
    scores = []
    for doc in documents:
        # LLM outputs a score (0-10), not text
        score = llm.score(query, doc)  # Returns float, not text
        scores.append(score)
    return scores

# ❌ NOT ALLOWED: LLM generates text explanations
def llm_reranking_with_reasoning(query, documents):
    for doc in documents:
        explanation = llm.generate(f"Explain why {doc} is relevant...")  # Text generation
        # This is NOT allowed
```

**Recommendation:** ✅ **USE THIS** - But ensure LLM only outputs scores, not text

**Expected nDCG@10:** 0.85-0.92 (if properly implemented)

---

#### 2. **ColBERT-Style Multi-Vector Retrieval** ✅
**Status:** ✅ **FULLY COMPLIANT**

**Why Compliant:**
- Token-level embeddings are retrieval-only
- MaxSim scoring is pure retrieval
- Cross-encoder reranking is allowed
- No text generation involved

**Recommendation:** ✅ **USE THIS** - Zero compliance risk

**Expected nDCG@10:** 0.80-0.88

---

#### 3. **Ensemble with Meta-Learner Fusion** ✅
**Status:** ✅ **FULLY COMPLIANT**

**Why Compliant:**
- Combines multiple retrieval models
- Meta-learner learns fusion weights (not generation)
- All components are retrieval-only
- No text generation involved

**Recommendation:** ✅ **USE THIS** - Zero compliance risk

**Expected nDCG@10:** 0.82-0.90

---

#### 4. **Generative Query Expansion** ✅
**Status:** ✅ **FULLY COMPLIANT**

**Why Compliant:**
- LLM used for query preprocessing only
- Generates query variants (not answers)
- Query rewriting is explicitly allowed
- No answer generation

**Compliant Implementation:**
```python
# ✅ ALLOWED: LLM generates query variants
def generative_query_expansion(query, conversation_history):
    prompt = f"Generate 3 alternative phrasings for this query: {query}"
    query_variants = llm.generate(prompt)  # Returns query strings, not answers
    return query_variants  # These are used for retrieval, not as answers
```

**Recommendation:** ✅ **USE THIS** - Fully compliant

**Expected nDCG@10:** 0.75-0.82

---

#### 5. **BGE-v2-Large Asymmetric Encoding** ✅
**Status:** ✅ **FULLY COMPLIANT**

**Why Compliant:**
- Uses larger retrieval model
- Asymmetric prompts are model features
- No text generation
- Pure retrieval improvement

**Recommendation:** ✅ **USE THIS** - Zero compliance risk

**Expected nDCG@10:** 0.70-0.78

---

#### 6. **Iterative Retrieval with Generative Feedback** ⚠️ **NEEDS CLARIFICATION**
**Status:** ⚠️ **CONDITIONALLY COMPLIANT**

**Original Proposal:**
- Uses LLM to generate feedback based on retrieved documents
- Expands query with feedback terms

**Compliance Analysis:**
- ✅ **ALLOWED:** If feedback is query expansion terms (preprocessing)
- ❌ **NOT ALLOWED:** If feedback is answer generation or text explanations

**Compliant Version:**
```python
# ✅ ALLOWED: Generate query expansion terms
def iterative_retrieval_with_feedback(query, initial_results):
    # LLM generates expansion terms (not answers)
    feedback_prompt = f"Based on these documents, what terms should be added to improve retrieval?\nQuery: {query}\nDocuments: {initial_results[:5]}"
    expansion_terms = llm.generate(feedback_prompt)  # Returns terms, not answers
    expanded_query = f"{query} {expansion_terms}"
    return retriever.retrieve(expanded_query)
```

**Recommendation:** ✅ **USE THIS** - But ensure feedback is query expansion only, not answer generation

**Expected nDCG@10:** 0.72-0.80

---

## Summary: Compliance Status

| Experiment | Compliance Status | Modification Needed | Expected nDCG@10 |
|------------|------------------|-------------------|------------------|
| Multi-Stage with LLM Reranking | ⚠️ Conditional | Ensure LLM only scores (no text) | 0.85-0.92 |
| ColBERT Multi-Vector | ✅ Fully Compliant | None | 0.80-0.88 |
| Ensemble Meta-Learner | ✅ Fully Compliant | None | 0.82-0.90 |
| Generative Query Expansion | ✅ Fully Compliant | None | 0.75-0.82 |
| BGE-v2-Large Asymmetric | ✅ Fully Compliant | None | 0.70-0.78 |
| Iterative Generative Feedback | ⚠️ Conditional | Ensure feedback is query expansion only | 0.72-0.80 |

---

## Recommended Compliant Implementation

### Priority 1: Zero-Risk, High-Impact
1. ✅ **ColBERT Multi-Vector Retrieval** (0.80-0.88)
2. ✅ **Ensemble Meta-Learner Fusion** (0.82-0.90)
3. ✅ **BGE-v2-Large Asymmetric** (0.70-0.78)

**Combined Expected:** 0.85-0.95 nDCG@10

### Priority 2: Low-Risk, High-Impact
4. ✅ **Generative Query Expansion** (0.75-0.82)
5. ⚠️ **Multi-Stage with LLM Scoring** (0.85-0.92) - Modified version
6. ⚠️ **Iterative Retrieval with Query Expansion** (0.72-0.80) - Modified version

---

## Key Modifications Required

### 1. LLM Reranking → LLM Relevance Scoring
**Change:**
- ❌ **Before:** LLM generates text explanations for relevance
- ✅ **After:** LLM outputs relevance scores (0-10) only

**Implementation:**
```python
# Compliant version
def llm_relevance_scoring(query, documents):
    scores = []
    for doc in documents:
        # Use structured output or scoring API
        score = llm.score(
            prompt=f"Rate relevance 0-10: Query: {query}\nDocument: {doc}",
            output_type="score"  # Returns float, not text
        )
        scores.append(score)
    return scores
```

### 2. Generative Feedback → Query Expansion Feedback
**Change:**
- ❌ **Before:** LLM generates answer-like feedback
- ✅ **After:** LLM generates query expansion terms only

**Implementation:**
```python
# Compliant version
def iterative_query_expansion(query, retrieved_docs):
    # LLM generates expansion terms (not answers)
    expansion = llm.generate(
        prompt=f"What terms should be added to this query for better retrieval?\nQuery: {query}",
        output_type="terms"  # Returns terms, not answers
    )
    return f"{query} {expansion}"
```

---

## Final Compliance Checklist

Before implementing, ensure:

- [ ] ✅ All LLM usage is for query preprocessing or relevance scoring (not text generation)
- [ ] ✅ No answer generation or text explanations
- [ ] ✅ All techniques are retrieval-only
- [ ] ✅ No metadata usage beyond corpus domain
- [ ] ✅ Training uses only relevance labels (not generation feedback)
- [ ] ✅ Output format: `document_id` + `score` only

---

## Conclusion

**Most experiments are COMPLIANT** with Task A (Retrieval Only) requirements from [MTRAGEval](https://ibm.github.io/mt-rag-benchmark/MTRAGEval/).

**Key Modifications Needed:**
1. Ensure LLM reranking only outputs scores (not text)
2. Ensure generative feedback is query expansion only (not answers)

**Recommended Approach:**
- Start with fully compliant experiments (ColBERT, Ensemble, BGE-v2-Large)
- Add conditional experiments with proper modifications
- Combined expected: **0.85-0.95 nDCG@10** ✅

**Compliance Risk:** ✅ **LOW** - All experiments can be made compliant with minor modifications

---

**Last Updated:** 2025-12-21  
**Compliance Status:** ✅ Most experiments compliant, minor modifications needed  
**Reference:** [MTRAGEval Task A Rules](https://ibm.github.io/mt-rag-benchmark/MTRAGEval/)

