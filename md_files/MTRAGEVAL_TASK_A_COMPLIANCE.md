# MTRAGEval Task A Compliance Check

This document ensures all experiments comply with MTRAGEval Task A (Retrieval Only) requirements.

**Task Information**: [MTRAGEval Official Page](https://ibm.github.io/mt-rag-benchmark/MTRAGEval/)
**Repository**: [IBM/mt-rag-benchmark](https://github.com/IBM/mt-rag-benchmark)

---

## 📋 Task A Requirements

### Evaluation Criteria
- **Ranking Metric**: nDCG (Normalized Discounted Cumulative Gain)
- **Task**: Retrieval Only
- **Evaluation Period**: January 10-20, 2026

### Key Constraints

#### ✅ ALLOWED During Training (Trial/Training Data)
1. **Metadata Access**: Can use all metadata from training data:
   - Question type (e.g., factoid)
   - Answerability (e.g., unanswerable, answerable)
   - Multi-turn type (e.g., follow-up, clarification)
   - Domain information
   - Full conversation history

2. **Data Usage**: Can use:
   - Training data (MTRAG Benchmark)
   - Trial data
   - Any publicly available models/embeddings
   - Any training techniques (fine-tuning, domain-specific, etc.)

3. **Model Training**: Can train:
   - Domain-specific models
   - Multi-domain models
   - Ensemble models
   - Any retrieval architecture

#### ❌ NOT ALLOWED During Evaluation
1. **Metadata Access**: Will NOT be provided:
   - Question type
   - Answerability information
   - Multi-turn type
   - Conversation history details
   - **ONLY domain will be provided** (e.g., ClapNQ, Govt, Cloud, FiQA)

2. **Data Restrictions**:
   - Cannot use evaluation data for training
   - Cannot access metadata not provided
   - Must work with domain-only information

### Output Format Requirements

According to [evaluation README](scripts/evaluation/README.md), retrieval results must be in this format:

```json
{
  "task_id": "unique_id",
  "contexts": [
    {
      "document_id": "822086267_6698-7277-0-579",
      "score": 18.759138,
      "text": "...",  // optional
      "title": "...", // optional
      "source": ""    // optional
    }
  ]
}
```

**Required Fields**:
- `document_id`: Must match corpus document IDs
- `score`: Retrieval score (higher = more relevant)

**Optional Fields**:
- `text`, `title`, `source`: Can be included but not required for evaluation

---

## ✅ Compliance Check: Our Experiments

### Phase 1: Foundation Experiments
**Status**: ✅ **COMPLIANT**

- **Baseline, Epochs 3, Epochs 5**: 
  - Use training data with metadata (allowed)
  - Train on all domains
  - No evaluation-time metadata dependency
  - ✅ **COMPLIANT**

### Phase 2: Hyperparameter Tuning
**Status**: ✅ **COMPLIANT**

- **Learning Rate Experiments, Augmentation**:
  - Use training data with metadata (allowed)
  - Data augmentation uses training data only
  - No evaluation-time metadata dependency
  - ✅ **COMPLIANT**

### Phase 3: Hybrid & Reranking
**Status**: ✅ **COMPLIANT**

- **Hybrid Retrieval, Reranking**:
  - Combine BM25 + dense retrieval (allowed)
  - Cross-encoder reranking (allowed)
  - No evaluation-time metadata dependency
  - ✅ **COMPLIANT**

### Phase 4: Advanced Techniques
**Status**: ✅ **COMPLIANT**

- **Domain-Specific Models**:
  - Train separate models per domain (allowed)
  - Use domain information from training data (allowed)
  - At evaluation: Only need domain to select model (domain will be provided)
  - ✅ **COMPLIANT**

- **Hard Negative Mining**:
  - Training technique only (allowed)
  - No evaluation-time metadata dependency
  - ✅ **COMPLIANT**

- **BGE-Large**:
  - Larger model architecture (allowed)
  - No evaluation-time metadata dependency
  - ✅ **COMPLIANT**

### Phase 5: Ensemble & Advanced
**Status**: ✅ **COMPLIANT**

- **Ensemble Methods**:
  - Combine multiple models (allowed)
  - Use domain to select ensemble strategy (domain will be provided)
  - ✅ **COMPLIANT**

- **Combined Techniques**:
  - Domain-specific + hard negatives (allowed)
  - Training technique only
  - ✅ **COMPLIANT**

- **Reranking**:
  - Cross-encoder reranking (allowed)
  - No evaluation-time metadata dependency
  - ✅ **COMPLIANT**

- **Query Expansion**:
  - Expand queries before retrieval (allowed)
  - Can use LLMs or synonym expansion
  - No evaluation-time metadata dependency
  - ✅ **COMPLIANT**

- **Hybrid Learned Weights**:
  - Learn optimal BM25/dense weights (allowed)
  - Can optimize per domain (domain will be provided)
  - ✅ **COMPLIANT**

---

## ⚠️ Potential Compliance Issues & Solutions

### Issue 1: Using Metadata During Evaluation
**Risk**: If any experiment tries to use question type, answerability, or multi-turn type during evaluation

**Solution**: 
- ✅ All our experiments only use domain information at evaluation time
- ✅ Domain will be provided during evaluation
- ✅ No experiments access metadata not provided

**Status**: ✅ **SAFE**

---

### Issue 2: Query Rewriting/Expansion
**Question**: Can we rewrite queries during evaluation?

**Answer**: 
- ✅ **YES** - Query rewriting/expansion is allowed
- The benchmark shows "Query Rewrite" as a valid setup (see README results)
- We can use LLMs, synonym expansion, etc.
- ✅ **COMPLIANT**

---

### Issue 3: Domain-Specific Models
**Question**: Can we use different models per domain?

**Answer**:
- ✅ **YES** - Domain-specific models are allowed
- Domain will be provided during evaluation
- We can select appropriate model based on domain
- ✅ **COMPLIANT**

---

### Issue 4: Ensemble Methods
**Question**: Can we combine multiple models?

**Answer**:
- ✅ **YES** - Ensemble methods are allowed
- Can combine any models trained on training data
- Can use RRF, weighted average, etc.
- ✅ **COMPLIANT**

---

### Issue 5: Using External Models/APIs
**Question**: Can we use external LLMs for query expansion?

**Answer**:
- ✅ **YES** - Can use any publicly available models/APIs
- LLM-based query expansion is allowed
- External embeddings/models are allowed
- ✅ **COMPLIANT**

---

## 📝 Evaluation Submission Requirements

### Format Compliance
All experiments must output results in the required format:

```json
{
  "task_id": "string",
  "contexts": [
    {
      "document_id": "string",  // REQUIRED
      "score": float,           // REQUIRED
      "text": "string",         // Optional
      "title": "string",        // Optional
      "source": "string"        // Optional
    }
  ]
}
```

### Our Compliance
- ✅ All evaluation scripts output correct format
- ✅ `document_id` matches corpus IDs
- ✅ `score` is retrieval score
- ✅ Results can be evaluated with official scripts

---

## 🎯 Recommended Approach for Evaluation

### Strategy
1. **Use Domain-Specific Models**: 
   - Train separate models per domain (already done)
   - At evaluation: Use domain to select model
   - ✅ Domain will be provided

2. **Ensemble Best Models**:
   - Combine domain-specific models
   - Use RRF or weighted average
   - ✅ No metadata dependency

3. **Query Expansion**:
   - Use LLM-based expansion (if available)
   - Or synonym-based expansion
   - ✅ Allowed technique

4. **Reranking**:
   - Use cross-encoder reranking
   - Fine-tune on training data
   - ✅ Allowed technique

5. **Hybrid Retrieval**:
   - Combine BM25 + dense
   - Learn optimal weights per domain
   - ✅ Domain will be provided

### What We CANNOT Do
- ❌ Use question type during evaluation (not provided)
- ❌ Use answerability information (not provided)
- ❌ Use multi-turn type (not provided)
- ❌ Access conversation history details (not provided)
- ❌ Use evaluation data for training

### What We CAN Do
- ✅ Use domain information (will be provided)
- ✅ Use query text (will be provided)
- ✅ Use any models trained on training data
- ✅ Use query expansion/rewriting
- ✅ Use ensemble methods
- ✅ Use reranking
- ✅ Use hybrid retrieval
- ✅ Use external models/APIs

---

## ✅ Final Compliance Status

### All Experiments: ✅ **COMPLIANT**

**Summary**:
- ✅ All experiments use only training data with metadata (allowed)
- ✅ No experiments depend on evaluation-time metadata
- ✅ Domain-specific models only need domain (will be provided)
- ✅ All techniques (ensemble, reranking, expansion) are allowed
- ✅ Output format matches requirements
- ✅ Retrieval evaluation scripts verified - no metadata usage

**Code Verification**:
- ✅ `evaluate_advanced_models.py`: Only uses domain, queries, corpus (no metadata)
- ✅ `evaluate_finetuned_bge.py`: Only uses domain, queries, corpus (no metadata)
- ✅ All Phase 5 evaluation scripts: Only use domain information
- ⚠️ Metadata found only in generation evaluation scripts (Task B/C, not Task A)

**No violations detected.**

---

## 📋 Pre-Evaluation Checklist

Before submitting to MTRAGEval:

- [ ] Verify all models only use domain information at evaluation time
- [ ] Test evaluation script with official format
- [ ] Ensure document_ids match corpus IDs
- [ ] Verify scores are properly normalized/ranked
- [ ] Test on trial data (if available)
- [ ] Document all techniques used
- [ ] Ensure no evaluation data leakage

---

## 🔗 References

- [MTRAGEval Official Page](https://ibm.github.io/mt-rag-benchmark/MTRAGEval/)
- [IBM/mt-rag-benchmark Repository](https://github.com/IBM/mt-rag-benchmark)
- [Evaluation README](scripts/evaluation/README.md)
- [Retrieval Tasks README](human/retrieval_tasks/README.md)

---

*Last Updated: 2025-12-13*
*Compliance Status: ✅ ALL EXPERIMENTS COMPLIANT*

