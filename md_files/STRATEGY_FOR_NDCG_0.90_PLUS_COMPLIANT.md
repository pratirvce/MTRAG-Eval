# Strategic Experiments for nDCG@10 > 0.90 on MTRAGEval Task A (Retrieval Only) - COMPLIANCE VERIFIED

**Target:** nDCG@10 > 0.90 (current best: 0.5101)  
**Gap:** Need +0.39 improvement (76% relative improvement)  
**Benchmark:** [MTRAGEval Task A - Retrieval Only](https://ibm.github.io/mt-rag-benchmark/MTRAGEval/)  
**Compliance Status:** ✅ All experiments verified for Task A (Retrieval Only) compliance

---

## Task A Compliance Rules (from MTRAGEval)

### ✅ Allowed:
1. **Retrieval-only systems** - No text generation
2. **Query processing/rewriting** - LLMs can be used for query expansion/rewriting (preprocessing)
3. **Relevance scoring** - Models can score/rank documents (no text generation)
4. **Multi-stage retrieval** - Dense → Sparse → Reranking pipelines
5. **Ensemble methods** - Combining multiple retrieval models
6. **Training on MTRAG data** - Using training data for fine-tuning

### ❌ NOT Allowed:
1. **Text generation** - Cannot generate answers or responses
2. **Generation feedback** - Cannot use generation quality to optimize retrieval
3. **Metadata usage** - Cannot use question type, answerability, multi-turn type (only corpus domain available)
4. **Generation objectives** - Cannot optimize for generation quality

### Evaluation Format:
- Output: `contexts` list with `document_id` and `score` (required)
- Metric: nDCG@10 (normalized Discounted Cumulative Gain)
- Input: Corpus domain (e.g., ClapNQ, Govt) + conversation history

---

## Current Performance Baseline

| System | nDCG@10 | Notes |
|--------|---------|-------|
| **Current Best (Our Experiments)** | **0.5101** | best_paper_large_model_finetuning |
| Elser + Query Rewrite (MTRAG Paper) | 0.54 | Best baseline from paper |
| BGE-base + Query Rewrite (MTRAG Paper) | 0.38 | Strong baseline |
| BGE-base Last Turn Only | 0.30 | Without conversation context |

**Key Insight:** Query rewriting provides +0.08 improvement (0.30 → 0.38), showing conversation context is critical.

---

## Strategic Analysis: What's Needed for nDCG@10 > 0.90

### Gap Analysis
- **Current:** 0.5101
- **Target:** 0.90
- **Required Improvement:** +0.3899 (76% relative improvement)

### Critical Success Factors (Task A Compliant)

1. **Multi-Stage Retrieval** ✅ (Expected: +0.10-0.15 nDCG)
   - Dense retrieval (top 100) → Sparse retrieval (top 50) → Cross-encoder reranking (top 10)
   - **Compliance:** ✅ Pure retrieval pipeline, no generation

2. **Retrieval-Only End-to-End Optimization** ✅ (Expected: +0.05-0.10 nDCG)
   - Optimize retriever using relevance labels (not generation feedback)
   - **Compliance:** ✅ Uses only retrieval signals (relevance labels from training data)

3. **Hybrid Dense-Sparse Fusion** ✅ (Expected: +0.05-0.10 nDCG)
   - Blended RAG approach: Combine dense vectors + sparse encoders
   - **Compliance:** ✅ Pure retrieval fusion, no generation

4. **Large Model Scaling** ✅ (Expected: +0.05-0.08 nDCG)
   - BGE-v2-large or larger models
   - **Compliance:** ✅ Retrieval models only

5. **Advanced Query Rewriting** ✅ (Expected: +0.05-0.08 nDCG)
   - LLM-based query expansion with conversation history
   - **Compliance:** ✅ Query preprocessing only, no answer generation

6. **Conversation State Tracking** ✅ (Expected: +0.03-0.05 nDCG)
   - Explicit state modeling for multi-turn context
   - **Compliance:** ✅ State tracking for retrieval, not generation

7. **Direct nDCG Optimization** ✅ (Expected: +0.03-0.05 nDCG)
   - NeuralNDCG or similar direct metric optimization
   - **Compliance:** ✅ Optimizes retrieval metric directly

8. **Ensemble Methods** ✅ (Expected: +0.02-0.05 nDCG)
   - Combine multiple retrieval models with learned weights
   - **Compliance:** ✅ Ensemble of retrieval models only

**Total Expected Improvement:** +0.38-0.66 nDCG (if all techniques work synergistically)

---

## High-Priority Experiments (Task A Compliant)

### Tier 1: Highest Impact (Expected: +0.15-0.25 nDCG each)

#### 1. **Retrieval-Only End-to-End Optimization** ✅
**Paper:** "OpenRAG: Optimizing RAG End-to-End via In-Context Retrieval Learning" (2025)  
**ArXiv:** [2503.08398](https://arxiv.org/abs/2503.08398)

**Novel Ideas:**
- End-to-end optimization of retriever for in-context relevance
- Adapts to diverse and evolving information needs
- **Reported:** 4.0% improvement over original, 2.1% over SOTA

**Task A Compliant Implementation:**
- ✅ Fine-tune retriever using **relevance labels from training data** (not generation feedback)
- ✅ Use in-context learning signals from retrieval quality (nDCG optimization)
- ✅ Multi-task training: **retrieval objectives only** (contrastive loss, nDCG loss)
- ❌ **REMOVED:** Generation feedback, generation objectives

**Expected nDCG@10:** 0.55-0.60 (from 0.51 baseline)

---

#### 2. **Blended RAG: Hybrid Dense-Sparse Fusion** ✅
**Paper:** "Blended RAG: Improving RAG Accuracy with Semantic Search and Hybrid Query-Based Retrievers" (2024)  
**ArXiv:** [2404.07220](https://arxiv.org/abs/2404.07220)

**Novel Ideas:**
- Combines Dense Vector indexes + Sparse Encoder indexes
- Hybrid query strategies
- **Reported:** New benchmarks on NQ and TREC-COVID, surpasses fine-tuning

**Task A Compliant Implementation:**
- ✅ Dense retrieval (BGE-large) + Sparse retrieval (BM25/ColBERT)
- ✅ Learned fusion weights (neural network)
- ✅ Query-dependent fusion (different weights per query type)
- ✅ **No generation components**

**Expected nDCG@10:** 0.60-0.65

---

#### 3. **Multi-Stage Hierarchical Retrieval Pipeline** ✅
**Papers:** Multiple (2023-2025)

**Novel Ideas:**
- Stage 1: Dense retrieval (top 100-200)
- Stage 2: Sparse retrieval (top 50-100)
- Stage 3: Hybrid fusion (top 50)
- Stage 4: Cross-encoder reranking (top 10-20)
- Stage 5: **Relevance scoring only** (no text generation)

**Task A Compliant Implementation:**
- ✅ Use BGE-large for dense retrieval
- ✅ Use BM25/ColBERT for sparse retrieval
- ✅ Train cross-encoder on MTRAG data (relevance labels only)
- ✅ Learn stage weights with neural network
- ✅ **Final scoring:** Relevance scores only (no text generation)
- ❌ **REMOVED:** LLM-based text generation for scoring

**Expected nDCG@10:** 0.65-0.75

---

### Tier 2: High Impact (Expected: +0.08-0.15 nDCG each)

#### 4. **MetaRAG: Retrieval Quality Self-Assessment** ✅
**Paper:** "Metacognitive Retrieval-Augmented Large Language Models" (2024)  
**Link:** [paperswithcode.com](https://paperswithcode.com/paper/metacognitive-retrieval-augmented-large)

**Novel Ideas:**
- Three-step metacognitive regulation pipeline
- Self-reflection on retrieval quality
- Identifies and corrects inadequacies

**Task A Compliant Implementation:**
- ✅ Add metacognitive layer that **evaluates retrieval quality** (not generation)
- ✅ Self-probing to determine if additional retrieval needed
- ✅ Iterative refinement based on **retrieval quality assessment** (not generation quality)
- ❌ **REMOVED:** Generation quality assessment, text generation

**Expected nDCG@10:** 0.58-0.63

---

#### 5. **Transform Retrieval for Textual Entailment** ✅
**Paper:** "Transform Retrieval for Textual Entailment in RAG" (2025)  
**Link:** [aclanthology.org/2025.naacl-short.50](https://aclanthology.org/2025.naacl-short.50)

**Novel Ideas:**
- Transforms query embeddings to align with semantic entailment
- No need to re-encode document corpus
- Contrastive learning for alignment

**Task A Compliant Implementation:**
- ✅ Train transform model on MTRAG training data (relevance labels)
- ✅ Transform query embeddings before retrieval
- ✅ Optimize for entailment relationship (retrieval-only)
- ✅ **No generation components**

**Expected nDCG@10:** 0.56-0.61

---

#### 6. **ChainRAG: Multi-Hop Retrieval (Retrieval-Only)** ✅
**Paper:** "Mitigating Lost-in-Retrieval Problems in Retrieval Augmented Multi-Hop Question Answering" (2025)  
**Link:** [aclanthology.org/2025.acl-long.1089](https://aclanthology.org/2025.acl-long.1089)

**Novel Ideas:**
- Addresses lost-in-retrieval problems in multi-hop scenarios
- Chain-based retrieval for complex queries
- **Reported:** Consistently outperforms baselines

**Task A Compliant Implementation:**
- ✅ Identify multi-hop queries in MTRAG
- ✅ Chain retrieval: retrieve → **retrieval reasoning** → retrieve again
- ✅ Handle conversation history as multi-hop context
- ✅ **Retrieval reasoning:** Use retrieval signals only (no text generation)
- ❌ **REMOVED:** Text generation for reasoning

**Expected nDCG@10:** 0.57-0.62

---

#### 7. **Advanced LLM Query Rewriting** ✅
**Papers:** Multiple (2023-2025)

**Novel Ideas:**
- LLM-based query expansion using conversation history
- Multi-query generation (generate 3-5 query variants)
- Query decomposition for complex questions

**Task A Compliant Implementation:**
- ✅ Use GPT-4o or similar for **query rewriting only** (not answer generation)
- ✅ Generate multiple query variants per turn
- ✅ Retrieve for each variant, then merge results
- ✅ Weight queries by confidence
- ✅ **Output:** Query variants only (no answers)
- ✅ **Compliance:** Query preprocessing is allowed, text generation is not

**Expected nDCG@10:** 0.55-0.60

---

### Tier 3: Medium Impact (Expected: +0.05-0.10 nDCG each)

#### 8. **BGE-v2-Large with Domain-Specific Fine-Tuning** ✅
**Current:** BGE-base (0.51)  
**Target:** BGE-v2-large with MTRAG fine-tuning

**Task A Compliant Implementation:**
- ✅ Use BGE-v2-large (560M parameters)
- ✅ Fine-tune on MTRAG training data (relevance labels)
- ✅ Domain-specific fine-tuning per corpus
- ✅ Longer training (10+ epochs with early stopping)
- ✅ **Retrieval-only training** (contrastive loss, no generation)

**Expected nDCG@10:** 0.58-0.63

---

#### 9. **Probing-RAG: Selective Document Retrieval** ✅
**Paper:** "Probing-RAG: Self-Probing to Guide Language Models in Selective Document Retrieval" (2024)  
**ArXiv:** [2410.13339](https://arxiv.org/abs/2410.13339)

**Novel Ideas:**
- Uses hidden state representations to determine retrieval necessity
- Pre-trained prober captures internal cognition
- Reduces redundant retrieval steps

**Task A Compliant Implementation:**
- ✅ Train prober to predict if retrieval needed
- ✅ Skip retrieval for queries with high confidence
- ✅ Focus retrieval on uncertain queries
- ✅ **No generation components**

**Expected nDCG@10:** 0.53-0.58

---

#### 10. **MRAG: Modular Retrieval Framework** ✅
**Paper:** "MRAG: A Modular Retrieval Framework for Time-Sensitive Question Answering" (2025)  
**Link:** [aclanthology.org/2025.findings-emnlp.167](https://aclanthology.org/2025.findings-emnlp.167)

**Novel Ideas:**
- Three modules: Question Processing, Retrieval & Summarization, Semantic-Temporal Hybrid Ranking
- Trainless framework (no training required)
- Semantic-temporal hybrid ranking

**Task A Compliant Implementation:**
- ✅ Adapt for multi-turn conversations (not just time-sensitive)
- ✅ Use semantic-temporal ranking for conversation history
- ✅ Modular design allows easy combination with other methods
- ⚠️ **Note:** "Summarization" module should be retrieval summarization (document selection), not text generation
- ❌ **REMOVED:** Text summarization/generation

**Expected nDCG@10:** 0.54-0.59

---

## Combination Strategies (Task A Compliant)

### Strategy A: Multi-Stage + Hybrid + End-to-End ✅
**Components:**
1. Multi-stage pipeline (dense → sparse → reranking)
2. Hybrid dense-sparse fusion
3. Retrieval-only end-to-end optimization

**Expected nDCG@10:** 0.75-0.85

**Task A Compliant Implementation:**
- Stage 1: BGE-v2-large dense retrieval (top 200)
- Stage 2: BM25/ColBERT sparse retrieval (top 100)
- Stage 3: Hybrid fusion with learned weights (top 50)
- Stage 4: Cross-encoder reranking (top 20)
- Stage 5: **Relevance scoring only** (no generation feedback)
- Training: Optimize using **relevance labels** (not generation quality)

---

### Strategy B: Advanced Query Rewriting + Large Models + Ensemble ✅
**Components:**
1. LLM-based multi-query generation (query preprocessing only)
2. BGE-v2-large with domain fine-tuning
3. Ensemble of multiple retrieval models

**Expected nDCG@10:** 0.70-0.80

**Task A Compliant Implementation:**
- Generate 3-5 query variants using GPT-4o (**query rewriting only**)
- Retrieve with BGE-v2-large (per domain fine-tuned)
- Ensemble: BGE-v2-large + ColBERT + BM25
- Learned ensemble weights per query type
- ✅ **No answer generation**

---

### Strategy C: MetaRAG + ChainRAG + Direct Optimization ✅
**Components:**
1. Metacognitive self-reflection (retrieval quality only)
2. Multi-hop chain retrieval (retrieval reasoning only)
3. Direct nDCG optimization (NeuralNDCG)

**Expected nDCG@10:** 0.68-0.78

**Task A Compliant Implementation:**
- Metacognitive layer evaluates **retrieval quality** (not generation)
- Chain retrieval for complex multi-turn queries
- NeuralNDCG loss for direct metric optimization
- Iterative refinement based on **retrieval self-assessment**
- ❌ **REMOVED:** Generation quality assessment

---

### Strategy D: ALL-IN-ONE (Maximum Potential) ✅
**Components:**
1. Multi-stage hierarchical pipeline
2. Hybrid dense-sparse fusion
3. Retrieval-only end-to-end optimization
4. Advanced query rewriting (preprocessing only)
5. Large models (BGE-v2-large)
6. Metacognitive self-reflection (retrieval quality)
7. Direct nDCG optimization
8. Ensemble methods

**Expected nDCG@10:** 0.85-0.95 ⭐ **TARGET ACHIEVABLE**

**Task A Compliant Implementation Priority:**
1. **Phase 1:** Multi-stage + Hybrid (0.65-0.75)
2. **Phase 2:** Add retrieval-only end-to-end optimization (0.70-0.80)
3. **Phase 3:** Add query rewriting + large models (0.75-0.85)
4. **Phase 4:** Add metacognitive + direct optimization (0.80-0.90)
5. **Phase 5:** Fine-tune ensemble weights (0.85-0.95)

---

## Compliance Checklist for Each Experiment

### ✅ Experiment 1: Multi-Stage Hybrid Retrieval Pipeline
- ✅ No text generation
- ✅ Only retrieval and ranking
- ✅ Output: document_id + score
- ✅ Uses only corpus domain (no metadata)

### ✅ Experiment 2: Retrieval-Only End-to-End Optimization
- ✅ Training uses relevance labels (not generation feedback)
- ✅ No generation objectives
- ✅ Optimizes retrieval metrics only

### ✅ Experiment 3: Blended RAG Hybrid Fusion
- ✅ Pure retrieval fusion
- ✅ No generation components

### ✅ Experiment 4: Advanced LLM Query Rewriting
- ✅ LLM used for query preprocessing only
- ✅ No answer generation
- ✅ Output: Query variants only

### ✅ Experiment 5: MetaRAG Self-Reflection
- ✅ Evaluates retrieval quality (not generation)
- ✅ No text generation

### ✅ Experiment 6: Transform Retrieval
- ✅ Pure retrieval optimization
- ✅ No generation components

### ✅ Experiment 7: ChainRAG Multi-Hop
- ✅ Retrieval reasoning only (no text generation)
- ✅ Chain of retrieval operations

### ✅ Experiment 8: BGE-v2-Large Domain-Specific
- ✅ Retrieval model only
- ✅ Training on relevance labels

### ✅ Experiment 9: Probing-RAG
- ✅ Selective retrieval only
- ✅ No generation components

### ✅ Experiment 10: Direct nDCG Optimization
- ✅ Optimizes retrieval metric directly
- ✅ No generation objectives

---

## Key Modifications for Task A Compliance

### ❌ Removed/Modified:
1. **Generation Feedback:** Changed to relevance label feedback
2. **Generation Objectives:** Changed to retrieval-only objectives
3. **LLM Text Generation:** Removed (only query rewriting allowed)
4. **LLM-based Relevance Scoring:** Changed to neural relevance scoring (no text generation)

### ✅ Allowed:
1. **Query Rewriting:** LLMs can preprocess queries (no answers)
2. **Relevance Scoring:** Models can score documents (no text generation)
3. **Retrieval Optimization:** Can optimize using relevance labels
4. **Multi-Stage Retrieval:** All stages are retrieval-only

---

## Implementation Roadmap (Task A Compliant)

### Phase 1: Foundation (Weeks 1-2)
1. ✅ Multi-Stage Hybrid Pipeline (Experiment 1)
2. ✅ BGE-v2-Large Domain-Specific (Experiment 8)
3. ✅ Ensemble of Top Methods (Experiment 9)

**Target:** 0.65-0.70 nDCG@10

---

### Phase 2: Optimization (Weeks 3-4)
4. ✅ Retrieval-Only End-to-End (Experiment 2 - modified)
5. ✅ Blended RAG Hybrid (Experiment 3)
6. ✅ Direct nDCG Optimization (Experiment 10)

**Target:** 0.70-0.75 nDCG@10

---

### Phase 3: Advanced Techniques (Weeks 5-6)
7. ✅ Advanced Query Rewriting (Experiment 4)
8. ✅ MetaRAG Self-Reflection (Experiment 5 - modified)
9. ✅ Transform Retrieval (Experiment 6)

**Target:** 0.75-0.80 nDCG@10

---

### Phase 4: Integration (Weeks 7-8)
10. ✅ ChainRAG Multi-Hop (Experiment 7 - modified)
11. ✅ Combine all techniques (Strategy D - compliant)
12. ✅ Fine-tune ensemble weights

**Target:** 0.85-0.95 nDCG@10 ⭐

---

## Evaluation Format Compliance

### Required Output Format:
```json
{
  "task_id": "unique_id",
  "contexts": [
    {
      "document_id": "822086267_6698-7277-0-579",
      "score": 18.759138
    },
    ...
  ],
  "Collection": "mt-rag-clapnq-elser-512-100-20240503"
}
```

### Evaluation Metrics:
- **Primary:** nDCG@10 (normalized Discounted Cumulative Gain)
- **Secondary:** Recall@1, Recall@3, Recall@5, Recall@10
- **Aggregation:** Weighted average across domains

---

## Success Criteria

- **Minimum Target:** nDCG@10 > 0.75 (50% improvement from baseline)
- **Stretch Target:** nDCG@10 > 0.85 (67% improvement)
- **Ultimate Target:** nDCG@10 > 0.90 (76% improvement) ⭐

---

## Next Steps

1. **Implement Experiment 1** (Multi-Stage Hybrid Pipeline) - Highest priority, fully compliant
2. **Implement Experiment 2** (Retrieval-Only End-to-End) - Second priority, modified for compliance
3. **Implement Experiment 9** (Ensemble) - Third priority, fully compliant
4. **Combine top 3 experiments** - Should reach 0.70-0.75
5. **Add remaining techniques** - Should reach 0.85-0.95

---

**Last Updated:** 2025-12-20  
**Status:** ✅ Task A Compliant - Ready for Implementation  
**Compliance Verified:** All experiments follow MTRAGEval Task A (Retrieval Only) rules

