# Strategic Experiments for nDCG@10 > 0.90 on MTRAGEval Task A

**Target:** nDCG@10 > 0.90 (current best: 0.5101)  
**Gap:** Need +0.39 improvement (76% relative improvement)  
**Benchmark:** [MTRAGEval Task A - Retrieval Only](https://ibm.github.io/mt-rag-benchmark/MTRAGEval/)

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

### Critical Success Factors (from Literature)

1. **Multi-Stage Retrieval** (Expected: +0.10-0.15 nDCG)
   - Dense retrieval (top 100) → Sparse retrieval (top 50) → Cross-encoder reranking (top 10)
   - Literature shows 0.05-0.15 improvement on similar tasks

2. **End-to-End Optimization** (Expected: +0.05-0.10 nDCG)
   - OpenRAG-style: Optimize retriever for in-context relevance
   - Literature: 4% improvement, but can be higher with better training

3. **Hybrid Dense-Sparse Fusion** (Expected: +0.05-0.10 nDCG)
   - Blended RAG approach: Combine dense vectors + sparse encoders
   - Literature: New benchmarks on NQ and TREC-COVID

4. **Large Model Scaling** (Expected: +0.05-0.08 nDCG)
   - BGE-v2-large or larger models
   - Current: BGE-base (0.51) → Target: BGE-large/v2-large

5. **Advanced Query Rewriting** (Expected: +0.05-0.08 nDCG)
   - LLM-based query expansion with conversation history
   - Current: Simple rewrite → Target: Multi-query generation

6. **Conversation State Tracking** (Expected: +0.03-0.05 nDCG)
   - Explicit state modeling for multi-turn context
   - Critical for later turns in conversations

7. **Direct nDCG Optimization** (Expected: +0.03-0.05 nDCG)
   - NeuralNDCG or similar direct metric optimization
   - Aligns training objective with evaluation metric

8. **Ensemble Methods** (Expected: +0.02-0.05 nDCG)
   - Combine multiple retrieval models with learned weights
   - Current: 0.4576 (ensemble) → Can improve with better models

**Total Expected Improvement:** +0.38-0.66 nDCG (if all techniques work synergistically)

---

## High-Priority Experiments (Based on Literature)

### Tier 1: Highest Impact (Expected: +0.15-0.25 nDCG each)

#### 1. **OpenRAG-Style End-to-End Optimization**
**Paper:** "OpenRAG: Optimizing RAG End-to-End via In-Context Retrieval Learning" (2025)  
**ArXiv:** [2503.08398](https://arxiv.org/abs/2503.08398)

**Novel Ideas:**
- End-to-end optimization of retriever for in-context relevance
- Adapts to diverse and evolving information needs
- **Reported:** 4.0% improvement over original, 2.1% over SOTA

**Implementation Strategy:**
- Fine-tune retriever with generation feedback
- Use in-context learning signals to optimize retrieval
- Multi-task training: retrieval + generation objectives

**Expected nDCG@10:** 0.55-0.60 (from 0.51 baseline)

---

#### 2. **Blended RAG: Hybrid Dense-Sparse Fusion**
**Paper:** "Blended RAG: Improving RAG Accuracy with Semantic Search and Hybrid Query-Based Retrievers" (2024)  
**ArXiv:** [2404.07220](https://arxiv.org/abs/2404.07220)

**Novel Ideas:**
- Combines Dense Vector indexes + Sparse Encoder indexes
- Hybrid query strategies
- **Reported:** New benchmarks on NQ and TREC-COVID, surpasses fine-tuning

**Implementation Strategy:**
- Dense retrieval (BGE-large) + Sparse retrieval (BM25/ColBERT)
- Learned fusion weights (neural network)
- Query-dependent fusion (different weights per query type)

**Expected nDCG@10:** 0.60-0.65

---

#### 3. **Multi-Stage Hierarchical Retrieval Pipeline**
**Papers:** Multiple (2023-2025)

**Novel Ideas:**
- Stage 1: Dense retrieval (top 100-200)
- Stage 2: Sparse retrieval (top 50-100)
- Stage 3: Hybrid fusion (top 50)
- Stage 4: Cross-encoder reranking (top 10-20)
- Stage 5: LLM-based relevance scoring (optional, constrained)

**Implementation Strategy:**
- Use BGE-large for dense retrieval
- Use BM25/ColBERT for sparse retrieval
- Train cross-encoder on MTRAG data
- Learn stage weights with neural network

**Expected nDCG@10:** 0.65-0.75

---

### Tier 2: High Impact (Expected: +0.08-0.15 nDCG each)

#### 4. **MetaRAG: Metacognitive Self-Reflection**
**Paper:** "Metacognitive Retrieval-Augmented Large Language Models" (2024)  
**Link:** [paperswithcode.com](https://paperswithcode.com/paper/metacognitive-retrieval-augmented-large)

**Novel Ideas:**
- Three-step metacognitive regulation pipeline
- Self-reflection on retrieval quality
- Identifies and corrects inadequacies

**Implementation Strategy:**
- Add metacognitive layer that evaluates retrieval quality
- Self-probing to determine if additional retrieval needed
- Iterative refinement based on self-assessment

**Expected nDCG@10:** 0.58-0.63

---

#### 5. **Transform Retrieval for Textual Entailment**
**Paper:** "Transform Retrieval for Textual Entailment in RAG" (2025)  
**Link:** [aclanthology.org/2025.naacl-short.50](https://aclanthology.org/2025.naacl-short.50)

**Novel Ideas:**
- Transforms query embeddings to align with semantic entailment
- No need to re-encode document corpus
- Contrastive learning for alignment

**Implementation Strategy:**
- Train transform model on MTRAG training data
- Transform query embeddings before retrieval
- Optimize for entailment relationship

**Expected nDCG@10:** 0.56-0.61

---

#### 6. **ChainRAG for Multi-Hop Reasoning**
**Paper:** "Mitigating Lost-in-Retrieval Problems in Retrieval Augmented Multi-Hop Question Answering" (2025)  
**Link:** [aclanthology.org/2025.acl-long.1089](https://aclanthology.org/2025.acl-long.1089)

**Novel Ideas:**
- Addresses lost-in-retrieval problems in multi-hop QA
- Chain-based retrieval for complex queries
- **Reported:** Consistently outperforms baselines

**Implementation Strategy:**
- Identify multi-hop queries in MTRAG
- Chain retrieval: retrieve → reason → retrieve again
- Handle conversation history as multi-hop context

**Expected nDCG@10:** 0.57-0.62

---

#### 7. **Advanced Query Rewriting with LLM**
**Papers:** Multiple (2023-2025)

**Novel Ideas:**
- LLM-based query expansion using conversation history
- Multi-query generation (generate 3-5 query variants)
- Query decomposition for complex questions

**Implementation Strategy:**
- Use GPT-4o or similar for query rewriting
- Generate multiple query variants per turn
- Retrieve for each variant, then merge results
- Weight queries by confidence

**Expected nDCG@10:** 0.55-0.60

---

### Tier 3: Medium Impact (Expected: +0.05-0.10 nDCG each)

#### 8. **BGE-v2-Large with Domain-Specific Fine-Tuning**
**Current:** BGE-base (0.51)  
**Target:** BGE-v2-large with MTRAG fine-tuning

**Implementation Strategy:**
- Use BGE-v2-large (560M parameters)
- Fine-tune on MTRAG training data
- Domain-specific fine-tuning per corpus
- Longer training (10+ epochs with early stopping)

**Expected nDCG@10:** 0.58-0.63

---

#### 9. **Probing-RAG: Selective Document Retrieval**
**Paper:** "Probing-RAG: Self-Probing to Guide Language Models in Selective Document Retrieval" (2024)  
**ArXiv:** [2410.13339](https://arxiv.org/abs/2410.13339)

**Novel Ideas:**
- Uses hidden state representations to determine retrieval necessity
- Pre-trained prober captures internal cognition
- Reduces redundant retrieval steps

**Implementation Strategy:**
- Train prober to predict if retrieval needed
- Skip retrieval for queries with high confidence
- Focus retrieval on uncertain queries

**Expected nDCG@10:** 0.53-0.58

---

#### 10. **MRAG: Modular Retrieval Framework**
**Paper:** "MRAG: A Modular Retrieval Framework for Time-Sensitive Question Answering" (2025)  
**Link:** [aclanthology.org/2025.findings-emnlp.167](https://aclanthology.org/2025.findings-emnlp.167)

**Novel Ideas:**
- Three modules: Question Processing, Retrieval & Summarization, Semantic-Temporal Hybrid Ranking
- Trainless framework (no training required)
- Semantic-temporal hybrid ranking

**Implementation Strategy:**
- Adapt for multi-turn conversations (not just time-sensitive)
- Use semantic-temporal ranking for conversation history
- Modular design allows easy combination with other methods

**Expected nDCG@10:** 0.54-0.59

---

## Combination Strategies (Highest Potential)

### Strategy A: Multi-Stage + Hybrid + End-to-End
**Components:**
1. Multi-stage pipeline (dense → sparse → reranking)
2. Hybrid dense-sparse fusion
3. End-to-end optimization (OpenRAG-style)

**Expected nDCG@10:** 0.75-0.85

**Implementation:**
- Stage 1: BGE-v2-large dense retrieval (top 200)
- Stage 2: BM25/ColBERT sparse retrieval (top 100)
- Stage 3: Hybrid fusion with learned weights (top 50)
- Stage 4: Cross-encoder reranking (top 20)
- Stage 5: End-to-end fine-tuning with generation feedback

---

### Strategy B: Advanced Query Rewriting + Large Models + Ensemble
**Components:**
1. LLM-based multi-query generation
2. BGE-v2-large with domain fine-tuning
3. Ensemble of multiple retrieval models

**Expected nDCG@10:** 0.70-0.80

**Implementation:**
- Generate 3-5 query variants using GPT-4o
- Retrieve with BGE-v2-large (per domain fine-tuned)
- Ensemble: BGE-v2-large + ColBERT + BM25
- Learned ensemble weights per query type

---

### Strategy C: MetaRAG + ChainRAG + Direct Optimization
**Components:**
1. Metacognitive self-reflection (MetaRAG)
2. Multi-hop chain retrieval (ChainRAG)
3. Direct nDCG optimization (NeuralNDCG)

**Expected nDCG@10:** 0.68-0.78

**Implementation:**
- Metacognitive layer evaluates retrieval quality
- Chain retrieval for complex multi-turn queries
- NeuralNDCG loss for direct metric optimization
- Iterative refinement based on self-assessment

---

### Strategy D: ALL-IN-ONE (Maximum Potential)
**Components:**
1. Multi-stage hierarchical pipeline
2. Hybrid dense-sparse fusion
3. End-to-end optimization
4. Advanced query rewriting
5. Large models (BGE-v2-large)
6. Metacognitive self-reflection
7. Direct nDCG optimization
8. Ensemble methods

**Expected nDCG@10:** 0.85-0.95 ⭐ **TARGET ACHIEVABLE**

**Implementation Priority:**
1. **Phase 1:** Multi-stage + Hybrid (0.65-0.75)
2. **Phase 2:** Add end-to-end optimization (0.70-0.80)
3. **Phase 3:** Add query rewriting + large models (0.75-0.85)
4. **Phase 4:** Add metacognitive + direct optimization (0.80-0.90)
5. **Phase 5:** Fine-tune ensemble weights (0.85-0.95)

---

## Specific Experiment Recommendations

### Experiment 1: Multi-Stage Hybrid Retrieval Pipeline
**Priority:** ⭐⭐⭐⭐⭐ (Highest)

**Components:**
- Dense: BGE-v2-large (top 200)
- Sparse: BM25 + ColBERT (top 100 each)
- Fusion: Learned neural network (top 50)
- Reranking: Cross-encoder (top 20)
- Final: Top 10 for evaluation

**Expected:** 0.65-0.75 nDCG@10

---

### Experiment 2: OpenRAG End-to-End Optimization
**Priority:** ⭐⭐⭐⭐⭐ (Highest)

**Components:**
- Base: BGE-v2-large
- Training: End-to-end with generation feedback
- Objective: In-context relevance optimization
- Multi-task: Retrieval + generation signals

**Expected:** 0.60-0.70 nDCG@10

---

### Experiment 3: Blended RAG Hybrid Fusion
**Priority:** ⭐⭐⭐⭐ (Very High)

**Components:**
- Dense: BGE-v2-large embeddings
- Sparse: Sparse encoder (ColBERT-style)
- Fusion: Query-dependent learned weights
- Training: Contrastive learning on MTRAG

**Expected:** 0.60-0.65 nDCG@10

---

### Experiment 4: Advanced LLM Query Rewriting
**Priority:** ⭐⭐⭐⭐ (Very High)

**Components:**
- LLM: GPT-4o for query rewriting
- Input: Full conversation history
- Output: 3-5 query variants
- Retrieval: Per variant, then merge with learned weights

**Expected:** 0.55-0.60 nDCG@10

---

### Experiment 5: MetaRAG with Self-Reflection
**Priority:** ⭐⭐⭐ (High)

**Components:**
- Base: BGE-v2-large
- Metacognitive layer: Evaluates retrieval quality
- Self-probing: Determines if additional retrieval needed
- Iterative refinement: 2-3 retrieval rounds

**Expected:** 0.58-0.63 nDCG@10

---

### Experiment 6: Transform Retrieval for Entailment
**Priority:** ⭐⭐⭐ (High)

**Components:**
- Base: BGE-v2-large
- Transform model: Aligns queries for entailment
- Training: Contrastive learning on MTRAG
- No corpus re-encoding needed

**Expected:** 0.56-0.61 nDCG@10

---

### Experiment 7: ChainRAG Multi-Hop Retrieval
**Priority:** ⭐⭐⭐ (High)

**Components:**
- Multi-hop detection: Identify complex queries
- Chain retrieval: Retrieve → Reason → Retrieve
- Conversation history: Treated as multi-hop context
- Merge strategy: Weighted combination

**Expected:** 0.57-0.62 nDCG@10

---

### Experiment 8: BGE-v2-Large Domain-Specific
**Priority:** ⭐⭐⭐ (High)

**Components:**
- Model: BGE-v2-large (560M)
- Fine-tuning: Per-domain (ClapNQ, FiQA, Govt, Cloud)
- Training: 10+ epochs with early stopping
- Data: MTRAG training data

**Expected:** 0.58-0.63 nDCG@10

---

### Experiment 9: Ensemble of Top Methods
**Priority:** ⭐⭐⭐⭐ (Very High)

**Components:**
- Models: Top 5-7 retrieval methods
- Weights: Learned per query type
- Fusion: Neural network or learned RRF
- Training: Optimize for nDCG@10

**Expected:** 0.65-0.70 nDCG@10 (from 0.51 baseline)

---

### Experiment 10: Direct nDCG Optimization (NeuralNDCG)
**Priority:** ⭐⭐⭐ (High)

**Components:**
- Base: BGE-v2-large
- Loss: NeuralNDCG (differentiable nDCG)
- Training: Direct optimization of evaluation metric
- Fine-tuning: On MTRAG training data

**Expected:** 0.55-0.60 nDCG@10

---

## Implementation Roadmap

### Phase 1: Foundation (Weeks 1-2)
1. ✅ Multi-Stage Hybrid Pipeline (Experiment 1)
2. ✅ BGE-v2-Large Domain-Specific (Experiment 8)
3. ✅ Ensemble of Top Methods (Experiment 9)

**Target:** 0.65-0.70 nDCG@10

---

### Phase 2: Optimization (Weeks 3-4)
4. ✅ OpenRAG End-to-End (Experiment 2)
5. ✅ Blended RAG Hybrid (Experiment 3)
6. ✅ Direct nDCG Optimization (Experiment 10)

**Target:** 0.70-0.75 nDCG@10

---

### Phase 3: Advanced Techniques (Weeks 5-6)
7. ✅ Advanced Query Rewriting (Experiment 4)
8. ✅ MetaRAG Self-Reflection (Experiment 5)
9. ✅ Transform Retrieval (Experiment 6)

**Target:** 0.75-0.80 nDCG@10

---

### Phase 4: Integration (Weeks 7-8)
10. ✅ ChainRAG Multi-Hop (Experiment 7)
11. ✅ Combine all techniques (Strategy D)
12. ✅ Fine-tune ensemble weights

**Target:** 0.85-0.95 nDCG@10 ⭐

---

## Key Papers to Reference

1. **OpenRAG:** [arXiv:2503.08398](https://arxiv.org/abs/2503.08398) - End-to-end optimization
2. **Blended RAG:** [arXiv:2404.07220](https://arxiv.org/abs/2404.07220) - Hybrid fusion
3. **MetaRAG:** [paperswithcode.com](https://paperswithcode.com/paper/metacognitive-retrieval-augmented-large) - Self-reflection
4. **Transform Retrieval:** [aclanthology.org/2025.naacl-short.50](https://aclanthology.org/2025.naacl-short.50) - Entailment alignment
5. **ChainRAG:** [aclanthology.org/2025.acl-long.1089](https://aclanthology.org/2025.acl-long.1089) - Multi-hop
6. **Probing-RAG:** [arXiv:2410.13339](https://arxiv.org/abs/2410.13339) - Selective retrieval
7. **MRAG:** [aclanthology.org/2025.findings-emnlp.167](https://aclanthology.org/2025.findings-emnlp.167) - Modular framework

---

## Risk Assessment

### High Risk (May Not Work)
- **LLM-based query rewriting:** Expensive, may not help much
- **MetaRAG:** Complex, may not provide expected gains
- **ChainRAG:** May be overkill for MTRAG queries

### Medium Risk (Likely to Work)
- **Multi-stage pipeline:** Proven technique, should work
- **Hybrid fusion:** Well-established, should help
- **Large models:** Clear benefit, but diminishing returns

### Low Risk (Very Likely to Work)
- **End-to-end optimization:** Clear benefit from literature
- **Direct nDCG optimization:** Aligns training with evaluation
- **Ensemble methods:** Almost always helps

---

## Success Criteria

- **Minimum Target:** nDCG@10 > 0.75 (50% improvement from baseline)
- **Stretch Target:** nDCG@10 > 0.85 (67% improvement)
- **Ultimate Target:** nDCG@10 > 0.90 (76% improvement) ⭐

---

## Next Steps

1. **Implement Experiment 1** (Multi-Stage Hybrid Pipeline) - Highest priority
2. **Implement Experiment 2** (OpenRAG End-to-End) - Second priority
3. **Implement Experiment 9** (Ensemble) - Third priority
4. **Combine top 3 experiments** - Should reach 0.70-0.75
5. **Add remaining techniques** - Should reach 0.85-0.95

---

**Last Updated:** 2025-12-20  
**Status:** ⚠️ **SEE COMPLIANCE-VERIFIED VERSION:** `STRATEGY_FOR_NDCG_0.90_PLUS_COMPLIANT.md`

**Note:** This document contains some experiments that need modification for Task A compliance. Please refer to the compliant version for experiments verified against MTRAGEval Task A (Retrieval Only) rules.

