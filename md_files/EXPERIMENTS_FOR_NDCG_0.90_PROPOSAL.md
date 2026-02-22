# Experiments to Achieve nDCG@10 > 0.90

**Current Best:** 0.4442 (tier1_adversarial_curriculum)  
**Target:** > 0.90  
**Gap:** Need to improve by ~2x

## Strategy Overview

To reach nDCG@10 > 0.90, we need to combine multiple high-impact techniques:

1. **Multi-Stage Retrieval with Cross-Encoder Reranking** (Expected: +0.15-0.25)
2. **LLM-Powered Query Expansion & Reranking** (Expected: +0.10-0.15)
3. **Ensemble of Specialized Models** (Expected: +0.10-0.15)
4. **ColBERT-Style Multi-Vector Retrieval** (Expected: +0.05-0.10)
5. **Generative Relevance Feedback** (Expected: +0.05-0.10)
6. **Large Model Scaling (BGE-v2-large)** (Expected: +0.05-0.10)

---

## Priority 1: High-Impact Experiments (Expected nDCG@10: 0.85-0.95)

### 1. **Multi-Stage Retrieval with LLM Reranking**
**Expected:** 0.85-0.92 nDCG@10

**Approach:**
- Stage 1: Dense retrieval (BGE-base) → Top 100 candidates
- Stage 2: Sparse retrieval (BM25) → Top 100 candidates
- Stage 3: Hybrid fusion (Reciprocal Rank Fusion) → Top 50
- Stage 4: **LLM-based reranking** (GPT-4o-mini or similar) → Final top 10

**Key Innovation:**
- Use LLM to score query-document pairs with detailed reasoning
- LLM prompt: "Given this query and document, rate relevance 0-10 with explanation"
- Fine-tune reranking weights based on LLM scores

**Implementation:**
```python
# Pseudo-code structure
def multi_stage_llm_reranking(query, corpus):
    # Stage 1: Dense retrieval
    dense_results = dense_retriever.retrieve(query, top_k=100)
    
    # Stage 2: Sparse retrieval
    sparse_results = bm25_retriever.retrieve(query, top_k=100)
    
    # Stage 3: Hybrid fusion
    fused_results = reciprocal_rank_fusion(dense_results, sparse_results, top_k=50)
    
    # Stage 4: LLM reranking
    llm_scores = []
    for doc_id, doc_text in fused_results[:50]:
        score = llm_reranker.score(query, doc_text)  # LLM-based scoring
        llm_scores.append((doc_id, score))
    
    return sorted(llm_scores, key=lambda x: x[1], reverse=True)[:10]
```

**Files to create:**
- `train_multistage_llm_reranking.py`
- `train_multistage_llm_reranking_tier1.py`

---

### 2. **ColBERT-Style Multi-Vector Retrieval with Reranking**
**Expected:** 0.80-0.88 nDCG@10

**Approach:**
- Use token-level embeddings (ColBERT-style)
- MaxSim scoring: max similarity between query tokens and document tokens
- Add cross-encoder reranking on top 20 candidates

**Key Innovation:**
- Token-level matching captures fine-grained relevance
- More expressive than single-vector embeddings
- Cross-encoder provides final precision boost

**Implementation:**
```python
def colbert_retrieval(query, corpus):
    # Token-level embeddings
    query_tokens = model.encode_queries(query, return_tokens=True)
    doc_tokens = model.encode_documents(corpus, return_tokens=True)
    
    # MaxSim scoring
    scores = maxsim_scoring(query_tokens, doc_tokens)
    top_20 = get_top_k(scores, k=20)
    
    # Cross-encoder reranking
    reranked = cross_encoder_rerank(query, top_20)
    return reranked[:10]
```

**Files to create:**
- `train_colbert_multivector.py`
- `train_colbert_multivector_tier1.py`

---

### 3. **Ensemble with Meta-Learner Fusion**
**Expected:** 0.82-0.90 nDCG@10

**Approach:**
- Train 5 specialized models:
  1. Dense retriever (BGE-base fine-tuned)
  2. Sparse retriever (BM25 + learned weights)
  3. Cross-encoder reranker
  4. Multi-vector retriever (ColBERT-style)
  5. LLM-based reranker
- Use meta-learner (small neural network) to learn optimal fusion weights
- Meta-learner takes query features + individual model scores → final score

**Key Innovation:**
- Adaptive fusion based on query characteristics
- Learns when to trust which model
- Can handle different query types optimally

**Implementation:**
```python
class MetaLearnerFusion(nn.Module):
    def __init__(self):
        self.fusion_net = nn.Sequential(
            nn.Linear(5 + query_features_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 5),  # 5 model weights
            nn.Softmax(dim=1)
        )
    
    def forward(self, query_features, model_scores):
        # model_scores: [batch, 5] (5 model scores)
        # query_features: [batch, query_features_dim]
        combined = torch.cat([model_scores, query_features], dim=1)
        weights = self.fusion_net(combined)
        final_score = torch.sum(weights * model_scores, dim=1)
        return final_score
```

**Files to create:**
- `train_ensemble_meta_learner.py`
- `train_ensemble_meta_learner_tier1.py`

---

## Priority 2: Medium-Impact Experiments (Expected nDCG@10: 0.75-0.85)

### 4. **Generative Query Expansion with Retrieval**
**Expected:** 0.75-0.82 nDCG@10

**Approach:**
- Use LLM to generate 3-5 query variations
- Retrieve with each variation
- Combine results with learned fusion
- Rerank final candidates

**Key Innovation:**
- Captures multiple query intents
- Handles ambiguity in multi-turn conversations
- Generates domain-specific expansions

**Implementation:**
```python
def generative_query_expansion(query, conversation_history):
    # Generate query variations
    prompt = f"Given this conversation history and query, generate 3 alternative phrasings:\n{conversation_history}\nQuery: {query}"
    variations = llm.generate(prompt, n=3)
    
    # Retrieve with each variation
    all_results = []
    for var in [query] + variations:
        results = retriever.retrieve(var, top_k=20)
        all_results.extend(results)
    
    # Deduplicate and rerank
    unique_results = deduplicate(all_results)
    final = reranker.rerank(query, unique_results, top_k=10)
    return final
```

**Files to create:**
- `train_generative_query_expansion.py`
- `train_generative_query_expansion_tier1.py`

---

### 5. **BGE-v2-Large with Asymmetric Encoding**
**Expected:** 0.70-0.78 nDCG@10

**Approach:**
- Use BGE-v2-large (560M parameters) instead of BGE-base
- Asymmetric encoding: different prompts for queries vs documents
- Fine-tune on multi-turn conversation data
- Longer training (5-7 epochs)

**Key Innovation:**
- Larger model capacity
- Better semantic understanding
- Asymmetric prompts optimize for retrieval task

**Implementation:**
```python
# Query encoding
query_prompt = "Represent this query for retrieving relevant documents: "
query_emb = model.encode(query_prompt + query)

# Document encoding
doc_prompt = "Represent this document for being retrieved: "
doc_emb = model.encode(doc_prompt + document)
```

**Files to create:**
- `train_bge_v2_large_asymmetric.py`
- `train_bge_v2_large_asymmetric_tier1.py`

---

### 6. **Iterative Retrieval with Generative Feedback**
**Expected:** 0.72-0.80 nDCG@10

**Approach:**
- Initial retrieval → Top 20
- Use LLM to generate relevance feedback based on top results
- Expand query with feedback terms
- Re-retrieve with expanded query
- Final reranking

**Key Innovation:**
- Iterative refinement
- Uses retrieved content to improve query
- Mimics human relevance feedback

**Implementation:**
```python
def iterative_retrieval_with_feedback(query, corpus):
    # Initial retrieval
    initial_results = retriever.retrieve(query, top_k=20)
    
    # Generate feedback
    feedback_prompt = f"Based on these retrieved documents, what terms should be added to improve retrieval?\nQuery: {query}\nDocuments: {initial_results[:5]}"
    feedback = llm.generate(feedback_prompt)
    
    # Expand query
    expanded_query = f"{query} {feedback}"
    
    # Re-retrieve
    final_results = retriever.retrieve(expanded_query, top_k=10)
    return final_results
```

**Files to create:**
- `train_iterative_generative_feedback.py`
- `train_iterative_generative_feedback_tier1.py`

---

## Priority 3: Supporting Experiments (Expected nDCG@10: 0.65-0.75)

### 7. **Cross-Encoder Fine-Tuned Reranker**
**Expected:** 0.65-0.72 nDCG@10

**Approach:**
- Fine-tune cross-encoder (e.g., ms-marco-MiniLM) on MT-RAG data
- Use as final reranking stage
- Train with hard negatives from dense retrieval

**Files to create:**
- `train_cross_encoder_reranker.py`
- `train_cross_encoder_reranker_tier1.py`

---

### 8. **Learned Sparse-Dense Hybrid Fusion**
**Expected:** 0.68-0.75 nDCG@10

**Approach:**
- Train neural network to learn optimal fusion weights
- Input: Dense scores, sparse scores, query features
- Output: Combined score
- End-to-end training

**Files to create:**
- `train_learned_hybrid_fusion.py`
- `train_learned_hybrid_fusion_tier1.py`

---

## Implementation Priority

### Phase 1 (Immediate - Highest Impact):
1. ✅ Multi-Stage Retrieval with LLM Reranking
2. ✅ ColBERT-Style Multi-Vector Retrieval
3. ✅ Ensemble with Meta-Learner Fusion

### Phase 2 (Next - Medium Impact):
4. ✅ Generative Query Expansion
5. ✅ BGE-v2-Large with Asymmetric Encoding
6. ✅ Iterative Retrieval with Generative Feedback

### Phase 3 (Supporting):
7. ✅ Cross-Encoder Fine-Tuned Reranker
8. ✅ Learned Sparse-Dense Hybrid Fusion

---

## Expected Combined Performance

If we combine the top 3 approaches:
- **Multi-Stage LLM Reranking:** 0.85-0.92
- **ColBERT Multi-Vector:** 0.80-0.88
- **Ensemble Meta-Learner:** 0.82-0.90

**Combined Ensemble:** Could reach **0.90-0.95** nDCG@10

---

## Key Technical Considerations

1. **LLM API Costs:** Use efficient models (GPT-4o-mini, Claude Haiku) for reranking
2. **Latency:** Cache LLM responses, batch processing
3. **Memory:** Use gradient checkpointing, mixed precision
4. **Training Data:** Use all 4 domains, longer training
5. **Evaluation:** Ensure using trained models, not base models

---

## Next Steps

1. Implement Priority 1 experiments (3 experiments)
2. Run and evaluate
3. If < 0.90, implement Priority 2 experiments
4. Combine best approaches into final ensemble
5. Fine-tune ensemble weights on validation set

