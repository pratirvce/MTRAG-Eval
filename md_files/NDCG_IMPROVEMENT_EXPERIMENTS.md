# Experiments to Improve nDCG Score for Task A Retrieval

**Current Best nDCG@10**: **0.4539** (phase5_ensemble_domain_specific)  
**Target nDCG@10**: **0.50+** (10% improvement)  
**Baseline nDCG@10**: 0.3000 (Paper baseline)

---

## 🎯 Why nDCG Matters for Task A

nDCG (Normalized Discounted Cumulative Gain) is the **official ranking metric** for MTRAGEval Task A:
- **Penalizes lower-ranked relevant documents** (position matters)
- **Rewards putting highly relevant documents at the top**
- **Is the metric used for leaderboard ranking**

Unlike Recall, nDCG specifically measures **ranking quality**, not just retrieval coverage.

---

## 📊 Current Status

| Metric | Baseline | Current Best | Improvement | Target |
|--------|----------|--------------|-------------|--------|
| **nDCG@10** | 0.3000 | **0.4539** | **+51.3%** | **0.50+** |
| Recall@10 | 0.3800 | 0.5441 | +43.2% | - |

**Gap to Target**: Need +0.0461 nDCG@10 (10.2% improvement from current best)

---

## 🚀 High-Priority Experiments (Recommended First)

### 1. ⭐⭐⭐⭐⭐ **Fine-Tuned Cross-Encoder Reranking** (HIGHEST POTENTIAL)

**Why This Improves nDCG**:
- Cross-encoders excel at **ranking quality** (precise relevance scoring)
- Current reranking used pre-trained models (underperformed)
- Fine-tuning on domain data should significantly improve precision
- **Directly targets nDCG improvement** (better top-K ranking)

**Implementation**:
- Fine-tune `cross-encoder/ms-marco-MiniLM-L-12-v2` on domain-specific data
- Use (query, positive, negative) triplets from training data
- Rerank top 50-100 results from best dense model
- Test different cross-encoder architectures (L-6, L-12, L-24)

**Expected Improvement**: **+5-12% nDCG@10**
- If successful: **0.48-0.51 nDCG@10** ✅

**Effort**: Medium (similar to bi-encoder training)  
**Risk**: Low (proven technique, just needs proper implementation)

**Experiments to Run**:
- `phase6_cross_encoder_finetuned_clapnq`
- `phase6_cross_encoder_finetuned_govt`
- `phase6_cross_encoder_finetuned_multi`
- `phase6_cross_encoder_finetuned_ensemble` (rerank ensemble results)

---

### 2. ⭐⭐⭐⭐⭐ **LLM-Based Query Expansion** (HIGH IMPACT)

**Why This Improves nDCG**:
- LLMs can generate **better query variations** than simple synonym expansion
- More context-aware expansions → better matching
- Can generate domain-specific terminology
- **Query expansion on ClapNQ already achieved 0.4981 nDCG@10** on that domain

**Implementation**:
- Use GPT-4/Claude or local LLM (Llama, Mistral) for query expansion
- Prompt: "Expand this query for better document retrieval: [query]"
- Generate multiple query variations per original query
- Combine results from all variations (RRF or weighted)

**Expected Improvement**: **+3-8% nDCG@10**
- If successful: **0.47-0.49 nDCG@10**

**Effort**: Medium (need API access or local LLM setup)  
**Risk**: Low (query expansion already proven effective)

**Experiments to Run**:
- `phase6_llm_query_expansion_gpt4`
- `phase6_llm_query_expansion_claude`
- `phase6_llm_query_expansion_local` (using local LLM)
- `phase6_llm_query_expansion_multi_variations` (multiple expansions per query)

---

### 3. ⭐⭐⭐⭐ **Multi-Stage Retrieval Pipeline** (GOOD IMPACT)

**Why This Improves nDCG**:
- **Stage 1**: Fast retrieval (top 100) with best dense model
- **Stage 2**: Rerank (top 50) with fine-tuned cross-encoder
- **Stage 3**: Final rerank (top 20) with larger model or ensemble
- Each stage refines ranking quality → better nDCG

**Implementation**:
- Stage 1: Use `phase5_ensemble_domain_specific` (retrieve top 100)
- Stage 2: Fine-tuned cross-encoder (rerank to top 50)
- Stage 3: Optional - larger cross-encoder or LLM-based reranking

**Expected Improvement**: **+4-10% nDCG@10**
- If successful: **0.47-0.50 nDCG@10**

**Effort**: Medium (need to chain models efficiently)  
**Risk**: Low (combines proven techniques)

**Experiments to Run**:
- `phase6_multistage_2stage` (dense + cross-encoder)
- `phase6_multistage_3stage` (dense + cross-encoder + LLM)
- `phase6_multistage_domain_specific` (different stages per domain)

---

### 4. ⭐⭐⭐⭐ **Improved Hard Negative Mining** (HIGH POTENTIAL IF FIXED)

**Why This Improves nDCG**:
- **Hard negatives improve ranking discrimination**
- Model learns to distinguish subtle differences
- Better at ranking relevant documents higher
- **Directly targets nDCG improvement** (better top-K ranking)

**Current Status**: Previous attempts failed (0.1444-0.1456 nDCG@10)

**Improvements Needed**:
- **Softer Hard Negatives**: Not the hardest, but moderately difficult
- **Curriculum Learning**: Start with easy negatives, gradually increase difficulty
- **In-Batch Hard Negatives**: Mine within batch (more efficient, proven to work)
- **Better Loss Function**: Use `MultipleNegativesRankingLoss` with hard negatives

**Expected Improvement**: **+5-12% nDCG@10** (if done correctly)
- If successful: **0.48-0.51 nDCG@10**

**Effort**: Medium-High (need to debug and refine)  
**Risk**: Moderate (previous attempts failed, but technique is proven)

**Experiments to Run**:
- `phase6_hard_negatives_inbatch` (in-batch mining)
- `phase6_hard_negatives_curriculum` (curriculum learning)
- `phase6_hard_negatives_softer` (moderate difficulty negatives)
- `phase6_hard_negatives_domain_specific` (domain-specific mining)

---

### 5. ⭐⭐⭐⭐ **Domain-Specific Query Expansion Strategies** (GOOD IMPACT)

**Why This Improves nDCG**:
- Different domains need different expansion strategies
- **ClapNQ**: Entity linking, fact expansion
- **Govt**: Legal/bureaucratic terminology
- **FiQA**: Financial terms, market context
- **Cloud**: Technical/API terminology

**Implementation**:
- Domain-specific synonym dictionaries
- Domain-specific query paraphrasing
- Use domain knowledge bases (WordNet, domain-specific ontologies)
- Combine with LLM-based expansion

**Expected Improvement**: **+2-6% nDCG@10**
- If successful: **0.46-0.48 nDCG@10**

**Effort**: Medium (need domain expertise/resources)  
**Risk**: Low (query expansion already proven)

**Experiments to Run**:
- `phase6_query_expansion_domain_specific_clapnq`
- `phase6_query_expansion_domain_specific_govt`
- `phase6_query_expansion_domain_specific_fiqa`
- `phase6_query_expansion_domain_specific_cloud`

---

## 🔬 Medium-Priority Experiments

### 6. ⭐⭐⭐ **Learning to Rank (LTR) Loss Functions** (GOOD IMPACT)

**Why This Improves nDCG**:
- **Listwise losses** directly optimize ranking metrics
- **Pairwise losses** improve relative ranking
- Better than contrastive losses for ranking quality

**Implementation**:
- Use `CoSENTLoss` or `RankingLoss` from sentence-transformers
- Train with relevance labels (not just positive/negative)
- Optimize directly for nDCG

**Expected Improvement**: **+2-5% nDCG@10**
- If successful: **0.46-0.48 nDCG@10**

**Effort**: Medium (need to adapt training pipeline)  
**Risk**: Moderate (different loss functions may need tuning)

**Experiments to Run**:
- `phase6_ltr_listwise_loss`
- `phase6_ltr_pairwise_loss`
- `phase6_ltr_ndcg_optimized`

---

### 7. ⭐⭐⭐ **Learned Hybrid Fusion Weights** (GOOD IMPACT)

**Why This Improves nDCG**:
- Current hybrid uses fixed alpha (0.3, 0.5, 0.7)
- **Learn optimal weights per domain** using validation set
- **Query-dependent weights** based on query characteristics
- Better balance between BM25 and dense signals

**Implementation**:
- Use validation set to optimize weights
- Grid search or gradient-based optimization
- Test per-domain vs. global weights
- Query-dependent weight learning

**Expected Improvement**: **+2-5% nDCG@10**
- If successful: **0.46-0.48 nDCG@10**

**Effort**: Medium (optimization needed)  
**Risk**: Low (hybrid retrieval already working)

**Experiments to Run**:
- `phase6_hybrid_learned_weights_per_domain`
- `phase6_hybrid_learned_weights_global`
- `phase6_hybrid_query_dependent_weights`

---

### 8. ⭐⭐⭐ **Pseudo-Relevance Feedback** (GOOD IMPACT)

**Why This Improves nDCG**:
- Use top retrieved results to expand query
- **Extract key terms** from top results
- **Re-retrieve** with expanded query
- Improves precision for relevant documents

**Implementation**:
- Step 1: Initial retrieval (top 10-20)
- Step 2: Extract keywords/entities from top results
- Step 3: Expand query with extracted terms
- Step 4: Re-retrieve with expanded query

**Expected Improvement**: **+2-5% nDCG@10**
- If successful: **0.46-0.48 nDCG@10**

**Effort**: Medium (need keyword extraction)  
**Risk**: Low (proven technique)

**Experiments to Run**:
- `phase6_pseudo_relevance_feedback_basic`
- `phase6_pseudo_relevance_feedback_llm_extraction`
- `phase6_pseudo_relevance_feedback_domain_specific`

---

### 9. ⭐⭐⭐ **Advanced Ensemble Strategies** (MODERATE IMPACT)

**Why This Improves nDCG**:
- Combine best techniques: Hard negatives + Query expansion + Hybrid
- **Learned ensemble weights** (not just RRF)
- **Per-domain ensemble strategies**
- **Query-dependent ensemble** (different strategies per query type)

**Implementation**:
- Ensemble: Hard negatives models + Query expansion + Hybrid results
- Learn optimal weights using validation set
- Test different combination methods (RRF, weighted, learned)

**Expected Improvement**: **+2-4% nDCG@10**
- If successful: **0.46-0.47 nDCG@10**

**Effort**: Medium (need to combine multiple techniques)  
**Risk**: Low (ensemble already proven effective)

**Experiments to Run**:
- `phase6_ensemble_all_techniques`
- `phase6_ensemble_learned_weights`
- `phase6_ensemble_per_domain`
- `phase6_ensemble_query_dependent`

---

### 10. ⭐⭐⭐ **Longer Training for Domain-Specific Models** (MODERATE IMPACT)

**Why This Improves nDCG**:
- Current: 7 epochs for domain-specific
- **Test 10, 15, 20 epochs** with early stopping
- May improve ranking quality with more training

**Implementation**:
- Increase epochs with validation monitoring
- Use early stopping to prevent overfitting
- Test different learning rate schedules

**Expected Improvement**: **+1-4% nDCG@10**
- If successful: **0.46-0.47 nDCG@10**

**Effort**: Low (just change config)  
**Risk**: Low (may have diminishing returns)

**Experiments to Run**:
- `phase6_domain_specific_clapnq_10epochs`
- `phase6_domain_specific_clapnq_15epochs`
- `phase6_domain_specific_govt_10epochs`
- `phase6_domain_specific_govt_15epochs`

---

## 🔍 Lower-Priority / Exploratory Experiments

### 11. ⭐⭐ **Alternative Embedding Models** (EXPLORATORY)

**Models to Test**:
- **E5 Models**: `intfloat/e5-base-v2`, `intfloat/e5-large-v2`
- **GTE Models**: `BAAI/gte-base`, `BAAI/gte-large`
- **Contriever**: `facebook/contriever`
- **ColBERT**: Multi-vector retrieval

**Expected Improvement**: Variable (0-8%)  
**Effort**: Medium (need to adapt training scripts)  
**Risk**: Moderate (may not outperform BGE)

---

### 12. ⭐⭐ **Query-Dependent Retrieval** (EXPLORATORY)

**Why This Improves nDCG**:
- Different query types need different strategies
- **Factual queries**: Entity-based retrieval
- **Conceptual queries**: Dense retrieval
- **Long queries**: Hybrid retrieval
- **Short queries**: Keyword-based

**Expected Improvement**: +1-4% nDCG@10  
**Effort**: Medium (need query classification)  
**Risk**: Moderate (complexity may not pay off)

---

### 13. ⭐⭐ **Curriculum Learning** (EXPLORATORY)

**Why This Improves nDCG**:
- Start with easy examples, gradually add harder ones
- May help with hard negative training
- Better convergence → better ranking

**Expected Improvement**: +1-3% nDCG@10  
**Effort**: Medium (need difficulty scoring)  
**Risk**: Moderate (may not help significantly)

---

## 📊 Recommended Experiment Priority

### Immediate (Highest Impact, Doable):

1. **Fine-Tuned Cross-Encoder Reranking** ⭐⭐⭐⭐⭐
   - Expected: +5-12% nDCG@10
   - Effort: Medium
   - **Target**: 0.48-0.51 nDCG@10

2. **LLM-Based Query Expansion** ⭐⭐⭐⭐⭐
   - Expected: +3-8% nDCG@10
   - Effort: Medium
   - **Target**: 0.47-0.49 nDCG@10

3. **Multi-Stage Retrieval Pipeline** ⭐⭐⭐⭐
   - Expected: +4-10% nDCG@10
   - Effort: Medium
   - **Target**: 0.47-0.50 nDCG@10

4. **Improved Hard Negative Mining** ⭐⭐⭐⭐
   - Expected: +5-12% nDCG@10 (if fixed)
   - Effort: Medium-High
   - **Target**: 0.48-0.51 nDCG@10

### Short-Term (Good Impact):

5. **Domain-Specific Query Expansion** ⭐⭐⭐⭐
6. **Learning to Rank Loss Functions** ⭐⭐⭐
7. **Learned Hybrid Fusion Weights** ⭐⭐⭐
8. **Pseudo-Relevance Feedback** ⭐⭐⭐

### Medium-Term (Exploratory):

9. **Advanced Ensemble Strategies** ⭐⭐⭐
10. **Longer Training for Domain-Specific** ⭐⭐⭐
11. **Alternative Embedding Models** ⭐⭐
12. **Query-Dependent Retrieval** ⭐⭐

---

## 🎯 Expected nDCG Trajectory

### Current Best: 0.4539

### Scenario 1: Cross-Encoder Success (Most Likely)
- Cross-encoder: +8% → **0.4902**
- LLM query expansion: +3% → **0.5049**
- Multi-stage: +2% → **0.5150**
- **Final**: **~0.51 nDCG@10** ✅ (Exceeds target!)

### Scenario 2: Moderate Success
- Cross-encoder: +5% → **0.4766**
- LLM query expansion: +2% → **0.4861**
- Multi-stage: +1% → **0.4910**
- **Final**: **~0.49 nDCG@10** ✅ (Close to target)

### Scenario 3: Limited Success
- Cross-encoder: +3% → **0.4676**
- LLM query expansion: +1% → **0.4723**
- Multi-stage: +0.5% → **0.4746**
- **Final**: **~0.47 nDCG@10** (Still improvement)

---

## 💡 Key Insights for nDCG Improvement

### What Works for nDCG:
1. ✅ **Ensemble methods** - Best overall (0.4539 nDCG@10)
2. ✅ **Query expansion** - Proven on ClapNQ (0.4981 nDCG@10 on that domain)
3. ✅ **Domain-specific models** - Strong single-domain performance
4. ✅ **Hybrid retrieval** - Currently testing (may help)

### What Doesn't Work (So Far):
1. ❌ **Pre-trained reranking** - Underperformed (0.3250 nDCG@10)
2. ❌ **Hard negatives (current impl)** - Failed (0.1444 nDCG@10)
3. ❌ **Simple hybrid (Phase 3)** - Underperformed

### What to Focus On:
1. **Ranking quality** (not just retrieval coverage)
2. **Top-K precision** (nDCG rewards top positions)
3. **Relevance discrimination** (distinguishing relevant from similar)
4. **Multi-stage refinement** (progressively improve ranking)

---

## 📝 Implementation Notes

### For Cross-Encoder Fine-Tuning:
- Use domain-specific training data
- Fine-tune on (query, positive, negative) triplets
- Test different architectures (L-6, L-12, L-24)
- Rerank top 50-100 from best dense model

### For LLM Query Expansion:
- Use GPT-4/Claude API or local LLM
- Generate 3-5 query variations per original query
- Combine results using RRF or weighted average
- Domain-specific prompts may help

### For Multi-Stage Pipeline:
- Stage 1: Fast retrieval (top 100) - use best ensemble
- Stage 2: Cross-encoder reranking (top 50)
- Stage 3: Optional - larger model or LLM reranking (top 20)

### For Hard Negative Mining:
- Use in-batch negative mining (proven to work)
- Start with moderate difficulty, not hardest
- Consider curriculum learning
- Use proper loss function (MultipleNegativesRankingLoss)

---

## 🎯 Action Plan

### Week 1 (Immediate):
1. **Fine-tune cross-encoder** on domain-specific data
2. **Test LLM query expansion** (start with GPT-4 API)
3. **Implement multi-stage pipeline** (2-stage first)

### Week 2 (Short-term):
4. **Fix hard negative mining** (try in-batch approach)
5. **Domain-specific query expansion** strategies
6. **Test learning to rank** loss functions

### Week 3+ (If needed):
7. **Advanced ensemble** of all techniques
8. **Longer training** for domain-specific models
9. **Exploratory techniques** (alternative models, etc.)

---

## 📈 Summary Table

| Experiment | Expected nDCG@10 | Expected Improvement | Effort | Priority |
|------------|------------------|---------------------|--------|----------|
| **Fine-Tuned Cross-Encoder** | 0.48-0.51 | +5-12% | Medium | ⭐⭐⭐⭐⭐ |
| **LLM Query Expansion** | 0.47-0.49 | +3-8% | Medium | ⭐⭐⭐⭐⭐ |
| **Multi-Stage Pipeline** | 0.47-0.50 | +4-10% | Medium | ⭐⭐⭐⭐ |
| **Improved Hard Negatives** | 0.48-0.51 | +5-12% | Medium-High | ⭐⭐⭐⭐ |
| **Domain-Specific QE** | 0.46-0.48 | +2-6% | Medium | ⭐⭐⭐⭐ |
| **Learning to Rank** | 0.46-0.48 | +2-5% | Medium | ⭐⭐⭐ |
| **Learned Hybrid Weights** | 0.46-0.48 | +2-5% | Medium | ⭐⭐⭐ |
| **Pseudo-Relevance Feedback** | 0.46-0.48 | +2-5% | Medium | ⭐⭐⭐ |

**Best Case Scenario**: Combining top 3 techniques could reach **0.51+ nDCG@10** ✅

---

*Last Updated: 2025-12-16*  
*Current Best nDCG@10: 0.4539*  
*Target: 0.50+ nDCG@10*

