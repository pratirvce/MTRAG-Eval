# Top Experiments for ACL Submission & Leaderboard Dominance

**Target**: Beat Elser's **0.54 nDCG@10** (paper's best result)  
**Current Best**: **0.4539 nDCG@10**  
**Gap**: Need **+19% improvement** to beat Elser  
**Timeline**: Evaluation Jan 10-20, 2026 | Paper due Feb 2026

**Reference**: [MTRAGEval Task A](https://ibm.github.io/mt-rag-benchmark/MTRAGEval/) - Ranking uses **nDCG**

---

## 🎯 Critical Path to Top Leaderboard

### Current Competitive Position

| System | nDCG@10 | Your Gap |
|--------|---------|----------|
| **Elser (Query Rewrite)** | **0.54** | -16% (need to beat) |
| Elser (Last Turn) | 0.49 | -7% |
| **Your Best** | **0.4539** | - |
| BGE-base (Query Rewrite) | 0.38 | +19% ✅ |
| BGE-base (Last Turn) | 0.30 | +51% ✅ |

**Goal**: Achieve **0.55+ nDCG@10** to secure top leaderboard position

---

## 🚀 Tier 1: MUST DO Experiments (Highest Impact)

### 1. ⭐⭐⭐⭐⭐ **Fine-Tuned Cross-Encoder Reranking** (CRITICAL)

**Why This is Essential**:
- **Cross-encoders are the gold standard** for ranking quality
- **Elser likely uses cross-encoder** (or similar) - you need to match/beat it
- **Fine-tuning on domain data** should outperform pre-trained
- **Directly optimizes nDCG** (ranking metric)

**Implementation Details**:
```python
# Fine-tune cross-encoder on domain-specific data
Model: cross-encoder/ms-marco-MiniLM-L-12-v2 (or L-24)
Training: (query, positive, negative) triplets from training data
Rerank: Top 100 from phase5_ensemble_domain_specific → Top 50
```

**Expected Results**:
- **nDCG@10**: **0.49-0.52** (+8-15% improvement)
- **Beats Elser**: If >0.54 ✅

**Experiments to Run**:
1. `phase6_cross_encoder_finetuned_ensemble`
   - Fine-tune on all domains combined
   - Rerank ensemble results
   - **Expected**: 0.49-0.52 nDCG@10

2. `phase6_cross_encoder_finetuned_per_domain`
   - Domain-specific cross-encoders
   - Rerank per-domain results
   - **Expected**: 0.50-0.53 nDCG@10 (better per-domain)

3. `phase6_cross_encoder_large`
   - Use larger model (L-24 or larger)
   - Final reranking stage
   - **Expected**: 0.51-0.54 nDCG@10

**Paper Contribution**: "Domain-Adapted Cross-Encoder Reranking for Multi-Turn RAG"

**Time Required**: 3-5 days (training + evaluation)

---

### 2. ⭐⭐⭐⭐⭐ **Multi-Stage Retrieval Pipeline** (CRITICAL)

**Why This is Essential**:
- **Novel contribution** for multi-turn RAG (publishable)
- **Progressive refinement** improves ranking quality
- **Combines best techniques** systematically
- **Clear methodology** (easy to explain in paper)

**Implementation Details**:
```
Stage 1: Fast Retrieval (Top 100)
  - Use: phase5_ensemble_domain_specific
  - Method: Dense retrieval with ensemble
  - Time: Fast (~1-2 seconds)

Stage 2: Cross-Encoder Reranking (Top 50)
  - Use: Fine-tuned cross-encoder
  - Method: Precise relevance scoring
  - Time: Moderate (~5-10 seconds)

Stage 3: Final Reranking (Top 20) [Optional]
  - Use: Larger cross-encoder or LLM
  - Method: Highest precision
  - Time: Slower (~10-20 seconds)
```

**Expected Results**:
- **2-Stage**: **0.48-0.51 nDCG@10** (+6-12% improvement)
- **3-Stage**: **0.50-0.53 nDCG@10** (+10-17% improvement)

**Experiments to Run**:
1. `phase6_multistage_2stage`
   - Dense + Cross-encoder
   - **Expected**: 0.48-0.51 nDCG@10

2. `phase6_multistage_3stage`
   - Dense + Cross-encoder + Final reranking
   - **Expected**: 0.50-0.53 nDCG@10

3. `phase6_multistage_adaptive`
   - Adaptive stage selection based on query
   - **Expected**: 0.51-0.54 nDCG@10

**Paper Contribution**: "Progressive Multi-Stage Retrieval for Multi-Turn RAG Conversations"

**Time Required**: 2-4 days (implementation + evaluation)

---

### 3. ⭐⭐⭐⭐⭐ **LLM-Based Multi-Query Expansion** (CRITICAL)

**Why This is Essential**:
- **Query expansion already proven** (0.4981 nDCG@10 on ClapNQ)
- **LLMs generate better variations** than simple expansion
- **Modern, publishable approach** (LLMs are hot topic)
- **Multi-query generation** is novel for multi-turn RAG

**Implementation Details**:
```python
# Generate multiple query variations using LLM
Prompt: "Generate 3-5 query variations for better document retrieval: [query]"
LLM: GPT-4, Claude, or local model (Llama, Mistral)
Combination: RRF or learned weights
```

**Expected Results**:
- **nDCG@10**: **0.48-0.51** (+6-12% improvement)
- **With domain-specific prompts**: **0.50-0.53 nDCG@10**

**Experiments to Run**:
1. `phase6_llm_query_expansion_gpt4_multi`
   - GPT-4 generates 3-5 variations
   - Combine with RRF
   - **Expected**: 0.48-0.51 nDCG@10

2. `phase6_llm_query_expansion_domain_specific`
   - Domain-specific prompts
   - **Expected**: 0.50-0.53 nDCG@10

3. `phase6_llm_query_expansion_ensemble`
   - Combine with other techniques
   - **Expected**: 0.51-0.54 nDCG@10

**Paper Contribution**: "LLM-Enhanced Multi-Query Expansion for Multi-Turn RAG"

**Time Required**: 2-3 days (API setup + evaluation)

---

### 4. ⭐⭐⭐⭐ **In-Batch Hard Negative Mining** (HIGH PRIORITY)

**Why This is Essential**:
- **Previous attempts failed** - but technique is proven
- **In-batch mining is modern** and efficient
- **Directly improves ranking** (better discrimination)
- **Publishable** - can discuss why previous failed, how this succeeds

**Implementation Details**:
```python
# In-batch hard negative mining
Loss: MultipleNegativesRankingLoss
Negatives: Mine within batch (not full corpus)
Training: Start from best domain-specific models
```

**Expected Results**:
- **nDCG@10**: **0.48-0.51** (+6-12% improvement)
- **If successful**: Could reach 0.50-0.53 nDCG@10

**Experiments to Run**:
1. `phase6_hard_negatives_inbatch_clapnq`
   - In-batch mining for ClapNQ
   - **Expected**: 0.48-0.51 nDCG@10 (on ClapNQ)

2. `phase6_hard_negatives_inbatch_govt`
   - In-batch mining for Govt
   - **Expected**: 0.48-0.51 nDCG@10 (on Govt)

3. `phase6_hard_negatives_curriculum`
   - Curriculum learning (gradual difficulty)
   - **Expected**: 0.49-0.52 nDCG@10

**Paper Contribution**: "In-Batch Hard Negative Mining for Multi-Turn RAG Ranking"

**Time Required**: 4-6 days (training is slow)

---

## 🔬 Tier 2: SHOULD DO Experiments (Strong Impact)

### 5. ⭐⭐⭐⭐ **Domain-Adaptive Hybrid Retrieval**

**Why This Helps**:
- **Learn optimal weights per domain** (not fixed alpha)
- **Publishable** - domain adaptation angle
- **Hybrid retrieval currently testing** - optimize it

**Expected**: +3-7% nDCG@10 → **0.47-0.49 nDCG@10**

### 6. ⭐⭐⭐⭐ **Advanced Ensemble of All Techniques**

**Why This Helps**:
- **Combine all best techniques** systematically
- **Learned ensemble weights** (not just RRF)
- **Strong results** - ensemble already best

**Expected**: +3-6% nDCG@10 → **0.47-0.48 nDCG@10**

### 7. ⭐⭐⭐ **Learning to Rank with Listwise Loss**

**Why This Helps**:
- **Direct nDCG optimization** - listwise losses optimize ranking
- **Technical depth** - less common in RAG
- **Publishable** - clear technical contribution

**Expected**: +2-5% nDCG@10 → **0.46-0.48 nDCG@10**

---

## 📊 Expected Final Results (Combining Techniques)

### Scenario 1: All Techniques Success (Best Case)
- **Base**: 0.4539
- **Cross-encoder**: +10% → **0.4993**
- **Multi-stage**: +5% → **0.5243**
- **LLM expansion**: +3% → **0.5400**
- **Hard negatives**: +2% → **0.5508**
- **Final**: **~0.55 nDCG@10** ✅ **BEATS ELSER!**

### Scenario 2: Most Techniques Success (Realistic)
- **Base**: 0.4539
- **Cross-encoder**: +8% → **0.4902**
- **Multi-stage**: +4% → **0.5098**
- **LLM expansion**: +2% → **0.5200**
- **Final**: **~0.52 nDCG@10** ✅ **Competitive with Elser**

### Scenario 3: Core Techniques Only (Conservative)
- **Base**: 0.4539
- **Cross-encoder**: +6% → **0.4816**
- **Multi-stage**: +3% → **0.4960**
- **Final**: **~0.50 nDCG@10** ✅ **Strong improvement**

---

## 🎯 Recommended Experiment Order

### Week 1 (Highest Priority):
1. **Cross-encoder fine-tuning** (3-4 days)
   - Highest impact on nDCG
   - Essential for beating Elser
   - **Target**: 0.49-0.52 nDCG@10

2. **Multi-stage pipeline** (2-3 days)
   - Novel contribution
   - Strong results
   - **Target**: 0.48-0.51 nDCG@10

### Week 2:
3. **LLM query expansion** (2-3 days)
   - Modern approach
   - Proven effective
   - **Target**: 0.48-0.51 nDCG@10

4. **Combine techniques** (1-2 days)
   - Cross-encoder + Multi-stage
   - **Target**: 0.50-0.53 nDCG@10

### Week 3:
5. **Hard negative mining** (4-6 days)
   - If time permits
   - High potential
   - **Target**: 0.48-0.51 nDCG@10

6. **Final ensemble** (2-3 days)
   - Combine all techniques
   - **Target**: **0.52-0.55 nDCG@10** ✅

---

## 📝 Paper-Ready Results Table (Projected)

| Method | nDCG@10 | R@10 | vs Baseline | vs Elser | Status |
|--------|---------|------|-------------|----------|--------|
| **Baseline (BGE)** | 0.30 | 0.38 | - | - | Reference |
| **Elser (Paper Best)** | 0.54 | 0.64 | +80% | - | To Beat |
| **Your: Ensemble** | 0.4539 | 0.5441 | +51% | -16% | ✅ Current |
| **Your: + Cross-Encoder** | 0.49-0.52 | 0.55-0.58 | +63-73% | -9% to -4% | 🎯 Target |
| **Your: + Multi-Stage** | 0.50-0.53 | 0.56-0.59 | +67-77% | -7% to -2% | 🎯 Target |
| **Your: + LLM Expansion** | 0.52-0.55 | 0.57-0.60 | +73-83% | -4% to +2% | 🎯 **BEATS ELSER** |

---

## 🏆 Success Metrics for ACL

### Minimum for Acceptance:
- ✅ **nDCG@10 ≥ 0.50** (strong improvement)
- ✅ **Clear technical contribution**
- ✅ **Comprehensive evaluation**

### For Top Leaderboard:
- ✅ **nDCG@10 ≥ 0.54** (beat/match Elser)
- ✅ **Consistent across domains**
- ✅ **Novel methodology**

### For #1 Position:
- ✅ **nDCG@10 ≥ 0.55** (exceed Elser)
- ✅ **Multiple novel contributions**
- ✅ **Strong analysis**

---

## 💡 Key Insights for Paper

### What Makes Your Approach Novel:
1. **Multi-stage pipeline** for multi-turn RAG (not done before)
2. **Domain-adaptive cross-encoder** reranking
3. **LLM-enhanced query expansion** for multi-turn queries
4. **Comprehensive ensemble** combining all techniques

### What to Emphasize:
- **Systematic approach** - not just throwing techniques together
- **Domain adaptation** - key insight for multi-domain RAG
- **Progressive refinement** - multi-stage improves ranking
- **Comprehensive evaluation** - all domains, detailed analysis

---

## 🎯 Final Recommendations

### Start Immediately:
1. ✅ **Fine-tune cross-encoder** (highest impact, 3-4 days)
2. ✅ **Implement multi-stage pipeline** (novel contribution, 2-3 days)
3. ✅ **Set up LLM query expansion** (modern approach, 2-3 days)

### If Time Permits:
4. ⚪ **In-batch hard negative mining** (high potential, 4-6 days)
5. ⚪ **Learned hybrid weights** (optimization, 2-3 days)
6. ⚪ **Final ensemble** (combine all, 2-3 days)

### Expected Outcome:
- **Best Case**: **0.55 nDCG@10** (beats Elser, #1 position) ✅
- **Realistic**: **0.52 nDCG@10** (competitive, top 3) ✅
- **Conservative**: **0.50 nDCG@10** (strong, top 5) ✅

---

*Strategy optimized for ACL 2026 submission and MTRAGEval Task A leaderboard*  
*Reference: [MTRAGEval Official](https://ibm.github.io/mt-rag-benchmark/MTRAGEval/)*

