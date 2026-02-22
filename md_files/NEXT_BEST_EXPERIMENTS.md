# Next Best Experiments to Run

**Last Updated**: 2025-12-17  
**Current Best**: 0.4576 nDCG@10 (Contrastive Learning)  
**Target**: Beat Elser's 0.54 nDCG@10  
**Gap**: Need +18% improvement to reach 0.54

---

## 📊 Current Status

### ✅ Completed Experiments
- **Contrastive Learning**: 0.4576 nDCG@10 (BEST)
- **Iterative Refinement**: 0.2231 nDCG@10 (low, needs improvement)
- **Ensemble Advanced**: 0.2192 nDCG@10
- **Multi-Stage 3-Stage**: 0.2480 nDCG@10

### 🟢 Currently Running
- Cross-Encoder Evaluation (GPU 0)
- Cross-Attention Rerun (GPU 1)
- LLM Query Expansion (GPU 2)
- Multi-Stage 2-Stage (GPU 3)
- Hierarchical Multi-Granularity (GPU 1, long-running)

---

## 🚀 Top 10 Next Best Experiments (Priority Order)

### 1. ⭐⭐⭐⭐⭐ **Cross-Encoder with Larger Model** (HIGHEST PRIORITY)

**Why This Is Critical**:
- ✅ Cross-encoders are proven to work (gold standard for ranking)
- ✅ Larger models (L-24) typically outperform smaller ones
- ✅ Directly optimizes nDCG (ranking metric)
- ✅ Can combine with fine-tuned cross-encoder results

**Expected Results**:
- **nDCG@10**: **0.51-0.54** (+11-18% improvement)
- **Potential**: Could beat Elser's 0.54!

**Implementation**:
- Use `cross-encoder/ms-marco-MiniLM-L-24-v2` or larger
- Rerank top 100 from best ensemble/contrastive results
- Fine-tune on domain-specific data if time permits

**Time**: 2-4 days  
**Complexity**: Medium  
**GPU**: 1 GPU needed

---

### 2. ⭐⭐⭐⭐⭐ **Domain-Specific Cross-Encoder Per Domain** (CRITICAL)

**Why This Is Critical**:
- ✅ Domain-specific fine-tuning outperforms general models
- ✅ Each domain has unique characteristics
- ✅ Can achieve 0.50-0.53 nDCG@10 per domain
- ✅ Strong publication value

**Expected Results**:
- **nDCG@10**: **0.50-0.53** (+9-16% improvement)
- **Per-domain**: Better than combined model

**Implementation**:
- Train separate cross-encoder for each domain (clapnq, fiqa, govt, cloud)
- Use domain-specific training data
- Rerank per-domain results

**Time**: 4-6 days (can run in parallel)  
**Complexity**: Medium  
**GPU**: 4 GPUs (one per domain) or sequential

---

### 3. ⭐⭐⭐⭐⭐ **Ensemble of Best Methods** (CRITICAL)

**Why This Is Critical**:
- ✅ Current best: Contrastive Learning (0.4576)
- ✅ Multiple strong methods can be combined
- ✅ Ensemble typically outperforms individual methods
- ✅ Fast to implement (no training needed)

**Expected Results**:
- **nDCG@10**: **0.48-0.52** (+5-14% improvement)
- **If best methods work**: Could reach 0.52-0.55 nDCG@10

**Implementation**:
- Combine: Contrastive Learning + Cross-Encoder + LLM Expansion + Multi-Stage
- Use Reciprocal Rank Fusion (RRF) or learned weights
- Test different combination strategies

**Time**: 1-2 days  
**Complexity**: Low  
**GPU**: 1 GPU (for evaluation)

---

### 4. ⭐⭐⭐⭐ **3-Stage Multi-Stage with Fine-Tuned Models** (HIGH PRIORITY)

**Why This Is Critical**:
- ✅ 2-stage already running, 3-stage adds final refinement
- ✅ Uses fine-tuned models at each stage
- ✅ Progressive refinement improves ranking
- ✅ Novel contribution for publication

**Expected Results**:
- **nDCG@10**: **0.50-0.53** (+9-16% improvement)
- **Better than 2-stage**: +2-5% additional improvement

**Implementation**:
- Stage 1: Best dense retrieval (Contrastive Learning or Ensemble)
- Stage 2: Fine-tuned cross-encoder reranking
- Stage 3: Larger cross-encoder or LLM-based reranking

**Time**: 3-5 days  
**Complexity**: Medium  
**GPU**: 1 GPU needed

---

### 5. ⭐⭐⭐⭐ **LLM Query Expansion with Domain-Specific Prompts** (HIGH PRIORITY)

**Why This Is Critical**:
- ✅ LLM expansion already running (basic version)
- ✅ Domain-specific prompts improve quality
- ✅ Can achieve 0.50-0.53 nDCG@10
- ✅ Modern, publishable approach

**Expected Results**:
- **nDCG@10**: **0.50-0.53** (+9-16% improvement)
- **Better than basic LLM expansion**: +2-5% additional improvement

**Implementation**:
- Create domain-specific prompts for each domain
- Use GPT-4 or Claude for query generation
- Combine with RRF or learned weights

**Time**: 2-3 days  
**Complexity**: Low-Medium  
**GPU**: 1 GPU (or API-based, no GPU needed)

---

### 6. ⭐⭐⭐⭐ **In-Batch Hard Negative Mining** (HIGH PRIORITY)

**Why This Is Critical**:
- ✅ Proven technique for improving ranking
- ✅ Previous attempts may have failed due to implementation
- ✅ Directly improves discrimination between relevant/irrelevant
- ✅ Can fine-tune best models further

**Expected Results**:
- **nDCG@10**: **0.48-0.51** (+5-12% improvement)
- **If successful**: Could reach 0.50-0.53 nDCG@10

**Implementation**:
- Use MultipleNegativesRankingLoss with in-batch mining
- Start from best domain-specific models
- Train on all domains or per-domain

**Time**: 4-6 days  
**Complexity**: Medium-High  
**GPU**: 1 GPU needed

---

### 7. ⭐⭐⭐ **Adaptive Multi-Stage Retrieval** (MEDIUM PRIORITY)

**Why This Is Critical**:
- ✅ Adapts number of stages based on query complexity
- ✅ Novel contribution for publication
- ✅ Efficient (doesn't always use all stages)
- ✅ Can achieve 0.51-0.54 nDCG@10

**Expected Results**:
- **nDCG@10**: **0.51-0.54** (+11-18% improvement)
- **Efficiency**: Faster than always using 3 stages

**Implementation**:
- Classify query complexity (simple vs. complex)
- Simple queries: 2-stage pipeline
- Complex queries: 3-stage pipeline
- Use query features or LLM to determine complexity

**Time**: 3-4 days  
**Complexity**: Medium  
**GPU**: 1 GPU needed

---

### 8. ⭐⭐⭐ **Pseudo-Relevance Feedback with LLM Expansion** (MEDIUM PRIORITY)

**Why This Is Critical**:
- ✅ Uses initial results to expand query
- ✅ LLM generates better expansions than simple methods
- ✅ Novel combination for multi-turn RAG
- ✅ Can achieve 0.49-0.52 nDCG@10

**Expected Results**:
- **nDCG@10**: **0.49-0.52** (+7-14% improvement)
- **Better than basic expansion**: Uses feedback from results

**Implementation**:
- Initial retrieval (top 10-20)
- Extract key terms/concepts from top results
- Use LLM to generate expanded query
- Re-retrieve with expanded query
- Combine both rounds

**Time**: 2-3 days  
**Complexity**: Medium  
**GPU**: 1 GPU (or API-based)

---

### 9. ⭐⭐⭐ **Hybrid Dense-Sparse Retrieval with Learned Weights** (MEDIUM PRIORITY)

**Why This Is Critical**:
- ✅ Combines best of both worlds (dense + sparse)
- ✅ Learned weights outperform fixed weights
- ✅ Can achieve 0.48-0.51 nDCG@10
- ✅ Proven technique in retrieval

**Expected Results**:
- **nDCG@10**: **0.48-0.51** (+5-12% improvement)
- **Better than fixed weights**: +2-4% additional improvement

**Implementation**:
- Dense: Best model (Contrastive Learning or Ensemble)
- Sparse: BM25 or Elser
- Learn optimal combination weights on validation set
- Apply to test set

**Time**: 2-3 days  
**Complexity**: Medium  
**GPU**: 1 GPU needed

---

### 10. ⭐⭐⭐ **Curriculum Learning for Hard Negatives** (MEDIUM PRIORITY)

**Why This Is Critical**:
- ✅ Gradual difficulty increase improves training
- ✅ Better than hard negatives from start
- ✅ Can fine-tune best models further
- ✅ Can achieve 0.49-0.52 nDCG@10

**Expected Results**:
- **nDCG@10**: **0.49-0.52** (+7-14% improvement)
- **Better than standard hard negatives**: +1-3% additional improvement

**Implementation**:
- Start with easy negatives (random)
- Gradually increase difficulty (harder negatives)
- Train in stages with increasing difficulty
- Use best models as starting point

**Time**: 5-7 days  
**Complexity**: High  
**GPU**: 1 GPU needed

---

## 📊 Comparison Table

| Rank | Experiment | Expected nDCG@10 | Impact | Complexity | Time | Priority | Novelty |
|------|------------|------------------|--------|------------|------|----------|---------|
| 🥇 | **Cross-Encoder Large** | **0.51-0.54** | +11-18% | Medium | 2-4 days | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| 🥈 | **Domain-Specific Cross-Encoder** | **0.50-0.53** | +9-16% | Medium | 4-6 days | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| 🥉 | **Ensemble Best Methods** | **0.48-0.52** | +5-14% | Low | 1-2 days | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| 4 | **3-Stage Multi-Stage** | **0.50-0.53** | +9-16% | Medium | 3-5 days | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| 5 | **LLM Expansion Domain-Specific** | **0.50-0.53** | +9-16% | Low-Medium | 2-3 days | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| 6 | **In-Batch Hard Negatives** | **0.48-0.51** | +5-12% | Medium-High | 4-6 days | ⭐⭐⭐⭐ | ⭐⭐⭐ |
| 7 | **Adaptive Multi-Stage** | **0.51-0.54** | +11-18% | Medium | 3-4 days | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| 8 | **Pseudo-Relevance Feedback** | **0.49-0.52** | +7-14% | Medium | 2-3 days | ⭐⭐⭐ | ⭐⭐⭐⭐ |
| 9 | **Hybrid Learned Weights** | **0.48-0.51** | +5-12% | Medium | 2-3 days | ⭐⭐⭐ | ⭐⭐⭐ |
| 10 | **Curriculum Learning** | **0.49-0.52** | +7-14% | High | 5-7 days | ⭐⭐⭐ | ⭐⭐⭐⭐ |

---

## 🎯 Recommended Implementation Strategy

### **Week 1: Critical Experiments** (Highest Impact)

1. **Ensemble Best Methods** (Priority 1)
   - Fastest to implement (1-2 days)
   - Can combine all running experiments
   - Expected: 0.48-0.52 nDCG@10
   - **Start immediately** ✅

2. **Cross-Encoder Large Model** (Priority 2)
   - High expected impact (0.51-0.54)
   - Could beat Elser!
   - Time: 2-4 days
   - **Start after ensemble** ✅

3. **Domain-Specific Cross-Encoder** (Priority 3)
   - Can run in parallel (4 GPUs)
   - Expected: 0.50-0.53
   - Time: 4-6 days
   - **Start in parallel with Priority 2** ✅

### **Week 2: High-Impact Experiments**

4. **3-Stage Multi-Stage** (Priority 4)
   - Builds on 2-stage results
   - Expected: 0.50-0.53
   - Time: 3-5 days

5. **LLM Expansion Domain-Specific** (Priority 5)
   - Builds on basic LLM expansion
   - Expected: 0.50-0.53
   - Time: 2-3 days

6. **In-Batch Hard Negatives** (Priority 6)
   - Fine-tune best models further
   - Expected: 0.48-0.51
   - Time: 4-6 days

### **Week 3: Novel Contributions**

7. **Adaptive Multi-Stage** (Priority 7)
   - Novel contribution
   - Expected: 0.51-0.54
   - Time: 3-4 days

8. **Pseudo-Relevance Feedback** (Priority 8)
   - Novel combination
   - Expected: 0.49-0.52
   - Time: 2-3 days

---

## 💡 Expected Combined Results

### **Best Case Scenario** (Top 3 Experiments Work):
- **Base**: 0.4576 (Contrastive Learning)
- **Ensemble**: +10% → **0.5034**
- **Cross-Encoder Large**: +8% → **0.5437** ✅ **BEATS ELSER!**
- **Domain-Specific**: +6% → **0.5767** ✅ **TIER 1 TARGET!**

### **Realistic Scenario** (Top 2 Experiments Work):
- **Base**: 0.4576
- **Ensemble**: +8% → **0.4942**
- **Cross-Encoder Large**: +6% → **0.5243** ✅ **COMPETITIVE!**

### **Conservative Scenario** (Top 1 Experiment Works):
- **Base**: 0.4576
- **Ensemble**: +8% → **0.4942** ✅ **STRONG IMPROVEMENT!**

---

## ✅ Action Items

### **Immediate Next Steps**:

1. **Start Ensemble Best Methods** (Priority 1)
   - Wait for running experiments to complete
   - Combine: Contrastive Learning + Cross-Encoder + LLM Expansion + Multi-Stage
   - Expected: 0.48-0.52 nDCG@10
   - **Time**: 1-2 days
   - **Can start**: After running experiments complete

2. **Start Cross-Encoder Large Model** (Priority 2)
   - Use `cross-encoder/ms-marco-MiniLM-L-24-v2`
   - Rerank best results
   - Expected: 0.51-0.54 nDCG@10
   - **Time**: 2-4 days
   - **Can start**: Immediately (if GPU available)

3. **Start Domain-Specific Cross-Encoder** (Priority 3)
   - Train 4 separate models (one per domain)
   - Can run in parallel on 4 GPUs
   - Expected: 0.50-0.53 nDCG@10
   - **Time**: 4-6 days
   - **Can start**: After Priority 2 or in parallel

---

## 📝 Summary

**Top 3 Recommended Experiments**:
1. **Ensemble Best Methods** ⭐⭐⭐⭐⭐ (Fastest, good impact)
2. **Cross-Encoder Large Model** ⭐⭐⭐⭐⭐ (Highest impact, could beat Elser)
3. **Domain-Specific Cross-Encoder** ⭐⭐⭐⭐⭐ (Strong impact, publication value)

**Expected Final Result**: **0.50-0.55 nDCG@10** (beats Elser's 0.54!) ✅

---

*Based on: NEXT_TIER1_EXPERIMENTS_RECOMMENDED.md, ACL_TOP_EXPERIMENTS.md, NOVEL_HIGH_IMPACT_EXPERIMENTS.md, and current experiment results*

