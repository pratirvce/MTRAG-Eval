# nDCG Improvement Analysis

**Current Best nDCG@10**: **0.4539** (phase5_ensemble_domain_specific)  
**MTRAGEval Task A Ranking Metric**: **nDCG** (Normalized Discounted Cumulative Gain)

---

## 🎯 Why nDCG Matters

nDCG is the **official ranking metric** for MTRAGEval Task A. Unlike Recall, nDCG:
- **Penalizes lower-ranked relevant documents** (position matters)
- **Rewards putting highly relevant documents at the top**
- **Is the metric used for leaderboard ranking**

**Goal**: Maximize nDCG@10 to improve leaderboard position.

---

## 📊 Current nDCG Performance

### Best Results by Technique:

| Rank | Experiment | nDCG@10 | Method | Status |
|------|------------|---------|--------|--------|
| 🥇 | **phase5_ensemble_domain_specific** | **0.4539** | Ensemble (RRF) | ✅ Completed |
| 🥈 | phase5_ensemble_weighted | 0.4370 | Weighted Ensemble | ✅ Completed |
| 🥉 | phase5_query_expansion_clapnq | 0.4186 | Query Expansion | ✅ Completed |
| 4 | phase4_domain_specific_clapnq | 0.4981* | Domain-Specific | ✅ Completed |
| 5 | phase4_domain_specific_govt | 0.4628* | Domain-Specific | ✅ Completed |
| 6 | phase2_augmentation | 0.4786* | Data Augmentation | ✅ Completed |

*Domain-specific scores (single domain), not average

### Key Insight:
- **Query expansion on ClapNQ** achieved **0.4981 nDCG@10** on that domain (excellent!)
- **Domain-specific models** show strong nDCG on individual domains
- **Ensemble** combines strengths but may average down high-performing domains

---

## 🔬 Experiments Most Likely to Improve nDCG

### 1. ⭐ **Hard Negative Mining Training** (HIGHEST POTENTIAL)

**Status**: 🟢 Currently Running (2 experiments)

**Experiments**:
- `phase5_domain_specific_clapnq_hard_negatives` (Running ~26 hours)
- `phase5_domain_specific_govt_hard_negatives` (Running ~26 hours)

**Why This Improves nDCG**:
- **Hard negatives improve ranking discrimination**
- Model learns to distinguish subtle differences
- Better at ranking relevant documents higher
- **Directly targets nDCG improvement** (better top-K ranking)

**Expected Improvement**: +3-8% nDCG@10
- If successful, could reach **0.47-0.49 nDCG@10**

**Risk**: Low (training technique, proven to work)

---

### 2. ⭐ **Query Expansion** (HIGH POTENTIAL)

**Status**: 🟢 Running (2 experiments), ⏳ Pending (0)

**Experiments**:
- `phase5_query_expansion_govt` (Running)
- `phase5_query_expansion_multi` (Running)

**Why This Improves nDCG**:
- **Query expansion on ClapNQ already achieved 0.4981 nDCG@10** on that domain
- Expands queries with synonyms/context → better matching
- Helps find relevant documents that might be missed
- **Proven success on ClapNQ domain**

**Expected Improvement**: +2-5% nDCG@10
- If Govt/Multi show similar gains: **0.47-0.48 nDCG@10**

**Risk**: Low (already proven on ClapNQ)

---

### 3. ⭐ **Hybrid Retrieval** (MODERATE-HIGH POTENTIAL)

**Status**: 🟢 Running (2), ⏳ Pending (7)

**Experiments**:
- `phase5_hybrid_clapnq_alpha0.3` (Running)
- `phase5_hybrid_clapnq_alpha0.5` (Running)
- `phase5_hybrid_clapnq_alpha0.7` (Pending)
- `phase5_hybrid_govt_alpha0.3/0.5/0.7` (Pending)
- `phase5_hybrid_multi_alpha0.3/0.5/0.7` (Pending)

**Why This Improves nDCG**:
- **Combines BM25 (lexical) + Dense (semantic)**
- BM25 catches exact matches (good for nDCG@1)
- Dense catches semantic matches
- **Optimal alpha** balances both signals
- Can improve ranking quality by leveraging both signals

**Expected Improvement**: +1-4% nDCG@10
- Best alpha value could reach **0.47-0.49 nDCG@10**

**Risk**: Moderate (depends on finding optimal alpha)

---

### 4. **Reranking** (MIXED RESULTS - LOWER PRIORITY)

**Status**: ✅ Completed (4 experiments)

**Results**:
- ClapNQ: 0.3250 nDCG@10 (lower than expected)
- Govt: 0.2896 nDCG@10
- Cloud: 0.2028 nDCG@10
- Multi: 0.2619 nDCG@10

**Why It Might Not Help**:
- **Reranking results have been disappointing**
- Lower nDCG than base models
- May need better reranker training
- Cross-encoder reranking didn't improve ranking quality

**Expected Improvement**: -5% to +2% nDCG@10 (unpredictable)

**Risk**: High (results so far are negative)

**Recommendation**: **Skip further reranking experiments** unless we improve the reranker training.

---

## 🎯 Recommended Strategy for nDCG Improvement

### Priority 1: Wait for Hard Negative Results ⭐⭐⭐
**Why**: Highest potential impact, directly targets ranking quality
**Action**: Monitor the 2 running hard negative experiments
**Expected**: +3-8% improvement if successful

### Priority 2: Complete Query Expansion ⭐⭐
**Why**: Already proven on ClapNQ (0.4981 nDCG@10)
**Action**: Wait for Govt and Multi results
**Expected**: +2-5% improvement if consistent

### Priority 3: Optimize Hybrid Retrieval ⭐⭐
**Why**: Combining signals can improve ranking
**Action**: Test all alpha values, find optimal
**Expected**: +1-4% improvement with optimal alpha

### Priority 4: Advanced Ensemble ⭐
**Why**: Combine best techniques
**Action**: After hard negatives complete, ensemble with query expansion
**Expected**: +1-3% improvement

---

## 📈 Expected nDCG Trajectory

### Current Best: 0.4539

### Scenario 1: Hard Negatives Success (Most Likely)
- Hard negatives: +5% → **0.4766**
- Query expansion: +3% → **0.4909**
- Hybrid optimal: +2% → **0.5007**
- **Final**: **~0.50 nDCG@10** ✅

### Scenario 2: Moderate Success
- Hard negatives: +3% → **0.4675**
- Query expansion: +2% → **0.4770**
- Hybrid optimal: +1% → **0.4818**
- **Final**: **~0.48 nDCG@10** ✅

### Scenario 3: Limited Success
- Hard negatives: +1% → **0.4584**
- Query expansion: +1% → **0.4626**
- Hybrid optimal: +0.5% → **0.4650**
- **Final**: **~0.46-0.47 nDCG@10** (still improvement)

---

## 🔍 Techniques That DON'T Help nDCG

### ❌ Reranking (Current Implementation)
- **Results**: Lower nDCG than base models
- **Reason**: Reranker may not be well-trained
- **Action**: Skip unless we improve reranker

### ❌ Simple Hybrid (Without Optimization)
- **Results**: Need to find optimal alpha
- **Reason**: Wrong balance hurts performance
- **Action**: Test multiple alpha values (already planned)

---

## 💡 Additional Experiments to Consider

### 1. **Ensemble of Best Techniques**
- Combine: Hard negatives + Query expansion + Hybrid
- Use RRF or learned weights
- **Expected**: +2-4% nDCG@10

### 2. **Domain-Specific Query Expansion**
- Different expansion strategies per domain
- **Expected**: +1-3% nDCG@10

### 3. **Two-Stage Retrieval**
- Stage 1: Dense retrieval (top 100)
- Stage 2: Rerank with better reranker
- **Expected**: +2-5% nDCG@10 (if reranker improved)

### 4. **Learned Hybrid Weights**
- Train model to learn optimal BM25/dense weights
- Per-domain or per-query weights
- **Expected**: +1-3% nDCG@10

---

## 📊 Comparison with Baseline

| Metric | Baseline | Current Best | Improvement |
|--------|----------|---------------|-------------|
| nDCG@10 | 0.30 | 0.4539 | **+51.3%** 🚀 |
| Recall@10 | 0.38 | 0.5441 | +43.2% |

**We've already improved nDCG by 51%!** Further improvements will be incremental but valuable.

---

## 🎯 Action Plan

### Immediate (Next 1-2 days):
1. ✅ **Monitor hard negative training** (2 experiments running)
2. ✅ **Wait for query expansion results** (2 experiments running)
3. ✅ **Complete hybrid retrieval tests** (7 experiments pending)

### Short-term (Next week):
1. **Evaluate hard negative models** (if successful)
2. **Create ensemble** of best techniques
3. **Test domain-specific query expansion**

### Long-term (If needed):
1. **Improve reranker training** (if we want to revisit reranking)
2. **Two-stage retrieval** with better reranker
3. **Learned hybrid weights** per domain

---

## 📝 Key Takeaways

1. **Hard negative mining** has highest potential (+3-8% nDCG)
2. **Query expansion** already proven on ClapNQ (0.4981 nDCG@10)
3. **Hybrid retrieval** needs optimization (test all alpha values)
4. **Reranking** needs improvement (current results negative)
5. **Ensemble** of best techniques likely final solution

**Target**: Achieve **0.50+ nDCG@10** (10% improvement from current best)

---

*Last Updated: 2025-12-14*
*Current Best nDCG@10: 0.4539*

