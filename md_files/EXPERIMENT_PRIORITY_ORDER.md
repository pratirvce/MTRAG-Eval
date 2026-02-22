# 🎯 Experiment Priority & Execution Order for Maximum nDCG

**Current Best**: **0.4539 nDCG@10**  
**Target to Beat**: **0.54 nDCG@10** (Elser - paper's best)  
**Your Goal**: **0.55+ nDCG@10** (Top leaderboard position)

---

## 📊 Priority Ranking (By Expected Impact on nDCG)

### 🥇 TIER 1: CRITICAL - Highest Impact (Start Immediately)

#### 1. ⭐⭐⭐⭐⭐ **Fine-Tuned Cross-Encoder Reranking** (MUST DO FIRST)

**Priority**: **#1 - HIGHEST**  
**Expected nDCG@10**: **0.49-0.52** (+8-15% improvement)  
**Time**: 3-5 days  
**Impact**: **CRITICAL** - Directly optimizes ranking quality

**Why This is #1**:
- **Cross-encoders are the gold standard** for ranking (nDCG optimization)
- **Elser likely uses cross-encoder** - you need to match/beat it
- **Highest single-technique impact** on nDCG
- **Foundation for other experiments** (multi-stage needs this)

**Experiment**: `phase6_cross_encoder_finetuned_ensemble`

**Dependencies**: None (can start immediately)

**Command**:
```bash
python train_cross_encoder_finetuned.py \
    --config experiments/retrieval/phase6_cross_encoder_finetuned_ensemble/config.json \
    --gpu_id 0 \
    --resume
```

---

#### 2. ⭐⭐⭐⭐⭐ **Multi-Stage Retrieval Pipeline** (DO AFTER #1)

**Priority**: **#2 - HIGH**  
**Expected nDCG@10**: **0.50-0.53** (+10-17% improvement)  
**Time**: 2-4 days  
**Impact**: **HIGH** - Novel contribution + Strong results

**Why This is #2**:
- **Uses fine-tuned cross-encoder from #1** (dependency)
- **Novel contribution** for multi-turn RAG (publishable)
- **Progressive refinement** improves ranking quality
- **Combines best techniques** systematically

**Experiment**: `phase6_multistage_2stage`

**Dependencies**: 
- Needs fine-tuned cross-encoder from #1 (or use pre-trained as fallback)
- Can start in parallel if using pre-trained cross-encoder initially

**Command**:
```bash
# After cross-encoder completes, update config to use fine-tuned model:
# "stage2_model": "./models/phase6_cross_encoder_finetuned_ensemble"

python train_multistage_retrieval.py \
    --config experiments/retrieval/phase6_multistage_2stage/config.json \
    --gpu_id 1 \
    --resume
```

---

#### 3. ⭐⭐⭐⭐ **LLM-Based Multi-Query Expansion** (CAN RUN IN PARALLEL)

**Priority**: **#3 - HIGH**  
**Expected nDCG@10**: **0.48-0.51** (+6-12% improvement)  
**Time**: 2-3 days  
**Impact**: **HIGH** - Modern approach, proven effective

**Why This is #3**:
- **Independent of other experiments** (can run in parallel)
- **Query expansion already proven** (0.4981 nDCG@10 on ClapNQ)
- **LLMs generate better variations** than simple expansion
- **Modern, publishable approach**

**Experiment**: `phase6_llm_query_expansion_gpt4_multi`

**Dependencies**: None (can start immediately in parallel)

**Command**:
```bash
python train_llm_query_expansion.py \
    --config experiments/retrieval/phase6_llm_query_expansion_gpt4_multi/config.json \
    --gpu_id 2 \
    --resume
```

---

### 🥈 TIER 2: HIGH VALUE - Run After Tier 1

#### 4. ⭐⭐⭐⭐ **Multi-Stage with Fine-Tuned Cross-Encoder** (COMBINE #1 + #2)

**Priority**: **#4 - HIGH**  
**Expected nDCG@10**: **0.51-0.54** (+12-19% improvement)  
**Time**: 1-2 days (just re-run with updated config)  
**Impact**: **HIGH** - Combines best of #1 and #2

**Why This is #4**:
- **Re-run multi-stage** using fine-tuned cross-encoder from #1
- **Expected to beat Elser** (0.54 nDCG@10)
- **Quick to run** (just evaluation, no training)

**Experiment**: `phase6_multistage_2stage_finetuned` (update config)

**Dependencies**: 
- Needs completed #1 (fine-tuned cross-encoder)
- Needs completed #2 (multi-stage pipeline working)

---

#### 5. ⭐⭐⭐⭐ **Advanced Ensemble of All Techniques** (FINAL COMBINATION)

**Priority**: **#5 - HIGH**  
**Expected nDCG@10**: **0.52-0.55** (+14-21% improvement)  
**Time**: 1-2 days  
**Impact**: **HIGHEST** - Combines all best techniques

**Why This is #5**:
- **Combines**: Cross-encoder + Multi-stage + LLM expansion
- **Expected to beat Elser** and reach top leaderboard
- **Final optimization** before submission

**Experiment**: `phase6_ensemble_all_techniques` (create new)

**Dependencies**: 
- Needs completed #1, #2, #3
- Combines all results

---

### 🥉 TIER 3: OPTIONAL - If Time Permits

#### 6. ⭐⭐⭐ **In-Batch Hard Negative Mining** (IF TIME PERMITS)

**Priority**: **#6 - MEDIUM**  
**Expected nDCG@10**: **0.48-0.51** (+6-12% improvement)  
**Time**: 4-6 days  
**Impact**: **MEDIUM** - High potential but time-consuming

**Why This is #6**:
- **Previous attempts failed** (but technique is proven)
- **Time-consuming** (4-6 days training)
- **Lower priority** than cross-encoder + multi-stage

**Dependencies**: None

---

## 🎯 OPTIMAL EXECUTION ORDER

### Phase 1: Core Techniques (Week 1) - START NOW

**Day 1-5: Run in Parallel**
```
GPU 0: Cross-encoder fine-tuning (#1) - 3-5 days
GPU 1: Multi-stage (with pre-trained cross-encoder) (#2) - 2-4 days  
GPU 2: LLM query expansion (#3) - 2-3 days
```

**Why This Order**:
- **#1 is critical** - Start immediately, highest impact
- **#2 and #3 can run in parallel** - Independent, no conflicts
- **#2 can use pre-trained cross-encoder** initially, then re-run with fine-tuned

### Phase 2: Optimization (Week 2)

**Day 6-7: Re-run with Fine-Tuned Model**
```
GPU 0: Multi-stage with fine-tuned cross-encoder (#4) - 1-2 days
GPU 1: (Free for other experiments)
GPU 2: (Free for other experiments)
```

**Day 8-10: Final Ensemble**
```
GPU 0: Advanced ensemble (#5) - 1-2 days
```

### Phase 3: Optional (Week 3 - If Time)

**Day 11-16: Hard Negatives (Optional)**
```
GPU 0: In-batch hard negative mining (#6) - 4-6 days
```

---

## 📈 Expected Results Progression

### After Phase 1 (Week 1):
- **Cross-encoder**: 0.49-0.52 nDCG@10 ✅
- **Multi-stage**: 0.48-0.51 nDCG@10 ✅
- **LLM expansion**: 0.48-0.51 nDCG@10 ✅
- **Best so far**: **0.52 nDCG@10** (competitive with Elser)

### After Phase 2 (Week 2):
- **Multi-stage (fine-tuned)**: 0.51-0.54 nDCG@10 ✅
- **Final ensemble**: 0.52-0.55 nDCG@10 ✅
- **Best so far**: **0.55 nDCG@10** (BEATS ELSER! 🏆)

### After Phase 3 (Week 3 - Optional):
- **With hard negatives**: 0.53-0.56 nDCG@10 ✅
- **Best so far**: **0.56 nDCG@10** (Top leaderboard position)

---

## 🚀 RECOMMENDED START SEQUENCE

### Immediate (Today):

1. **Start Cross-Encoder (#1)** - GPU 0
   ```bash
   python train_cross_encoder_finetuned.py \
       --config experiments/retrieval/phase6_cross_encoder_finetuned_ensemble/config.json \
       --gpu_id 0 \
       --resume
   ```

2. **Start Multi-Stage (#2)** - GPU 1 (with pre-trained cross-encoder)
   ```bash
   python train_multistage_retrieval.py \
       --config experiments/retrieval/phase6_multistage_2stage/config.json \
       --gpu_id 1 \
       --resume
   ```

3. **Start LLM Expansion (#3)** - GPU 2
   ```bash
   python train_llm_query_expansion.py \
       --config experiments/retrieval/phase6_llm_query_expansion_gpt4_multi/config.json \
       --gpu_id 2 \
       --resume
   ```

### After Cross-Encoder Completes (Day 3-5):

4. **Re-run Multi-Stage (#4)** - GPU 0 (with fine-tuned cross-encoder)
   - Update config: `"stage2_model": "./models/phase6_cross_encoder_finetuned_ensemble"`
   - Re-run evaluation

5. **Create Final Ensemble (#5)** - GPU 0
   - Combine all results from #1, #2, #3, #4

---

## 📊 Decision Matrix

| Experiment | Priority | Impact | Time | Dependencies | Start When |
|------------|----------|--------|------|--------------|------------|
| **Cross-Encoder** | #1 | ⭐⭐⭐⭐⭐ | 3-5d | None | **NOW** |
| **Multi-Stage** | #2 | ⭐⭐⭐⭐⭐ | 2-4d | None* | **NOW** (parallel) |
| **LLM Expansion** | #3 | ⭐⭐⭐⭐ | 2-3d | None | **NOW** (parallel) |
| **Multi-Stage (Fine-tuned)** | #4 | ⭐⭐⭐⭐⭐ | 1-2d | #1 | After #1 |
| **Final Ensemble** | #5 | ⭐⭐⭐⭐⭐ | 1-2d | #1, #2, #3 | After #1, #2, #3 |
| **Hard Negatives** | #6 | ⭐⭐⭐ | 4-6d | None | If time permits |

*Multi-stage can use pre-trained cross-encoder initially, then re-run with fine-tuned

---

## 🎯 Key Insights

### Why This Order Works:

1. **#1 (Cross-encoder) is foundation** - Other experiments benefit from it
2. **#2 and #3 are independent** - Can run in parallel, no conflicts
3. **#4 combines #1 + #2** - Re-run after #1 completes for best results
4. **#5 is final optimization** - Combines everything for maximum nDCG

### Expected Final Score:

- **Minimum (Tier 1 only)**: **0.52 nDCG@10** ✅ (Competitive)
- **Realistic (Tier 1 + #4)**: **0.54 nDCG@10** ✅ (Matches Elser)
- **Best Case (All Tier 1 + #5)**: **0.55 nDCG@10** ✅ (Beats Elser!)
- **With Tier 3**: **0.56 nDCG@10** ✅ (Top leaderboard)

---

## ✅ Action Plan

### Today (Immediate):
1. ✅ Start #1 (Cross-encoder) on GPU 0
2. ✅ Start #2 (Multi-stage) on GPU 1  
3. ✅ Start #3 (LLM expansion) on GPU 2

### This Week:
4. ⏳ Monitor #1, #2, #3 progress
5. ⏳ When #1 completes → Start #4 (Multi-stage with fine-tuned)

### Next Week:
6. ⏳ When #1, #2, #3 complete → Start #5 (Final ensemble)
7. ⏳ Analyze results and prepare for submission

---

**Bottom Line**: Start #1, #2, #3 in parallel TODAY. They're the highest impact experiments and will get you to **0.52+ nDCG@10** (competitive with Elser). Then combine them for **0.55 nDCG@10** (beats Elser)! 🏆

