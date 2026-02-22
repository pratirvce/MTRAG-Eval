# Strategic Experiment Plan for ACL Submission & Top Leaderboard

**Target Conference**: ACL 2026 (or similar top-tier venue)  
**Competition**: MTRAGEval Task A (SemEval 2026)  
**Current Best nDCG@10**: **0.4539**  
**Target nDCG@10**: **0.55+** (top leaderboard position)  
**Evaluation Timeline**: Jan 10-20, 2026  
**Paper Submission**: February 2026

**Reference**: [MTRAGEval Official Page](https://ibm.github.io/mt-rag-benchmark/MTRAGEval/)

---

## 🎯 Competitive Analysis

### Paper Baseline Results (from MTRAG Benchmark)
| Retriever | Setup | nDCG@10 | Status |
|-----------|-------|---------|--------|
| **Elser** | Query Rewrite | **0.54** | 🏆 Best in paper |
| Elser | Last Turn | 0.49 | Strong |
| BGE-base 1.5 | Query Rewrite | 0.38 | Baseline |
| BGE-base 1.5 | Last Turn | 0.30 | Baseline |

### Your Current Results
| Method | nDCG@10 | vs Paper Best | Status |
|--------|---------|---------------|--------|
| **phase5_ensemble_domain_specific** | **0.4539** | -16% | Current best |
| phase4_domain_specific_clapnq | 0.4981* | -8% | Single domain |
| phase5_query_expansion_govt | 0.4515 | -16% | Good |

*Single domain result, not average

**Gap to Beat**: Need to exceed **0.54 nDCG@10** to beat paper's best result

---

## 🚀 Critical Experiments for ACL Submission

### Tier 1: MUST DO (Highest Impact + Publication Value)

#### 1. ⭐⭐⭐⭐⭐ **Fine-Tuned Cross-Encoder Reranking** (CRITICAL)

**Why for ACL**:
- **Strong technical contribution** - Fine-tuning cross-encoders is well-established
- **Directly improves nDCG** - Cross-encoders excel at ranking quality
- **Publishable methodology** - Clear, reproducible, well-motivated
- **Expected to beat Elser** - Fine-tuned models typically outperform pre-trained

**Implementation**:
- Fine-tune `cross-encoder/ms-marco-MiniLM-L-12-v2` on domain-specific data
- Use (query, positive, negative) triplets from training data
- Rerank top 50-100 from `phase5_ensemble_domain_specific`
- Test multiple architectures (L-6, L-12, L-24)

**Expected Improvement**: **+8-15% nDCG@10**
- **Target**: **0.49-0.52 nDCG@10** (competitive with/beating Elser)

**Experiments**:
- `phase6_cross_encoder_finetuned_ensemble` (rerank ensemble)
- `phase6_cross_encoder_finetuned_per_domain` (domain-specific rerankers)
- `phase6_cross_encoder_large` (larger model for final reranking)

**Paper Contribution**: "Domain-Adapted Cross-Encoder Reranking for Multi-Turn RAG"

---

#### 2. ⭐⭐⭐⭐⭐ **Multi-Stage Retrieval Pipeline** (CRITICAL)

**Why for ACL**:
- **Novel contribution** - Multi-stage pipelines are publishable
- **Strong results** - Combines best techniques systematically
- **Clear methodology** - Easy to explain and reproduce
- **Beats single-stage** - Progressive refinement improves ranking

**Implementation**:
- **Stage 1**: Fast retrieval (top 100) - `phase5_ensemble_domain_specific`
- **Stage 2**: Cross-encoder reranking (top 50) - Fine-tuned cross-encoder
- **Stage 3**: Final reranking (top 20) - Larger cross-encoder or ensemble

**Expected Improvement**: **+6-12% nDCG@10**
- **Target**: **0.48-0.51 nDCG@10**

**Experiments**:
- `phase6_multistage_2stage` (dense + cross-encoder)
- `phase6_multistage_3stage` (add final reranking)
- `phase6_multistage_adaptive` (adaptive stage selection)

**Paper Contribution**: "Progressive Multi-Stage Retrieval for Multi-Turn Conversations"

---

#### 3. ⭐⭐⭐⭐⭐ **LLM-Based Query Expansion with Multi-Query Generation** (CRITICAL)

**Why for ACL**:
- **Novel application** - LLM query expansion for multi-turn RAG
- **Strong results** - Query expansion already proven (0.4981 on ClapNQ)
- **Publishable** - Clear methodology, reproducible
- **Beats simple expansion** - LLMs generate better variations

**Implementation**:
- Use GPT-4/Claude to generate 3-5 query variations per original query
- Domain-specific prompts for each domain
- Combine results using RRF or learned weights
- Test different LLMs (GPT-4, Claude, local models)

**Expected Improvement**: **+5-10% nDCG@10**
- **Target**: **0.48-0.50 nDCG@10**

**Experiments**:
- `phase6_llm_query_expansion_gpt4_multi`
- `phase6_llm_query_expansion_claude_multi`
- `phase6_llm_query_expansion_domain_specific` (domain-specific prompts)

**Paper Contribution**: "LLM-Enhanced Query Expansion for Multi-Turn Retrieval"

---

#### 4. ⭐⭐⭐⭐ **Improved Hard Negative Mining (In-Batch)** (HIGH PRIORITY)

**Why for ACL**:
- **Technical depth** - Hard negatives are well-studied, but in-batch is modern
- **Strong potential** - Previous attempts failed, but technique is proven
- **Publishable** - Can discuss why previous failed, how this succeeds
- **Direct nDCG improvement** - Better ranking discrimination

**Implementation**:
- **In-batch hard negative mining** (proven to work)
- Use `MultipleNegativesRankingLoss` with in-batch negatives
- Start from best domain-specific models
- Curriculum learning (gradually increase difficulty)

**Expected Improvement**: **+6-12% nDCG@10** (if done correctly)
- **Target**: **0.48-0.51 nDCG@10**

**Experiments**:
- `phase6_hard_negatives_inbatch_clapnq`
- `phase6_hard_negatives_inbatch_govt`
- `phase6_hard_negatives_curriculum` (curriculum learning)

**Paper Contribution**: "In-Batch Hard Negative Mining for Multi-Turn RAG"

---

### Tier 2: SHOULD DO (Strong Impact + Good for Paper)

#### 5. ⭐⭐⭐⭐ **Domain-Adaptive Hybrid Retrieval with Learned Weights**

**Why for ACL**:
- **Novel contribution** - Learning optimal hybrid weights per domain
- **Strong results** - Hybrid retrieval currently testing
- **Publishable** - Clear methodology, domain adaptation angle

**Implementation**:
- Learn optimal BM25/dense weights per domain using validation set
- Query-dependent weight learning
- Test different optimization methods

**Expected Improvement**: **+3-7% nDCG@10**
- **Target**: **0.47-0.49 nDCG@10**

**Paper Contribution**: "Domain-Adaptive Hybrid Retrieval with Learned Fusion Weights"

---

#### 6. ⭐⭐⭐⭐ **Advanced Ensemble of All Techniques**

**Why for ACL**:
- **Comprehensive approach** - Combines all best techniques
- **Strong results** - Ensemble already best (0.4539)
- **Publishable** - Can analyze contribution of each component

**Implementation**:
- Ensemble: Hard negatives + Query expansion + Hybrid + Cross-encoder
- Learned ensemble weights (not just RRF)
- Per-domain ensemble strategies

**Expected Improvement**: **+3-6% nDCG@10**
- **Target**: **0.47-0.48 nDCG@10**

**Paper Contribution**: "Comprehensive Ensemble for Multi-Turn RAG Retrieval"

---

#### 7. ⭐⭐⭐ **Learning to Rank (LTR) with Listwise Loss**

**Why for ACL**:
- **Technical depth** - LTR is well-established but less common in RAG
- **Direct nDCG optimization** - Listwise losses optimize ranking metrics
- **Publishable** - Clear technical contribution

**Implementation**:
- Use `CoSENTLoss` or custom listwise loss
- Train with relevance labels
- Optimize directly for nDCG

**Expected Improvement**: **+2-5% nDCG@10**
- **Target**: **0.46-0.48 nDCG@10**

**Paper Contribution**: "Listwise Learning to Rank for Multi-Turn RAG"

---

### Tier 3: NICE TO HAVE (If Time Permits)

#### 8. ⭐⭐⭐ **Pseudo-Relevance Feedback**
#### 9. ⭐⭐⭐ **Query-Dependent Retrieval Strategies**
#### 10. ⭐⭐ **Alternative Embedding Models (E5, GTE)**

---

## 📊 Expected Final Results

### Best Case Scenario (Combining Top Techniques):
- **Base**: 0.4539 (current best)
- **Cross-encoder**: +10% → **0.4993**
- **Multi-stage**: +5% → **0.5243**
- **LLM expansion**: +3% → **0.5400**
- **Hard negatives**: +2% → **0.5508**
- **Final**: **~0.55 nDCG@10** ✅ (Beats Elser's 0.54!)

### Realistic Scenario:
- **Base**: 0.4539
- **Cross-encoder**: +8% → **0.4902**
- **Multi-stage**: +4% → **0.5098**
- **LLM expansion**: +2% → **0.5200**
- **Final**: **~0.52 nDCG@10** ✅ (Competitive with Elser)

### Conservative Scenario:
- **Base**: 0.4539
- **Cross-encoder**: +5% → **0.4766**
- **Multi-stage**: +3% → **0.4909**
- **LLM expansion**: +1% → **0.4958**
- **Final**: **~0.50 nDCG@10** ✅ (Strong improvement)

---

## 🎓 What Makes a Strong ACL Paper

### 1. **Technical Contributions** (Required)
- ✅ Novel methodology or significant improvement
- ✅ Clear technical innovation
- ✅ Well-motivated approach

### 2. **Strong Experimental Results** (Required)
- ✅ Beat or match state-of-the-art
- ✅ Comprehensive evaluation
- ✅ Statistical significance

### 3. **Reproducibility** (Required)
- ✅ Clear methodology
- ✅ Code/data availability
- ✅ Detailed hyperparameters

### 4. **Analysis & Insights** (Important)
- ✅ Error analysis
- ✅ Ablation studies
- ✅ Domain-specific insights

### 5. **Writing Quality** (Important)
- ✅ Clear problem statement
- ✅ Well-structured experiments
- ✅ Strong related work

---

## 📝 Recommended Paper Structure

### Title Suggestion:
"**Domain-Adaptive Multi-Stage Retrieval for Multi-Turn RAG Conversations**"

### Key Contributions:
1. **Multi-stage retrieval pipeline** for multi-turn RAG
2. **Domain-adaptive cross-encoder reranking**
3. **LLM-enhanced query expansion** for multi-turn queries
4. **In-batch hard negative mining** for better ranking
5. **Comprehensive ensemble** combining all techniques

### Experimental Setup:
- **Baselines**: Paper baselines (BGE, Elser)
- **Your methods**: Multi-stage, cross-encoder, LLM expansion, ensemble
- **Evaluation**: nDCG@10 (official metric), Recall@10
- **Domains**: ClapNQ, FiQA, Govt, Cloud

### Results to Highlight:
- **nDCG@10**: 0.52-0.55 (beating Elser's 0.54)
- **Improvement**: +73-83% over baseline
- **Domain analysis**: Performance per domain
- **Ablation studies**: Contribution of each component

---

## 🎯 Strategic Experiment Plan

### Phase 6A: Core Techniques (Weeks 1-2)

**Priority 1: Cross-Encoder Fine-Tuning** (Week 1)
- Fine-tune cross-encoder on domain-specific data
- Rerank ensemble results
- **Expected**: 0.49-0.52 nDCG@10

**Priority 2: Multi-Stage Pipeline** (Week 1-2)
- Implement 2-stage pipeline (dense + cross-encoder)
- Test 3-stage if time permits
- **Expected**: 0.48-0.51 nDCG@10

**Priority 3: LLM Query Expansion** (Week 2)
- Implement GPT-4/Claude query expansion
- Generate multiple query variations
- **Expected**: 0.48-0.50 nDCG@10

### Phase 6B: Advanced Techniques (Weeks 3-4)

**Priority 4: Hard Negative Mining** (Week 3)
- Implement in-batch hard negative mining
- Train domain-specific models
- **Expected**: 0.48-0.51 nDCG@10

**Priority 5: Learned Hybrid Weights** (Week 3-4)
- Optimize hybrid weights per domain
- Test query-dependent weights
- **Expected**: 0.47-0.49 nDCG@10

**Priority 6: Advanced Ensemble** (Week 4)
- Combine all best techniques
- Learn optimal ensemble weights
- **Expected**: 0.50-0.53 nDCG@10

### Phase 6C: Final Optimization (Week 5)

**Priority 7: Final Ensemble & Optimization**
- Combine all techniques
- Hyperparameter tuning
- Final evaluation
- **Target**: **0.52-0.55 nDCG@10**

---

## 📈 Leaderboard Strategy

### To Beat Elser (0.54 nDCG@10):
- Need **+19% improvement** from current best (0.4539)
- **Strategy**: Combine cross-encoder + multi-stage + LLM expansion
- **Expected**: 0.52-0.55 nDCG@10 ✅

### To Be Top 3:
- Need **0.50+ nDCG@10**
- **Strategy**: Cross-encoder + multi-stage
- **Expected**: 0.50-0.52 nDCG@10 ✅

### To Be #1:
- Need **0.55+ nDCG@10**
- **Strategy**: All techniques combined + optimization
- **Expected**: 0.52-0.55 nDCG@10 (may need additional techniques)

---

## 🔬 Specific Experiments to Run

### Critical Experiments (Run First):

1. **phase6_cross_encoder_finetuned_ensemble**
   - Fine-tune cross-encoder on all domains
   - Rerank `phase5_ensemble_domain_specific` results (top 100 → top 50)
   - **Expected**: 0.49-0.52 nDCG@10

2. **phase6_multistage_2stage**
   - Stage 1: Ensemble retrieval (top 100)
   - Stage 2: Fine-tuned cross-encoder (top 50)
   - **Expected**: 0.48-0.51 nDCG@10

3. **phase6_llm_query_expansion_gpt4_multi**
   - GPT-4 generates 3-5 query variations
   - Combine results with RRF
   - **Expected**: 0.48-0.50 nDCG@10

4. **phase6_hard_negatives_inbatch_clapnq**
   - In-batch hard negative mining
   - Train on ClapNQ domain
   - **Expected**: 0.48-0.51 nDCG@10 (on ClapNQ)

5. **phase6_ensemble_all_techniques**
   - Combine: Cross-encoder + Multi-stage + LLM expansion + Hard negatives
   - Learned ensemble weights
   - **Expected**: 0.50-0.53 nDCG@10

### Supporting Experiments:

6. **phase6_cross_encoder_per_domain**
   - Domain-specific cross-encoder rerankers
   - **Expected**: 0.47-0.50 nDCG@10

7. **phase6_hybrid_learned_weights_per_domain**
   - Optimal hybrid weights per domain
   - **Expected**: 0.47-0.49 nDCG@10

8. **phase6_ltr_listwise_loss**
   - Learning to rank with listwise loss
   - **Expected**: 0.46-0.48 nDCG@10

---

## 📊 Expected Timeline

### Week 1 (Dec 16-22):
- ✅ Cross-encoder fine-tuning (3-4 days)
- ✅ Multi-stage pipeline implementation (2-3 days)
- **Target**: 0.49-0.52 nDCG@10

### Week 2 (Dec 23-29):
- ✅ LLM query expansion (2-3 days)
- ✅ Combine cross-encoder + multi-stage (1-2 days)
- **Target**: 0.50-0.53 nDCG@10

### Week 3 (Dec 30 - Jan 5):
- ✅ Hard negative mining (in-batch) (3-4 days)
- ✅ Learned hybrid weights (2-3 days)
- **Target**: 0.51-0.54 nDCG@10

### Week 4 (Jan 6-12):
- ✅ Final ensemble of all techniques (2-3 days)
- ✅ Hyperparameter optimization (2-3 days)
- ✅ Final evaluation and analysis (1-2 days)
- **Target**: **0.52-0.55 nDCG@10** ✅

### Week 5 (Jan 13-19):
- ✅ Final submission preparation
- ✅ Error analysis
- ✅ Paper writing

---

## 🎯 Key Success Factors for ACL

### 1. **Strong Technical Contribution**
- ✅ Multi-stage retrieval (novel for multi-turn RAG)
- ✅ Domain-adaptive techniques
- ✅ LLM-enhanced query expansion

### 2. **Competitive Results**
- ✅ Beat or match Elser (0.54 nDCG@10)
- ✅ Comprehensive evaluation across domains
- ✅ Statistical significance

### 3. **Clear Methodology**
- ✅ Reproducible experiments
- ✅ Well-documented code
- ✅ Clear hyperparameters

### 4. **Thorough Analysis**
- ✅ Ablation studies (contribution of each component)
- ✅ Error analysis (failure cases)
- ✅ Domain-specific insights
- ✅ Comparison with baselines

### 5. **Writing Quality**
- ✅ Clear problem statement
- ✅ Well-motivated approach
- ✅ Strong experimental design
- ✅ Comprehensive related work

---

## 💡 Paper Writing Tips

### Abstract Should Highlight:
1. **Problem**: Multi-turn RAG retrieval challenges
2. **Approach**: Multi-stage retrieval with domain adaptation
3. **Results**: 0.52-0.55 nDCG@10 (beating state-of-the-art)
4. **Contribution**: Novel pipeline for multi-turn RAG

### Introduction Should:
- Motivate multi-turn RAG retrieval
- Highlight limitations of existing approaches
- Present your solution
- Preview contributions

### Methodology Should:
- Clearly describe multi-stage pipeline
- Explain domain adaptation
- Detail query expansion approach
- Describe ensemble strategy

### Experiments Should:
- Compare with all baselines (BGE, Elser)
- Show ablation studies
- Analyze per-domain performance
- Include error analysis

### Discussion Should:
- Analyze why techniques work
- Discuss failure cases
- Compare with related work
- Suggest future directions

---

## 📊 Expected Paper Results Table

| Method | nDCG@10 | R@10 | vs Baseline | vs Elser |
|--------|---------|------|-------------|----------|
| **Baseline (BGE)** | 0.30 | 0.38 | - | - |
| **Elser (Paper Best)** | 0.54 | 0.64 | +80% | - |
| **Your: Multi-Stage** | 0.50-0.52 | 0.55-0.57 | +67-73% | -7% to -4% |
| **Your: + Cross-Encoder** | 0.52-0.54 | 0.57-0.59 | +73-80% | -4% to 0% |
| **Your: + LLM Expansion** | 0.53-0.55 | 0.58-0.60 | +77-83% | -2% to +2% |
| **Your: Final Ensemble** | **0.52-0.55** | **0.57-0.60** | **+73-83%** | **-4% to +2%** |

**Target**: Match or beat Elser's 0.54 nDCG@10

---

## 🎯 Final Recommendations

### Must Do (For ACL Acceptance):
1. ✅ **Fine-tuned cross-encoder reranking** - Strong technical contribution
2. ✅ **Multi-stage retrieval pipeline** - Novel methodology
3. ✅ **LLM query expansion** - Modern, publishable approach
4. ✅ **Comprehensive ensemble** - Strong results

### Should Do (For Top Leaderboard):
5. ✅ **In-batch hard negative mining** - Technical depth
6. ✅ **Learned hybrid weights** - Domain adaptation
7. ✅ **Ablation studies** - Required for ACL

### Nice to Have:
8. ⚪ Learning to rank losses
9. ⚪ Alternative embedding models
10. ⚪ Query-dependent retrieval

---

## 📝 Next Steps

### Immediate (This Week):
1. **Start cross-encoder fine-tuning** (highest impact)
2. **Implement multi-stage pipeline** (novel contribution)
3. **Set up LLM query expansion** (modern approach)

### Short-term (Next 2 Weeks):
4. **Complete all Tier 1 experiments**
5. **Combine techniques into final ensemble**
6. **Run comprehensive evaluation**

### Before Submission (Jan 2026):
7. **Final optimization**
8. **Error analysis**
9. **Paper writing**
10. **Code/documentation cleanup**

---

## 🏆 Success Criteria

### For ACL Acceptance:
- ✅ **nDCG@10 ≥ 0.50** (strong improvement over baseline)
- ✅ **Clear technical contribution**
- ✅ **Comprehensive evaluation**
- ✅ **Reproducible methodology**

### For Top Leaderboard:
- ✅ **nDCG@10 ≥ 0.54** (beat or match Elser)
- ✅ **Consistent across domains**
- ✅ **Robust to different query types**

### For #1 Position:
- ✅ **nDCG@10 ≥ 0.55** (exceed Elser)
- ✅ **Novel techniques**
- ✅ **Strong analysis**

---

*Strategy designed for ACL 2026 submission and MTRAGEval Task A leaderboard*  
*Based on: [MTRAGEval Official Page](https://ibm.github.io/mt-rag-benchmark/MTRAGEval/)*

