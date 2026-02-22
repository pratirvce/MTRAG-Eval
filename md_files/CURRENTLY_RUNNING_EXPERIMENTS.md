# Currently Running Experiments - Complete Explanation

**Last Updated**: 2025-12-16 22:20  
**Total Running**: 5-6 experiments across 6 GPUs  
**Status**: Most GPUs fully utilized

---

## 📊 Overview

We are currently running **5 experiments in parallel**:
- **2 Phase 6 Priority Experiments** (for ACL submission - highest impact on nDCG)
- **3 Hybrid Retrieval Experiments** (resumed on free GPUs)
- **1 Phase 6 Experiment**: Stopped (needs fix) - GPU 2 available

All experiments support **checkpointing and resume** - they can be paused and resumed without losing progress.

---

## 🚀 Phase 6 Priority Experiments (ACL Submission)

These are the **critical experiments** designed to beat Elser's 0.54 nDCG@10 and achieve top leaderboard position for ACL submission.

---

### 1. ⭐⭐⭐⭐⭐ Cross-Encoder Fine-Tuning (Priority #1)

**Experiment Name**: `phase6_cross_encoder_finetuned_ensemble`  
**GPU**: 0  
**Status**: ✅ Running  
**Utilization**: 30-95% (varies with training phase)

#### What This Experiment Does:
Fine-tunes a cross-encoder model on domain-specific data to improve ranking quality. Cross-encoders are the gold standard for ranking because they can see both query and document together, allowing for precise relevance scoring.

#### Technical Details:
- **Base Model**: `cross-encoder/ms-marco-MiniLM-L-12-v2`
- **Training Data**: All 4 domains (clapnq, fiqa, govt, cloud) combined
- **Training Method**: Binary classification (relevant vs. non-relevant pairs)
- **Epochs**: 3
- **Batch Size**: 16
- **Learning Rate**: 2e-5

#### Why This is Critical:
- **Highest single-technique impact** on nDCG improvement
- **Directly optimizes ranking quality** (nDCG metric)
- **Elser likely uses cross-encoder** - we need to match/beat it
- **Foundation for other experiments** (multi-stage will use this)

#### Expected Results:
- **nDCG@10**: **0.49-0.52** (+8-15% improvement from current 0.4539)
- **Time**: 3-5 days
- **Impact**: If successful, could reach **0.52 nDCG@10** (competitive with Elser)

#### Output Location:
- **Model**: `models/phase6_cross_encoder_finetuned_ensemble/`
- **Checkpoints**: `models/phase6_cross_encoder_finetuned_ensemble/checkpoints/`
- **Log**: `experiments/retrieval/phase6_cross_encoder_finetuned_ensemble/training.log`
- **Results**: `experiments/retrieval/phase6_cross_encoder_finetuned_ensemble/results.json` (after evaluation)

#### Monitor:
```bash
tail -f experiments/retrieval/phase6_cross_encoder_finetuned_ensemble/training.log
```

---

### 2. ⭐⭐⭐⭐⭐ Multi-Stage Retrieval Pipeline (Priority #2)

**Experiment Name**: `phase6_multistage_2stage`  
**GPU**: 1  
**Status**: ✅ Running  
**Utilization**: 100%

#### What This Experiment Does:
Implements a progressive multi-stage retrieval pipeline that refines results at each stage:
- **Stage 1**: Fast dense retrieval (top 100) using ensemble of domain-specific models
- **Stage 2**: Cross-encoder reranking (top 50) using pre-trained cross-encoder
- **Stage 3**: (Optional) Final reranking with larger model

#### Technical Details:
- **Stage 1 Model**: `./models/domain_specific_clapnq` (ensemble of domain models)
- **Stage 2 Model**: `cross-encoder/ms-marco-MiniLM-L-12-v2` (pre-trained, will use fine-tuned later)
- **Stage 1 Top-K**: 100 documents
- **Stage 2 Top-K**: 50 documents (after reranking)
- **Domains**: All 4 domains (clapnq, fiqa, govt, cloud)

#### Why This is Critical:
- **Novel contribution** for multi-turn RAG (publishable)
- **Progressive refinement** improves ranking quality at each stage
- **Combines best techniques** systematically
- **Clear methodology** (easy to explain in paper)

#### Expected Results:
- **nDCG@10**: **0.48-0.51** (+6-12% improvement)
- **With fine-tuned cross-encoder**: **0.51-0.54** (+12-19% improvement)
- **Time**: 2-4 days
- **Impact**: Novel approach that could beat Elser

#### Output Location:
- **Checkpoints**: `experiments/retrieval/phase6_multistage_2stage/checkpoints/`
- **Log**: `experiments/retrieval/phase6_multistage_2stage/training.log`
- **Results**: `experiments/retrieval/phase6_multistage_2stage/results.json`

#### Monitor:
```bash
tail -f experiments/retrieval/phase6_multistage_2stage/training.log
```

#### Next Step:
After cross-encoder (#1) completes, this will be re-run with the fine-tuned cross-encoder for even better results.

---

### 3. ⭐⭐⭐⭐ LLM-Based Multi-Query Expansion (Priority #3)

**Experiment Name**: `phase6_llm_query_expansion_gpt4_multi`  
**GPU**: 2  
**Status**: ⚠️ **Stopped** (Error encountered)  
**Utilization**: 0% (GPU free)

#### What This Experiment Does:
Uses GPT-4 to generate multiple query variations for each original query, then combines retrieval results from all variations using Reciprocal Rank Fusion (RRF).

#### Technical Details:
- **LLM**: GPT-4 (or Claude as fallback)
- **Query Variations**: 3-5 variations per original query
- **Base Model**: `./models/domain_specific_clapnq` (ensemble of domain models)
- **Combination Method**: Reciprocal Rank Fusion (RRF)
- **Domains**: All 4 domains

#### Why This is Critical:
- **Query expansion already proven effective** (0.4981 nDCG@10 on ClapNQ in previous experiments)
- **LLMs generate better variations** than simple synonym expansion
- **Modern, publishable approach** (LLMs are hot topic)
- **Independent of other experiments** (can run in parallel)

#### Expected Results:
- **nDCG@10**: **0.48-0.51** (+6-12% improvement)
- **With domain-specific prompts**: **0.50-0.53** nDCG@10
- **Time**: 2-3 days
- **Impact**: Modern approach that's publishable

#### Output Location:
- **Checkpoints**: `experiments/retrieval/phase6_llm_query_expansion_gpt4_multi/checkpoints/`
- **Log**: `experiments/retrieval/phase6_llm_query_expansion_gpt4_multi/training.log`
- **Results**: `experiments/retrieval/phase6_llm_query_expansion_gpt4_multi/results.json`

#### Monitor:
```bash
tail -f experiments/retrieval/phase6_llm_query_expansion_gpt4_multi/training.log
```

#### Note:
⚠️ **This experiment encountered an error and stopped**. The error appears to be in the dense retrieval search function. GPU 2 is now free and can be used for other experiments or this can be fixed and restarted.

---

## 🔄 Hybrid Retrieval Experiments (Resumed)

These experiments test different alpha values (weights) for combining BM25 and dense retrieval. They were paused to free up GPUs for Phase 6 priority experiments, then resumed on free GPUs.

---

### 4. Hybrid Retrieval - Govt Domain (Alpha 0.7)

**Experiment Name**: `phase5_hybrid_govt_alpha0.7`  
**GPU**: 3  
**Status**: ✅ Running  
**Utilization**: 100%

#### What This Experiment Does:
Combines BM25 (lexical) and dense (semantic) retrieval for the Govt domain with alpha=0.7, meaning 70% weight on dense retrieval and 30% on BM25.

#### Technical Details:
- **Domain**: Govt
- **Alpha**: 0.7 (70% dense, 30% BM25)
- **Dense Model**: Domain-specific fine-tuned model
- **BM25**: Using `rank_bm25` library
- **Combination**: Weighted score fusion

#### Why This is Running:
- Tests optimal hybrid weights for Govt domain
- Part of comprehensive evaluation of hybrid retrieval
- Resumed on free GPU to maximize GPU utilization

#### Expected Results:
- **nDCG@10**: Varies by domain and alpha value
- **Time**: 1-2 days
- **Impact**: Helps determine optimal hybrid weights per domain

#### Output Location:
- **Log**: `experiments/retrieval/phase5_hybrid_govt_alpha0.7/training.log`
- **Results**: `experiments/retrieval/phase5_hybrid_govt_alpha0.7/results.json`

---

### 5. Hybrid Retrieval - Multi Domain (Alpha 0.3)

**Experiment Name**: `phase5_hybrid_multi_alpha0.3`  
**GPU**: 4  
**Status**: ✅ Running  
**Utilization**: 100%

#### What This Experiment Does:
Combines BM25 and dense retrieval across all domains with alpha=0.3, meaning 30% weight on dense retrieval and 70% on BM25.

#### Technical Details:
- **Domain**: Multi (all domains combined)
- **Alpha**: 0.3 (30% dense, 70% BM25)
- **Dense Model**: Multi-domain fine-tuned model
- **BM25**: Using `rank_bm25` library

#### Why This is Running:
- Tests lower alpha value (more BM25, less dense)
- Part of comprehensive evaluation
- Resumed on free GPU

#### Expected Results:
- **nDCG@10**: Varies by alpha value
- **Time**: 1-2 days

---

### 6. Hybrid Retrieval - Multi Domain (Alpha 0.5)

**Experiment Name**: `phase5_hybrid_multi_alpha0.5`  
**GPU**: 5  
**Status**: ✅ Running  
**Utilization**: 100%

#### What This Experiment Does:
Combines BM25 and dense retrieval across all domains with alpha=0.5, meaning equal weight (50% dense, 50% BM25).

#### Technical Details:
- **Domain**: Multi (all domains combined)
- **Alpha**: 0.5 (50% dense, 50% BM25 - balanced)
- **Dense Model**: Multi-domain fine-tuned model
- **BM25**: Using `rank_bm25` library

#### Why This is Running:
- Tests balanced hybrid approach
- Part of comprehensive evaluation
- Resumed on free GPU

#### Expected Results:
- **nDCG@10**: Varies by alpha value
- **Time**: 1-2 days

---

## 📊 GPU Allocation Summary

| GPU | Experiment | Type | Priority | Utilization | Status |
|-----|------------|------|----------|-------------|--------|
| 0 | Cross-encoder fine-tuning | Phase 6 | #1 (Highest) | 96% | ✅ Running |
| 1 | Multi-stage retrieval | Phase 6 | #2 (High) | 100% | ✅ Running |
| 2 | LLM query expansion | Phase 6 | #3 (High) | 0% | ⚠️ Check status |
| 3 | Hybrid govt alpha0.7 | Hybrid | Medium | 100% | ✅ Running |
| 4 | Hybrid multi alpha0.3 | Hybrid | Medium | 100% | ✅ Running |
| 5 | Hybrid multi alpha0.5 | Hybrid | Medium | 100% | ✅ Running |

**5-6 experiments actively running!** 🎯

---

## 🎯 Expected Final Results

### After Phase 6 Experiments Complete:

| Experiment | Expected nDCG@10 | Improvement |
|------------|------------------|-------------|
| Cross-encoder | 0.49-0.52 | +8-15% |
| Multi-stage | 0.48-0.51 | +6-12% |
| LLM expansion | 0.48-0.51 | +6-12% |
| **Best Individual** | **0.52** | **+14.6%** |

### After Combining All Techniques:

- **Multi-stage (with fine-tuned cross-encoder)**: 0.51-0.54 nDCG@10
- **Final Ensemble**: **0.52-0.55 nDCG@10** ✅
- **Target**: Beat Elser's 0.54 nDCG@10

---

## ⏳ Timeline

### Week 1 (Current):
- **Day 1-5**: Cross-encoder fine-tuning (GPU 0)
- **Day 1-4**: Multi-stage retrieval (GPU 1)
- **Day 1-3**: LLM query expansion (GPU 2)
- **Day 1-2**: Hybrid experiments (GPUs 3, 4, 5)

### Week 2:
- **After #1 completes**: Re-run multi-stage with fine-tuned cross-encoder
- **After all complete**: Create final ensemble combining all techniques
- **Resume remaining hybrid experiments**: 6 more hybrid experiments pending

---

## 🔧 Management Commands

### Check All Running Experiments
```bash
ps aux | grep -E "train_cross_encoder|train_multistage|train_llm_query|train_hybrid" | grep -v grep
```

### Monitor GPU Usage
```bash
watch -n 1 nvidia-smi
```

### Monitor Specific Experiment
```bash
# Phase 6 experiments
tail -f experiments/retrieval/phase6_*/training.log

# Hybrid experiments
tail -f experiments/retrieval/phase5_hybrid_*/training.log
```

### Check Experiment Status
```bash
# Check if experiment completed
ls experiments/retrieval/<experiment_name>/results.json

# View results
cat experiments/retrieval/<experiment_name>/results.json | jq '.average'
```

### Pause an Experiment
```bash
# Find PID
ps aux | grep <experiment_name> | grep -v grep

# Pause gracefully (saves checkpoint)
kill -TERM <PID>
```

### Resume an Experiment
```bash
# All experiments support --resume flag
python train_<script>.py --config <config_path> --gpu_id <gpu> --resume
```

---

## 📝 Experiment Dependencies

### Phase 6 Experiments:
- **Cross-encoder (#1)**: Independent, can run alone
- **Multi-stage (#2)**: Can use pre-trained cross-encoder initially, then re-run with fine-tuned
- **LLM expansion (#3)**: Independent, can run alone

### After Cross-Encoder Completes:
- **Multi-stage will be re-run** with fine-tuned cross-encoder for better results
- **Final ensemble** will combine all three techniques

---

## 🎓 Paper Contribution

These running experiments will provide:

1. **Domain-Adaptive Cross-Encoder Reranking** - Technical depth, directly optimizes nDCG
2. **Multi-Stage Retrieval Pipeline** - Novel contribution for multi-turn RAG
3. **LLM-Enhanced Query Expansion** - Modern approach, publishable
4. **Hybrid Retrieval Analysis** - Comprehensive evaluation of different alpha values

All designed to achieve **0.52-0.55 nDCG@10** and beat Elser's 0.54 for top leaderboard position! 🏆

---

## ⏸️ Pending Experiments

### Remaining Hybrid Experiments (6):
1. `phase5_hybrid_clapnq_alpha0.3`
2. `phase5_hybrid_clapnq_alpha0.5`
3. `phase5_hybrid_clapnq_alpha0.7`
4. `phase5_hybrid_govt_alpha0.3`
5. `phase5_hybrid_govt_alpha0.5`
6. `phase5_hybrid_multi_alpha0.7`

These will automatically resume when Phase 6 experiments complete and free up GPUs.

---

## ✅ Summary

- **Total Running**: 5 experiments
- **Phase 6 Priority**: 2 experiments running (GPUs 0, 1), 1 stopped (GPU 2)
- **Hybrid Experiments**: 3 experiments (GPUs 3, 4, 5)
- **GPU Utilization**: 5 GPUs in use, 1 GPU free (GPU 2)
- **All experiments support**: Checkpointing, pause/resume, graceful shutdown

**5 experiments actively running!** 🚀  
**GPU 2 is free** - can be used to restart LLM expansion (after fix) or run other experiments.

---

*For detailed strategy and expected results, see:*
- `ACL_SUBMISSION_STRATEGY.md` - Complete strategy for ACL submission
- `EXPERIMENT_PRIORITY_ORDER.md` - Priority ranking and execution order
- `PHASE6_START_GUIDE.md` - How to manage Phase 6 experiments

