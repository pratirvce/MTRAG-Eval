# Experiment Queue - Next Experiments to Run When GPUs Free Up

**Last Updated**: 2025-12-16 23:55  
**Status**: Active queue for automatic experiment scheduling

---

## 🎯 **Queue Strategy**

This document lists experiments in **priority order** to start automatically when GPUs become available. Experiments are prioritized by:
1. **Expected Impact** (nDCG improvement)
2. **Novelty/Publication Value**
3. **Implementation Status** (ready vs needs implementation)
4. **Dependencies** (none preferred)

---

## 📊 **Currently Running Experiments**

| GPU | Experiment | Status | Expected Completion |
|-----|------------|--------|---------------------|
| **GPU 0** | Cross-Encoder Evaluation | 🔄 Running | ~2-3 hours |
| **GPU 1** | (Unknown/Idle) | 🔄 Running | Unknown |
| **GPU 2** | Cross-Attention Query-Document | 🔄 Running | ~4-8 hours |
| **GPU 3** | Iterative Refinement Retrieval | 🔄 Running | ~2-4 hours |
| **GPU 4** | Ensemble Domain-Specific | 🔄 Running | ~1-2 hours |
| **GPU 5** | Multi-Stage 2-Stage Fine-Tuned | 🔄 Running | ~1-2 hours |

**Next GPU Available**: GPU 4/5 (expected ~1-2 hours), then GPU 0/3 (~2-4 hours)

---

## 🚀 **Priority Queue: Next Experiments to Run**

### **Tier 1: High Priority Novel Experiments (Ready to Implement)**

#### **1. ⭐⭐⭐⭐ Hierarchical Multi-Granularity Retrieval** (HIGHEST PRIORITY)

**Status**: ⏸️ **Not yet implemented** - Ready to implement  
**Priority**: **HIGHEST** - Next Tier 1 experiment after Cross-Attention  
**Expected Impact**: 0.49-0.52 nDCG@10 (+8-14% improvement)

**Why Next**:
- ✅ Pure retrieval - Multiple retrieval granularities
- ✅ No external dependencies - Uses existing models
- ✅ Good impact - +8-14% nDCG improvement
- ✅ Medium complexity - Faster to implement than Contrastive Learning
- ✅ Novel approach - Multi-granularity retrieval strategy

**Implementation Details**:
- **Script**: `train_hierarchical_multigranularity_retrieval.py` (needs creation)
- **Config**: `experiments/retrieval/phase8_hierarchical_multigranularity/config.json` (needs creation)
- **Time to Implement**: 3-4 days
- **Time to Run**: 3-4 hours per domain (12-16 hours total)
- **GPU Memory**: ~2-4 GB

**How It Works**:
1. Retrieve at **sentence level** (fine-grained)
2. Retrieve at **paragraph level** (medium-grained)
3. Retrieve at **document level** (coarse-grained)
4. Combine results using **Reciprocal Rank Fusion (RRF)**
5. Optional: Rerank with cross-encoder

**Publication Value**: ⭐⭐⭐⭐
- **Novel**: "Hierarchical Multi-Granularity Retrieval for Multi-Turn RAG"
- **Technical Depth**: Multi-granularity retrieval, RRF combination
- **Good Contribution**: Novel retrieval strategy

**Action**: **Implement when GPU becomes available** (GPU 1, 4, or 5)

---

#### **2. ⭐⭐⭐⭐ Contrastive Learning with Conversation-Document Pairs** (HIGH PRIORITY)

**Status**: ⏸️ **Not yet implemented** - Ready to implement  
**Priority**: **HIGH** - High impact but requires training  
**Expected Impact**: 0.49-0.52 nDCG@10 (+8-15% improvement)

**Why Next**:
- ✅ Pure retrieval - Trains retrieval models
- ✅ No external dependencies - Uses existing models
- ✅ High impact - +8-15% nDCG improvement
- ⚠️ Requires training - Longer time commitment (5-7 days)
- ✅ Novel training approach - Contrastive learning

**Implementation Details**:
- **Script**: `train_contrastive_conversation_doc.py` (needs creation)
- **Config**: `experiments/retrieval/phase8_contrastive_learning/config.json` (needs creation)
- **Time to Implement**: 5-7 days (includes training)
- **Training Time**: 1-2 days (fine-tuning on conversation-document pairs)
- **Evaluation Time**: 3-4 hours per domain (12-16 hours total)
- **GPU Memory**: ~4-6 GB

**How It Works**:
1. Create **positive pairs**: (conversation, relevant_document)
2. Create **negative pairs**: (conversation, irrelevant_document)
3. Train model with **contrastive loss** (MultipleNegativesRankingLoss)
4. Fine-tune on conversation-document pairs
5. Retrieve using fine-tuned model

**Publication Value**: ⭐⭐⭐⭐
- **Novel**: "Contrastive Learning for Conversation-Document Retrieval"
- **Technical Depth**: Training methodology, contrastive learning
- **Good Contribution**: Novel training approach

**Action**: **Implement after Hierarchical Multi-Granularity** (or in parallel if multiple GPUs available)

---

### **Tier 2: Medium Priority Experiments (Ready to Run)**

#### **3. ⭐⭐⭐ LLM-Powered Query Rewriting with Reasoning** (MEDIUM PRIORITY)

**Status**: ⏸️ **Partially implemented** - Needs completion  
**Priority**: **MEDIUM** - High impact but requires LLM API  
**Expected Impact**: 0.51-0.54 nDCG@10 (+12-20% improvement)

**Why Next**:
- ✅ Improves retrieval - Better queries = better retrieval
- ⚠️ Requires LLM API - Needs GPT-4 or similar (API costs)
- ✅ High impact - +12-20% nDCG improvement
- ⚠️ Slower execution - API calls add latency
- ✅ Fast to implement - API-based, quick to set up

**Implementation Details**:
- **Script**: `train_llm_query_rewriting_reasoning.py` (needs creation)
- **Config**: `experiments/retrieval/phase8_llm_query_rewriting/config.json` (needs creation)
- **Time to Implement**: 2-3 days
- **Time to Run**: 4-6 hours per domain (16-24 hours total, slower due to API calls)
- **GPU Memory**: ~2-4 GB (mostly for retrieval, not LLM)

**How It Works**:
1. **LLM reasons** about what information is needed from conversation
2. **LLM rewrites query** based on reasoning
3. **Retrieve** with rewritten query
4. Optional: Combine original and rewritten query results

**Publication Value**: ⭐⭐⭐⭐
- **Novel**: "Reasoning-Based Query Rewriting for Multi-Turn RAG"
- **Technical Depth**: LLM reasoning, query rewriting
- **Good Contribution**: Combines LLM reasoning with retrieval

**Action**: **Implement if time permits** (after Tier 1 experiments)

---

#### **4. ⭐⭐⭐ Pseudo-Relevance Feedback with LLM Expansion** (MEDIUM PRIORITY)

**Status**: ⏸️ **Partially implemented** - Similar to existing LLM expansion  
**Priority**: **MEDIUM** - Good impact but requires LLM API  
**Expected Impact**: 0.50-0.53 nDCG@10 (+10-16% improvement)

**Why Next**:
- ✅ Improves retrieval - Query expansion helps recall
- ⚠️ Requires LLM API - Needs GPT-4 or similar (API costs)
- ✅ Good impact - +10-16% nDCG improvement
- ⚠️ Two-stage - Initial retrieval + expansion (slower)
- ✅ Similar to existing - Can build on `train_llm_query_expansion.py`

**Implementation Details**:
- **Script**: `train_pseudo_relevance_feedback_llm.py` (needs creation)
- **Config**: `experiments/retrieval/phase8_pseudo_relevance_feedback/config.json` (needs creation)
- **Time to Implement**: 2-3 days
- **Time to Run**: 4-6 hours per domain (16-24 hours total)
- **GPU Memory**: ~2-4 GB

**How It Works**:
1. **Initial retrieval** with original query
2. **Extract key information** from top results
3. **LLM expands query** based on key information
4. **Retrieve again** with expanded query
5. **Combine results** using RRF

**Publication Value**: ⭐⭐⭐
- **Novel**: "Pseudo-Relevance Feedback with LLM for Multi-Turn RAG"
- **Technical Depth**: Combines traditional IR with modern LLMs
- **Good Contribution**: Query expansion with feedback

**Action**: **Implement if time permits** (after Tier 1 experiments)

---

### **Tier 3: Lower Priority Experiments (Optional)**

#### **5. ⭐⭐⭐ Adaptive Retrieval Strategy Selection** (LOWER PRIORITY)

**Status**: ⏸️ **Not yet implemented**  
**Priority**: **LOWER** - Experimental approach  
**Expected Impact**: 0.48-0.51 nDCG@10 (+6-12% improvement)

**Why Lower Priority**:
- ✅ Pure retrieval - Strategy selection for retrieval
- ⚠️ Requires classification - Need to classify query types
- ⚠️ Complex - Multiple strategies to implement
- ⚠️ Lower impact - +6-12% improvement (less than others)

**Action**: **Implement only if time permits** (after all Tier 1 & 2 experiments)

---

#### **6. ⭐⭐ Graph-Based Knowledge Retrieval** (EXPERIMENTAL)

**Status**: ⏸️ **Not yet implemented**  
**Priority**: **LOWEST** - Very experimental  
**Expected Impact**: 0.48-0.51 nDCG@10 (+5-12% improvement)

**Why Lowest Priority**:
- ✅ Retrieval-focused - Graph traversal for retrieval
- ⚠️ Very experimental - Unproven approach
- ⚠️ Complex - Requires graph construction
- ⚠️ Lower impact - +5-12% improvement

**Action**: **Implement only if time permits** (experimental, low priority)

---

## 📋 **Implementation Checklist**

### **When GPU 1 Becomes Available (~20-30 minutes)**:
- [ ] **Start**: Hierarchical Multi-Granularity Retrieval
  - Implement `train_hierarchical_multigranularity_retrieval.py`
  - Create config file
  - Start on GPU 1

### **When GPU 4/5 Become Available (~1-2 hours)**:
- [ ] **Start**: Contrastive Learning (if Hierarchical not started yet)
  - OR start in parallel if multiple GPUs available

### **When All Tier 1 Experiments Complete**:
- [ ] **Consider**: LLM Query Rewriting (if API access available)
- [ ] **Consider**: Pseudo-Relevance Feedback (if API access available)

---

## 🎯 **Recommended Execution Order**

### **Phase 1: Immediate (Next 1-2 Hours)**
1. ✅ **Cross-Attention Query-Document** - Already running (GPU 2)
2. ⏳ **Hierarchical Multi-Granularity** - Start when GPU 1/4/5 free up

### **Phase 2: Short-Term (Next 1-2 Days)**
3. ⏳ **Contrastive Learning** - Start when GPU available (after Hierarchical or in parallel)
4. ⏳ **LLM Query Rewriting** - If time permits and API access available

### **Phase 3: Long-Term (If Time Permits)**
5. ⏳ **Pseudo-Relevance Feedback** - If time permits
6. ⏳ **Adaptive Strategy Selection** - Optional
7. ⏳ **Graph-Based Retrieval** - Experimental, optional

---

## 📊 **Expected Combined Results**

### **Scenario 1: All Tier 1 Experiments Complete**
- **Base**: 0.4539
- **Cross-Attention**: +12% → **0.5084**
- **Hierarchical**: +10% → **0.5592**
- **Contrastive**: +10% → **0.6151**
- **Final**: **~0.55-0.62 nDCG@10** ✅ **EXCEPTIONAL!**

### **Scenario 2: Top 2 Tier 1 Experiments Complete**
- **Base**: 0.4539
- **Cross-Attention**: +10% → **0.4993**
- **Hierarchical**: +8% → **0.5392**
- **Final**: **~0.50-0.54 nDCG@10** ✅ **STRONG!**

---

## 🔄 **Automatic Scheduling Script**

A script has been created to automatically start the next experiment in the queue when a GPU becomes available.

**Script Location**: `start_next_experiment_in_queue.sh` ✅ **CREATED**

**How to Use**:
```bash
# Run the queue manager to start next experiments
./start_next_experiment_in_queue.sh

# Or run it periodically (every 30 minutes) using cron:
# */30 * * * * cd /path/to/mt-rag-benchmark && ./start_next_experiment_in_queue.sh
```

**What It Does**:
1. **Monitors GPU availability** (checks `nvidia-smi` for free GPUs)
2. **Checks experiment queue** (this document)
3. **Starts next priority experiment** when GPU free
4. **Logs experiment start** and provides monitoring commands

**Current Queue Priority**:
1. Hierarchical Multi-Granularity Retrieval (when implemented)
2. Contrastive Learning (when implemented)

---

## ✅ **Action Items**

### **Immediate**:
1. **Implement Hierarchical Multi-Granularity Retrieval** (Priority 1)
   - Script: `train_hierarchical_multigranularity_retrieval.py`
   - Config: `experiments/retrieval/phase8_hierarchical_multigranularity/config.json`
   - Start when GPU 1/4/5 free up

### **Short-Term**:
2. **Implement Contrastive Learning** (Priority 2)
   - Script: `train_contrastive_conversation_doc.py`
   - Config: `experiments/retrieval/phase8_contrastive_learning/config.json`
   - Start when GPU available (after Hierarchical or in parallel)

### **Long-Term**:
3. **Consider LLM Query Rewriting** (if API access available)
4. **Consider Pseudo-Relevance Feedback** (if time permits)

---

## 📝 **Notes**

- **Cross-Attention**: Currently running on GPU 2 (just started, ~4-8 hours)
- **Next Available GPU**: GPU 1 (~20-30 minutes), then GPU 4/5 (~1-2 hours)
- **Recommended Next**: Hierarchical Multi-Granularity Retrieval
- **Queue Updates**: This document should be updated as experiments start/complete

---

**Queue is ready! Next experiment to start: Hierarchical Multi-Granularity Retrieval** 🚀

