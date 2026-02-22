# 🚀 Novel Experiments Implementation Guide

## ✅ Completed Implementations (2/8)

### 1. CATR: Conversation-Aware Temporal Retrieval
- **Files Created:**
  - `train_catr_temporal_retrieval.py` - Core implementation with temporal attention
  - `train_catr_temporal_retrieval_tier1.py` - Tier1 wrapper
- **Status:** ✅ Complete and added to auto-runner
- **Task A Compliant:** ✅ Yes (retrieval only, no text generation)

### 2. UQ-Ret: Uncertainty-Quantified Retrieval
- **Files Created:**
  - `train_uq_uncertainty_retrieval.py` - Core implementation with uncertainty estimation
  - `train_uq_uncertainty_retrieval_tier1.py` - Tier1 wrapper
- **Status:** ✅ Complete and added to auto-runner
- **Task A Compliant:** ✅ Yes (retrieval with confidence scores, no text generation)

## ⏳ Remaining Implementations (6/8)

### 3. Diff-Ret: Differentiable Retrieval
**Priority:** Highest (Expected: 0.58-0.68 nDCG@10)

**Core Novel Contribution:**
- Gumbel-Softmax for differentiable top-k selection
- End-to-end optimization directly for nDCG
- Gradient flow through discrete retrieval

**Implementation Template:**
```python
# Key components needed:
1. DifferentiableTopK class using Gumbel-Softmax
2. Direct nDCG loss function (differentiable)
3. End-to-end training loop
4. Integration with BGE-large base model
```

**Files to Create:**
- `train_diff_retrieval.py`
- `train_diff_retrieval_tier1.py`

**Task A Compliance:** ✅ Retrieval optimization only, no text generation

---

### 4. Meta-Ret: Cross-Domain Meta-Learning
**Priority:** High (Expected: 0.48-0.58 nDCG@10)

**Core Novel Contribution:**
- MAML (Model-Agnostic Meta-Learning) for retrieval
- Few-shot domain adaptation
- Cross-domain knowledge transfer

**Implementation Template:**
```python
# Key components needed:
1. MAML meta-learner class
2. Support/query set splitting per domain
3. Inner loop (domain adaptation) and outer loop (meta-update)
4. Prototype-based few-shot matching
```

**Files to Create:**
- `train_meta_retrieval.py`
- `train_meta_retrieval_tier1.py`

**Task A Compliance:** ✅ Retrieval model training, no text generation

---

### 5. Causal-Ret: Causal Retrieval
**Priority:** High (Expected: 0.50-0.60 nDCG@10)

**Core Novel Contribution:**
- Causal graph construction for query-document relationships
- Intervention analysis (what if we change query terms?)
- Counterfactual reasoning for retrieval

**Implementation Template:**
```python
# Key components needed:
1. CausalGraphBuilder - constructs query-document causal graph
2. InterventionModule - simulates query modifications
3. CounterfactualGenerator - generates counterfactual queries
4. CausalRegularizer - uses causal structure for regularization
```

**Files to Create:**
- `train_causal_retrieval.py` (improved version of existing)
- `train_causal_retrieval_tier1.py`

**Task A Compliance:** ✅ Retrieval with causal reasoning, no text generation

---

### 6. GEP-Ret: Graph-Enhanced Retrieval
**Priority:** High (Expected: 0.55-0.65 nDCG@10)

**Core Novel Contribution:**
- Entity extraction and knowledge graph construction
- GNN-based entity propagation
- Hybrid fusion of graph signals with dense retrieval

**Implementation Template:**
```python
# Key components needed:
1. EntityExtractor - extracts entities from documents
2. KnowledgeGraphBuilder - builds domain-specific graphs
3. GNNEncoder - GCN/GraphSAGE for graph encoding
4. HybridScorer - combines dense + graph scores
```

**Files to Create:**
- `train_gep_graph_retrieval.py` (improved version)
- `train_gep_graph_retrieval_tier1.py`

**Task A Compliance:** ✅ Retrieval with graph signals, no text generation

---

### 7. X-Ret: Explainable Retrieval
**Priority:** Medium (Expected: 0.52-0.62 nDCG@10)

**Core Novel Contribution:**
- Attention-based rationales (highlighting, not text generation)
- Contrastive explanations (why doc A > doc B)
- Faithfulness metrics for explanations

**Implementation Template:**
```python
# Key components needed:
1. AttentionExtractor - extracts query-document attention
2. RationaleHighlighter - highlights key matching spans (no text gen)
3. ContrastiveExplainer - explains ranking differences
4. FaithfulnessScorer - verifies explanation quality
```

**Files to Create:**
- `train_x_explainable_retrieval.py`
- `train_x_explainable_retrieval_tier1.py`

**Task A Compliance:** ✅ Attention highlighting only, no text generation

---

### 8. CLCF-Ret: Contrastive Learning on Flows
**Priority:** Medium (Expected: 0.53-0.63 nDCG@10)

**Core Novel Contribution:**
- Contrastive learning on conversation flows (not just pairs)
- Turn-level representations
- Flow-aware negative sampling

**Implementation Template:**
```python
# Key components needed:
1. FlowEncoder - encodes conversation as sequence
2. TurnEncoder - encodes individual turns
3. FlowContrastiveLoss - contrastive learning on flows
4. CoherenceRegularizer - ensures conversation coherence
```

**Files to Create:**
- `train_clcf_contrastive_flow.py`
- `train_clcf_contrastive_flow_tier1.py`

**Task A Compliance:** ✅ Retrieval training, no text generation

---

## 📋 Implementation Strategy

### Phase 1: Core Structure (All 6 experiments)
1. Create base implementation files with:
   - Standard imports and setup
   - Core novel component classes
   - Evaluation framework integration
   - Task A compliance verification

### Phase 2: Tier1 Wrappers (All 6 experiments)
1. Create tier1 wrapper scripts following the pattern:
   - Argument parsing
   - Config file creation
   - Error handling
   - Integration with auto-runner

### Phase 3: Auto-Runner Integration
1. Add all 6 experiments to `PENDING_EXPERIMENTS` in `auto_start_fixed_experiments.py`
2. Set appropriate priorities (1 for high-impact, 2 for medium)
3. Add Task A compliance notes

### Phase 4: Testing & Refinement
1. Test each experiment on a small subset
2. Verify Task A compliance
3. Refine implementations based on results

---

## 🎯 Quick Start Implementation

For each remaining experiment, follow this pattern:

1. **Copy template from CATR or UQ-Ret:**
   ```bash
   cp train_catr_temporal_retrieval.py train_<experiment_name>.py
   ```

2. **Modify core components:**
   - Replace temporal attention with experiment-specific components
   - Update evaluation logic
   - Ensure Task A compliance

3. **Create tier1 wrapper:**
   ```bash
   cp train_catr_temporal_retrieval_tier1.py train_<experiment_name>_tier1.py
   ```

4. **Add to auto-runner:**
   - Add entry to `PENDING_EXPERIMENTS` list
   - Set priority and description

---

## ✅ Task A Compliance Checklist

For each experiment, verify:
- [ ] No text generation (no LLM text output)
- [ ] Only retrieval and ranking operations
- [ ] Query preprocessing allowed (expansion, rewriting)
- [ ] Cross-encoders allowed (scoring only)
- [ ] Confidence/uncertainty scores allowed
- [ ] Attention/explanation highlighting allowed (no text generation)
- [ ] All outputs are document IDs and scores

---

## 📊 Expected Performance Summary

| Experiment | Expected nDCG@10 | Novelty | Priority |
|------------|-----------------|---------|----------|
| CATR | 0.55-0.65 | ⭐⭐⭐⭐⭐ | 1 |
| UQ-Ret | 0.50-0.60 | ⭐⭐⭐⭐⭐ | 1 |
| Diff-Ret | 0.58-0.68 | ⭐⭐⭐⭐⭐ | 1 |
| Meta-Ret | 0.48-0.58 | ⭐⭐⭐⭐⭐ | 1 |
| Causal-Ret | 0.50-0.60 | ⭐⭐⭐⭐⭐ | 1 |
| GEP-Ret | 0.55-0.65 | ⭐⭐⭐⭐ | 2 |
| X-Ret | 0.52-0.62 | ⭐⭐⭐⭐ | 2 |
| CLCF-Ret | 0.53-0.63 | ⭐⭐⭐⭐ | 2 |

---

## 🚀 Next Steps

1. **Immediate:** Implement Diff-Ret (highest expected performance)
2. **Next:** Implement GEP-Ret and Meta-Ret (strong theoretical contributions)
3. **Then:** Implement Causal-Ret, X-Ret, CLCF-Ret
4. **Finally:** Add all to auto-runner and test

---

## 📝 Notes

- All experiments build on BGE-large as base model
- All experiments are Task A compliant (retrieval only)
- All experiments have clear novel contributions for Tier 1 publication
- Implementations can be extended iteratively based on results

