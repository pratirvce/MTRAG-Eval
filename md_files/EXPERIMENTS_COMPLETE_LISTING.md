# MTRAGEval Task A Experiments - Complete Listing

**Last Updated:** 2025-12-20T21:04:41.238053

---

## 📊 Summary Statistics

- **Completed:** 31
- **Running:** 6
- **Pending:** 55
- **Failed:** 8

### Novelty Distribution

- **High Novelty:** 31
- **Medium Novelty:** 28
- **Low Novelty:** 33

---

## ✅ Completed Experiments

| Experiment Name | Novelty | nDCG@10 | Description |
|----------------|---------|---------|-------------|
| `tier1_adversarial_curriculum` | Medium | 0.4442 | Adversarial Hard Negative Mining with Curriculum Learning |
| `task_a_hybrid_dynamic_fusion` | Medium | 0.2423 | Hybrid Dense-Sparse with Learned Dynamic Fusion (Task A - Expected: 0.72-0.82 nDCG@10) |
| `task_a_multistage_hierarchical_retrieval` | Medium | 0.2394 | Multi-Stage Hierarchical Retrieval with Cross-Encoder Reranking (Task A - Expected: 0.65-0.75 nDCG@1... |
| `tier1_cross_attention_query_document_fixed` | Medium | 0.2200 | Cross-Attention Query-Document Interaction (fixes applied) |
| `tier1_cross_attention_query_document` | Medium | 0.2200 | Cross-Attention Query-Document Interaction (CUDA device fix applied) |
| `best_paper_hierarchical_routing_fixed` | Medium | 0.2166 | Hierarchical Multi-Stage with Learned Routing (indentation fix applied) |
| `tier1_llm_distillation` | Low | 0.1986 | Knowledge Distillation from Large Language Models |
| `task_a_cross_encoder_conversation_context` | Low | 0.1949 | Transformer-Based Cross-Encoder with Full Conversation Context (Task A - Expected: 0.68-0.78 nDCG@10... |
| `tier1_hybrid_lexical_semantic` | Medium | 0.1796 | Hybrid Lexical-Semantic Retrieval with Learned Fusion (Expected: 0.52-0.56 nDCG@10) |
| `tier1_momentum_contrastive` | Medium | 0.1796 | Contrastive Learning with Momentum Encoder (Expected: 0.52-0.56 nDCG@10) |
| `tier1_nas_retrieval` | Medium | 0.1796 | Neural Architecture Search for Retrieval (Expected: 0.53-0.57 nDCG@10) |
| `tier1_continual_learning` | Medium | 0.1796 | Continual Learning for Retrieval (Expected: 0.52-0.56 nDCG@10) |
| `tier1_explainable_retrieval` | High | 0.1796 | Explainable Retrieval with Attention Visualization (Expected: 0.52-0.56 nDCG@10) |
| `tier1_adversarial_robustness` | Medium | 0.1796 | Adversarial Robustness for Retrieval (Expected: 0.52-0.56 nDCG@10) |
| `tier1_synthetic_data` | Low | 0.1796 | Synthetic Data Generation for Retrieval (Expected: 0.53-0.57 nDCG@10) |
| `task_a_differentiable_end_to_end` | High | 0.1796 | Differentiable End-to-End Retrieval Pipeline (Task A - Expected: 0.70-0.80 nDCG@10) |
| `tier1_causal_inference` | High | 0.1796 | Causal Inference for Multi-Turn Retrieval (Expected: 0.54-0.58 nDCG@10) |
| `tier1_learned_indices` | High | 0.1796 | Learned Indices for Neural Retrieval (Expected: 0.53-0.57 nDCG@10) |
| `tier1_rlhf_retrieval` | High | 0.1796 | RLHF for Retrieval (Expected: 0.54-0.58 nDCG@10) |
| `tier1_differentiable_retrieval` | High | 0.1796 | Differentiable End-to-End Retrieval Pipeline (Expected: 0.53-0.57 nDCG@10) |
| `tier1_graph_enhanced_reranking` | Medium | 0.1796 | Graph-Enhanced Adaptive Re-Ranking (GEAR) (Expected: 0.52-0.56 nDCG@10) |
| `tier1_uncertainty_aware` | High | 0.1796 | Uncertainty-Aware Retrieval with Confidence Calibration (Expected: 0.50-0.54 nDCG@10) |
| `tier1_foundation_distillation` | Low | 0.1796 | Foundation Model Distillation for Retrieval (Expected: 0.54-0.58 nDCG@10) |
| `tier1_query_decomposition` | Low | 0.1796 | Retrieval with Learned Query Decomposition (Expected: 0.50-0.54 nDCG@10) |
| `tier1_knowledge_graph` | Low | 0.1796 | Knowledge Graph-Enhanced Retrieval (Expected: 0.53-0.57 nDCG@10) |
| `tier1_curriculum_contrastive` | Medium | 0.1796 | Curriculum Contrastive Learning with Query Rewriting (Expected: 0.52-0.56 nDCG@10) |
| `tier1_mixture_experts` | Low | 0.1796 | Mixture of Retrieval Experts (MoRE) (Expected: 0.52-0.56 nDCG@10) |
| `tier1_retrieval_as_generation` | High | 0.1796 | Retrieval as Generation (RAG-Retrieval) (Expected: 0.54-0.58 nDCG@10) |
| `tier1_multi_turn_state_tracking` | Medium | 0.1796 | Multi-Turn Conversation State Tracking with Retrieval (Expected: 0.51-0.55 nDCG@10) |
| `tier1_graph_aware_retrieval` | Low | 0.1738 | Conversation Graph-Aware Retrieval |
| `tier1_multitask_retrieval` | Low | 0.0185 | Multi-Task Learning for Retrieval |

---

## 🔄 Running Experiments

| Experiment Name | Novelty | Status | Description |
|----------------|---------|--------|-------------|
| `tier1_causal_inference_fixed` | High | Running | Causal Inference for Multi-Turn Retrieval (evaluation bug fixed) |
| `tier1_curriculum_contrastive_fixed` | Medium | Running | Curriculum Contrastive Learning (evaluation bug fixed - now uses trained model) |
| `tier1_neural_ndcg` | Low | Running | Direct nDCG Optimization with NeuralNDCG (Expected: 0.53-0.57 nDCG@10) |
| `tier1_rl_adaptive_retrieval` | Low | Running | Reinforcement Learning for Adaptive Retrieval (Expected: 0.52-0.56 nDCG@10) |
| `tier1_temporal_attention` | Medium | Running | Temporal Attention for Conversation History (Expected: 0.51-0.55 nDCG@10) |
| `tier1_uncertainty_aware_fixed` | High | Running | Uncertainty-Aware Retrieval (evaluation bug fixed) |

---

## ⏳ Pending Experiments

| Experiment Name | Novelty | Priority | Description |
|----------------|---------|----------|-------------|
| `task_a_enhanced_multistage_llm_scoring` | Medium | 1 | Enhanced Multi-Stage Pipeline with LLM-Based Relevance Scoring (Task A - Expected: 0.85-0.92 nDCG@10... |
| `task_a_advanced_contrastive_hierarchical` | Medium | 1 | Advanced Contrastive Learning with Hierarchical Loss (Task A - Expected: 0.70-0.80 nDCG@10) |
| `task_a_model_scaling_bge_v2` | Low | 1 | Model Scaling Improvements (BGE-v2-large, longer training, domain-specific) (Task A - Expected: 0.61... |
| `task_a_specialized_ensemble_meta_learner` | High | 1 | Specialized Ensemble with Meta-Learner (Task A - Expected: 0.75-0.85 nDCG@10) |
| `task_a_llm_query_expansion_constrained` | Medium | 1 | LLM-Powered Query Expansion (Task A - Expected: 0.65-0.75 nDCG@10) ⚠️ CONSTRAINT: LLM generates quer... |
| `task_a_multistage_hierarchical_retrieval` | Medium | 1 | Multi-Stage Hierarchical Retrieval with Cross-Encoder Reranking (Task A - Expected: 0.65-0.75 nDCG@1... |
| `task_a_hybrid_dynamic_fusion` | Medium | 1 | Hybrid Dense-Sparse with Learned Dynamic Fusion (Task A - Expected: 0.72-0.82 nDCG@10) |
| `task_a_optimized_hybrid_reranking` | Medium | 1 | Optimized Hybrid Search with Reranking - All Improvements Combined (Task A - Expected: 0.60-0.75 nDC... |
| `task_a_semantic_chunking_retrieval` | Low | 1 | Semantic Chunking Retrieval: Boundary-Aware Chunking (Task A - Expected: +0.03-0.06 nDCG@10). Chunks... |
| `novel_catr_temporal_retrieval` | High | 1 | CATR: Conversation-Aware Temporal Retrieval - Models temporal dynamics of multi-turn conversations (... |
| `novel_uq_uncertainty_retrieval` | High | 1 | UQ-Ret: Uncertainty-Quantified Retrieval - Provides calibrated confidence scores for retrieval (Task... |
| `novel_diff_retrieval` | High | 1 | Diff-Ret: Differentiable Retrieval - Fully differentiable pipeline with end-to-end nDCG optimization... |
| `novel_meta_retrieval` | High | 1 | Meta-Ret: Cross-Domain Meta-Learning - Fast adaptation to new domains with few examples (Task A - Ex... |
| `novel_causal_retrieval` | High | 1 | Causal-Ret: Causal Retrieval - Uses causal inference to understand query-document relationships (Tas... |
| `task_a_cross_encoder_conversation_context` | Low | 1 | Transformer-Based Cross-Encoder with Full Conversation Context (Task A - Expected: 0.68-0.78 nDCG@10... |
| `tier1_curriculum_contrastive` | Medium | 1 | Curriculum Contrastive Learning with Query Rewriting (Expected: 0.52-0.56 nDCG@10) |
| `tier1_causal_inference` | High | 1 | Causal Inference for Multi-Turn Retrieval (Expected: 0.54-0.58 nDCG@10) |
| `tier1_foundation_distillation` | Low | 1 | Foundation Model Distillation for Retrieval (Expected: 0.54-0.58 nDCG@10) |
| `tier1_learned_indices` | High | 1 | Learned Indices for Neural Retrieval (Expected: 0.53-0.57 nDCG@10) |
| `task_a_openrag_retrieval_only` | Low | 1 | OpenRAG-Style Retrieval-Only End-to-End Optimization (Task A - Expected: 0.55-0.60 nDCG@10) ✅ Uses r... |
| `task_a_metarag_retrieval` | Low | 1 | MetaRAG: Retrieval Quality Self-Assessment (Task A - Expected: 0.58-0.63 nDCG@10) ✅ Evaluates retrie... |
| `task_a_late_chunking_retrieval` | Low | 1 | Late Chunking Retrieval: Embed full document first, then chunk (preserves global context) (Task A - ... |
| `task_a_colbert_retrieval` | Low | 1 | ColBERT-Style Multi-Vector Retrieval: Token-level embeddings with MaxSim scoring (Task A - Expected:... |
| `task_a_bge_v2_large_asymmetric` | Low | 1 | BGE-v2-Large with Asymmetric Encoding: State-of-the-art model (560M params) with query/document prom... |
| `task_a_enhanced_graph_construction` | Low | 2 | Enhanced Graph Construction and GNN Architectures (Task A - Expected: 0.60-0.75 nDCG@10) |
| `task_a_rlhf_retrieval_constrained` | High | 2 | RLHF for Retrieval (Task A - Expected: 0.72-0.82 nDCG@10) ⚠️ CONSTRAINT: RLHF optimizes retrieval me... |
| `task_a_advanced_training_techniques` | Medium | 2 | Advanced Training Techniques (hard negatives, curriculum, adversarial) (Task A - Expected: 0.59-0.63... |
| `task_a_enhanced_differentiable_pipeline` | High | 2 | Enhanced Differentiable End-to-End Pipeline (Task A - Expected: 0.70-0.85 nDCG@10) |
| `tier1_mmr_retrieval` | Low | 2 | MMR (Maximal Marginal Relevance) Retrieval: Balances relevance and diversity (Task A - Expected: +0.... |
| `tier1_hyde_retrieval` | Low | 2 | HyDE (Hypothetical Document Embeddings) Retrieval: Generates hypothetical documents for retrieval (T... |
| `tier1_cg_rag_retrieval` | Low | 2 | CG-RAG (Contextualized Graph RAG) Retrieval: Builds knowledge graph, contextualizes based on query/c... |
| `tier1_gfm_rag_retrieval` | Low | 2 | GFM-RAG (Graph Foundation Model for RAG) Retrieval: Uses foundation models for graph encoding, appli... |
| `novel_gep_graph_retrieval` | High | 2 | GEP-Ret: Graph-Enhanced Retrieval - Uses GNNs to propagate entity relationships (Task A - Expected: ... |
| `novel_x_explainable_retrieval` | High | 2 | X-Ret: Explainable Retrieval - Provides attention-based explanations (Task A - Expected: 0.52-0.62 n... |
| `novel_clcf_contrastive_flow` | High | 2 | CLCF-Ret: Contrastive Learning on Conversation Flows - Applies contrastive learning to conversation ... |
| `task_a_llm_powered_retrieval` | Low | 2 | LLM-Powered Retrieval with In-Context Learning (Task A - Expected: 0.70-0.80 nDCG@10) |
| `task_a_differentiable_end_to_end` | High | 2 | Differentiable End-to-End Retrieval Pipeline (Task A - Expected: 0.70-0.80 nDCG@10) |
| `task_a_graph_neural_retrieval` | High | 2 | Conversation-Aware Graph Neural Retrieval (Task A - Expected: 0.68-0.78 nDCG@10) |
| `task_a_domain_specific_ensemble` | Medium | 2 | Ensemble of Domain-Specific Retrievers (Task A - Expected: 0.75-0.85 nDCG@10) |
| `tier1_multi_turn_state_tracking` | Medium | 2 | Multi-Turn Conversation State Tracking with Retrieval (Expected: 0.51-0.55 nDCG@10) |
| `tier1_mixture_experts` | Low | 2 | Mixture of Retrieval Experts (MoRE) (Expected: 0.52-0.56 nDCG@10) |
| `tier1_graph_enhanced_reranking` | Medium | 2 | Graph-Enhanced Adaptive Re-Ranking (GEAR) (Expected: 0.52-0.56 nDCG@10) |
| `tier1_differentiable_retrieval` | High | 2 | Differentiable End-to-End Retrieval Pipeline (Expected: 0.53-0.57 nDCG@10) |
| `tier1_retrieval_as_generation` | High | 2 | Retrieval as Generation (RAG-Retrieval) (Expected: 0.54-0.58 nDCG@10) |
| `tier1_rlhf_retrieval` | High | 2 | RLHF for Retrieval (Expected: 0.54-0.58 nDCG@10) |
| `tier1_synthetic_data` | Low | 2 | Synthetic Data Generation for Retrieval (Expected: 0.53-0.57 nDCG@10) |
| `tier1_knowledge_graph` | Low | 2 | Knowledge Graph-Enhanced Retrieval (Expected: 0.53-0.57 nDCG@10) |
| `task_a_transform_retrieval` | High | 2 | Transform Retrieval for Textual Entailment (Task A - Expected: 0.56-0.61 nDCG@10) ✅ Pure retrieval o... |
| `task_a_chainrag_retrieval` | Low | 2 | ChainRAG: Multi-Hop Retrieval (Retrieval-Only) (Task A - Expected: 0.57-0.62 nDCG@10) ✅ Retrieval re... |
| `task_a_probing_rag` | Low | 2 | Probing-RAG: Selective Document Retrieval (Task A - Expected: 0.53-0.58 nDCG@10) ✅ Uses hidden state... |
| `tier1_uncertainty_aware` | High | 3 | Uncertainty-Aware Retrieval with Confidence Calibration (Expected: 0.50-0.54 nDCG@10) |
| `tier1_query_decomposition` | Low | 3 | Retrieval with Learned Query Decomposition (Expected: 0.50-0.54 nDCG@10) |
| `tier1_meta_learning` | High | 4 | Meta-Learning for Fast Domain Adaptation (Expected: 0.51-0.55 nDCG@10) |
| `tier1_cross_domain_transfer` | Medium | 4 | Cross-Domain Transfer Learning with Domain Adversarial Training (Expected: 0.51-0.55 nDCG@10) |
| `tier1_prompt_based_retrieval` | Low | 4 | Prompt-Based Retrieval with In-Context Learning (Expected: 0.52-0.56 nDCG@10) |

---

## ❌ Failed Experiments

| Experiment Name | Novelty | Error |
|----------------|---------|-------|
| `tier1_enhanced_contrastive_hardnegatives_fixed` | Low | CUDA out of memory. Tried to allocate 2.00 MiB. GPU 0 has a total capacity of... |
| `tier1_learning_to_rank_listwise_fixed` | Low | Evaluator retriever is not set! |
| `tier1_enhanced_contrastive` | Low | Processed 2 negative embeddings but expected 1 |
| `tier1_counterfactual_augmentation` | Medium | element 0 of tensors does not require grad and does not have a grad_fn |
| `tier1_mixture_experts_fixed` | Low | Process died without completing |
| `tier1_semantic_drift` | Medium | element 0 of tensors does not require grad and does not have a grad_fn |
| `tier1_multi_granularity` | Medium | element 0 of tensors does not require grad and does not have a grad_fn |
| `tier1_memory_augmented` | Low | Process died without completing |

---

## 📈 Top Performers (nDCG@10)

| Rank | Experiment Name | Novelty | nDCG@10 |
|------|----------------|---------|---------|
| 1 | `tier1_adversarial_curriculum` | Medium | 0.4442 |
| 2 | `task_a_hybrid_dynamic_fusion` | Medium | 0.2423 |
| 3 | `task_a_multistage_hierarchical_retrieval` | Medium | 0.2394 |
| 4 | `tier1_cross_attention_query_document_fixed` | Medium | 0.2200 |
| 5 | `tier1_cross_attention_query_document` | Medium | 0.2200 |
| 6 | `best_paper_hierarchical_routing_fixed` | Medium | 0.2166 |
| 7 | `tier1_llm_distillation` | Low | 0.1986 |
| 8 | `task_a_cross_encoder_conversation_context` | Low | 0.1949 |
| 9 | `tier1_hybrid_lexical_semantic` | Medium | 0.1796 |
| 10 | `tier1_momentum_contrastive` | Medium | 0.1796 |
| 11 | `tier1_nas_retrieval` | Medium | 0.1796 |
| 12 | `tier1_continual_learning` | Medium | 0.1796 |
| 13 | `tier1_explainable_retrieval` | High | 0.1796 |
| 14 | `tier1_adversarial_robustness` | Medium | 0.1796 |
| 15 | `tier1_synthetic_data` | Low | 0.1796 |

---

## 📝 Notes

- **Novelty Levels:**
  - **High:** Novel architectures, new paradigms, unexplored directions
  - **Medium:** Combinations of techniques, enhancements of existing methods
  - **Low:** Standard techniques, baseline improvements, hyperparameter tuning

- **Priority:** Lower number = higher priority for execution

- **nDCG@10:** Primary evaluation metric for Task A (Retrieval Only)
