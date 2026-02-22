# All Experiments: Status, Novelty, and Explanations

**Generated:** generate_all_experiments_report.py

This report consolidates experiment status, novelty, and descriptions into a single list.

## Summary

- **Completed**: 34
- **Running**: 6
- **Pending**: 52
- **Stopped**: 129
- **Failed**: 10
- **Run-Unknown**: 52

## Experiment List (All)

| Experiment | Status | Novel Idea | Novelty | nDCG@10 | Description |
|-----------|--------|------------|---------|--------|-------------|
| `best_paper_hierarchical_routing_fixed` | completed | Yes | Medium | 0.2166 | Hierarchical Multi-Stage with Learned Routing (indentation fix applied) |
| `ensemble_domain_specific_models` | completed | No |  |  | No description found in source files. |
| `ensemble_weighted_fusion` | completed | No |  |  | No description found in source files. |
| `reranking_clapnq` | completed | No |  |  | No description found in source files. |
| `task_a_cross_encoder_conversation_context` | completed | No | Low | 0.1949 | Transformer-Based Cross-Encoder with Full Conversation Context (Task A - Expected: 0.68-0.78 nDCG@10... |
| `task_a_differentiable_end_to_end` | completed | Yes | High | 0.1796 | Differentiable End-to-End Retrieval Pipeline (Task A - Expected: 0.70-0.80 nDCG@10) |
| `task_a_hybrid_dynamic_fusion` | completed | Yes | Medium | 0.2423 | Hybrid Dense-Sparse with Learned Dynamic Fusion (Task A - Expected: 0.72-0.82 nDCG@10) |
| `task_a_multistage_hierarchical_retrieval` | completed | Yes | Medium | 0.2394 | Multi-Stage Hierarchical Retrieval with Cross-Encoder Reranking (Task A - Expected: 0.65-0.75 nDCG@1... |
| `tier1_adversarial_curriculum` | completed | Yes | Medium | 0.4442 | Adversarial Hard Negative Mining with Curriculum Learning |
| `tier1_adversarial_robustness` | completed | Yes | Medium | 0.1796 | Adversarial Robustness for Retrieval (Expected: 0.52-0.56 nDCG@10) |
| `tier1_causal_inference` | completed | Yes | High | 0.1796 | Causal Inference for Multi-Turn Retrieval (Expected: 0.54-0.58 nDCG@10) |
| `tier1_continual_learning` | completed | Yes | Medium | 0.1796 | Continual Learning for Retrieval (Expected: 0.52-0.56 nDCG@10) |
| `tier1_cross_attention_query_document` | completed | Yes | Medium | 0.2200 | Cross-Attention Query-Document Interaction (CUDA device fix applied) |
| `tier1_cross_attention_query_document_fixed` | completed | Yes | Medium | 0.2200 | Cross-Attention Query-Document Interaction (fixes applied) |
| `tier1_curriculum_contrastive` | completed | Yes | Medium | 0.1796 | Curriculum Contrastive Learning with Query Rewriting (Expected: 0.52-0.56 nDCG@10) |
| `tier1_differentiable_retrieval` | completed | Yes | High | 0.1796 | Differentiable End-to-End Retrieval Pipeline (Expected: 0.53-0.57 nDCG@10) |
| `tier1_explainable_retrieval` | completed | Yes | High | 0.1796 | Explainable Retrieval with Attention Visualization (Expected: 0.52-0.56 nDCG@10) |
| `tier1_foundation_distillation` | completed | No | Low | 0.1796 | Foundation Model Distillation for Retrieval (Expected: 0.54-0.58 nDCG@10) |
| `tier1_graph_aware_retrieval` | completed | No | Low | 0.1738 | Conversation Graph-Aware Retrieval |
| `tier1_graph_enhanced_reranking` | completed | Yes | Medium | 0.1796 | Graph-Enhanced Adaptive Re-Ranking (GEAR) (Expected: 0.52-0.56 nDCG@10) |
| `tier1_hybrid_lexical_semantic` | completed | Yes | Medium | 0.1796 | Hybrid Lexical-Semantic Retrieval with Learned Fusion (Expected: 0.52-0.56 nDCG@10) |
| `tier1_knowledge_graph` | completed | No | Low | 0.1796 | Knowledge Graph-Enhanced Retrieval (Expected: 0.53-0.57 nDCG@10) |
| `tier1_learned_indices` | completed | Yes | High | 0.1796 | Learned Indices for Neural Retrieval (Expected: 0.53-0.57 nDCG@10) |
| `tier1_llm_distillation` | completed | No | Low | 0.1986 | Knowledge Distillation from Large Language Models |
| `tier1_mixture_experts` | completed | No | Low | 0.1796 | Mixture of Retrieval Experts (MoRE) (Expected: 0.52-0.56 nDCG@10) |
| `tier1_momentum_contrastive` | completed | Yes | Medium | 0.1796 | Contrastive Learning with Momentum Encoder (Expected: 0.52-0.56 nDCG@10) |
| `tier1_multi_turn_state_tracking` | completed | Yes | Medium | 0.1796 | Multi-Turn Conversation State Tracking with Retrieval (Expected: 0.51-0.55 nDCG@10) |
| `tier1_multitask_retrieval` | completed | No | Low | 0.0185 | Multi-Task Learning for Retrieval |
| `tier1_nas_retrieval` | completed | Yes | Medium | 0.1796 | Neural Architecture Search for Retrieval (Expected: 0.53-0.57 nDCG@10) |
| `tier1_query_decomposition` | completed | No | Low | 0.1796 | Retrieval with Learned Query Decomposition (Expected: 0.50-0.54 nDCG@10) |
| `tier1_retrieval_as_generation` | completed | Yes | High | 0.1796 | Retrieval as Generation (RAG-Retrieval) (Expected: 0.54-0.58 nDCG@10) |
| `tier1_rlhf_retrieval` | completed | Yes | High | 0.1796 | RLHF for Retrieval (Expected: 0.54-0.58 nDCG@10) |
| `tier1_synthetic_data` | completed | No | Low | 0.1796 | Synthetic Data Generation for Retrieval (Expected: 0.53-0.57 nDCG@10) |
| `tier1_uncertainty_aware` | completed | Yes | High | 0.1796 | Uncertainty-Aware Retrieval with Confidence Calibration (Expected: 0.50-0.54 nDCG@10) |
| `tier1_causal_inference_fixed` | running | Yes | High |  | Causal Inference for Multi-Turn Retrieval (evaluation bug fixed) |
| `tier1_curriculum_contrastive_fixed` | running | Yes | Medium |  | Curriculum Contrastive Learning (evaluation bug fixed - now uses trained model) |
| `tier1_neural_ndcg` | running | No | Low |  | Direct nDCG Optimization with NeuralNDCG (Expected: 0.53-0.57 nDCG@10) |
| `tier1_rl_adaptive_retrieval` | running | No | Low |  | Reinforcement Learning for Adaptive Retrieval (Expected: 0.52-0.56 nDCG@10) |
| `tier1_temporal_attention` | running | Yes | Medium |  | Temporal Attention for Conversation History (Expected: 0.51-0.55 nDCG@10) |
| `tier1_uncertainty_aware_fixed` | running | Yes | High |  | Uncertainty-Aware Retrieval (evaluation bug fixed) |
| `hybrid_clapnq_alpha_0.3` | pending | No |  |  | No description found in source files. |
| `hybrid_clapnq_alpha_0.5` | pending | No |  |  | No description found in source files. |
| `hybrid_clapnq_alpha_0.7` | pending | No |  |  | No description found in source files. |
| `hybrid_govt_alpha_0.3` | pending | No |  |  | No description found in source files. |
| `hybrid_govt_alpha_0.5` | pending | No |  |  | No description found in source files. |
| `hybrid_govt_alpha_0.7` | pending | No |  |  | No description found in source files. |
| `hybrid_multi_alpha_0.3` | pending | No |  |  | No description found in source files. |
| `hybrid_multi_alpha_0.5` | pending | No |  |  | No description found in source files. |
| `hybrid_multi_alpha_0.7` | pending | No |  |  | No description found in source files. |
| `novel_catr_temporal_retrieval` | pending | Yes | High |  | CATR: Conversation-Aware Temporal Retrieval - Models temporal dynamics of multi-turn conversations (... |
| `novel_causal_retrieval` | pending | Yes | High |  | Causal-Ret: Causal Retrieval - Uses causal inference to understand query-document relationships (Tas... |
| `novel_clcf_contrastive_flow` | pending | Yes | High |  | CLCF-Ret: Contrastive Learning on Conversation Flows - Applies contrastive learning to conversation ... |
| `novel_diff_retrieval` | pending | Yes | High |  | Diff-Ret: Differentiable Retrieval - Fully differentiable pipeline with end-to-end nDCG optimization... |
| `novel_gep_graph_retrieval` | pending | Yes | High |  | GEP-Ret: Graph-Enhanced Retrieval - Uses GNNs to propagate entity relationships (Task A - Expected: ... |
| `novel_meta_retrieval` | pending | Yes | High |  | Meta-Ret: Cross-Domain Meta-Learning - Fast adaptation to new domains with few examples (Task A - Ex... |
| `novel_uq_uncertainty_retrieval` | pending | Yes | High |  | UQ-Ret: Uncertainty-Quantified Retrieval - Provides calibrated confidence scores for retrieval (Task... |
| `novel_x_explainable_retrieval` | pending | Yes | High |  | X-Ret: Explainable Retrieval - Provides attention-based explanations (Task A - Expected: 0.52-0.62 n... |
| `query_expansion_clapnq` | pending | No |  |  | No description found in source files. |
| `query_expansion_govt` | pending | No |  |  | No description found in source files. |
| `query_expansion_multi_domain` | pending | No |  |  | No description found in source files. |
| `reranking_cloud` | pending | No |  |  | No description found in source files. |
| `reranking_govt` | pending | No |  |  | No description found in source files. |
| `reranking_multi_domain` | pending | No |  |  | No description found in source files. |
| `task_a_advanced_contrastive_hierarchical` | pending | Yes | Medium |  | Advanced Contrastive Learning with Hierarchical Loss (Task A - Expected: 0.70-0.80 nDCG@10) |
| `task_a_advanced_training_techniques` | pending | Yes | Medium |  | Advanced Training Techniques (hard negatives, curriculum, adversarial) (Task A - Expected: 0.59-0.63... |
| `task_a_bge_v2_large_asymmetric` | pending | No | Low |  | BGE-v2-Large with Asymmetric Encoding: State-of-the-art model (560M params) with query/document prom... |
| `task_a_chainrag_retrieval` | pending | No | Low |  | ChainRAG: Multi-Hop Retrieval (Retrieval-Only) (Task A - Expected: 0.57-0.62 nDCG@10) ✅ Retrieval re... |
| `task_a_colbert_retrieval` | pending | No | Low |  | ColBERT-Style Multi-Vector Retrieval: Token-level embeddings with MaxSim scoring (Task A - Expected:... |
| `task_a_domain_specific_ensemble` | pending | Yes | Medium |  | Ensemble of Domain-Specific Retrievers (Task A - Expected: 0.75-0.85 nDCG@10) |
| `task_a_enhanced_differentiable_pipeline` | pending | Yes | High |  | Enhanced Differentiable End-to-End Pipeline (Task A - Expected: 0.70-0.85 nDCG@10) |
| `task_a_enhanced_graph_construction` | pending | No | Low |  | Enhanced Graph Construction and GNN Architectures (Task A - Expected: 0.60-0.75 nDCG@10) |
| `task_a_enhanced_multistage_llm_scoring` | pending | Yes | Medium |  | Enhanced Multi-Stage Pipeline with LLM-Based Relevance Scoring (Task A - Expected: 0.85-0.92 nDCG@10... |
| `task_a_graph_neural_retrieval` | pending | Yes | High |  | Conversation-Aware Graph Neural Retrieval (Task A - Expected: 0.68-0.78 nDCG@10) |
| `task_a_late_chunking_retrieval` | pending | No | Low |  | Late Chunking Retrieval: Embed full document first, then chunk (preserves global context) (Task A - ... |
| `task_a_llm_powered_retrieval` | pending | No | Low |  | LLM-Powered Retrieval with In-Context Learning (Task A - Expected: 0.70-0.80 nDCG@10) |
| `task_a_llm_query_expansion_constrained` | pending | Yes | Medium |  | LLM-Powered Query Expansion (Task A - Expected: 0.65-0.75 nDCG@10) ⚠️ CONSTRAINT: LLM generates quer... |
| `task_a_metarag_retrieval` | pending | No | Low |  | MetaRAG: Retrieval Quality Self-Assessment (Task A - Expected: 0.58-0.63 nDCG@10) ✅ Evaluates retrie... |
| `task_a_model_scaling_bge_v2` | pending | No | Low |  | Model Scaling Improvements (BGE-v2-large, longer training, domain-specific) (Task A - Expected: 0.61... |
| `task_a_openrag_retrieval_only` | pending | No | Low |  | OpenRAG-Style Retrieval-Only End-to-End Optimization (Task A - Expected: 0.55-0.60 nDCG@10) ✅ Uses r... |
| `task_a_optimized_hybrid_reranking` | pending | Yes | Medium |  | Optimized Hybrid Search with Reranking - All Improvements Combined (Task A - Expected: 0.60-0.75 nDC... |
| `task_a_probing_rag` | pending | No | Low |  | Probing-RAG: Selective Document Retrieval (Task A - Expected: 0.53-0.58 nDCG@10) ✅ Uses hidden state... |
| `task_a_rlhf_retrieval_constrained` | pending | Yes | High |  | RLHF for Retrieval (Task A - Expected: 0.72-0.82 nDCG@10) ⚠️ CONSTRAINT: RLHF optimizes retrieval me... |
| `task_a_semantic_chunking_retrieval` | pending | No | Low |  | Semantic Chunking Retrieval: Boundary-Aware Chunking (Task A - Expected: +0.03-0.06 nDCG@10). Chunks... |
| `task_a_specialized_ensemble_meta_learner` | pending | Yes | High |  | Specialized Ensemble with Meta-Learner (Task A - Expected: 0.75-0.85 nDCG@10) |
| `task_a_transform_retrieval` | pending | Yes | High |  | Transform Retrieval for Textual Entailment (Task A - Expected: 0.56-0.61 nDCG@10) ✅ Pure retrieval o... |
| `tier1_cg_rag_retrieval` | pending | No | Low |  | CG-RAG (Contextualized Graph RAG) Retrieval: Builds knowledge graph, contextualizes based on query/c... |
| `tier1_cross_domain_transfer` | pending | Yes | Medium |  | Cross-Domain Transfer Learning with Domain Adversarial Training (Expected: 0.51-0.55 nDCG@10) |
| `tier1_gfm_rag_retrieval` | pending | No | Low |  | GFM-RAG (Graph Foundation Model for RAG) Retrieval: Uses foundation models for graph encoding, appli... |
| `tier1_hyde_retrieval` | pending | No | Low |  | HyDE (Hypothetical Document Embeddings) Retrieval: Generates hypothetical documents for retrieval (T... |
| `tier1_meta_learning` | pending | Yes | High |  | Meta-Learning for Fast Domain Adaptation (Expected: 0.51-0.55 nDCG@10) |
| `tier1_mmr_retrieval` | pending | No | Low |  | MMR (Maximal Marginal Relevance) Retrieval: Balances relevance and diversity (Task A - Expected: +0.... |
| `tier1_prompt_based_retrieval` | pending | No | Low |  | Prompt-Based Retrieval with In-Context Learning (Expected: 0.52-0.56 nDCG@10) |
| `advanced_contrastive_hierarchical` | stopped | No |  |  | No description found in source files. |
| `adversarial_curriculum_learning` | stopped | No |  |  | No description found in source files. |
| `adversarial_robustness_training` | stopped | No |  |  | No description found in source files. |
| `baseline_3_epochs` | stopped | No |  |  | No description found in source files. |
| `baseline_5_epochs` | stopped | No |  |  | No description found in source files. |
| `baseline_bge_finetuned` | stopped | No |  |  | No description found in source files. |
| `bge_large_model_finetuning` | stopped | No |  |  | No description found in source files. |
| `causal_inference_retrieval` | stopped | No |  |  | No description found in source files. |
| `causal_inference_retrieval_fixed` | stopped | No |  |  | No description found in source files. |
| `continual_learning_adaptation` | stopped | No |  |  | No description found in source files. |
| `contrastive_learning_finetuning` | stopped | No |  |  | No description found in source files. |
| `conversation_aware_attention` | stopped | No |  |  | No description found in source files. |
| `cosine_similarity_loss` | stopped | No |  |  | No description found in source files. |
| `counterfactual_data_augmentation` | stopped | No |  |  | No description found in source files. |
| `cross_attention_mechanism` | stopped | No |  |  | No description found in source files. |
| `cross_attention_mechanism_v3` | stopped | No |  |  | No description found in source files. |
| `cross_attention_query_document` | stopped | No |  |  | No description found in source files. |
| `cross_attention_query_document_fixed` | stopped | No |  |  | No description found in source files. |
| `cross_attention_rerun` | stopped | No |  |  | No description found in source files. |
| `cross_attention_rerun_fixed` | stopped | No |  |  | No description found in source files. |
| `cross_attention_rerun_fixed_v2` | stopped | No |  |  | No description found in source files. |
| `cross_domain_transfer_learning` | stopped | No |  |  | No description found in source files. |
| `cross_encoder_conversation_context` | stopped | No |  |  | No description found in source files. |
| `cross_encoder_domain_specific` | stopped | No |  |  | No description found in source files. |
| `cross_encoder_evaluation` | stopped | No |  |  | No description found in source files. |
| `cross_encoder_finetuned` | stopped | No |  |  | No description found in source files. |
| `cross_encoder_finetuned_ensemble` | stopped | No |  |  | No description found in source files. |
| `cross_encoder_large_model` | stopped | No |  |  | No description found in source files. |
| `cross_encoder_reranking` | stopped | No |  |  | No description found in source files. |
| `cross_encoder_reranking_evaluation` | stopped | No |  |  | No description found in source files. |
| `curriculum_contrastive_learning` | stopped | No |  |  | No description found in source files. |
| `curriculum_contrastive_learning_fixed` | stopped | No |  |  | No description found in source files. |
| `data_augmentation_retrieval` | stopped | No |  |  | No description found in source files. |
| `differentiable_end_to_end` | stopped | No |  |  | No description found in source files. |
| `differentiable_retrieval` | stopped | No |  |  | No description found in source files. |
| `differentiable_retrieval_fixed` | stopped | No |  |  | No description found in source files. |
| `domain_specific_ensemble` | stopped | No |  |  | No description found in source files. |
| `domain_specific_finetuning_all_domains` | stopped | No |  |  | No description found in source files. |
| `domain_specific_finetuning_clapnq` | stopped | No |  |  | No description found in source files. |
| `domain_specific_finetuning_cloud` | stopped | No |  |  | No description found in source files. |
| `domain_specific_finetuning_fiqa` | stopped | No |  |  | No description found in source files. |
| `domain_specific_finetuning_govt` | stopped | No |  |  | No description found in source files. |
| `enhanced_contrastive_hard_negatives` | stopped | No |  |  | No description found in source files. |
| `enhanced_contrastive_hard_negatives_fixed` | stopped | No |  |  | No description found in source files. |
| `enhanced_contrastive_improved` | stopped | No |  |  | No description found in source files. |
| `enhanced_contrastive_learning` | stopped | No |  |  | No description found in source files. |
| `enhanced_multistage_llm_scoring` | stopped | No |  |  | No description found in source files. |
| `ensemble_advanced_methods` | stopped | No |  |  | No description found in source files. |
| `ensemble_best_performing` | stopped | No |  |  | No description found in source files. |
| `ensemble_best_performing_short` | stopped | No |  |  | No description found in source files. |
| `explainable_retrieval` | stopped | No |  |  | No description found in source files. |
| `foundation_model_distillation` | stopped | No |  |  | No description found in source files. |
| `foundation_model_distillation_fixed` | stopped | No |  |  | No description found in source files. |
| `graph_aware_retrieval` | stopped | No |  |  | No description found in source files. |
| `graph_aware_retrieval_short` | stopped | No |  |  | No description found in source files. |
| `graph_enhanced_reranking` | stopped | No |  |  | No description found in source files. |
| `graph_enhanced_reranking_fixed` | stopped | No |  |  | No description found in source files. |
| `hard_negatives_cosine_similarity` | stopped | No |  |  | No description found in source files. |
| `hard_negatives_mining_5` | stopped | No |  |  | No description found in source files. |
| `hard_negatives_triplet_loss` | stopped | No |  |  | No description found in source files. |
| `hierarchical_multigranularity_retrieval` | stopped | No |  |  | No description found in source files. |
| `hierarchical_routing` | stopped | No |  |  | No description found in source files. |
| `hierarchical_routing_fixed` | stopped | No |  |  | No description found in source files. |
| `hybrid_dynamic_fusion` | stopped | No |  |  | No description found in source files. |
| `hybrid_lexical_semantic_fusion` | stopped | No |  |  | No description found in source files. |
| `hybrid_lexical_semantic_reranking` | stopped | No |  |  | No description found in source files. |
| `hybrid_reranking_fusion` | stopped | No |  |  | No description found in source files. |
| `iterative_query_refinement` | stopped | No |  |  | No description found in source files. |
| `iterative_refinement_improved` | stopped | No |  |  | No description found in source files. |
| `knowledge_distillation_llm` | stopped | No |  |  | No description found in source files. |
| `knowledge_graph_enhanced` | stopped | No |  |  | No description found in source files. |
| `large_model_finetuning` | stopped | No |  |  | No description found in source files. |
| `learned_index_structures` | stopped | No |  |  | No description found in source files. |
| `learned_index_structures_fixed` | stopped | No |  |  | No description found in source files. |
| `learned_reciprocal_rank_fusion` | stopped | No |  |  | No description found in source files. |
| `learning_rate_1e5` | stopped | No |  |  | No description found in source files. |
| `learning_rate_5e5` | stopped | No |  |  | No description found in source files. |
| `learning_to_rank` | stopped | No |  |  | No description found in source files. |
| `learning_to_rank_listwise` | stopped | No |  |  | No description found in source files. |
| `learning_to_rank_listwise_fixed` | stopped | No |  |  | No description found in source files. |
| `llm_distillation` | stopped | No |  |  | No description found in source files. |
| `llm_query_expansion` | stopped | No |  |  | No description found in source files. |
| `llm_query_expansion_constrained` | stopped | No |  |  | No description found in source files. |
| `llm_query_expansion_gpt4` | stopped | No |  |  | No description found in source files. |
| `logs` | stopped | No |  |  | No description found in source files. |
| `memory_augmented_retrieval` | stopped | No |  |  | No description found in source files. |
| `meta_learning_adaptation` | stopped | No |  |  | No description found in source files. |
| `meta_learning_adaptation_fixed` | stopped | No |  |  | No description found in source files. |
| `mixture_of_experts` | stopped | No |  |  | No description found in source files. |
| `mixture_of_experts_fixed` | stopped | No |  |  | No description found in source files. |
| `model_scaling_bge_v2` | stopped | No |  |  | No description found in source files. |
| `momentum_contrastive_learning` | stopped | No |  |  | No description found in source files. |
| `multi_granularity_encoding` | stopped | No |  |  | No description found in source files. |
| `multi_turn_state_tracking` | stopped | No |  |  | No description found in source files. |
| `multi_turn_state_tracking_fixed` | stopped | No |  |  | No description found in source files. |
| `multistage_hierarchical_retrieval` | stopped | No |  |  | No description found in source files. |
| `multistage_retrieval_2stage` | stopped | No |  |  | No description found in source files. |
| `multistage_retrieval_2stage_finetuned` | stopped | No |  |  | No description found in source files. |
| `multistage_retrieval_3stage` | stopped | No |  |  | No description found in source files. |
| `multistage_retrieval_3stage_finetuned` | stopped | No |  |  | No description found in source files. |
| `multistage_retrieval_base` | stopped | No |  |  | No description found in source files. |
| `multitask_learning` | stopped | No |  |  | No description found in source files. |
| `multitask_learning_retrieval` | stopped | No |  |  | No description found in source files. |
| `neural_architecture_search` | stopped | No |  |  | No description found in source files. |
| `neural_ndcg_optimization` | stopped | No |  |  | No description found in source files. |
| `prompt_based_retrieval` | stopped | No |  |  | No description found in source files. |
| `pseudo_relevance_feedback` | stopped | No |  |  | No description found in source files. |
| `qdit_transformer_architecture` | stopped | No |  |  | No description found in source files. |
| `query_decomposition_retrieval` | stopped | No |  |  | No description found in source files. |
| `reinforcement_learning_adaptive` | stopped | No |  |  | No description found in source files. |
| `reinforcement_learning_adaptive_short` | stopped | No |  |  | No description found in source files. |
| `reinforcement_learning_human_feedback` | stopped | No |  |  | No description found in source files. |
| `retrieval_as_generation` | stopped | No |  |  | No description found in source files. |
| `rl_adaptive_retrieval` | stopped | No |  |  | No description found in source files. |
| `rl_adaptive_retrieval_fixed` | stopped | No |  |  | No description found in source files. |
| `semantic_drift_adaptation` | stopped | No |  |  | No description found in source files. |
| `specialized_ensemble_meta_learner` | stopped | No |  |  | No description found in source files. |
| `synthetic_data_augmentation` | stopped | No |  |  | No description found in source files. |
| `temporal_attention_mechanism` | stopped | No |  |  | No description found in source files. |
| `temporal_memory_retrieval` | stopped | No |  |  | No description found in source files. |
| `temporal_memory_retrieval_fixed` | stopped | No |  |  | No description found in source files. |
| `test` | stopped | No |  |  | No description found in source files. |
| `tier1_temporal_memory` | stopped | No |  |  | No description found in source files. |
| `trained_model_clapnq` | stopped | No |  |  | No description found in source files. |
| `trained_model_cloud` | stopped | No |  |  | No description found in source files. |
| `trained_model_fiqa` | stopped | No |  |  | No description found in source files. |
| `trained_model_govt` | stopped | No |  |  | No description found in source files. |
| `uncertainty_aware_retrieval` | stopped | No |  |  | No description found in source files. |
| `uncertainty_aware_retrieval_fixed` | stopped | No |  |  | No description found in source files. |
| `domain_specific_clapnq_hard_negatives` | failed | No |  |  | No description found in source files. |
| `domain_specific_govt_hard_negatives` | failed | No |  |  | No description found in source files. |
| `tier1_counterfactual_augmentation` | failed | Yes | Medium |  | element 0 of tensors does not require grad and does not have a grad_fn |
| `tier1_enhanced_contrastive` | failed | No | Low |  | Processed 2 negative embeddings but expected 1 |
| `tier1_enhanced_contrastive_hardnegatives_fixed` | failed | No | Low |  | CUDA out of memory. Tried to allocate 2.00 MiB. GPU 0 has a total capacity of... |
| `tier1_learning_to_rank_listwise_fixed` | failed | No | Low |  | Evaluator retriever is not set! |
| `tier1_memory_augmented` | failed | No | Low |  | Process died without completing |
| `tier1_mixture_experts_fixed` | failed | No | Low |  | Process died without completing |
| `tier1_multi_granularity` | failed | Yes | Medium |  | element 0 of tensors does not require grad and does not have a grad_fn |
| `tier1_semantic_drift` | failed | Yes | Medium |  | element 0 of tensors does not require grad and does not have a grad_fn |
| `best_paper_adversarial_curriculum` | run-unknown | No |  |  | Adversarial Hard Negative Mining with Curriculum Learning |
| `best_paper_graph_aware_retrieval` | run-unknown | No |  |  | Conversation Graph-Aware Retrieval |
| `best_paper_hierarchical_routing` | run-unknown | No |  |  | Hierarchical Multi-Stage with Learned Routing |
| `best_paper_large_model_finetuning` | run-unknown | No |  |  | Large Model Fine-Tuning (BGE-Large or Larger) |
| `best_paper_learned_rrf` | run-unknown | No |  |  | Learned Reciprocal Rank Fusion with Neural Weighting |
| `best_paper_llm_distillation` | run-unknown | No |  |  | Knowledge Distillation from Large Language Models |
| `best_paper_meta_learning` | run-unknown | No |  |  | Cross-Domain Transfer Learning with Meta-Learning (MAML) |
| `best_paper_meta_learning_fixed` | run-unknown | No |  |  | Cross-domain meta-learning with MAML (fixed version) |
| `best_paper_multitask_retrieval` | run-unknown | No |  |  | Multi-Task Learning: Retrieval + Reranking (Retrieval-Only, adapted for Task A) |
| `best_paper_rl_adaptive_retrieval` | run-unknown | No |  |  | Reinforcement Learning for Adaptive Retrieval |
| `best_paper_rl_adaptive_retrieval_fixed` | run-unknown | No |  |  | Reinforcement learning for adaptive retrieval (fixed version) |
| `best_paper_temporal_memory` | run-unknown | No |  |  | Temporal Memory Networks for Conversation Context |
| `best_paper_temporal_memory_fixed` | run-unknown | No |  |  | Temporal memory networks for conversation context (fixed version) |
| `multistage_retrieval` | run-unknown | No |  |  | Multi-Stage Retrieval Pipeline with Checkpointing and Resume Support |
| `phase1_baseline` | run-unknown | No |  |  | Baseline dense retrieval using BGE-base model |
| `phase3_hybrid_reranking` | run-unknown | No |  |  | Hybrid retrieval combining dense and sparse methods with reranking |
| `phase3_reranking` | run-unknown | No |  |  | Dense retrieval with cross-encoder reranking |
| `phase4_hard_negatives_5neg` | run-unknown | No |  |  | Contrastive learning with 5 hard negative examples per query |
| `phase5_ensemble_domain_specific` | run-unknown | No |  |  | Ensemble of domain-specific retrievers for each domain |
| `phase5_ensemble_weighted` | run-unknown | No |  |  | Weighted ensemble of multiple retrieval models |
| `phase5_query_expansion_clapnq` | run-unknown | No |  |  | Query expansion using LLM for ClapNQ domain |
| `phase5_query_expansion_govt` | run-unknown | No |  |  | Query expansion using LLM for Govt domain |
| `phase5_query_expansion_multi` | run-unknown | No |  |  | Query expansion using LLM across multiple domains |
| `phase5_reranking_clapnq` | run-unknown | No |  |  | Cross-encoder reranking for ClapNQ domain |
| `phase5_reranking_cloud` | run-unknown | No |  |  | Cross-encoder reranking for Cloud domain |
| `phase5_reranking_govt` | run-unknown | No |  |  | Cross-encoder reranking for Govt domain |
| `phase5_reranking_multi_domain` | run-unknown | No |  |  | Cross-encoder reranking across multiple domains |
| `phase6_cross_encoder_evaluation` | run-unknown | No |  |  | Evaluation of cross-encoder models for reranking |
| `phase6_llm_query_expansion_gpt4_multi` | run-unknown | No |  |  | GPT-4 powered query expansion across multiple domains |
| `phase6_multistage_2stage` | run-unknown | No |  |  | Two-stage retrieval: dense retrieval followed by reranking |
| `phase6_multistage_2stage_finetuned` | run-unknown | No |  |  | Two-stage retrieval with fine-tuned models |
| `phase7_conversation_aware_attention` | run-unknown | No |  |  | Attention mechanism that considers conversation history |
| `phase7_iterative_refinement` | run-unknown | No |  |  | Iterative query refinement and re-retrieval |
| `phase8_cross_attention_query_document` | run-unknown | No |  |  | Cross-attention between query and document embeddings |
| `phase8_cross_attention_query_document_fixed` | run-unknown | No |  |  | Cross-attention between query and document embeddings (fixed version) |
| `tier1_contrastive_learning` | run-unknown | No |  |  | Standard contrastive learning for retrieval |
| `tier1_cross_attention` | run-unknown | No |  |  | Cross-attention mechanism for query-document interaction |
| `tier1_cross_attention_fixed_v3` | run-unknown | No |  |  | Cross-attention mechanism (fixed version 3) |
| `tier1_cross_attention_rerun` | run-unknown | No |  |  | Cross-attention mechanism rerun |
| `tier1_cross_attention_rerun_fixed` | run-unknown | No |  |  | Cross-attention mechanism rerun (fixed) |
| `tier1_cross_attention_rerun_fixed_v2` | run-unknown | No |  |  | Cross-attention mechanism rerun (fixed version 2) |
| `tier1_cross_encoder_domain_specific` | run-unknown | No |  |  | Domain-specific cross-encoder models for reranking |
| `tier1_cross_encoder_evaluation` | run-unknown | No |  |  | Evaluation framework for cross-encoder models |
| `tier1_cross_encoder_large` | run-unknown | No |  |  | Large cross-encoder model for reranking |
| `tier1_ensemble_best_methods` | run-unknown | No |  |  | Ensemble combining best performing retrieval methods |
| `tier1_hierarchical_multigranularity` | run-unknown | No |  |  | Hierarchical retrieval at multiple granularity levels |
| `tier1_iterative_refinement_improved` | run-unknown | No |  |  | Improved iterative query refinement and retrieval |
| `tier1_learning_to_rank_listwise` | run-unknown | No |  |  | Learning-to-Rank with Listwise Loss for Multi-Turn RAG Retrieval |
| `tier1_pseudo_relevance_feedback` | run-unknown | No |  |  | Pseudo-Relevance Feedback with LLM Expansion for Multi-Turn RAG Retrieval |
| `tier1_qdit_transformer` | run-unknown | No |  |  | Query-Document Interaction Transformer (QDIT) |
| `tier2_ensemble_advanced` | run-unknown | No |  |  | Advanced ensemble methods combining multiple retrieval strategies |
| `tier2_multistage_3stage` | run-unknown | No |  |  | Three-stage retrieval pipeline: dense, sparse, and reranking |

## Notes

- **Novel Idea** is marked `Yes` when novelty is High/Medium, the name starts with `novel_`, or it appears in novel proposal/implementation files.
- **run-unknown** indicates the experiment appears in historical run lists without a clear completion status in the status files.
