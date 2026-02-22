#!/usr/bin/env python3
"""
Automated Fixed and Pending Experiments Runner
Automatically starts fixed experiments and new/pending experiments when GPUs become available
"""

import os
import sys
import subprocess
import pathlib
import json
import time
import logging
import signal
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Fixed experiments that need re-run (fixes have been applied)
FIXED_EXPERIMENTS = [
    {
        "name": "enhanced_contrastive_learning_hardnegatives_fixed",
        "script": "train_enhanced_contrastive_tier1.py",
        "priority": 1,  # Highest priority - fixes applied
        "description": "Enhanced Contrastive Learning with Hard Negatives (tensor size mismatch fix applied)",
        "fix_applied": "Added tensor shape verification and batch size checks"
    },
    {
        "name": "learning_to_rank_listwise_fixed",
        "script": "train_learning_to_rank_tier1.py",
        "priority": 1,  # Highest priority - fixes applied
        "description": "Learning-to-Rank with Listwise Loss (evaluator fix applied)",
        "fix_applied": "Evaluator reuse fixed (removed None retriever creation)"
    },
    {
        "name": "hierarchical_routing_fixed",
        "script": "train_hierarchical_routing_tier1.py",
        "priority": 1,  # Highest priority - fixes applied
        "description": "Hierarchical Multi-Stage with Learned Routing (indentation fix applied)",
        "fix_applied": "Indentation bug fixed in run_hierarchical_routing_evaluation"
    },
    {
        "name": "cross_attention_mechanism_query_document",
        "script": "train_cross_attention_tier1.py",
        "priority": 1,  # Highest priority - fixes applied
        "description": "Cross-Attention Query-Document Interaction (CUDA device fix applied)",
        "fix_applied": "Fixed CUDA device handling when CUDA_VISIBLE_DEVICES is set"
    },
    {
        "name": "cosine_similarity_loss",
        "script": "train_cosine_similarity_loss_tier1.py",
        "priority": 2,  # Medium priority - fixes applied
        "description": "Cosine Similarity Loss Training (wrapper script created)",
        "fix_applied": "Created tier1 wrapper script with proper GPU and output_dir support"
    },
    {
        "name": "enhanced_contrastive_learning",
        "script": "train_enhanced_contrastive_tier1.py",
        "priority": 2,  # Medium priority - fixes applied
        "description": "Enhanced Contrastive Learning (tensor size mismatch fix applied)",
        "fix_applied": "Added tensor shape verification and batch size checks"
    },
    {
        "name": "neural_ndcg_optimization",
        "script": "train_neural_ndcg_tier1.py",
        "priority": 1,  # Highest priority - gradient tracking fix applied
        "description": "Direct nDCG Optimization with NeuralNDCG (gradient tracking fix applied)",
        "fix_applied": "Fixed gradient tracking by using internal forward pass instead of model.encode()"
    },
    {
        "name": "graph_enhanced_reranking",
        "script": "train_graph_enhanced_reranking_tier1.py",
        "priority": 2,  # High priority - tensor boolean fix applied
        "description": "Graph-Enhanced Adaptive Re-Ranking (tensor boolean fix applied)",
        "fix_applied": "Fixed tensor boolean ambiguity by replacing 'or' with explicit None checks"
    },
    {
        "name": "multi_turn_state_tracking",
        "script": "train_state_tracking_tier1.py",
        "priority": 2,  # Medium priority - OOM fix applied
        "description": "Multi-Turn Conversation State Tracking (OOM fix applied)",
        "fix_applied": "Reduced batch size to 2 and increased gradient accumulation to 4"
    },
    {
        "name": "cross_encoder_conversation_context",
        "script": "train_cross_encoder_conversation_context_tier1.py",
        "priority": 1,  # Highest priority - BEIR API fix applied
        "description": "Transformer-Based Cross-Encoder with Full Conversation Context (BEIR API fix applied)",
        "fix_applied": "Fixed DenseRetrievalExactSearch.retrieve() API issue - using EvaluateRetrieval wrapper"
    },
    {
        "name": "hybrid_dynamic_fusion",
        "script": "train_hybrid_dynamic_fusion_tier1.py",
        "priority": 1,  # Highest priority - BEIR API and BM25 fix applied
        "description": "Hybrid Dense-Sparse with Learned Dynamic Fusion (BEIR API and BM25 fix applied)",
        "fix_applied": "Fixed DenseRetrievalExactSearch.retrieve() API and made BM25Search optional"
    },
    {
        "name": "multistage_hierarchical_retrieval",
        "script": "train_multistage_hierarchical_retrieval_tier1.py",
        "priority": 1,  # Highest priority - BEIR API and BM25 fix applied
        "description": "Multi-Stage Hierarchical Retrieval (BEIR API and BM25 fix applied)",
        "fix_applied": "Fixed DenseRetrievalExactSearch.retrieve() API and made BM25Search optional"
    },
    {
        "name": "differentiable_end_to_end",
        "script": "train_differentiable_end_to_end_tier1.py",
        "priority": 2,  # High priority - BEIR API fix applied
        "description": "Differentiable End-to-End Retrieval Pipeline (BEIR API fix applied)",
        "fix_applied": "Fixed DenseRetrievalExactSearch.retrieve() API - using EvaluateRetrieval wrapper"
    },
    {
        "name": "domain_specific_ensemble",
        "script": "train_domain_specific_ensemble_tier1.py",
        "priority": 2,  # High priority - subprocess fix applied
        "description": "Ensemble of Domain-Specific Retrievers (subprocess output redirection fix applied)",
        "fix_applied": "Fixed subprocess output redirection to use log files instead of PIPE"
    },
    # CRITICAL FIX: Experiments that were evaluating with base model instead of trained model
    {
        "name": "curriculum_contrastive_learning_fixed",
        "script": "train_curriculum_contrastive_tier1.py",
        "priority": 1,  # Highest priority - critical evaluation bug fixed
        "description": "Curriculum Contrastive Learning (evaluation bug fixed - now uses trained model)",
        "fix_applied": "Fixed evaluation to use trained model instead of base model, added model saving",
        "no_resume": True  # Force fresh training to get correct scores
    },
    {
        "name": "causal_inference_retrieval_fixed",
        "script": "train_causal_inference_tier1.py",
        "priority": 1,  # Highest priority - critical evaluation bug fixed
        "description": "Causal Inference for Multi-Turn Retrieval (evaluation bug fixed)",
        "fix_applied": "Fixed evaluation to use trained model instead of base model, added model saving",
        "no_resume": True
    },
    {
        "name": "uncertainty_aware_retrieval_fixed",
        "script": "train_uncertainty_aware_tier1.py",
        "priority": 1,  # Highest priority - critical evaluation bug fixed
        "description": "Uncertainty-Aware Retrieval (evaluation bug fixed)",
        "fix_applied": "Fixed evaluation to use trained model instead of base model, added model saving",
        "no_resume": True
    },
    {
        "name": "mixture_of_experts_fixed",
        "script": "train_mixture_experts_tier1.py",
        "priority": 1,  # Highest priority - critical evaluation bug fixed
        "description": "Mixture of Retrieval Experts (evaluation bug fixed)",
        "fix_applied": "Fixed evaluation to use trained model instead of base model, added model saving",
        "no_resume": True
    },
    {
        "name": "multi_turn_state_tracking_fixed",
        "script": "train_state_tracking_tier1.py",
        "priority": 1,  # Highest priority - critical evaluation bug fixed
        "description": "Multi-Turn Conversation State Tracking (evaluation bug fixed)",
        "fix_applied": "Fixed evaluation to use trained model instead of base model, added model saving",
        "no_resume": True
    },
    {
        "name": "learned_index_structures_fixed",
        "script": "train_learned_indices_tier1.py",
        "priority": 1,  # Highest priority - critical evaluation bug fixed
        "description": "Learned Indices for Neural Retrieval (evaluation bug fixed)",
        "fix_applied": "Fixed evaluation to use trained model instead of base model, added model saving",
        "no_resume": True
    },
    {
        "name": "differentiable_retrieval_fixed",
        "script": "train_differentiable_retrieval_tier1.py",
        "priority": 1,  # Highest priority - critical evaluation bug fixed
        "description": "Differentiable End-to-End Retrieval (evaluation bug fixed)",
        "fix_applied": "Fixed evaluation to use trained model instead of base model, added model saving",
        "no_resume": True
    },
    {
        "name": "foundation_model_distillation_fixed",
        "script": "train_foundation_distillation_tier1.py",
        "priority": 1,  # Highest priority - critical evaluation bug fixed
        "description": "Foundation Model Distillation (evaluation bug fixed)",
        "fix_applied": "Fixed evaluation to use trained model instead of base model, added model saving",
        "no_resume": True
    },
    {
        "name": "graph_enhanced_reranking_fixed",
        "script": "train_graph_enhanced_reranking_tier1.py",
        "priority": 1,  # Highest priority - critical evaluation bug fixed
        "description": "Graph-Enhanced Adaptive Re-Ranking (evaluation bug fixed)",
        "fix_applied": "Fixed evaluation to use trained model instead of base model, added model saving",
        "no_resume": True
    },
    {
        "name": "explainable_retrieval_fixed",
        "script": "train_explainable_retrieval_tier1.py",
        "priority": 1,  # Highest priority - critical evaluation bug fixed
        "description": "Explainable Retrieval with Attention Visualization (evaluation bug fixed)",
        "fix_applied": "Fixed evaluation to use trained model instead of base model, added model saving",
        "no_resume": True
    },
    {
        "name": "reinforcement_learning_human_feedback_fixed",
        "script": "train_rlhf_retrieval_tier1.py",
        "priority": 1,  # Highest priority - critical evaluation bug fixed
        "description": "RLHF for Retrieval (evaluation bug fixed)",
        "fix_applied": "Fixed evaluation to use trained model instead of base model, added model saving",
        "no_resume": True
    },
    {
        "name": "retrieval_as_generation_fixed",
        "script": "train_retrieval_as_generation_tier1.py",
        "priority": 1,  # Highest priority - critical evaluation bug fixed
        "description": "Retrieval as Generation (RAG-Retrieval) (evaluation bug fixed)",
        "fix_applied": "Fixed evaluation to use trained model instead of base model, added model saving",
        "no_resume": True
    },
    # Phase 2 Improvements: Enhanced experiments with better training strategies
    {
        "name": "enhanced_contrastive_learning_improved",
        "script": "train_enhanced_contrastive_improved_tier1.py",
        "priority": 1,  # Highest priority - Phase 2 improvements
        "description": "Enhanced Contrastive Learning with Phase 2 Improvements (better negatives, loss, training)",
        "fix_applied": "Added improved negative sampling, temperature-scaled margin loss, longer training, cosine LR schedule",
        "no_resume": True
    },
    # Fixed experiments - gradient errors resolved
    {
        "name": "counterfactual_data_augmentation_fixed",
        "script": "train_counterfactual_tier1.py",
        "priority": 2,  # High priority - gradient error fixed
        "description": "Contrastive Learning with Counterfactual Augmentation (gradient error fixed)",
        "fix_applied": "Fixed gradient flow by using model's internal forward pass with proper gradient tracking",
        "no_resume": True
    },
    {
        "name": "semantic_drift_adaptation_fixed",
        "script": "train_semantic_drift_tier1.py",
        "priority": 2,  # High priority - gradient error fixed
        "description": "Semantic Drift Detection with Adaptive Retrieval (gradient error fixed)",
        "fix_applied": "Fixed gradient flow by using model's internal forward pass with proper gradient tracking",
        "no_resume": True
    },
    {
        "name": "multi_granularity_encoding_fixed",
        "script": "train_multi_granularity_tier1.py",
        "priority": 2,  # High priority - gradient error fixed
        "description": "Multi-Granularity Contrastive Learning (gradient error fixed)",
        "fix_applied": "Fixed gradient flow by using model's internal forward pass with proper gradient tracking",
        "no_resume": True
    },
    # Fixed experiments - other errors resolved
    {
        "name": "enhanced_contrastive_learning_fixed_v2",
        "script": "train_enhanced_contrastive_tier1.py",
        "priority": 2,  # High priority - negative embedding count fixed
        "description": "Enhanced Contrastive Learning (negative embedding count error fixed)",
        "fix_applied": "Fixed negative embedding count mismatch by ensuring num_hard_negatives=1 is properly enforced",
        "no_resume": True
    },
    {
        "name": "enhanced_contrastive_learning_hardnegatives_fixed_v2",
        "script": "train_enhanced_contrastive_tier1.py",
        "priority": 1,  # Highest priority - OOM fixed
        "description": "Enhanced Contrastive Learning with Hard Negatives (OOM error fixed)",
        "fix_applied": "Reduced batch size to 2, increased gradient accumulation to 4, enabled FP16",
        "no_resume": True
    },
    {
        "name": "query_decomposition_retrieval_fixed",
        "script": "train_query_decomposition_tier1.py",
        "priority": 2,  # High priority - gradient error fixed
        "description": "Retrieval with Learned Query Decomposition (gradient error fixed)",
        "fix_applied": "Fixed gradient flow by using model's internal forward pass with proper gradient tracking",
        "no_resume": True
    },
    {
        "name": "learning_to_rank_listwise_fixed_v2",
        "script": "train_learning_to_rank_tier1.py",
        "priority": 1,  # Highest priority - evaluator error fixed
        "description": "Learning-to-Rank with Listwise Loss (evaluator retriever fix applied)",
        "fix_applied": "Fixed evaluator retriever not set error by recreating evaluator before evaluation",
        "no_resume": True
    }
]

# Pending/new experiments that should be started
PENDING_EXPERIMENTS = [
    # NEW: High-Impact Experiments for nDCG@10 > 0.90 (Priority 1 - Highest)
    {
        "name": "tier1_colbert_multivector",
        "script": "train_colbert_retrieval_tier1.py",
        "priority": 1,  # Highest priority - fully compliant, high impact
        "description": "ColBERT-Style Multi-Vector Retrieval (Task A Compliant - Expected: 0.80-0.88 nDCG@10)"
    },
    {
        "name": "tier1_ensemble_meta_learner",
        "script": "train_ensemble_meta_learner_tier1.py",
        "priority": 1,  # Highest priority - fully compliant, high impact
        "description": "Ensemble with Meta-Learner Fusion (Task A Compliant - Expected: 0.82-0.90 nDCG@10)"
    },
    {
        "name": "tier1_generative_query_expansion",
        "script": "train_generative_query_expansion_tier1.py",
        "priority": 1,  # Highest priority - fully compliant
        "description": "Generative Query Expansion with LLM (Task A Compliant - Expected: 0.75-0.82 nDCG@10)"
    },
    {
        "name": "multistage_retrieval_base_llm_scoring",
        "script": "train_multistage_llm_scoring_tier1.py",
        "priority": 1,  # Highest priority - conditionally compliant
        "description": "Multi-Stage Retrieval with LLM Scoring (Task A Compliant - Expected: 0.85-0.92 nDCG@10) ⚠️ LLM outputs scores only"
    },
    {
        "name": "tier1_iterative_query_expansion",
        "script": "train_iterative_query_expansion_tier1.py",
        "priority": 1,  # Highest priority - conditionally compliant
        "description": "Iterative Retrieval with Query Expansion (Task A Compliant - Expected: 0.72-0.80 nDCG@10) ⚠️ Feedback is query expansion only"
    },
    {
        "name": "tier1_bge_v2_large_asymmetric",
        "script": "train_bge_v2_large_asymmetric_tier1.py",
        "priority": 1,  # Highest priority - fully compliant
        "description": "BGE-v2-Large with Asymmetric Encoding (Task A Compliant - Expected: 0.70-0.78 nDCG@10)"
    },
    {
        "name": "tier1_listwise_ndcg_reranker",
        "script": "train_listwise_ndcg_reranker_tier1.py",
        "priority": 1,  # Highest priority - listwise nDCG@10 reranker (ListT5-style)
        "description": "Hybrid top-100 retrieval + listwise reranking directly targeting nDCG@10 (ListT5-style, Task A Compliant - Expected: 0.80-0.88 nDCG@10)"
    },
    # NEW: Novel Experiment Ideas (Priority 1 - Highest)
    {
        "name": "tier1_adarewriter",
        "script": "train_adarewriter_tier1.py",
        "priority": 1,  # Highest priority - test-time adaptive query reformulation
        "description": "AdaRewriter: Test-Time Adaptive Query Reformulation with Reward Model (Task A Compliant - Expected: 0.78-0.85 nDCG@10)"
    },
    {
        "name": "tier1_rl_query_augmentation",
        "script": "train_rl_query_augmentation_tier1.py",
        "priority": 1,  # Highest priority - RL-optimized query augmentation
        "description": "RL-Optimized Query Augmentation: Policy optimized by retrieval rewards (Task A Compliant - Expected: 0.80-0.88 nDCG@10)"
    },
    {
        "name": "tier1_rl_query_reformulation",
        "script": "train_rl_query_reformulation_tier1.py",
        "priority": 1,  # Highest priority - ConvSearch-R1 inspired retrieval-aligned RL
        "description": "Retrieval-aligned RL Query Reformulation: No supervised rewrites, directly optimizes retrieval ranking with rank-incentive rewards (Task A Compliant - Expected: 0.75-0.85 nDCG@10)"
    },
    {
        "name": "tier1_retriever_in_loop_rewrite",
        "script": "train_retriever_in_loop_rewrite_tier1.py",
        "priority": 1,  # Highest priority - retriever-in-the-loop rewrite selection
        "description": "Retriever-in-the-loop multi-rewrite bandit: selects rewrites by predicted retrieval gain ΔnDCG (Task A Compliant - Expected: 0.80-0.88 nDCG@10)"
    },
    {
        "name": "tier1_ndcg_metric_aligned",
        "script": "train_ndcg_metric_aligned_tier1.py",
        "priority": 1,  # Highest priority - metric-aligned nDCG training
        "description": "Differentiable nDCG@10 training (Lambda-style) for dense retriever (Task A Compliant - Expected: 0.78-0.86 nDCG@10)"
    },
    {
        "name": "tier1_conversational_negative_mining",
        "script": "train_conversational_negative_mining_tier1.py",
        "priority": 1,  # Highest priority - conversational hard negative mining
        "description": "Conversational Negative Mining: previous-turn relevant, same-entity wrong-facet, and partial-answer negatives with curriculum (Task A Compliant - Expected: 0.78-0.86 nDCG@10)"
    },
    {
        "name": "tier1_hybrid_dynamic_fusion_ndcg",
        "script": "train_hybrid_dynamic_fusion_ndcg_tier1.py",
        "priority": 1,  # Highest priority - dynamic hybrid fusion
        "description": "Hybrid lexical/sparse/dense fusion with per-query learned weights targeting nDCG@10 (Task A Compliant - Expected: 0.80-0.90 nDCG@10)"
    },
    {
        "name": "tier1_passage_query_generation",
        "script": "train_passage_query_generation_tier1.py",
        "priority": 1,  # Highest priority - passage-style query variants
        "description": "Passage-Query Generation: title/definition/procedure-style query variants fused via RRF for documentation-like corpora (Task A Compliant - Expected: 0.78-0.86 nDCG@10)"
    },
    {
        "name": "tier1_combined_dynamic_ensemble",
        "script": "train_combined_dynamic_ensemble_tier1.py",
        "priority": 1,  # Highest priority - combined dense + routing + passage fusion
        "description": "Combined Dynamic Ensemble: dense baseline + domain-conditioned routing + passage-style query fusion via RRF (Task A Compliant - Expected: 0.82-0.92 nDCG@10)"
    },
    {
        "name": "tier1_iterative_clarification_rewrite",
        "script": "train_iterative_clarification_rewrite_tier1.py",
        "priority": 1,  # Highest priority - ICR-inspired clarification-rewrite loop
        "description": "Iterative Clarification-and-Rewrite Loop: Internal clarification hypotheses → multiple rewrites → RRF fusion (Task A Compliant - Expected: 0.78-0.88 nDCG@10)"
    },
    {
        "name": "tier1_domain_conditioned_routing",
        "script": "train_domain_conditioned_routing_tier1.py",
        "priority": 1,  # Highest priority - domain-conditioned routing
        "description": "Domain-Conditioned Retrieval + Routing: domain-specific adapters and routing over base/domain retrievers (Task A Compliant - Expected: 0.78-0.86 nDCG@10)"
    },
    {
        "name": "tier1_eclipse_dimension_importance",
        "script": "train_eclipse_dimension_importance_tier1.py",
        "priority": 1,  # Highest priority - dimension importance estimation
        "description": "ECLIPSE: Contrastive Dimension Importance Estimation (Task A Compliant - Expected: 0.75-0.83 nDCG@10)"
    },
    {
        "name": "tier1_intent_graph_retrieval",
        "script": "train_intent_graph_retrieval_tier1.py",
        "priority": 1,  # Highest priority - intent-driven graph patterns
        "description": "Dual-Retrieval with Intent-Driven Graph Patterns (Task A Compliant - Expected: 0.77-0.85 nDCG@10)"
    },
    {
        "name": "tier1_multi_query_sparse_dense",
        "script": "train_multi_query_sparse_dense_tier1.py",
        "priority": 1,  # Highest priority - multi-query sparse+dense fusion
        "description": "Multi-Query Sparse + Dense Rewrites: Multiple rewrites through sparse and dense retrievers (Task A Compliant - Expected: 0.80-0.88 nDCG@10)"
    },
    {
        "name": "tier1_test_time_scaling",
        "script": "train_test_time_scaling_tier1.py",
        "priority": 1,  # Highest priority - test-time scaling
        "description": "Test-Time Scaling & Iterative Reranking: Iterative refinement with LLM reranking (Task A Compliant - Expected: 0.82-0.90 nDCG@10) ⚠️ LLM outputs scores only"
    },
    {
        "name": "tier1_learned_sparse_splade",
        "script": "train_learned_sparse_splade_tier1.py",
        "priority": 1,  # Highest priority - learned sparse retrieval
        "description": "Learned Sparse Retrieval (SPLADE): Combines lexical and semantic signals (Task A Compliant - Expected: 0.78-0.86 nDCG@10)"
    },
    # Task A - Ultra High Performance Retrieval Experiments (Priority 1 - Highest)
    # NEW: Fully Applicable Experiments from Improvement Analysis
    {
        "name": "enhanced_multistage_llm_scoring",
        "script": "train_enhanced_multistage_llm_scoring_tier1.py",
        "priority": 1,  # Highest priority - 5-stage pipeline with LLM scoring (constrained)
        "description": "Enhanced Multi-Stage Pipeline with LLM-Based Relevance Scoring (Task A - Expected: 0.85-0.92 nDCG@10) ⚠️ CONSTRAINT: LLM outputs scores only, NOT text generation"
    },
    {
        "name": "advanced_contrastive_hierarchical",
        "script": "train_advanced_contrastive_hierarchical_tier1.py",
        "priority": 1,  # Highest priority - advanced contrastive learning
        "description": "Advanced Contrastive Learning with Hierarchical Loss (Task A - Expected: 0.70-0.80 nDCG@10)"
    },
    {
        "name": "model_scaling_bge_v2",
        "script": "train_model_scaling_bge_v2_tier1.py",
        "priority": 1,  # Highest priority - model scaling improvements
        "description": "Model Scaling Improvements (BGE-v2-large, longer training, domain-specific) (Task A - Expected: 0.61-0.66 nDCG@10)"
    },
    {
        "name": "specialized_ensemble_meta_learner",
        "script": "train_specialized_ensemble_meta_learner_tier1.py",
        "priority": 1,  # Highest priority - specialized ensemble
        "description": "Specialized Ensemble with Meta-Learner (Task A - Expected: 0.75-0.85 nDCG@10)"
    },
    {
        "name": "llm_query_expansion_constrained",
        "script": "train_llm_query_expansion_constrained_tier1.py",
        "priority": 1,  # Highest priority - LLM query expansion (constrained)
        "description": "LLM-Powered Query Expansion (Task A - Expected: 0.65-0.75 nDCG@10) ⚠️ CONSTRAINT: LLM generates query expansions only, NOT text responses"
    },
    {
        "name": "task_a_enhanced_graph_construction",
        "script": "train_enhanced_graph_construction_tier1.py",
        "priority": 2,  # High priority - enhanced graph construction
        "description": "Enhanced Graph Construction and GNN Architectures (Task A - Expected: 0.60-0.75 nDCG@10)"
    },
    {
        "name": "task_a_rlhf_retrieval_constrained",
        "script": "train_rlhf_retrieval_constrained_tier1.py",
        "priority": 2,  # High priority - RLHF for retrieval (constrained)
        "description": "RLHF for Retrieval (Task A - Expected: 0.72-0.82 nDCG@10) ⚠️ CONSTRAINT: RLHF optimizes retrieval metrics only, NOT generation quality"
    },
    {
        "name": "task_a_advanced_training_techniques",
        "script": "train_advanced_training_techniques_tier1.py",
        "priority": 2,  # High priority - advanced training techniques
        "description": "Advanced Training Techniques (hard negatives, curriculum, adversarial) (Task A - Expected: 0.59-0.63 nDCG@10)"
    },
    {
        "name": "task_a_enhanced_differentiable_pipeline",
        "script": "train_enhanced_differentiable_pipeline_tier1.py",
        "priority": 2,  # High priority - enhanced differentiable pipeline
        "description": "Enhanced Differentiable End-to-End Pipeline (Task A - Expected: 0.70-0.85 nDCG@10)"
    },
    {
        "name": "multistage_hierarchical_retrieval",
        "script": "train_multistage_hierarchical_retrieval_tier1.py",
        "priority": 1,  # Highest priority - most promising for Task A
        "description": "Multi-Stage Hierarchical Retrieval with Cross-Encoder Reranking (Task A - Expected: 0.65-0.75 nDCG@10)"
    },
    {
        "name": "hybrid_dynamic_fusion",
        "script": "train_hybrid_dynamic_fusion_tier1.py",
        "priority": 1,  # Highest priority - high expected performance
        "description": "Hybrid Dense-Sparse with Learned Dynamic Fusion (Task A - Expected: 0.72-0.82 nDCG@10)"
    },
    {
        "name": "task_a_optimized_hybrid_reranking",
        "script": "train_optimized_hybrid_reranking_tier1.py",
        "priority": 1,  # Highest priority - all improvements combined
        "description": "Optimized Hybrid Search with Reranking - All Improvements Combined (Task A - Expected: 0.60-0.75 nDCG@10). Combines: BGE-large, fine-tuned cross-encoder, learned fusion, query expansion, optimized pipeline"
    },
    {
        "name": "task_a_semantic_chunking_retrieval",
        "script": "train_semantic_chunking_retrieval_tier1.py",
        "priority": 1,  # Highest priority - semantic chunking for better retrieval
        "description": "Semantic Chunking Retrieval: Boundary-Aware Chunking (Task A - Expected: +0.03-0.06 nDCG@10). Chunks at paragraph/sentence boundaries, preserves semantic coherence, proper document ID mapping"
    },
    {
        "name": "tier1_mmr_retrieval",
        "script": "train_mmr_retrieval_tier1.py",
        "priority": 2,  # High priority - diversity-aware retrieval
        "description": "MMR (Maximal Marginal Relevance) Retrieval: Balances relevance and diversity (Task A - Expected: +0.02-0.05 nDCG@10). Reduces redundancy, improves coverage of different aspects"
    },
    {
        "name": "tier1_hyde_retrieval",
        "script": "train_hyde_retrieval_tier1.py",
        "priority": 2,  # High priority - HyDE for semantic gap bridging
        "description": "HyDE (Hypothetical Document Embeddings) Retrieval: Generates hypothetical documents for retrieval (Task A - Expected: +0.03-0.06 nDCG@10). Bridges query-document semantic gap"
    },
    {
        "name": "tier1_cg_rag_retrieval",
        "script": "train_cg_rag_retrieval_tier1.py",
        "priority": 2,  # High priority - CG-RAG for contextualized graph retrieval
        "description": "CG-RAG (Contextualized Graph RAG) Retrieval: Builds knowledge graph, contextualizes based on query/conversation (Task A - Expected: +0.03-0.06 nDCG@10). Captures document relationships and context"
    },
    {
        "name": "tier1_gfm_rag_retrieval",
        "script": "train_gfm_rag_retrieval_tier1.py",
        "priority": 2,  # High priority - GFM-RAG for graph foundation model retrieval
        "description": "GFM-RAG (Graph Foundation Model for RAG) Retrieval: Uses foundation models for graph encoding, applies GNN for graph reasoning (Task A - Expected: +0.04-0.07 nDCG@10). Foundation model understanding of graph structure"
    },
    # Novel Experiments for Tier 1 Conference Publication (Task A Compliant)
    {
        "name": "novel_catr_temporal_retrieval",
        "script": "train_catr_temporal_retrieval_tier1.py",
        "priority": 1,  # Highest priority - novel temporal retrieval
        "description": "CATR: Conversation-Aware Temporal Retrieval - Models temporal dynamics of multi-turn conversations (Task A - Expected: 0.55-0.65 nDCG@10). Novel: Temporal attention, query evolution modeling, information state tracking"
    },
    {
        "name": "novel_uq_uncertainty_retrieval",
        "script": "train_uq_uncertainty_retrieval_tier1.py",
        "priority": 1,  # Highest priority - novel uncertainty quantification
        "description": "UQ-Ret: Uncertainty-Quantified Retrieval - Provides calibrated confidence scores for retrieval (Task A - Expected: 0.50-0.60 nDCG@10). Novel: Ensemble-based uncertainty, confidence calibration, adaptive retrieval"
    },
    {
        "name": "novel_diff_retrieval",
        "script": "train_diff_retrieval_tier1.py",
        "priority": 1,  # Highest priority - novel differentiable retrieval
        "description": "Diff-Ret: Differentiable Retrieval - Fully differentiable pipeline with end-to-end nDCG optimization (Task A - Expected: 0.58-0.68 nDCG@10). Novel: Gumbel-Softmax top-k, direct nDCG loss, gradient flow through retrieval"
    },
    {
        "name": "novel_meta_retrieval",
        "script": "train_meta_retrieval_tier1.py",
        "priority": 1,  # Highest priority - novel meta-learning
        "description": "Meta-Ret: Cross-Domain Meta-Learning - Fast adaptation to new domains with few examples (Task A - Expected: 0.48-0.58 nDCG@10). Novel: MAML framework, few-shot adaptation, cross-domain transfer"
    },
    {
        "name": "novel_causal_retrieval",
        "script": "train_causal_retrieval_tier1.py",
        "priority": 1,  # Highest priority - novel causal inference
        "description": "Causal-Ret: Causal Retrieval - Uses causal inference to understand query-document relationships (Task A - Expected: 0.50-0.60 nDCG@10). Novel: Causal graph construction, intervention analysis, causal regularization"
    },
    {
        "name": "novel_gep_graph_retrieval",
        "script": "train_gep_graph_retrieval_tier1.py",
        "priority": 2,  # High priority - novel graph-enhanced retrieval
        "description": "GEP-Ret: Graph-Enhanced Retrieval - Uses GNNs to propagate entity relationships (Task A - Expected: 0.55-0.65 nDCG@10). Novel: Entity extraction, graph construction, GNN propagation, hybrid fusion"
    },
    {
        "name": "novel_x_explainable_retrieval",
        "script": "train_x_explainable_retrieval_tier1.py",
        "priority": 2,  # High priority - novel explainable retrieval
        "description": "X-Ret: Explainable Retrieval - Provides attention-based explanations (Task A - Expected: 0.52-0.62 nDCG@10). Novel: Cross-attention rationales, contrastive explanations, faithfulness metrics (no text generation)"
    },
    {
        "name": "novel_clcf_contrastive_flow",
        "script": "train_clcf_contrastive_flow_tier1.py",
        "priority": 2,  # High priority - novel contrastive learning on flows
        "description": "CLCF-Ret: Contrastive Learning on Conversation Flows - Applies contrastive learning to conversation flows (Task A - Expected: 0.53-0.63 nDCG@10). Novel: Flow encoding, turn-level representations, flow-aware negatives"
    },
    {
        "name": "cross_encoder_conversation_context",
        "script": "train_cross_encoder_conversation_context_tier1.py",
        "priority": 1,  # Highest priority - high expected performance
        "description": "Transformer-Based Cross-Encoder with Full Conversation Context (Task A - Expected: 0.68-0.78 nDCG@10)"
    },
    {
        "name": "task_a_llm_powered_retrieval",
        "script": "train_llm_powered_retrieval_tier1.py",
        "priority": 2,  # High priority - requires LLM access
        "description": "LLM-Powered Retrieval with In-Context Learning (Task A - Expected: 0.70-0.80 nDCG@10)"
    },
    {
        "name": "differentiable_end_to_end",
        "script": "train_differentiable_end_to_end_tier1.py",
        "priority": 2,  # High priority - complex but powerful
        "description": "Differentiable End-to-End Retrieval Pipeline (Task A - Expected: 0.70-0.80 nDCG@10)"
    },
    {
        "name": "task_a_graph_neural_retrieval",
        "script": "train_graph_neural_retrieval_tier1.py",
        "priority": 2,  # High priority - complex
        "description": "Conversation-Aware Graph Neural Retrieval (Task A - Expected: 0.68-0.78 nDCG@10)"
    },
    {
        "name": "domain_specific_ensemble",
        "script": "train_domain_specific_ensemble_tier1.py",
        "priority": 2,  # High priority
        "description": "Ensemble of Domain-Specific Retrievers (Task A - Expected: 0.75-0.85 nDCG@10)"
    },
    {
        "name": "curriculum_contrastive_learning",
        "script": "train_curriculum_contrastive_tier1.py",
        "priority": 1,  # Highest priority - novel approach with high expected performance
        "description": "Curriculum Contrastive Learning with Query Rewriting (Expected: 0.52-0.56 nDCG@10)"
    },
    {
        "name": "cross_attention_mechanism_query_document_fixed",
        "script": "train_cross_attention_tier1.py",
        "priority": 2,
        "description": "Cross-Attention Query-Document Interaction (fixes applied)"
    },
    {
        "name": "multi_turn_state_tracking",
        "script": "train_state_tracking_tier1.py",
        "priority": 2,
        "description": "Multi-Turn Conversation State Tracking with Retrieval (Expected: 0.51-0.55 nDCG@10)"
    },
    {
        "name": "mixture_of_experts",
        "script": "train_mixture_experts_tier1.py",
        "priority": 2,
        "description": "Mixture of Retrieval Experts (MoRE) (Expected: 0.52-0.56 nDCG@10)"
    },
    {
        "name": "uncertainty_aware_retrieval",
        "script": "train_uncertainty_aware_tier1.py",
        "priority": 3,
        "description": "Uncertainty-Aware Retrieval with Confidence Calibration (Expected: 0.50-0.54 nDCG@10)"
    },
    {
        "name": "query_decomposition_retrieval",
        "script": "train_query_decomposition_tier1.py",
        "priority": 3,
        "description": "Retrieval with Learned Query Decomposition (Expected: 0.50-0.54 nDCG@10)"
    },
    {
        "name": "counterfactual_data_augmentation",
        "script": "train_counterfactual_tier1.py",
        "priority": 3,
        "description": "Contrastive Learning with Counterfactual Augmentation (Expected: 0.51-0.55 nDCG@10)"
    },
    {
        "name": "semantic_drift_adaptation",
        "script": "train_semantic_drift_tier1.py",
        "priority": 3,
        "description": "Semantic Drift Detection with Adaptive Retrieval (Expected: 0.50-0.54 nDCG@10)"
    },
    {
        "name": "multi_granularity_encoding",
        "script": "train_multi_granularity_tier1.py",
        "priority": 3,
        "description": "Multi-Granularity Contrastive Learning (Expected: 0.52-0.56 nDCG@10)"
    },
    {
        "name": "adversarial_curriculum_learning",
        "script": "train_adversarial_curriculum_tier1.py",
        "priority": 4,
        "description": "Adversarial Hard Negative Mining with Curriculum Learning"
    },
    {
        "name": "graph_aware_retrieval",
        "script": "train_graph_aware_tier1.py",
        "priority": 4,
        "description": "Conversation Graph-Aware Retrieval"
    },
    {
        "name": "knowledge_distillation_llm",
        "script": "train_llm_distillation_tier1.py",
        "priority": 4,
        "description": "Knowledge Distillation from Large Language Models"
    },
    {
        "name": "multitask_learning_retrieval",
        "script": "train_multitask_retrieval_tier1.py",
        "priority": 4,
        "description": "Multi-Task Learning for Retrieval"
    },
    # Additional Novel Experiments (from ADDITIONAL_NOVEL_EXPERIMENTS.md)
    {
        "name": "neural_ndcg_optimization",
        "script": "train_neural_ndcg_tier1.py",
        "priority": 1,  # Highest priority - direct nDCG optimization
        "description": "Direct nDCG Optimization with NeuralNDCG (Expected: 0.53-0.57 nDCG@10)"
    },
    {
        "name": "graph_enhanced_reranking",
        "script": "train_graph_enhanced_reranking_tier1.py",
        "priority": 2,  # High priority - novel graph-based approach
        "description": "Graph-Enhanced Adaptive Re-Ranking (GEAR) (Expected: 0.52-0.56 nDCG@10)"
    },
    {
        "name": "hybrid_lexical_semantic_fusion",
        "script": "train_hybrid_lexical_semantic_tier1.py",
        "priority": 3,  # Medium priority - practical hybrid approach
        "description": "Hybrid Lexical-Semantic Retrieval with Learned Fusion (Expected: 0.52-0.56 nDCG@10)"
    },
    {
        "name": "momentum_contrastive_learning",
        "script": "train_momentum_contrastive_tier1.py",
        "priority": 3,  # Medium priority - proven technique
        "description": "Contrastive Learning with Momentum Encoder (Expected: 0.52-0.56 nDCG@10)"
    },
    {
        "name": "memory_augmented_retrieval",
        "script": "train_memory_augmented_tier1.py",
        "priority": 3,  # Medium priority - novel memory approach
        "description": "Memory-Augmented Neural Retrieval (MANR) (Expected: 0.52-0.56 nDCG@10)"
    },
    {
        "name": "temporal_attention_mechanism",
        "script": "train_temporal_attention_tier1.py",
        "priority": 4,  # Lower priority
        "description": "Temporal Attention for Conversation History (Expected: 0.51-0.55 nDCG@10)"
    },
    {
        "name": "reinforcement_learning_adaptive",
        "script": "train_rl_adaptive_retrieval_tier1.py",
        "priority": 4,  # Lower priority - complex implementation
        "description": "Reinforcement Learning for Adaptive Retrieval (Expected: 0.52-0.56 nDCG@10)"
    },
    {
        "name": "meta_learning_adaptation",
        "script": "train_meta_learning_tier1.py",
        "priority": 4,  # Lower priority
        "description": "Meta-Learning for Fast Domain Adaptation (Expected: 0.51-0.55 nDCG@10)"
    },
    {
        "name": "cross_domain_transfer_learning",
        "script": "train_cross_domain_transfer_tier1.py",
        "priority": 4,  # Lower priority
        "description": "Cross-Domain Transfer Learning with Domain Adversarial Training (Expected: 0.51-0.55 nDCG@10)"
    },
    {
        "name": "prompt_based_retrieval",
        "script": "train_prompt_based_retrieval_tier1.py",
        "priority": 4,  # Lower priority - requires LLM access
        "description": "Prompt-Based Retrieval with In-Context Learning (Expected: 0.52-0.56 nDCG@10)"
    },
    # Tier 1 Research Ideas (from TIER1_RESEARCH_IDEAS.md) - Novel directions for top-tier conferences
    {
        "name": "causal_inference_retrieval",
        "script": "train_causal_inference_tier1.py",
        "priority": 1,  # Highest priority - hot topic, novel application
        "description": "Causal Inference for Multi-Turn Retrieval (Expected: 0.54-0.58 nDCG@10)"
    },
    {
        "name": "foundation_model_distillation",
        "script": "train_foundation_distillation_tier1.py",
        "priority": 1,  # Highest priority - foundation models are hot
        "description": "Foundation Model Distillation for Retrieval (Expected: 0.54-0.58 nDCG@10)"
    },
    {
        "name": "learned_index_structures",
        "script": "train_learned_indices_tier1.py",
        "priority": 1,  # Highest priority - cutting-edge, novel for IR
        "description": "Learned Indices for Neural Retrieval (Expected: 0.53-0.57 nDCG@10)"
    },
    {
        "name": "differentiable_retrieval",
        "script": "train_differentiable_retrieval_tier1.py",
        "priority": 2,  # High priority - addresses fundamental limitation
        "description": "Differentiable End-to-End Retrieval Pipeline (Expected: 0.53-0.57 nDCG@10)"
    },
    {
        "name": "retrieval_as_generation",
        "script": "train_retrieval_as_generation_tier1.py",
        "priority": 2,  # High priority - paradigm shift
        "description": "Retrieval as Generation (RAG-Retrieval) (Expected: 0.54-0.58 nDCG@10)"
    },
    {
        "name": "reinforcement_learning_human_feedback",
        "script": "train_rlhf_retrieval_tier1.py",
        "priority": 2,  # High priority - RLHF is hot topic
        "description": "RLHF for Retrieval (Expected: 0.54-0.58 nDCG@10)"
    },
    {
        "name": "synthetic_data_augmentation",
        "script": "train_synthetic_data_tier1.py",
        "priority": 2,  # High priority - data efficiency
        "description": "Synthetic Data Generation for Retrieval (Expected: 0.53-0.57 nDCG@10)"
    },
    {
        "name": "knowledge_graph_enhanced",
        "script": "train_knowledge_graph_tier1.py",
        "priority": 2,  # High priority - proven in NLP
        "description": "Knowledge Graph-Enhanced Retrieval (Expected: 0.53-0.57 nDCG@10)"
    },
    {
        "name": "neural_architecture_search",
        "script": "train_nas_retrieval_tier1.py",
        "priority": 3,  # Medium priority - AutoML for retrieval
        "description": "Neural Architecture Search for Retrieval (Expected: 0.53-0.57 nDCG@10)"
    },
    {
        "name": "continual_learning_adaptation",
        "script": "train_continual_learning_tier1.py",
        "priority": 3,  # Medium priority - practical for deployment
        "description": "Continual Learning for Retrieval (Expected: 0.52-0.56 nDCG@10)"
    },
    {
        "name": "explainable_retrieval",
        "script": "train_explainable_retrieval_tier1.py",
        "priority": 3,  # Medium priority - interpretability
        "description": "Explainable Retrieval with Attention Visualization (Expected: 0.52-0.56 nDCG@10)"
    },
    {
        "name": "adversarial_robustness_training",
        "script": "train_adversarial_robustness_tier1.py",
        "priority": 3,  # Medium priority - security/robustness
        "description": "Adversarial Robustness for Retrieval (Expected: 0.52-0.56 nDCG@10)"
    },
    # NEW: Strategy-compliant experiments from STRATEGY_FOR_NDCG_0.90_PLUS_COMPLIANT.md
    {
        "name": "task_a_openrag_retrieval_only",
        "script": "train_openrag_retrieval_only_tier1.py",
        "priority": 1,  # Highest priority - retrieval-only end-to-end optimization
        "description": "OpenRAG-Style Retrieval-Only End-to-End Optimization (Task A - Expected: 0.55-0.60 nDCG@10) ✅ Uses relevance labels only, no generation feedback"
    },
    {
        "name": "task_a_metarag_retrieval",
        "script": "train_metarag_retrieval_tier1.py",
        "priority": 1,  # Highest priority - metacognitive retrieval quality assessment
        "description": "MetaRAG: Retrieval Quality Self-Assessment (Task A - Expected: 0.58-0.63 nDCG@10) ✅ Evaluates retrieval quality only, not generation"
    },
    {
        "name": "task_a_transform_retrieval",
        "script": "train_transform_retrieval_tier1.py",
        "priority": 2,  # High priority - entailment alignment
        "description": "Transform Retrieval for Textual Entailment (Task A - Expected: 0.56-0.61 nDCG@10) ✅ Pure retrieval optimization"
    },
    {
        "name": "task_a_chainrag_retrieval",
        "script": "train_chainrag_retrieval_tier1.py",
        "priority": 2,  # High priority - multi-hop retrieval
        "description": "ChainRAG: Multi-Hop Retrieval (Retrieval-Only) (Task A - Expected: 0.57-0.62 nDCG@10) ✅ Retrieval reasoning only, no text generation"
    },
    {
        "name": "task_a_probing_rag",
        "script": "train_probing_rag_tier1.py",
        "priority": 2,  # High priority - selective retrieval
        "description": "Probing-RAG: Selective Document Retrieval (Task A - Expected: 0.53-0.58 nDCG@10) ✅ Uses hidden states for retrieval necessity, no generation"
    },
    # NEW: State-of-the-art Embedding & Chunking Techniques (from web research)
    {
        "name": "task_a_late_chunking_retrieval",
        "script": "train_late_chunking_retrieval_tier1.py",
        "priority": 1,  # Highest priority - novel chunking technique with high impact
        "description": "Late Chunking Retrieval: Embed full document first, then chunk (preserves global context) (Task A - Expected: +0.05-0.10 nDCG@10) ✅ Paper: arXiv:2409.04701"
    },
    {
        "name": "task_a_colbert_retrieval",
        "script": "train_colbert_retrieval_tier1.py",
        "priority": 1,  # Highest priority - state-of-the-art multi-vector embeddings
        "description": "ColBERT-Style Multi-Vector Retrieval: Token-level embeddings with MaxSim scoring (Task A - Expected: +0.04-0.08 nDCG@10) ✅ Fine-grained token-to-token matching"
    },
    {
        "name": "task_a_bge_v2_large_asymmetric",
        "script": "train_bge_v2_large_asymmetric_tier1.py",
        "priority": 1,  # Highest priority - largest model with asymmetric encoding
        "description": "BGE-v2-Large with Asymmetric Encoding: State-of-the-art model (560M params) with query/document prompts (Task A - Expected: +0.05-0.10 nDCG@10) ✅ Best embedding model available"
    }
]

@dataclass
class ExperimentStatus:
    """Status of an experiment"""
    name: str
    status: str  # pending, running, completed, failed
    gpu: Optional[int] = None
    pid: Optional[int] = None
    start_time: Optional[str] = None
    error: Optional[str] = None
    retry_count: int = 0

class AutoFixedExperimentsRunner:
    """Automatically starts fixed and pending experiments when GPUs become available"""
    
    def __init__(self, experiments_dir: str = "experiments/retrieval", status_file: str = "auto_fixed_experiments_status.json"):
        self.experiments_dir = pathlib.Path(experiments_dir)
        self.status_file = pathlib.Path(status_file)
        self.experiment_statuses: Dict[str, ExperimentStatus] = {}
        self.max_retries = 3
        self.check_interval = 60  # Check every 1 minute (more responsive)
        self.min_gpu_utilization = 10  # GPU is free if utilization < 10%
        self.min_free_memory_mb = 2000  # At least 2GB free memory
        self.load_status()
        self.running = True
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info("Received shutdown signal, saving status and exiting...")
        self.running = False
        self.save_status()
        sys.exit(0)
    
    def load_status(self):
        """Load previous status if exists"""
        if self.status_file.exists():
            try:
                with open(self.status_file, 'r') as f:
                    data = json.load(f)
                    for name, status_data in data.get('experiment_statuses', {}).items():
                        self.experiment_statuses[name] = ExperimentStatus(**status_data)
                logger.info(f"Loaded status for {len(self.experiment_statuses)} experiments")
            except Exception as e:
                logger.warning(f"Could not load status: {e}")
    
    def save_status(self):
        """Save current status"""
        data = {
            'experiment_statuses': {
                name: {
                    'name': status.name,
                    'status': status.status,
                    'gpu': status.gpu,
                    'pid': status.pid,
                    'start_time': status.start_time,
                    'error': status.error,
                    'retry_count': status.retry_count
                }
                for name, status in self.experiment_statuses.items()
            },
            'last_updated': datetime.now().isoformat()
        }
        with open(self.status_file, 'w') as f:
            json.dump(data, f, indent=2)
        logger.debug("Status saved")
    
    def get_free_gpus(self) -> List[int]:
        """Get list of free GPUs (utilization < threshold and sufficient memory)"""
        # GPUs to exclude from auto-runner (reserved for other purposes)
        EXCLUDED_GPUS = {0, 1}  # Exclude GPUs 0 and 1
        
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=index,utilization.gpu,memory.used,memory.total', 
                 '--format=csv,noheader,nounits'],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode == 0:
                free_gpus = []
                # Get GPUs currently in use by tracked experiments
                used_gpus = set()
                for exp_name, status in self.experiment_statuses.items():
                    if status.status == 'running' and status.gpu is not None:
                        used_gpus.add(status.gpu)
                
                for line in result.stdout.strip().split('\n'):
                    if ',' in line:
                        parts = [p.strip() for p in line.split(',')]
                        if len(parts) >= 4:
                            gpu_id = int(parts[0])
                            
                            # Skip excluded GPUs
                            if gpu_id in EXCLUDED_GPUS:
                                logger.debug(f"GPU {gpu_id} is excluded from auto-runner")
                                continue
                            
                            util = int(parts[1]) if parts[1].isdigit() else 0
                            mem_used = int(parts[2]) if parts[2].isdigit() else 0
                            mem_total = int(parts[3]) if parts[3].isdigit() else 0
                            mem_free = mem_total - mem_used
                            
                            # GPU is free if:
                            # 1. Low utilization AND sufficient free memory
                            # 2. OR not tracked as in use by any experiment
                            is_free = (util < self.min_gpu_utilization and mem_free >= self.min_free_memory_mb)
                            is_not_tracked = (gpu_id not in used_gpus)
                            
                            if is_free or (is_not_tracked and util < 50 and mem_free >= self.min_free_memory_mb):
                                free_gpus.append(gpu_id)
                                logger.debug(f"GPU {gpu_id} is free: {util}% util, {mem_free}MB free")
                return free_gpus
        except Exception as e:
            logger.error(f"Error checking GPU status: {e}")
        return []
    
    def check_experiment_status(self, exp_name: str) -> Tuple[str, Optional[str]]:
        """
        Check if experiment is completed, running, or failed
        Returns: (status, error_message)
        """
        exp_dir = self.experiments_dir / exp_name
        
        # Check if results.json exists and is valid
        results_file = exp_dir / "results.json"
        if results_file.exists():
            try:
                with open(results_file, 'r') as f:
                    results = json.load(f)
                    if results.get('status') == 'failed':
                        error = results.get('error', 'Unknown error')
                        return ('failed', error)
                    elif 'average' in results or 'domains' in results:
                        # Has valid results
                        return ('completed', None)
            except Exception as e:
                logger.warning(f"Error reading results.json for {exp_name}: {e}")
        
        # Check if process is running (both in status file and actually running)
        if exp_name in self.experiment_statuses:
            status = self.experiment_statuses[exp_name]
            if status.pid:
                try:
                    # Check if process is still running
                    os.kill(status.pid, 0)
                    # Also check if it's actually a training process
                    try:
                        result = subprocess.run(['ps', '-p', str(status.pid), '-o', 'comm='], 
                                              capture_output=True, text=True, timeout=5)
                        if 'python' in result.stdout.lower() or 'train_' in result.stdout:
                            return ('running', None)
                        else:
                            # Process exists but not a Python training process
                            return ('failed', 'Process is not a training process')
                    except:
                        return ('running', None)  # Assume running if we can't check
                except OSError:
                    # Process is dead
                    logger.info(f"Process {status.pid} for {exp_name} is no longer running")
                    # Check if it completed successfully (check results.json first)
                    if results_file.exists():
                        try:
                            with open(results_file, 'r') as f:
                                results = json.load(f)
                                # Check if it has valid results (not just error status)
                                if 'average' in results or 'domains' in results:
                                    logger.info(f"Experiment {exp_name} has valid results despite process death")
                                    return ('completed', None)
                                elif results.get('status') == 'failed':
                                    # Has error in results.json
                                    error = results.get('error', 'Unknown error')
                                    return ('failed', error)
                        except Exception as e:
                            logger.warning(f"Error reading results.json for {exp_name}: {e}")
                    
                    # Also check for log files to get more context
                    log_files = list(exp_dir.glob("*.log"))
                    if log_files:
                        # Try to extract error from log
                        try:
                            with open(log_files[0], 'r') as f:
                                log_content = f.read()
                                # Look for common errors in logs
                                if 'Traceback' in log_content or 'Error' in log_content:
                                    # Extract last error line
                                    lines = log_content.split('\n')
                                    for line in reversed(lines[-50:]):  # Check last 50 lines
                                        if 'Error' in line or 'Exception' in line:
                                            return ('failed', f"Process died: {line[:100]}")
                        except:
                            pass
                    
                    return ('failed', 'Process died without completing')
        
        # Check for checkpoint (experiment in progress)
        checkpoint_file = exp_dir / "checkpoint.json"
        if checkpoint_file.exists():
            return ('in_progress', None)
        
        return ('pending', None)
    
    def is_experiment_running(self, exp_name: str) -> bool:
        """Check if experiment is currently running"""
        if exp_name not in self.experiment_statuses:
            return False
        
        status = self.experiment_statuses[exp_name]
        if status.status != 'running' or not status.pid:
            return False
        
        try:
            os.kill(status.pid, 0)
            return True
        except OSError:
            return False
    
    def start_experiment(self, exp_config: Dict, gpu: int, resume: bool = True) -> Optional[int]:
        """
        Start an experiment on the specified GPU
        Returns: PID if successful, None otherwise
        """
        exp_name = exp_config['name']
        script = exp_config['script']
        output_dir = self.experiments_dir / exp_name
        
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Build command
        script_path = pathlib.Path(__file__).parent / script
        if not script_path.exists():
            logger.error(f"Script not found: {script_path}")
            return None
        
        log_file = output_dir / "training.log"
        
        cmd = [
            sys.executable,
            str(script_path),
            "--experiment_name", exp_name,
            "--gpu", str(gpu),
            "--output_dir", str(self.experiments_dir)
        ]
        
        # Handle --no-resume flag if specified in experiment config
        if exp_config.get('no_resume', False):
            cmd.append("--no-resume")
            logger.info(f"Using --no-resume flag for {exp_name} (fresh training)")
        # Otherwise, resume is handled by checking checkpoint existence in the script itself
        
        logger.info(f"Starting {exp_name} on GPU {gpu}...")
        logger.debug(f"Command: {' '.join(cmd)}")
        
        try:
            # Start process in background
            with open(log_file, 'a') as log:
                process = subprocess.Popen(
                    cmd,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    cwd=pathlib.Path(__file__).parent,
                    env=os.environ.copy()
                )
            
            # Update status
            self.experiment_statuses[exp_name] = ExperimentStatus(
                name=exp_name,
                status='running',
                gpu=gpu,
                pid=process.pid,
                start_time=datetime.now().isoformat()
            )
            self.save_status()
            
            logger.info(f"✅ Started {exp_name} on GPU {gpu} (PID: {process.pid})")
            return process.pid
        
        except Exception as e:
            logger.error(f"Failed to start {exp_name}: {e}")
            self.experiment_statuses[exp_name] = ExperimentStatus(
                name=exp_name,
                status='failed',
                error=str(e),
                retry_count=1
            )
            self.save_status()
            return None
    
    def get_experiments_to_run(self) -> List[Dict]:
        """Get list of experiments that need to be run, sorted by priority"""
        all_experiments = FIXED_EXPERIMENTS + PENDING_EXPERIMENTS
        to_run = []
        
        for exp_config in all_experiments:
            exp_name = exp_config['name']
            status, error = self.check_experiment_status(exp_name)
            
            if status == 'completed':
                continue  # Already completed
            
            if status == 'running' or self.is_experiment_running(exp_name):
                continue  # Already running
            
            if status == 'failed':
                # Check retry count
                if exp_name in self.experiment_statuses:
                    retry_count = self.experiment_statuses[exp_name].retry_count
                    if retry_count >= self.max_retries:
                        logger.warning(f"{exp_name} has exceeded max retries ({self.max_retries})")
                        continue
            
            to_run.append(exp_config)
        
        # Sort by priority (lower number = higher priority)
        to_run.sort(key=lambda x: x['priority'])
        return to_run
    
    def monitor_and_start(self):
        """Main monitoring loop"""
        logger.info("=" * 60)
        logger.info("Auto Fixed Experiments Runner Started")
        logger.info("=" * 60)
        logger.info(f"Monitoring {len(FIXED_EXPERIMENTS)} fixed experiments and {len(PENDING_EXPERIMENTS)} pending experiments")
        logger.info(f"Check interval: {self.check_interval} seconds")
        logger.info(f"GPU free threshold: <{self.min_gpu_utilization}% utilization, >{self.min_free_memory_mb}MB free memory")
        logger.info("=" * 60)
        
        iteration = 0
        while self.running:
            iteration += 1
            logger.info(f"\n--- Iteration {iteration} ---")
            logger.info(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            
            # Update status of running experiments
            for exp_name, status in list(self.experiment_statuses.items()):
                if status.status == 'running':
                    new_status, error = self.check_experiment_status(exp_name)
                    if new_status != 'running':
                        logger.info(f"Experiment {exp_name} status changed: {status.status} -> {new_status}")
                        status.status = new_status
                        if error:
                            status.error = error
                        if new_status == 'completed':
                            logger.info(f"✅ {exp_name} completed successfully!")
                        elif new_status == 'failed':
                            status.retry_count += 1
                            logger.warning(f"❌ {exp_name} failed: {error}")
            
            # Detect and track orphaned processes (running but not in status file)
            try:
                result = subprocess.run(['ps', 'aux'], capture_output=True, text=True, timeout=10)
                lines = result.stdout.split('\n')
                tracked_pids = {str(s.pid) for s in self.experiment_statuses.values() if s.pid}
                
                for line in lines:
                    if 'train_' in line and 'tier1' in line and 'python' in line.lower():
                        parts = line.split()
                        if len(parts) > 1:
                            pid = parts[1]
                            if pid not in tracked_pids:
                                # Extract experiment name and GPU
                                exp_name = None
                                gpu = None
                                for i, part in enumerate(parts):
                                    if part == '--experiment_name' and i + 1 < len(parts):
                                        exp_name = parts[i + 1]
                                    elif part == '--gpu' and i + 1 < len(parts):
                                        gpu = int(parts[i + 1])
                                    elif 'experiments/retrieval/' in part:
                                        exp_name = part.split('/')[-1]
                                
                                # Check if experiment name already exists OR if PID is already tracked
                                pid_already_tracked = any(s.pid == int(pid) for s in self.experiment_statuses.values() if s.pid)
                                
                                # Check if same experiment is already running (prevent duplicates)
                                exp_already_running = any(
                                    s.name == exp_name and s.status == 'running' 
                                    for s in self.experiment_statuses.values()
                                )
                                
                                if exp_name and gpu is not None and not exp_already_running and exp_name not in self.experiment_statuses and not pid_already_tracked:
                                    logger.info(f"🔍 Detected orphaned process: {exp_name} (PID: {pid}, GPU: {gpu})")
                                    self.experiment_statuses[exp_name] = ExperimentStatus(
                                        name=exp_name,
                                        status='running',
                                        gpu=gpu,
                                        pid=int(pid),
                                        start_time=datetime.now().isoformat()
                                    )
                                    self.save_status()
                                elif exp_already_running:
                                    logger.warning(f"⚠️  Skipping duplicate: {exp_name} already running (PID: {pid} on GPU {gpu} ignored)")
                                elif exp_name in self.experiment_statuses or pid_already_tracked:
                                    logger.debug(f"Skipping duplicate detection: {exp_name} (PID: {pid}) already tracked")
            except Exception as e:
                logger.debug(f"Error detecting orphaned processes: {e}")
            
            # Get free GPUs
            free_gpus = self.get_free_gpus()
            logger.info(f"Free GPUs: {free_gpus}")
            
            if not free_gpus:
                logger.info("No free GPUs available, waiting...")
            else:
                # Get experiments to run
                experiments_to_run = self.get_experiments_to_run()
                logger.info(f"Experiments to run: {len(experiments_to_run)}")
                
                # Start experiments on free GPUs
                started = 0
                for exp_config in experiments_to_run:
                    if not free_gpus:
                        break
                    
                    exp_name = exp_config['name']
                    status, error = self.check_experiment_status(exp_name)
                    
                    if status in ['running', 'completed']:
                        continue
                    
                    # Additional check: prevent starting if already running anywhere
                    if self.is_experiment_running(exp_name):
                        logger.warning(f"Skipping {exp_name}: already running (detected by process check)")
                        continue
                    
                    # Assign GPU
                    gpu = free_gpus.pop(0)
                    
                    # Determine if we should resume (unless no_resume is set)
                    if exp_config.get('no_resume', False):
                        resume = False  # Force fresh training
                    else:
                        exp_dir = self.experiments_dir / exp_name
                        checkpoint_file = exp_dir / "checkpoint.json"
                        resume = checkpoint_file.exists() and status == 'in_progress'
                    
                    # Start experiment
                    pid = self.start_experiment(exp_config, gpu, resume=resume)
                    if pid:
                        started += 1
                        logger.info(f"Started {exp_name} on GPU {gpu} (resume={resume})")
                    else:
                        # Put GPU back if start failed
                        free_gpus.insert(0, gpu)
                
                if started > 0:
                    logger.info(f"Started {started} experiment(s)")
            
            # Save status
            self.save_status()
            
            # Print summary
            self.print_summary()
            
            # Wait before next check
            if self.running:
                logger.info(f"Waiting {self.check_interval} seconds before next check...")
                time.sleep(self.check_interval)
    
    def print_summary(self):
        """Print summary of experiment statuses"""
        logger.info("\n" + "=" * 60)
        logger.info("Experiment Status Summary")
        logger.info("=" * 60)
        
        by_status = {}
        for exp_name, status in self.experiment_statuses.items():
            s = status.status
            if s not in by_status:
                by_status[s] = []
            by_status[s].append(exp_name)
        
        for status in ['running', 'completed', 'failed', 'pending', 'in_progress']:
            if status in by_status:
                logger.info(f"{status.upper()}: {len(by_status[status])} - {', '.join(by_status[status][:5])}")
                if len(by_status[status]) > 5:
                    logger.info(f"  ... and {len(by_status[status]) - 5} more")
        
        logger.info("=" * 60 + "\n")

def main():
    """Main entry point"""
    runner = AutoFixedExperimentsRunner()
    
    try:
        runner.monitor_and_start()
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
    finally:
        runner.save_status()
        logger.info("Exiting...")

if __name__ == "__main__":
    main()

