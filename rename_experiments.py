#!/usr/bin/env python3
"""
Safely rename experiments with standard journal-style names.
This script:
1. Renames experiment directories
2. Updates all JSON tracking files
3. Updates config.json files if they reference the experiment name
4. Creates a mapping file for reference
"""

import json
import pathlib
import shutil
import re
from typing import Dict, List, Tuple
from datetime import datetime

# Mapping from old names to new journal-style names
EXPERIMENT_NAME_MAPPING = {
    # Phase 1: Baseline
    "phase1_baseline": "baseline_bge_finetuned",
    "phase1_epochs3": "baseline_3_epochs",
    "phase1_epochs5": "baseline_5_epochs",
    
    # Phase 2: Hyperparameter Optimization
    "phase2_augmentation": "data_augmentation_retrieval",
    "phase2_lr1e5": "learning_rate_1e5",
    "phase2_lr5e5": "learning_rate_5e5",
    "phase2_cosine_loss": "cosine_similarity_loss",
    
    # Phase 3: Reranking
    "phase3_reranking": "cross_encoder_reranking",
    "phase3_hybrid": "hybrid_lexical_semantic_reranking",
    "phase3_hybrid_reranking": "hybrid_reranking_fusion",
    
    # Phase 4: Domain-Specific Fine-tuning
    "phase4_domain_specific_clapnq": "domain_specific_finetuning_clapnq",
    "phase4_domain_specific_fiqa": "domain_specific_finetuning_fiqa",
    "phase4_domain_specific_govt": "domain_specific_finetuning_govt",
    "phase4_domain_specific_cloud": "domain_specific_finetuning_cloud",
    "phase4_domain_specific_all": "domain_specific_finetuning_all_domains",
    "phase4_hard_negatives_5neg": "hard_negatives_mining_5",
    "phase4_hard_negatives_cosine": "hard_negatives_cosine_similarity",
    "phase4_hard_negatives_triplet": "hard_negatives_triplet_loss",
    "phase4_bge_large": "bge_large_model_finetuning",
    
    # Phase 5: Query Expansion and Ensembles
    "phase5_query_expansion_clapnq": "query_expansion_clapnq",
    "phase5_query_expansion_govt": "query_expansion_govt",
    "phase5_query_expansion_multi": "query_expansion_multi_domain",
    "phase5_ensemble_domain_specific": "ensemble_domain_specific_models",
    "phase5_ensemble_weighted": "ensemble_weighted_fusion",
    "phase5_reranking_clapnq": "reranking_clapnq",
    "phase5_reranking_govt": "reranking_govt",
    "phase5_reranking_cloud": "reranking_cloud",
    "phase5_reranking_multi_domain": "reranking_multi_domain",
    "phase5_domain_specific_clapnq_hard_negatives": "domain_specific_clapnq_hard_negatives",
    "phase5_domain_specific_govt_hard_negatives": "domain_specific_govt_hard_negatives",
    "phase5_hybrid_clapnq_alpha0.3": "hybrid_clapnq_alpha_0.3",
    "phase5_hybrid_clapnq_alpha0.5": "hybrid_clapnq_alpha_0.5",
    "phase5_hybrid_clapnq_alpha0.7": "hybrid_clapnq_alpha_0.7",
    "phase5_hybrid_govt_alpha0.3": "hybrid_govt_alpha_0.3",
    "phase5_hybrid_govt_alpha0.5": "hybrid_govt_alpha_0.5",
    "phase5_hybrid_govt_alpha0.7": "hybrid_govt_alpha_0.7",
    "phase5_hybrid_multi_alpha0.3": "hybrid_multi_alpha_0.3",
    "phase5_hybrid_multi_alpha0.5": "hybrid_multi_alpha_0.5",
    "phase5_hybrid_multi_alpha0.7": "hybrid_multi_alpha_0.7",
    
    # Phase 6: LLM-Based Methods
    "phase6_llm_query_expansion_gpt4_multi": "llm_query_expansion_gpt4",
    "phase6_multistage_2stage": "multistage_retrieval_2stage",
    "phase6_multistage_2stage_finetuned": "multistage_retrieval_2stage_finetuned",
    "phase6_cross_encoder_evaluation": "cross_encoder_evaluation",
    "phase6_cross_encoder_finetuned_ensemble": "cross_encoder_finetuned_ensemble",
    
    # Phase 7: Conversation-Aware
    "phase7_conversation_aware_attention": "conversation_aware_attention",
    "phase7_iterative_refinement": "iterative_query_refinement",
    
    # Phase 8
    "phase8_cross_attention_query_document": "cross_attention_query_document",
    "phase8_cross_attention_query_document_fixed": "cross_attention_query_document_fixed",
    
    # Tier 1: Advanced Techniques
    "tier1_contrastive_learning": "contrastive_learning_finetuning",
    "tier1_meta_learning": "meta_learning_adaptation",
    "tier1_adversarial_curriculum": "adversarial_curriculum_learning",
    "tier1_cross_encoder_evaluation": "cross_encoder_reranking_evaluation",
    "tier1_cross_encoder_large": "cross_encoder_large_model",
    "tier1_cross_encoder_domain_specific": "cross_encoder_domain_specific",
    "tier1_cross_encoder_finetuned": "cross_encoder_finetuned",
    "tier1_hierarchical_multigranularity": "hierarchical_multigranularity_retrieval",
    "tier1_iterative_refinement_improved": "iterative_refinement_improved",
    "tier1_cross_attention": "cross_attention_mechanism",
    "tier1_cross_attention_fixed_v3": "cross_attention_mechanism_v3",
    "tier1_cross_attention_query_document": "cross_attention_query_document",
    "tier1_cross_attention_query_document_fixed": "cross_attention_query_document_fixed",
    "tier1_cross_attention_rerun": "cross_attention_rerun",
    "tier1_cross_attention_rerun_fixed": "cross_attention_rerun_fixed",
    "tier1_cross_attention_rerun_fixed_v2": "cross_attention_rerun_fixed_v2",
    "tier1_llm_distillation": "knowledge_distillation_llm",
    "tier1_learning_to_rank_listwise": "learning_to_rank_listwise",
    "tier1_learning_to_rank_listwise_fixed": "learning_to_rank_listwise_fixed",
    "tier1_synthetic_data": "synthetic_data_augmentation",
    "tier1_cross_domain_transfer": "cross_domain_transfer_learning",
    "tier1_semantic_drift": "semantic_drift_adaptation",
    "tier1_causal_inference": "causal_inference_retrieval",
    "tier1_causal_inference_fixed": "causal_inference_retrieval_fixed",
    "tier1_explainable_retrieval": "explainable_retrieval",
    "tier1_counterfactual_augmentation": "counterfactual_data_augmentation",
    "tier1_memory_augmented": "memory_augmented_retrieval",
    "tier1_learned_indices": "learned_index_structures",
    "tier1_learned_indices_fixed": "learned_index_structures_fixed",
    "tier1_momentum_contrastive": "momentum_contrastive_learning",
    "tier1_rlhf_retrieval": "reinforcement_learning_human_feedback",
    "tier1_temporal_attention": "temporal_attention_mechanism",
    "tier1_adversarial_robustness": "adversarial_robustness_training",
    "tier1_prompt_based_retrieval": "prompt_based_retrieval",
    "tier1_enhanced_contrastive": "enhanced_contrastive_learning",
    "tier1_enhanced_contrastive_hardnegatives": "enhanced_contrastive_hard_negatives",
    "tier1_enhanced_contrastive_hardnegatives_fixed": "enhanced_contrastive_hard_negatives_fixed",
    "tier1_enhanced_contrastive_improved": "enhanced_contrastive_improved",
    "tier1_differentiable_retrieval": "differentiable_retrieval",
    "tier1_differentiable_retrieval_fixed": "differentiable_retrieval_fixed",
    "tier1_rl_adaptive_retrieval": "reinforcement_learning_adaptive",
    "tier1_rl_adaptive": "reinforcement_learning_adaptive_short",
    "tier1_nas_retrieval": "neural_architecture_search",
    "tier1_graph_enhanced_reranking": "graph_enhanced_reranking",
    "tier1_graph_enhanced_reranking_fixed": "graph_enhanced_reranking_fixed",
    "tier1_multi_granularity": "multi_granularity_encoding",
    "tier1_hybrid_lexical_semantic": "hybrid_lexical_semantic_fusion",
    "tier1_uncertainty_aware": "uncertainty_aware_retrieval",
    "tier1_uncertainty_aware_fixed": "uncertainty_aware_retrieval_fixed",
    "tier1_foundation_distillation": "foundation_model_distillation",
    "tier1_foundation_distillation_fixed": "foundation_model_distillation_fixed",
    "tier1_query_decomposition": "query_decomposition_retrieval",
    "tier1_knowledge_graph": "knowledge_graph_enhanced",
    "tier1_curriculum_contrastive": "curriculum_contrastive_learning",
    "tier1_curriculum_contrastive_fixed": "curriculum_contrastive_learning_fixed",
    "tier1_mixture_experts": "mixture_of_experts",
    "tier1_mixture_experts_fixed": "mixture_of_experts_fixed",
    "tier1_continual_learning": "continual_learning_adaptation",
    "tier1_retrieval_as_generation": "retrieval_as_generation",
    "tier1_multi_turn_state_tracking": "multi_turn_state_tracking",
    "tier1_multi_turn_state_tracking_fixed": "multi_turn_state_tracking_fixed",
    "tier1_graph_aware_retrieval": "graph_aware_retrieval",
    "tier1_graph_aware": "graph_aware_retrieval_short",
    "tier1_pseudo_relevance_feedback": "pseudo_relevance_feedback",
    "tier1_qdit_transformer": "qdit_transformer_architecture",
    "tier1_multitask_retrieval": "multitask_learning_retrieval",
    "tier1_ensemble_best_methods": "ensemble_best_performing",
    "tier1_ensemble_best": "ensemble_best_performing_short",
    "tier1_hierarchical_routing": "hierarchical_routing",
    "tier1_learned_rrf": "learned_reciprocal_rank_fusion",
    "tier1_neural_ndcg": "neural_ndcg_optimization",
    "tier1_multistage": "multistage_retrieval",
    "tier1_multistage_2stage": "multistage_retrieval_2stage",
    "tier1_multistage_3stage_finetuned": "multistage_retrieval_3stage_finetuned",
    "tier1_large_model": "large_model_finetuning",
    "tier1_learning_to_rank": "learning_to_rank",
    "tier1_llm_query_expansion": "llm_query_expansion",
    
    # Best Paper Methods
    "best_paper_large_model_finetuning": "large_model_finetuning",
    "best_paper_adversarial_curriculum": "adversarial_curriculum_learning",
    "best_paper_meta_learning": "meta_learning_adaptation",
    "best_paper_meta_learning_fixed": "meta_learning_adaptation_fixed",
    "best_paper_hierarchical_routing": "hierarchical_routing",
    "best_paper_hierarchical_routing_fixed": "hierarchical_routing_fixed",
    "best_paper_llm_distillation": "llm_distillation",
    "best_paper_graph_aware_retrieval": "graph_aware_retrieval",
    "best_paper_learned_rrf": "learned_reciprocal_rank_fusion",
    "best_paper_rl_adaptive_retrieval": "rl_adaptive_retrieval",
    "best_paper_rl_adaptive_retrieval_fixed": "rl_adaptive_retrieval_fixed",
    "best_paper_temporal_memory": "temporal_memory_retrieval",
    "best_paper_temporal_memory_fixed": "temporal_memory_retrieval_fixed",
    "best_paper_multitask_retrieval": "multitask_learning",
    
    # Task A
    "task_a_hybrid_dynamic_fusion": "hybrid_dynamic_fusion",
    "task_a_multistage_hierarchical_retrieval": "multistage_hierarchical_retrieval",
    "task_a_cross_encoder_conversation_context": "cross_encoder_conversation_context",
    "task_a_differentiable_end_to_end": "differentiable_end_to_end",
    "task_a_advanced_contrastive_hierarchical": "advanced_contrastive_hierarchical",
    "task_a_domain_specific_ensemble": "domain_specific_ensemble",
    "task_a_enhanced_multistage_llm_scoring": "enhanced_multistage_llm_scoring",
    "task_a_llm_query_expansion_constrained": "llm_query_expansion_constrained",
    "task_a_model_scaling_bge_v2": "model_scaling_bge_v2",
    "task_a_specialized_ensemble_meta_learner": "specialized_ensemble_meta_learner",
    
    # Other
    "multistage_retrieval": "multistage_retrieval_base",
    "tier2_multistage_3stage": "multistage_retrieval_3stage",
    "tier2_ensemble_advanced": "ensemble_advanced_methods",
}

EXPERIMENTS_DIR = pathlib.Path("experiments/retrieval")
STATUS_FILES = [
    "experiment_status.json",
    "stopped_experiments_resume_info.json",
    "fixed_experiments_status.json",
    "tier1_experiments_status.json",
    "auto_experiments_status.json",
    "auto_fixed_experiments_status.json",
]

def update_json_file(filepath: pathlib.Path, mapping: Dict[str, str], dry_run: bool = False):
    """Update experiment names in a JSON file."""
    if not filepath.exists():
        return False
    
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        updated = False
        new_data = {}
        
        if isinstance(data, dict):
            for key, value in data.items():
                if key in mapping:
                    new_key = mapping[key]
                    new_data[new_key] = value
                    if isinstance(value, dict) and 'experiment_name' in value:
                        value['experiment_name'] = new_key
                    # Update paths in the value
                    if isinstance(value, dict):
                        for path_key in ['config_path', 'log_path', 'output_dir', 'checkpoint_path']:
                            if path_key in value and value[path_key]:
                                old_path = value[path_key]
                                for old_name, new_name in mapping.items():
                                    if old_name in str(old_path):
                                        value[path_key] = str(old_path).replace(old_name, new_name)
                                        updated = True
                    updated = True
                else:
                    new_data[key] = value
        
        elif isinstance(data, dict) and 'stopped_experiments' in data:
            # Handle stopped_experiments_resume_info.json format
            new_data = data.copy()
            if 'stopped_experiments' in new_data:
                updated_exps = []
                for exp in new_data['stopped_experiments']:
                    if exp.get('experiment_name') in mapping:
                        exp['experiment_name'] = mapping[exp['experiment_name']]
                        # Update output_dir
                        if 'output_dir' in exp and exp['output_dir']:
                            for old_name, new_name in mapping.items():
                                if old_name in exp['output_dir']:
                                    exp['output_dir'] = exp['output_dir'].replace(old_name, new_name)
                                    updated = True
                        updated = True
                    updated_exps.append(exp)
                new_data['stopped_experiments'] = updated_exps
        
        if updated and not dry_run:
            with open(filepath, 'w') as f:
                json.dump(new_data, f, indent=2)
            return True
        elif updated:
            return True
        
        return False
    except Exception as e:
        print(f"Error updating {filepath}: {e}")
        return False

def update_config_file(config_path: pathlib.Path, mapping: Dict[str, str], dry_run: bool = False):
    """Update experiment name in config.json if present."""
    if not config_path.exists():
        return False
    
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        updated = False
        if 'experiment_name' in config and config['experiment_name'] in mapping:
            if not dry_run:
                config['experiment_name'] = mapping[config['experiment_name']]
            updated = True
        
        # Update any paths in config
        for key, value in config.items():
            if isinstance(value, str) and 'experiments/retrieval' in value:
                for old_name, new_name in mapping.items():
                    if old_name in value:
                        if not dry_run:
                            config[key] = value.replace(old_name, new_name)
                        updated = True
                        break
        
        if updated and not dry_run:
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
        
        return updated
    except Exception as e:
        print(f"Error updating config {config_path}: {e}")
        return False

def rename_experiment_directory(old_name: str, new_name: str, dry_run: bool = False) -> bool:
    """Rename an experiment directory."""
    old_dir = EXPERIMENTS_DIR / old_name
    new_dir = EXPERIMENTS_DIR / new_name
    
    if not old_dir.exists():
        return False
    
    if new_dir.exists():
        print(f"⚠️  Warning: {new_dir} already exists, skipping {old_name}")
        return False
    
    if not dry_run:
        try:
            old_dir.rename(new_dir)
            print(f"✅ Renamed: {old_name} → {new_name}")
            return True
        except Exception as e:
            print(f"❌ Error renaming {old_name}: {e}")
            return False
    else:
        print(f"[DRY RUN] Would rename: {old_name} → {new_name}")
        return True

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Rename experiments with journal-style names')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be renamed without doing it')
    parser.add_argument('--mapping-file', type=str, help='Custom mapping JSON file (optional)')
    parser.add_argument('--experiment', type=str, help='Rename a single experiment (old_name:new_name)')
    
    args = parser.parse_args()
    
    # Load custom mapping if provided
    mapping = EXPERIMENT_NAME_MAPPING.copy()
    if args.mapping_file:
        with open(args.mapping_file, 'r') as f:
            mapping.update(json.load(f))
    
    # Handle single experiment rename
    if args.experiment:
        if ':' not in args.experiment:
            print("Error: --experiment format should be old_name:new_name")
            return
        old_name, new_name = args.experiment.split(':', 1)
        mapping = {old_name: new_name}
    
    print("=" * 80)
    print("EXPERIMENT RENAMING TOOL")
    print("=" * 80)
    print(f"Mode: {'DRY RUN' if args.dry_run else 'LIVE'}")
    print(f"Experiments to rename: {len(mapping)}")
    print()
    
    # Find which experiments actually exist
    existing_experiments = {name: new_name for name, new_name in mapping.items() 
                           if (EXPERIMENTS_DIR / name).exists()}
    
    print(f"Found {len(existing_experiments)} existing experiments to rename")
    print()
    
    # Step 1: Rename directories
    print("Step 1: Renaming experiment directories...")
    renamed = 0
    for old_name, new_name in existing_experiments.items():
        if rename_experiment_directory(old_name, new_name, args.dry_run):
            renamed += 1
    print(f"Renamed {renamed} directories\n")
    
    # Step 2: Update JSON status files
    print("Step 2: Updating JSON status files...")
    status_updated = 0
    for status_file in STATUS_FILES:
        filepath = pathlib.Path(status_file)
        if update_json_file(filepath, mapping, args.dry_run):
            status_updated += 1
            print(f"  ✅ Updated {status_file}")
    print(f"Updated {status_updated} status files\n")
    
    # Step 3: Update config.json files in renamed directories
    print("Step 3: Updating config.json files...")
    config_updated = 0
    for old_name, new_name in existing_experiments.items():
        config_path = EXPERIMENTS_DIR / new_name / "config.json"
        if update_config_file(config_path, mapping, args.dry_run):
            config_updated += 1
    print(f"Updated {config_updated} config files\n")
    
    # Save mapping for reference
    mapping_file = pathlib.Path("experiment_name_mapping.json")
    if not args.dry_run:
        mapping_data = {
            "created_at": datetime.now().isoformat(),
            "mapping": mapping,
            "renamed_count": len(existing_experiments)
        }
        with open(mapping_file, 'w') as f:
            json.dump(mapping_data, f, indent=2)
        print(f"✅ Saved mapping to {mapping_file}")
    
    print()
    print("=" * 80)
    print("RENAMING COMPLETE!")
    print("=" * 80)
    print("\n⚠️  IMPORTANT NOTES:")
    print("1. Results files (results.json) are NOT affected - they don't contain experiment names")
    print("2. Resume functionality will work with new names")
    print("3. Check experiment_status.json and resume files to verify updates")
    print("4. You may need to restart any running experiments with new names")

if __name__ == "__main__":
    main()

