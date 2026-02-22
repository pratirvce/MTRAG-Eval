"""
Setup Phase 5 Experiments
Register all recommended experiments with the experiment runner
"""

import json
import pathlib
import subprocess
import sys
from typing import List, Dict

EXPERIMENTS_DIR = "experiments/retrieval"

def create_ensemble_configs() -> List[Dict]:
    """Create configs for ensemble experiments"""
    configs = []
    
    # Ensemble of best domain-specific models
    configs.append({
        "experiment_name": "phase5_ensemble_domain_specific",
        "model_paths": [
            "./models/domain_specific_clapnq",
            "./models/domain_specific_govt",
            "./models/domain_specific_cloud"
        ],
        "ensemble_method": "rrf",
        "domains": ["clapnq", "fiqa", "govt", "cloud"],
        "use_data_splits": True
    })
    
    # Ensemble with weighted average
    configs.append({
        "experiment_name": "phase5_ensemble_weighted",
        "model_paths": [
            "./models/domain_specific_clapnq",
            "./models/domain_specific_govt",
            "./models/domain_specific_cloud",
            "./models/phase2_augmentation"
        ],
        "ensemble_method": "weighted",
        "weights": [0.4, 0.3, 0.2, 0.1],  # Higher weight for better models
        "domains": ["clapnq", "fiqa", "govt", "cloud"],
        "use_data_splits": True
    })
    
    return configs

def create_combined_techniques_configs() -> List[Dict]:
    """Create configs for combined techniques (domain-specific + hard negatives)"""
    configs = []
    domains = ["clapnq", "govt"]  # Start with best performing domains
    
    for domain in domains:
        configs.append({
            "experiment_name": f"phase5_domain_specific_{domain}_hard_negatives",
            "domain": domain,
            "base_model": "BAAI/bge-base-en-v1.5",
            "use_pretrained_multi_domain": True,
            "pretrained_multi_domain_path": "./models/phase1_epochs5",
            "epochs": 5,
            "batch_size": 32,
            "learning_rate": 2e-5,
            "num_hard_negatives": 3,
            "loss_function": "CosineSimilarityLoss",
            "use_validation": True,
            "use_data_splits": True,
            "warmup_steps": 100,
            "evaluation_steps": 200,
            "save_best_model": True,
            "save_checkpoints": True,
            "checkpoint_steps": 1000,
            "seed": 42,
            "output_path": f"./models/domain_specific_{domain}_hard_negatives"
        })
    
    return configs

def create_reranking_configs() -> List[Dict]:
    """Create configs for reranking experiments"""
    configs = []
    
    # Reranking with domain-specific models
    domain_models = {
        "clapnq": "./models/domain_specific_clapnq",
        "govt": "./models/domain_specific_govt",
        "cloud": "./models/domain_specific_cloud"
    }
    
    for domain, model_path in domain_models.items():
        configs.append({
            "experiment_name": f"phase5_reranking_{domain}",
            "base_model_path": model_path,
            "cross_encoder_model": "cross-encoder/ms-marco-MiniLM-L-12-v2",
            "top_k": 100,
            "rerank_top_k": 20,
            "domains": [domain],
            "use_data_splits": True
        })
    
    # Multi-domain reranking
    configs.append({
        "experiment_name": "phase5_reranking_multi_domain",
        "base_model_path": "./models/phase2_augmentation",
        "cross_encoder_model": "cross-encoder/ms-marco-MiniLM-L-12-v2",
        "top_k": 100,
        "rerank_top_k": 20,
        "domains": ["clapnq", "fiqa", "govt", "cloud"],
        "use_data_splits": True
    })
    
    return configs

def create_query_expansion_configs() -> List[Dict]:
    """Create configs for query expansion experiments"""
    configs = []
    
    # Query expansion with best models
    models = [
        ("phase5_query_expansion_clapnq", "./models/domain_specific_clapnq"),
        ("phase5_query_expansion_govt", "./models/domain_specific_govt"),
        ("phase5_query_expansion_multi", "./models/phase2_augmentation")
    ]
    
    for exp_name, model_path in models:
        configs.append({
            "experiment_name": exp_name,
            "model_path": model_path,
            "expansion_method": "synonym",  # Can be enhanced with LLM
            "domains": ["clapnq", "fiqa", "govt", "cloud"],
            "use_data_splits": True
        })
    
    return configs

def create_hybrid_configs() -> List[Dict]:
    """Create configs for hybrid retrieval experiments"""
    configs = []
    
    # Test different alpha values
    alphas = [0.3, 0.5, 0.7]
    models = [
        ("phase5_hybrid_clapnq", "./models/domain_specific_clapnq"),
        ("phase5_hybrid_govt", "./models/domain_specific_govt"),
        ("phase5_hybrid_multi", "./models/phase2_augmentation")
    ]
    
    for exp_base, model_path in models:
        for alpha in alphas:
            configs.append({
                "experiment_name": f"{exp_base}_alpha{alpha}",
                "model_path": model_path,
                "alpha": alpha,
                "domains": ["clapnq", "fiqa", "govt", "cloud"],
                "use_data_splits": True
            })
    
    return configs

def register_experiments():
    """Register all experiments with the experiment runner"""
    all_configs = []
    
    # Collect all configs
    print("Creating experiment configs...")
    all_configs.extend(create_ensemble_configs())
    all_configs.extend(create_combined_techniques_configs())
    all_configs.extend(create_reranking_configs())
    all_configs.extend(create_query_expansion_configs())
    all_configs.extend(create_hybrid_configs())
    
    print(f"Total experiments to register: {len(all_configs)}")
    
    # Save configs and register
    experiments_dir = pathlib.Path(EXPERIMENTS_DIR)
    experiments_dir.mkdir(parents=True, exist_ok=True)
    
    registered = []
    for config in all_configs:
        exp_name = config["experiment_name"]
        exp_dir = experiments_dir / exp_name
        exp_dir.mkdir(parents=True, exist_ok=True)
        
        config_file = exp_dir / "config.json"
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        
        # Register with experiment runner
        try:
            result = subprocess.run(
                [
                    sys.executable,
                    "experiment_runner.py",
                    "--action", "register",
                    "--experiment", exp_name,
                    "--config", str(config_file)
                ],
                capture_output=True,
                text=True,
                cwd=pathlib.Path.cwd()
            )
            
            if result.returncode == 0:
                registered.append(exp_name)
                print(f"✅ Registered: {exp_name}")
            else:
                print(f"❌ Failed to register {exp_name}: {result.stderr}")
        except Exception as e:
            print(f"❌ Error registering {exp_name}: {e}")
    
    print(f"\n✅ Successfully registered {len(registered)}/{len(all_configs)} experiments")
    return registered

def print_summary():
    """Print summary of registered experiments"""
    print("\n" + "="*60)
    print("PHASE 5 EXPERIMENTS SUMMARY")
    print("="*60)
    print("\n1. Ensemble Methods (2 experiments):")
    print("   - phase5_ensemble_domain_specific (RRF)")
    print("   - phase5_ensemble_weighted (Weighted average)")
    
    print("\n2. Combined Techniques (2 experiments):")
    print("   - phase5_domain_specific_clapnq_hard_negatives")
    print("   - phase5_domain_specific_govt_hard_negatives")
    
    print("\n3. Reranking (4 experiments):")
    print("   - phase5_reranking_clapnq")
    print("   - phase5_reranking_govt")
    print("   - phase5_reranking_cloud")
    print("   - phase5_reranking_multi_domain")
    
    print("\n4. Query Expansion (3 experiments):")
    print("   - phase5_query_expansion_clapnq")
    print("   - phase5_query_expansion_govt")
    print("   - phase5_query_expansion_multi")
    
    print("\n5. Hybrid Retrieval (9 experiments):")
    print("   - phase5_hybrid_clapnq_alpha{0.3,0.5,0.7}")
    print("   - phase5_hybrid_govt_alpha{0.3,0.5,0.7}")
    print("   - phase5_hybrid_multi_alpha{0.3,0.5,0.7}")
    
    print("\n" + "="*60)
    print("TOTAL: 20 experiments")
    print("="*60)
    print("\nTo start experiments:")
    print("  python experiment_runner.py --action run_parallel")
    print("\nTo check status:")
    print("  python experiment_runner.py --action status")
    print("\nTo pause all:")
    print("  python experiment_runner.py --action pause --experiment <name>")
    print("\nTo resume:")
    print("  python experiment_runner.py --action resume --experiment <name>")

if __name__ == "__main__":
    print("Setting up Phase 5 experiments...")
    registered = register_experiments()
    print_summary()

