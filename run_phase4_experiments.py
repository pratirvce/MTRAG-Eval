"""
Phase 4: Advanced Retrieval Experiments
This script runs advanced experiments to improve retrieval scores:
1. Hard negative mining
2. Domain-specific fine-tuning
3. Larger base models
4. Advanced loss functions
5. Multi-stage training
"""

import json
import pathlib
import subprocess
import sys
import logging
from datetime import datetime

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)

EXPERIMENTS_DIR = pathlib.Path("experiments/retrieval")

# Phase 4 Experiment Configurations
PHASE4_EXPERIMENTS = {
    "phase4_hard_negatives_cosine": {
        "description": "Hard negative mining with CosineSimilarityLoss",
        "script": "train_advanced_bge.py",
        "config": {
            "experiment_name": "phase4_hard_negatives_cosine",
            "base_model": "BAAI/bge-base-en-v1.5",
            "epochs": 5,
            "batch_size": 32,
            "learning_rate": 2e-5,
            "loss_function": "CosineSimilarityLoss",
            "num_hard_negatives": 3,
            "use_hard_negatives": True,
            "use_validation": True,
            "use_data_splits": True,
            "warmup_steps": 100,
            "evaluation_steps": 500,
            "save_best_model": True,
            "save_checkpoints": True,
            "checkpoint_steps": 1000,
            "seed": 42
        }
    },
    "phase4_hard_negatives_triplet": {
        "description": "Hard negative mining with TripletLoss",
        "script": "train_advanced_bge.py",
        "config": {
            "experiment_name": "phase4_hard_negatives_triplet",
            "base_model": "BAAI/bge-base-en-v1.5",
            "epochs": 5,
            "batch_size": 32,
            "learning_rate": 2e-5,
            "loss_function": "TripletLoss",
            "num_hard_negatives": 3,
            "use_hard_negatives": True,
            "use_validation": True,
            "use_data_splits": True,
            "warmup_steps": 100,
            "evaluation_steps": 500,
            "save_best_model": True,
            "save_checkpoints": True,
            "checkpoint_steps": 1000,
            "seed": 42
        }
    },
    "phase4_hard_negatives_5neg": {
        "description": "Hard negatives with 5 negatives per positive",
        "script": "train_advanced_bge.py",
        "config": {
            "experiment_name": "phase4_hard_negatives_5neg",
            "base_model": "BAAI/bge-base-en-v1.5",
            "epochs": 5,
            "batch_size": 24,  # Smaller batch due to more negatives
            "learning_rate": 2e-5,
            "loss_function": "CosineSimilarityLoss",
            "num_hard_negatives": 5,
            "use_hard_negatives": True,
            "use_validation": True,
            "use_data_splits": True,
            "warmup_steps": 100,
            "evaluation_steps": 500,
            "save_best_model": True,
            "save_checkpoints": True,
            "checkpoint_steps": 1000,
            "seed": 42
        }
    },
    "phase4_bge_large": {
        "description": "Larger BGE model (bge-large-en-v1.5)",
        "script": "train_advanced_bge.py",
        "config": {
            "experiment_name": "phase4_bge_large",
            "base_model": "BAAI/bge-large-en-v1.5",
            "epochs": 3,  # Fewer epochs for larger model
            "batch_size": 16,  # Smaller batch for larger model
            "learning_rate": 1e-5,  # Lower LR for larger model
            "loss_function": "MultipleNegativesRankingLoss",
            "num_hard_negatives": 3,
            "use_hard_negatives": True,
            "use_validation": True,
            "use_data_splits": True,
            "warmup_steps": 100,
            "evaluation_steps": 300,
            "save_best_model": True,
            "save_checkpoints": True,
            "checkpoint_steps": 1000,
            "seed": 42
        }
    },
    "phase4_domain_specific_clapnq": {
        "description": "Domain-specific model for ClapNQ",
        "script": "train_domain_specific_bge.py",
        "config": {
            "domain": "clapnq",
            "base_model": "BAAI/bge-base-en-v1.5",
            "epochs": 7,
            "batch_size": 32,
            "learning_rate": 1e-5,
            "use_pretrained_multi_domain": True,
            "pretrained_multi_domain_path": "./models/phase1_epochs5",
            "use_validation": True,
            "use_data_splits": True,
            "warmup_steps": 50,
            "evaluation_steps": 200,
            "save_best_model": True,
            "seed": 42
        }
    },
    "phase4_domain_specific_fiqa": {
        "description": "Domain-specific model for FiQA (worst performing)",
        "script": "train_domain_specific_bge.py",
        "config": {
            "domain": "fiqa",
            "base_model": "BAAI/bge-base-en-v1.5",
            "epochs": 7,
            "batch_size": 32,
            "learning_rate": 1e-5,
            "use_pretrained_multi_domain": True,
            "pretrained_multi_domain_path": "./models/phase1_epochs5",
            "use_validation": True,
            "use_data_splits": True,
            "warmup_steps": 50,
            "evaluation_steps": 200,
            "save_best_model": True,
            "seed": 42
        }
    },
    "phase4_domain_specific_govt": {
        "description": "Domain-specific model for Govt",
        "script": "train_domain_specific_bge.py",
        "config": {
            "domain": "govt",
            "base_model": "BAAI/bge-base-en-v1.5",
            "epochs": 7,
            "batch_size": 32,
            "learning_rate": 1e-5,
            "use_pretrained_multi_domain": True,
            "pretrained_multi_domain_path": "./models/phase1_epochs5",
            "use_validation": True,
            "use_data_splits": True,
            "warmup_steps": 50,
            "evaluation_steps": 200,
            "save_best_model": True,
            "seed": 42
        }
    },
    "phase4_domain_specific_cloud": {
        "description": "Domain-specific model for Cloud",
        "script": "train_domain_specific_bge.py",
        "config": {
            "domain": "cloud",
            "base_model": "BAAI/bge-base-en-v1.5",
            "epochs": 7,
            "batch_size": 32,
            "learning_rate": 1e-5,
            "use_pretrained_multi_domain": True,
            "pretrained_multi_domain_path": "./models/phase1_epochs5",
            "use_validation": True,
            "use_data_splits": True,
            "warmup_steps": 50,
            "evaluation_steps": 200,
            "save_best_model": True,
            "seed": 42
        }
    }
}

def run_experiment(exp_name, exp_config):
    """Run a single experiment."""
    logging.info(f"\n{'='*60}")
    logging.info(f"Starting experiment: {exp_name}")
    logging.info(f"Description: {exp_config['description']}")
    logging.info(f"{'='*60}\n")
    
    # Create experiment directory
    exp_dir = EXPERIMENTS_DIR / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config
    config_file = exp_dir / "config.json"
    with open(config_file, 'w') as f:
        json.dump(exp_config['config'], f, indent=2)
    
    # Prepare command
    script = exp_config['script']
    config_path = str(config_file)
    
    # Build command
    cmd = [
        sys.executable, script,
        "--config", config_path
    ]
    
    # Add domain if specified
    if 'domain' in exp_config['config']:
        cmd.extend(["--domain", exp_config['config']['domain']])
    
    # Run training
    log_file = exp_dir / "training.log"
    try:
        with open(log_file, 'w') as f:
            result = subprocess.run(
                cmd,
                stdout=f,
                stderr=subprocess.STDOUT,
                check=True
            )
        logging.info(f"✅ Experiment {exp_name} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        logging.error(f"❌ Experiment {exp_name} failed: {e}")
        return False

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Run Phase 4 advanced experiments')
    parser.add_argument('--experiment', type=str, help='Specific experiment to run (or "all")')
    parser.add_argument('--list', action='store_true', help='List all available experiments')
    
    args = parser.parse_args()
    
    if args.list:
        print("\nAvailable Phase 4 Experiments:")
        print("="*60)
        for exp_name, exp_config in PHASE4_EXPERIMENTS.items():
            print(f"\n{exp_name}:")
            print(f"  Description: {exp_config['description']}")
            print(f"  Script: {exp_config['script']}")
        return
    
    if args.experiment == "all" or args.experiment is None:
        # Run all experiments
        results = {}
        for exp_name, exp_config in PHASE4_EXPERIMENTS.items():
            results[exp_name] = run_experiment(exp_name, exp_config)
        
        # Summary
        print("\n" + "="*60)
        print("Experiment Summary:")
        print("="*60)
        for exp_name, success in results.items():
            status = "✅ SUCCESS" if success else "❌ FAILED"
            print(f"{exp_name}: {status}")
    else:
        if args.experiment not in PHASE4_EXPERIMENTS:
            print(f"Error: Unknown experiment '{args.experiment}'")
            print("Use --list to see available experiments")
            return
        run_experiment(args.experiment, PHASE4_EXPERIMENTS[args.experiment])

if __name__ == "__main__":
    main()

