#!/usr/bin/env python3
"""
Experiment Runner for Retrieval Improvements
Runs all phases systematically and records results.
"""
import json
import subprocess
import time
import pathlib
from datetime import datetime
import logging

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

RESULTS_DIR = pathlib.Path("experiments/retrieval")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

def save_experiment_config(phase, experiment_name, config, results_file):
    """Save experiment configuration."""
    exp_config = {
        'phase': phase,
        'experiment_name': experiment_name,
        'config': config,
        'timestamp': datetime.now().isoformat(),
        'results_file': str(results_file)
    }
    
    config_file = RESULTS_DIR / f"{experiment_name}_config.json"
    with open(config_file, 'w') as f:
        json.dump(exp_config, f, indent=2)
    
    return config_file

def run_training(config_file, log_file):
    """Run training script."""
    cmd = [
        "python", "train_improved_bge.py",
        "--config", str(config_file)
    ]
    
    logging.info(f"Running training: {' '.join(cmd)}")
    with open(log_file, 'w') as log:
        result = subprocess.run(
            cmd,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True
        )
    
    return result.returncode == 0

def run_evaluation(model_path, config_file, results_file, log_file):
    """Run evaluation script."""
    cmd = [
        "python", "evaluate_improved_bge.py",
        "--model_path", model_path,
        "--config", str(config_file),
        "--output", str(results_file)
    ]
    
    logging.info(f"Running evaluation: {' '.join(cmd)}")
    with open(log_file, 'w') as log:
        result = subprocess.run(
            cmd,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True
        )
    
    return result.returncode == 0

def run_phase(phase_name, experiments):
    """Run all experiments in a phase."""
    logging.info(f"\n{'='*60}")
    logging.info(f"Starting Phase: {phase_name}")
    logging.info(f"{'='*60}\n")
    
    phase_results = []
    
    for exp_name, config in experiments.items():
        logging.info(f"\n--- Experiment: {exp_name} ---")
        
        # Create experiment directory
        exp_dir = RESULTS_DIR / exp_name
        exp_dir.mkdir(parents=True, exist_ok=True)
        
        # Save config
        config_file = exp_dir / "config.json"
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        
        # Training (skip if using existing model)
        train_log = exp_dir / "training.log"
        
        # Check if we should use an existing model
        use_existing_model = config.get('use_existing_model', False)
        existing_model_path = config.get('existing_model_path', None)
        
        if use_existing_model and existing_model_path:
            model_path = existing_model_path
            logging.info(f"Using existing model: {model_path}")
            if not pathlib.Path(model_path).exists():
                logging.error(f"Existing model not found: {model_path}")
                continue
        else:
            model_path = f"./models/{exp_name}"
            logging.info(f"Training model: {exp_name}")
            train_success = run_training(config_file, train_log)
            
            if not train_success:
                logging.error(f"Training failed for {exp_name}. Check {train_log}")
                continue
            
            # Wait a bit for model to be fully saved
            time.sleep(5)
        
        # Evaluation
        eval_log = exp_dir / "evaluation.log"
        results_file = exp_dir / "results.json"
        
        logging.info(f"Evaluating model: {exp_name}")
        eval_success = run_evaluation(model_path, config_file, results_file, eval_log)
        
        if eval_success and results_file.exists():
            with open(results_file, 'r') as f:
                results = json.load(f)
            phase_results.append({
                'experiment': exp_name,
                'results': results
            })
            logging.info(f"✅ {exp_name} completed. Recall@10: {results['averages']['Recall@10']:.4f}")
        else:
            logging.error(f"Evaluation failed for {exp_name}. Check {eval_log}")
    
    return phase_results

def main():
    """Run all experiment phases."""
    import os
    
    # Save PID
    pid_file = RESULTS_DIR / "logs" / "current_experiment.pid"
    pid_file.parent.mkdir(parents=True, exist_ok=True)
    with open(pid_file, 'w') as f:
        f.write(str(os.getpid()))
    logging.info(f"PID saved to: {pid_file}")
    
    # Phase 1: Quick Wins
    phase1_experiments = {
        "phase1_baseline": {
            "experiment_name": "phase1_baseline",
            "base_model": "BAAI/bge-base-en-v1.5",
            "epochs": 1,
            "batch_size": 16,
            "learning_rate": 2e-5,
            "loss_function": "MultipleNegativesRankingLoss",
            "use_augmentation": False,
            "use_validation": False,
            "use_data_splits": False,
            "warmup_steps": 100,
            "output_path": "./models/phase1_baseline"
        },
        "phase1_epochs3": {
            "experiment_name": "phase1_epochs3",
            "base_model": "BAAI/bge-base-en-v1.5",
            "epochs": 3,
            "batch_size": 32,
            "learning_rate": 2e-5,
            "loss_function": "MultipleNegativesRankingLoss",
            "use_augmentation": False,
            "use_validation": True,
            "use_data_splits": True,
            "warmup_steps": 100,
            "evaluation_steps": 500,
            "save_best_model": True,
            "output_path": "./models/phase1_epochs3"
        },
        "phase1_epochs5": {
            "experiment_name": "phase1_epochs5",
            "base_model": "BAAI/bge-base-en-v1.5",
            "epochs": 5,
            "batch_size": 32,
            "learning_rate": 2e-5,
            "loss_function": "MultipleNegativesRankingLoss",
            "use_augmentation": False,
            "use_validation": True,
            "use_data_splits": True,
            "warmup_steps": 100,
            "evaluation_steps": 500,
            "save_best_model": True,
            "output_path": "./models/phase1_epochs5"
        }
    }
    
    # Phase 2: Training Improvements
    phase2_experiments = {
        "phase2_lr1e5": {
            "experiment_name": "phase2_lr1e5",
            "base_model": "BAAI/bge-base-en-v1.5",
            "epochs": 3,
            "batch_size": 32,
            "learning_rate": 1e-5,
            "loss_function": "MultipleNegativesRankingLoss",
            "use_augmentation": False,
            "use_validation": True,
            "use_data_splits": True,
            "warmup_steps": 100,
            "evaluation_steps": 500,
            "save_best_model": True,
            "output_path": "./models/phase2_lr1e5"
        },
        "phase2_lr5e5": {
            "experiment_name": "phase2_lr5e5",
            "base_model": "BAAI/bge-base-en-v1.5",
            "epochs": 3,
            "batch_size": 32,
            "learning_rate": 5e-5,
            "loss_function": "MultipleNegativesRankingLoss",
            "use_augmentation": False,
            "use_validation": True,
            "use_data_splits": True,
            "warmup_steps": 100,
            "evaluation_steps": 500,
            "save_best_model": True,
            "output_path": "./models/phase2_lr5e5"
        },
        "phase2_cosine_loss": {
            "experiment_name": "phase2_cosine_loss",
            "base_model": "BAAI/bge-base-en-v1.5",
            "epochs": 3,
            "batch_size": 32,
            "learning_rate": 2e-5,
            "loss_function": "CosineSimilarityLoss",
            "use_augmentation": False,
            "use_validation": True,
            "use_data_splits": True,
            "warmup_steps": 100,
            "evaluation_steps": 500,
            "save_best_model": True,
            "output_path": "./models/phase2_cosine_loss"
        },
        "phase2_augmentation": {
            "experiment_name": "phase2_augmentation",
            "base_model": "BAAI/bge-base-en-v1.5",
            "epochs": 3,
            "batch_size": 32,
            "learning_rate": 2e-5,
            "loss_function": "MultipleNegativesRankingLoss",
            "use_augmentation": True,
            "use_validation": True,
            "use_data_splits": True,
            "warmup_steps": 100,
            "evaluation_steps": 500,
            "save_best_model": True,
            "output_path": "./models/phase2_augmentation"
        }
    }
    
    # Phase 3: Retrieval Strategy
    # Skip hybrid experiments (require Elasticsearch, marginal gains)
    # Use best model (phase2_augmentation) for reranking
    phase3_experiments = {
        "phase3_reranking_best_model": {
            "experiment_name": "phase3_reranking_best_model",
            "use_existing_model": True,
            "existing_model_path": "./models/phase2_augmentation",
            "retriever_type": "dense",
            "use_reranking": True,
            "reranker_model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
            "top_k_rerank": 100,
            "use_data_splits": True
        }
    }
    
    all_results = {}
    
    # Run Phase 1
    phase1_results = run_phase("Phase 1: Quick Wins", phase1_experiments)
    all_results['phase1'] = phase1_results
    
    # Run Phase 2 (use best model from Phase 1)
    phase2_results = run_phase("Phase 2: Training Improvements", phase2_experiments)
    all_results['phase2'] = phase2_results
    
    # Run Phase 3 (use best model from Phase 2)
    phase3_results = run_phase("Phase 3: Retrieval Strategy", phase3_experiments)
    all_results['phase3'] = phase3_results
    
    # Save summary
    summary_file = RESULTS_DIR / "experiment_summary.json"
    with open(summary_file, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'results': all_results
        }, f, indent=2)
    
    logging.info(f"\n{'='*60}")
    logging.info("All experiments completed!")
    logging.info(f"Results saved to: {summary_file}")
    logging.info(f"{'='*60}\n")
    
    # Print summary
    print("\n" + "="*60)
    print("EXPERIMENT SUMMARY")
    print("="*60)
    
    for phase, results in all_results.items():
        print(f"\n{phase.upper()}:")
        for exp in results:
            if 'results' in exp:
                avg = exp['results'].get('averages', {})
                print(f"  {exp['experiment']}: R@10={avg.get('Recall@10', 0):.4f}, nDCG@10={avg.get('nDCG@10', 0):.4f}")

if __name__ == "__main__":
    main()

