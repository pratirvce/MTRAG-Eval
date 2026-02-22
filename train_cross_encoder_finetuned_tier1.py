#!/usr/bin/env python3
"""
Tier 1: Fine-Tuned Cross-Encoder Reranking
Priority 1 - Highest impact experiment
"""

import sys
import pathlib
import argparse
import json

# Add parent directory to path to import existing scripts
sys.path.insert(0, str(pathlib.Path(__file__).parent))

# Import the existing cross-encoder training function
from train_cross_encoder_finetuned import train_cross_encoder

def main():
    parser = argparse.ArgumentParser(description='Tier 1: Fine-Tuned Cross-Encoder')
    parser.add_argument('--experiment_name', type=str, required=True)
    parser.add_argument('--gpu', type=int, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--resume', action='store_true', help='Resume from checkpoint')
    parser.add_argument('--no-resume', dest='resume', action='store_false', help='Do not resume')
    parser.set_defaults(resume=True)
    
    args = parser.parse_args()
    
    # Create config for cross-encoder training
    config = {
        'experiment_name': args.experiment_name,
        'base_model': 'cross-encoder/ms-marco-MiniLM-L-12-v2',
        'domains': ['clapnq', 'fiqa', 'govt', 'cloud'],
        'use_data_splits': True,
        'epochs': 3,
        'batch_size': 16,
        'learning_rate': 2e-5,
        'warmup_steps': 100,
        'checkpoint_save_steps': 500,
        'resume': args.resume,
        'seed': 42
    }
    
    # Create output directory structure
    output_path = pathlib.Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save config
    config_file = output_path / "config.json"
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    
    # Run training
    try:
        model_path = train_cross_encoder(config, gpu_id=args.gpu)
        
        # Save results placeholder (actual evaluation will be done separately)
        results = {
            'experiment': args.experiment_name,
            'status': 'training_completed',
            'model_path': str(model_path) if model_path else None
        }
        
        results_file = output_path / "results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
            
    except Exception as e:
        # Save error info
        error_info = {
            'experiment': args.experiment_name,
            'status': 'failed',
            'error': str(e)
        }
        results_file = output_path / "results.json"
        with open(results_file, 'w') as f:
            json.dump(error_info, f, indent=2)
        raise

if __name__ == "__main__":
    main()

