#!/usr/bin/env python3
"""
Tier 1: Phase 2 Cosine Loss Training
Cosine Similarity Loss instead of MultipleNegativesRankingLoss
"""

import sys
import os
import pathlib
import argparse
import json
import logging

sys.path.insert(0, str(pathlib.Path(__file__).parent))

logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_name', type=str, required=True)
    parser.add_argument('--gpu', type=int, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--resume', action='store_true', default=True)
    parser.add_argument('--no-resume', dest='resume', action='store_false')
    
    args = parser.parse_args()
    
    # Set CUDA_VISIBLE_DEVICES before importing torch
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    logging.info(f"Set CUDA_VISIBLE_DEVICES={args.gpu}")
    
    from train_improved_bge import run_training
    
    # Ensure output_dir includes experiment name
    output_path = pathlib.Path(args.output_dir) / args.experiment_name
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create config file for train_improved_bge
    config = {
        'experiment_name': args.experiment_name,
        'base_model': 'BAAI/bge-base-en-v1.5',  # Changed from model_path to base_model
        'output_path': str(output_path / "model"),
        'use_data_splits': True,
        'epochs': 3,
        'batch_size': 16,
        'learning_rate': 2e-5,
        'loss_function': 'CosineSimilarityLoss',
        'use_validation': True,
        'save_checkpoints': True,
        'checkpoint_steps': 1000,
        'warmup_steps': 100,
        'evaluation_steps': 500,
        'save_best_model': True
    }
    
    config_file = output_path / "config.json"
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    
    logging.info("="*80)
    logging.info("PHASE 2: COSINE SIMILARITY LOSS TRAINING")
    logging.info("="*80)
    logging.info(f"Config: {config_file}")
    logging.info(f"Output: {output_path}")
    logging.info("="*80)
    
    try:
        # run_training expects config dict
        run_training(config)
        logging.info(f"✅ Phase 2 cosine loss training completed")
    except Exception as e:
        logging.error(f"Failed: {e}", exc_info=True)
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

