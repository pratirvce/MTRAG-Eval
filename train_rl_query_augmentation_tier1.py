#!/usr/bin/env python3
"""
Tier 1: RL-Optimized Query Augmentation
Policy optimized by retrieval rewards (nDCG)

Task A Compliant: ✅ Query augmentation is preprocessing only
Expected: 0.80-0.88 nDCG@10
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
    
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    logging.info(f"Set CUDA_VISIBLE_DEVICES={args.gpu}")
    
    from train_rl_query_augmentation import train_rl_query_augmentation
    
    output_path = pathlib.Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    config = {
        'model_path': 'BAAI/bge-base-en-v1.5',
        'rl_epochs': 3,
        'output_dir': str(output_path),
        'domains': ['clapnq', 'fiqa', 'govt', 'cloud'],
        'use_data_splits': True,
        'resume': args.resume
    }
    
    config_file = output_path / "config.json"
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    
    logging.info("="*80)
    logging.info("RL-OPTIMIZED QUERY AUGMENTATION")
    logging.info("="*80)
    logging.info("Novel Features:")
    logging.info("  1. Treat query rewriter as RL agent")
    logging.info("  2. Reward = retrieval nDCG@K")
    logging.info("  3. Train policy with PPO + KL regularization")
    logging.info("  4. Directly optimizes retrieval performance")
    logging.info("Task A Compliant: Query augmentation only (no answer generation)")
    logging.info("="*80)
    
    try:
        results_path = train_rl_query_augmentation(config, gpu_id=args.gpu)
        if results_path:
            logging.info(f"✅ RL query augmentation completed: {results_path}")
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
