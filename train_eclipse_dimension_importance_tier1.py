#!/usr/bin/env python3
"""
Tier 1: ECLIPSE - Contrastive Dimension Importance Estimation
Suppresses noisy dimensions via pseudo-irrelevant feedback

Task A Compliant: ✅ Pure retrieval enhancement
Expected: 0.75-0.83 nDCG@10
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
    
    from train_eclipse_dimension_importance import train_eclipse
    
    output_path = pathlib.Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    config = {
        'model_path': 'BAAI/bge-base-en-v1.5',
        'eclipse_epochs': 5,
        'output_dir': str(output_path),
        'domains': ['clapnq', 'fiqa', 'govt', 'cloud'],
        'use_data_splits': True,
        'resume': args.resume
    }
    
    config_file = output_path / "config.json"
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    
    logging.info("="*80)
    logging.info("ECLIPSE: CONTRASTIVE DIMENSION IMPORTANCE ESTIMATION")
    logging.info("="*80)
    logging.info("Novel Features:")
    logging.info("  1. Estimates importance of each embedding dimension")
    logging.info("  2. Uses pseudo-irrelevant feedback to identify noisy dimensions")
    logging.info("  3. Reweights embeddings to suppress noise")
    logging.info("  4. Improves dense retrieval by emphasizing informative dimensions")
    logging.info("Task A Compliant: Pure retrieval enhancement (no generation)")
    logging.info("="*80)
    
    try:
        results_path = train_eclipse(config, gpu_id=args.gpu)
        if results_path:
            logging.info(f"✅ ECLIPSE completed: {results_path}")
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

