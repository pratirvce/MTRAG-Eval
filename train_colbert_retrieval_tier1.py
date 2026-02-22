#!/usr/bin/env python3
"""
Tier 1: ColBERT-Style Multi-Vector Retrieval
Token-level embeddings with MaxSim scoring
Task A Compliant: ✅ Pure retrieval, no generation
Expected: +0.04-0.08 nDCG@10 improvement
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
    
    from train_colbert_retrieval import train_colbert_retrieval
    
    output_path = pathlib.Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    config = {
        'model_path': 'BAAI/bge-base-en-v1.5',
        'output_dir': str(output_path),
        'domains': ['clapnq', 'fiqa', 'govt', 'cloud'],
        'use_data_splits': True,
        'max_query_length': 32,
        'max_doc_length': 180,
        'fine_tune': False,  # Can set to True for fine-tuning
        'epochs': 3,
        'batch_size': 4,
        'gradient_accumulation_steps': 4,
        'use_fp16': True,
        'resume': args.resume
    }
    
    config_file = output_path / "config.json"
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    
    logging.info("="*80)
    logging.info("COLBERT-STYLE MULTI-VECTOR RETRIEVAL")
    logging.info("="*80)
    logging.info("Novel Features:")
    logging.info("  1. Token-level embeddings (not single-vector)")
    logging.info("  2. MaxSim scoring (max similarity per query token)")
    logging.info("  3. Fine-grained matching (token-to-token)")
    logging.info("  4. Better for partial relevance")
    logging.info("="*80)
    
    try:
        results_path = train_colbert_retrieval(config, gpu_id=args.gpu)
        if results_path:
            logging.info(f"✅ ColBERT retrieval completed: {results_path}")
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

