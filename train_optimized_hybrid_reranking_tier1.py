#!/usr/bin/env python3
"""
Tier 1: Optimized Hybrid Search with Reranking - All Improvements Combined
Expected: 0.60-0.75 nDCG@10
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
    
    from train_optimized_hybrid_reranking import train_optimized_hybrid_reranking
    
    output_path = pathlib.Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    config = {
        'model_path': 'BAAI/bge-large-en-v1.5',  # Large model for better performance
        'cross_encoder_model': 'cross-encoder/ms-marco-MiniLM-L-6-v2',
        'output_dir': str(output_path),
        'domains': ['clapnq', 'fiqa', 'govt', 'cloud'],
        'use_data_splits': True,
        'epochs': 3,
        'batch_size': 8,
        'stage1_top_k': 200,  # Dense retrieval
        'stage2_top_k': 100,  # Sparse retrieval
        'stage3_top_k': 50,   # After fusion
        'stage4_top_k': 20,   # After reranking
        'final_top_k': 10,    # Final output
        'resume': args.resume
    }
    
    config_file = output_path / "config.json"
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    
    logging.info("="*80)
    logging.info("OPTIMIZED HYBRID SEARCH WITH RERANKING")
    logging.info("="*80)
    logging.info("Improvements:")
    logging.info("  1. BGE-large for dense retrieval")
    logging.info("  2. Fine-tuned cross-encoder on MTRAG data")
    logging.info("  3. Learned hybrid fusion with domain/query adaptation")
    logging.info("  4. Query expansion (Task A compliant)")
    logging.info("  5. Optimized multi-stage pipeline")
    logging.info("Expected: 0.60-0.75 nDCG@10")
    logging.info("="*80)
    
    try:
        results_path = train_optimized_hybrid_reranking(config, gpu_id=args.gpu)
        if results_path:
            logging.info(f"✅ Optimized hybrid reranking completed: {results_path}")
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

