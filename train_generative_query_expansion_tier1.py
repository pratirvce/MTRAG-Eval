#!/usr/bin/env python3
"""
Tier 1: Generative Query Expansion with LLM
Uses LLM to generate query variants for better retrieval

Task A Compliant: ✅ Query preprocessing only, no answer generation
Expected: 0.75-0.82 nDCG@10
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
    
    from train_generative_query_expansion import train_generative_query_expansion
    
    output_path = pathlib.Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    config = {
        'model_path': 'BAAI/bge-base-en-v1.5',
        'llm_model': 'microsoft/Phi-3-mini-4k-instruct',  # Small, efficient LLM
        'use_local_llm': True,  # Use local LLM for query expansion
        'output_dir': str(output_path),
        'domains': ['clapnq', 'fiqa', 'govt', 'cloud'],
        'use_data_splits': True,
        'resume': args.resume
    }
    
    config_file = output_path / "config.json"
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    
    logging.info("="*80)
    logging.info("GENERATIVE QUERY EXPANSION WITH LLM")
    logging.info("="*80)
    logging.info("Novel Features:")
    logging.info("  1. LLM generates 3-5 query variants per query")
    logging.info("  2. Uses conversation history for context-aware expansion")
    logging.info("  3. Merges results using Reciprocal Rank Fusion")
    logging.info("  4. Task A Compliant: Query preprocessing only (no answers)")
    logging.info("="*80)
    
    try:
        results_path = train_generative_query_expansion(config, gpu_id=args.gpu)
        if results_path:
            logging.info(f"✅ Generative query expansion completed: {results_path}")
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

