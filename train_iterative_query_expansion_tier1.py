#!/usr/bin/env python3
"""
Tier 1: Iterative Retrieval with Query Expansion Feedback
Initial retrieval → LLM generates expansion terms → Re-retrieve

Task A Compliant: ✅ Feedback is query expansion terms only, not answers
Expected: 0.72-0.80 nDCG@10
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
    
    from train_iterative_query_expansion import train_iterative_query_expansion
    
    output_path = pathlib.Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    config = {
        'model_path': 'BAAI/bge-base-en-v1.5',
        'llm_model': 'microsoft/Phi-3-mini-4k-instruct',
        'use_local_llm': True,
        'output_dir': str(output_path),
        'domains': ['clapnq', 'fiqa', 'govt', 'cloud'],
        'use_data_splits': True,
        'resume': args.resume
    }
    
    config_file = output_path / "config.json"
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    
    logging.info("="*80)
    logging.info("ITERATIVE RETRIEVAL WITH QUERY EXPANSION")
    logging.info("="*80)
    logging.info("Pipeline:")
    logging.info("  1. Initial retrieval → Top 20")
    logging.info("  2. LLM generates expansion terms from top documents")
    logging.info("  3. Re-retrieve with expanded query → Final top 10")
    logging.info("Task A Compliant: Feedback is query expansion terms only (not answers)")
    logging.info("="*80)
    
    try:
        results_path = train_iterative_query_expansion(config, gpu_id=args.gpu)
        if results_path:
            logging.info(f"✅ Iterative query expansion completed: {results_path}")
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

