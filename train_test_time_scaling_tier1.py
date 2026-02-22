#!/usr/bin/env python3
"""
Tier 1: Test-Time Scaling & Iterative Reranking
Iterative refinement: retrieve → rerank → refine → re-retrieve

Task A Compliant: ✅ LLM only scores, not generates text
Expected: 0.82-0.90 nDCG@10
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
    
    from train_test_time_scaling import train_test_time_scaling
    
    output_path = pathlib.Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    config = {
        'model_path': 'BAAI/bge-base-en-v1.5',
        'llm_model': 'microsoft/Phi-3-mini-4k-instruct',
        'use_local_llm': True,
        'num_iterations': 2,
        'output_dir': str(output_path),
        'domains': ['clapnq', 'fiqa', 'govt', 'cloud'],
        'use_data_splits': True,
        'resume': args.resume
    }
    
    config_file = output_path / "config.json"
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    
    logging.info("="*80)
    logging.info("TEST-TIME SCALING & ITERATIVE RERANKING")
    logging.info("="*80)
    logging.info("Novel Features:")
    logging.info("  1. Initial retrieval → Top candidates")
    logging.info("  2. LLM reranks top candidates (scores only)")
    logging.info("  3. Refine query based on reranked documents")
    logging.info("  4. Re-retrieve with refined query")
    logging.info("  5. Aggregate scores across iterations")
    logging.info("Task A Compliant: LLM only outputs scores (not text)")
    logging.info("="*80)
    
    try:
        results_path = train_test_time_scaling(config, gpu_id=args.gpu)
        if results_path:
            logging.info(f"✅ Test-time scaling completed: {results_path}")
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

