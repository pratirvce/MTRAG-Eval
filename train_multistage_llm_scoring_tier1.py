#!/usr/bin/env python3
"""
Tier 1: Multi-Stage Retrieval with LLM Relevance Scoring
Multi-stage pipeline with LLM-based scoring

Task A Compliant: ✅ LLM only outputs scores (0-10), not text
Expected: 0.85-0.92 nDCG@10
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
    
    from train_multistage_llm_scoring import train_multistage_llm_scoring
    
    output_path = pathlib.Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    config = {
        'model_path': 'BAAI/bge-base-en-v1.5',
        'llm_model': 'microsoft/Phi-3-mini-4k-instruct',
        'use_local_llm': True,
        'use_cross_encoder': True,
        'output_dir': str(output_path),
        'domains': ['clapnq', 'fiqa', 'govt', 'cloud'],
        'use_data_splits': True,
        'resume': args.resume
    }
    
    config_file = output_path / "config.json"
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    
    logging.info("="*80)
    logging.info("MULTI-STAGE RETRIEVAL WITH LLM SCORING")
    logging.info("="*80)
    logging.info("Pipeline Stages:")
    logging.info("  1. Dense retrieval (BGE-base) → Top 100")
    logging.info("  2. Sparse retrieval (BM25) → Merge with dense")
    logging.info("  3. Cross-encoder reranking → Top 50")
    logging.info("  4. LLM relevance scoring → Top 20 → Final top 10")
    logging.info("Task A Compliant: LLM only outputs scores (0-10), not text")
    logging.info("="*80)
    
    try:
        results_path = train_multistage_llm_scoring(config, gpu_id=args.gpu)
        if results_path:
            logging.info(f"✅ Multi-stage LLM scoring completed: {results_path}")
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

