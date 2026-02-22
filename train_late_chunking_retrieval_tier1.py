#!/usr/bin/env python3
"""
Tier 1: Late Chunking Retrieval
Contextual Chunk Embeddings Using Long-Context Embedding Models
Task A Compliant: ✅ Pure retrieval, no generation
Expected: +0.05-0.10 nDCG@10 improvement
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
    
    from train_late_chunking_retrieval import train_late_chunking_retrieval
    
    output_path = pathlib.Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    config = {
        'model_path': 'BAAI/bge-large-en-v1.5',  # Use large model for better context
        'output_dir': str(output_path),
        'domains': ['clapnq', 'fiqa', 'govt', 'cloud'],
        'use_data_splits': True,
        'chunk_size': 512,
        'chunk_overlap': 100,
        'max_doc_length': 2048,  # Long context for full document embedding
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
    logging.info("LATE CHUNKING RETRIEVAL")
    logging.info("="*80)
    logging.info("Novel Features:")
    logging.info("  1. Embed full document first (preserves global context)")
    logging.info("  2. Chunk after embedding (contextual chunk embeddings)")
    logging.info("  3. Recursive chunking (paragraph → sentence → word)")
    logging.info("  4. Asymmetric query encoding (with prompts)")
    logging.info("  5. BGE-large model (better long-context understanding)")
    logging.info("="*80)
    
    try:
        results_path = train_late_chunking_retrieval(config, gpu_id=args.gpu)
        if results_path:
            logging.info(f"✅ Late Chunking retrieval completed: {results_path}")
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

