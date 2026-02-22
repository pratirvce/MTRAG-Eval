#!/usr/bin/env python3
"""
Tier 1: Curriculum Contrastive Learning with Hard Negatives and Query Rewriting
Novel approach combining curriculum learning, query rewriting, and multi-granularity negatives
Expected: 0.52-0.56 nDCG@10
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
    
    # ALWAYS set CUDA_VISIBLE_DEVICES to the specified GPU BEFORE importing anything that uses torch
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    logging.info(f"Set CUDA_VISIBLE_DEVICES={args.gpu}")
    
    # Now import after CUDA_VISIBLE_DEVICES is set
    from train_curriculum_contrastive import train_curriculum_contrastive
    
    output_path = pathlib.Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    config = {
        'model_path': 'BAAI/bge-base-en-v1.5',
        'output_dir': str(output_path),
        'domains': ['clapnq', 'fiqa', 'govt', 'cloud'],
        'use_data_splits': True,
        'epochs': 3,
        'batch_size': 2,  # Memory optimized (reduced from 4 to fix OOM)
        'num_hard_negatives': 2,  # Reduced for memory
        'gradient_accumulation_steps': 2,  # Effective batch size = 8
        'use_fp16': True,  # Mixed precision
        'resume': args.resume
    }
    
    config_file = output_path / "config.json"
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    
    logging.info("="*80)
    logging.info("CURRICULUM CONTRASTIVE LEARNING WITH QUERY REWRITING")
    logging.info("="*80)
    logging.info("Novel Features:")
    logging.info("  1. Curriculum Learning: Progressive negative difficulty (random → BM25 → adversarial)")
    logging.info("  2. Query Rewriting: Context-aware query expansion from conversation history")
    logging.info("  3. Multi-Granularity Negatives: Document/sentence/phrase level mining")
    logging.info("  4. Hierarchical Contrastive Loss: Learning at multiple granularities")
    logging.info("="*80)
    
    try:
        results_path = train_curriculum_contrastive(config, gpu_id=args.gpu)
        if results_path:
            logging.info(f"✅ Curriculum contrastive learning completed: {results_path}")
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

