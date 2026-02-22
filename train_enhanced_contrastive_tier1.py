#!/usr/bin/env python3
"""
Tier 1: Enhanced Contrastive Learning with Hard Negative Mining
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
    # This MUST be done before importing train_enhanced_contrastive_hardnegatives
    # because that module imports torch at the top level
    # We override any existing value to ensure we use the correct GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    logging.info(f"Set CUDA_VISIBLE_DEVICES={args.gpu} (overriding any existing value)")
    
    # Now import after CUDA_VISIBLE_DEVICES is set
    from train_enhanced_contrastive_hardnegatives import train_enhanced_contrastive
    
    output_path = pathlib.Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    config = {
        'model_path': 'BAAI/bge-base-en-v1.5',
        'output_dir': str(output_path),
        'domains': ['clapnq', 'fiqa', 'govt', 'cloud'],
        'use_data_splits': True,
        'epochs': 3,
        'batch_size': 1,  # Reduced to 1 to avoid OOM errors
        'num_hard_negatives': 1,  # Reduced from 2 to 1 to reduce memory usage
        'gradient_accumulation_steps': 4,  # Increase accumulation to maintain effective batch size
        'use_fp16': True,  # Enable mixed precision training
        'resume': args.resume
    }
    
    config_file = output_path / "config.json"
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    
    try:
        results_path = train_enhanced_contrastive(config, gpu_id=args.gpu)
        if results_path:
            logging.info(f"✅ Enhanced contrastive learning completed: {results_path}")
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

