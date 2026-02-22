#!/usr/bin/env python3
"""
Tier 1: Cross-Encoder with Larger Model (L-24)
Uses larger cross-encoder model for better reranking
"""

import sys
import pathlib
import argparse
import json
import logging

sys.path.insert(0, str(pathlib.Path(__file__).parent))

logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO)

def main():
    parser = argparse.ArgumentParser(description='Tier 1: Cross-Encoder Large Model')
    parser.add_argument('--experiment_name', type=str, required=True)
    parser.add_argument('--gpu', type=int, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--resume', action='store_true', help='Resume from checkpoint')
    parser.add_argument('--no-resume', dest='resume', action='store_false', help='Do not resume')
    parser.set_defaults(resume=True)
    
    args = parser.parse_args()
    
    # Import evaluation script (similar to evaluate_cross_encoder_tier1.py)
    from evaluate_cross_encoder_tier1 import evaluate_cross_encoder_reranking
    
    output_path = pathlib.Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Use larger model
    model_path = "cross-encoder/ms-marco-MiniLM-L-24-v2"
    
    # If fine-tuned model exists, use it; otherwise use pre-trained
    fine_tuned_path = pathlib.Path("models/tier1_cross_encoder_finetuned")
    if fine_tuned_path.exists():
        model_path = str(fine_tuned_path)
        logging.info(f"Using fine-tuned model: {model_path}")
    else:
        logging.info(f"Using pre-trained large model: {model_path}")
    
    config = {
        'model_path': model_path,
        'output_dir': str(output_path),
        'domains': ['clapnq', 'fiqa', 'govt', 'cloud'],
        'use_data_splits': True,
        'top_k': 100,
        'rerank_top_k': 10,
        'resume': args.resume
    }
    
    config_file = output_path / "config.json"
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    
    try:
        # Set GPU before evaluation
        import os
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
        
        # Use device 0 since CUDA_VISIBLE_DEVICES is set
        results = evaluate_cross_encoder_reranking(model_path, 0, output_path)
        logging.info("✅ Cross-encoder large model evaluation completed successfully")
    except Exception as e:
        logging.error(f"Cross-encoder large model evaluation failed: {e}", exc_info=True)
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

