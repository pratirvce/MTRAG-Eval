#!/usr/bin/env python3
"""
Tier 1: Cross-Attention Query-Document Interaction
Priority 2 - High novelty experiment
"""

import sys
import pathlib
import argparse
import json
import logging

# Add parent directory to path
sys.path.insert(0, str(pathlib.Path(__file__).parent))

logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO)

def main():
    parser = argparse.ArgumentParser(description='Tier 1: Cross-Attention Query-Document')
    parser.add_argument('--experiment_name', type=str, required=True)
    parser.add_argument('--gpu', type=int, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--resume', action='store_true', help='Resume from checkpoint')
    parser.add_argument('--no-resume', dest='resume', action='store_false', help='Do not resume')
    parser.set_defaults(resume=True)
    
    args = parser.parse_args()
    
    # Import and use existing cross-attention script
    from train_cross_attention_retrieval import run_cross_attention_retrieval
    
    output_path = pathlib.Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create config matching the expected format
    config = {
        'model_path': 'BAAI/bge-base-en-v1.5',
        'output_dir': str(output_path),
        'domains': ['clapnq', 'fiqa', 'govt', 'cloud'],
        'use_data_splits': True,
        'top_k': 100,
        'batch_size': 32,
        'doc_batch_size': 1000,
        'resume': args.resume,
        'use_attention': True
    }
    
    # Save config
    config_file = output_path / "config.json"
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    
    # Run experiment
    try:
        results_path = run_cross_attention_retrieval(config, gpu_id=args.gpu)
        
        # Results are already saved by run_cross_attention_retrieval
        if results_path:
            logging.info(f"✅ Cross-attention experiment completed. Results: {results_path}")
    except Exception as e:
        logging.error(f"Cross-attention experiment failed: {e}", exc_info=True)
        # Save error info
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
