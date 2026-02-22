#!/usr/bin/env python3
"""
Tier 1: Improved Iterative Refinement Retrieval
Priority 4 - High potential if fixed
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
    parser = argparse.ArgumentParser(description='Tier 1: Improved Iterative Refinement')
    parser.add_argument('--experiment_name', type=str, required=True)
    parser.add_argument('--gpu', type=int, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--resume', action='store_true', help='Resume from checkpoint')
    parser.add_argument('--no-resume', dest='resume', action='store_false', help='Do not resume')
    parser.set_defaults(resume=True)
    
    args = parser.parse_args()
    
    from train_iterative_refinement_retrieval import run_iterative_refinement_evaluation
    
    output_path = pathlib.Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create config - improved version with better parameters
    config = {
        'experiment_name': args.experiment_name,
        'model_path': 'BAAI/bge-base-en-v1.5',
        'domains': ['clapnq', 'fiqa', 'govt', 'cloud'],
        'use_data_splits': True,
        'round1_top_k': 100,
        'round2_top_k': 100,
        'final_top_k': 100,
        'resume': args.resume
    }
    
    config_file = output_path / "config.json"
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    
    # Run experiment
    try:
        results_file = run_iterative_refinement_evaluation(config, gpu_id=args.gpu)
        
        # Results are already saved by run_iterative_refinement_evaluation
        if results_file:
            logging.info(f"✅ Iterative refinement experiment completed. Results: {results_file}")
    except Exception as e:
        logging.error(f"Iterative refinement experiment failed: {e}", exc_info=True)
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
