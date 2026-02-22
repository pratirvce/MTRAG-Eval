#!/usr/bin/env python3
"""
Template Tier 1 Wrapper for Novel Experiments
Task A Compliant: Retrieval only

Copy this template and update:
1. Import statement to your experiment function
2. Experiment name in config
3. Output directory name
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_name', type=str, required=True)
    parser.add_argument('--gpu', type=int, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--resume', action='store_true', default=True)
    parser.add_argument('--no-resume', dest='resume', action='store_false')
    
    args = parser.parse_args()
    
    # UPDATE: Import your experiment function
    from train_novel_experiment_template import run_novel_experiment
    
    output_path = pathlib.Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # UPDATE: Configure your experiment parameters
    config = {
        'model_path': 'BAAI/bge-large-en-v1.5',  # Or your model
        'output_dir': str(output_path),
        'domains': ['clapnq', 'fiqa', 'govt', 'cloud'],
        'use_data_splits': True,
        # Add your experiment-specific parameters here
        'resume': args.resume
    }
    
    config_file = output_path / "config.json"
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    
    try:
        # UPDATE: Call your experiment function
        results_path = run_novel_experiment(config, gpu_id=args.gpu)
        if results_path:
            logging.info(f"✅ Novel Experiment completed: {results_path}")
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

