#!/usr/bin/env python3
"""
Tier 1: Multi-Stage Retrieval Pipeline
Wrapper for multi-stage retrieval experiment (2-stage version)
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
    parser = argparse.ArgumentParser(description='Tier 1: Multi-Stage Retrieval')
    parser.add_argument('--experiment_name', type=str, required=True)
    parser.add_argument('--gpu', type=int, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--resume', action='store_true', help='Resume from checkpoint')
    parser.add_argument('--no-resume', dest='resume', action='store_false', help='Do not resume')
    parser.set_defaults(resume=True)
    
    args = parser.parse_args()
    
    from train_multistage_retrieval import run_multistage_evaluation
    
    output_path = pathlib.Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    config = {
        'output_dir': str(output_path),
        'domains': ['clapnq', 'fiqa', 'govt', 'cloud'],
        'use_data_splits': True,
        'stage1_model': 'BAAI/bge-base-en-v1.5',
        'stage2_model': 'cross-encoder/ms-marco-MiniLM-L-12-v2',
        'stage3_model': None,  # 2-stage only
        'stage1_top_k': 100,
        'stage2_top_k': 10,
        'stage3_top_k': None,
        'resume': args.resume
    }
    
    config_file = output_path / "config.json"
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    
    try:
        results = run_multistage_evaluation(config, gpu_id=args.gpu)
        logging.info("✅ Multi-stage retrieval completed successfully")
    except Exception as e:
        logging.error(f"Multi-stage retrieval failed: {e}", exc_info=True)
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

