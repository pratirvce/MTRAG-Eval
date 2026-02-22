#!/usr/bin/env python3
"""
Tier 1: Pseudo-Relevance Feedback with LLM Expansion
Priority 5 - Modern LLM-based approach
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
    parser = argparse.ArgumentParser(description='Tier 1: Pseudo-Relevance Feedback (LLM)')
    parser.add_argument('--experiment_name', type=str, required=True)
    parser.add_argument('--gpu', type=int, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--resume', action='store_true', help='Resume from checkpoint')
    parser.add_argument('--no-resume', dest='resume', action='store_false', help='Do not resume')
    parser.set_defaults(resume=True)
    
    args = parser.parse_args()
    
    from train_pseudo_relevance_feedback import run_pseudo_relevance_feedback_evaluation
    
    output_path = pathlib.Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    config = {
        'output_dir': str(output_path),
        'domains': ['clapnq', 'fiqa', 'govt', 'cloud'],
        'use_data_splits': True,
        'model_path': 'BAAI/bge-base-en-v1.5',
        'llm_provider': 'openai',  # or 'claude', 'local'
        'top_k_feedback': 10,
        'resume': args.resume
    }
    
    config_file = output_path / "config.json"
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    
    try:
        results = run_pseudo_relevance_feedback_evaluation(config, gpu_id=args.gpu)
        logging.info("✅ Pseudo-Relevance Feedback completed successfully")
    except Exception as e:
        logging.error(f"Pseudo-Relevance Feedback failed: {e}", exc_info=True)
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

