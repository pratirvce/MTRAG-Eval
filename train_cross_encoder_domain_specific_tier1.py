#!/usr/bin/env python3
"""
Tier 1: Domain-Specific Cross-Encoder
Trains separate cross-encoder for each domain
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
    parser = argparse.ArgumentParser(description='Tier 1: Domain-Specific Cross-Encoder')
    parser.add_argument('--experiment_name', type=str, required=True)
    parser.add_argument('--gpu', type=int, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--resume', action='store_true', help='Resume from checkpoint')
    parser.add_argument('--no-resume', dest='resume', action='store_false', help='Do not resume')
    parser.set_defaults(resume=True)
    
    args = parser.parse_args()
    
    from train_cross_encoder_finetuned import train_cross_encoder
    
    output_path = pathlib.Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Train one model per domain (can be parallelized)
    domains = ['clapnq', 'fiqa', 'govt', 'cloud']
    all_results = {}
    
    for domain in domains:
        logging.info(f"\n{'='*60}")
        logging.info(f"Training cross-encoder for domain: {domain}")
        logging.info(f"{'='*60}")
        
        domain_output = output_path / domain
        domain_output.mkdir(parents=True, exist_ok=True)
        
        config = {
            'experiment_name': f"{args.experiment_name}_{domain}",
            'base_model': 'cross-encoder/ms-marco-MiniLM-L-12-v2',
            'domains': [domain],  # Single domain
            'use_data_splits': True,
            'epochs': 3,
            'batch_size': 16,
            'learning_rate': 2e-5,
            'warmup_steps': 100,
            'checkpoint_save_steps': 500,
            'resume': args.resume,
            'seed': 42
        }
        
        try:
            # Set GPU before training
            import os
            os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
            
            model_path = train_cross_encoder(config, gpu_id=0)  # Use 0 since CUDA_VISIBLE_DEVICES is set
            
            # Evaluate the domain-specific model
            from evaluate_cross_encoder_tier1 import evaluate_cross_encoder_reranking
            domain_results = evaluate_cross_encoder_reranking(str(model_path), 0, domain_output)  # Use 0 since CUDA_VISIBLE_DEVICES is set
            
            if domain_results and domain in domain_results:
                all_results[domain] = domain_results[domain]
                logging.info(f"✅ Domain {domain} completed: nDCG@10 = {domain_results[domain].get('nDCG@10', 0):.4f}")
        except Exception as e:
            logging.error(f"Error training/evaluating for domain {domain}: {e}", exc_info=True)
            continue
    
    # Calculate averages
    if all_results:
        import numpy as np
        avg_results = {
            "Recall@1": np.mean([r["Recall@1"] for r in all_results.values()]),
            "Recall@3": np.mean([r["Recall@3"] for r in all_results.values()]),
            "Recall@5": np.mean([r["Recall@5"] for r in all_results.values()]),
            "Recall@10": np.mean([r["Recall@10"] for r in all_results.values()]),
            "nDCG@1": np.mean([r["nDCG@1"] for r in all_results.values()]),
            "nDCG@3": np.mean([r["nDCG@3"] for r in all_results.values()]),
            "nDCG@5": np.mean([r["nDCG@5"] for r in all_results.values()]),
            "nDCG@10": np.mean([r["nDCG@10"] for r in all_results.values()])
        }
        all_results["average"] = avg_results
        
        logging.info("\n" + "="*60)
        logging.info("DOMAIN-SPECIFIC CROSS-ENCODER RESULTS:")
        logging.info("="*60)
        logging.info(f"Average Recall@10: {avg_results['Recall@10']:.4f}")
        logging.info(f"Average nDCG@10: {avg_results['nDCG@10']:.4f}")
        
        # Save results
        results_file = output_path / "results.json"
        with open(results_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        logging.info(f"Results saved to: {results_file}")

if __name__ == "__main__":
    main()

