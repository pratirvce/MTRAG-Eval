#!/usr/bin/env python3
"""
Tier 1: CG-RAG (Contextualized Graph RAG) Retrieval
Task A - Retrieval Only
Expected: +0.03-0.06 nDCG@10 improvement
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
    
    # Set CUDA_VISIBLE_DEVICES before importing torch
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    logging.info(f"Set CUDA_VISIBLE_DEVICES={args.gpu}")
    
    from train_cg_rag_retrieval import train_cg_rag_retrieval
    
    output_path = pathlib.Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    config = {
        'model_path': 'BAAI/bge-base-en-v1.5',
        'output_dir': str(output_path),
        'domains': ['clapnq', 'fiqa', 'govt', 'cloud'],
        'use_data_splits': True,
        'epochs': 3,
        'batch_size': 8,
        'gradient_accumulation_steps': 2,
        'use_fp16': True,
        'fine_tune': True,
        'resume': args.resume
    }
    
    config_file = output_path / "config.json"
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    
    logging.info("="*80)
    logging.info("CG-RAG (CONTEXTUALIZED GRAPH RAG) RETRIEVAL")
    logging.info("="*80)
    logging.info("Features:")
    logging.info("  1. Builds knowledge graph from corpus documents")
    logging.info("  2. Extracts entities and relationships")
    logging.info("  3. Contextualizes graph based on query/conversation")
    logging.info("  4. Combines dense retrieval with graph-based scores")
    logging.info("  5. Uses contextualized graph for retrieval")
    logging.info("⚠️  TASK A COMPLIANCE:")
    logging.info("  • Builds graph from documents (corpus processing - allowed)")
    logging.info("  • Contextualizes graph based on query (query processing - allowed)")
    logging.info("  • No text generation involved")
    logging.info("  • Pure retrieval method")
    logging.info("  Reference: https://ibm.github.io/mt-rag-benchmark/MTRAGEval/")
    logging.info("Expected: +0.03-0.06 nDCG@10 improvement")
    logging.info("="*80)
    
    try:
        results_path = train_cg_rag_retrieval(config, gpu_id=args.gpu)
        if results_path:
            logging.info(f"✅ CG-RAG retrieval completed: {results_path}")
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

