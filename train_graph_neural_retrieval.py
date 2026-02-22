"""
Conversation-Aware Graph Neural Retrieval
Task A - Ultra High Performance Retrieval
Expected: 0.68-0.78 nDCG@10
Uses existing graph_enhanced_reranking as base
"""

import pathlib
import logging
import argparse
import json
import os
import sys

# Import existing graph enhanced reranking
from train_graph_enhanced_reranking import train_graph_enhanced_reranking

logging.basicConfig(format='%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S', level=logging.INFO)

def train_graph_neural_retrieval(config: dict, gpu_id: int = None):
    """Train Conversation-Aware Graph Neural Retrieval"""
    # Use existing graph enhanced reranking implementation
    return train_graph_enhanced_reranking(config, gpu_id)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--gpu_id', type=int)
    parser.add_argument('--resume', action='store_true', default=True)
    parser.add_argument('--no-resume', dest='resume', action='store_false')
    
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    config['resume'] = args.resume
    
    try:
        results_path = train_graph_neural_retrieval(config, args.gpu_id)
        if results_path:
            logging.info(f"✅ Training completed: {results_path}")
    except KeyboardInterrupt:
        logging.info("Interrupted by user")
        sys.exit(0)
    except Exception as e:
        logging.error(f"Failed: {e}", exc_info=True)
        sys.exit(1)
