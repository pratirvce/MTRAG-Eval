#!/usr/bin/env python3
"""
Tier1 Wrapper for Transformer-Based Cross-Encoder with Full Conversation Context
Task A - Ultra High Performance Retrieval
Expected: 0.68-0.78 nDCG@10
"""

import os
import sys
import json
import pathlib
import argparse

# Set CUDA_VISIBLE_DEVICES before importing torch
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_name', type=str, required=True)
    parser.add_argument('--gpu', type=int, required=True)
    args, unknown = parser.parse_known_args()
    
    # Set GPU before any torch imports
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    
    # Now import the training script
    from train_cross_encoder_conversation_context import train_cross_encoder_conversation_context
    
    # Setup paths
    experiment_name = args.experiment_name
    output_dir = pathlib.Path(f"experiments/retrieval/{experiment_name}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create config
    config = {
        "output_dir": str(output_dir),
        "model_path": "BAAI/bge-base-en-v1.5",
        "cross_encoder_model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
        "use_data_splits": True,
        "domains": ["clapnq", "fiqa", "govt", "cloud"],
        "resume": True,
        "first_stage_top_k": 100,
        "rerank_top_k": 10
    }
    
    # Save config
    config_file = output_dir / "config.json"
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    
    # Run training
    try:
        results_path = train_cross_encoder_conversation_context(config, gpu_id=args.gpu)
        if results_path:
            print(f"✅ Experiment completed: {results_path}")
            sys.exit(0)
        else:
            print("❌ Experiment failed")
            sys.exit(1)
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

