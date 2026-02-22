#!/usr/bin/env python3
"""
Tier1 Wrapper for Hybrid Dense-Sparse with Learned Dynamic Fusion
Task A - Ultra High Performance Retrieval
Expected: 0.72-0.82 nDCG@10
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
    from train_hybrid_dynamic_fusion import train_hybrid_dynamic_fusion
    
    # Setup paths
    experiment_name = args.experiment_name
    output_dir = pathlib.Path(f"experiments/retrieval/{experiment_name}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create config
    config = {
        "output_dir": str(output_dir),
        "model_path": "BAAI/bge-base-en-v1.5",
        "epochs": 3,
        "batch_size": 4,
        "gradient_accumulation_steps": 2,
        "use_fp16": True,
        "use_data_splits": True,
        "domains": ["clapnq", "fiqa", "govt", "cloud"],
        "resume": True
    }
    
    # Save config
    config_file = output_dir / "config.json"
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    
    # Run training
    try:
        results_path = train_hybrid_dynamic_fusion(config, gpu_id=args.gpu)
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

