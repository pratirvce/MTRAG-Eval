#!/usr/bin/env python3
"""
Wrapper script for Model Scaling Improvements
"""

import os
import sys
import argparse
import pathlib

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_name", type=str, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--output_dir", type=str, default="experiments/retrieval")
    args = parser.parse_args()
    
    # Set GPU before importing torch
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    
    # Import and run
    sys.path.insert(0, str(pathlib.Path(__file__).parent))
    from train_model_scaling_bge_v2 import main as run_main
    
    # Override sys.argv
    sys.argv = [
        "train_model_scaling_bge_v2.py",
        "--experiment_name", args.experiment_name,
        "--gpu", str(args.gpu),
        "--output_dir", args.output_dir,
        "--model_name", "BAAI/bge-large-en-v1.5",  # Can be upgraded to BGE-v2-large
        "--epochs", "5",
        "--batch_size", "8",  # Reduced for memory
        "--gradient_accumulation_steps", "4",
        "--learning_rate", "2e-5",
        "--domain_specific"  # Train separate models per domain
    ]
    
    run_main()

if __name__ == "__main__":
    main()

