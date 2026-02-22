#!/usr/bin/env python3
"""
Wrapper script for Advanced Contrastive Learning with Hierarchical Loss
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
    from train_advanced_contrastive_hierarchical import main as run_main
    
    # Override sys.argv
    sys.argv = [
        "train_advanced_contrastive_hierarchical.py",
        "--experiment_name", args.experiment_name,
        "--gpu", str(args.gpu),
        "--output_dir", args.output_dir,
        "--use_data_splits",
        "--epochs", "3",
        "--batch_size", "8",  # Reduced for memory
        "--learning_rate", "2e-5"
    ]
    
    run_main()

if __name__ == "__main__":
    main()

