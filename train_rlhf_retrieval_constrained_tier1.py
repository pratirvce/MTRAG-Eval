#!/usr/bin/env python3
"""
Wrapper script for RLHF for Retrieval (Task A Constrained)
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
    from train_rlhf_retrieval_constrained import main as run_main
    
    # Override sys.argv
    sys.argv = [
        "train_rlhf_retrieval_constrained.py",
        "--experiment_name", args.experiment_name,
        "--gpu", str(args.gpu),
        "--output_dir", args.output_dir,
        "--use_data_splits",
        "--num_epochs", "3"  # Reduced for efficiency
    ]
    
    run_main()

if __name__ == "__main__":
    main()

