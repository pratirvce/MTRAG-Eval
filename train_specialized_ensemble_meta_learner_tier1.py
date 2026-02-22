#!/usr/bin/env python3
"""
Wrapper script for Specialized Ensemble with Meta-Learner
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
    from train_specialized_ensemble_meta_learner import main as run_main
    
    # Override sys.argv
    sys.argv = [
        "train_specialized_ensemble_meta_learner.py",
        "--experiment_name", args.experiment_name,
        "--gpu", str(args.gpu),
        "--output_dir", args.output_dir,
        "--epochs", "3",
        "--batch_size", "8"  # Reduced for memory
    ]
    
    run_main()

if __name__ == "__main__":
    main()

