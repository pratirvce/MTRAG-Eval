#!/usr/bin/env python3
"""Tier1 Wrapper for Differentiable End-to-End Retrieval Pipeline"""
import os, sys, json, pathlib, argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_name', type=str, required=True)
    parser.add_argument('--gpu', type=int, required=True)
    args = parser.parse_known_args()[0]
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    from train_differentiable_end_to_end import train_differentiable_end_to_end
    output_dir = pathlib.Path(f"experiments/retrieval/{args.experiment_name}")
    output_dir.mkdir(parents=True, exist_ok=True)
    config = {"output_dir": str(output_dir), "model_path": "BAAI/bge-base-en-v1.5", "epochs": 3, "batch_size": 4, "gradient_accumulation_steps": 2, "use_fp16": True, "use_data_splits": True, "domains": ["clapnq", "fiqa", "govt", "cloud"], "resume": True, "candidate_pool_size": 100}
    with open(output_dir / "config.json", 'w') as f: json.dump(config, f, indent=2)
    try:
        results_path = train_differentiable_end_to_end(config, gpu_id=args.gpu)
        print(f"✅ Experiment completed: {results_path}" if results_path else "❌ Experiment failed")
        sys.exit(0 if results_path else 1)
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback; traceback.print_exc()
        sys.exit(1)
