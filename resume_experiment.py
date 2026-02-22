#!/usr/bin/env python3
"""
Resume a paused experiment from the latest checkpoint.

Usage:
    python resume_experiment.py --experiment phase4_hard_negatives_cosine [--checkpoint PATH] [--gpu_id 0]
"""
import argparse
import json
import pathlib
import logging
import sys
from sentence_transformers import SentenceTransformer
from train_advanced_bge import run_advanced_training

logging.basicConfig(
    format='%(asctime)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO
)

def find_latest_checkpoint(experiment_name: str) -> pathlib.Path:
    """Find the latest checkpoint for an experiment."""
    # Check two possible locations:
    # 1. experiments/retrieval/{exp}/models/{exp}-checkpoints/checkpoint-*
    # 2. ./models/{exp}-checkpoints/checkpoint-*
    
    exp_dir = pathlib.Path(f"experiments/retrieval/{experiment_name}")
    models_dir_exp = exp_dir / "models" / f"{experiment_name}-checkpoints"
    models_dir_root = pathlib.Path(f"./models/{experiment_name}-checkpoints")
    
    checkpoint_dirs = []
    
    # Check experiment-specific models directory
    if models_dir_exp.exists():
        checkpoint_dirs.extend(list(models_dir_exp.glob("checkpoint-*")))
    
    # Check root models directory
    if models_dir_root.exists():
        checkpoint_dirs.extend(list(models_dir_root.glob("checkpoint-*")))
    
    if not checkpoint_dirs:
        return None
    
    # Sort by checkpoint number (extract from path)
    def get_step(cp):
        try:
            return int(cp.name.split("-")[-1])
        except:
            return 0
    
    latest = sorted(checkpoint_dirs, key=get_step, reverse=True)[0]
    return latest

def load_checkpoint_info(checkpoint_path: pathlib.Path) -> dict:
    """Load training state from checkpoint."""
    state_file = checkpoint_path / "trainer_state.json"
    if not state_file.exists():
        logging.warning(f"trainer_state.json not found in {checkpoint_path}")
        return None
    
    try:
        with open(state_file) as f:
            state = json.load(f)
        
        return {
            "global_step": state.get("global_step", 0),
            "epoch": state.get("epoch", 0.0),
            "best_model_checkpoint": state.get("best_model_checkpoint"),
            "max_steps": state.get("max_steps"),
            "num_train_epochs": state.get("num_train_epochs", 1)
        }
    except Exception as e:
        logging.warning(f"Error loading checkpoint info: {e}")
        return None

def resume_training(experiment_name: str, checkpoint_path: pathlib.Path = None, gpu_id: int = None):
    """Resume training from checkpoint."""
    exp_dir = pathlib.Path(f"experiments/retrieval/{experiment_name}")
    config_file = exp_dir / "config.json"
    
    if not config_file.exists():
        raise FileNotFoundError(f"Config not found: {config_file}")
    
    with open(config_file) as f:
        config = json.load(f)
    
    # Find latest checkpoint if not specified
    if checkpoint_path is None:
        checkpoint_path = find_latest_checkpoint(experiment_name)
        if checkpoint_path is None:
            raise ValueError(
                f"No checkpoint found for {experiment_name}.\n"
                f"Searched in:\n"
                f"  - {exp_dir / 'models' / f'{experiment_name}-checkpoints'}\n"
                f"  - ./models/{experiment_name}-checkpoints\n"
                f"\nNote: Checkpoints are saved every 1000 steps. If training hasn't reached step 1000 yet, no checkpoint exists."
            )
    
    checkpoint_path = pathlib.Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint path does not exist: {checkpoint_path}")
    
    logging.info("="*60)
    logging.info(f"🔄 RESUMING EXPERIMENT: {experiment_name}")
    logging.info("="*60)
    logging.info(f"📂 Checkpoint: {checkpoint_path}")
    
    # Load checkpoint info
    checkpoint_info = load_checkpoint_info(checkpoint_path)
    if checkpoint_info:
        logging.info(f"   Step: {checkpoint_info['global_step']}")
        logging.info(f"   Epoch: {checkpoint_info['epoch']:.2f}")
        if checkpoint_info.get('max_steps'):
            progress = (checkpoint_info['global_step'] / checkpoint_info['max_steps']) * 100
            logging.info(f"   Progress: {progress:.1f}%")
    else:
        logging.warning("Could not load checkpoint info, but checkpoint directory exists")
    
    # Modify config to resume from checkpoint
    # The checkpoint path becomes the base_model (it's a saved SentenceTransformer model)
    config['resume_from_checkpoint'] = str(checkpoint_path.absolute())
    config['base_model'] = str(checkpoint_path.absolute())  # Load from checkpoint
    
    if gpu_id is not None:
        config['gpu_id'] = gpu_id
    
    logging.info(f"🚀 Resuming training...")
    logging.info(f"   Will load model from: {checkpoint_path}")
    
    try:
        result = run_advanced_training(config)
        logging.info(f"✅ Training resumed and completed successfully!")
        return result
    except Exception as e:
        logging.error(f"❌ Error during resumed training: {e}")
        raise

def main():
    parser = argparse.ArgumentParser(
        description='Resume a paused experiment from checkpoint',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Resume with auto-detected latest checkpoint
  python resume_experiment.py --experiment phase4_hard_negatives_cosine

  # Resume from specific checkpoint
  python resume_experiment.py --experiment phase4_hard_negatives_cosine \\
      --checkpoint ./models/phase4_hard_negatives_cosine-checkpoints/checkpoint-2000

  # Resume on specific GPU
  python resume_experiment.py --experiment phase4_hard_negatives_cosine --gpu_id 0
        """
    )
    parser.add_argument(
        "--experiment",
        required=True,
        help="Experiment name (e.g., phase4_hard_negatives_cosine)"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        help="Specific checkpoint path (optional, will auto-detect latest)"
    )
    parser.add_argument(
        "--gpu_id",
        type=int,
        help="GPU ID to use for training"
    )
    
    args = parser.parse_args()
    
    try:
        checkpoint_path = pathlib.Path(args.checkpoint) if args.checkpoint else None
        resume_training(args.experiment, checkpoint_path, args.gpu_id)
    except Exception as e:
        logging.error(f"Failed to resume experiment: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

