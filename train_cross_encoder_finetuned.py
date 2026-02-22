"""
Fine-tune Cross-Encoder for Reranking with Checkpointing and Resume Support
This is the CRITICAL experiment for ACL submission - highest impact on nDCG
"""

import pathlib
import logging
import argparse
import json
import os
import random
import numpy as np
from typing import Dict, List, Optional, Tuple
from beir.datasets.data_loader import GenericDataLoader
from sentence_transformers import CrossEncoder, InputExample
from sentence_transformers.cross_encoder.evaluation import CEBinaryClassificationEvaluator
from torch.utils.data import DataLoader
import torch
import signal
import sys
from datetime import datetime

logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO)

MTRAG_DOMAINS = ["clapnq", "fiqa", "govt", "cloud"]

# Global variables for graceful shutdown
shutdown_requested = False
current_checkpoint_dir = None

def signal_handler(sig, frame):
    """Handle SIGINT (Ctrl+C) gracefully"""
    global shutdown_requested
    logging.info("\n⚠️  Shutdown signal received. Saving checkpoint and exiting gracefully...")
    shutdown_requested = True

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

def set_seed(seed=42):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_training_data(domain: str, data_root: pathlib.Path, use_data_splits: bool = True) -> List[InputExample]:
    """
    Load training data for cross-encoder fine-tuning.
    Creates (query, positive_doc, label=1) and (query, negative_doc, label=0) pairs.
    """
    logging.info(f"Loading training data for domain: {domain}...")
    
    corpus_file = data_root / "corpora" / "passage_level" / f"{domain}.jsonl"
    
    if use_data_splits:
        query_file = data_root / "data_splits" / "retrieval_tasks" / domain / "train" / f"{domain}_questions.jsonl"
        qrels_file = data_root / "data_splits" / "retrieval_tasks" / domain / "train" / "qrels" / "dev.tsv"
    else:
        query_file = data_root / "human" / "retrieval_tasks" / domain / f"{domain}_questions.jsonl"
        qrels_file = data_root / "human" / "retrieval_tasks" / domain / "qrels" / "dev.tsv"
    
    # Fallback to original data if splits don't exist
    if use_data_splits and not query_file.exists():
        logging.warning(f"Data splits not found for {domain}, using original data")
        query_file = data_root / "human" / "retrieval_tasks" / domain / f"{domain}_questions.jsonl"
        qrels_file = data_root / "human" / "retrieval_tasks" / domain / "qrels" / "dev.tsv"
    
    try:
        corpus, queries, qrels = GenericDataLoader(
            corpus_file=str(corpus_file),
            query_file=str(query_file),
            qrels_file=str(qrels_file)
        ).load_custom()
    except Exception as e:
        logging.error(f"Error loading data for {domain}: {e}")
        return []
    
    examples = []
    
    # Create positive examples (query, relevant_doc, label=1)
    for query_id, doc_scores in qrels.items():
        query_text = queries.get(query_id)
        if not query_text:
            continue
        
        # Get positive documents (score > 0)
        positive_docs = [doc_id for doc_id, score in doc_scores.items() if score > 0]
        
        for doc_id in positive_docs:
            doc = corpus.get(doc_id)
            if doc:
                title = doc.get("title", "")
                text = doc.get("text", "")
                doc_text = f"{title} {text}".strip() if title else text
                examples.append(InputExample(texts=[query_text, doc_text], label=1))
        
        # Create negative examples (query, non-relevant_doc, label=0)
        # Sample negatives from corpus (not in positive set)
        positive_set = set(positive_docs)
        negative_candidates = [doc_id for doc_id in corpus.keys() if doc_id not in positive_set]
        
        # Sample up to 2 negatives per query (to balance dataset)
        num_negatives = min(2, len(negative_candidates))
        if num_negatives > 0:
            negative_docs = random.sample(negative_candidates, num_negatives)
            for doc_id in negative_docs:
                doc = corpus.get(doc_id)
                if doc:
                    title = doc.get("title", "")
                    text = doc.get("text", "")
                    doc_text = f"{title} {text}".strip() if title else text
                    examples.append(InputExample(texts=[query_text, doc_text], label=0))
    
    logging.info(f"Created {len(examples)} training examples for {domain}")
    return examples

def load_checkpoint(checkpoint_dir: pathlib.Path) -> Optional[Dict]:
    """Load checkpoint information if it exists"""
    checkpoint_file = checkpoint_dir / "checkpoint_info.json"
    if checkpoint_file.exists():
        with open(checkpoint_file, 'r') as f:
            return json.load(f)
    return None

def save_checkpoint(checkpoint_dir: pathlib.Path, epoch: int, step: int, model_path: str):
    """Save checkpoint information"""
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_info = {
        "epoch": epoch,
        "step": step,
        "model_path": model_path,
        "timestamp": datetime.now().isoformat()
    }
    checkpoint_file = checkpoint_dir / "checkpoint_info.json"
    with open(checkpoint_file, 'w') as f:
        json.dump(checkpoint_info, f, indent=2)
    logging.info(f"Checkpoint saved: epoch={epoch}, step={step}")

def train_cross_encoder(
    config: Dict,
    gpu_id: Optional[int] = None
) -> str:
    """
    Fine-tune cross-encoder model with checkpointing and resume support.
    
    Returns:
        Path to the fine-tuned model
    """
    global shutdown_requested, current_checkpoint_dir
    
    experiment_name = config.get('experiment_name', 'cross_encoder_finetuned')
    base_model = config.get('base_model', 'cross-encoder/ms-marco-MiniLM-L-12-v2')
    domains = config.get('domains', MTRAG_DOMAINS)
    use_data_splits = config.get('use_data_splits', True)
    epochs = config.get('epochs', 3)
    batch_size = config.get('batch_size', 16)
    learning_rate = config.get('learning_rate', 2e-5)
    warmup_steps = config.get('warmup_steps', 100)
    checkpoint_save_steps = config.get('checkpoint_save_steps', 500)
    resume = config.get('resume', True)
    
    # Set GPU
    if gpu_id is not None:
        # Only set if not already set (to avoid conflicts)
        if 'CUDA_VISIBLE_DEVICES' not in os.environ:
            os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        logging.info(f"Using GPU {gpu_id} (CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')})")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available():
        actual_device = torch.cuda.current_device()
        logging.info(f"Device: {device} (actual device: {actual_device})")
    else:
        logging.info(f"Device: {device}")
    
    # Set seed
    seed = config.get('seed', 42)
    set_seed(seed)
    
    # Setup paths
    data_root = pathlib.Path(".")
    output_dir = pathlib.Path("models") / experiment_name
    checkpoint_dir = output_dir / "checkpoints"
    current_checkpoint_dir = checkpoint_dir
    
    # Load checkpoint if resuming
    start_epoch = 0
    start_step = 0
    checkpoint_info = None
    
    if resume:
        checkpoint_info = load_checkpoint(checkpoint_dir)
        if checkpoint_info:
            start_epoch = checkpoint_info.get('epoch', 0)
            start_step = checkpoint_info.get('step', 0)
            model_path = checkpoint_info.get('model_path')
            if model_path and pathlib.Path(model_path).exists():
                logging.info(f"Resuming from checkpoint: epoch={start_epoch}, step={start_step}")
                base_model = model_path
            else:
                logging.warning("Checkpoint model not found, starting from base model")
                start_epoch = 0
                start_step = 0
    
    # Load training data
    logging.info("Loading training data...")
    all_examples = []
    for domain in domains:
        examples = load_training_data(domain, data_root, use_data_splits)
        all_examples.extend(examples)
    
    if len(all_examples) == 0:
        logging.error("No training examples found!")
        return None
    
    logging.info(f"Total training examples: {len(all_examples)}")
    
    # Shuffle examples
    random.shuffle(all_examples)
    
    # Custom collate function for CrossEncoder
    def collate_fn(batch):
        texts = [[ex.texts[0], ex.texts[1]] for ex in batch]
        labels = [ex.label for ex in batch]
        return texts, labels
    
    # Create data loader with custom collate function
    train_dataloader = DataLoader(all_examples, shuffle=True, batch_size=batch_size, collate_fn=collate_fn)
    
    # Load model
    logging.info(f"Loading base model: {base_model}")
    model = CrossEncoder(base_model, num_labels=1, max_length=512)
    model.to(device)
    
    # Setup optimizer
    from torch.optim import AdamW
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    
    # Training loop with checkpointing
    logging.info(f"Starting training: epochs={epochs}, batch_size={batch_size}, lr={learning_rate}")
    logging.info(f"Total steps per epoch: {len(train_dataloader)}")
    
    total_steps = 0
    steps_since_checkpoint = 0
    
    try:
        for epoch in range(start_epoch, epochs):
            if shutdown_requested:
                logging.info("Shutdown requested, saving checkpoint...")
                save_checkpoint(checkpoint_dir, epoch, total_steps, str(output_dir))
                break
            
            logging.info(f"\n{'='*60}")
            logging.info(f"Epoch {epoch + 1}/{epochs}")
            logging.info(f"{'='*60}")
            
            epoch_loss = 0.0
            num_batches = 0
            
            for batch_idx, batch in enumerate(train_dataloader):
                if shutdown_requested:
                    logging.info("Shutdown requested, saving checkpoint...")
                    save_checkpoint(checkpoint_dir, epoch, total_steps, str(output_dir))
                    break
                
                # Skip batches if resuming
                if epoch == start_epoch and batch_idx < (start_step // batch_size):
                    continue
                
                # Batch is already prepared by collate_fn
                texts, labels = batch
                labels = torch.tensor(labels, dtype=torch.float32).to(device)
                
                # Forward pass - use model's internal forward method for training
                # CrossEncoder has a tokenizer attribute
                # Tokenize the input pairs [query, doc]
                features = model.tokenizer(texts, padding=True, truncation=True, 
                                          return_tensors='pt', max_length=512)
                features = {k: v.to(device) for k, v in features.items()}
                
                # Forward through the model (model.model is the underlying transformer)
                outputs = model.model(**features)
                scores = outputs.logits.squeeze(-1)  # Get logits and squeeze
                
                # Compute loss (binary cross-entropy with logits)
                loss_fn = torch.nn.BCEWithLogitsLoss()
                loss = loss_fn(scores, labels)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
                total_steps += 1
                steps_since_checkpoint += 1
                
                # Save checkpoint periodically
                if steps_since_checkpoint >= checkpoint_save_steps:
                    checkpoint_path = checkpoint_dir / f"checkpoint-{total_steps}"
                    checkpoint_path.mkdir(parents=True, exist_ok=True)
                    model.save(str(checkpoint_path))
                    save_checkpoint(checkpoint_dir, epoch, total_steps, str(checkpoint_path))
                    steps_since_checkpoint = 0
                    logging.info(f"Checkpoint saved at step {total_steps}")
                
                # Log progress
                if (batch_idx + 1) % 100 == 0:
                    avg_loss = epoch_loss / num_batches
                    logging.info(f"Step {total_steps} | Batch {batch_idx + 1}/{len(train_dataloader)} | Loss: {avg_loss:.4f}")
            
            if shutdown_requested:
                break
            
            # Save epoch checkpoint
            epoch_checkpoint = checkpoint_dir / f"epoch-{epoch + 1}"
            epoch_checkpoint.mkdir(parents=True, exist_ok=True)
            model.save(str(epoch_checkpoint))
            save_checkpoint(checkpoint_dir, epoch + 1, total_steps, str(epoch_checkpoint))
            
            avg_epoch_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
            logging.info(f"Epoch {epoch + 1} completed | Avg Loss: {avg_epoch_loss:.4f}")
        
        # Save final model
        if not shutdown_requested:
            logging.info("Saving final model...")
            output_dir.mkdir(parents=True, exist_ok=True)
            model.save(str(output_dir))
            save_checkpoint(checkpoint_dir, epochs, total_steps, str(output_dir))
            logging.info(f"Final model saved to: {output_dir}")
            return str(output_dir)
        else:
            logging.info("Training interrupted. Use --resume to continue from checkpoint.")
            return str(checkpoint_dir)
    
    except Exception as e:
        logging.error(f"Error during training: {e}", exc_info=True)
        save_checkpoint(checkpoint_dir, epoch, total_steps, str(output_dir))
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Fine-tune Cross-Encoder with Checkpointing')
    parser.add_argument('--config', type=str, required=True, help='Config JSON file')
    parser.add_argument('--gpu_id', type=int, help='GPU ID')
    parser.add_argument('--resume', action='store_true', default=True, help='Resume from checkpoint if available')
    parser.add_argument('--no-resume', dest='resume', action='store_false', help='Do not resume from checkpoint')
    
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    config['resume'] = args.resume
    
    try:
        model_path = train_cross_encoder(config, args.gpu_id)
        if model_path:
            logging.info(f"✅ Training completed. Model saved to: {model_path}")
    except KeyboardInterrupt:
        logging.info("Training interrupted by user")
        sys.exit(0)
    except Exception as e:
        logging.error(f"Training failed: {e}", exc_info=True)
        sys.exit(1)

