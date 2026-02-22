"""
Advanced BGE Training with Hard Negative Mining, Domain-Specific Fine-tuning,
and Advanced Loss Functions to Improve Retrieval Performance
"""

import pathlib
import logging
import random
import numpy as np
import argparse
import json
import os
from datetime import datetime
from typing import List, Dict, Tuple, Optional
from beir.datasets.data_loader import GenericDataLoader
from sentence_transformers import SentenceTransformer, InputExample, losses
from sentence_transformers.evaluation import InformationRetrievalEvaluator
from torch.utils.data import DataLoader
import torch
from sklearn.metrics.pairwise import cosine_similarity

# --- Configuration ---
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO)

def set_seed(seed=42):
    """Sets the seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logging.info(f"🔒 Random seed set to {seed} for reproducibility.")

BASE_MODEL_NAME = "BAAI/bge-base-en-v1.5"
MTRAG_DOMAINS = ["clapnq", "fiqa", "govt", "cloud"]

def load_mtrag_examples(domain, split="train", use_data_splits=True):
    """Loads training examples from a domain."""
    logging.info(f"Loading MTRAG domain: {domain} (split: {split})...")
    data_root = pathlib.Path(".")
    
    if use_data_splits:
        corpus_file = data_root / "corpora" / "passage_level" / f"{domain}.jsonl"
        query_file = data_root / "data_splits" / "retrieval_tasks" / domain / split / f"{domain}_questions.jsonl"
        qrels_file = data_root / "data_splits" / "retrieval_tasks" / domain / split / "qrels" / "dev.tsv"
    else:
        corpus_file = data_root / "corpora" / "passage_level" / f"{domain}.jsonl"
        query_file = data_root / "human" / "retrieval_tasks" / domain / f"{domain}_questions.jsonl"
        qrels_file = data_root / "human" / "retrieval_tasks" / domain / "qrels" / "dev.tsv"
    
    if not query_file.exists() and use_data_splits:
        logging.warning(f"Split file not found: {query_file}, using original location")
        query_file = data_root / "human" / "retrieval_tasks" / domain / f"{domain}_questions.jsonl"
        qrels_file = data_root / "human" / "retrieval_tasks" / domain / "qrels" / "dev.tsv"
    
    try:
        corpus, queries, qrels = GenericDataLoader(
            corpus_file=str(corpus_file),
            query_file=str(query_file),
            qrels_file=str(qrels_file)
        ).load_custom()
    except Exception as e:
        logging.error(f"Error loading {domain}: {e}. Skipping.")
        return [], {}, {}, {}
    
    return corpus, queries, qrels

def mine_hard_negatives(
    model: SentenceTransformer,
    query_text: str,
    positive_doc_ids: List[str],
    corpus: Dict,
    corpus_ids: List[str],
    top_k: int = 5,
    device: str = "cuda"
) -> List[str]:
    """
    Mine hard negatives - passages that are similar to query but not relevant.
    These are challenging negatives that help the model learn fine-grained distinctions.
    """
    # Get query embedding
    query_emb = model.encode([query_text], convert_to_numpy=True, show_progress_bar=False)
    
    # Get all passage embeddings
    all_passage_texts = []
    all_passage_ids = []
    for doc_id in corpus_ids:
        if doc_id not in positive_doc_ids:  # Only consider non-positive passages
            doc = corpus.get(doc_id)
            if doc:
                title = doc.get("title", "")
                text = doc.get("text", "")
                doc_text = f"{title} {text}".strip() if title else text
                all_passage_texts.append(doc_text)
                all_passage_ids.append(doc_id)
    
    if len(all_passage_texts) == 0:
        return []
    
    # Encode passages in batches
    passage_embs = model.encode(
        all_passage_texts,
        batch_size=32,
        convert_to_numpy=True,
        show_progress_bar=False
    )
    
    # Compute similarities
    similarities = cosine_similarity(query_emb, passage_embs)[0]
    
    # Get top-k most similar (but incorrect) passages
    top_indices = np.argsort(similarities)[::-1][:top_k]
    hard_negatives = [all_passage_ids[idx] for idx in top_indices]
    
    return hard_negatives

def create_training_examples_with_hard_negatives(
    model: SentenceTransformer,
    corpus: Dict,
    queries: Dict,
    qrels: Dict,
    num_hard_negatives: int = 3,
    use_hard_negatives: bool = True,
    device: str = "cuda"
) -> List[InputExample]:
    """
    Create training examples with hard negatives.
    For each positive pair, we mine hard negatives that are similar but incorrect.
    """
    examples = []
    corpus_ids = list(corpus.keys())
    
    for query_id, doc_infos in qrels.items():
        query_text = queries.get(query_id)
        if not query_text:
            continue
        
        positive_doc_ids = [doc_id for doc_id, score in doc_infos.items() if score > 0]
        
        # Create positive examples
        for doc_id in positive_doc_ids:
            doc = corpus.get(doc_id)
            if doc:
                title = doc.get("title", "")
                text = doc.get("text", "")
                doc_text = f"{title} {text}".strip() if title else text
                
                # Add positive example
                examples.append(InputExample(texts=[query_text, doc_text], label=1.0))
        
        # Mine hard negatives if enabled
        if use_hard_negatives and len(positive_doc_ids) > 0:
            hard_neg_ids = mine_hard_negatives(
                model, query_text, positive_doc_ids, corpus, corpus_ids,
                top_k=num_hard_negatives, device=device
            )
            
            for neg_doc_id in hard_neg_ids:
                neg_doc = corpus.get(neg_doc_id)
                if neg_doc:
                    title = neg_doc.get("title", "")
                    text = neg_doc.get("text", "")
                    doc_text = f"{title} {text}".strip() if title else text
                    examples.append(InputExample(texts=[query_text, doc_text], label=0.0))
    
    return examples

def load_training_data_advanced(
    model: SentenceTransformer,
    domains: List[str],
    split: str = "train",
    use_hard_negatives: bool = True,
    num_hard_negatives: int = 3,
    use_data_splits: bool = True,
    device: str = "cuda"
) -> List[InputExample]:
    """Load training data with hard negative mining."""
    all_examples = []
    
    for domain in domains:
        corpus, queries, qrels = load_mtrag_examples(domain, split=split, use_data_splits=use_data_splits)
        
        if not corpus or not queries or not qrels:
            continue
        
        logging.info(f"Mining hard negatives for {domain}...")
        domain_examples = create_training_examples_with_hard_negatives(
            model, corpus, queries, qrels,
            num_hard_negatives=num_hard_negatives,
            use_hard_negatives=use_hard_negatives,
            device=device
        )
        
        all_examples.extend(domain_examples)
        logging.info(f"Created {len(domain_examples)} examples for {domain}")
    
    return all_examples

def load_validation_data(domains: List[str], use_data_splits: bool = True):
    """Load validation data for evaluation during training."""
    val_corpus = {}
    val_queries = {}
    val_qrels = {}
    
    for domain in domains:
        corpus, queries, qrels = load_mtrag_examples(domain, split="val", use_data_splits=use_data_splits)
        if corpus:
            for k, v in corpus.items():
                val_corpus[f"{domain}_{k}"] = v
            for k, v in queries.items():
                val_queries[f"{domain}_{k}"] = v
            for k, v in qrels.items():
                val_qrels[f"{domain}_{k}"] = {f"{domain}_{dk}": dv for dk, dv in v.items()}
    
    return val_corpus, val_queries, val_qrels

def run_advanced_training(config: Dict):
    """Run advanced training with hard negative mining and improved techniques."""
    set_seed(config.get('seed', 42))
    
    logging.info("="*60)
    
    # Check if resuming from checkpoint
    resume_from_checkpoint = config.get('resume_from_checkpoint')
    if resume_from_checkpoint:
        logging.info(f"🔄 RESUMING training: {config['experiment_name']}")
        logging.info(f"   From checkpoint: {resume_from_checkpoint}")
    else:
        logging.info(f"🚀 Starting NEW training: {config['experiment_name']}")
    
    logging.info("="*60)
    
    # Support GPU device assignment
    if not torch.cuda.is_available():
        logging.warning("⚠️ CUDA not available. Training will be slow.")
        device = "cpu"
    else:
        # If CUDA_VISIBLE_DEVICES is set, use device 0 (it will be the selected GPU)
        # Otherwise, use the specified gpu_id
        gpu_id = config.get('gpu_id')
        if 'CUDA_VISIBLE_DEVICES' in os.environ:
            # GPU already selected via environment variable, use device 0
            device = "cuda:0"
            logging.info(f"Using GPU (via CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']})")
        elif gpu_id is not None:
            if gpu_id >= torch.cuda.device_count():
                logging.warning(f"GPU {gpu_id} not available, using default GPU")
                device = "cuda"
            else:
                device = f"cuda:{gpu_id}"
                logging.info(f"Using GPU {gpu_id}")
        else:
            device = "cuda"
    
    # Load model (from checkpoint or base model)
    if resume_from_checkpoint and pathlib.Path(resume_from_checkpoint).exists():
        checkpoint_path = pathlib.Path(resume_from_checkpoint)
        logging.info(f"📂 Loading model from checkpoint: {checkpoint_path}")
        model = SentenceTransformer(str(checkpoint_path.absolute()), device=device)
        
        # Try to load checkpoint info to adjust training parameters
        state_file = checkpoint_path / "trainer_state.json"
        if state_file.exists():
            try:
                with open(state_file) as f:
                    checkpoint_state = json.load(f)
                completed_epochs = checkpoint_state.get('epoch', 0.0)
                completed_steps = checkpoint_state.get('global_step', 0)
                total_epochs = config.get('epochs', 5)
                
                logging.info(f"   Checkpoint state: {completed_steps} steps, {completed_epochs:.2f} epochs completed")
                
                # Note: SentenceTransformer's fit() will continue from where it left off
                # but we can't easily adjust epochs, so we keep the original epoch count
                # It will continue training for the full epoch count, which is acceptable
                logging.info(f"   Will continue training for {total_epochs} total epochs")
            except Exception as e:
                logging.warning(f"   Could not load checkpoint state: {e}")
    else:
        base_model_name = config.get('base_model', BASE_MODEL_NAME)
        logging.info(f"Loading base model: {base_model_name}")
        model = SentenceTransformer(base_model_name, device=device)
    
    # Load training data with hard negatives
    # If resuming, we can skip hard negative mining if training examples were cached
    # However, to be safe and ensure consistency, we'll redo mining but it's faster
    # with the resumed model (already trained, better embeddings)
    if resume_from_checkpoint:
        logging.info("🔄 Resuming: Re-mining hard negatives with resumed model (faster than initial mining)...")
    else:
        logging.info("Loading training data with hard negative mining...")
    
    train_examples = load_training_data_advanced(
        model=model,  # Use current model (checkpoint or base) for hard negative mining
        domains=MTRAG_DOMAINS,
        split="train",
        use_hard_negatives=config.get('use_hard_negatives', True),
        num_hard_negatives=config.get('num_hard_negatives', 3),
        use_data_splits=config.get('use_data_splits', True),
        device=device
    )
    
    logging.info(f"Total training examples: {len(train_examples)}")
    
    if not train_examples:
        logging.error("No training examples found. Exiting.")
        return None
    
    # Create dataloader
    train_dataloader = DataLoader(
        train_examples,
        shuffle=True,
        batch_size=config.get('batch_size', 32)
    )
    
    # Setup advanced loss function
    loss_type = config.get('loss_function', 'CosineSimilarityLoss')
    if loss_type == 'CosineSimilarityLoss':
        # Use CosineSimilarityLoss for explicit positive/negative pairs
        train_loss = losses.CosineSimilarityLoss(model=model)
    elif loss_type == 'MultipleNegativesRankingLoss':
        train_loss = losses.MultipleNegativesRankingLoss(model=model)
    elif loss_type == 'TripletLoss':
        from sentence_transformers.losses import TripletLoss, TripletDistanceMetric
        train_loss = TripletLoss(
            model=model,
            distance_metric=TripletDistanceMetric.COSINE,
            triplet_margin=1.0
        )
    else:
        train_loss = losses.CosineSimilarityLoss(model=model)
    
    logging.info(f"Using loss function: {loss_type}")
    
    # Setup validation evaluator
    evaluator = None
    if config.get('use_validation', True):
        try:
            val_corpus, val_queries, val_qrels = load_validation_data(
                MTRAG_DOMAINS,
                use_data_splits=config.get('use_data_splits', True)
            )
            if val_corpus and val_queries and val_qrels:
                evaluator = InformationRetrievalEvaluator(
                    queries=val_queries,
                    corpus=val_corpus,
                    relevant_docs=val_qrels,
                    show_progress_bar=True,
                    name="validation"
                )
                logging.info(f"✅ Validation evaluator created with {len(val_queries)} queries")
        except Exception as e:
            logging.warning(f"Could not create validation evaluator: {e}")
    
    # Training parameters
    output_path = config.get('output_path', f"./models/{config['experiment_name']}")
    pathlib.Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Start training
    if resume_from_checkpoint:
        logging.info(f"Resuming training config:")
    else:
        logging.info(f"Training config:")
    logging.info(f"  - Epochs: {config.get('epochs', 5)}")
    logging.info(f"  - Batch Size: {config.get('batch_size', 32)}")
    logging.info(f"  - Learning Rate: {config.get('learning_rate', 2e-5)}")
    logging.info(f"  - Hard Negatives: {config.get('num_hard_negatives', 3)} per positive")
    logging.info(f"  - Output: {output_path}")
    
    # Note: SentenceTransformer's fit() method will automatically continue from checkpoint
    # if the output_path matches and checkpoints exist. When resuming from a specific
    # checkpoint, we've already loaded that model, so training will continue normally.
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=config.get('epochs', 5),
        warmup_steps=config.get('warmup_steps', 100),
        optimizer_params={'lr': config.get('learning_rate', 2e-5)},
        evaluator=evaluator,
        evaluation_steps=config.get('evaluation_steps', 500) if evaluator else None,
        output_path=output_path,
        save_best_model=config.get('save_best_model', True) if evaluator else False,
        show_progress_bar=True,
        checkpoint_save_steps=config.get('checkpoint_steps', 1000),
        checkpoint_path=f"{output_path}-checkpoints" if config.get('save_checkpoints', True) else None
    )
    
    logging.info(f"✅ Training complete. Model saved to: {output_path}")
    return output_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Advanced BGE Training with Hard Negatives')
    parser.add_argument('--config', type=str, help='Path to config JSON file')
    parser.add_argument('--experiment_name', type=str, default='advanced_experiment_1')
    parser.add_argument('--base_model', type=str, default='BAAI/bge-base-en-v1.5')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=2e-5)
    parser.add_argument('--loss_function', type=str, default='CosineSimilarityLoss',
                        choices=['CosineSimilarityLoss', 'MultipleNegativesRankingLoss', 'TripletLoss'])
    parser.add_argument('--num_hard_negatives', type=int, default=3)
    parser.add_argument('--use_hard_negatives', action='store_true', default=True)
    parser.add_argument('--use_validation', action='store_true', default=True)
    parser.add_argument('--use_data_splits', action='store_true', default=True)
    
    args = parser.parse_args()
    
    if args.config:
        with open(args.config, 'r') as f:
            config = json.load(f)
    else:
        config = {
            'experiment_name': args.experiment_name,
            'base_model': args.base_model,
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'learning_rate': args.learning_rate,
            'loss_function': args.loss_function,
            'num_hard_negatives': args.num_hard_negatives,
            'use_hard_negatives': args.use_hard_negatives,
            'use_validation': args.use_validation,
            'use_data_splits': args.use_data_splits,
            'warmup_steps': 100,
            'evaluation_steps': 500,
            'save_best_model': True,
            'save_checkpoints': True,
            'checkpoint_steps': 1000,
            'seed': 42
        }
    
    run_advanced_training(config)

