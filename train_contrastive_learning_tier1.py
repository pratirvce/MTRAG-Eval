#!/usr/bin/env python3
"""
Tier 1: Contrastive Learning with Conversation-Document Pairs
Priority 5 - Training-based experiment
Trains retrieval model using contrastive learning on conversation-document pairs
"""

import sys
import pathlib
import argparse
import json
import logging
import numpy as np
from typing import Dict, List, Optional
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.models import SentenceBERT
from beir.retrieval.search.dense import DenseRetrievalExactSearch
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
import torch
import signal
import random

logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO)

MTRAG_DOMAINS = ["clapnq", "fiqa", "govt", "cloud"]
shutdown_requested = False

def signal_handler(sig, frame):
    global shutdown_requested
    logging.info("\n⚠️  Shutdown signal received. Saving checkpoint...")
    shutdown_requested = True

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_training_pairs(domain: str, data_root: pathlib.Path, use_data_splits: bool = True) -> List[InputExample]:
    """Load conversation-document pairs for contrastive learning"""
    logging.info(f"Loading training pairs for domain: {domain}...")
    
    corpus_file = data_root / "corpora" / "passage_level" / f"{domain}.jsonl"
    
    if use_data_splits:
        query_file = data_root / "data_splits" / "retrieval_tasks" / domain / "train" / f"{domain}_questions.jsonl"
        qrels_file = data_root / "data_splits" / "retrieval_tasks" / domain / "train" / "qrels" / "dev.tsv"
    else:
        query_file = data_root / "human" / "retrieval_tasks" / domain / f"{domain}_questions.jsonl"
        qrels_file = data_root / "human" / "retrieval_tasks" / domain / "qrels" / "dev.tsv"
    
    if not query_file.exists() and use_data_splits:
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
    
    # Create positive pairs: (query, relevant_doc)
    for query_id, doc_scores in qrels.items():
        query_text = queries.get(query_id)
        if not query_text:
            continue
        
        # Get positive documents
        positive_docs = [doc_id for doc_id, score in doc_scores.items() if score > 0]
        
        for doc_id in positive_docs:
            doc = corpus.get(doc_id)
            if doc:
                title = doc.get("title", "")
                text = doc.get("text", "")
                doc_text = f"{title} {text}".strip() if title else text
                examples.append(InputExample(texts=[query_text, doc_text], label=1.0))
    
    logging.info(f"Created {len(examples)} positive pairs for {domain}")
    return examples

def train_contrastive_model(config: Dict, gpu_id: int, output_dir: pathlib.Path):
    """Train model with contrastive learning"""
    base_model = config.get('base_model', 'BAAI/bge-base-en-v1.5')
    domains = config.get('domains', MTRAG_DOMAINS)
    use_data_splits = config.get('use_data_splits', True)
    epochs = config.get('epochs', 3)
    batch_size = config.get('batch_size', 16)
    learning_rate = config.get('learning_rate', 2e-5)
    resume = config.get('resume', True)
    
    # Set GPU
    if gpu_id is not None:
        import os
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Using device: {device}")
    
    set_seed(42)
    
    # Check for existing model
    model_dir = output_dir / "model"
    start_epoch = 0
    
    if resume and model_dir.exists():
        try:
            logging.info(f"Loading existing model from {model_dir}")
            model = SentenceTransformer(str(model_dir), device=device)
            # Try to load checkpoint info
            checkpoint_file = output_dir / "checkpoint.json"
            if checkpoint_file.exists():
                with open(checkpoint_file, 'r') as f:
                    checkpoint = json.load(f)
                    start_epoch = checkpoint.get('epoch', 0)
                    logging.info(f"Resuming from epoch {start_epoch}")
        except:
            logging.info("Could not load existing model, starting fresh")
            model = SentenceTransformer(base_model, device=device)
    else:
        model = SentenceTransformer(base_model, device=device)
    
    # Load training data
    logging.info("Loading training data...")
    all_examples = []
    data_root = pathlib.Path(".")
    
    for domain in domains:
        examples = load_training_pairs(domain, data_root, use_data_splits)
        all_examples.extend(examples)
    
    if len(all_examples) == 0:
        logging.error("No training examples found!")
        return None
    
    logging.info(f"Total training examples: {len(all_examples)}")
    random.shuffle(all_examples)
    
    # Create data loader
    train_dataloader = DataLoader(all_examples, shuffle=True, batch_size=batch_size)
    
    # Define loss function (MultipleNegativesRankingLoss for contrastive learning)
    train_loss = losses.MultipleNegativesRankingLoss(model)
    
    # Training loop
    logging.info(f"Starting training: epochs={epochs}, batch_size={batch_size}, lr={learning_rate}")
    
    try:
        for epoch in range(start_epoch, epochs):
            if shutdown_requested:
                break
            
            logging.info(f"\n{'='*60}")
            logging.info(f"Epoch {epoch + 1}/{epochs}")
            logging.info(f"{'='*60}")
            
            model.fit(
                train_objectives=[(train_dataloader, train_loss)],
                epochs=1,
                warmup_steps=100,
                optimizer_params={'lr': learning_rate},
                show_progress_bar=True
            )
            
            # Save checkpoint
            checkpoint_path = output_dir / f"checkpoint_epoch_{epoch + 1}"
            checkpoint_path.mkdir(parents=True, exist_ok=True)
            model.save(str(checkpoint_path))
            
            checkpoint_info = {
                'epoch': epoch + 1,
                'timestamp': str(pathlib.Path(__file__).stat().st_mtime)
            }
            with open(output_dir / "checkpoint.json", 'w') as f:
                json.dump(checkpoint_info, f, indent=2)
            
            if shutdown_requested:
                break
        
        # Save final model
        if not shutdown_requested:
            model_dir.mkdir(parents=True, exist_ok=True)
            model.save(str(model_dir))
            logging.info(f"✅ Final model saved to: {model_dir}")
    except Exception as e:
        logging.error(f"Training error: {e}", exc_info=True)
        raise
    
    return str(model_dir)

def evaluate_contrastive_model(model_path: str, config: Dict, gpu_id: int, output_dir: pathlib.Path):
    """Evaluate the trained contrastive model"""
    domains = config.get('domains', MTRAG_DOMAINS)
    use_data_splits = config.get('use_data_splits', True)
    top_k = config.get('top_k', 100)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_root = pathlib.Path(".")
    all_results = {}
    
    # Load model
    logging.info(f"Loading trained model: {model_path}")
    model = SentenceBERT(model_path, device=device)
    retriever = DenseRetrievalExactSearch(model, batch_size=128)
    
    for domain in domains:
        if shutdown_requested:
            break
        
        logging.info(f"\n{'='*60}")
        logging.info(f"Evaluating on domain: {domain}")
        logging.info(f"{'='*60}")
        
        # Load test data
        corpus_file = data_root / "corpora" / "passage_level" / f"{domain}.jsonl"
        
        if use_data_splits:
            query_file = data_root / "data_splits" / "retrieval_tasks" / domain / "test" / f"{domain}_questions.jsonl"
            qrels_file = data_root / "data_splits" / "retrieval_tasks" / domain / "test" / "qrels" / "dev.tsv"
        else:
            query_file = data_root / "human" / "retrieval_tasks" / domain / f"{domain}_questions.jsonl"
            qrels_file = data_root / "human" / "retrieval_tasks" / domain / "qrels" / "dev.tsv"
        
        if not query_file.exists() and use_data_splits:
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
            continue
        
        # Retrieve
        logging.info(f"Retrieving (top-{top_k})...")
        evaluator = EvaluateRetrieval(retriever, k_values=[top_k])
        results = evaluator.retrieve(corpus, queries)
        
        # Evaluate
        k_values = [1, 3, 5, 10]
        eval_evaluator = EvaluateRetrieval(None, k_values=k_values)
        ndcg, _map, recall, precision = eval_evaluator.evaluate(qrels, results, k_values)
        
        all_results[domain] = {
            "Recall@1": recall.get('Recall@1', 0),
            "Recall@3": recall.get('Recall@3', 0),
            "Recall@5": recall.get('Recall@5', 0),
            "Recall@10": recall.get('Recall@10', 0),
            "nDCG@1": ndcg.get('NDCG@1', 0),
            "nDCG@3": ndcg.get('NDCG@3', 0),
            "nDCG@5": ndcg.get('NDCG@5', 0),
            "nDCG@10": ndcg.get('NDCG@10', 0)
        }
        
        logging.info(f"Domain {domain} - Recall@10: {recall.get('Recall@10', 0):.4f}, nDCG@10: {ndcg.get('NDCG@10', 0):.4f}")
    
    # Calculate average
    if all_results:
        avg_results = {
            "Recall@1": np.mean([r["Recall@1"] for r in all_results.values()]),
            "Recall@3": np.mean([r["Recall@3"] for r in all_results.values()]),
            "Recall@5": np.mean([r["Recall@5"] for r in all_results.values()]),
            "Recall@10": np.mean([r["Recall@10"] for r in all_results.values()]),
            "nDCG@1": np.mean([r["nDCG@1"] for r in all_results.values()]),
            "nDCG@3": np.mean([r["nDCG@3"] for r in all_results.values()]),
            "nDCG@5": np.mean([r["nDCG@5"] for r in all_results.values()]),
            "nDCG@10": np.mean([r["nDCG@10"] for r in all_results.values()])
        }
        
        all_results["average"] = avg_results
        logging.info(f"\n{'='*60}")
        logging.info("Average Results:")
        logging.info(f"Recall@10: {avg_results['Recall@10']:.4f}")
        logging.info(f"nDCG@10: {avg_results['nDCG@10']:.4f}")
        logging.info(f"{'='*60}")
    
    # Save results
    results_file = output_dir / "results.json"
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    logging.info(f"✅ Results saved to: {results_file}")
    return str(results_file)

def main():
    parser = argparse.ArgumentParser(description='Tier 1: Contrastive Learning')
    parser.add_argument('--experiment_name', type=str, required=True)
    parser.add_argument('--gpu', type=int, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--resume', action='store_true', help='Resume from checkpoint')
    parser.add_argument('--no-resume', dest='resume', action='store_false', help='Do not resume')
    parser.set_defaults(resume=True)
    
    args = parser.parse_args()
    
    output_path = pathlib.Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    config = {
        'base_model': 'BAAI/bge-base-en-v1.5',
        'domains': ['clapnq', 'fiqa', 'govt', 'cloud'],
        'use_data_splits': True,
        'epochs': 3,
        'batch_size': 16,
        'learning_rate': 2e-5,
        'top_k': 100,
        'resume': args.resume
    }
    
    config_file = output_path / "config.json"
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    
    try:
        # Step 1: Train model
        logging.info("Step 1: Training contrastive model...")
        model_path = train_contrastive_model(config, args.gpu, output_path)
        
        if shutdown_requested:
            logging.info("Training interrupted. Model saved at checkpoint.")
            return
        
        # Step 2: Evaluate model
        logging.info("Step 2: Evaluating trained model...")
        results_file = evaluate_contrastive_model(model_path, config, args.gpu, output_path)
        
        logging.info(f"✅ Contrastive learning experiment completed")
    except Exception as e:
        logging.error(f"Contrastive learning experiment failed: {e}", exc_info=True)
        error_info = {
            'experiment': args.experiment_name,
            'status': 'failed',
            'error': str(e)
        }
        results_file = output_path / "results.json"
        with open(results_file, 'w') as f:
            json.dump(error_info, f, indent=2)
        raise

if __name__ == "__main__":
    main()
