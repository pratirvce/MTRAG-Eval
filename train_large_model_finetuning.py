"""
Large Model Fine-Tuning (BGE-Large or Larger)
Fine-tune larger models on all domains with hard negatives
Expected: 0.60-0.65 nDCG@10
"""

import pathlib
import logging
import argparse
import json
import os
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
import sys
from datetime import datetime

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

def load_training_examples(domains: List[str], use_data_splits: bool = True) -> List[InputExample]:
    """Load training examples from all domains"""
    examples = []
    data_root = pathlib.Path(".")
    
    for domain in domains:
        if use_data_splits:
            train_dir = data_root / "data_splits" / "retrieval_tasks" / domain / "train"
            corpus_file = data_root / "corpora" / "passage_level" / f"{domain}.jsonl"  # Corpus is not in train dir
            query_file = train_dir / f"{domain}_questions.jsonl"
            qrels_file = train_dir / "qrels" / "dev.tsv"
        else:
            corpus_file = data_root / "corpora" / "passage_level" / f"{domain}.jsonl"
            query_file = data_root / "human" / "retrieval_tasks" / domain / f"{domain}_questions.jsonl"
            qrels_file = data_root / "human" / "retrieval_tasks" / domain / "qrels" / "dev.tsv"
        
        try:
            corpus, queries, qrels = GenericDataLoader(
                corpus_file=str(corpus_file),
                query_file=str(query_file),
                qrels_file=str(qrels_file)
            ).load_custom()
            
            # Create training examples
            for query_id, query_text in queries.items():
                if query_id in qrels:
                    for doc_id, score in qrels[query_id].items():
                        if score > 0 and doc_id in corpus:
                            doc_text = corpus[doc_id].get('text', '')
                            examples.append(InputExample(texts=[query_text, doc_text]))
            
            logging.info(f"Loaded {len(examples)} examples from {domain}")
        except Exception as e:
            logging.warning(f"Error loading {domain}: {e}")
            continue
    
    return examples

def run_large_model_finetuning(config: Dict, gpu_id: Optional[int] = None) -> Optional[str]:
    """Run Large Model Fine-Tuning"""
    global shutdown_requested
    
    if gpu_id is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        logging.info(f"Using GPU {gpu_id}")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Device: {device}")
    
    output_dir = pathlib.Path(config.get('output_dir', 'experiments/retrieval/large_model'))
    output_dir.mkdir(parents=True, exist_ok=True)
    
    model_name = config.get('model_path', 'BAAI/bge-large-en-v1.5')
    domains = config.get('domains', MTRAG_DOMAINS)
    epochs = config.get('epochs', 3)
    batch_size = config.get('batch_size', 8)  # Smaller batch for large models
    use_data_splits = config.get('use_data_splits', True)
    resume = config.get('resume', True)
    
    # Load checkpoint
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    model_path = checkpoint_dir / "model"
    
    if resume and model_path.exists():
        logging.info(f"Resuming from checkpoint: {model_path}")
        model = SentenceTransformer(str(model_path))
    else:
        logging.info(f"Loading base model: {model_name}")
        model = SentenceTransformer(model_name)
    
    # Load training data
    logging.info("Loading training examples...")
    train_examples = load_training_examples(domains, use_data_splits)
    
    if not train_examples:
        logging.error("No training examples found!")
        return None
    
    logging.info(f"Total training examples: {len(train_examples)}")
    
    # Create data loader
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=batch_size)
    
    # Define loss
    train_loss = losses.MultipleNegativesRankingLoss(model)
    
    # Training
    logging.info(f"Starting training for {epochs} epochs...")
    warmup_steps = int(len(train_dataloader) * 0.1)
    
    try:
        model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            epochs=epochs,
            warmup_steps=warmup_steps,
            output_path=str(model_path),
            show_progress_bar=True,
            checkpoint_path=str(checkpoint_dir) if resume else None,
            checkpoint_save_steps=len(train_dataloader)
        )
    except KeyboardInterrupt:
        logging.info("Training interrupted, saving checkpoint...")
        model.save(str(model_path))
    
    # Evaluation
    logging.info("Evaluating on test set...")
    all_results = {}
    
    for domain in domains:
        if shutdown_requested:
            break
        
        logging.info(f"\n{'='*60}")
        logging.info(f"Evaluating domain: {domain}")
        logging.info(f"{'='*60}")
        
        data_root = pathlib.Path(".")
        corpus_file = data_root / "corpora" / "passage_level" / f"{domain}.jsonl"
        query_file = data_root / "human" / "retrieval_tasks" / domain / f"{domain}_questions.jsonl"
        qrels_file = data_root / "human" / "retrieval_tasks" / domain / "qrels" / "dev.tsv"
        
        try:
            corpus, queries, qrels = GenericDataLoader(
                corpus_file=str(corpus_file),
                query_file=str(query_file),
                qrels_file=str(qrels_file)
            ).load_custom()
        except Exception as e:
            logging.error(f"Error loading {domain}: {e}")
            continue
        
        # Create retriever
        retriever_model = SentenceBERT(str(model_path), device=device)
        retriever = DenseRetrievalExactSearch(retriever_model, batch_size=64)
        
        # Retrieve
        evaluator = EvaluateRetrieval(retriever, k_values=[1, 3, 5, 10])
        results = evaluator.retrieve(corpus, queries)
        
        # Evaluate
        ndcg, _map, recall, precision = evaluator.evaluate(qrels, results, [1, 3, 5, 10])
        
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
        
        logging.info(f"Domain {domain} - nDCG@10: {ndcg.get('NDCG@10', 0):.4f}")
    
    # Save results
    if all_results:
        avg_results = {
            metric: np.mean([res.get(metric, 0) for res in all_results.values()])
            for metric in list(all_results.values())[0].keys()
        }
        
        final_results = {
            "average": avg_results,
            "domains": all_results,
            "model": model_name
        }
        
        results_file = output_dir / "results.json"
        with open(results_file, 'w') as f:
            json.dump(final_results, f, indent=2)
        
        logging.info(f"\n{'='*60}")
        logging.info("Average Results:")
        for metric, value in avg_results.items():
            logging.info(f"  {metric}: {value:.4f}")
        logging.info(f"{'='*60}\n")
        
        return str(output_dir)
    
    return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--gpu_id', type=int)
    parser.add_argument('--resume', action='store_true', default=True)
    parser.add_argument('--no-resume', dest='resume', action='store_false')
    
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    config['resume'] = args.resume
    
    try:
        results_path = run_large_model_finetuning(config, args.gpu_id)
        if results_path:
            logging.info(f"✅ Large Model Fine-Tuning completed: {results_path}")
    except KeyboardInterrupt:
        logging.info("Interrupted by user")
        sys.exit(0)
    except Exception as e:
        logging.error(f"Failed: {e}", exc_info=True)
        sys.exit(1)

