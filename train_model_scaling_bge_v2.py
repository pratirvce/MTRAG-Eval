"""
Model Scaling Improvements for Task A
- Upgrade to BGE-v2-large or E5-large-v2
- Longer training (5-10 epochs)
- Domain-specific fine-tuning
- Larger batch sizes with gradient accumulation

Expected: 0.61-0.66 nDCG@10 (from baseline 0.51)
"""

import pathlib
import logging
import argparse
import json
import os
import numpy as np
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

def load_training_pairs(domain: str, data_root: pathlib.Path, use_data_splits: bool = True) -> list:
    """Load training pairs"""
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
    for qid, rel_docs in qrels.items():
        if qid not in queries:
            continue
        query_text = queries[qid]
        
        for doc_id, rel_score in rel_docs.items():
            if rel_score > 0 and doc_id in corpus:
                examples.append(InputExample(texts=[query_text, corpus[doc_id]], label=float(rel_score)))
    
    return examples

def train_domain_specific_model(
    domain: str,
    data_root: pathlib.Path,
    output_dir: pathlib.Path,
    gpu_id: int = 0,
    model_name: str = "BAAI/bge-large-en-v1.5",
    epochs: int = 5,
    batch_size: int = 16,
    gradient_accumulation_steps: int = 2,
    learning_rate: float = 2e-5,
    warmup_steps: int = 100
):
    """Train domain-specific model with scaling improvements"""
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    device = torch.device(f'cuda:0' if torch.cuda.is_available() else 'cpu')
    
    logging.info(f"🚀 Training domain-specific model for {domain}")
    logging.info(f"Model: {model_name}, Epochs: {epochs}, Batch Size: {batch_size}")
    logging.info(f"Gradient Accumulation Steps: {gradient_accumulation_steps}")
    
    # Load training data for this domain
    examples = load_training_pairs(domain, data_root, use_data_splits=True)
    
    if not examples:
        # Fallback to all domains
        logging.warning(f"No training data for {domain}, using all domains")
        examples = []
        for d in MTRAG_DOMAINS:
            examples.extend(load_training_pairs(d, data_root, use_data_splits=True))
    
    logging.info(f"Total training examples: {len(examples)}")
    
    if not examples:
        logging.error("No training examples found")
        return None
    
    # Initialize model
    model = SentenceTransformer(model_name)
    model.to(device)
    
    # DataLoader with larger effective batch size
    train_dataloader = DataLoader(examples, shuffle=True, batch_size=batch_size)
    
    # Loss function
    train_loss = losses.MultipleNegativesRankingLoss(model=model)
    
    # Training with gradient accumulation
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=epochs,
        warmup_steps=warmup_steps,
        output_path=str(output_dir / f"model_{domain}"),
        show_progress_bar=True,
        checkpoint_save_steps=1000,
        optimizer_params={'lr': learning_rate},
        use_amp=True  # Mixed precision for memory efficiency
    )
    
    logging.info(f"✅ Model saved to {output_dir / f'model_{domain}'}")
    return model

def evaluate_model(
    model_path: str,
    domain: str,
    data_root: pathlib.Path,
    use_data_splits: bool = True
) -> dict:
    """Evaluate trained model"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
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
        return None
    
    # Create retriever
    beir_model = SentenceBERT(model_path, device=device.type)
    retriever = DenseRetrievalExactSearch(beir_model, batch_size=128)
    evaluator = EvaluateRetrieval(retriever, k_values=[1, 3, 5, 10])
    
    # Retrieve and evaluate
    results = evaluator.retrieve(corpus, queries)
    ndcg, _map, recall, precision = evaluator.evaluate(qrels, results, [1, 3, 5, 10])
    
    return {
        "domain": domain,
        "Recall@1": recall['Recall@1'],
        "Recall@3": recall['Recall@3'],
        "Recall@5": recall['Recall@5'],
        "Recall@10": recall['Recall@10'],
        "nDCG@1": ndcg['NDCG@1'],
        "nDCG@3": ndcg['NDCG@3'],
        "nDCG@5": ndcg['NDCG@5'],
        "nDCG@10": ndcg['NDCG@10'],
    }

def main():
    parser = argparse.ArgumentParser(description="Model Scaling Improvements")
    parser.add_argument("--experiment_name", type=str, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--output_dir", type=str, default="experiments/retrieval")
    parser.add_argument("--model_name", type=str, default="BAAI/bge-large-en-v1.5")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--domain_specific", action="store_true", help="Train separate models per domain")
    args = parser.parse_args()
    
    output_dir = pathlib.Path(args.output_dir) / args.experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)
    data_root = pathlib.Path(".")
    
    all_results = {}
    
    if args.domain_specific:
        # Train separate models per domain
        for domain in MTRAG_DOMAINS:
            logging.info(f"Training domain-specific model for {domain}")
            model = train_domain_specific_model(
                domain=domain,
                data_root=data_root,
                output_dir=output_dir,
                gpu_id=args.gpu,
                model_name=args.model_name,
                epochs=args.epochs,
                batch_size=args.batch_size,
                gradient_accumulation_steps=args.gradient_accumulation_steps,
                learning_rate=args.learning_rate
            )
            
            if model:
                # Evaluate on this domain
                results = evaluate_model(
                    str(output_dir / f"model_{domain}"),
                    domain,
                    data_root,
                    use_data_splits=True
                )
                if results:
                    all_results[domain] = results
    else:
        # Train single model on all domains
        logging.info("Training single model on all domains")
        all_examples = []
        for domain in MTRAG_DOMAINS:
            examples = load_training_pairs(domain, data_root, use_data_splits=True)
            all_examples.extend(examples)
        
        if all_examples:
            model = SentenceTransformer(args.model_name)
            train_dataloader = DataLoader(all_examples, shuffle=True, batch_size=args.batch_size)
            train_loss = losses.MultipleNegativesRankingLoss(model=model)
            
            model.fit(
                train_objectives=[(train_dataloader, train_loss)],
                epochs=args.epochs,
                warmup_steps=100,
                output_path=str(output_dir / "model"),
                show_progress_bar=True,
                checkpoint_save_steps=1000,
                optimizer_params={'lr': args.learning_rate},
                use_amp=True
            )
            
            # Evaluate on all domains
            for domain in MTRAG_DOMAINS:
                results = evaluate_model(
                    str(output_dir / "model"),
                    domain,
                    data_root,
                    use_data_splits=True
                )
                if results:
                    all_results[domain] = results
    
    # Calculate averages
    if all_results:
        avg_results = {
            "Recall@1": np.mean([r["Recall@1"] for r in all_results.values()]),
            "Recall@3": np.mean([r["Recall@3"] for r in all_results.values()]),
            "Recall@5": np.mean([r["Recall@5"] for r in all_results.values()]),
            "Recall@10": np.mean([r["Recall@10"] for r in all_results.values()]),
            "nDCG@1": np.mean([r["nDCG@1"] for r in all_results.values()]),
            "nDCG@3": np.mean([r["nDCG@3"] for r in all_results.values()]),
            "nDCG@5": np.mean([r["nDCG@5"] for r in all_results.values()]),
            "nDCG@10": np.mean([r["nDCG@10"] for r in all_results.values()]),
        }
        all_results["average"] = avg_results
        
        # Save results
        results_file = output_dir / "results.json"
        with open(results_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        logging.info(f"✅ Final average: nDCG@10={avg_results['nDCG@10']:.4f}")
        logging.info(f"✅ Results saved to {results_file}")

if __name__ == "__main__":
    main()

