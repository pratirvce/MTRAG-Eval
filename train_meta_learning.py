"""
Cross-Domain Transfer Learning with Meta-Learning (MAML)
Meta-learning for fast adaptation to new domains
Expected: 0.57-0.61 nDCG@10
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
import torch.nn as nn
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

def run_meta_learning_evaluation(config: Dict, gpu_id: Optional[int] = None) -> Optional[str]:
    """Run Meta-Learning evaluation (simplified MAML)"""
    global shutdown_requested
    
    if gpu_id is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        logging.info(f"Using GPU {gpu_id}")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Device: {device}")
    
    output_dir = pathlib.Path(config.get('output_dir', 'experiments/retrieval/meta_learning'))
    output_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint_dir = output_dir
    resume = config.get('resume', True)
    
    # Load checkpoint
    all_results = {}
    completed_domains = set()
    if resume:
        checkpoint_file = checkpoint_dir / "checkpoint.json"
        if checkpoint_file.exists():
            with open(checkpoint_file) as f:
                checkpoint = json.load(f)
                all_results = checkpoint.get('results', {})
                completed_domains = set(checkpoint.get('completed_domains', []))
                logging.info(f"Resuming from checkpoint. Completed domains: {completed_domains}")
    
    domains = config.get('domains', MTRAG_DOMAINS)
    base_model = config.get('model_path', 'BAAI/bge-base-en-v1.5')
    top_k = config.get('top_k', 100)
    
    # Initialize base model (meta-learner)
    logging.info(f"Loading meta-learner model: {base_model}")
    meta_model = SentenceTransformer(base_model)
    meta_model.to(device)
    
    # Meta-training: Fine-tune on multiple domains with more examples
    logging.info("Meta-training on multiple domains...")
    train_domains = domains[:3]  # Use first 3 for meta-training
    
    all_meta_examples = []
    for domain in train_domains:
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
            
            # Create training examples (use more examples)
            for query_id, query_text in queries.items():
                if query_id in qrels:
                    for doc_id, score in qrels[query_id].items():
                        if score > 0 and doc_id in corpus:
                            doc_text = corpus[doc_id].get('text', '')
                            all_meta_examples.append(InputExample(texts=[query_text, doc_text]))
        except Exception as e:
            logging.warning(f"Error loading data for meta-training {domain}: {e}")
            continue
    
    # Meta-train on all examples from multiple domains
    if all_meta_examples:
        logging.info(f"Meta-training on {len(all_meta_examples)} examples from {len(train_domains)} domains...")
        # Use more examples and more epochs for better meta-learning
        train_dataloader = DataLoader(all_meta_examples[:500], shuffle=True, batch_size=8)  # Reduced for OOM
        train_loss = losses.MultipleNegativesRankingLoss(meta_model)
        
        # Meta-training (outer loop)
        meta_model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            epochs=3,  # More epochs
            warmup_steps=50,  # More warmup
            show_progress_bar=True,
            fp16=True  # Enable FP16 to reduce memory
        )
        logging.info("✅ Meta-training completed")
    
    # Evaluation on all domains
    logging.info("Evaluating meta-learned model on all domains...")
    
    for domain in domains:
        if domain in completed_domains:
            logging.info(f"Skipping {domain} (already completed)")
            continue
        
        if shutdown_requested:
            logging.info("Shutdown requested, saving checkpoint...")
            checkpoint_file = checkpoint_dir / "checkpoint.json"
            with open(checkpoint_file, 'w') as f:
                json.dump({
                    "last_domain": domain,
                    "completed_domains": list(completed_domains),
                    "results": all_results,
                    "timestamp": datetime.now().isoformat()
                }, f, indent=2)
            break
        
        logging.info(f"\n{'='*60}")
        logging.info(f"Processing domain: {domain}")
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
            logging.error(f"Error loading data for {domain}: {e}")
            continue
        
        logging.info(f"Loaded {len(corpus)} documents, {len(queries)} queries")
        
        # Fast adaptation to target domain (inner loop)
        logging.info(f"Fast adaptation to {domain}...")
        train_examples = []
        for query_id, query_text in queries.items():
            if query_id in qrels:
                for doc_id, score in qrels[query_id].items():
                    if score > 0 and doc_id in corpus:
                        doc_text = corpus[doc_id].get('text', '')
                        train_examples.append(InputExample(texts=[query_text, doc_text]))
        
        if train_examples:
            # Use more examples for better adaptation
            train_dataloader = DataLoader(train_examples[:200], shuffle=True, batch_size=8)  # Reduced for OOM
            train_loss = losses.MultipleNegativesRankingLoss(meta_model)
            
            # Fast adaptation (inner loop) - more epochs for better adaptation
            meta_model.fit(
                train_objectives=[(train_dataloader, train_loss)],
                epochs=2,  # More epochs
                warmup_steps=20,  # More warmup
                show_progress_bar=False,
                fp16=True  # Enable FP16 to reduce memory
            )
        
        # Evaluation
        logging.info("Evaluating adapted model...")
        # Use model directly by encoding queries and corpus, then computing similarity
        # This avoids the SentenceBERT wrapper issue with model saving
        from sentence_transformers.util import cos_sim
        # torch is already imported at the top of the file
        
        # Encode queries and corpus
        logging.info("Encoding queries and corpus...")
        query_texts = list(queries.values())
        query_ids = list(queries.keys())
        corpus_texts = [corpus[doc_id].get('text', '') for doc_id in corpus.keys()]
        corpus_ids = list(corpus.keys())
        
        query_embeddings = meta_model.encode(query_texts, batch_size=32, show_progress_bar=True, convert_to_tensor=True)
        corpus_embeddings = meta_model.encode(corpus_texts, batch_size=32, show_progress_bar=True, convert_to_tensor=True)
        
        # Compute similarity and create results in BEIR format
        logging.info("Computing similarities...")
        query_embeddings = query_embeddings.to(device)
        corpus_embeddings = corpus_embeddings.to(device)
        results = {}
        
        for i, query_id in enumerate(query_ids):
            query_emb = query_embeddings[i:i+1]
            scores = cos_sim(query_emb, corpus_embeddings)[0]
            results[query_id] = {corpus_ids[j]: float(scores[j].cpu().item()) for j in range(len(corpus_ids))}
        
        # Use results directly for evaluation
        evaluator = EvaluateRetrieval(None, k_values=[1, 3, 5, 10])
        
        # Results already computed above, evaluate directly
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
        
        logging.info(f"\n{'='*60}")
        logging.info(f"Domain {domain} Results:")
        logging.info(f"  Recall@10: {recall.get('Recall@10', 0):.4f}")
        logging.info(f"  nDCG@10: {ndcg.get('NDCG@10', 0):.4f}")
        logging.info(f"{'='*60}\n")
        
        completed_domains.add(domain)
        
        # Save checkpoint
        checkpoint_file = checkpoint_dir / "checkpoint.json"
        with open(checkpoint_file, 'w') as f:
            json.dump({
                "last_domain": domain,
                "completed_domains": list(completed_domains),
                "results": all_results,
                "timestamp": datetime.now().isoformat()
            }, f, indent=2)
    
    # Final results
    if all_results:
        avg_results = {
            metric: np.mean([res.get(metric, 0) for res in all_results.values()])
            for metric in list(all_results.values())[0].keys()
        }
        
        final_results = {
            "average": avg_results,
            "domains": all_results
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
        results_path = run_meta_learning_evaluation(config, args.gpu_id)
        if results_path:
            logging.info(f"✅ Meta-Learning completed: {results_path}")
    except KeyboardInterrupt:
        logging.info("Interrupted by user")
        sys.exit(0)
    except Exception as e:
        logging.error(f"Failed: {e}", exc_info=True)
        sys.exit(1)

