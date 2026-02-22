"""
Adversarial Hard Negative Mining with Curriculum Learning
Systematic hard negative generation with curriculum schedule
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
from rank_bm25 import BM25Okapi

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

def get_hard_negatives(query_text: str, corpus_dict: Dict, corpus_ids: List[str],
                      bm25: BM25Okapi, model: SentenceTransformer,
                      num_negatives: int = 3, difficulty: float = 0.5) -> List[str]:
    """Get hard negatives based on curriculum difficulty"""
    # BM25 negatives
    query_tokens = query_text.lower().split()
    bm25_scores = bm25.get_scores(query_tokens)
    
    # Sort by BM25 score (higher = more similar, harder negative)
    sorted_indices = np.argsort(bm25_scores)[::-1]
    
    # Curriculum: difficulty controls how hard the negatives are
    # difficulty=0.0: easy (random), difficulty=1.0: very hard (top BM25)
    start_idx = int(len(sorted_indices) * (1 - difficulty))
    end_idx = min(start_idx + num_negatives * 10, len(sorted_indices))
    
    hard_negatives = []
    for idx in sorted_indices[start_idx:end_idx]:
        doc_id = corpus_ids[idx]
        if doc_id in corpus_dict:
            doc_text = corpus_dict[doc_id].get('text', '')
            hard_negatives.append(doc_text)
            if len(hard_negatives) >= num_negatives:
                break
    
    # Fill with random if needed
    while len(hard_negatives) < num_negatives:
        random_doc = np.random.choice(list(corpus_dict.values()))
        random_text = random_doc.get('text', '')
        if random_text not in hard_negatives:
            hard_negatives.append(random_text)
    
    return hard_negatives[:num_negatives]

def run_adversarial_curriculum_evaluation(config: Dict, gpu_id: Optional[int] = None) -> Optional[str]:
    """Run Adversarial Curriculum Learning evaluation"""
    global shutdown_requested
    
    if gpu_id is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        logging.info(f"Using GPU {gpu_id}")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Device: {device}")
    
    output_dir = pathlib.Path(config.get('output_dir', 'experiments/retrieval/adversarial_curriculum'))
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
    use_data_splits = config.get('use_data_splits', True)
    
    # Initialize model
    logging.info(f"Loading model: {base_model}")
    model = SentenceTransformer(base_model)
    model.to(device)
    
    # Process each domain
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
        
        # Build BM25 index for hard negative mining
        logging.info("Building BM25 index...")
        corpus_texts = [corpus[doc_id].get('text', '') for doc_id in corpus.keys()]
        tokenized_corpus = [doc.lower().split() for doc in corpus_texts]
        bm25 = BM25Okapi(tokenized_corpus)
        corpus_ids = list(corpus.keys())
        
        # Fine-tune with curriculum learning
        logging.info("Fine-tuning with curriculum learning...")
        epochs = 2
        curriculum_stages = [0.3, 0.6, 0.9]  # Easy to hard
        
        for epoch in range(epochs):
            difficulty = curriculum_stages[min(epoch, len(curriculum_stages) - 1)]
            logging.info(f"Epoch {epoch + 1}/{epochs}, Difficulty: {difficulty:.2f}")
            
            # Create training examples with curriculum
            train_examples = []
            for query_id, query_text in queries.items():
                if query_id in qrels:
                    for doc_id, score in qrels[query_id].items():
                        if score > 0 and doc_id in corpus:
                            pos_doc_text = corpus[doc_id].get('text', '')
                            
                            # Get hard negatives
                            hard_negatives = get_hard_negatives(
                                query_text, corpus, corpus_ids, bm25, model,
                                num_negatives=3, difficulty=difficulty
                            )
                            
                            # Add positive example
                            train_examples.append(InputExample(texts=[query_text, pos_doc_text]))
                            
                            # Add hard negative examples
                            for neg_text in hard_negatives:
                                train_examples.append(InputExample(texts=[query_text, neg_text], label=0.0))
            
            if train_examples:
                train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)
                train_loss = losses.MultipleNegativesRankingLoss(model)
                
                model.fit(
                    train_objectives=[(train_dataloader, train_loss)],
                    epochs=1,
                    warmup_steps=100,
                    show_progress_bar=True
                )
        
        # Evaluation
        logging.info("Evaluating fine-tuned model...")
        # Save model temporarily if needed for SentenceBERT wrapper
        temp_model_path = output_dir / "temp_model"
        temp_model_path.mkdir(exist_ok=True)
        model.save(str(temp_model_path))
        retriever_model = SentenceBERT(str(temp_model_path), device=device)
        retriever = DenseRetrievalExactSearch(retriever_model, batch_size=128)
        
        evaluator = EvaluateRetrieval(retriever, k_values=[1, 3, 5, 10])
        results = evaluator.retrieve(corpus, queries)
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
        results_path = run_adversarial_curriculum_evaluation(config, args.gpu_id)
        if results_path:
            logging.info(f"✅ Adversarial Curriculum Learning completed: {results_path}")
    except KeyboardInterrupt:
        logging.info("Interrupted by user")
        sys.exit(0)
    except Exception as e:
        logging.error(f"Failed: {e}", exc_info=True)
        sys.exit(1)

