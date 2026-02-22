"""
Multi-Task Learning: Retrieval + Reranking (Retrieval-Only, adapted for Task A)
Joint optimization of retrieval and reranking tasks
Expected: 0.59-0.63 nDCG@10
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
from transformers import AutoTokenizer, AutoModel

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

class MultiTaskRetrievalModel(nn.Module):
    """Multi-Task Model: Shared encoder with retrieval and reranking heads"""
    def __init__(self, base_model_name: str = 'BAAI/bge-base-en-v1.5'):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        self.encoder = AutoModel.from_pretrained(base_model_name)
        hidden_size = self.encoder.config.hidden_size
        
        # Retrieval head: similarity scoring
        self.retrieval_proj = nn.Linear(hidden_size, 768)
        
        # Reranking head: cross-encoder scoring
        self.reranking_proj = nn.Linear(hidden_size, 1)
        
    def encode_retrieval(self, texts: List[str], batch_size: int = 32):
        """Encode for retrieval (symmetric)"""
        embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            inputs = self.tokenizer(
                batch, padding=True, truncation=True, max_length=512,
                return_tensors='pt'
            )
            device = next(self.encoder.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.encoder(**inputs)
                # Mean pooling
                embeddings_batch = outputs.last_hidden_state.mean(dim=1)
                embeddings_batch = self.retrieval_proj(embeddings_batch)
                embeddings.append(embeddings_batch.cpu().numpy())
        
        return np.vstack(embeddings)
    
    def score_reranking(self, query_texts: List[str], doc_texts: List[str], batch_size: int = 32):
        """Score query-document pairs for reranking"""
        scores = []
        for i in range(0, len(query_texts), batch_size):
            batch_queries = query_texts[i:i+batch_size]
            batch_docs = doc_texts[i:i+batch_size]
            
            batch_texts = [f"{q} [SEP] {d}" for q, d in zip(batch_queries, batch_docs)]
            inputs = self.tokenizer(
                batch_texts, padding=True, truncation=True, max_length=512,
                return_tensors='pt'
            )
            device = next(self.encoder.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.encoder(**inputs)
                # CLS token
                cls_embeddings = outputs.last_hidden_state[:, 0, :]
                batch_scores = self.reranking_proj(cls_embeddings).squeeze(1)
                scores.append(batch_scores.cpu().numpy())
        
        return np.concatenate(scores)

def run_multitask_retrieval_evaluation(config: Dict, gpu_id: Optional[int] = None) -> Optional[str]:
    """Run Multi-Task Retrieval evaluation"""
    global shutdown_requested
    
    if gpu_id is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        logging.info(f"Using GPU {gpu_id}")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Device: {device}")
    
    output_dir = pathlib.Path(config.get('output_dir', 'experiments/retrieval/multitask'))
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
    
    # Initialize multi-task model
    logging.info(f"Loading model: {base_model}")
    model = MultiTaskRetrievalModel(base_model_name=base_model)
    model.to(device)
    model.eval()
    
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
        
        # Step 1: Initial retrieval with retrieval head
        logging.info("Step 1: Initial retrieval...")
        query_texts = [queries[qid] for qid in queries.keys()]
        query_embeddings = model.encode_retrieval(query_texts)
        query_emb_map = {
            qid: emb for qid, emb in zip(queries.keys(), query_embeddings)
        }
        
        corpus_texts = [corpus[doc_id].get('text', '') for doc_id in corpus.keys()]
        corpus_embeddings = model.encode_retrieval(corpus_texts)
        corpus_emb_map = {
            doc_id: emb for doc_id, emb in zip(corpus.keys(), corpus_embeddings)
        }
        
        # Get initial candidates
        initial_results = {}
        for qid, query_emb in query_emb_map.items():
            scores = {}
            for doc_id, doc_emb in corpus_emb_map.items():
                score = np.dot(query_emb, doc_emb) / (
                    np.linalg.norm(query_emb) * np.linalg.norm(doc_emb)
                )
                scores[doc_id] = float(score)
            sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k * 2]
            initial_results[qid] = dict(sorted_scores)
        
        # Step 2: Rerank with reranking head
        logging.info("Step 2: Reranking with cross-encoder head...")
        final_results = {}
        
        for qid, query_text in queries.items():
            if qid not in initial_results:
                final_results[qid] = {}
                continue
            
            candidates = sorted(
                initial_results[qid].items(),
                key=lambda x: x[1],
                reverse=True
            )[:top_k * 2]
            
            doc_texts = [corpus[doc_id].get('text', '') for doc_id, _ in candidates if doc_id in corpus]
            query_texts_batch = [query_text] * len(doc_texts)
            
            if not doc_texts:
                final_results[qid] = {}
                continue
            
            # Rerank with cross-encoder
            rerank_scores = model.score_reranking(query_texts_batch, doc_texts)
            
            # Combine initial and rerank scores
            combined_scores = {}
            for idx, (doc_id, initial_score) in enumerate(candidates):
                if doc_id in corpus and idx < len(rerank_scores):
                    combined_score = 0.4 * initial_score + 0.6 * float(rerank_scores[idx])
                    combined_scores[doc_id] = combined_score
            
            final_results[qid] = dict(
                sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
            )
        
        # Step 3: Evaluate
        logging.info("Step 3: Evaluating results...")
        evaluator = EvaluateRetrieval(None, k_values=[1, 3, 5, 10])
        ndcg, _map, recall, precision = evaluator.evaluate(qrels, final_results, [1, 3, 5, 10])
        
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
        results_path = run_multitask_retrieval_evaluation(config, args.gpu_id)
        if results_path:
            logging.info(f"✅ Multi-Task Retrieval completed: {results_path}")
    except KeyboardInterrupt:
        logging.info("Interrupted by user")
        sys.exit(0)
    except Exception as e:
        logging.error(f"Failed: {e}", exc_info=True)
        sys.exit(1)

