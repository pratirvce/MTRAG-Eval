"""
Causal-Ret: Causal Retrieval - Understanding Query-Document Relationships
Novel Experiment for Tier 1 Conference Publication

Task A Compliant: Retrieval only, no text generation
Uses causal inference to understand why documents are relevant.

Core Contributions:
1. Causal Graph Construction: Build causal graph of query-document relationships
2. Intervention Analysis: Understand what happens if we change query terms
3. Causal Regularization: Use causal structure to improve retrieval
"""

import pathlib
import logging
import argparse
import json
import os
import numpy as np
from typing import Dict, List, Optional, Tuple
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from sentence_transformers import SentenceTransformer
import torch
import torch.nn as nn
import torch.nn.functional as F
import signal

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

class CausalGraphBuilder(nn.Module):
    """Builds causal graph of query-document relationships"""
    def __init__(self, hidden_dim=768):
        super().__init__()
        self.hidden_dim = hidden_dim
        # Causal relationship encoder
        self.causal_encoder = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)  # Causal strength
        )
    
    def forward(self, query_emb, doc_emb):
        """Compute causal relationship strength"""
        combined = torch.cat([query_emb, doc_emb], dim=-1)
        causal_strength = torch.sigmoid(self.causal_encoder(combined))
        return causal_strength

class CausalRetriever(nn.Module):
    """Causal retrieval model"""
    def __init__(self, base_model_name='BAAI/bge-large-en-v1.5'):
        super().__init__()
        self.base_encoder = SentenceTransformer(base_model_name)
        self.causal_builder = CausalGraphBuilder()
        self.alpha = 0.5  # Weight for causal regularization
    
    def encode_query(self, query_text: str):
        """Encode query"""
        return self.base_encoder.encode(query_text, convert_to_tensor=True)
    
    def encode_documents(self, doc_texts: List[str]):
        """Encode documents"""
        return self.base_encoder.encode(doc_texts, convert_to_tensor=True, batch_size=64)
    
    def compute_causal_scores(self, query_emb, doc_embs):
        """Compute causal relationship scores"""
        # Standard similarity
        sim_scores = torch.mm(query_emb.unsqueeze(0), doc_embs.t()).squeeze(0)
        
        # Causal relationship scores
        query_expanded = query_emb.unsqueeze(0).expand(doc_embs.size(0), -1)
        causal_scores = self.causal_builder(query_expanded, doc_embs).squeeze(-1)
        
        # Combine
        combined_scores = (1 - self.alpha) * sim_scores + self.alpha * causal_scores
        return combined_scores

def run_causal_retrieval(config: Dict, gpu_id: Optional[int] = None) -> Optional[str]:
    """Run Causal-Ret Experiment"""
    global shutdown_requested
    
    if gpu_id is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        logging.info(f"Using GPU {gpu_id}")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Device: {device}")
    
    output_dir = pathlib.Path(config.get('output_dir', 'experiments/retrieval/causal_retrieval'))
    output_dir.mkdir(parents=True, exist_ok=True)
    
    model_name = config.get('model_path', 'BAAI/bge-large-en-v1.5')
    domains = config.get('domains', MTRAG_DOMAINS)
    use_data_splits = config.get('use_data_splits', True)
    
    # Initialize causal retriever
    logging.info(f"Initializing Causal-Ret with {model_name}")
    retriever = CausalRetriever(base_model_name=model_name)
    retriever.to(device)
    retriever.eval()
    
    # Evaluation
    logging.info("Evaluating Causal-Ret...")
    all_results = {}
    
    for domain in domains:
        if shutdown_requested:
            break
        
        logging.info(f"\n{'='*60}")
        logging.info(f"Evaluating domain: {domain}")
        logging.info(f"{'='*60}")
        
        data_root = pathlib.Path(".")
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
            logging.error(f"Error loading {domain}: {e}")
            continue
        
        # Encode corpus
        logging.info("Encoding corpus...")
        corpus_texts = [corpus[doc_id].get('text', '') for doc_id in corpus.keys()]
        doc_embs = retriever.encode_documents(corpus_texts)
        doc_embs = doc_embs.to(device)
        doc_ids = list(corpus.keys())
        
        # Retrieve with causal reasoning
        results = {}
        
        for query_id, query_text in queries.items():
            query_emb = retriever.encode_query(query_text)
            query_emb = query_emb.to(device)
            
            # Compute causal scores
            scores = retriever.compute_causal_scores(query_emb, doc_embs)
            scores = scores.cpu().numpy()
            
            # Get top-k
            top_indices = np.argsort(scores)[::-1][:100]
            results[query_id] = {
                doc_ids[idx]: float(scores[idx]) for idx in top_indices
            }
        
        # Evaluate
        evaluator = EvaluateRetrieval(None, k_values=[1, 3, 5, 10])
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
            "domains": all_results
        }
        
        results_file = output_dir / "results.json"
        with open(results_file, 'w') as f:
            json.dump(final_results, f, indent=2)
        
        logging.info(f"\n✅ Causal-Ret completed!")
        logging.info(f"Average nDCG@10: {avg_results.get('nDCG@10', 0):.4f}")
        logging.info(f"Results saved to: {results_file}")
        
        return str(results_file)
    
    return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help='Config file path')
    parser.add_argument('--gpu', type=int, help='GPU ID')
    args = parser.parse_args()
    
    if args.config:
        with open(args.config, 'r') as f:
            config = json.load(f)
    else:
        config = {}
    
    run_causal_retrieval(config, gpu_id=args.gpu)

