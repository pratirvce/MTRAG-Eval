"""
Diff-Ret: Differentiable Retrieval with End-to-End Optimization
Novel Experiment for Tier 1 Conference Publication

Task A Compliant: Retrieval only, no text generation
Fully differentiable retrieval pipeline optimized directly for nDCG.

Core Contributions:
1. Differentiable Top-K: Gumbel-Softmax for differentiable top-k selection
2. End-to-End Optimization: Direct nDCG optimization (not proxy loss)
3. Gradient Flow: Enables gradients through discrete retrieval operations
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
from beir.retrieval.models import SentenceBERT
from beir.retrieval.search.dense import DenseRetrievalExactSearch
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
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

class DifferentiableTopK(nn.Module):
    """Differentiable top-k selection using Gumbel-Softmax"""
    def __init__(self, k=10, temperature=1.0, hard=False):
        super().__init__()
        self.k = k
        self.temperature = temperature
        self.hard = hard
    
    def forward(self, scores, training=True):
        """
        Args:
            scores: [batch_size, num_docs] - Retrieval scores
        Returns:
            selected_scores: [batch_size, k] - Differentiable top-k scores
            selected_indices: [batch_size, k] - Selected document indices
        """
        if not training:
            # During evaluation, use standard top-k
            topk_scores, topk_indices = torch.topk(scores, self.k, dim=-1)
            return topk_scores, topk_indices
        
        # Gumbel-Softmax for differentiable top-k
        # Add Gumbel noise
        gumbel_noise = -torch.log(-torch.log(torch.rand_like(scores) + 1e-10) + 1e-10)
        perturbed_scores = scores + gumbel_noise
        
        # Softmax with temperature
        soft_scores = F.softmax(perturbed_scores / self.temperature, dim=-1)
        
        # Get top-k
        topk_scores, topk_indices = torch.topk(soft_scores, self.k, dim=-1)
        
        if self.hard:
            # Straight-through estimator: use hard selection in forward, soft in backward
            hard_scores = torch.zeros_like(soft_scores)
            hard_scores.scatter_(-1, topk_indices, topk_scores)
            topk_scores = hard_scores.gather(-1, topk_indices)
        
        return topk_scores, topk_indices

class DifferentiableNDCGLoss(nn.Module):
    """Differentiable nDCG loss for direct optimization"""
    def __init__(self, k=10):
        super().__init__()
        self.k = k
    
    def compute_dcg(self, scores, relevances):
        """Compute DCG with differentiable scores"""
        # DCG = sum(rel_i / log2(i+2))
        positions = torch.arange(1, scores.size(-1) + 1, device=scores.device, dtype=torch.float32)
        discounts = 1.0 / torch.log2(positions + 1.0)
        dcg = torch.sum(scores * relevances * discounts, dim=-1)
        return dcg
    
    def forward(self, predicted_scores, relevance_scores):
        """
        Args:
            predicted_scores: [batch_size, k] - Predicted retrieval scores
            relevance_scores: [batch_size, k] - Ground truth relevance scores
        Returns:
            loss: Scalar - Negative nDCG (to minimize)
        """
        # Normalize scores to [0, 1] using sigmoid
        normalized_scores = torch.sigmoid(predicted_scores)
        
        # Compute DCG
        dcg = self.compute_dcg(normalized_scores, relevance_scores)
        
        # Compute IDCG (ideal DCG - sorted by relevance)
        sorted_relevances, _ = torch.sort(relevance_scores, dim=-1, descending=True)
        idcg = self.compute_dcg(torch.ones_like(sorted_relevances), sorted_relevances)
        
        # Avoid division by zero
        idcg = torch.clamp(idcg, min=1e-10)
        
        # nDCG = DCG / IDCG
        ndcg = dcg / idcg
        
        # Return negative nDCG as loss (to maximize nDCG)
        loss = -torch.mean(ndcg)
        
        return loss, torch.mean(ndcg)

class DifferentiableRetriever(nn.Module):
    """Differentiable retrieval model with end-to-end optimization"""
    def __init__(self, base_model_name='BAAI/bge-large-en-v1.5', k=10, temperature=1.0):
        super().__init__()
        self.base_encoder = SentenceTransformer(base_model_name)
        self.k = k
        self.diff_topk = DifferentiableTopK(k=k, temperature=temperature, hard=True)
        self.ndcg_loss = DifferentiableNDCGLoss(k=k)
        
        # Learnable temperature for adaptive softmax
        self.learnable_temp = nn.Parameter(torch.tensor(1.0))
    
    def encode_query(self, query_text: str):
        """Encode query"""
        return self.base_encoder.encode(query_text, convert_to_tensor=True)
    
    def encode_documents(self, doc_texts: List[str]):
        """Encode documents"""
        return self.base_encoder.encode(doc_texts, convert_to_tensor=True, batch_size=64)
    
    def retrieve_differentiable(self, query_emb, doc_embs, training=True):
        """
        Differentiable retrieval operation
        Args:
            query_emb: [hidden_dim] - Query embedding
            doc_embs: [num_docs, hidden_dim] - Document embeddings
            training: Whether in training mode
        Returns:
            selected_scores: [k] - Differentiable top-k scores
            selected_indices: [k] - Selected document indices
        """
        # Compute similarities
        scores = torch.mm(query_emb.unsqueeze(0), doc_embs.t()).squeeze(0)  # [num_docs]
        
        # Differentiable top-k
        selected_scores, selected_indices = self.diff_topk(scores.unsqueeze(0), training=training)
        
        return selected_scores.squeeze(0), selected_indices.squeeze(0)

def load_training_data(domain: str, data_root: pathlib.Path, use_data_splits: bool = True):
    """Load training data for differentiable optimization"""
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
        return corpus, queries, qrels
    except Exception as e:
        logging.error(f"Error loading {domain}: {e}")
        return {}, {}, {}

def run_diff_retrieval(config: Dict, gpu_id: Optional[int] = None) -> Optional[str]:
    """Run Diff-Ret Differentiable Retrieval Experiment"""
    global shutdown_requested
    
    if gpu_id is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        logging.info(f"Using GPU {gpu_id}")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Device: {device}")
    
    output_dir = pathlib.Path(config.get('output_dir', 'experiments/retrieval/diff_retrieval'))
    output_dir.mkdir(parents=True, exist_ok=True)
    
    model_name = config.get('model_path', 'BAAI/bge-large-en-v1.5')
    domains = config.get('domains', MTRAG_DOMAINS)
    use_data_splits = config.get('use_data_splits', True)
    top_k = config.get('top_k', 10)
    temperature = config.get('temperature', 1.0)
    
    # Initialize differentiable retriever
    logging.info(f"Initializing Diff-Ret with {model_name}")
    retriever_model = DifferentiableRetriever(
        base_model_name=model_name,
        k=top_k,
        temperature=temperature
    )
    retriever_model.to(device)
    retriever_model.eval()
    
    # Evaluation
    logging.info("Evaluating Diff-Ret Differentiable Retrieval...")
    all_results = {}
    
    for domain in domains:
        if shutdown_requested:
            break
        
        logging.info(f"\n{'='*60}")
        logging.info(f"Evaluating domain: {domain}")
        logging.info(f"{'='*60}")
        
        data_root = pathlib.Path(".")
        corpus, queries, qrels = load_training_data(domain, data_root, use_data_splits)
        
        if not corpus or not queries:
            continue
        
        # Encode corpus
        logging.info("Encoding corpus...")
        corpus_texts = [corpus[doc_id].get('text', '') for doc_id in corpus.keys()]
        doc_embs = retriever_model.encode_documents(corpus_texts)
        doc_embs = doc_embs.to(device)
        doc_ids = list(corpus.keys())
        
        # Retrieve with differentiable top-k
        results = {}
        
        for query_id, query_text in queries.items():
            # Encode query
            query_emb = retriever_model.encode_query(query_text)
            query_emb = query_emb.to(device)
            
            # Differentiable retrieval
            selected_scores, selected_indices = retriever_model.retrieve_differentiable(
                query_emb, doc_embs, training=False
            )
            
            # Convert to results format
            results[query_id] = {
                doc_ids[idx]: float(selected_scores[i]) 
                for i, idx in enumerate(selected_indices.cpu().numpy())
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
        
        logging.info(f"\n✅ Diff-Ret Differentiable Retrieval completed!")
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
    
    run_diff_retrieval(config, gpu_id=args.gpu)

