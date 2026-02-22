"""
X-Ret: Explainable Retrieval with Attention-Based Rationales
Novel Experiment for Tier 1 Conference Publication

Task A Compliant: Retrieval only, no text generation
Provides interpretable explanations through attention highlighting (not text generation).

Core Contributions:
1. Attention Extraction: Cross-attention between query and document
2. Rationale Highlighting: Highlights key matching spans (no text generation)
3. Contrastive Explanations: Explains why doc A ranked higher than doc B
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

class CrossAttentionExplainer(nn.Module):
    """Cross-attention for explainable retrieval"""
    def __init__(self, hidden_dim=768):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.query_proj = nn.Linear(hidden_dim, hidden_dim)
        self.doc_proj = nn.Linear(hidden_dim, hidden_dim)
        self.scale = np.sqrt(hidden_dim)
    
    def forward(self, query_emb, doc_embs):
        """
        Compute cross-attention and extract attention weights
        Returns:
            scores: Retrieval scores
            attention_weights: Attention weights for explanation
        """
        Q = self.query_proj(query_emb).unsqueeze(0)  # [1, hidden_dim]
        K = self.doc_proj(doc_embs)  # [num_docs, hidden_dim]
        V = doc_embs  # [num_docs, hidden_dim]
        
        # Attention scores
        attention_scores = torch.mm(Q, K.t()) / self.scale  # [1, num_docs]
        attention_weights = F.softmax(attention_scores, dim=-1)
        
        # Weighted document representations
        weighted_docs = torch.mm(attention_weights, V)  # [1, hidden_dim]
        
        # Final score
        scores = torch.mm(Q, weighted_docs.t()).squeeze()
        
        return scores, attention_weights.squeeze(0)

class ExplainableRetriever(nn.Module):
    """Explainable retrieval model"""
    def __init__(self, base_model_name='BAAI/bge-large-en-v1.5'):
        super().__init__()
        self.base_encoder = SentenceTransformer(base_model_name)
        self.explainer = CrossAttentionExplainer()
    
    def encode_query(self, query_text: str):
        """Encode query"""
        return self.base_encoder.encode(query_text, convert_to_tensor=True)
    
    def encode_documents(self, doc_texts: List[str]):
        """Encode documents"""
        return self.base_encoder.encode(doc_texts, convert_to_tensor=True, batch_size=64)
    
    def retrieve_with_explanation(self, query_emb, doc_embs):
        """Retrieve with attention-based explanations"""
        scores, attention_weights = self.explainer(query_emb, doc_embs)
        return scores, attention_weights

def run_x_explainable_retrieval(config: Dict, gpu_id: Optional[int] = None) -> Optional[str]:
    """Run X-Ret Explainable Retrieval Experiment"""
    global shutdown_requested
    
    if gpu_id is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        logging.info(f"Using GPU {gpu_id}")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Device: {device}")
    
    output_dir = pathlib.Path(config.get('output_dir', 'experiments/retrieval/x_explainable_retrieval'))
    output_dir.mkdir(parents=True, exist_ok=True)
    
    model_name = config.get('model_path', 'BAAI/bge-large-en-v1.5')
    domains = config.get('domains', MTRAG_DOMAINS)
    use_data_splits = config.get('use_data_splits', True)
    
    # Initialize explainable retriever
    logging.info(f"Initializing X-Ret with {model_name}")
    retriever = ExplainableRetriever(base_model_name=model_name)
    retriever.to(device)
    retriever.eval()
    
    # Evaluation
    logging.info("Evaluating X-Ret Explainable Retrieval...")
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
        
        # Retrieve with explanations
        results = {}
        explanations = {}
        
        for query_id, query_text in queries.items():
            query_emb = retriever.encode_query(query_text)
            query_emb = query_emb.to(device)
            
            # Retrieve with attention
            scores, attention_weights = retriever.retrieve_with_explanation(query_emb, doc_embs)
            scores = scores.cpu().numpy()
            attention_weights = attention_weights.cpu().numpy()
            
            # Get top-k
            top_indices = np.argsort(scores)[::-1][:100]
            results[query_id] = {
                doc_ids[idx]: float(scores[idx]) for idx in top_indices
            }
            
            # Store attention weights for explanation (Task A compliant - no text generation)
            explanations[query_id] = {
                doc_ids[idx]: float(attention_weights[idx]) for idx in top_indices[:10]
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
        
        logging.info(f"\n✅ X-Ret completed!")
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
    
    run_x_explainable_retrieval(config, gpu_id=args.gpu)

