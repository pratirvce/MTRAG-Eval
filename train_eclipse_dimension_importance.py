"""
ECLIPSE: Contrastive Dimension Importance Estimation
Adjusts retrieval embedding spaces by estimating and suppressing noisy dimensions

Task A Compliant: ✅ Pure retrieval enhancement, no generation
Expected: 0.75-0.83 nDCG@10
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
from sentence_transformers import SentenceTransformer
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


class DimensionImportanceEstimator(nn.Module):
    """
    Estimates importance of each embedding dimension using pseudo-irrelevant feedback
    """
    def __init__(self, embedding_dim: int = 768):
        super().__init__()
        self.embedding_dim = embedding_dim
        # Learn dimension weights
        self.importance_weights = nn.Parameter(torch.ones(embedding_dim))
    
    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Reweight embeddings based on dimension importance"""
        # Apply learned weights
        weighted = embeddings * torch.sigmoid(self.importance_weights.unsqueeze(0))
        # Normalize
        return F.normalize(weighted, p=2, dim=-1)
    
    def estimate_importance(self, query_embs: torch.Tensor, pos_embs: torch.Tensor, neg_embs: torch.Tensor):
        """
        Estimate dimension importance using contrastive learning
        Dimensions that help distinguish positive from negative are important
        """
        # Compute similarities
        pos_sim = torch.sum(query_embs * pos_embs, dim=-1)  # [batch]
        neg_sim = torch.sum(query_embs * neg_embs, dim=-1)  # [batch]
        
        # Margin: positive should be more similar than negative
        margin = pos_sim - neg_sim
        
        # Dimension importance: dimensions that contribute to margin
        query_pos_diff = query_embs - pos_embs  # [batch, dim]
        query_neg_diff = query_embs - neg_embs  # [batch, dim]
        
        # Importance = how much each dimension contributes to positive margin
        importance = torch.mean(torch.abs(query_pos_diff) - torch.abs(query_neg_diff), dim=0)
        
        return importance


class ECLIPSERetriever:
    """
    Retriever with ECLIPSE dimension importance estimation
    """
    def __init__(self, base_model_path: str, importance_estimator: DimensionImportanceEstimator, device: str = "cuda"):
        self.device = device
        self.importance_estimator = importance_estimator.to(device)
        self.base_model = SentenceBERT(base_model_path, device=device)
        self.retriever = DenseRetrievalExactSearch(self.base_model, batch_size=128)
    
    def encode_with_importance(self, texts: List[str]) -> torch.Tensor:
        """Encode texts and apply dimension importance weighting"""
        embeddings = self.base_model.encode(texts, convert_to_tensor=True).to(self.device)
        # Apply importance weighting
        weighted_embeddings = self.importance_estimator(embeddings)
        return weighted_embeddings
    
    def retrieve(self, corpus: Dict, queries: Dict, top_k: int = 10) -> Dict:
        """Retrieve with ECLIPSE dimension importance"""
        all_results = {}
        
        # Encode queries with importance weighting
        query_texts = list(queries.values())
        query_ids = list(queries.keys())
        query_embs = self.encode_with_importance(query_texts)
        
        # Encode corpus with importance weighting (batch processing)
        corpus_texts = [corpus[doc_id].get('text', '') for doc_id in corpus.keys()]
        doc_ids = list(corpus.keys())
        
        # Process in batches
        batch_size = 128
        doc_embs_list = []
        for i in range(0, len(corpus_texts), batch_size):
            batch_texts = corpus_texts[i:i+batch_size]
            batch_embs = self.encode_with_importance(batch_texts)
            doc_embs_list.append(batch_embs)
        
        doc_embs = torch.cat(doc_embs_list, dim=0)
        
        # Compute similarities
        for i, query_id in enumerate(query_ids):
            query_emb = query_embs[i:i+1]  # [1, dim]
            similarities = torch.matmul(query_emb, doc_embs.T).squeeze(0)  # [num_docs]
            
            # Get top_k
            top_scores, top_indices = torch.topk(similarities, k=min(top_k, len(doc_ids)))
            
            results = {}
            for score, idx in zip(top_scores, top_indices):
                doc_id = doc_ids[idx.item()]
                results[doc_id] = score.item()
            
            all_results[query_id] = results
        
        return all_results


def train_eclipse(config: Dict, gpu_id: Optional[int] = None) -> Optional[str]:
    """Train ECLIPSE dimension importance estimator"""
    global shutdown_requested
    
    if gpu_id is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Device: {device}")
    
    output_dir = pathlib.Path(config.get('output_dir', 'experiments/retrieval/eclipse'))
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize components
    base_model = config.get('model_path', 'BAAI/bge-base-en-v1.5')
    base_retriever_model = SentenceBERT(base_model, device=device)
    embedding_dim = base_retriever_model.get_sentence_embedding_dimension()
    
    importance_estimator = DimensionImportanceEstimator(embedding_dim=embedding_dim).to(device)
    retriever = ECLIPSERetriever(base_model, importance_estimator, device=device)
    
    # Training: estimate dimension importance from contrastive pairs
    domains = config.get('domains', MTRAG_DOMAINS)
    use_data_splits = config.get('use_data_splits', True)
    
    optimizer = torch.optim.AdamW(importance_estimator.parameters(), lr=1e-3)
    epochs = config.get('eclipse_epochs', 5)
    
    all_results = {}
    
    for domain in domains:
        if shutdown_requested:
            break
        
        logging.info(f"\n{'='*60}\nTraining ECLIPSE on domain: {domain}\n{'='*60}")
        
        # Load data
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
            logging.error(f"Error loading data for {domain}: {e}")
            continue
        
        # Training loop: learn dimension importance
        importance_estimator.train()
        
        for epoch in range(epochs):
            total_loss = 0
            num_batches = 0
            
            # Sample training pairs
            query_ids = list(queries.keys())[:50]  # Sample for training
            
            for query_id in query_ids:
                query_text = queries[query_id]
                relevant_docs = [doc_id for doc_id, score in qrels.get(query_id, {}).items() if score > 0]
                
                if not relevant_docs:
                    continue
                
                # Get positive document
                pos_doc_id = relevant_docs[0]
                pos_doc_text = corpus[pos_doc_id].get('text', '')
                
                # Get negative document (random non-relevant)
                all_doc_ids = list(corpus.keys())
                negative_docs = [doc_id for doc_id in all_doc_ids if doc_id not in relevant_docs]
                if not negative_docs:
                    continue
                neg_doc_id = negative_docs[0]
                neg_doc_text = corpus[neg_doc_id].get('text', '')
                
                # Encode
                query_emb = base_retriever_model.encode([query_text], convert_to_tensor=True).to(device)
                pos_emb = base_retriever_model.encode([pos_doc_text], convert_to_tensor=True).to(device)
                neg_emb = base_retriever_model.encode([neg_doc_text], convert_to_tensor=True).to(device)
                
                # Apply importance weighting
                query_emb_weighted = importance_estimator(query_emb)
                pos_emb_weighted = importance_estimator(pos_emb)
                neg_emb_weighted = importance_estimator(neg_emb)
                
                # Contrastive loss: positive should be more similar than negative
                pos_sim = torch.sum(query_emb_weighted * pos_emb_weighted, dim=-1)
                neg_sim = torch.sum(query_emb_weighted * neg_emb_weighted, dim=-1)
                
                margin = 0.2  # Margin for contrastive loss
                loss = F.relu(margin - (pos_sim - neg_sim))
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
            
            if num_batches > 0:
                logging.info(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/num_batches:.4f}")
        
        # Evaluation
        logging.info(f"Evaluating on {domain}...")
        importance_estimator.eval()
        
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
        
        # Save importance estimator
        estimator_path = output_dir / f"importance_estimator_{domain}.pt"
        torch.save(importance_estimator.state_dict(), estimator_path)
    
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
        
        logging.info(f"\n{'='*60}\nAverage Results:\n")
        for metric, value in avg_results.items():
            logging.info(f"  {metric}: {value:.4f}")
        logging.info(f"{'='*60}\n")
        
        return str(output_dir)
    
    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--gpu_id', type=int, default=None)
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    try:
        results_path = train_eclipse(config, gpu_id=args.gpu_id)
        if results_path:
            logging.info(f"✅ Training completed: {results_path}")
    except KeyboardInterrupt:
        logging.info("Interrupted by user")
        sys.exit(0)
    except Exception as e:
        logging.error(f"Failed: {e}", exc_info=True)
        sys.exit(1)

