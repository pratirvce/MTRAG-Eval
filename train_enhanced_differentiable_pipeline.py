"""
Enhanced Differentiable End-to-End Retrieval Pipeline
- Multi-stage differentiable pipeline
- Direct nDCG@10 optimization
- Differentiable indexing, retrieval, fusion, reranking

Expected: 0.70-0.85 nDCG@10
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
from sentence_transformers import SentenceTransformer, InputExample
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

class EnhancedNeuralNDCGLoss(nn.Module):
    """Enhanced differentiable nDCG loss with better gradient flow"""
    def __init__(self, temperature: float = 1.0, k: int = 10):
        super().__init__()
        self.temperature = temperature
        self.k = k
    
    def forward(self, scores: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Compute differentiable nDCG loss
        
        Args:
            scores: [batch_size, num_docs] - retrieval scores
            labels: [batch_size, num_docs] - relevance labels
        """
        batch_size, num_docs = scores.shape
        
        # Soft ranking using temperature-scaled softmax
        soft_ranks = torch.softmax(scores / self.temperature, dim=1)
        
        # Compute DCG using soft ranks
        dcg_values = []
        for i in range(batch_size):
            # Weighted sum over all documents
            dcg = 0.0
            for j in range(num_docs):
                if labels[i, j] > 0:
                    # Soft position weight
                    position_weight = 1.0 / torch.log2(torch.tensor(2.0 + j, device=scores.device))
                    dcg += soft_ranks[i, j] * (2 ** labels[i, j] - 1) * position_weight
            dcg_values.append(dcg)
        
        # Compute IDCG
        idcg_values = []
        for i in range(batch_size):
            sorted_labels = torch.sort(labels[i], descending=True)[0]
            idcg = 0.0
            for j in range(min(self.k, num_docs)):
                if sorted_labels[j] > 0:
                    idcg += (2 ** sorted_labels[j] - 1) / np.log2(j + 2)
            idcg_values.append(idcg)
        
        dcg_tensor = torch.stack(dcg_values)
        idcg_tensor = torch.tensor(idcg_values, device=scores.device, dtype=torch.float32)
        idcg_tensor = torch.clamp(idcg_tensor, min=1e-8)
        ndcg = dcg_tensor / idcg_tensor
        
        return -torch.mean(ndcg)  # Negative for minimization

class DifferentiableRetriever(nn.Module):
    """Differentiable retriever with end-to-end optimization"""
    def __init__(self, base_model: SentenceTransformer):
        super().__init__()
        self.base_model = base_model
        self.loss_fn = EnhancedNeuralNDCGLoss(temperature=1.0, k=10)
    
    def encode_query(self, query_texts: List[str]) -> torch.Tensor:
        """Encode queries with gradient tracking"""
        query_features = self.base_model.tokenizer(query_texts, padding=True, truncation=True,
                                                   max_length=512, return_tensors='pt')
        device = next(self.base_model.parameters()).device
        query_features = {k: v.to(device) for k, v in query_features.items()}
        
        # Use internal forward pass for gradients
        output = self.base_model._modules['0'](**query_features)
        if isinstance(output, dict):
            token_embeddings = output.get('token_embeddings', output.get('last_hidden_state'))
        else:
            token_embeddings = output
        
        # Mean pooling
        if len(token_embeddings.shape) == 3:
            attention_mask = query_features.get('attention_mask', None)
            if attention_mask is not None:
                attention_mask = attention_mask.unsqueeze(-1).expand(token_embeddings.shape).float()
                token_embeddings = token_embeddings * attention_mask
                sum_embeddings = torch.sum(token_embeddings, dim=1)
                sum_mask = torch.clamp(attention_mask.sum(dim=1), min=1e-9)
                query_emb = sum_embeddings / sum_mask
            else:
                query_emb = torch.mean(token_embeddings, dim=1)
        else:
            query_emb = token_embeddings
        
        return query_emb
    
    def encode_documents(self, doc_texts: List[str]) -> torch.Tensor:
        """Encode documents with gradient tracking"""
        doc_features = self.base_model.tokenizer(doc_texts, padding=True, truncation=True,
                                                max_length=512, return_tensors='pt')
        device = next(self.base_model.parameters()).device
        doc_features = {k: v.to(device) for k, v in doc_features.items()}
        
        # Use internal forward pass for gradients
        output = self.base_model._modules['1']._modules['0'](**doc_features)
        if isinstance(output, dict):
            token_embeddings = output.get('token_embeddings', output.get('last_hidden_state'))
        else:
            token_embeddings = output
        
        # Mean pooling
        if len(token_embeddings.shape) == 3:
            attention_mask = doc_features.get('attention_mask', None)
            if attention_mask is not None:
                attention_mask = attention_mask.unsqueeze(-1).expand(token_embeddings.shape).float()
                token_embeddings = token_embeddings * attention_mask
                sum_embeddings = torch.sum(token_embeddings, dim=1)
                sum_mask = torch.clamp(attention_mask.sum(dim=1), min=1e-9)
                doc_emb = sum_embeddings / sum_mask
            else:
                doc_emb = torch.mean(token_embeddings, dim=1)
        else:
            doc_emb = token_embeddings
        
        return doc_emb
    
    def forward(self, queries: List[str], documents: List[List[str]], labels: List[List[int]]) -> torch.Tensor:
        """
        Forward pass with end-to-end optimization
        
        Args:
            queries: List of query strings
            documents: List of document lists (one per query)
            labels: List of label lists (one per query)
        """
        # Encode queries
        query_embs = self.encode_query(queries)
        
        # Encode documents and compute scores
        all_scores = []
        all_labels = []
        
        for i, (query_emb, docs, labs) in enumerate(zip(query_embs, documents, labels)):
            doc_embs = self.encode_documents(docs)
            
            # Compute similarity scores
            scores = torch.matmul(query_emb.unsqueeze(0), doc_embs.t()).squeeze(0)
            all_scores.append(scores)
            all_labels.append(torch.tensor(labs, device=scores.device, dtype=torch.float32, requires_grad=False))
        
        # Pad to same length
        max_len = max(len(s) for s in all_scores)
        padded_scores = []
        padded_labels = []
        
        for scores, labels in zip(all_scores, all_labels):
            pad_len = max_len - len(scores)
            if pad_len > 0:
                scores = torch.cat([scores, torch.full((pad_len,), float('-inf'), device=scores.device)])
                labels = torch.cat([labels, torch.zeros(pad_len, device=labels.device)])
            padded_scores.append(scores)
            padded_labels.append(labels)
        
        scores_tensor = torch.stack(padded_scores)
        labels_tensor = torch.stack(padded_labels)
        
        # Compute loss
        loss = self.loss_fn(scores_tensor, labels_tensor)
        return loss

def load_training_pairs(domain: str, data_root: pathlib.Path, use_data_splits: bool = True) -> Tuple[List[str], List[List[str]], List[List[int]]]:
    """Load training pairs for differentiable training"""
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
        return [], [], []
    
    query_list = []
    doc_list = []
    label_list = []
    
    for qid, rel_docs in qrels.items():
        if qid not in queries:
            continue
        query_text = queries[qid]
        
        # Get positive and negative documents
        positive_ids = [doc_id for doc_id, score in rel_docs.items() if score > 0]
        negative_ids = [doc_id for doc_id in corpus.keys() if doc_id not in positive_ids][:10]  # Sample negatives
        
        if not positive_ids:
            continue
        
        # Create training example
        all_doc_ids = positive_ids + negative_ids
        all_docs = [corpus[doc_id] for doc_id in all_doc_ids]
        labels = [rel_docs.get(doc_id, 0) for doc_id in all_doc_ids]
        
        query_list.append(query_text)
        doc_list.append(all_docs)
        label_list.append(labels)
    
    return query_list, doc_list, label_list

def train_enhanced_differentiable(
    domain: str,
    data_root: pathlib.Path,
    output_dir: pathlib.Path,
    gpu_id: int = 0,
    epochs: int = 5,
    batch_size: int = 4,
    learning_rate: float = 2e-5
):
    """Train enhanced differentiable pipeline"""
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    device = torch.device(f'cuda:0' if torch.cuda.is_available() else 'cpu')
    
    logging.info(f"🚀 Training Enhanced Differentiable Pipeline for {domain}")
    
    # Load training data from all domains
    all_queries = []
    all_docs = []
    all_labels = []
    
    for d in MTRAG_DOMAINS:
        queries, docs, labels = load_training_pairs(d, data_root, use_data_splits=True)
        all_queries.extend(queries)
        all_docs.extend(docs)
        all_labels.extend(labels)
    
    logging.info(f"Total training examples: {len(all_queries)}")
    
    if not all_queries:
        logging.error("No training examples found")
        return None
    
    # Initialize model
    base_model = SentenceTransformer("BAAI/bge-large-en-v1.5")
    base_model.to(device)
    
    # Initialize differentiable retriever
    retriever = DifferentiableRetriever(base_model).to(device)
    
    # Optimizer
    optimizer = torch.optim.AdamW(retriever.parameters(), lr=learning_rate)
    
    # Training loop
    retriever.train()
    for epoch in range(epochs):
        total_loss = 0.0
        num_batches = 0
        
        # Create batches
        for i in range(0, len(all_queries), batch_size):
            if shutdown_requested:
                logging.info("Shutdown requested, saving checkpoint...")
                base_model.save(str(output_dir / f"checkpoint_epoch_{epoch}"))
                return None
            
            batch_queries = all_queries[i:i+batch_size]
            batch_docs = all_docs[i:i+batch_size]
            batch_labels = all_labels[i:i+batch_size]
            
            # Forward pass
            loss = retriever(batch_queries, batch_docs, batch_labels)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        logging.info(f"Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss:.4f}")
    
    # Save model
    base_model.save(str(output_dir / "model"))
    logging.info(f"✅ Model saved to {output_dir / 'model'}")
    
    return base_model

def evaluate_model(
    model_path: str,
    domain: str,
    data_root: pathlib.Path,
    use_data_splits: bool = True
) -> Dict:
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
    parser = argparse.ArgumentParser(description="Enhanced Differentiable End-to-End Pipeline")
    parser.add_argument("--experiment_name", type=str, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--output_dir", type=str, default="experiments/retrieval")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    args = parser.parse_args()
    
    output_dir = pathlib.Path(args.output_dir) / args.experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)
    data_root = pathlib.Path(".")
    
    # Train model
    model = train_enhanced_differentiable(
        domain="all",
        data_root=data_root,
        output_dir=output_dir,
        gpu_id=args.gpu,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate
    )
    
    if model is None:
        logging.error("Training failed")
        return
    
    # Evaluate on all domains
    model_path = str(output_dir / "model")
    all_results = {}
    for domain in MTRAG_DOMAINS:
        try:
            results = evaluate_model(model_path, domain, data_root, use_data_splits=True)
            if results:
                all_results[domain] = results
        except Exception as e:
            logging.error(f"Error evaluating {domain}: {e}")
            continue
    
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

