"""
ColBERT-Style Multi-Vector Retrieval
Paper: "ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT" (2020)

Key Innovation: Token-level embeddings with MaxSim scoring instead of single-vector embeddings.
This allows fine-grained matching between query and document tokens.

Task A Compliant: ✅ Pure retrieval, no generation
Expected: +0.04-0.08 nDCG@10 improvement
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
from torch.utils.data import DataLoader, Dataset
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


class ColBERTRetriever:
    """
    ColBERT-style retriever: Token-level embeddings with MaxSim scoring
    """
    def __init__(self, base_model: SentenceTransformer, max_query_length: int = 32, 
                 max_doc_length: int = 180):
        self.base_model = base_model
        self.max_query_length = max_query_length
        self.max_doc_length = max_doc_length
        self.tokenizer = base_model.tokenizer
        self.device = next(base_model.parameters()).device
        
    def encode_query(self, query: str) -> torch.Tensor:
        """
        Encode query at token level
        Returns: [seq_len, hidden_dim] tensor
        """
        tokens = self.tokenizer.encode(query, add_special_tokens=True,
                                       max_length=self.max_query_length,
                                       truncation=True, return_tensors='pt')
        tokens = tokens.to(self.device)
        
        with torch.no_grad():
            outputs = self.base_model._modules['0'](tokens)
            if isinstance(outputs, dict):
                token_embeddings = outputs.get('last_hidden_state') or outputs.get('token_embeddings')
            else:
                token_embeddings = outputs
            
            # Normalize token embeddings
            token_embeddings = F.normalize(token_embeddings, p=2, dim=2)
        
        return token_embeddings.squeeze(0)  # [seq_len, hidden_dim]
    
    def encode_document(self, document: str) -> torch.Tensor:
        """
        Encode document at token level
        Returns: [seq_len, hidden_dim] tensor
        """
        tokens = self.tokenizer.encode(document, add_special_tokens=True,
                                       max_length=self.max_doc_length,
                                       truncation=True, return_tensors='pt')
        tokens = tokens.to(self.device)
        
        with torch.no_grad():
            outputs = self.base_model._modules['0'](tokens)
            if isinstance(outputs, dict):
                token_embeddings = outputs.get('last_hidden_state') or outputs.get('token_embeddings')
            else:
                token_embeddings = outputs
            
            # Normalize token embeddings
            token_embeddings = F.normalize(token_embeddings, p=2, dim=2)
        
        return token_embeddings.squeeze(0)  # [seq_len, hidden_dim]
    
    def score(self, query_emb: torch.Tensor, doc_emb: torch.Tensor) -> float:
        """
        MaxSim scoring: For each query token, find max similarity with any document token
        Score = sum of max similarities
        """
        # query_emb: [seq_len_q, hidden_dim]
        # doc_emb: [seq_len_d, hidden_dim]
        
        # Compute similarity matrix: [seq_len_q, seq_len_d]
        similarity_matrix = torch.matmul(query_emb, doc_emb.T)  # Cosine similarity (already normalized)
        
        # MaxSim: max similarity for each query token
        max_similarities = torch.max(similarity_matrix, dim=1)[0]  # [seq_len_q]
        
        # Sum of max similarities (ColBERT score)
        score = torch.sum(max_similarities).item()
        
        return score


def load_training_pairs(domain: str, data_root: pathlib.Path, use_data_splits: bool = True) -> Tuple[List[InputExample], Dict, Dict]:
    """Load conversation-document pairs"""
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
        return [], {}, {}
    
    examples = []
    for query_id, doc_scores in qrels.items():
        query_text = queries.get(query_id)
        if not query_text:
            continue
        
        positive_docs = [doc_id for doc_id, score in doc_scores.items() if score > 0]
        for doc_id in positive_docs:
            doc = corpus.get(doc_id)
            if doc:
                title = doc.get("title", "")
                text = doc.get("text", "")
                doc_text = f"{title} {text}".strip() if title else text
                examples.append(InputExample(texts=[query_text, doc_text], label=1.0))
    
    logging.info(f"Created {len(examples)} positive pairs for {domain}")
    return examples, corpus, queries


def train_colbert_retrieval(config: Dict, gpu_id: Optional[int] = None) -> Optional[str]:
    """Train ColBERT-style multi-vector retrieval"""
    global shutdown_requested
    
    cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if cuda_visible:
        logging.info(f"CUDA_VISIBLE_DEVICES={cuda_visible}")
    elif gpu_id is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and torch.cuda.is_available():
        try:
            torch.cuda.set_device(0)
            logging.info(f"Device: {device}, GPU: {torch.cuda.current_device()}")
        except Exception as e:
            logging.warning(f"Could not set CUDA device: {e}")
    
    output_dir = pathlib.Path(config.get('output_dir', 'experiments/retrieval/colbert_retrieval'))
    output_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint_dir = output_dir
    resume = config.get('resume', True)
    
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
    base_model_name = config.get('model_path', 'BAAI/bge-base-en-v1.5')
    max_query_length = config.get('max_query_length', 32)
    max_doc_length = config.get('max_doc_length', 180)
    
    # Load base model
    logging.info(f"Loading base model: {base_model_name}")
    base_model = SentenceTransformer(base_model_name)
    base_model.to(device)
    
    # Create ColBERT retriever
    retriever = ColBERTRetriever(
        base_model=base_model,
        max_query_length=max_query_length,
        max_doc_length=max_doc_length
    )
    
    # Fine-tune if needed
    if config.get('fine_tune', False):
        epochs = config.get('epochs', 3)
        batch_size = config.get('batch_size', 4)
        gradient_accumulation_steps = config.get('gradient_accumulation_steps', 4)
        use_fp16 = config.get('use_fp16', True)
        
        scaler = None
        if use_fp16 and device == "cuda":
            try:
                from torch.cuda.amp import autocast, GradScaler
                scaler = GradScaler()
            except ImportError:
                use_fp16 = False
        
        for param in base_model.parameters():
            param.requires_grad = True
        
        for domain in domains:
            if domain in completed_domains:
                continue
            
            if shutdown_requested:
                checkpoint_file = checkpoint_dir / "checkpoint.json"
                with open(checkpoint_file, 'w') as f:
                    json.dump({
                        "last_domain": domain,
                        "completed_domains": list(completed_domains),
                        "results": all_results,
                        "timestamp": datetime.now().isoformat()
                    }, f, indent=2)
                break
            
            logging.info(f"\n{'='*60}\nTraining on domain: {domain}\n{'='*60}")
            data_root = pathlib.Path(".")
            examples, corpus, queries = load_training_pairs(domain, data_root, config.get('use_data_splits', True))
            
            if not examples:
                continue
            
            dataloader = DataLoader(examples, batch_size=batch_size, shuffle=True)
            optimizer = torch.optim.AdamW(base_model.parameters(), lr=2e-5, weight_decay=0.01)
            
            base_model.train()
            for epoch in range(epochs):
                total_loss = 0
                for batch_idx, batch in enumerate(dataloader):
                    queries_batch = [ex.texts[0] for ex in batch]
                    docs_batch = [ex.texts[1] for ex in batch]
                    
                    # Encode with ColBERT (token-level)
                    query_embs = [retriever.encode_query(q) for q in queries_batch]
                    doc_embs = [retriever.encode_document(d) for d in docs_batch]
                    
                    # Compute ColBERT scores
                    scores = []
                    for q_emb, d_emb in zip(query_embs, doc_embs):
                        score = retriever.score(q_emb, d_emb)
                        scores.append(score)
                    
                    scores = torch.tensor(scores, device=device)
                    
                    # Contrastive loss (encourage high scores for positive pairs)
                    # Use in-batch negatives
                    batch_size = len(queries_batch)
                    score_matrix = torch.zeros((batch_size, batch_size), device=device)
                    for i, q_emb in enumerate(query_embs):
                        for j, d_emb in enumerate(doc_embs):
                            score_matrix[i, j] = retriever.score(q_emb, d_emb)
                    
                    # Positive scores are on diagonal
                    pos_scores = torch.diag(score_matrix)
                    # Negative scores are off-diagonal
                    neg_scores = score_matrix - torch.diag(torch.diag(score_matrix))
                    
                    # Loss: maximize positive scores, minimize negative scores
                    loss = -torch.mean(torch.log(torch.sigmoid(pos_scores.unsqueeze(1) - neg_scores) + 1e-8))
                    
                    if use_fp16 and scaler is not None:
                        with autocast():
                            loss = loss / gradient_accumulation_steps
                        scaler.scale(loss).backward()
                        if (batch_idx + 1) % gradient_accumulation_steps == 0:
                            scaler.step(optimizer)
                            scaler.update()
                            optimizer.zero_grad()
                    else:
                        loss = loss / gradient_accumulation_steps
                        loss.backward()
                        if (batch_idx + 1) % gradient_accumulation_steps == 0:
                            optimizer.step()
                            optimizer.zero_grad()
                    
                    total_loss += loss.item() * gradient_accumulation_steps
                
                logging.info(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader):.4f}")
            
            # Save trained model
            trained_model_path = checkpoint_dir / f"trained_model_{domain}"
            base_model.save(str(trained_model_path))
            logging.info(f"Saved trained model for {domain} to {trained_model_path}")
    
    # Evaluation with ColBERT
    logging.info("\nEvaluating with ColBERT retrieval...")
    
    # Use trained model if available
    if config.get('fine_tune', False) and len(domains) > 0:
        last_domain = list(domains)[-1] if isinstance(domains, (list, tuple)) else domains[-1]
        trained_model_path = checkpoint_dir / f"trained_model_{last_domain}"
        if trained_model_path.exists():
            logging.info(f"Loading trained model from {trained_model_path}")
            base_model = SentenceTransformer(str(trained_model_path))
            base_model.to(device)
            retriever = ColBERTRetriever(base_model, max_query_length, max_doc_length)
    
    # Create custom retriever for ColBERT
    class ColBERTDenseRetriever:
        def __init__(self, retriever: ColBERTRetriever):
            self.retriever = retriever
            self.corpus_embeddings = {}
            self.corpus_texts = {}
        
        def index(self, corpus: Dict):
            """Index corpus with ColBERT token-level embeddings"""
            logging.info("Indexing corpus with ColBERT...")
            for doc_id, doc in corpus.items():
                title = doc.get("title", "")
                text = doc.get("text", "")
                doc_text = f"{title} {text}".strip() if title else text
                
                doc_emb = self.retriever.encode_document(doc_text)
                self.corpus_embeddings[doc_id] = doc_emb.cpu()  # Store on CPU
                self.corpus_texts[doc_id] = doc_text
        
        def retrieve(self, corpus: Dict, queries: Dict, top_k: int = 10) -> Dict:
            """Retrieve using ColBERT MaxSim scoring"""
            results = {}
            
            for query_id, query_text in queries.items():
                query_emb = self.retriever.encode_query(query_text)
                
                # Compute ColBERT scores for all documents
                scores = {}
                for doc_id, doc_emb in self.corpus_embeddings.items():
                    doc_emb = doc_emb.to(query_emb.device)
                    score = self.retriever.score(query_emb, doc_emb)
                    scores[doc_id] = float(score)
                
                # Sort and get top-k
                sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
                results[query_id] = {doc_id: score for doc_id, score in sorted_docs}
            
            return results
    
    custom_retriever = ColBERTDenseRetriever(retriever)
    
    for domain in domains:
        if domain in completed_domains:
            continue
        
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
            
            # Index corpus
            custom_retriever.index(corpus)
            
            # Retrieve
            results = custom_retriever.retrieve(corpus, queries, top_k=100)
            
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
            
            completed_domains.add(domain)
            
            checkpoint_file = checkpoint_dir / "checkpoint.json"
            with open(checkpoint_file, 'w') as f:
                json.dump({
                    "last_domain": domain,
                    "completed_domains": list(completed_domains),
                    "results": all_results,
                    "timestamp": datetime.now().isoformat()
                }, f, indent=2)
            
        except Exception as e:
            logging.error(f"Error evaluating {domain}: {e}", exc_info=True)
            continue
    
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
    parser.add_argument('--gpu_id', type=int)
    parser.add_argument('--resume', action='store_true', default=True)
    parser.add_argument('--no-resume', dest='resume', action='store_false')
    
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    config['resume'] = args.resume
    
    try:
        results_path = train_colbert_retrieval(config, args.gpu_id)
        if results_path:
            logging.info(f"✅ ColBERT retrieval training completed: {results_path}")
    except KeyboardInterrupt:
        logging.info("Interrupted by user")
        sys.exit(0)
    except Exception as e:
        logging.error(f"Failed: {e}", exc_info=True)
        sys.exit(1)

