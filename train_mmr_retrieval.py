"""
MMR (Maximal Marginal Relevance) Retrieval
Balances relevance and diversity in retrieval results
Task A - Retrieval Only
Expected: +0.02-0.05 nDCG@10 improvement

MMR Algorithm:
  MMR = λ * relevance - (1-λ) * max_similarity_to_selected
  Selects documents that maximize relevance while minimizing redundancy
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

class MMRRetriever:
    """
    Maximal Marginal Relevance Retriever
    Balances relevance and diversity in retrieval results
    """
    def __init__(self, base_model: SentenceTransformer, lambda_param: float = 0.6):
        """
        Args:
            base_model: SentenceTransformer model for encoding
            lambda_param: Trade-off between relevance (λ) and diversity (1-λ)
                          Higher λ = more relevance, lower λ = more diversity
        """
        self.base_model = base_model
        self.lambda_param = lambda_param
    
    def mmr_rerank(self, query: str, candidates: Dict[str, float], 
                   corpus: Dict[str, Dict], top_k: int = 10) -> Dict[str, float]:
        """
        Rerank candidates using MMR algorithm
        
        Args:
            query: Query text
            candidates: Dict of {doc_id: relevance_score} from initial retrieval
            corpus: Full corpus dict
            top_k: Number of documents to return
        
        Returns:
            Dict of {doc_id: mmr_score} for top_k documents
        """
        if not candidates:
            return {}
        
        # Encode query
        query_emb = self.base_model.encode(query, convert_to_numpy=True, show_progress_bar=False)
        query_emb = query_emb / (np.linalg.norm(query_emb) + 1e-8)  # Normalize
        
        # Get candidate documents and their embeddings
        candidate_ids = list(candidates.keys())
        candidate_texts = []
        for doc_id in candidate_ids:
            doc = corpus.get(doc_id, {})
            title = doc.get('title', '')
            text = doc.get('text', '')
            doc_text = f"{title}\n\n{text}".strip() if title else text
            candidate_texts.append(doc_text)
        
        # Encode all candidates
        candidate_embs = self.base_model.encode(
            candidate_texts,
            batch_size=128,
            show_progress_bar=False,
            convert_to_numpy=True
        )
        # Normalize embeddings
        candidate_embs = candidate_embs / (np.linalg.norm(candidate_embs, axis=1, keepdims=True) + 1e-8)
        
        # Compute relevance scores (cosine similarity with query)
        relevance_scores = np.dot(candidate_embs, query_emb)
        
        # MMR algorithm: Greedy selection
        selected = []
        selected_embs = []
        remaining = list(range(len(candidate_ids)))
        mmr_scores = {}
        
        # Select first document (highest relevance)
        if remaining:
            first_idx = np.argmax(relevance_scores)
            selected.append(first_idx)
            selected_embs.append(candidate_embs[first_idx])
            remaining.remove(first_idx)
            doc_id = candidate_ids[first_idx]
            mmr_scores[doc_id] = float(relevance_scores[first_idx])
        
        # Select remaining documents using MMR
        while len(selected) < top_k and remaining:
            best_mmr = -float('inf')
            best_idx = None
            
            for idx in remaining:
                relevance = relevance_scores[idx]
                
                # Compute max similarity to already selected documents
                if selected_embs:
                    similarities = np.dot(candidate_embs[idx], np.array(selected_embs).T)
                    max_similarity = np.max(similarities)
                else:
                    max_similarity = 0.0
                
                # MMR score
                mmr = self.lambda_param * relevance - (1 - self.lambda_param) * max_similarity
                
                if mmr > best_mmr:
                    best_mmr = mmr
                    best_idx = idx
            
            if best_idx is not None:
                selected.append(best_idx)
                selected_embs.append(candidate_embs[best_idx])
                remaining.remove(best_idx)
                doc_id = candidate_ids[best_idx]
                mmr_scores[doc_id] = float(best_mmr)
            else:
                break
        
        return mmr_scores
    
    def retrieve(self, corpus: Dict[str, Dict], queries: Dict[str, str], 
                 initial_top_k: int = 100, final_top_k: int = 10) -> Dict[str, Dict[str, float]]:
        """
        Retrieve using MMR reranking
        
        Args:
            corpus: Full corpus
            queries: Dict of {query_id: query_text}
            initial_top_k: Number of candidates from initial retrieval
            final_top_k: Number of documents to return after MMR
        
        Returns:
            Dict of {query_id: {doc_id: mmr_score}}
        """
        # Initial dense retrieval
        dense_retriever = DenseRetrievalExactSearch(
            SentenceBERT(self.base_model, device=self.base_model.device),
            batch_size=128
        )
        evaluator = EvaluateRetrieval(dense_retriever, k_values=[initial_top_k])
        initial_results = evaluator.retrieve(corpus, queries)
        
        # MMR reranking for each query
        mmr_results = {}
        for query_id, query_text in queries.items():
            candidates = initial_results.get(query_id, {})
            if not candidates:
                mmr_results[query_id] = {}
                continue
            
            # Rerank using MMR
            mmr_scores = self.mmr_rerank(query_text, candidates, corpus, top_k=final_top_k)
            mmr_results[query_id] = mmr_scores
        
        return mmr_results

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
                doc_text = f"{title}\n\n{text}".strip() if title else text
                examples.append(InputExample(texts=[query_text, doc_text], label=1.0))
    
    logging.info(f"Created {len(examples)} positive pairs for {domain}")
    return examples, corpus, queries

def train_mmr_retrieval(config: Dict, gpu_id: Optional[int] = None) -> Optional[str]:
    """Train retrieval with MMR reranking"""
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
    
    output_dir = pathlib.Path(config.get('output_dir', 'experiments/retrieval/mmr_retrieval'))
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
    lambda_param = config.get('lambda_param', 0.6)  # MMR trade-off parameter
    initial_top_k = config.get('initial_top_k', 100)  # Initial retrieval candidates
    final_top_k = config.get('final_top_k', 10)  # Final results after MMR
    
    # Load base model
    logging.info(f"Loading base model: {base_model_name}")
    base_model = SentenceTransformer(base_model_name)
    base_model.to(device)
    
    # Fine-tune on MTRAG data (optional)
    epochs = config.get('epochs', 3)
    batch_size = config.get('batch_size', 8)
    gradient_accumulation_steps = config.get('gradient_accumulation_steps', 2)
    use_fp16 = config.get('use_fp16', True)
    
    scaler = None
    if use_fp16 and device == "cuda":
        try:
            from torch.cuda.amp import autocast, GradScaler
            scaler = GradScaler()
            logging.info("Mixed precision training (FP16) enabled")
        except ImportError:
            use_fp16 = False
    
    for param in base_model.parameters():
        param.requires_grad = True
    
    # Training loop
    if config.get('fine_tune', True):
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
                optimizer.zero_grad()
                
                for batch_idx, batch in enumerate(dataloader):
                    queries_batch = [ex.texts[0] for ex in batch]
                    positives_batch = [ex.texts[1] for ex in batch]
                    
                    # Encode with asymmetric prompts
                    query_prompts = [f"Represent this sentence for searching relevant passages: {q}" for q in queries_batch]
                    doc_prompts = [f"Represent this sentence for retrieval: {d}" for d in positives_batch]
                    
                    if use_fp16 and scaler is not None:
                        with autocast():
                            query_embs = base_model.encode(query_prompts, convert_to_tensor=True, show_progress_bar=False)
                            pos_embs = base_model.encode(doc_prompts, convert_to_tensor=True, show_progress_bar=False)
                            
                            query_embs = F.normalize(query_embs, p=2, dim=1)
                            pos_embs = F.normalize(pos_embs, p=2, dim=1)
                            
                            similarities = torch.sum(query_embs * pos_embs, dim=1)
                            loss = -torch.mean(torch.log(torch.sigmoid(similarities / 0.05) + 1e-8))
                            loss = loss / gradient_accumulation_steps
                        
                        scaler.scale(loss).backward()
                        if (batch_idx + 1) % gradient_accumulation_steps == 0:
                            scaler.step(optimizer)
                            scaler.update()
                            optimizer.zero_grad()
                    else:
                        query_embs = base_model.encode(query_prompts, convert_to_tensor=True, show_progress_bar=False)
                        pos_embs = base_model.encode(doc_prompts, convert_to_tensor=True, show_progress_bar=False)
                        
                        query_embs = F.normalize(query_embs, p=2, dim=1)
                        pos_embs = F.normalize(pos_embs, p=2, dim=1)
                        
                        similarities = torch.sum(query_embs * pos_embs, dim=1)
                        loss = -torch.mean(torch.log(torch.sigmoid(similarities / 0.05) + 1e-8))
                        loss = loss / gradient_accumulation_steps
                        loss.backward()
                        
                        if (batch_idx + 1) % gradient_accumulation_steps == 0:
                            optimizer.step()
                            optimizer.zero_grad()
                    
                    total_loss += loss.item() * gradient_accumulation_steps
                    
                    if device == "cuda" and (batch_idx + 1) % 10 == 0:
                        torch.cuda.empty_cache()
                
                logging.info(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader):.4f}")
        
        # Save trained model
        trained_model_path = checkpoint_dir / "trained_model"
        base_model.save(str(trained_model_path))
        logging.info(f"Saved trained model to {trained_model_path}")
    
    # Reload trained model if available
    trained_model_path = checkpoint_dir / "trained_model"
    if trained_model_path.exists():
        logging.info(f"Loading trained model from {trained_model_path}")
        base_model = SentenceTransformer(str(trained_model_path))
        base_model.to(device)
    
    # Evaluation with MMR
    logging.info("\nEvaluating with MMR retrieval...")
    mmr_retriever = MMRRetriever(base_model, lambda_param=lambda_param)
    
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
            
            # Retrieve with MMR
            logging.info(f"MMR retrieval for {domain} (λ={lambda_param})...")
            mmr_results = mmr_retriever.retrieve(
                corpus, queries, 
                initial_top_k=initial_top_k, 
                final_top_k=final_top_k
            )
            
            # Evaluate
            evaluator = EvaluateRetrieval(None, k_values=[1, 3, 5, 10])
            ndcg, _map, recall, precision = evaluator.evaluate(qrels, mmr_results, [1, 3, 5, 10])
            
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
            
            logging.info(f"\n{domain} Results:")
            logging.info(f"  Recall@10: {recall.get('Recall@10', 0):.4f}")
            logging.info(f"  nDCG@10: {ndcg.get('NDCG@10', 0):.4f}")
            
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
        results_path = train_mmr_retrieval(config, args.gpu_id)
        if results_path:
            logging.info(f"✅ MMR retrieval completed: {results_path}")
    except KeyboardInterrupt:
        logging.info("Interrupted by user")
        sys.exit(0)
    except Exception as e:
        logging.error(f"Failed: {e}", exc_info=True)
        sys.exit(1)

