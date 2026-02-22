"""
Mixture of Retrieval Experts (MoRE)
Train multiple specialized retrievers and learn to route queries to appropriate expert
Expected: 0.52-0.56 nDCG@10
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
from torch.utils.data.dataloader import default_collate
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


def input_example_collate_fn(batch):
    """Custom collate function for InputExample objects"""
    # Extract texts from InputExample objects
    texts = [example.texts for example in batch]
    # Return as list of (query, positive) pairs
    return texts

def signal_handler(sig, frame):
    global shutdown_requested
    logging.info("\n⚠️  Shutdown signal received. Saving checkpoint...")
    shutdown_requested = True

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


class ExpertRouter(nn.Module):
    """Routes queries to appropriate expert"""
    def __init__(self, input_dim: int = 768, num_experts: int = 4):
        super().__init__()
        self.num_experts = num_experts
        self.router = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Linear(input_dim // 2, num_experts),
            nn.Softmax(dim=1)
        )
    
    def forward(self, query_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            query_emb: [batch, input_dim]
        Returns:
            expert_weights: [batch, num_experts]
        """
        return self.router(query_emb)


class MixtureOfExpertsRetriever(nn.Module):
    """Mixture of Retrieval Experts"""
    def __init__(self, base_model_path: str, num_experts: int = 4, device: str = "cuda"):
        super().__init__()
        self.num_experts = num_experts
        self.device = device
        
        # Create multiple expert models
        self.experts = nn.ModuleList([
            SentenceTransformer(base_model_path) for _ in range(num_experts)
        ])
        
        # Router
        hidden_dim = self.experts[0].get_sentence_embedding_dimension()
        self.router = ExpertRouter(hidden_dim, num_experts)
        
        # Expert descriptions
        self.expert_types = [
            "early_conversation",  # Broad retrieval
            "mid_conversation",    # Focused retrieval
            "late_conversation",   # Refinement retrieval
            "topic_shift"          # Topic shift detection
        ]
    
    def encode_query(self, query_texts: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode query and get expert weights"""
        # Get base query embedding using forward pass for gradients
        query_features = self.experts[0].tokenizer(query_texts, padding=True, truncation=True,
                                                   max_length=512, return_tensors='pt')
        query_features = {k: v.to(self.device) for k, v in query_features.items()}
        query_emb = self.experts[0](query_features)['sentence_embedding']
        
        # Route to experts
        expert_weights = self.router(query_emb)  # [batch, num_experts]
        
        # Get expert embeddings using forward pass
        expert_embs = []
        for expert in self.experts:
            emb = expert(query_features)['sentence_embedding']
            expert_embs.append(emb)
        
        expert_embs = torch.stack(expert_embs, dim=1)  # [batch, num_experts, dim]
        
        # Weighted combination
        expert_weights_expanded = expert_weights.unsqueeze(2)  # [batch, num_experts, 1]
        combined_emb = torch.sum(expert_embs * expert_weights_expanded, dim=1)  # [batch, dim]
        
        return combined_emb, expert_weights
    
    def encode_documents(self, doc_texts: List[str]) -> torch.Tensor:
        """Encode documents using first expert with forward pass for gradients"""
        doc_features = self.experts[0].tokenizer(doc_texts, padding=True, truncation=True,
                                                 max_length=512, return_tensors='pt')
        doc_features = {k: v.to(self.device) for k, v in doc_features.items()}
        return self.experts[0](doc_features)['sentence_embedding']
    
    def save(self, path: str):
        """Save the model to disk"""
        import pathlib
        save_path = pathlib.Path(path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save each expert
        for i, expert in enumerate(self.experts):
            expert_path = save_path / f"expert_{i}"
            expert.save(str(expert_path))
        
        # Save router
        router_path = save_path / "router.pt"
        torch.save(self.router.state_dict(), router_path)
        
        # Save metadata
        metadata = {
            'num_experts': self.num_experts,
            'expert_types': self.expert_types,
            'hidden_dim': self.experts[0].get_sentence_embedding_dimension()
        }
        import json
        with open(save_path / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)


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


def train_mixture_experts(config: Dict, gpu_id: Optional[int] = None) -> Optional[str]:
    """Train mixture of experts retriever"""
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
    
    output_dir = pathlib.Path(config.get('output_dir', 'experiments/retrieval/mixture_experts'))
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
    
    domains = config.get('domains', MTRAG_DOMAINS)
    base_model = config.get('model_path', 'BAAI/bge-base-en-v1.5')
    epochs = config.get('epochs', 3)
    batch_size = config.get('batch_size', 4)
    gradient_accumulation_steps = config.get('gradient_accumulation_steps', 2)
    use_fp16 = config.get('use_fp16', True)
    num_experts = config.get('num_experts', 4)
    
    # Create model
    logging.info(f"Creating Mixture of Experts with {num_experts} experts")
    model = MixtureOfExpertsRetriever(base_model, num_experts=num_experts, device=device)
    model.to(device)
    
    scaler = None
    if use_fp16 and device == "cuda":
        try:
            from torch.cuda.amp import autocast, GradScaler
            scaler = GradScaler()
        except ImportError:
            use_fp16 = False
    
    for param in model.parameters():
        param.requires_grad = True
    
    # Training
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
        
        dataloader = DataLoader(examples, batch_size=batch_size, shuffle=True, collate_fn=input_example_collate_fn)
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
        
        model.train()
        for epoch in range(epochs):
            total_loss = 0
            optimizer.zero_grad()
            
            for batch_idx, batch_texts in enumerate(dataloader):
                # batch_texts is a list of [query, positive] pairs
                queries_batch = [texts[0] for texts in batch_texts]
                positives_batch = [texts[1] for texts in batch_texts]
                
                # Encode with experts
                query_embs, expert_weights = model.encode_query(queries_batch)
                pos_embs = model.encode_documents(positives_batch)
                
                # Normalize
                query_embs = F.normalize(query_embs, p=2, dim=1)
                pos_embs = F.normalize(pos_embs, p=2, dim=1)
                
                # Contrastive loss
                similarities = torch.sum(query_embs * pos_embs, dim=1)
                loss = -torch.mean(torch.log(torch.sigmoid(similarities / 0.05) + 1e-8))
                
                # Add expert diversity loss (encourage different experts)
                expert_diversity = -torch.mean(torch.sum(expert_weights * torch.log(expert_weights + 1e-8), dim=1))
                loss = loss + 0.1 * expert_diversity
                
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
                
                if device == "cuda" and (batch_idx + 1) % 10 == 0:
                    torch.cuda.empty_cache()
            
            logging.info(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader):.4f}")
        
        # Save trained model after each domain
        trained_model_path = checkpoint_dir / f"trained_model_{domain}"
        model.save(str(trained_model_path))
        logging.info(f"Saved trained model for {domain} to {trained_model_path}")
    
    # Evaluation using the trained model
    logging.info("\nEvaluating trained model...")
    # Use the trained model from the last domain
    if len(domains) > 0:
        last_domain = list(domains)[-1] if isinstance(domains, (list, tuple)) else domains[-1]
        trained_model_path = checkpoint_dir / f"trained_model_{last_domain}"
        if trained_model_path.exists():
            logging.info(f"Loading trained model from {trained_model_path}")
            retriever = DenseRetrievalExactSearch(SentenceBERT(str(trained_model_path), device=device), batch_size=128)
        else:
            logging.warning(f"Trained model not found at {trained_model_path}, using base model")
            retriever = DenseRetrievalExactSearch(SentenceBERT(base_model, device=device), batch_size=128)
    else:
        logging.warning("No domains to evaluate, using base model")
        retriever = DenseRetrievalExactSearch(SentenceBERT(base_model, device=device), batch_size=128)
    evaluator = EvaluateRetrieval(retriever, k_values=[1, 3, 5, 10])
    
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
            logging.error(f"Error evaluating {domain}: {e}")
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
        results_path = train_mixture_experts(config, args.gpu_id)
        if results_path:
            logging.info(f"✅ Mixture of experts training completed: {results_path}")
    except KeyboardInterrupt:
        logging.info("Interrupted by user")
        sys.exit(0)
    except Exception as e:
        logging.error(f"Failed: {e}", exc_info=True)
        sys.exit(1)

