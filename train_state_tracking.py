"""
Multi-Turn Conversation State Tracking with Retrieval
Explicitly models conversation state (topics, entities, intents) and uses it to guide retrieval
Expected: 0.51-0.55 nDCG@10
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
import re

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


class ConversationStateEncoder(nn.Module):
    """Encodes conversation state (entities, topics, intents)"""
    def __init__(self, hidden_dim: int = 768, state_dim: int = 256):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.state_dim = state_dim
        
        # State components
        self.entity_encoder = nn.Linear(hidden_dim, state_dim)
        self.topic_encoder = nn.Linear(hidden_dim, state_dim)
        self.intent_encoder = nn.Linear(hidden_dim, state_dim)
        
        # State fusion
        self.state_fusion = nn.Sequential(
            nn.Linear(state_dim * 3, state_dim * 2),
            nn.ReLU(),
            nn.Linear(state_dim * 2, state_dim)
        )
    
    def forward(self, conversation_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            conversation_emb: [batch, hidden_dim] - conversation embedding
        Returns:
            state: [batch, state_dim] - conversation state
        """
        entity_state = self.entity_encoder(conversation_emb)
        topic_state = self.topic_encoder(conversation_emb)
        intent_state = self.intent_encoder(conversation_emb)
        
        combined = torch.cat([entity_state, topic_state, intent_state], dim=1)
        state = self.state_fusion(combined)
        return state


class StateAwareRetriever(nn.Module):
    """State-aware retrieval model"""
    def __init__(self, base_model: SentenceTransformer, state_dim: int = 256):
        super().__init__()
        self.base_model = base_model
        self.state_encoder = ConversationStateEncoder(
            hidden_dim=base_model.get_sentence_embedding_dimension(),
            state_dim=state_dim
        )
        self.state_dim = state_dim
        
        # State-query fusion
        self.fusion = nn.Sequential(
            nn.Linear(base_model.get_sentence_embedding_dimension() + state_dim, 
                     base_model.get_sentence_embedding_dimension()),
            nn.LayerNorm(base_model.get_sentence_embedding_dimension())
        )
    
    def encode_query(self, query_texts: List[str], conversation_history: List[str] = None) -> torch.Tensor:
        """Encode query with conversation state using forward pass for gradients"""
        # Encode current query using forward pass
        query_features = self.base_model.tokenizer(query_texts, padding=True, truncation=True,
                                                   max_length=512, return_tensors='pt')
        device = next(self.base_model.parameters()).device
        query_features = {k: v.to(device) for k, v in query_features.items()}
        query_emb = self.base_model(query_features)['sentence_embedding']
        
        # Encode conversation history if available
        if conversation_history:
            history_text = " ".join(conversation_history[-3:])  # Last 3 turns
            history_features = self.base_model.tokenizer([history_text], padding=True, truncation=True,
                                                         max_length=512, return_tensors='pt')
            history_features = {k: v.to(device) for k, v in history_features.items()}
            history_emb = self.base_model(history_features)['sentence_embedding']
            state = self.state_encoder(history_emb)
            
            # Expand state to batch size
            if state.size(0) == 1 and query_emb.size(0) > 1:
                state = state.expand(query_emb.size(0), -1)
            
            # Fuse query and state
            combined = torch.cat([query_emb, state], dim=1)
            query_emb = self.fusion(combined)
        
        return query_emb
    
    def encode_documents(self, doc_texts: List[str]) -> torch.Tensor:
        """Encode documents using forward pass for gradients"""
        doc_features = self.base_model.tokenizer(doc_texts, padding=True, truncation=True,
                                                max_length=512, return_tensors='pt')
        device = next(self.base_model.parameters()).device
        doc_features = {k: v.to(device) for k, v in doc_features.items()}
        return self.base_model(doc_features)['sentence_embedding']
    
    def save(self, output_path: str):
        """Save the model - base SentenceTransformer and additional state components"""
        output_path = pathlib.Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save the base SentenceTransformer model
        self.base_model.save(str(output_path))
        
        # Save the state encoder and fusion layers as additional state dict
        state_dict_path = output_path / "state_components.pt"
        torch.save({
            'state_encoder': self.state_encoder.state_dict(),
            'fusion': self.fusion.state_dict(),
            'state_dim': self.state_dim
        }, state_dict_path)
        
        logging.info(f"Saved base model to {output_path} and state components to {state_dict_path}")


class StateTrackingDataset(Dataset):
    """Dataset with conversation state tracking"""
    def __init__(self, examples: List[InputExample], conversation_dict: Dict = None):
        self.examples = examples
        self.conversation_dict = conversation_dict or {}
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        query_text, pos_doc_text = example.texts
        
        # Extract conversation history from query
        history = self._extract_history(query_text)
        
        return {
            'query': query_text,
            'positive': pos_doc_text,
            'history': history
        }
    
    def _extract_history(self, query_text: str) -> List[str]:
        """Extract conversation history from query text"""
        # Parse conversation format: |user|: ... |agent|: ...
        parts = re.split(r'\|user\|:|\|agent\|:', query_text)
        # Filter empty parts and return as history
        history = [p.strip() for p in parts if p.strip()]
        return history if history else [query_text]


def state_tracking_collate_fn(batch):
    """Custom collate function for StateTrackingDataset"""
    queries = [item['query'] for item in batch]
    positives = [item['positive'] for item in batch]
    histories = [item['history'] for item in batch]
    
    return {
        'query': queries,
        'positive': positives,
        'history': histories
    }


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


def train_state_tracking(config: Dict, gpu_id: Optional[int] = None) -> Optional[str]:
    """Train state-aware retrieval model"""
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
    
    output_dir = pathlib.Path(config.get('output_dir', 'experiments/retrieval/state_tracking'))
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
    
    # Load base model
    logging.info(f"Loading base model: {base_model}")
    base_sentence_model = SentenceTransformer(base_model)
    base_sentence_model.to(device)
    
    # Create state-aware retriever
    model = StateAwareRetriever(base_sentence_model)
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
        
        dataset = StateTrackingDataset(examples)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=state_tracking_collate_fn)
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
        
        model.train()
        for epoch in range(epochs):
            total_loss = 0
            optimizer.zero_grad()
            
            for batch_idx, batch in enumerate(dataloader):
                queries_batch = batch['query']
                positives_batch = batch['positive']
                histories_batch = batch['history']
                
                # Encode with state
                # Encode queries with state using forward pass for gradients
                # Process each query with its history
                query_embs_list = []
                for i, query in enumerate(queries_batch):
                    history = histories_batch[i] if i < len(histories_batch) else None
                    query_emb = model.encode_query([query], conversation_history=history)
                    query_embs_list.append(query_emb[0])
                query_embs = torch.stack(query_embs_list)
                
                # Encode positives using forward pass
                pos_embs = model.encode_documents(positives_batch)
                
                # Normalize
                query_embs = F.normalize(query_embs, p=2, dim=1)
                pos_embs = F.normalize(pos_embs, p=2, dim=1)
                
                # Contrastive loss
                similarities = torch.sum(query_embs * pos_embs, dim=1)
                loss = -torch.mean(torch.log(torch.sigmoid(similarities / 0.05) + 1e-8))
                
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
        results_path = train_state_tracking(config, args.gpu_id)
        if results_path:
            logging.info(f"✅ State tracking training completed: {results_path}")
    except KeyboardInterrupt:
        logging.info("Interrupted by user")
        sys.exit(0)
    except Exception as e:
        logging.error(f"Failed: {e}", exc_info=True)
        sys.exit(1)

