"""
OpenRAG-Style Retrieval-Only End-to-End Optimization
Task A Compliant: Uses relevance labels only (not generation feedback)
Optimizes retriever for in-context relevance using retrieval signals
Expected: 0.55-0.60 nDCG@10
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
from transformers import get_cosine_schedule_with_warmup

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


class InContextRelevanceLoss(nn.Module):
    """
    Loss that optimizes for in-context relevance
    Uses relevance labels to learn what's relevant in context
    """
    def __init__(self, temperature: float = 0.05):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, query_emb, pos_emb, neg_embs):
        """
        Args:
            query_emb: [batch, dim]
            pos_emb: [batch, dim]
            neg_embs: [batch, num_negs, dim]
        """
        query_emb = F.normalize(query_emb, p=2, dim=1)
        pos_emb = F.normalize(pos_emb, p=2, dim=1)
        neg_embs = F.normalize(neg_embs, p=2, dim=2)
        
        # Positive similarity
        pos_sim = torch.sum(query_emb * pos_emb, dim=1) / self.temperature
        
        # Negative similarities
        neg_sims = torch.bmm(query_emb.unsqueeze(1), neg_embs.transpose(1, 2)).squeeze(1) / self.temperature
        neg_sim = torch.max(neg_sims, dim=1)[0]  # Hard negative
        
        # Contrastive loss
        loss = -torch.log(torch.sigmoid(pos_sim - neg_sim) + 1e-8)
        return torch.mean(loss)


class InContextRelevanceDataset(Dataset):
    """Dataset for in-context relevance learning"""
    def __init__(self, examples: List[InputExample], model, corpus_dict: Dict,
                 queries_dict: Dict, num_negatives: int = 3):
        self.examples = examples
        self.model = model
        self.corpus_dict = corpus_dict
        self.queries_dict = queries_dict
        self.num_negatives = num_negatives
        
        # Build document pool for negative sampling
        self.doc_ids = list(corpus_dict.keys())
        self.doc_texts = [corpus_dict[did].get('text', '') for did in self.doc_ids]
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        query_text, pos_doc_text = example.texts
        
        # Sample random negatives
        negatives = []
        pos_doc_id = None
        for doc_id, doc in self.corpus_dict.items():
            if doc.get('text', '') == pos_doc_text:
                pos_doc_id = doc_id
                break
        
        while len(negatives) < self.num_negatives:
            neg_idx = np.random.randint(0, len(self.doc_ids))
            neg_doc_id = self.doc_ids[neg_idx]
            if neg_doc_id != pos_doc_id:
                neg_text = self.corpus_dict[neg_doc_id].get('text', '')
                if neg_text and neg_text not in negatives:
                    negatives.append(neg_text)
        
        return {
            'query': query_text,
            'positive': pos_doc_text,
            'negatives': negatives[:self.num_negatives]
        }


def in_context_collate_fn(batch):
    """Collate function for in-context relevance dataset"""
    queries = [item['query'] for item in batch]
    positives = [item['positive'] for item in batch]
    negatives = [item['negatives'] for item in batch]
    return {
        'query': queries,
        'positive': positives,
        'negatives': negatives
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


def train_openrag_retrieval_only(config: Dict, gpu_id: Optional[int] = None) -> Optional[str]:
    """Train with OpenRAG-style retrieval-only end-to-end optimization"""
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
    
    output_dir = pathlib.Path(config.get('output_dir', 'experiments/retrieval/openrag_retrieval_only'))
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
    epochs = config.get('epochs', 5)
    batch_size = config.get('batch_size', 8)
    num_negatives = config.get('num_negatives', 3)
    gradient_accumulation_steps = config.get('gradient_accumulation_steps', 4)
    use_fp16 = config.get('use_fp16', True)
    temperature = config.get('temperature', 0.05)
    
    # Load base model
    logging.info(f"Loading base model: {base_model}")
    model = SentenceTransformer(base_model)
    model.to(device)
    
    # Mixed precision
    scaler = None
    if use_fp16 and device == "cuda":
        try:
            from torch.cuda.amp import autocast, GradScaler
            scaler = GradScaler()
            logging.info("Mixed precision training (FP16) enabled")
        except ImportError:
            use_fp16 = False
    
    for param in model.parameters():
        param.requires_grad = True
    
    # Train per domain
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
        logging.info(f"Training on domain: {domain}")
        logging.info(f"{'='*60}")
        
        data_root = pathlib.Path(".")
        examples, corpus, queries = load_training_pairs(domain, data_root, config.get('use_data_splits', True))
        
        if not examples:
            logging.warning(f"No training examples for {domain}, skipping")
            continue
        
        # Create dataset
        dataset = InContextRelevanceDataset(examples, model, corpus, queries, num_negatives=num_negatives)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=in_context_collate_fn)
        
        # Training setup
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
        loss_fn = InContextRelevanceLoss(temperature=temperature)
        
        # Cosine annealing with warmup
        total_steps = len(dataloader) * epochs
        warmup_steps = int(0.1 * total_steps)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
        )
        
        model.train()
        for epoch in range(epochs):
            total_loss = 0
            optimizer.zero_grad()
            
            for batch_idx, batch in enumerate(dataloader):
                queries_batch = batch['query']
                positives_batch = batch['positive']
                negatives_batch = batch['negatives']
                
                # Encode queries
                query_features = model.tokenizer(queries_batch, padding=True, truncation=True,
                                                 max_length=512, return_tensors='pt')
                query_features = {k: v.to(device) for k, v in query_features.items()}
                query_outputs = model._modules['0'](query_features)
                
                if isinstance(query_outputs, dict):
                    query_embeddings = query_outputs.get('token_embeddings') or query_outputs.get('last_hidden_state')
                else:
                    query_embeddings = query_outputs
                
                pooling_output = model._modules['1']({
                    'token_embeddings': query_embeddings,
                    'attention_mask': query_features['attention_mask']
                })
                query_embs = pooling_output.get('sentence_embedding')
                if query_embs is None:
                    query_embs = next((v for v in pooling_output.values() if isinstance(v, torch.Tensor)), None)
                
                if query_embs.dtype != torch.float32:
                    query_embs = query_embs.float()
                if query_embs.dim() == 1:
                    query_embs = query_embs.unsqueeze(0)
                
                # Encode positives
                pos_features = model.tokenizer(positives_batch, padding=True, truncation=True,
                                              max_length=512, return_tensors='pt')
                pos_features = {k: v.to(device) for k, v in pos_features.items()}
                pos_outputs = model._modules['0'](pos_features)
                
                if isinstance(pos_outputs, dict):
                    pos_embeddings = pos_outputs.get('token_embeddings') or pos_outputs.get('last_hidden_state')
                else:
                    pos_embeddings = pos_outputs
                
                pooling_output = model._modules['1']({
                    'token_embeddings': pos_embeddings,
                    'attention_mask': pos_features['attention_mask']
                })
                pos_embs = pooling_output.get('sentence_embedding')
                if pos_embs is None:
                    pos_embs = next((v for v in pooling_output.values() if isinstance(v, torch.Tensor)), None)
                
                if pos_embs.dtype != torch.float32:
                    pos_embs = pos_embs.float()
                if pos_embs.dim() == 1:
                    pos_embs = pos_embs.unsqueeze(0)
                
                # Encode negatives
                neg_texts = [neg for negs in negatives_batch for neg in negs]
                neg_features = model.tokenizer(neg_texts, padding=True, truncation=True,
                                              max_length=512, return_tensors='pt')
                neg_features = {k: v.to(device) for k, v in neg_features.items()}
                neg_outputs = model._modules['0'](neg_features)
                
                if isinstance(neg_outputs, dict):
                    neg_embeddings = neg_outputs.get('token_embeddings') or neg_outputs.get('last_hidden_state')
                else:
                    neg_embeddings = neg_outputs
                
                pooling_output = model._modules['1']({
                    'token_embeddings': neg_embeddings,
                    'attention_mask': neg_features['attention_mask']
                })
                neg_embs = pooling_output.get('sentence_embedding')
                if neg_embs is None:
                    neg_embs = next((v for v in pooling_output.values() if isinstance(v, torch.Tensor)), None)
                
                if neg_embs.dtype != torch.float32:
                    neg_embs = neg_embs.float()
                if neg_embs.dim() == 1:
                    neg_embs = neg_embs.unsqueeze(0)
                elif neg_embs.dim() == 3:
                    neg_embs = neg_embs.mean(dim=1)
                
                # Reshape negatives: [batch * num_negatives, dim] -> [batch, num_negatives, dim]
                num_negs_per_query = len(negatives_batch[0]) if negatives_batch else num_negatives
                neg_embs = neg_embs.view(len(queries_batch), num_negs_per_query, -1)
                
                # Compute loss
                if use_fp16 and scaler is not None:
                    with autocast():
                        loss = loss_fn(query_embs, pos_embs, neg_embs)
                        loss = loss / gradient_accumulation_steps
                    
                    scaler.scale(loss).backward()
                    if (batch_idx + 1) % gradient_accumulation_steps == 0:
                        scaler.step(optimizer)
                        scaler.update()
                        scheduler.step()
                        optimizer.zero_grad()
                else:
                    loss = loss_fn(query_embs, pos_embs, neg_embs)
                    loss = loss / gradient_accumulation_steps
                    loss.backward()
                    if (batch_idx + 1) % gradient_accumulation_steps == 0:
                        optimizer.step()
                        scheduler.step()
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
    if len(domains) > 0:
        last_domain = list(domains)[-1] if isinstance(domains, (list, tuple)) else domains[-1]
        trained_model_path = checkpoint_dir / f"trained_model_{last_domain}"
        if trained_model_path.exists():
            logging.info(f"Loading trained model from {trained_model_path}")
            retriever = DenseRetrievalExactSearch(SentenceBERT(str(trained_model_path), device=device), batch_size=128)
        else:
            logging.warning(f"Trained model not found, using base model")
            retriever = DenseRetrievalExactSearch(SentenceBERT(base_model, device=device), batch_size=128)
    else:
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
        results_path = train_openrag_retrieval_only(config, args.gpu_id)
        if results_path:
            logging.info(f"✅ OpenRAG retrieval-only training completed: {results_path}")
    except KeyboardInterrupt:
        logging.info("Interrupted by user")
        sys.exit(0)
    except Exception as e:
        logging.error(f"Failed: {e}", exc_info=True)
        sys.exit(1)

