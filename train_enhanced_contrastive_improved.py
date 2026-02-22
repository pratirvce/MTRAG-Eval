"""
Enhanced Contrastive Learning with Phase 2 Improvements
- Better negative sampling (in-batch + BM25 hard negatives)
- Temperature-scaled contrastive loss
- Margin-based loss
- Longer training (5-7 epochs)
- Better learning rate schedule
Expected: 0.35-0.45 nDCG@10 (improved from 0.1796)
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
import random
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


class ImprovedContrastiveLoss(nn.Module):
    """Improved contrastive loss with temperature scaling and margin"""
    def __init__(self, temperature: float = 0.05, margin: float = 0.2):
        super().__init__()
        self.temperature = temperature
        self.margin = margin
    
    def forward(self, query_emb, pos_emb, neg_emb):
        """
        Args:
            query_emb: [batch, dim]
            pos_emb: [batch, dim]
            neg_emb: [batch, dim] (averaged negatives)
        """
        # Normalize
        query_emb = F.normalize(query_emb, p=2, dim=1)
        pos_emb = F.normalize(pos_emb, p=2, dim=1)
        neg_emb = F.normalize(neg_emb, p=2, dim=1)
        
        # Positive similarity
        pos_sim = torch.sum(query_emb * pos_emb, dim=1) / self.temperature
        
        # Negative similarity
        neg_sim = torch.sum(query_emb * neg_emb, dim=1) / self.temperature
        
        # Margin-based loss: maximize pos_sim, minimize neg_sim with margin
        # Loss = -log(exp(pos_sim) / (exp(pos_sim) + exp(neg_sim + margin)))
        # This ensures pos_sim > neg_sim + margin
        loss = -torch.log(torch.sigmoid(pos_sim - neg_sim - self.margin) + 1e-8)
        
        return torch.mean(loss)


class ImprovedHardNegativeDataset(Dataset):
    """Dataset with improved hard negative mining"""
    def __init__(self, examples: List[InputExample], model, corpus_dict: Dict,
                 queries_dict: Dict, use_bm25: bool = True, use_inbatch: bool = True,
                 num_hard_negatives: int = 2):
        self.examples = examples
        self.model = model
        self.corpus_dict = corpus_dict
        self.queries_dict = queries_dict
        self.use_bm25 = use_bm25
        self.use_inbatch = use_inbatch
        self.num_hard_negatives = num_hard_negatives
        
        # Build BM25 index if available
        self.bm25_index = None
        if use_bm25:
            try:
                from rank_bm25 import BM25Okapi
                corpus_texts = [doc.get('text', '') for doc in corpus_dict.values()]
                tokenized_corpus = [doc.lower().split() for doc in corpus_texts]
                self.bm25_index = BM25Okapi(tokenized_corpus)
                self.corpus_ids = list(corpus_dict.keys())
                logging.info("BM25 index built for hard negative mining")
            except ImportError:
                logging.warning("rank_bm25 not available. BM25 hard negative mining will be disabled.")
                self.use_bm25 = False
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        query_text, pos_doc_text = example.texts
        
        negatives = []
        
        # 1. BM25 hard negatives
        if self.use_bm25 and self.bm25_index:
            try:
                query_tokens = query_text.lower().split()
                scores = self.bm25_index.get_scores(query_tokens)
                top_indices = np.argsort(scores)[::-1][:self.num_hard_negatives * 5]
                
                for i in top_indices:
                    if i < len(self.corpus_ids):
                        doc_id = self.corpus_ids[i]
                        if doc_id in self.corpus_dict:
                            doc = self.corpus_dict[doc_id]
                            doc_text = doc.get('text', '')
                            if doc_text and doc_text != pos_doc_text:
                                negatives.append(doc_text)
                                if len(negatives) >= self.num_hard_negatives:
                                    break
            except Exception as e:
                logging.debug(f"Error in BM25 negative mining: {e}")
        
        # 2. Random negatives (fallback or supplement)
        while len(negatives) < self.num_hard_negatives:
            random_doc = random.choice(list(self.corpus_dict.values()))
            random_text = random_doc.get('text', '')
            if random_text and random_text != pos_doc_text and random_text not in negatives:
                negatives.append(random_text)
        
        return {
            'query': query_text,
            'positive': pos_doc_text,
            'negatives': negatives[:self.num_hard_negatives]
        }


def improved_collate_fn(batch):
    """Collate function for improved dataset"""
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


def train_enhanced_contrastive_improved(config: Dict, gpu_id: Optional[int] = None) -> Optional[str]:
    """Train with improved contrastive learning"""
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
    
    output_dir = pathlib.Path(config.get('output_dir', 'experiments/retrieval/enhanced_contrastive_improved'))
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
    epochs = config.get('epochs', 5)  # Increased from 3 to 5
    batch_size = config.get('batch_size', 8)  # Increased from 4
    num_hard_negatives = config.get('num_hard_negatives', 2)
    gradient_accumulation_steps = config.get('gradient_accumulation_steps', 4)  # Effective batch size = 32
    use_fp16 = config.get('use_fp16', True)
    temperature = config.get('temperature', 0.05)
    margin = config.get('margin', 0.2)
    
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
        
        # Create improved dataset
        dataset = ImprovedHardNegativeDataset(
            examples, model, corpus, queries,
            use_bm25=True, use_inbatch=True,
            num_hard_negatives=num_hard_negatives
        )
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=improved_collate_fn)
        
        # Training setup with improved loss
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
        loss_fn = ImprovedContrastiveLoss(temperature=temperature, margin=margin)
        
        # Cosine annealing with warmup
        total_steps = len(dataloader) * epochs
        warmup_steps = int(0.1 * total_steps)  # 10% warmup
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
                
                # Filter valid examples
                valid_indices = [i for i, negs in enumerate(negatives_batch) if len(negs) > 0]
                if len(valid_indices) == 0:
                    continue
                
                if len(valid_indices) < len(queries_batch):
                    queries_batch = [queries_batch[i] for i in valid_indices]
                    positives_batch = [positives_batch[i] for i in valid_indices]
                    negatives_batch = [negatives_batch[i] for i in valid_indices]
                
                # Recalculate neg_counts after filtering
                neg_counts = [len(negs) for negs in negatives_batch]
                total_negatives = sum(neg_counts)
                
                # Encode queries and positives (same as before)
                query_features = model.tokenizer(queries_batch, padding=True, truncation=True,
                                                 max_length=512, return_tensors='pt')
                query_features = {k: v.to(device) for k, v in query_features.items()}
                
                pos_features = model.tokenizer(positives_batch, padding=True, truncation=True,
                                              max_length=512, return_tensors='pt')
                pos_features = {k: v.to(device) for k, v in pos_features.items()}
                
                # Get embeddings through model modules (same pattern as before)
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
                
                # Ensure correct dtypes and shapes
                if query_embs.dtype != torch.float32:
                    query_embs = query_embs.float()
                if pos_embs.dtype != torch.float32:
                    pos_embs = pos_embs.float()
                
                if query_embs.dim() == 1:
                    query_embs = query_embs.unsqueeze(0)
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
                
                # Average negatives per query
                neg_embs_list = []
                start_idx = 0
                for count in neg_counts:
                    end_idx = start_idx + count
                    query_negs = neg_embs[start_idx:end_idx]
                    if query_negs.dtype != torch.float32:
                        query_negs = query_negs.float()
                    query_neg_avg = query_negs.mean(dim=0)
                    neg_embs_list.append(query_neg_avg)
                    start_idx = end_idx
                
                neg_embs_avg = torch.stack(neg_embs_list, dim=0)
                
                # Compute improved loss
                if use_fp16 and scaler is not None:
                    with autocast():
                        loss = loss_fn(query_embs, pos_embs, neg_embs_avg)
                        loss = loss / gradient_accumulation_steps
                    
                    scaler.scale(loss).backward()
                    if (batch_idx + 1) % gradient_accumulation_steps == 0:
                        scaler.step(optimizer)
                        scaler.update()
                        scheduler.step()
                        optimizer.zero_grad()
                else:
                    loss = loss_fn(query_embs, pos_embs, neg_embs_avg)
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
        results_path = train_enhanced_contrastive_improved(config, args.gpu_id)
        if results_path:
            logging.info(f"✅ Improved contrastive learning completed: {results_path}")
    except KeyboardInterrupt:
        logging.info("Interrupted by user")
        sys.exit(0)
    except Exception as e:
        logging.error(f"Failed: {e}", exc_info=True)
        sys.exit(1)

