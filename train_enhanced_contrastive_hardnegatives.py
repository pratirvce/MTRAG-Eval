"""
Enhanced Contrastive Learning with Systematic Hard Negative Mining
Novel approach: Multiple hard negative sources (in-batch, BM25, adversarial, dynamic)
Expected: 0.50-0.54 nDCG@10
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
from torch.utils.data import DataLoader, Dataset
import torch
import torch.nn.functional as F
import signal
import sys
from datetime import datetime
try:
    from rank_bm25 import BM25Okapi
    BM25_AVAILABLE = True
except ImportError:
    BM25Okapi = None
    BM25_AVAILABLE = False
    logging.warning("rank_bm25 not available. BM25 hard negative mining will be disabled.")
import random

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

class HardNegativeDataset(Dataset):
    """Dataset with hard negative mining"""
    def __init__(self, examples: List[InputExample], model, corpus_dict: Dict, 
                 queries_dict: Dict, use_bm25: bool = True, use_inbatch: bool = True,
                 num_hard_negatives: int = 3):
        self.examples = examples
        self.model = model
        self.corpus_dict = corpus_dict
        self.queries_dict = queries_dict
        self.use_bm25 = use_bm25
        self.use_inbatch = use_inbatch
        self.num_hard_negatives = num_hard_negatives
        
        # Build BM25 index for hard negative mining
        if use_bm25 and BM25_AVAILABLE and corpus_dict:
            corpus_texts = [doc.get('text', '') for doc in corpus_dict.values()]
            tokenized_corpus = [doc.lower().split() for doc in corpus_texts]
            self.bm25 = BM25Okapi(tokenized_corpus)
            self.corpus_ids = list(corpus_dict.keys())
        else:
            self.bm25 = None
            self.corpus_ids = None
            if use_bm25 and not BM25_AVAILABLE:
                logging.warning("BM25 requested but rank_bm25 not available. Disabling BM25 hard negatives.")
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        query_text, pos_doc_text = example.texts
        
        # Get hard negatives
        hard_negatives = []
        
        # 1. In-batch hard negatives (will be handled in loss)
        # 2. BM25 hard negatives
        if self.bm25 and self.corpus_ids:
            query_tokens = query_text.lower().split()
            scores = self.bm25.get_scores(query_tokens)
            # Get top documents that are NOT the positive
            top_indices = np.argsort(scores)[::-1][:self.num_hard_negatives * 10]
            for i in top_indices:
                doc_id = self.corpus_ids[i]
                if doc_id in self.corpus_dict:
                    doc_text = self.corpus_dict[doc_id].get('text', '')
                    if doc_text != pos_doc_text:  # Not the positive
                        hard_negatives.append(doc_text)
                        if len(hard_negatives) >= self.num_hard_negatives:
                            break
        
        # 3. Random negatives (fallback)
        while len(hard_negatives) < self.num_hard_negatives:
            random_doc = random.choice(list(self.corpus_dict.values()))
            random_text = random_doc.get('text', '')
            if random_text != pos_doc_text:
                hard_negatives.append(random_text)
        
        return {
            'query': query_text,
            'positive': pos_doc_text,
            'negatives': hard_negatives[:self.num_hard_negatives]
        }

class EnhancedContrastiveLoss(torch.nn.Module):
    """Enhanced contrastive loss with hard negatives"""
    def __init__(self, temperature: float = 0.05):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, query_emb, pos_emb, neg_embs):
        # Ensure all embeddings are float dtype before normalization
        # F.normalize() requires float or complex dtype, not Long/int
        if query_emb.dtype != torch.float32 and query_emb.dtype != torch.float64:
            query_emb = query_emb.float()
        if pos_emb.dtype != torch.float32 and pos_emb.dtype != torch.float64:
            pos_emb = pos_emb.float()
        if neg_embs.dtype != torch.float32 and neg_embs.dtype != torch.float64:
            neg_embs = neg_embs.float()
        
        # Normalize embeddings
        query_emb = F.normalize(query_emb, p=2, dim=1)
        pos_emb = F.normalize(pos_emb, p=2, dim=1)
        neg_embs = F.normalize(neg_embs, p=2, dim=1)
        
        # Positive similarity
        pos_sim = torch.sum(query_emb * pos_emb, dim=1) / self.temperature
        
        # Negative similarities
        neg_sims = torch.matmul(query_emb, neg_embs.t()) / self.temperature
        
        # Combine positive and negatives
        logits = torch.cat([pos_sim.unsqueeze(1), neg_sims], dim=1)
        labels = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)
        
        loss = F.cross_entropy(logits, labels)
        return loss

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

def train_enhanced_contrastive(config: Dict, gpu_id: Optional[int] = None) -> Optional[str]:
    """Train with enhanced contrastive learning"""
    global shutdown_requested
    
    # CUDA_VISIBLE_DEVICES should already be set by the caller before importing this module
    # If torch is already initialized, we can't change CUDA_VISIBLE_DEVICES
    cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if cuda_visible:
        logging.info(f"CUDA_VISIBLE_DEVICES={cuda_visible} (set before torch import)")
    elif gpu_id is not None:
        # Fallback: try to set it (may not work if torch already initialized)
        logging.warning(f"CUDA_VISIBLE_DEVICES not set, attempting to set to {gpu_id} (may not work)")
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Explicitly set the device if CUDA is available
    if device == "cuda" and torch.cuda.is_available():
        # If CUDA_VISIBLE_DEVICES was set correctly, GPU 0 is the specified GPU
        try:
            torch.cuda.set_device(0)
            actual_gpu = torch.cuda.current_device()
            gpu_name = torch.cuda.get_device_name(actual_gpu)
            logging.info(f"Device: {device}, Using GPU {actual_gpu} ({gpu_name})")
            logging.info(f"CUDA_VISIBLE_DEVICES={cuda_visible}, GPU will appear as device 0 to PyTorch")
        except Exception as e:
            logging.warning(f"Could not set CUDA device: {e}")
            logging.info(f"Device: {device}")
    else:
        logging.info(f"Device: {device}")
    
    output_dir = pathlib.Path(config.get('output_dir', 'experiments/retrieval/enhanced_contrastive'))
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
    epochs = config.get('epochs', 3)
    batch_size = config.get('batch_size', 16)
    num_hard_negatives = config.get('num_hard_negatives', 3)
    gradient_accumulation_steps = config.get('gradient_accumulation_steps', 1)
    use_fp16 = config.get('use_fp16', False)
    
    # Load base model
    logging.info(f"Loading base model: {base_model}")
    model = SentenceTransformer(base_model)
    model.to(device)
    
    # Enable mixed precision training if requested
    if use_fp16 and device == "cuda":
        try:
            from torch.cuda.amp import autocast, GradScaler
            scaler = GradScaler()
            logging.info("Mixed precision training (FP16) enabled")
        except ImportError:
            logging.warning("FP16 not available, falling back to FP32")
            use_fp16 = False
            scaler = None
    else:
        scaler = None
    
    # Ensure all parameters require gradients for training
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
        
        # Create dataset with hard negatives
        dataset = HardNegativeDataset(
            examples, model, corpus, queries,
            use_bm25=True, use_inbatch=True,
            num_hard_negatives=num_hard_negatives
        )
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Training loop
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
        loss_fn = EnhancedContrastiveLoss(temperature=0.05)
        
        model.train()
        for epoch in range(epochs):
            total_loss = 0
            optimizer.zero_grad()  # Zero gradients at the start of epoch
            
            for batch_idx, batch in enumerate(dataloader):
                queries_batch = batch['query']
                positives_batch = batch['positive']
                negatives_batch = batch['negatives']
                
                # Ensure all queries have at least one negative
                # Filter out any that don't (shouldn't happen, but safety check)
                valid_indices = []
                for i, negs in enumerate(negatives_batch):
                    if len(negs) > 0:
                        valid_indices.append(i)
                    else:
                        logging.warning(f"Query {i} has no negatives, will skip this example")
                
                if len(valid_indices) == 0:
                    logging.warning("No valid examples in batch (all have no negatives), skipping")
                    continue
                
                if len(valid_indices) < len(queries_batch):
                    # Filter batch to only include valid examples
                    queries_batch = [queries_batch[i] for i in valid_indices]
                    positives_batch = [positives_batch[i] for i in valid_indices]
                    negatives_batch = [negatives_batch[i] for i in valid_indices]
                    logging.debug(f"Filtered batch from {len(batch['query'])} to {len(queries_batch)} valid examples")
                
                # Recalculate neg_counts after filtering (CRITICAL FIX)
                neg_counts = [len(negs) for negs in negatives_batch]
                total_negatives = sum(neg_counts)
                
                # Encode with gradient tracking - manually forward through modules
                # Skip normalization module (2) as it may detach gradients - normalize in loss instead
                query_features = model.tokenizer(queries_batch, padding=True, truncation=True, 
                                                 max_length=512, return_tensors='pt')
                query_features = {k: v.to(device) for k, v in query_features.items()}
                query_outputs = model._modules['0'](query_features)  # Transformer module
                # Handle BaseModelOutput or dict format
                # SentenceTransformer's Transformer module returns dict with 'token_embeddings' key
                if isinstance(query_outputs, dict):
                    # First try 'token_embeddings' (SentenceTransformer format)
                    query_embeddings = query_outputs.get('token_embeddings')
                    if query_embeddings is None:
                        # Fallback to 'last_hidden_state' (HuggingFace format)
                        query_embeddings = query_outputs.get('last_hidden_state')
                    if query_embeddings is None:
                        # Find 3D tensor (batch, seq_len, hidden_dim)
                        for key, value in query_outputs.items():
                            if isinstance(value, torch.Tensor) and value.dim() == 3:
                                query_embeddings = value
                                break
                elif hasattr(query_outputs, 'last_hidden_state'):
                    query_embeddings = query_outputs.last_hidden_state
                elif isinstance(query_outputs, tuple):
                    query_embeddings = query_outputs[0]  # First element is usually hidden states
                elif hasattr(query_outputs, 'shape'):
                    # It's already a tensor
                    query_embeddings = query_outputs
                else:
                    # Last resort: try to use the output directly
                    query_embeddings = query_outputs
                
                # Ensure we have a tensor (not a dict or None)
                if query_embeddings is None or isinstance(query_embeddings, dict) or not hasattr(query_embeddings, 'shape'):
                    # If we can't extract a tensor, log the actual output type for debugging
                    logging.error(f"Could not extract token embeddings from model output. Type: {type(query_outputs)}, Output keys: {list(query_outputs.keys()) if isinstance(query_outputs, dict) else 'N/A'}")
                    raise ValueError(f"Model output format not supported. Expected tensor, got {type(query_outputs)}. "
                                   f"If dict, expected 'last_hidden_state' key or tensor values.")
                
                # Ensure query_embeddings has the correct shape [batch, seq_len, hidden_dim]
                if query_embeddings.dim() == 1:
                    # If 1D, it's likely [hidden_dim] - need to add batch and seq_len dimensions
                    query_embeddings = query_embeddings.unsqueeze(0).unsqueeze(0)  # [1, 1, hidden_dim]
                elif query_embeddings.dim() == 2:
                    # If 2D, could be [batch, hidden_dim] or [seq_len, hidden_dim]
                    if query_embeddings.size(0) == len(queries_batch):
                        # Likely [batch, hidden_dim] - need to add seq_len dimension
                        query_embeddings = query_embeddings.unsqueeze(1)  # [batch, 1, hidden_dim]
                    else:
                        # Likely [seq_len, hidden_dim] - need to add batch dimension
                        query_embeddings = query_embeddings.unsqueeze(0)  # [1, seq_len, hidden_dim]
                elif query_embeddings.dim() == 3:
                    # Should be [batch, seq_len, hidden_dim] - verify batch size matches
                    if query_embeddings.size(0) != len(queries_batch):
                        logging.warning(f"Batch size mismatch: embeddings {query_embeddings.size(0)} vs queries {len(queries_batch)}")
                else:
                    raise ValueError(f"Unexpected tensor dimension: {query_embeddings.dim()}. Expected 1, 2, or 3 dimensions.")
                
                # Use model's forward method through modules 0 and 1 (skip normalization for gradient tracking)
                # Pass through pooling module
                pooling_input = {'token_embeddings': query_embeddings, 
                               'attention_mask': query_features['attention_mask']}
                pooling_output = model._modules['1'](pooling_input)
                
                # Pooling module returns dict with 'sentence_embedding' key
                # The value should be [batch_size, hidden_dim] where hidden_dim=768
                if not isinstance(pooling_output, dict):
                    raise ValueError(f"Expected pooling output to be dict, got {type(pooling_output)}")
                
                query_embs = pooling_output.get('sentence_embedding')
                if query_embs is None:
                    # Log available keys for debugging
                    available_keys = list(pooling_output.keys())
                    logging.error(f"Pooling output does not have 'sentence_embedding' key. Available keys: {available_keys}")
                    # Try to get any tensor value as fallback
                    query_embs = next((v for v in pooling_output.values() if isinstance(v, torch.Tensor)), None)
                    if query_embs is not None:
                        logging.warning(f"Using fallback tensor from pooling output. Shape: {query_embs.shape}")
                
                if query_embs is None:
                    raise ValueError(f"Could not extract sentence embedding from pooling output. Keys: {list(pooling_output.keys())}")
                
                # Ensure float dtype
                if query_embs.dtype != torch.float32 and query_embs.dtype != torch.float64:
                    query_embs = query_embs.float()
                
                # Ensure query_embs has correct shape [batch_size, hidden_dim]
                # Pooling should already give us [batch, hidden_dim], but handle edge cases
                if query_embs.dim() == 1:
                    query_embs = query_embs.unsqueeze(0)  # [1, hidden_dim]
                elif query_embs.dim() == 3:
                    # Should not happen, but if it does, take mean
                    logging.warning(f"Query embeddings are 3D: {query_embs.shape}, taking mean")
                    query_embs = query_embs.mean(dim=1)  # [batch, hidden_dim]
                elif query_embs.dim() != 2:
                    raise ValueError(f"Unexpected query_embs dimension: {query_embs.dim()}, shape: {query_embs.shape}")
                
                # Verify batch size and embedding dimension
                if query_embs.size(0) != len(queries_batch):
                    raise ValueError(f"Query batch size mismatch: {query_embs.size(0)} vs {len(queries_batch)}")
                
                # The embedding dimension should be 768 for BGE-base
                # If it's not, we have a serious problem
                expected_dim = 768
                if query_embs.size(1) != expected_dim:
                    # Log detailed info for debugging
                    logging.error(f"CRITICAL: Query embedding dimension is {query_embs.size(1)}, expected {expected_dim}")
                    logging.error(f"Query embeddings full shape: {query_embs.shape}")
                    logging.error(f"Query embeddings dtype: {query_embs.dtype}")
                    logging.error(f"Pooling output keys: {list(pooling_output.keys())}")
                    # Check if we accidentally got sequence length instead of embedding dim
                    if query_embs.size(1) < 100:  # Likely sequence length
                        raise ValueError(f"Query embedding appears to have wrong shape. Got dim={query_embs.size(1)} (looks like seq_len), expected {expected_dim}. Full shape: {query_embs.shape}. This suggests pooling did not work correctly.")
                    raise ValueError(f"Query embedding dimension mismatch: got {query_embs.size(1)}, expected {expected_dim}")
                
                pos_features = model.tokenizer(positives_batch, padding=True, truncation=True,
                                              max_length=512, return_tensors='pt')
                pos_features = {k: v.to(device) for k, v in pos_features.items()}
                pos_outputs = model._modules['0'](pos_features)
                # Handle BaseModelOutput or dict format
                # SentenceTransformer's Transformer module returns dict with 'token_embeddings' key
                if isinstance(pos_outputs, dict):
                    # First try 'token_embeddings' (SentenceTransformer format)
                    pos_embeddings = pos_outputs.get('token_embeddings')
                    if pos_embeddings is None:
                        # Fallback to 'last_hidden_state' (HuggingFace format)
                        pos_embeddings = pos_outputs.get('last_hidden_state')
                    if pos_embeddings is None:
                        # Find 3D tensor (batch, seq_len, hidden_dim)
                        for key, value in pos_outputs.items():
                            if isinstance(value, torch.Tensor) and value.dim() == 3:
                                pos_embeddings = value
                                break
                elif hasattr(pos_outputs, 'last_hidden_state'):
                    pos_embeddings = pos_outputs.last_hidden_state
                elif isinstance(pos_outputs, tuple):
                    pos_embeddings = pos_outputs[0]
                else:
                    pos_embeddings = pos_outputs
                
                if pos_embeddings is None:
                    raise ValueError(f"Could not extract hidden states from pos model output. Type: {type(pos_outputs)}")
                
                # Ensure pos_embeddings has the correct shape [batch, seq_len, hidden_dim]
                if pos_embeddings.dim() == 1:
                    pos_embeddings = pos_embeddings.unsqueeze(0).unsqueeze(0)  # [1, 1, hidden_dim]
                elif pos_embeddings.dim() == 2:
                    if pos_embeddings.size(0) == len(positives_batch):
                        pos_embeddings = pos_embeddings.unsqueeze(1)  # [batch, 1, hidden_dim]
                    else:
                        pos_embeddings = pos_embeddings.unsqueeze(0)  # [1, seq_len, hidden_dim]
                elif pos_embeddings.dim() == 3:
                    if pos_embeddings.size(0) != len(positives_batch):
                        logging.warning(f"Pos batch size mismatch: embeddings {pos_embeddings.size(0)} vs positives {len(positives_batch)}")
                
                # Pass through pooling module
                pooling_output = model._modules['1']({'token_embeddings': pos_embeddings,
                                                      'attention_mask': pos_features['attention_mask']})
                
                # Extract sentence embedding - handle both dict and tensor outputs
                if isinstance(pooling_output, dict):
                    pos_embs = pooling_output.get('sentence_embedding')
                    if pos_embs is None:
                        # Try other possible keys
                        pos_embs = pooling_output.get('embeddings') or pooling_output.get('pooled_output')
                        if pos_embs is None and len(pooling_output) > 0:
                            # Get first tensor value
                            pos_embs = next((v for v in pooling_output.values() if isinstance(v, torch.Tensor)), None)
                else:
                    pos_embs = pooling_output
                
                if pos_embs is None:
                    raise ValueError(f"Could not extract sentence embedding from pooling output. Type: {type(pooling_output)}, Keys: {list(pooling_output.keys()) if isinstance(pooling_output, dict) else 'N/A'}")
                
                # Ensure float dtype (normalize() requires float/complex dtype)
                if pos_embs.dtype != torch.float32 and pos_embs.dtype != torch.float64:
                    pos_embs = pos_embs.float()
                
                # Ensure pos_embs has correct shape [batch_size, hidden_dim]
                if pos_embs.dim() == 1:
                    pos_embs = pos_embs.unsqueeze(0)  # [1, hidden_dim]
                elif pos_embs.dim() == 3:
                    # If [batch, seq_len, hidden_dim], take mean pooling or first token
                    pos_embs = pos_embs.mean(dim=1)  # [batch, hidden_dim]
                elif pos_embs.dim() != 2:
                    raise ValueError(f"Unexpected pos_embs dimension: {pos_embs.dim()}, shape: {pos_embs.shape}")
                
                # Verify batch size matches and embedding dimensions match query_embs
                if pos_embs.size(0) != len(positives_batch):
                    logging.error(f"Pos batch size mismatch: {pos_embs.size(0)} vs {len(positives_batch)}")
                    raise ValueError(f"Pos embedding batch size {pos_embs.size(0)} doesn't match positives batch size {len(positives_batch)}")
                
                if pos_embs.size(0) != query_embs.size(0):
                    logging.error(f"Batch size mismatch between query and pos: {query_embs.size(0)} vs {pos_embs.size(0)}")
                    raise ValueError(f"Query batch size {query_embs.size(0)} doesn't match pos batch size {pos_embs.size(0)}")
                
                if pos_embs.size(1) != query_embs.size(1):
                    logging.error(f"Embedding dimension mismatch: query {query_embs.size(1)} vs pos {pos_embs.size(1)}")
                    raise ValueError(f"Query embedding dim {query_embs.size(1)} doesn't match pos embedding dim {pos_embs.size(1)}")
                
                # Encode negatives
                # Track how many negatives each query has (already calculated after filtering above)
                # neg_counts and total_negatives are already set after filtering
                
                # Log negative counts for debugging
                logging.debug(f"Batch {batch_idx}: {len(queries_batch)} queries, negative counts: {neg_counts}, total: {total_negatives}")
                
                # Check if all queries have negatives
                queries_without_negs = [i for i, count in enumerate(neg_counts) if count == 0]
                if queries_without_negs:
                    logging.warning(f"Queries {queries_without_negs} have no negatives. Adding random negatives for these queries.")
                    # Add random negatives for queries without negatives
                    for query_idx in queries_without_negs:
                        # Get a random document from corpus as negative
                        if corpus:
                            random_doc_id = random.choice(list(corpus.keys()))
                            random_doc = corpus[random_doc_id]
                            random_text = random_doc.get('text', '')
                            if random_text:
                                negatives_batch[query_idx] = [random_text]
                                neg_counts[query_idx] = 1
                                total_negatives += 1
                                logging.debug(f"Added random negative for query {query_idx}")
                
                # Ensure we have at least one negative per query after fixing
                if total_negatives == 0:
                    logging.warning("No negatives found in batch after fixes, skipping this batch")
                    continue
                
                # Update neg_counts after fixes
                neg_counts = [len(negs) for negs in negatives_batch]
                total_negatives = sum(neg_counts)
                
                # Process negatives in chunks if there are too many to avoid OOM
                max_negatives_per_batch = 16  # Process at most 16 negatives at a time
                neg_embs_list = []
                
                if total_negatives > max_negatives_per_batch:
                    # Process negatives in chunks
                    neg_texts = [neg for negs in negatives_batch for neg in negs]
                    for chunk_start in range(0, total_negatives, max_negatives_per_batch):
                        chunk_end = min(chunk_start + max_negatives_per_batch, total_negatives)
                        neg_chunk = neg_texts[chunk_start:chunk_end]
                        
                        neg_features = model.tokenizer(neg_chunk, padding=True, truncation=True,
                                                      max_length=512, return_tensors='pt')
                        neg_features = {k: v.to(device) for k, v in neg_features.items()}
                        neg_outputs = model._modules['0'](neg_features)
                        
                        # Extract embeddings for this chunk
                        # SentenceTransformer's Transformer module returns dict with 'token_embeddings' key
                        if isinstance(neg_outputs, dict):
                            # First try 'token_embeddings' (SentenceTransformer format)
                            neg_embeddings_chunk = neg_outputs.get('token_embeddings')
                            if neg_embeddings_chunk is None:
                                # Fallback to 'last_hidden_state' (HuggingFace format)
                                neg_embeddings_chunk = neg_outputs.get('last_hidden_state')
                            if neg_embeddings_chunk is None:
                                # Find 3D tensor (batch, seq_len, hidden_dim)
                                for key, value in neg_outputs.items():
                                    if isinstance(value, torch.Tensor) and value.dim() == 3:
                                        neg_embeddings_chunk = value
                                        break
                        elif hasattr(neg_outputs, 'last_hidden_state'):
                            neg_embeddings_chunk = neg_outputs.last_hidden_state
                        elif isinstance(neg_outputs, tuple):
                            neg_embeddings_chunk = neg_outputs[0]
                        else:
                            neg_embeddings_chunk = neg_outputs
                        
                        if neg_embeddings_chunk is None:
                            raise ValueError(f"Could not extract hidden states from neg model output chunk")
                        
                        # Ensure correct shape
                        if neg_embeddings_chunk.dim() == 1:
                            neg_embeddings_chunk = neg_embeddings_chunk.unsqueeze(0).unsqueeze(0)
                        elif neg_embeddings_chunk.dim() == 2:
                            if neg_embeddings_chunk.size(0) == len(neg_chunk):
                                neg_embeddings_chunk = neg_embeddings_chunk.unsqueeze(1)
                            else:
                                neg_embeddings_chunk = neg_embeddings_chunk.unsqueeze(0)
                        
                        # Pass through pooling module
                        pooling_output = model._modules['1']({'token_embeddings': neg_embeddings_chunk,
                                                              'attention_mask': neg_features['attention_mask']})
                        
                        # Extract sentence embedding
                        if isinstance(pooling_output, dict):
                            neg_embs_chunk = pooling_output.get('sentence_embedding')
                            if neg_embs_chunk is None:
                                neg_embs_chunk = pooling_output.get('embeddings') or pooling_output.get('pooled_output')
                                if neg_embs_chunk is None and len(pooling_output) > 0:
                                    neg_embs_chunk = next((v for v in pooling_output.values() if isinstance(v, torch.Tensor)), None)
                        else:
                            neg_embs_chunk = pooling_output
                        
                        if neg_embs_chunk is None:
                            raise ValueError(f"Could not extract sentence embedding from neg chunk pooling output")
                        
                        if neg_embs_chunk.dtype != torch.float32 and neg_embs_chunk.dtype != torch.float64:
                            neg_embs_chunk = neg_embs_chunk.float()
                        
                        # Ensure correct shape [chunk_size, hidden_dim]
                        if neg_embs_chunk.dim() == 1:
                            neg_embs_chunk = neg_embs_chunk.unsqueeze(0)
                        elif neg_embs_chunk.dim() == 3:
                            neg_embs_chunk = neg_embs_chunk.mean(dim=1)  # [chunk_size, hidden_dim]
                        
                        neg_embs_list.append(neg_embs_chunk)
                        
                        # Clear cache after each chunk
                        if device == "cuda":
                            torch.cuda.empty_cache()
                    
                    # Concatenate all chunks
                    neg_embs = torch.cat(neg_embs_list, dim=0)
                else:
                    # Process all negatives at once (original code path)
                    neg_texts = [neg for negs in negatives_batch for neg in negs]
                    neg_features = model.tokenizer(neg_texts, padding=True, truncation=True,
                                                  max_length=512, return_tensors='pt')
                    neg_features = {k: v.to(device) for k, v in neg_features.items()}
                    neg_outputs = model._modules['0'](neg_features)
                    
                    # Handle BaseModelOutput or dict format
                    # SentenceTransformer's Transformer module returns dict with 'token_embeddings' key
                    if isinstance(neg_outputs, dict):
                        # First try 'token_embeddings' (SentenceTransformer format)
                        neg_embeddings = neg_outputs.get('token_embeddings')
                        if neg_embeddings is None:
                            # Fallback to 'last_hidden_state' (HuggingFace format)
                            neg_embeddings = neg_outputs.get('last_hidden_state')
                        if neg_embeddings is None:
                            # Find 3D tensor (batch, seq_len, hidden_dim)
                            for key, value in neg_outputs.items():
                                if isinstance(value, torch.Tensor) and value.dim() == 3:
                                    neg_embeddings = value
                                    break
                    elif hasattr(neg_outputs, 'last_hidden_state'):
                        neg_embeddings = neg_outputs.last_hidden_state
                    elif isinstance(neg_outputs, tuple):
                        neg_embeddings = neg_outputs[0]
                    else:
                        neg_embeddings = neg_outputs
                    
                    if neg_embeddings is None:
                        raise ValueError(f"Could not extract hidden states from neg model output. Type: {type(neg_outputs)}")
                    
                    # Ensure neg_embeddings has the correct shape [batch, seq_len, hidden_dim]
                    if neg_embeddings.dim() == 1:
                        neg_embeddings = neg_embeddings.unsqueeze(0).unsqueeze(0)  # [1, 1, hidden_dim]
                    elif neg_embeddings.dim() == 2:
                        if neg_embeddings.size(0) == total_negatives:
                            neg_embeddings = neg_embeddings.unsqueeze(1)  # [total_negatives, 1, hidden_dim]
                        else:
                            neg_embeddings = neg_embeddings.unsqueeze(0)  # [1, seq_len, hidden_dim]
                    elif neg_embeddings.dim() == 3:
                        if neg_embeddings.size(0) != total_negatives:
                            logging.warning(f"Neg batch size mismatch: embeddings {neg_embeddings.size(0)} vs negatives {total_negatives}")
                    
                    # Pass through pooling module
                    pooling_output = model._modules['1']({'token_embeddings': neg_embeddings,
                                                         'attention_mask': neg_features['attention_mask']})
                    
                    # Extract sentence embedding
                    if isinstance(pooling_output, dict):
                        neg_embs = pooling_output.get('sentence_embedding')
                        if neg_embs is None:
                            neg_embs = pooling_output.get('embeddings') or pooling_output.get('pooled_output')
                            if neg_embs is None and len(pooling_output) > 0:
                                neg_embs = next((v for v in pooling_output.values() if isinstance(v, torch.Tensor)), None)
                    else:
                        neg_embs = pooling_output
                    
                    if neg_embs is None:
                        raise ValueError(f"Could not extract sentence embedding from neg pooling output")
                    
                    # Ensure float dtype (mean() requires float/complex dtype)
                    if neg_embs.dtype != torch.float32 and neg_embs.dtype != torch.float64:
                        neg_embs = neg_embs.float()
                    
                    # Ensure correct shape [total_negatives, hidden_dim]
                    if neg_embs.dim() == 1:
                        neg_embs = neg_embs.unsqueeze(0)
                    elif neg_embs.dim() == 3:
                        neg_embs = neg_embs.mean(dim=1)  # [total_negatives, hidden_dim]
                
                # Handle variable numbers of negatives per query
                # Split neg_embs back into per-query groups (works for both chunked and non-chunked paths)
                neg_embs_list = []
                start_idx = 0
                
                # CRITICAL FIX: Ensure neg_counts matches the filtered queries_batch length FIRST
                # This must be done BEFORE checking neg_embs size to avoid errors
                # This can happen if filtering changed the batch size but neg_counts wasn't updated
                if len(neg_counts) != len(queries_batch):
                    logging.warning(f"neg_counts length ({len(neg_counts)}) doesn't match queries_batch length ({len(queries_batch)}). Recalculating...")
                    # Recalculate neg_counts to match the filtered batch
                    neg_counts = [len(negs) for negs in negatives_batch]
                    total_negatives = sum(neg_counts)
                    logging.info(f"Recalculated neg_counts: {neg_counts}, total: {total_negatives}")
                
                # Verify we have enough embeddings (after recalculation)
                if neg_embs.size(0) != total_negatives:
                    logging.error(f"Neg embeddings count mismatch: got {neg_embs.size(0)}, expected {total_negatives}")
                    logging.error(f"Neg counts per query: {neg_counts}, sum: {sum(neg_counts)}")
                    raise ValueError(f"Neg embeddings count {neg_embs.size(0)} doesn't match total negatives {total_negatives}")
                
                for i, count in enumerate(neg_counts):
                    if i >= len(queries_batch):
                        logging.warning(f"Skipping neg_counts[{i}] as it's beyond queries_batch length")
                        break
                    
                    if count == 0:
                        # If no negatives for this query, use zero embedding (shouldn't happen due to earlier check)
                        logging.warning(f"Query {i} has 0 negatives, using zero embedding")
                        zero_emb = torch.zeros(neg_embs.size(1), device=neg_embs.device, dtype=neg_embs.dtype)
                        neg_embs_list.append(zero_emb)
                        continue
                    
                    end_idx = start_idx + count
                    if end_idx > neg_embs.size(0):
                        logging.error(f"Index out of bounds: start_idx={start_idx}, end_idx={end_idx}, neg_embs.size(0)={neg_embs.size(0)}")
                        raise ValueError(f"Cannot slice neg_embs: start={start_idx}, end={end_idx}, size={neg_embs.size(0)}")
                    
                    query_negs = neg_embs[start_idx:end_idx]  # [count, hidden_dim]
                    # Ensure float dtype before averaging (mean() requires float/complex dtype)
                    if query_negs.dtype != torch.float32 and query_negs.dtype != torch.float64:
                        query_negs = query_negs.float()
                    # Average the negatives for this query
                    query_neg_avg = query_negs.mean(dim=0)  # [hidden_dim]
                    neg_embs_list.append(query_neg_avg)
                    start_idx = end_idx
                
                # Verify we processed all queries
                if len(neg_embs_list) != len(queries_batch):
                    logging.error(f"Processed {len(neg_embs_list)} negative embeddings but have {len(queries_batch)} queries")
                    logging.error(f"Neg counts: {neg_counts}, total: {sum(neg_counts)}")
                    logging.error(f"queries_batch length: {len(queries_batch)}, negatives_batch length: {len(negatives_batch)}")
                    # Try to fix by truncating or padding
                    if len(neg_embs_list) > len(queries_batch):
                        logging.warning(f"Truncating neg_embs_list from {len(neg_embs_list)} to {len(queries_batch)}")
                        neg_embs_list = neg_embs_list[:len(queries_batch)]
                    elif len(neg_embs_list) < len(queries_batch):
                        logging.warning(f"Padding neg_embs_list from {len(neg_embs_list)} to {len(queries_batch)}")
                        zero_emb = torch.zeros(neg_embs.size(1), device=neg_embs.device, dtype=neg_embs.dtype)
                        while len(neg_embs_list) < len(queries_batch):
                            neg_embs_list.append(zero_emb)
                
                # Stack into [batch_size, hidden_dim]
                neg_embs = torch.stack(neg_embs_list, dim=0)
                
                # Final verification before loss computation
                if neg_embs.size(0) != query_embs.size(0):
                    logging.error(f"Neg batch size {neg_embs.size(0)} doesn't match query batch size {query_embs.size(0)}")
                    raise ValueError(f"Neg batch size {neg_embs.size(0)} doesn't match query batch size {query_embs.size(0)}")
                
                if neg_embs.size(1) != query_embs.size(1):
                    logging.error(f"Neg embedding dim {neg_embs.size(1)} doesn't match query embedding dim {query_embs.size(1)}")
                    raise ValueError(f"Neg embedding dim {neg_embs.size(1)} doesn't match query embedding dim {query_embs.size(1)}")
                
                # Compute loss with mixed precision if enabled
                if use_fp16 and scaler is not None:
                    with autocast():
                        loss = loss_fn(query_embs, pos_embs, neg_embs)
                        loss = loss / gradient_accumulation_steps  # Scale loss for accumulation
                    
                    # Scale loss and backward
                    scaled_loss = scaler.scale(loss)
                    scaled_loss.backward()
                    
                    # Accumulate gradients
                    if (batch_idx + 1) % gradient_accumulation_steps == 0:
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad()
                else:
                    # Standard FP32 training
                    loss = loss_fn(query_embs, pos_embs, neg_embs)
                    loss = loss / gradient_accumulation_steps  # Scale loss for accumulation
                    loss.backward()
                    
                    # Accumulate gradients
                    if (batch_idx + 1) % gradient_accumulation_steps == 0:
                        optimizer.step()
                        optimizer.zero_grad()
                
                total_loss += loss.item() * gradient_accumulation_steps  # Scale back for logging
                
                # Clear CUDA cache periodically to free memory
                if device == "cuda" and (batch_idx + 1) % 10 == 0:
                    torch.cuda.empty_cache()
            
            logging.info(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader):.4f}")
    
    # Evaluate
    logging.info("\nEvaluating trained model...")
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
            
            # Save checkpoint
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
    
    # Final results
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
        results_path = train_enhanced_contrastive(config, args.gpu_id)
        if results_path:
            logging.info(f"✅ Enhanced contrastive learning completed: {results_path}")
    except KeyboardInterrupt:
        logging.info("Interrupted by user")
        sys.exit(0)
    except Exception as e:
        logging.error(f"Failed: {e}", exc_info=True)
        sys.exit(1)

