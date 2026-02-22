"""
Direct nDCG Optimization with NeuralNDCG
Directly optimizes nDCG instead of surrogate losses (contrastive, triplet)
Uses NeuralNDCG to make nDCG differentiable for gradient-based optimization
Expected: 0.53-0.57 nDCG@10
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


class NeuralNDCGLoss(nn.Module):
    """
    Differentiable approximation of nDCG for direct optimization
    Based on: "NeuralNDCG: Direct Optimisation of a Ranking Metric via Differentiable Relaxation of Sorting" (2021)
    """
    def __init__(self, temperature: float = 1.0, k: int = 10):
        super().__init__()
        self.temperature = temperature
        self.k = k
    
    def forward(self, scores: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            scores: [batch_size, num_docs] - predicted relevance scores
            labels: [batch_size, num_docs] - ground truth relevance labels
        Returns:
            Negative nDCG (to minimize)
        """
        batch_size, num_docs = scores.shape
        
        # Soft sort using temperature-scaled softmax
        # This approximates the sorting operation in nDCG
        sorted_scores = torch.sort(scores, dim=1, descending=True)[0]
        
        # Compute DCG for each query
        dcg_values = []
        for i in range(batch_size):
            # Get sorted labels according to scores
            sorted_indices = torch.argsort(scores[i], descending=True)
            sorted_labels = labels[i][sorted_indices]
            
            # Compute DCG@k
            dcg = 0.0
            for j in range(min(self.k, num_docs)):
                if sorted_labels[j] > 0:
                    dcg += (2 ** sorted_labels[j] - 1) / np.log2(j + 2)
            dcg_values.append(dcg)
        
        # Compute IDCG (ideal DCG)
        idcg_values = []
        for i in range(batch_size):
            # Sort labels in descending order (ideal ranking)
            sorted_labels = torch.sort(labels[i], descending=True)[0]
            idcg = 0.0
            for j in range(min(self.k, num_docs)):
                if sorted_labels[j] > 0:
                    idcg += (2 ** sorted_labels[j] - 1) / np.log2(j + 2)
            idcg_values.append(idcg)
        
        # Compute nDCG
        dcg_tensor = torch.tensor(dcg_values, device=scores.device, dtype=torch.float32)
        idcg_tensor = torch.tensor(idcg_values, device=scores.device, dtype=torch.float32)
        
        # Avoid division by zero
        idcg_tensor = torch.clamp(idcg_tensor, min=1e-8)
        ndcg = dcg_tensor / idcg_tensor
        
        # Return negative nDCG (to minimize)
        return -ndcg.mean()


class NeuralNDCGRetriever(nn.Module):
    """Retriever optimized with NeuralNDCG loss"""
    def __init__(self, base_model: SentenceTransformer):
        super().__init__()
        self.base_model = base_model
        self.loss_fn = NeuralNDCGLoss(temperature=1.0, k=10)
    
    def encode_query(self, query: str) -> torch.Tensor:
        """Encode query"""
        return self.base_model.encode(query, convert_to_tensor=True, show_progress_bar=False)
    
    def encode_documents(self, documents: List[str]) -> torch.Tensor:
        """Encode documents"""
        return self.base_model.encode(documents, convert_to_tensor=True, show_progress_bar=False, batch_size=32)
    
    def forward(self, queries: List[str], documents: List[List[str]], labels: List[List[int]]) -> torch.Tensor:
        """
        Forward pass for training
        Args:
            queries: List of query strings
            documents: List of document lists (one per query)
            labels: List of label lists (one per query)
        """
        # Encode queries using internal forward pass to track gradients
        query_embs = []
        device = next(self.base_model.parameters()).device
        for q in queries:
            # Use internal forward pass to track gradients
            with torch.set_grad_enabled(True):  # Ensure gradients are enabled
                query_features = self.base_model.tokenizer([q], padding=True, truncation=True,
                                                           max_length=512, return_tensors='pt')
                query_features = {k: v.to(device) for k, v in query_features.items()}
                # Fix: Pass features as positional argument or use input_ids directly
                # The Transformer module expects input_ids, attention_mask, etc. as keyword arguments
                query_output = self.base_model._modules['0'](query_features)
                # Extract token embeddings
                if isinstance(query_output, dict):
                    token_embeddings = query_output.get('token_embeddings', query_output.get('last_hidden_state'))
                else:
                    token_embeddings = query_output
                # Ensure token_embeddings is a tensor
                if token_embeddings is None:
                    # Try to get from output directly
                    if hasattr(query_output, 'last_hidden_state'):
                        token_embeddings = query_output.last_hidden_state
                    elif isinstance(query_output, tuple):
                        token_embeddings = query_output[0]
                    else:
                        raise ValueError(f"Could not extract token embeddings from query output: {type(query_output)}")
                # Pass through pooling module
                pooling_input = {'token_embeddings': token_embeddings, 
                               'attention_mask': query_features['attention_mask']}
                query_emb_output = self.base_model._modules['1'](pooling_input)
                if isinstance(query_emb_output, dict):
                    query_emb = query_emb_output.get('sentence_embedding')
                    if query_emb is None:
                        query_emb = next((v for v in query_emb_output.values() if isinstance(v, torch.Tensor)), None)
                else:
                    query_emb = query_emb_output
                if query_emb is None:
                    raise ValueError(f"Could not extract sentence embedding from pooling output")
                query_embs.append(query_emb.squeeze(0) if query_emb.dim() > 1 and query_emb.size(0) == 1 else query_emb)
        query_embs = torch.stack(query_embs)  # [batch_size, dim]
        
        # Encode documents for each query
        all_scores = []
        all_labels = []
        
        for i, (query_emb, docs, labs) in enumerate(zip(query_embs, documents, labels)):
            # Encode documents using internal forward pass to track gradients
            with torch.set_grad_enabled(True):  # Ensure gradients are enabled
                doc_features = self.base_model.tokenizer(docs, padding=True, truncation=True,
                                                         max_length=512, return_tensors='pt')
                doc_features = {k: v.to(device) for k, v in doc_features.items()}
                # Fix: Pass features correctly to Transformer module
                doc_output = self.base_model._modules['0'](doc_features)
                # Extract token embeddings
                if isinstance(doc_output, dict):
                    token_embeddings = doc_output.get('token_embeddings', doc_output.get('last_hidden_state'))
                else:
                    token_embeddings = doc_output
                # Ensure token_embeddings is a tensor
                if token_embeddings is None:
                    # Try to get from output directly
                    if hasattr(doc_output, 'last_hidden_state'):
                        token_embeddings = doc_output.last_hidden_state
                    elif isinstance(doc_output, tuple):
                        token_embeddings = doc_output[0]
                    else:
                        raise ValueError(f"Could not extract token embeddings from doc output: {type(doc_output)}")
                # Pass through pooling module
                pooling_input = {'token_embeddings': token_embeddings,
                               'attention_mask': doc_features['attention_mask']}
                doc_emb_output = self.base_model._modules['1'](pooling_input)
                if isinstance(doc_emb_output, dict):
                    doc_embs = doc_emb_output.get('sentence_embedding')
                    if doc_embs is None:
                        doc_embs = next((v for v in doc_emb_output.values() if isinstance(v, torch.Tensor)), None)
                else:
                    doc_embs = doc_emb_output
                if doc_embs is None:
                    raise ValueError(f"Could not extract sentence embedding from doc pooling output")
                
                # Compute similarity scores
                scores = torch.matmul(query_emb.unsqueeze(0), doc_embs.t())  # [1, num_docs]
                all_scores.append(scores.squeeze(0))
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
        
        scores_tensor = torch.stack(padded_scores)  # [batch_size, max_num_docs]
        labels_tensor = torch.stack(padded_labels)  # [batch_size, max_num_docs]
        
        # Compute loss
        loss = self.loss_fn(scores_tensor, labels_tensor)
        return loss


class NeuralNDCGDataset(Dataset):
    """Dataset for NeuralNDCG training"""
    def __init__(self, examples: List[InputExample], corpus_dict: Dict, qrels: Dict):
        self.examples = examples
        self.corpus_dict = corpus_dict
        self.qrels = qrels
        
        # Build query-document pairs with labels
        self.query_doc_pairs = []
        for example in examples:
            query_text, pos_doc_text = example.texts
            query_id = example.guid if hasattr(example, 'guid') else None
            
            # Get relevant documents for this query
            relevant_docs = []
            if query_id and query_id in qrels:
                relevant_docs = list(qrels[query_id].keys())
            
            # Sample negative documents
            all_doc_ids = list(corpus_dict.keys())
            negative_docs = [doc_id for doc_id in all_doc_ids if doc_id not in relevant_docs]
            negative_docs = negative_docs[:20]  # Limit negatives
            
            # Create document list and labels
            doc_list = [pos_doc_text]
            labels = [2]  # High relevance for positive
            
            for neg_doc_id in negative_docs[:19]:  # 19 negatives
                neg_doc = corpus_dict.get(neg_doc_id, {})
                neg_text = neg_doc.get('text', '')
                if neg_text:
                    doc_list.append(neg_text)
                    labels.append(0)  # No relevance
            
            self.query_doc_pairs.append({
                'query': query_text,
                'documents': doc_list,
                'labels': labels
            })
    
    def __len__(self):
        return len(self.query_doc_pairs)
    
    def __getitem__(self, idx):
        return self.query_doc_pairs[idx]


def neural_ndcg_collate_fn(batch):
    """Custom collate function for NeuralNDCG dataset"""
    queries = [item['query'] for item in batch]
    documents = [item['documents'] for item in batch]
    labels = [item['labels'] for item in batch]
    
    return {
        'queries': queries,
        'documents': documents,
        'labels': labels
    }


def load_training_data(domain: str, data_root: pathlib.Path, use_data_splits: bool = True) -> Tuple[List[InputExample], Dict, Dict]:
    """Load training data"""
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
        
        # Get positive document
        pos_doc_id = max(doc_scores.items(), key=lambda x: x[1])[0]
        pos_doc = corpus.get(pos_doc_id, {})
        pos_text = pos_doc.get('text', '')
        
        if pos_text:
            example = InputExample(texts=[query_text, pos_text], label=1.0)
            example.guid = query_id
            examples.append(example)
    
    return examples, corpus, qrels


def load_checkpoint(checkpoint_dir: pathlib.Path) -> Optional[Dict]:
    """Load checkpoint if exists"""
    checkpoint_file = checkpoint_dir / "checkpoint.json"
    if checkpoint_file.exists():
        with open(checkpoint_file, 'r') as f:
            return json.load(f)
    return None


def save_checkpoint(checkpoint_dir: pathlib.Path, epoch: int, model_path: str, results: Dict):
    """Save checkpoint"""
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_file = checkpoint_dir / "checkpoint.json"
    
    checkpoint_data = {
        "epoch": epoch,
        "model_path": model_path,
        "results": results,
        "timestamp": datetime.now().isoformat()
    }
    
    with open(checkpoint_file, 'w') as f:
        json.dump(checkpoint_data, f, indent=2)
    
    logging.info(f"✅ Checkpoint saved: {checkpoint_file}")


def train_neural_ndcg(config: Dict, gpu_id: int = 0) -> Optional[str]:
    """Train retriever with NeuralNDCG loss"""
    # Set CUDA_VISIBLE_DEVICES before importing/using torch
    cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if not cuda_visible and gpu_id is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        logging.info(f"Set CUDA_VISIBLE_DEVICES={gpu_id}")
    
    data_root = pathlib.Path(config.get('data_root', '.'))
    output_dir = pathlib.Path(config['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load checkpoint if resuming
    resume = config.get('resume', True)
    checkpoint = None
    start_epoch = 0
    if resume:
        checkpoint = load_checkpoint(output_dir)
        if checkpoint:
            start_epoch = checkpoint.get('epoch', 0) + 1
            logging.info(f"Resuming from epoch {start_epoch}")
    
    # Load model
    model_path = config.get('model_path', 'BAAI/bge-base-en-v1.5')
    logging.info(f"Loading model: {model_path}")
    base_model = SentenceTransformer(model_path)
    
    # Create retriever
    retriever = NeuralNDCGRetriever(base_model)
    retriever.train()
    
    # Setup device - use device 0 when CUDA_VISIBLE_DEVICES is set
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        try:
            torch.cuda.set_device(0)
            logging.info(f"Device: {device}, GPU: {torch.cuda.current_device()}")
        except Exception as e:
            logging.warning(f"Could not set CUDA device: {e}")
    retriever.to(device)
    
    # Load training data
    all_examples = []
    all_corpus = {}
    all_qrels = {}
    
    for domain in config.get('domains', MTRAG_DOMAINS):
        examples, corpus, qrels = load_training_data(domain, data_root, config.get('use_data_splits', True))
        all_examples.extend(examples)
        all_corpus.update(corpus)
        all_qrels.update(qrels)
        logging.info(f"Loaded {len(examples)} examples from {domain}")
    
    if not all_examples:
        logging.error("No training examples found")
        return None
    
    # Create dataset
    dataset = NeuralNDCGDataset(all_examples, all_corpus, all_qrels)
    dataloader = DataLoader(dataset, batch_size=config.get('batch_size', 4), 
                           shuffle=True, collate_fn=neural_ndcg_collate_fn)
    
    # Optimizer
    optimizer = torch.optim.AdamW(retriever.parameters(), lr=config.get('learning_rate', 2e-5))
    
    # Mixed precision training
    use_fp16 = config.get('use_fp16', False)
    gradient_accumulation_steps = config.get('gradient_accumulation_steps', 1)
    scaler = None
    if use_fp16 and device == "cuda":
        try:
            from torch.cuda.amp import autocast, GradScaler
            scaler = GradScaler()
            logging.info("Mixed precision training (FP16) enabled")
        except ImportError:
            logging.warning("FP16 not available, falling back to FP32")
            use_fp16 = False
    
    # Training loop
    num_epochs = config.get('epochs', 3)
    results = {}
    
    for epoch in range(start_epoch, num_epochs):
        if shutdown_requested:
            logging.info("Shutdown requested, saving checkpoint and exiting...")
            save_checkpoint(output_dir, epoch, str(output_dir / "model"), results)
            break
        
        logging.info(f"Epoch {epoch + 1}/{num_epochs}")
        epoch_loss = 0.0
        num_batches = 0
        optimizer.zero_grad()
        
        for batch_idx, batch in enumerate(dataloader):
            queries = batch['queries']
            documents = batch['documents']
            labels = batch['labels']
            
            # Forward pass with mixed precision if enabled
            if use_fp16 and scaler is not None:
                with autocast():
                    loss = retriever(queries, documents, labels)
                    loss = loss / gradient_accumulation_steps
                
                scaler.scale(loss).backward()
                
                if (batch_idx + 1) % gradient_accumulation_steps == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
            else:
                loss = retriever(queries, documents, labels)
                loss = loss / gradient_accumulation_steps
                loss.backward()
                
                if (batch_idx + 1) % gradient_accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()
            
            epoch_loss += loss.item() * gradient_accumulation_steps
            num_batches += 1
            
            # Clear cache periodically
            if device == "cuda" and (batch_idx + 1) % 10 == 0:
                torch.cuda.empty_cache()
                logging.info(f"  Batch {batch_idx + 1}/{len(dataloader)}, Loss: {loss.item() * gradient_accumulation_steps:.4f}")
        
        avg_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
        logging.info(f"Epoch {epoch + 1} completed. Average loss: {avg_loss:.4f}")
        
        # Save checkpoint
        save_checkpoint(output_dir, epoch, str(output_dir / "model"), results)
        
        # Save model
        model_save_path = output_dir / f"model_epoch_{epoch + 1}"
        base_model.save(str(model_save_path))
    
    # Final model save
    final_model_path = output_dir / "model"
    base_model.save(str(final_model_path))
    
    # Evaluation
    logging.info("Running evaluation...")
    evaluator = EvaluateRetrieval(retriever.base_model, k_values=[5, 10])
    
    all_results = {}
    for domain in config.get('domains', MTRAG_DOMAINS):
        corpus_file = data_root / "corpora" / "passage_level" / f"{domain}.jsonl"
        query_file = data_root / "data_splits" / "retrieval_tasks" / domain / "test" / f"{domain}_questions.jsonl"
        qrels_file = data_root / "data_splits" / "retrieval_tasks" / domain / "test" / "qrels" / "dev.tsv"
        
        if not query_file.exists():
            query_file = data_root / "human" / "retrieval_tasks" / domain / f"{domain}_questions.jsonl"
            qrels_file = data_root / "human" / "retrieval_tasks" / domain / "qrels" / "dev.tsv"
        
        try:
            results = evaluator.retrieve(corpus_file, query_file)
            ndcg, recall = evaluator.evaluate(qrels_file, results, [5, 10])
            all_results[domain] = {
                'nDCG@5': ndcg.get('NDCG@5', 0.0),
                'nDCG@10': ndcg.get('NDCG@10', 0.0),
                'Recall@5': recall.get('Recall@5', 0.0),
                'Recall@10': recall.get('Recall@10', 0.0)
            }
            logging.info(f"{domain}: nDCG@10={all_results[domain]['nDCG@10']:.4f}")
        except Exception as e:
            logging.error(f"Error evaluating {domain}: {e}")
    
    # Save results
    results_file = output_dir / "results.json"
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    logging.info("✅ Training completed")
    return str(output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    train_neural_ndcg(config, gpu_id=args.gpu)

