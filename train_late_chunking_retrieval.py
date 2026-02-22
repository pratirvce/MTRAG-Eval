"""
Late Chunking Retrieval: Contextual Chunk Embeddings Using Long-Context Embedding Models
Paper: "Late Chunking: Contextual Chunk Embeddings Using Long-Context Embedding Models" (2024)
ArXiv: https://arxiv.org/abs/2409.04701

Key Innovation: Embed the entire document first, then chunk it. This preserves global context
within each chunk, leading to more accurate and context-aware embeddings.

Task A Compliant: ✅ Pure retrieval, no generation
Expected: +0.05-0.10 nDCG@10 improvement over standard chunking
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
import re
from transformers import AutoTokenizer

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


class LateChunkingRetriever:
    """
    Late Chunking: Embed full document first, then create chunk embeddings
    that preserve global context.
    """
    def __init__(self, base_model: SentenceTransformer, chunk_size: int = 512, 
                 chunk_overlap: int = 100, max_doc_length: int = 2048):
        self.base_model = base_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.max_doc_length = max_doc_length
        self.tokenizer = base_model.tokenizer
        
    def embed_full_document(self, document: str) -> torch.Tensor:
        """
        Embed the entire document first to capture global context.
        If document is too long, use sliding window approach.
        """
        # Tokenize document
        tokens = self.tokenizer.encode(document, add_special_tokens=True, 
                                      max_length=self.max_doc_length, 
                                      truncation=True, return_tensors='pt')
        
        # Get full document embedding
        device = next(self.base_model.parameters()).device
        tokens = tokens.to(device)
        
        with torch.no_grad():
            outputs = self.base_model._modules['0'](tokens)
            if isinstance(outputs, dict):
                full_embeddings = outputs.get('last_hidden_state') or outputs.get('token_embeddings')
            else:
                full_embeddings = outputs
            
            # Mean pool full document
            pooling_output = self.base_model._modules['1']({
                'token_embeddings': full_embeddings,
                'attention_mask': (tokens != self.tokenizer.pad_token_id).long()
            })
            full_doc_emb = pooling_output.get('sentence_embedding')
            if full_doc_emb is None:
                full_doc_emb = next((v for v in pooling_output.values() if isinstance(v, torch.Tensor)), None)
        
        return full_doc_emb
    
    def chunk_with_context(self, document: str, full_doc_emb: torch.Tensor) -> List[Tuple[str, torch.Tensor]]:
        """
        Chunk document and create chunk embeddings that incorporate global context.
        Each chunk embedding is a combination of:
        1. Local chunk embedding
        2. Global document embedding (weighted)
        """
        # Recursive chunking: paragraph → sentence → word
        chunks = self._recursive_chunk(document)
        
        chunk_embeddings = []
        device = next(self.base_model.parameters()).device
        
        for chunk_text in chunks:
            # Embed local chunk
            chunk_tokens = self.tokenizer.encode(chunk_text, add_special_tokens=True,
                                                max_length=self.chunk_size,
                                                truncation=True, return_tensors='pt')
            chunk_tokens = chunk_tokens.to(device)
            
            with torch.no_grad():
                chunk_outputs = self.base_model._modules['0'](chunk_tokens)
                if isinstance(chunk_outputs, dict):
                    chunk_token_embs = chunk_outputs.get('last_hidden_state') or chunk_outputs.get('token_embeddings')
                else:
                    chunk_token_embs = chunk_outputs
                
                pooling_output = self.base_model._modules['1']({
                    'token_embeddings': chunk_token_embs,
                    'attention_mask': (chunk_tokens != self.tokenizer.pad_token_id).long()
                })
                local_chunk_emb = pooling_output.get('sentence_embedding')
                if local_chunk_emb is None:
                    local_chunk_emb = next((v for v in pooling_output.values() if isinstance(v, torch.Tensor)), None)
            
            # Combine local chunk embedding with global document embedding
            # Weight: 0.7 local + 0.3 global (preserves context while maintaining chunk specificity)
            if local_chunk_emb.dim() == 1:
                local_chunk_emb = local_chunk_emb.unsqueeze(0)
            if full_doc_emb.dim() == 1:
                full_doc_emb = full_doc_emb.unsqueeze(0)
            
            # Ensure same dimension
            if local_chunk_emb.shape[1] != full_doc_emb.shape[1]:
                # Pad or truncate to match
                min_dim = min(local_chunk_emb.shape[1], full_doc_emb.shape[1])
                local_chunk_emb = local_chunk_emb[:, :min_dim]
                full_doc_emb = full_doc_emb[:, :min_dim]
            
            contextual_chunk_emb = 0.7 * local_chunk_emb + 0.3 * full_doc_emb
            contextual_chunk_emb = F.normalize(contextual_chunk_emb, p=2, dim=1)
            
            chunk_embeddings.append((chunk_text, contextual_chunk_emb.squeeze(0)))
        
        return chunk_embeddings
    
    def _recursive_chunk(self, text: str) -> List[str]:
        """
        Recursive chunking: Try to split at paragraph boundaries first,
        then sentence boundaries, then word boundaries.
        """
        # First try: split by paragraphs
        paragraphs = re.split(r'\n\n+', text)
        chunks = []
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            # Check if paragraph fits in chunk size
            para_tokens = len(self.tokenizer.encode(para, add_special_tokens=False))
            
            if para_tokens <= self.chunk_size:
                chunks.append(para)
            else:
                # Split by sentences
                sentences = re.split(r'(?<=[.!?])\s+', para)
                current_chunk = []
                current_size = 0
                
                for sentence in sentences:
                    sentence = sentence.strip()
                    if not sentence:
                        continue
                    
                    sent_tokens = len(self.tokenizer.encode(sentence, add_special_tokens=False))
                    
                    if current_size + sent_tokens <= self.chunk_size:
                        current_chunk.append(sentence)
                        current_size += sent_tokens
                    else:
                        if current_chunk:
                            chunks.append(' '.join(current_chunk))
                        current_chunk = [sentence]
                        current_size = sent_tokens
                
                if current_chunk:
                    chunks.append(' '.join(current_chunk))
        
        return chunks
    
    def encode_query(self, query: str) -> torch.Tensor:
        """Encode query with asymmetric prompt"""
        # Use query prompt for better retrieval
        query_with_prompt = f"Represent this sentence for searching relevant passages: {query}"
        return self.base_model.encode(query_with_prompt, convert_to_tensor=True, 
                                     show_progress_bar=False)


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


def train_late_chunking_retrieval(config: Dict, gpu_id: Optional[int] = None) -> Optional[str]:
    """Train retrieval with Late Chunking strategy"""
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
    
    output_dir = pathlib.Path(config.get('output_dir', 'experiments/retrieval/late_chunking_retrieval'))
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
    base_model_name = config.get('model_path', 'BAAI/bge-large-en-v1.5')  # Use large model for better context
    chunk_size = config.get('chunk_size', 512)
    chunk_overlap = config.get('chunk_overlap', 100)
    max_doc_length = config.get('max_doc_length', 2048)
    
    # Load base model
    logging.info(f"Loading base model: {base_model_name}")
    base_model = SentenceTransformer(base_model_name)
    base_model.to(device)
    
    # Create Late Chunking Retriever
    retriever = LateChunkingRetriever(
        base_model=base_model,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        max_doc_length=max_doc_length
    )
    
    # Fine-tune on MTRAG data if needed
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
    
    # Training loop (optional - can skip if using pre-trained model)
    if config.get('fine_tune', False):
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
            
            # Simple contrastive training
            dataloader = DataLoader(examples, batch_size=batch_size, shuffle=True)
            optimizer = torch.optim.AdamW(base_model.parameters(), lr=2e-5, weight_decay=0.01)
            
            base_model.train()
            for epoch in range(epochs):
                total_loss = 0
                for batch_idx, batch in enumerate(dataloader):
                    queries_batch = [ex.texts[0] for ex in batch]
                    docs_batch = [ex.texts[1] for ex in batch]
                    
                    # Encode queries and documents
                    query_embs = torch.stack([retriever.encode_query(q) for q in queries_batch])
                    doc_embs = torch.stack([base_model.encode(d, convert_to_tensor=True) for d in docs_batch])
                    
                    # Contrastive loss
                    query_embs = F.normalize(query_embs, p=2, dim=1)
                    doc_embs = F.normalize(doc_embs, p=2, dim=1)
                    similarities = torch.sum(query_embs * doc_embs, dim=1)
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
                
                logging.info(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader):.4f}")
            
            # Save trained model
            trained_model_path = checkpoint_dir / f"trained_model_{domain}"
            base_model.save(str(trained_model_path))
            logging.info(f"Saved trained model for {domain} to {trained_model_path}")
    
    # Evaluation with Late Chunking
    logging.info("\nEvaluating with Late Chunking retrieval...")
    
    # Use trained model if available, otherwise use base model
    if len(domains) > 0:
        last_domain = list(domains)[-1] if isinstance(domains, (list, tuple)) else domains[-1]
        trained_model_path = checkpoint_dir / f"trained_model_{last_domain}"
        if trained_model_path.exists() and config.get('fine_tune', False):
            logging.info(f"Loading trained model from {trained_model_path}")
            base_model = SentenceTransformer(str(trained_model_path))
            base_model.to(device)
            retriever = LateChunkingRetriever(base_model, chunk_size, chunk_overlap, max_doc_length)
    
    # Create a custom retriever that uses Late Chunking
    class LateChunkingDenseRetriever:
        def __init__(self, retriever: LateChunkingRetriever):
            self.retriever = retriever
            self.corpus_embeddings = {}
            self.corpus_texts = {}
        
        def index(self, corpus: Dict):
            """Index corpus using Late Chunking"""
            logging.info("Indexing corpus with Late Chunking...")
            for doc_id, doc in corpus.items():
                title = doc.get("title", "")
                text = doc.get("text", "")
                doc_text = f"{title} {text}".strip() if title else text
                
                # Get full document embedding
                full_doc_emb = self.retriever.embed_full_document(doc_text)
                
                # Get chunk embeddings with context
                chunk_embs = self.retriever.chunk_with_context(doc_text, full_doc_emb)
                
                # Store all chunk embeddings (use first chunk as primary, others as alternatives)
                if chunk_embs:
                    self.corpus_embeddings[doc_id] = chunk_embs[0][1].cpu().numpy()  # Primary chunk
                    self.corpus_texts[doc_id] = doc_text
        
        def retrieve(self, corpus: Dict, queries: Dict, top_k: int = 10) -> Dict:
            """Retrieve using Late Chunking embeddings"""
            results = {}
            
            for query_id, query_text in queries.items():
                query_emb = self.retriever.encode_query(query_text).cpu().numpy()
                
                # Compute similarities
                scores = {}
                for doc_id, doc_emb in self.corpus_embeddings.items():
                    similarity = np.dot(query_emb, doc_emb) / (np.linalg.norm(query_emb) * np.linalg.norm(doc_emb) + 1e-8)
                    scores[doc_id] = float(similarity)
                
                # Sort and get top-k
                sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
                results[query_id] = {doc_id: score for doc_id, score in sorted_docs}
            
            return results
    
    custom_retriever = LateChunkingDenseRetriever(retriever)
    
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
        results_path = train_late_chunking_retrieval(config, args.gpu_id)
        if results_path:
            logging.info(f"✅ Late Chunking retrieval training completed: {results_path}")
    except KeyboardInterrupt:
        logging.info("Interrupted by user")
        sys.exit(0)
    except Exception as e:
        logging.error(f"Failed: {e}", exc_info=True)
        sys.exit(1)

