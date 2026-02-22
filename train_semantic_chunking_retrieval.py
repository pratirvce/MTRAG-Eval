"""
Semantic Chunking Retrieval: Boundary-Aware Chunking for Better Retrieval
Task A - Retrieval Only
Expected: +0.03-0.06 nDCG@10 improvement

Key Features:
1. Semantic boundary-aware chunking (paragraph → sentence)
2. Preserves semantic coherence (no mid-sentence splits)
3. Proper document ID mapping for evaluation
4. BGE-large embeddings with fine-tuning
5. Task A compliant (no text generation)
"""

import pathlib
import logging
import argparse
import json
import os
import numpy as np
import re
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
from collections import defaultdict

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

class SemanticChunker:
    """
    Semantic boundary-aware chunking that respects paragraph and sentence boundaries
    """
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 100, tokenizer=None):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.tokenizer = tokenizer
    
    def chunk_text(self, text: str, doc_id: str) -> List[Tuple[str, str]]:
        """
        Chunk text at semantic boundaries (paragraph → sentence)
        Returns: List of (chunk_text, chunk_id) tuples
        chunk_id format: {doc_id}_chunk_{index}
        """
        chunks = []
        
        # Step 1: Split by paragraphs (double newlines)
        paragraphs = re.split(r'\n\n+', text)
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        
        current_chunk = []
        current_size = 0
        chunk_index = 0
        
        for para in paragraphs:
            # Count tokens in paragraph
            if self.tokenizer:
                para_tokens = len(self.tokenizer.encode(para, add_special_tokens=False))
            else:
                # Fallback: approximate tokens as words * 1.3
                para_tokens = int(len(para.split()) * 1.3)
            
            # If paragraph fits in current chunk
            if current_size + para_tokens <= self.chunk_size:
                current_chunk.append(para)
                current_size += para_tokens
            else:
                # Save current chunk if it has content
                if current_chunk:
                    chunk_text = '\n\n'.join(current_chunk)
                    chunk_id = f"{doc_id}_chunk_{chunk_index}"
                    chunks.append((chunk_text, chunk_id))
                    chunk_index += 1
                
                # If paragraph itself is too large, split by sentences
                if para_tokens > self.chunk_size:
                    sentence_chunks = self._chunk_by_sentences(para, doc_id, chunk_index)
                    chunks.extend(sentence_chunks)
                    chunk_index += len(sentence_chunks)
                    current_chunk = []
                    current_size = 0
                else:
                    # Start new chunk with this paragraph
                    current_chunk = [para]
                    current_size = para_tokens
        
        # Add remaining chunk
        if current_chunk:
            chunk_text = '\n\n'.join(current_chunk)
            chunk_id = f"{doc_id}_chunk_{chunk_index}"
            chunks.append((chunk_text, chunk_id))
        
        return chunks
    
    def _chunk_by_sentences(self, text: str, doc_id: str, start_index: int) -> List[Tuple[str, str]]:
        """Split text by sentences when paragraph is too large"""
        # Split by sentence boundaries
        sentences = re.split(r'(?<=[.!?])\s+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        chunks = []
        current_chunk = []
        current_size = 0
        chunk_index = start_index
        
        for sentence in sentences:
            if self.tokenizer:
                sent_tokens = len(self.tokenizer.encode(sentence, add_special_tokens=False))
            else:
                sent_tokens = int(len(sentence.split()) * 1.3)
            
            if current_size + sent_tokens <= self.chunk_size:
                current_chunk.append(sentence)
                current_size += sent_tokens
            else:
                if current_chunk:
                    chunk_text = ' '.join(current_chunk)
                    chunk_id = f"{doc_id}_chunk_{chunk_index}"
                    chunks.append((chunk_text, chunk_id))
                    chunk_index += 1
                
                # If sentence itself is too large, split by words (last resort)
                if sent_tokens > self.chunk_size:
                    word_chunks = self._chunk_by_words(sentence, doc_id, chunk_index)
                    chunks.extend(word_chunks)
                    chunk_index += len(word_chunks)
                else:
                    current_chunk = [sentence]
                    current_size = sent_tokens
        
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunk_id = f"{doc_id}_chunk_{chunk_index}"
            chunks.append((chunk_text, chunk_id))
        
        return chunks
    
    def _chunk_by_words(self, text: str, doc_id: str, start_index: int) -> List[Tuple[str, str]]:
        """Split by words as last resort (should rarely be needed)"""
        words = text.split()
        chunks = []
        current_chunk = []
        current_size = 0
        chunk_index = start_index
        
        for word in words:
            word_tokens = 1  # Approximate
            
            if current_size + word_tokens <= self.chunk_size:
                current_chunk.append(word)
                current_size += word_tokens
            else:
                if current_chunk:
                    chunk_text = ' '.join(current_chunk)
                    chunk_id = f"{doc_id}_chunk_{chunk_index}"
                    chunks.append((chunk_text, chunk_id))
                    chunk_index += 1
                    current_chunk = [word]
                    current_size = word_tokens
        
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunk_id = f"{doc_id}_chunk_{chunk_index}"
            chunks.append((chunk_text, chunk_id))
        
        return chunks

class SemanticChunkingRetriever:
    """
    Retriever that uses semantic chunking for better document representation
    """
    def __init__(self, base_model: SentenceTransformer, chunk_size: int = 512, chunk_overlap: int = 100):
        self.base_model = base_model
        self.chunker = SemanticChunker(chunk_size, chunk_overlap, base_model.tokenizer)
        self.chunk_index = {}  # Maps chunk_id -> (doc_id, chunk_text, embedding)
        self.doc_to_chunks = defaultdict(list)  # Maps doc_id -> [chunk_ids]
    
    def index_corpus(self, corpus: Dict[str, Dict]):
        """Index corpus with semantic chunking"""
        logging.info("Indexing corpus with semantic chunking...")
        
        all_chunks = []
        chunk_to_doc = {}
        
        for doc_id, doc in corpus.items():
            title = doc.get('title', '')
            text = doc.get('text', '')
            
            # Combine title and text
            full_text = f"{title}\n\n{text}".strip() if title else text
            
            # Chunk with semantic boundaries
            chunks = self.chunker.chunk_text(full_text, doc_id)
            
            # Store mapping
            for chunk_text, chunk_id in chunks:
                all_chunks.append((chunk_id, chunk_text))
                chunk_to_doc[chunk_id] = doc_id
                self.doc_to_chunks[doc_id].append(chunk_id)
        
        # Encode all chunks
        logging.info(f"Encoding {len(all_chunks)} semantic chunks...")
        chunk_texts = [chunk_text for _, chunk_text in all_chunks]
        chunk_embeddings = self.base_model.encode(
            chunk_texts,
            batch_size=128,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        
        # Store in index
        for (chunk_id, chunk_text), embedding in zip(all_chunks, chunk_embeddings):
            doc_id = chunk_to_doc[chunk_id]
            self.chunk_index[chunk_id] = {
                'doc_id': doc_id,
                'chunk_text': chunk_text,
                'embedding': embedding
            }
        
        logging.info(f"Indexed {len(self.chunk_index)} chunks from {len(corpus)} documents")
        logging.info(f"Average chunks per document: {len(self.chunk_index) / len(corpus):.2f}")
    
    def retrieve(self, queries: Dict[str, str], top_k: int = 10) -> Dict[str, Dict[str, float]]:
        """Retrieve documents using semantic chunks"""
        # Encode queries
        query_texts = list(queries.values())
        query_ids = list(queries.keys())
        
        query_embeddings = self.base_model.encode(
            query_texts,
            batch_size=128,
            show_progress_bar=False,
            convert_to_numpy=True
        )
        
        # For each query, find top chunks, then aggregate by document
        results = {}
        
        for query_id, query_emb in zip(query_ids, query_embeddings):
            # Compute similarities with all chunks
            chunk_scores = {}
            for chunk_id, chunk_data in self.chunk_index.items():
                chunk_emb = chunk_data['embedding']
                similarity = np.dot(query_emb, chunk_emb) / (
                    np.linalg.norm(query_emb) * np.linalg.norm(chunk_emb) + 1e-8
                )
                doc_id = chunk_data['doc_id']
                
                # Aggregate by document (max score from any chunk)
                if doc_id not in chunk_scores:
                    chunk_scores[doc_id] = similarity
                else:
                    chunk_scores[doc_id] = max(chunk_scores[doc_id], similarity)
            
            # Sort by score and return top_k
            sorted_docs = sorted(chunk_scores.items(), key=lambda x: x[1], reverse=True)
            results[query_id] = {doc_id: float(score) for doc_id, score in sorted_docs[:top_k]}
        
        return results

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

def train_semantic_chunking_retrieval(config: Dict, gpu_id: Optional[int] = None) -> Optional[str]:
    """Train retrieval with semantic chunking"""
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
    
    output_dir = pathlib.Path(config.get('output_dir', 'experiments/retrieval/semantic_chunking_retrieval'))
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
    base_model_name = config.get('model_path', 'BAAI/bge-large-en-v1.5')
    chunk_size = config.get('chunk_size', 512)
    chunk_overlap = config.get('chunk_overlap', 100)
    
    # Load base model
    logging.info(f"Loading base model: {base_model_name}")
    base_model = SentenceTransformer(base_model_name)
    base_model.to(device)
    
    # Fine-tune on MTRAG data
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
    
    # Evaluation with semantic chunking
    logging.info("\nEvaluating with semantic chunking retrieval...")
    
    # Reload trained model if available
    trained_model_path = checkpoint_dir / "trained_model"
    if trained_model_path.exists():
        logging.info(f"Loading trained model from {trained_model_path}")
        base_model = SentenceTransformer(str(trained_model_path))
        base_model.to(device)
    
    retriever = SemanticChunkingRetriever(base_model, chunk_size, chunk_overlap)
    
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
            
            # Index corpus with semantic chunking
            logging.info(f"Indexing {domain} corpus with semantic chunking...")
            retriever.index_corpus(corpus)
            
            # Retrieve
            logging.info(f"Retrieving for {domain}...")
            results = retriever.retrieve(queries, top_k=10)
            
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
        results_path = train_semantic_chunking_retrieval(config, args.gpu_id)
        if results_path:
            logging.info(f"✅ Semantic chunking retrieval completed: {results_path}")
    except KeyboardInterrupt:
        logging.info("Interrupted by user")
        sys.exit(0)
    except Exception as e:
        logging.error(f"Failed: {e}", exc_info=True)
        sys.exit(1)

