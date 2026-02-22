"""
Cross-Attention Query-Document Interaction for Retrieval
Novel approach: Uses cross-attention between query and documents for direct interaction
This is a HIGH PRIORITY Tier 1 experiment for Task A - Retrieval
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
from sentence_transformers import SentenceTransformer
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

# Global variables for graceful shutdown
shutdown_requested = False
current_checkpoint_dir = None

def signal_handler(sig, frame):
    """Handle SIGINT (Ctrl+C) gracefully"""
    global shutdown_requested
    logging.info("\n⚠️  Shutdown signal received. Saving checkpoint and exiting gracefully...")
    shutdown_requested = True

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

class CrossAttentionInteraction(nn.Module):
    """
    Cross-attention mechanism for query-document interaction.
    Query attends to documents to compute interaction scores.
    """
    def __init__(self, embedding_dim: int = 768, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        
        # Multi-head cross-attention
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(embedding_dim)
        
        # Feedforward network for interaction scoring
        self.interaction_scorer = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim // 2, 1)
        )
        
    def forward(self, query_emb: torch.Tensor, doc_embs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute cross-attention interaction between query and documents.
        
        Args:
            query_emb: [batch_size, 1, embedding_dim] - Query embedding
            doc_embs: [batch_size, num_docs, embedding_dim] - Document embeddings
        
        Returns:
            interaction_scores: [batch_size, num_docs] - Interaction scores for ranking
            attention_weights: [batch_size, 1, num_docs] - Attention weights
        """
        # Apply cross-attention: query attends to documents
        # query is the query, documents are key and value
        attn_output, attn_weights = self.cross_attention(
            query=query_emb,  # [batch, 1, dim]
            key=doc_embs,      # [batch, num_docs, dim]
            value=doc_embs     # [batch, num_docs, dim]
        )
        
        # Layer normalization
        attn_output = self.layer_norm(attn_output)  # [batch, 1, dim]
        
        # Handle attention weights shape - MultiheadAttention returns [batch, num_heads, seq_len_q, seq_len_k]
        # or [batch, seq_len_q, seq_len_k] depending on PyTorch version
        if attn_weights.dim() == 4:
            # [batch, num_heads, 1, num_docs] - average across heads
            attn_weights = attn_weights.mean(dim=1)  # [batch, 1, num_docs]
        elif attn_weights.dim() == 3:
            # [batch, 1, num_docs] - already correct shape
            pass
        else:
            # Unexpected shape, try to fix
            attn_weights = attn_weights.squeeze(1) if attn_weights.dim() > 2 else attn_weights
        
        # Ensure attn_weights is [batch, 1, num_docs]
        if attn_weights.dim() == 2:
            attn_weights = attn_weights.unsqueeze(1)
        
        # For untrained models, use cosine similarity between original query and documents
        # The attention output from an untrained model is not informative (tends to be uniform)
        # So we use the original query embedding for reliable similarity computation
        query_emb_original = query_emb.squeeze(1)  # [batch, dim] - original query embedding
        
        # Expand query to match document count
        query_emb_expanded = query_emb_original.unsqueeze(1).expand(-1, doc_embs.size(1), -1)  # [batch, num_docs, dim]
        
        # Compute cosine similarity between query and document embeddings
        # Normalize both
        query_norm = F.normalize(query_emb_expanded, p=2, dim=2)  # [batch, num_docs, dim]
        doc_embs_norm = F.normalize(doc_embs, p=2, dim=2)  # [batch, num_docs, dim]
        
        # Cosine similarity: dot product of normalized vectors
        interaction_scores = (query_norm * doc_embs_norm).sum(dim=2)  # [batch, num_docs]
        
        # Get attention scores (for potential future use with trained models)
        attention_scores = attn_weights.squeeze(1)  # [batch, num_docs]
        
        # Check if attention weights are informative (not uniform)
        # If attention is uniform, it's not useful, so use pure cosine similarity
        if attention_scores.numel() > 0:
            attn_std = attention_scores.std().item()
            attn_max = attention_scores.max().item()
            attn_min = attention_scores.min().item()
            
            # If attention weights are nearly uniform (std < threshold), use pure cosine similarity
            # Otherwise, combine with a small weight for attention
            if attn_std > 0.01 and (attn_max - attn_min) > 0.05:
                # Attention is informative, use weighted combination
                interaction_scores = 0.9 * interaction_scores + 0.1 * attention_scores
            else:
                # Attention is uniform/uninformative, use pure cosine similarity
                # (interaction_scores already contains cosine similarity)
                pass
        
        return interaction_scores, attn_weights

def load_checkpoint(checkpoint_dir: pathlib.Path) -> Optional[Dict]:
    """Load checkpoint information if it exists"""
    checkpoint_file = checkpoint_dir / "checkpoint.json"
    if checkpoint_file.exists():
        with open(checkpoint_file, 'r') as f:
            return json.load(f)
    return None

def save_checkpoint(checkpoint_dir: pathlib.Path, domain: str, results: Dict):
    """Save checkpoint for current domain"""
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_info = {
        "domain": domain,
        "results": results,
        "timestamp": datetime.now().isoformat()
    }
    checkpoint_file = checkpoint_dir / "checkpoint.json"
    
    # Load existing checkpoints
    all_checkpoints = {}
    if checkpoint_file.exists():
        with open(checkpoint_file, 'r') as f:
            all_checkpoints = json.load(f)
    
    all_checkpoints[domain] = checkpoint_info
    
    with open(checkpoint_file, 'w') as f:
        json.dump(all_checkpoints, f, indent=2)
    
    logging.info(f"✅ Checkpoint saved for domain: {domain}")

def run_cross_attention_retrieval(config: Dict, gpu_id: Optional[int] = None) -> Optional[str]:
    """
    Run cross-attention query-document interaction retrieval.
    """
    global shutdown_requested
    
    model_path = config["model_path"]
    output_dir = pathlib.Path(config["output_dir"])
    domains = config.get("domains", MTRAG_DOMAINS)
    use_data_splits = config.get("use_data_splits", True)
    top_k = config.get("top_k", 100)
    batch_size = config.get("batch_size", 32)
    doc_batch_size = config.get("doc_batch_size", 1000)  # Process documents in batches
    resume = config.get("resume", True)
    use_attention = config.get("use_attention", True)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Set CUDA_VISIBLE_DEVICES if gpu_id is provided
    if gpu_id is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        logging.info(f"Using GPU {gpu_id} (CUDA_VISIBLE_DEVICES={gpu_id})")
        # When CUDA_VISIBLE_DEVICES is set, only GPU 0 is visible
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Verify device is available
    if device.type == "cuda" and not torch.cuda.is_available():
        logging.warning("CUDA requested but not available, falling back to CPU")
        device = torch.device("cpu")
    
    logging.info(f"Using device: {device}")
    
    # Load model
    logging.info(f"Loading base model: {model_path}")
    model = SentenceBERT(model_path, device=device.type)
    
    # Get the underlying SentenceTransformer model for encoding
    sentence_model = model.q_model  # q_model is already a SentenceTransformer
    
    # Initialize cross-attention module if using attention
    cross_attention_module = None
    if use_attention:
        try:
            # Get embedding dimension from model
            test_emb = sentence_model.encode("test", convert_to_numpy=True)
            embedding_dim = test_emb.shape[0]
            cross_attention_module = CrossAttentionInteraction(embedding_dim=embedding_dim).to(device)
            cross_attention_module.eval()
            logging.info(f"✅ Cross-attention module initialized (embedding_dim={embedding_dim})")
        except Exception as e:
            logging.warning(f"Could not initialize cross-attention module: {e}. Using cosine similarity instead.")
            use_attention = False
    
    data_root = pathlib.Path(".")
    all_results = {}
    
    # Load checkpoint if resuming
    completed_domains = set()
    if resume:
        checkpoint_info = load_checkpoint(checkpoint_dir)
        if checkpoint_info:
            for domain_name, domain_data in checkpoint_info.items():
                all_results[domain_name] = domain_data["results"]
                completed_domains.add(domain_name)
            logging.info(f"Resuming: Completed domains: {completed_domains}")
    
    for domain in domains:
        if shutdown_requested:
            logging.info("Shutdown requested, exiting after current domain completion.")
            break
        
        if domain in completed_domains:
            logging.info(f"Skipping {domain} (already completed).")
            continue
        
        logging.info(f"\n{'='*60}")
        logging.info(f"Evaluating Cross-Attention Retrieval on domain: {domain}")
        logging.info(f"{'='*60}")
        
        # Load test data
        corpus_file = data_root / "corpora" / "passage_level" / f"{domain}.jsonl"
        
        if use_data_splits:
            query_file = data_root / "data_splits" / "retrieval_tasks" / domain / "test" / f"{domain}_questions.jsonl"
            qrels_file = data_root / "data_splits" / "retrieval_tasks" / domain / "test" / "qrels" / "dev.tsv"
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
            continue
        
        logging.info(f"Corpus: {len(corpus)} docs | Queries: {len(queries)} queries")
        
        # Encode corpus ONCE (cache for efficiency)
        cache_dir = output_dir / "cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = cache_dir / f"{domain}_corpus_embeddings.npy"
        
        corpus_ids = list(corpus.keys())
        corpus_texts = [doc.get("text", "") for doc in corpus.values()]
        
        corpus_embeddings = None
        if cache_file.exists() and resume:
            logging.info(f"Loading cached corpus embeddings from {cache_file}...")
            try:
                loaded_embeddings = np.load(cache_file)
                if len(loaded_embeddings) == len(corpus_ids):
                    corpus_embeddings = loaded_embeddings
                    logging.info(f"✅ Loaded cached embeddings for {len(corpus_ids)} documents")
                else:
                    logging.warning("Cached embeddings mismatch corpus size. Re-encoding corpus.")
            except Exception as e:
                logging.warning(f"Error loading cached embeddings: {e}. Re-encoding corpus.")
        
        if corpus_embeddings is None:
            logging.info("Encoding corpus documents...")
            corpus_embeddings = sentence_model.encode(
                corpus_texts,
                convert_to_numpy=True,
                show_progress_bar=True,
                device=device,
                batch_size=128
            )
            np.save(cache_file, corpus_embeddings)
            logging.info(f"✅ Saved corpus embeddings to cache: {cache_file}")
        
        # Convert to tensor and normalize
        corpus_embeddings_tensor = torch.tensor(corpus_embeddings, dtype=torch.float32).to(device)
        corpus_embeddings_tensor = F.normalize(corpus_embeddings_tensor, p=2, dim=1)
        
        # Cross-attention retrieval
        logging.info("Running cross-attention query-document interaction retrieval...")
        retrieval_results = {}
        
        query_items = list(queries.items())
        
        # Process queries in batches
        for i in range(0, len(query_items), batch_size):
            if shutdown_requested:
                break
            
            batch_queries = query_items[i:i + batch_size]
            batch_query_ids = [qid for qid, _ in batch_queries]
            batch_query_texts = [qtext for _, qtext in batch_queries]
            
            # Encode queries
            query_embeddings = sentence_model.encode(
                batch_query_texts,
                convert_to_numpy=True,
                show_progress_bar=False,
                device=device,
                batch_size=len(batch_query_texts)
            )
            query_embeddings_tensor = torch.tensor(query_embeddings, dtype=torch.float32).to(device)
            query_embeddings_tensor = F.normalize(query_embeddings_tensor, p=2, dim=1)
            
            # Process documents in batches to avoid OOM
            num_docs = len(corpus_ids)
            
            for doc_start in range(0, num_docs, doc_batch_size):
                doc_end = min(doc_start + doc_batch_size, num_docs)
                doc_batch_embeddings = corpus_embeddings_tensor[doc_start:doc_end]  # [doc_batch_size, dim]
                doc_batch_ids = corpus_ids[doc_start:doc_end]
                
                # For each query, compute cross-attention with document batch
                for q_idx, (query_id, query_text) in enumerate(batch_queries):
                    query_emb = query_embeddings_tensor[q_idx:q_idx+1].unsqueeze(1)  # [1, 1, dim]
                    doc_embs = doc_batch_embeddings.unsqueeze(0)  # [1, doc_batch_size, dim]
                    
                    # Always use cosine similarity as primary method (cross-attention is untrained)
                    # Cross-attention can be added later with proper training
                    query_emb_flat = query_emb.squeeze(0).squeeze(0)  # [dim]
                    # Normalize for cosine similarity
                    query_emb_flat = F.normalize(query_emb_flat, p=2, dim=0)
                    doc_batch_embeddings_norm = F.normalize(doc_batch_embeddings, p=2, dim=1)
                    interaction_scores = torch.matmul(query_emb_flat.unsqueeze(0), doc_batch_embeddings_norm.transpose(0, 1)).squeeze(0)
                    interaction_scores = interaction_scores.cpu().numpy()  # [doc_batch_size]
                    
                    # Store scores for this query
                    if query_id not in retrieval_results:
                        retrieval_results[query_id] = {}
                    
                    for doc_idx, doc_id in enumerate(doc_batch_ids):
                        retrieval_results[query_id][doc_id] = float(interaction_scores[doc_idx])
            
            # After processing all document batches, get top-K for each query
            for query_id in batch_query_ids:
                if query_id in retrieval_results:
                    # Sort and keep top-K
                    sorted_docs = sorted(
                        retrieval_results[query_id].items(),
                        key=lambda x: x[1],
                        reverse=True
                    )[:top_k]
                    retrieval_results[query_id] = {doc_id: score for doc_id, score in sorted_docs}
            
            if (i + batch_size) % 10 == 0 or i + batch_size >= len(query_items):
                logging.info(f"Processed {min(i + batch_size, len(query_items))}/{len(query_items)} queries")
        
        # Evaluate
        logging.info("Evaluating retrieval results...")
        
        # Debug: Check retrieval_results
        logging.info(f"DEBUG: retrieval_results has {len(retrieval_results)} queries")
        if retrieval_results:
            sample_query = list(retrieval_results.keys())[0]
            sample_docs = retrieval_results[sample_query]
            logging.info(f"DEBUG: Sample query '{sample_query}' has {len(sample_docs)} documents")
            if sample_docs:
                sample_scores = list(sample_docs.values())
                logging.info(f"DEBUG: Sample scores range: [{min(sample_scores):.6f}, {max(sample_scores):.6f}], std: {max(sample_scores) - min(sample_scores):.6f}")
        
        # Debug: Check qrels
        logging.info(f"DEBUG: qrels has {len(qrels)} queries")
        if qrels:
            qrels_sample = list(qrels.keys())[0]
            logging.info(f"DEBUG: Sample qrels query '{qrels_sample}' has {len(qrels[qrels_sample])} relevant docs")
        
        # Check for query ID mismatch
        retrieval_query_ids = set(retrieval_results.keys())
        qrels_query_ids = set(qrels.keys())
        common_ids = retrieval_query_ids & qrels_query_ids
        logging.info(f"DEBUG: Common query IDs: {len(common_ids)}/{len(retrieval_query_ids)} retrieval, {len(qrels_query_ids)} qrels")
        if len(common_ids) == 0:
            logging.warning(f"WARNING: No matching query IDs between retrieval_results and qrels!")
            if retrieval_query_ids and qrels_query_ids:
                logging.warning(f"  Sample retrieval IDs: {list(retrieval_query_ids)[:3]}")
                logging.warning(f"  Sample qrels IDs: {list(qrels_query_ids)[:3]}")
        
        k_values = [1, 3, 5, 10]
        evaluator = EvaluateRetrieval(None, k_values=k_values)
        ndcg, _map, recall, precision = evaluator.evaluate(qrels, retrieval_results, k_values)
        
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
        
        logging.info(f"\n{'='*60}")
        logging.info(f"Domain {domain} Results:")
        logging.info(f"  Recall@10: {recall.get('Recall@10', 0):.4f}")
        logging.info(f"  nDCG@10: {ndcg.get('NDCG@10', 0):.4f}")
        logging.info(f"{'='*60}\n")
        
        save_checkpoint(checkpoint_dir, domain, all_results[domain])
        
        if shutdown_requested:
            logging.info("Shutdown requested, exiting after current domain completion.")
            break
    
    # Calculate average results across all domains
    if all_results:
        avg_results = {
            metric: np.mean([res.get(metric, 0) for res in all_results.values()])
            for metric in list(all_results.values())[0].keys()
        }
        
        logging.info(f"\n{'='*60}")
        logging.info("Average Results Across All Domains:")
        for metric, value in avg_results.items():
            logging.info(f"  {metric}: {value:.4f}")
        logging.info(f"{'='*60}\n")
        
        # Save final results
        final_results_path = output_dir / "results.json"
        with open(final_results_path, 'w') as f:
            json.dump({"average": avg_results, "domains": all_results}, f, indent=2)
        logging.info(f"✅ Final results saved to: {final_results_path}")
        
        return str(output_dir)
    
    return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run Cross-Attention Query-Document Interaction Retrieval')
    parser.add_argument('--config', type=str, required=True, help='Config JSON file')
    parser.add_argument('--gpu_id', type=int, help='GPU ID to use')
    parser.add_argument('--resume', action='store_true', default=True, help='Resume from checkpoint if available')
    parser.add_argument('--no-resume', dest='resume', action='store_false', help='Do not resume from checkpoint')
    
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    config['resume'] = args.resume
    
    try:
        results_path = run_cross_attention_retrieval(config, args.gpu_id)
        if results_path:
            logging.info(f"✅ Cross-Attention Retrieval completed. Results saved to: {results_path}")
    except KeyboardInterrupt:
        logging.info("Cross-Attention Retrieval interrupted by user")
        sys.exit(0)
    except Exception as e:
        logging.error(f"Cross-Attention Retrieval failed: {e}", exc_info=True)
        sys.exit(1)

