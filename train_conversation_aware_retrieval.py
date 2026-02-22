"""
Conversation-Aware Contextual Retrieval with Attention Mechanism
Novel approach: Uses full conversation history with attention, not just last turn
This is the HIGHEST PRIORITY experiment for Task A - Retrieval
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
import torch
import torch.nn as nn
import signal
import sys
from datetime import datetime
import re

# Add re import at top if not already there

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

class ConversationAttentionEncoder(nn.Module):
    """
    Attention mechanism to encode conversation history with focus on relevant turns
    """
    def __init__(self, embedding_dim: int = 768, num_heads: int = 8):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        
        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=num_heads,
            batch_first=True
        )
        
        # Layer norm and feedforward
        self.layer_norm = nn.LayerNorm(embedding_dim)
        self.feedforward = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 2),
            nn.GELU(),
            nn.Linear(embedding_dim * 2, embedding_dim)
        )
        self.layer_norm2 = nn.LayerNorm(embedding_dim)
        
    def forward(self, conversation_embeddings: torch.Tensor, current_query_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            conversation_embeddings: [batch_size, num_turns, embedding_dim]
            current_query_emb: [batch_size, embedding_dim]
        
        Returns:
            Contextual query embedding: [batch_size, embedding_dim]
        """
        batch_size = conversation_embeddings.size(0)
        
        # Use current query as query, conversation turns as key and value
        current_query_expanded = current_query_emb.unsqueeze(1)  # [batch, 1, dim]
        
        # Apply attention: query attends to conversation history
        attn_output, attn_weights = self.attention(
            query=current_query_expanded,
            key=conversation_embeddings,
            value=conversation_embeddings
        )
        
        # Residual connection and layer norm
        attn_output = self.layer_norm(attn_output.squeeze(1) + current_query_emb)
        
        # Feedforward
        ff_output = self.feedforward(attn_output)
        output = self.layer_norm2(ff_output + attn_output)
        
        return output

def parse_conversation_turns(query_text: str) -> Tuple[List[str], str]:
    """
    Parse conversation turns from query text.
    Format: |user|: ... |agent|: ... |user|: ...
    Returns: (conversation_turns, current_query)
    """
    # Split by turn markers
    turns = re.split(r'\|(user|agent)\|:', query_text)
    
    conversation_turns = []
    current_query = ""
    
    for i in range(1, len(turns), 2):
        if i + 1 < len(turns):
            speaker = turns[i].strip()
            text = turns[i + 1].strip()
            
            if text:
                turn_text = f"|{speaker}|: {text}"
                conversation_turns.append(turn_text)
                # Last user turn is the current query
                if speaker == "user":
                    current_query = text
    
    # If no turns found, use entire text as current query
    if not conversation_turns:
        current_query = query_text
        conversation_turns = [query_text]
    
    return conversation_turns, current_query

def encode_conversation_with_attention(
    model: SentenceBERT,
    conversation_turns: List[str],
    current_query: str,
    attention_encoder: Optional[ConversationAttentionEncoder] = None,
    device: str = "cuda"
) -> np.ndarray:
    """
    Encode conversation with attention mechanism.
    If attention_encoder is None, uses simple weighted average.
    """
    # SentenceBERT from BEIR uses q_model (SentenceTransformer) for encoding
    sentence_model = model.q_model
    
    if not conversation_turns:
        # Fallback to current query only
        return sentence_model.encode(current_query, convert_to_numpy=True)
    
    # Encode all conversation turns
    turn_embeddings = sentence_model.encode(conversation_turns, convert_to_numpy=True)
    current_query_emb = sentence_model.encode(current_query, convert_to_numpy=True)
    
    if attention_encoder is None:
        # Simple weighted average: more weight to recent turns
        weights = np.linspace(0.1, 1.0, len(turn_embeddings))
        weights = weights / weights.sum()
        conversation_emb = np.average(turn_embeddings, axis=0, weights=weights)
        # Combine with current query (weighted)
        final_emb = 0.7 * conversation_emb + 0.3 * current_query_emb
        return final_emb
    else:
        # Use attention mechanism
        turn_embeddings_tensor = torch.from_numpy(turn_embeddings).unsqueeze(0).to(device)
        current_query_emb_tensor = torch.from_numpy(current_query_emb).unsqueeze(0).to(device)
        
        with torch.no_grad():
            contextual_emb = attention_encoder(turn_embeddings_tensor, current_query_emb_tensor)
            return contextual_emb.cpu().numpy().squeeze(0)

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
    
    logging.info(f"Checkpoint saved for domain: {domain}")

def run_conversation_aware_retrieval(
    config: Dict,
    gpu_id: Optional[int] = None
) -> str:
    """
    Run conversation-aware retrieval with attention mechanism.
    
    Returns:
        Path to results file
    """
    global shutdown_requested, current_checkpoint_dir
    
    experiment_name = config.get('experiment_name', 'conversation_aware_retrieval')
    model_path = config.get('model_path', 'BAAI/bge-base-en-v1.5')
    domains = config.get('domains', MTRAG_DOMAINS)
    use_data_splits = config.get('use_data_splits', True)
    use_attention = config.get('use_attention', True)
    resume = config.get('resume', True)
    
    # Set GPU
    if gpu_id is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        logging.info(f"Using GPU {gpu_id}")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Device: {device}")
    
    # Setup paths
    data_root = pathlib.Path(".")
    output_dir = pathlib.Path("experiments/retrieval") / experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_dir / "checkpoints"
    current_checkpoint_dir = checkpoint_dir
    
    # Load checkpoint if resuming
    completed_domains = set()
    if resume:
        checkpoint_info = load_checkpoint(checkpoint_dir)
        if checkpoint_info:
            completed_domains = set(checkpoint_info.keys())
            logging.info(f"Resuming: Completed domains: {completed_domains}")
    
    # Load model
    logging.info(f"Loading model: {model_path}")
    model = SentenceBERT(model_path, device=device)
    
    # Initialize attention encoder if using attention
    attention_encoder = None
    if use_attention:
        try:
            # Get embedding dimension from model
            test_emb = model.q_model.encode("test", convert_to_numpy=True)
            embedding_dim = test_emb.shape[0]
            attention_encoder = ConversationAttentionEncoder(embedding_dim=embedding_dim).to(device)
            attention_encoder.eval()
            logging.info("Attention encoder initialized")
        except Exception as e:
            logging.warning(f"Could not initialize attention encoder: {e}. Using weighted average instead.")
            use_attention = False
    
    all_results = {}
    
    for domain in domains:
        if shutdown_requested:
            logging.info("Shutdown requested, saving checkpoint...")
            save_checkpoint(checkpoint_dir, domain, all_results.get(domain, {}))
            break
        
        if domain in completed_domains:
            logging.info(f"Skipping {domain} (already completed)")
            continue
        
        logging.info(f"\n{'='*60}")
        logging.info(f"Evaluating on domain: {domain}")
        logging.info(f"{'='*60}")
        
        # Load test data
        corpus_file = data_root / "corpora" / "passage_level" / f"{domain}.jsonl"
        
        if use_data_splits:
            query_file = data_root / "data_splits" / "retrieval_tasks" / domain / "test" / f"{domain}_questions.jsonl"
            qrels_file = data_root / "data_splits" / "retrieval_tasks" / domain / "test" / "qrels" / "dev.tsv"
        else:
            query_file = data_root / "human" / "retrieval_tasks" / domain / f"{domain}_questions.jsonl"
            qrels_file = data_root / "human" / "retrieval_tasks" / domain / "qrels" / "dev.tsv"
        
        # Fallback to original data if splits don't exist
        if use_data_splits and not query_file.exists():
            logging.warning(f"Data splits not found for {domain}, using original data")
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
        
        # Encode corpus ONCE (not per query batch!)
        # Check for cached corpus embeddings
        cache_dir = output_dir / "cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = cache_dir / f"{domain}_corpus_embeddings.npy"
        
        if cache_file.exists() and resume:
            logging.info(f"Loading cached corpus embeddings from {cache_file}...")
            try:
                corpus_embeddings = np.load(cache_file)
                doc_ids = list(corpus.keys())
                if len(corpus_embeddings) == len(doc_ids):
                    logging.info(f"✅ Loaded cached embeddings for {len(doc_ids)} documents")
                else:
                    logging.warning(f"Cache size mismatch. Re-encoding corpus...")
                    raise FileNotFoundError("Cache mismatch")
            except Exception as e:
                logging.warning(f"Error loading cache: {e}. Re-encoding corpus...")
                corpus_texts = [doc.get("text", "") for doc in corpus.values()]
                logging.info(f"Encoding corpus ({len(corpus_texts)} documents)...")
                corpus_embeddings = model.q_model.encode(corpus_texts, convert_to_numpy=True, batch_size=128)
                # Save cache
                np.save(cache_file, corpus_embeddings)
                logging.info(f"✅ Corpus embeddings cached to {cache_file}")
        else:
            # Encode corpus documents ONCE
            corpus_texts = [doc.get("text", "") for doc in corpus.values()]
            logging.info(f"Encoding corpus ({len(corpus_texts)} documents)...")
            corpus_embeddings = model.q_model.encode(corpus_texts, convert_to_numpy=True, batch_size=128)
            # Save cache
            np.save(cache_file, corpus_embeddings)
            logging.info(f"✅ Corpus embeddings cached to {cache_file}")
        
        doc_ids = list(corpus.keys())
        
        # Conversation-aware retrieval
        logging.info("Running conversation-aware retrieval...")
        retrieval_results = {}
        
        batch_size = 32
        query_items = list(queries.items())
        
        for i in range(0, len(query_items), batch_size):
            if shutdown_requested:
                break
            
            batch_queries = query_items[i:i + batch_size]
            batch_embeddings = []
            
            for query_id, query_text in batch_queries:
                try:
                    # Parse conversation turns
                    conversation_turns, current_query = parse_conversation_turns(query_text)
                    
                    # Encode with attention
                    query_emb = encode_conversation_with_attention(
                        model,
                        conversation_turns,
                        current_query,
                        attention_encoder,
                        device
                    )
                    batch_embeddings.append(query_emb)
                except Exception as e:
                    logging.warning(f"Error processing query {query_id}: {e}. Using simple encoding.")
                    # Fallback to simple encoding
                    query_emb = model.q_model.encode(query_text, convert_to_numpy=True)
                    batch_embeddings.append(query_emb)
            
            # Batch retrieve (corpus embeddings already computed)
            batch_embeddings = np.array(batch_embeddings)
            
            # Compute similarities
            scores = np.dot(batch_embeddings, corpus_embeddings.T)
            
            # Get top-k for each query
            for idx, (query_id, _) in enumerate(batch_queries):
                query_scores = scores[idx]
                top_indices = np.argsort(query_scores)[::-1][:100]
                retrieval_results[query_id] = {
                    doc_ids[j]: float(query_scores[j]) for j in top_indices
                }
            
            if (i + batch_size) % 100 == 0:
                logging.info(f"Processed {min(i + batch_size, len(query_items))}/{len(query_items)} queries")
        
        # Evaluate
        logging.info("Evaluating results...")
        k_values = [1, 3, 5, 10]
        evaluator = EvaluateRetrieval(None, k_values=k_values)
        ndcg, _map, recall, precision = evaluator.evaluate(qrels, retrieval_results, k_values)
        
        domain_results = {
            "Recall@1": recall.get('Recall@1', 0),
            "Recall@3": recall.get('Recall@3', 0),
            "Recall@5": recall.get('Recall@5', 0),
            "Recall@10": recall.get('Recall@10', 0),
            "nDCG@1": ndcg.get('NDCG@1', 0),
            "nDCG@3": ndcg.get('NDCG@3', 0),
            "nDCG@5": ndcg.get('NDCG@5', 0),
            "nDCG@10": ndcg.get('NDCG@10', 0)
        }
        
        all_results[domain] = domain_results
        
        logging.info(f"Domain {domain} - Recall@10: {recall.get('Recall@10', 0):.4f}, nDCG@10: {ndcg.get('NDCG@10', 0):.4f}")
        
        # Save checkpoint after each domain
        save_checkpoint(checkpoint_dir, domain, domain_results)
    
    # Calculate averages
    if all_results:
        avg_results = {
            "Recall@1": np.mean([r["Recall@1"] for r in all_results.values()]),
            "Recall@3": np.mean([r["Recall@3"] for r in all_results.values()]),
            "Recall@5": np.mean([r["Recall@5"] for r in all_results.values()]),
            "Recall@10": np.mean([r["Recall@10"] for r in all_results.values()]),
            "nDCG@1": np.mean([r["nDCG@1"] for r in all_results.values()]),
            "nDCG@3": np.mean([r["nDCG@3"] for r in all_results.values()]),
            "nDCG@5": np.mean([r["nDCG@5"] for r in all_results.values()]),
            "nDCG@10": np.mean([r["nDCG@10"] for r in all_results.values()])
        }
        all_results["average"] = avg_results
        
        logging.info(f"\n{'='*60}")
        logging.info("CONVERSATION-AWARE RETRIEVAL RESULTS:")
        logging.info(f"{'='*60}")
        logging.info(f"Average Recall@10: {avg_results['Recall@10']:.4f}")
        logging.info(f"Average nDCG@10: {avg_results['nDCG@10']:.4f}")
    
    # Save results
    results_file = output_dir / "results.json"
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    logging.info(f"Results saved to: {results_file}")
    logging.info("✅ Conversation-aware evaluation completed")
    
    return str(results_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Conversation-Aware Contextual Retrieval')
    parser.add_argument('--config', type=str, required=True, help='Path to config JSON file')
    parser.add_argument('--gpu_id', type=int, default=None, help='GPU ID to use')
    parser.add_argument('--resume', action='store_true', default=True, help='Resume from checkpoint')
    parser.add_argument('--no-resume', dest='resume', action='store_false', help='Do not resume from checkpoint')
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    config['resume'] = args.resume
    
    try:
        results_file = run_conversation_aware_retrieval(config, args.gpu_id)
        logging.info(f"✅ Experiment completed. Results: {results_file}")
    except KeyboardInterrupt:
        logging.info("⚠️  Experiment interrupted by user")
        sys.exit(1)
    except Exception as e:
        logging.error(f"❌ Experiment failed: {e}", exc_info=True)
        sys.exit(1)

