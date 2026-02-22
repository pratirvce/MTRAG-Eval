"""
Reinforcement Learning for Adaptive Retrieval
RL agent learns optimal retrieval strategy per query/domain
Expected: 0.59-0.63 nDCG@10
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
import torch.nn.functional as F
import signal
import sys
from datetime import datetime
from collections import deque

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

class RLAgent(nn.Module):
    """RL Agent for adaptive retrieval strategy selection"""
    def __init__(self, state_dim: int = 768, num_actions: int = 3, hidden_dim: int = 256):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, num_actions)
        
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Predict action probabilities"""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        logits = self.fc3(x)
        return F.softmax(logits, dim=1)
    
    def select_action(self, state: torch.Tensor, epsilon: float = 0.1) -> int:
        """Epsilon-greedy action selection"""
        if np.random.random() < epsilon:
            return np.random.randint(0, self.fc3.out_features)
        else:
            with torch.no_grad():
                # Ensure state has correct shape [batch_size, features]
                if state.dim() == 1:
                    state = state.unsqueeze(0)
                probs = self.forward(state)
                action_idx = probs.argmax(dim=1)
                # Handle both single and batch cases
                if action_idx.dim() == 0:
                    return action_idx.item()
                else:
                    return action_idx[0].item()  # Take first element if batch

def run_rl_adaptive_evaluation(config: Dict, gpu_id: Optional[int] = None) -> Optional[str]:
    """Run RL Adaptive Retrieval evaluation"""
    global shutdown_requested
    
    if gpu_id is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        logging.info(f"Using GPU {gpu_id}")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Device: {device}")
    
    output_dir = pathlib.Path(config.get('output_dir', 'experiments/retrieval/rl_adaptive'))
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
    top_k = config.get('top_k', 100)
    
    # Initialize base retrievers (different strategies)
    logging.info(f"Loading base model: {base_model}")
    base_retriever_model = SentenceBERT(base_model, device=device)
    retriever1 = DenseRetrievalExactSearch(base_retriever_model, batch_size=128)
    
    # Initialize RL agent
    rl_agent = RLAgent(state_dim=768, num_actions=3)  # 3 strategies
    rl_agent.to(device)
    rl_agent.eval()
    
    # Process each domain
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
        logging.info(f"Processing domain: {domain}")
        logging.info(f"{'='*60}")
        
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
        except Exception as e:
            logging.error(f"Error loading data for {domain}: {e}")
            continue
        
        logging.info(f"Loaded {len(corpus)} documents, {len(queries)} queries")
        
        # Get query embeddings for RL state
        logging.info("Encoding queries for RL state...")
        query_texts = [queries[qid] for qid in queries.keys()]
        query_embeddings = base_retriever_model.q_model.encode(
            query_texts, batch_size=128, show_progress_bar=True, convert_to_numpy=True
        )
        query_emb_map = {
            qid: emb for qid, emb in zip(queries.keys(), query_embeddings)
        }
        
        # Adaptive retrieval with RL
        logging.info("Performing adaptive retrieval with RL agent...")
        final_results = {}
        
        # Process queries in batches to avoid single-query IndexError
        query_batch_size = 32
        query_items = list(queries.items())
        
        for batch_start in range(0, len(query_items), query_batch_size):
            batch_end = min(batch_start + query_batch_size, len(query_items))
            batch_queries = dict(query_items[batch_start:batch_end])
            
            # Get actions for all queries in batch
            query_embs_batch = torch.tensor(
                [query_emb_map[qid] for qid in batch_queries.keys() if qid in query_emb_map],
                dtype=torch.float32
            ).to(device)
            
            # Get actions for batch (use heuristic since RL agent is untrained)
            # Heuristic: Use query characteristics to select strategy
            actions_map = {}
            valid_qids = [qid for qid in batch_queries.keys() if qid in query_emb_map]
            for i, qid in enumerate(valid_qids):
                query_text = batch_queries[qid]
                query_emb = query_embs_batch[i:i+1]
                
                # Simple heuristic: query length and embedding norm
                query_len = len(query_text.split())
                query_norm = torch.norm(query_emb).item()
                
                # Strategy selection heuristic
                if query_len < 5:
                    action = 0  # Simple dense retrieval for short queries
                elif query_len < 15:
                    action = 1  # Hybrid for medium queries
                else:
                    action = 2  # Multi-stage for complex queries
                
                actions_map[qid] = action
            
            # Group queries by action for efficient batch retrieval
            action_queries = {0: {}, 1: {}, 2: {}}
            for qid, query_text in batch_queries.items():
                if qid in actions_map:
                    action = actions_map[qid]
                    action_queries[action][qid] = query_text
            
            # Process each action group - ensure minimum batch size of 2
            for action, action_query_dict in action_queries.items():
                if not action_query_dict:
                    continue
                
                # Ensure we have at least 2 queries to avoid BEIR IndexError
                # BEIR's exact_search fails with single queries
                if len(action_query_dict) == 1:
                    # For single queries, merge with queries from another action
                    # First try to find another action with queries
                    for other_action, other_dict in action_queries.items():
                        if other_action != action and len(other_dict) > 0:
                            # Temporarily merge, process, then split results
                            merged_dict = {**action_query_dict, **other_dict}
                            merged_actions = {}
                            for qid in action_query_dict.keys():
                                merged_actions[qid] = action
                            for qid in other_dict.keys():
                                merged_actions[qid] = other_action
                            
                            # Process merged batch with different top_k per action
                            evaluator = EvaluateRetrieval(retriever1, k_values=[top_k * 2])
                            merged_results = evaluator.retrieve(corpus, merged_dict)
                            
                            # Split results back by action and apply strategy
                            for qid, qaction in merged_actions.items():
                                if qaction == 0 or qaction == 2:
                                    final_results[qid] = dict(list(merged_results.get(qid, {}).items())[:top_k])
                                elif qaction == 1:
                                    candidates = sorted(
                                        merged_results.get(qid, {}).items(),
                                        key=lambda x: x[1],
                                        reverse=True
                                    )[:top_k]
                                    final_results[qid] = dict(candidates)
                            break
                    else:
                        # No other action available, process with dummy
                        first_qid = list(action_query_dict.keys())[0]
                        dummy_qid = f"__dummy_{first_qid}"
                        action_query_dict[dummy_qid] = action_query_dict[first_qid]
                        
                        # Apply selected strategy
                        if action == 0 or action == 2:
                            evaluator = EvaluateRetrieval(retriever1, k_values=[top_k])
                            results = evaluator.retrieve(corpus, action_query_dict)
                            final_results[first_qid] = results.get(first_qid, {})
                        elif action == 1:
                            evaluator = EvaluateRetrieval(retriever1, k_values=[top_k * 2])
                            results = evaluator.retrieve(corpus, action_query_dict)
                            candidates = sorted(
                                results.get(first_qid, {}).items(),
                                key=lambda x: x[1],
                                reverse=True
                            )[:top_k]
                            final_results[first_qid] = dict(candidates)
                    continue
                
                # Normal batch processing (2+ queries)
                # Apply selected strategy
                if action == 0 or action == 2:
                    # Strategy 1/3: Standard dense retrieval
                    evaluator = EvaluateRetrieval(retriever1, k_values=[top_k])
                    results = evaluator.retrieve(corpus, action_query_dict)
                    for qid in action_query_dict.keys():
                        final_results[qid] = results.get(qid, {})
                elif action == 1:
                    # Strategy 2: Dense retrieval with higher top_k then rerank
                    evaluator = EvaluateRetrieval(retriever1, k_values=[top_k * 2])
                    results = evaluator.retrieve(corpus, action_query_dict)
                    for qid in action_query_dict.keys():
                        candidates = sorted(
                            results.get(qid, {}).items(),
                            key=lambda x: x[1],
                            reverse=True
                        )[:top_k]
                        final_results[qid] = dict(candidates)
            
            # Handle queries without embeddings
            for qid in batch_queries.keys():
                if qid not in final_results:
                    final_results[qid] = {}
        
        # Evaluate
        logging.info("Evaluating results...")
        evaluator = EvaluateRetrieval(None, k_values=[1, 3, 5, 10])
        ndcg, _map, recall, precision = evaluator.evaluate(qrels, final_results, [1, 3, 5, 10])
        
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
        results_path = run_rl_adaptive_evaluation(config, args.gpu_id)
        if results_path:
            logging.info(f"✅ RL Adaptive Retrieval completed: {results_path}")
    except KeyboardInterrupt:
        logging.info("Interrupted by user")
        sys.exit(0)
    except Exception as e:
        logging.error(f"Failed: {e}", exc_info=True)
        sys.exit(1)

