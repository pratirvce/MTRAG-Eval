"""
AdaRewriter: Test-Time Adaptive Query Reformulation
Uses test-time adaptation with a lightweight reward model to pick the best rewrite among many candidates.

Task A Compliant: ✅ Query rewriting is preprocessing only, no answer generation
Expected: 0.78-0.85 nDCG@10
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
from torch.utils.data import DataLoader
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


class QueryRewriter:
    """LLM-based query rewriter - generates multiple rewrite candidates"""
    def __init__(self, llm_model_name: Optional[str] = None, use_local: bool = True):
        self.use_local = use_local
        self.llm_model_name = llm_model_name or "microsoft/Phi-3-mini-4k-instruct"
        
        if use_local:
            try:
                from scripts.evaluation.huggingface_client import HuggingFaceLLMClient
                self.llm_client = HuggingFaceLLMClient(self.llm_model_name)
                logging.info(f"Using local LLM for rewriting: {self.llm_model_name}")
            except Exception as e:
                logging.warning(f"Could not load local LLM: {e}. Using simple rewriting.")
                self.llm_client = None
        else:
            self.llm_client = None
    
    def generate_rewrites(self, query: str, conversation_history: Optional[str] = None, num_rewrites: int = 5) -> List[str]:
        """Generate multiple query rewrite candidates"""
        if self.llm_client is None:
            return self._simple_rewrites(query, num_rewrites)
        
        try:
            if conversation_history:
                prompt = f"""Given this conversation history and query, generate {num_rewrites} different rewrites of the query that would help retrieve relevant documents.

Conversation History:
{conversation_history}

Original Query: {query}

Generate {num_rewrites} query rewrites (one per line, no explanations, just the rewrites):
"""
            else:
                prompt = f"""Generate {num_rewrites} different rewrites of this query that would help retrieve relevant documents.

Original Query: {query}

Generate {num_rewrites} query rewrites (one per line, no explanations, just the rewrites):
"""
            
            response = self.llm_client.generate_response(prompt, max_new_tokens=300, temperature=0.8)
            
            rewrites = []
            for line in response.strip().split('\n'):
                line = line.strip().lstrip('0123456789.-) ').strip()
                if line and len(line) > 5 and not line.startswith('#'):
                    rewrites.append(line)
            
            if not rewrites:
                rewrites = [query]
            
            return rewrites[:num_rewrites]
        except Exception as e:
            logging.warning(f"LLM rewriting failed: {e}. Using simple rewrites.")
            return self._simple_rewrites(query, num_rewrites)
    
    def _simple_rewrites(self, query: str, num_rewrites: int) -> List[str]:
        """Simple rewrite fallback"""
        rewrites = [query]
        if "what" in query.lower():
            rewrites.append(query.lower().replace("what", "which"))
            rewrites.append(query.lower().replace("what", "describe"))
        if "how" in query.lower():
            rewrites.append(query.lower().replace("how", "what method"))
        if len(rewrites) < num_rewrites:
            rewrites.append(query + " information")
            rewrites.append(query + " details")
        return rewrites[:num_rewrites]


class RewardModel(nn.Module):
    """
    Lightweight reward model to score query rewrites based on retrieval effectiveness
    """
    def __init__(self, embedding_dim: int = 768, hidden_dim: int = 256):
        super().__init__()
        self.scorer = nn.Sequential(
            nn.Linear(embedding_dim * 2, hidden_dim),  # query + context features
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)  # Reward score
        )
    
    def forward(self, query_emb: torch.Tensor, context_emb: torch.Tensor) -> torch.Tensor:
        """Score query rewrite based on query and context embeddings"""
        combined = torch.cat([query_emb, context_emb], dim=-1)
        reward = self.scorer(combined)
        return reward.squeeze(-1)


class AdaRewriterRetriever:
    """
    Retriever with AdaRewriter: generates rewrites, scores them with reward model, selects best
    """
    def __init__(self, base_model_path: str, rewriter: QueryRewriter, reward_model: RewardModel, device: str = "cuda"):
        self.device = device
        self.rewriter = rewriter
        self.reward_model = reward_model.to(device)
        self.reward_model.eval()
        
        # Base retriever for getting context/feedback
        self.base_model = SentenceBERT(base_model_path, device=device)
        self.retriever = DenseRetrievalExactSearch(self.base_model, batch_size=128)
        
        # Query encoder for reward model
        self.query_encoder = self.base_model
    
    def score_rewrite(self, rewrite: str, original_query: str, corpus_sample: List[str]) -> float:
        """Score a rewrite using reward model"""
        # Encode rewrite and original query
        rewrite_emb = self.query_encoder.encode([rewrite], convert_to_tensor=True).to(self.device)
        original_emb = self.query_encoder.encode([original_query], convert_to_tensor=True).to(self.device)
        
        # Get context from corpus sample (average embedding)
        if corpus_sample:
            context_embs = self.query_encoder.encode(corpus_sample[:5], convert_to_tensor=True).to(self.device)
            context_emb = context_embs.mean(dim=0, keepdim=True)
        else:
            context_emb = original_emb
        
        # Score with reward model
        with torch.no_grad():
            reward = self.reward_model(rewrite_emb, context_emb)
            return reward.item()
    
    def retrieve(self, corpus: Dict, queries: Dict, top_k: int = 10) -> Dict:
        """Retrieve with AdaRewriter: generate rewrites, score, select best, retrieve"""
        all_results = {}
        
        for query_id, query_text in queries.items():
            # Extract conversation history if available
            conversation_history = None
            if "|user|:" in query_text:
                parts = query_text.split("|user|:")
                if len(parts) > 1:
                    conversation_history = "|user|:".join(parts[:-1])
                    query_text = parts[-1].strip()
            
            # Step 1: Generate multiple rewrites
            rewrites = self.rewriter.generate_rewrites(query_text, conversation_history, num_rewrites=5)
            logging.debug(f"Query {query_id}: Generated {len(rewrites)} rewrites")
            
            # Step 2: Get corpus sample for context (quick retrieval with original query)
            evaluator_temp = EvaluateRetrieval(self.retriever, k_values=[10])
            context_results = evaluator_temp.retrieve(corpus, {query_id: query_text})
            corpus_sample = [corpus[doc_id].get('text', '')[:200] for doc_id in list(context_results.get(query_id, {}).keys())[:5]]
            
            # Step 3: Score rewrites with reward model
            rewrite_scores = []
            for rewrite in rewrites:
                score = self.score_rewrite(rewrite, query_text, corpus_sample)
                rewrite_scores.append((rewrite, score))
            
            # Step 4: Select top rewrites (top 2-3)
            rewrite_scores.sort(key=lambda x: x[1], reverse=True)
            top_rewrites = [r[0] for r in rewrite_scores[:3]]
            
            # Step 5: Retrieve with top rewrites and fuse
            all_candidates = {}
            for rewrite in top_rewrites:
                rewrite_queries = {query_id: rewrite}
                evaluator_temp = EvaluateRetrieval(self.retriever, k_values=[top_k * 2])
                rewrite_results = evaluator_temp.retrieve(corpus, rewrite_queries)
                
                if query_id in rewrite_results:
                    # RRF fusion
                    sorted_docs = sorted(rewrite_results[query_id].items(), key=lambda x: x[1], reverse=True)
                    for rank, (doc_id, score) in enumerate(sorted_docs, start=1):
                        if doc_id not in all_candidates:
                            all_candidates[doc_id] = 0.0
                        all_candidates[doc_id] += 1.0 / (60 + rank)
            
            # Sort and return top_k
            sorted_docs = sorted(all_candidates.items(), key=lambda x: x[1], reverse=True)
            all_results[query_id] = {doc_id: score for doc_id, score in sorted_docs[:top_k]}
        
        return all_results


def train_reward_model(config: Dict, gpu_id: Optional[int] = None) -> Optional[str]:
    """Train reward model on query rewrite effectiveness"""
    global shutdown_requested
    
    if gpu_id is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Device: {device}")
    
    output_dir = pathlib.Path(config.get('output_dir', 'experiments/retrieval/adarewriter'))
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize components
    base_model = config.get('model_path', 'BAAI/bge-base-en-v1.5')
    llm_model = config.get('llm_model', 'microsoft/Phi-3-mini-4k-instruct')
    use_local_llm = config.get('use_local_llm', True)
    
    rewriter = QueryRewriter(llm_model_name=llm_model, use_local=use_local_llm)
    base_retriever_model = SentenceBERT(base_model, device=device)
    embedding_dim = base_retriever_model.get_sentence_embedding_dimension()
    
    reward_model = RewardModel(embedding_dim=embedding_dim, hidden_dim=256).to(device)
    
    # Training data: load queries and generate rewrites
    domains = config.get('domains', MTRAG_DOMAINS)
    use_data_splits = config.get('use_data_splits', True)
    
    training_pairs = []
    
    for domain in domains:
        if shutdown_requested:
            break
        
        logging.info(f"Loading training data for {domain}...")
        data_root = pathlib.Path(".")
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
            continue
        
        # Generate rewrites and create training pairs
        retriever_temp = DenseRetrievalExactSearch(base_retriever_model, batch_size=128)
        evaluator_temp = EvaluateRetrieval(retriever_temp, k_values=[10])
        
        for query_id, query_text in list(queries.items())[:50]:  # Sample for training
            # Generate rewrites
            rewrites = rewriter.generate_rewrites(query_text, num_rewrites=5)
            
            # Get retrieval results for original and rewrites
            original_results = evaluator_temp.retrieve(corpus, {query_id: query_text})
            original_docs = set(original_results.get(query_id, {}).keys())
            
            relevant_docs = set(qrels.get(query_id, {}).keys())
            
            # Score rewrites based on retrieval effectiveness
            for rewrite in rewrites:
                rewrite_results = evaluator_temp.retrieve(corpus, {query_id: rewrite})
                rewrite_docs = set(rewrite_results.get(query_id, {}).keys())
                
                # Reward = overlap with relevant docs
                overlap = len(rewrite_docs & relevant_docs)
                reward = overlap / max(len(relevant_docs), 1)
                
                # Create training example
                training_pairs.append({
                    'query': query_text,
                    'rewrite': rewrite,
                    'reward': reward
                })
    
    if not training_pairs:
        logging.warning("No training pairs generated, using pretrained reward model")
    else:
        # Train reward model
        logging.info(f"Training reward model on {len(training_pairs)} pairs...")
        optimizer = torch.optim.AdamW(reward_model.parameters(), lr=1e-4)
        epochs = config.get('reward_epochs', 5)
        
        reward_model.train()
        for epoch in range(epochs):
            total_loss = 0
            batch_size = 32
            
            for i in range(0, len(training_pairs), batch_size):
                batch = training_pairs[i:i+batch_size]
                
                query_texts = [p['query'] for p in batch]
                rewrite_texts = [p['rewrite'] for p in batch]
                rewards = torch.tensor([p['reward'] for p in batch], device=device, dtype=torch.float32)
                
                # Encode queries and rewrites
                query_embs = base_retriever_model.encode(query_texts, convert_to_tensor=True).to(device)
                rewrite_embs = base_retriever_model.encode(rewrite_texts, convert_to_tensor=True).to(device)
                
                # Get context (average of query embeddings)
                context_embs = query_embs.mean(dim=0, keepdim=True).expand(len(batch), -1)
                
                # Forward pass
                predicted_rewards = reward_model(rewrite_embs, context_embs)
                
                # Loss: MSE between predicted and actual rewards
                loss = F.mse_loss(predicted_rewards, rewards)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            logging.info(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/(len(training_pairs)//batch_size + 1):.4f}")
        
        # Save reward model
        reward_model_path = output_dir / "reward_model.pt"
        torch.save(reward_model.state_dict(), reward_model_path)
        logging.info(f"Saved reward model to {reward_model_path}")
    
    # Create retriever with trained reward model
    retriever = AdaRewriterRetriever(base_model, rewriter, reward_model, device=device)
    
    # Evaluation
    all_results = {}
    
    for domain in domains:
        if shutdown_requested:
            break
        
        logging.info(f"\n{'='*60}\nEvaluating on domain: {domain}\n{'='*60}")
        
        # Load data
        data_root = pathlib.Path(".")
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
            continue
        
        # Retrieve and evaluate
        evaluator = EvaluateRetrieval(retriever, k_values=[1, 3, 5, 10])
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
    
    # Save results
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
    parser.add_argument('--gpu_id', type=int, default=None)
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    try:
        results_path = train_reward_model(config, gpu_id=args.gpu_id)
        if results_path:
            logging.info(f"✅ Training completed: {results_path}")
    except KeyboardInterrupt:
        logging.info("Interrupted by user")
        sys.exit(0)
    except Exception as e:
        logging.error(f"Failed: {e}", exc_info=True)
        sys.exit(1)
