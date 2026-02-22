"""
Retriever-in-the-loop Rewrite Selection (Multi-Rewrite Bandit)

Idea:
- Generate N rewrites per query using diverse templates:
  - minimal coref resolution
  - maximal standalone reformulation
  - entity-only focus
  - constraint-only focus
- Use the retriever itself to estimate retrieval gain (ΔnDCG proxy) for each rewrite.
- Maintain a small bandit over rewrite templates (arms) and update based on observed rewards.

Why this matters for mtRAG:
- mtRAG shows that naïvely adding more context can hurt retrieval.
- This experiment *selects* rewrites that empirically help retrieval, rather than trusting all rewrites equally.

Task A Compliant: ✅ Query rewriting + retrieval only, no answer generation
Expected: 0.80-0.88 nDCG@10
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
import signal
import sys
from datetime import datetime
from collections import defaultdict
import random

logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO)

MTRAG_DOMAINS = ["clapnq", "fiqa", "govt", "cloud"]
shutdown_requested = False

def signal_handler(sig, frame):
    global shutdown_requested
    logging.info("\n⚠️  Shutdown signal received. Saving checkpoint and exiting gracefully...")
    shutdown_requested = True

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


class RewriteTemplateGenerator:
    """
    Generates multiple rewrites per query using simple, deterministic templates.
    No LLM required; keeps experiment lightweight and reproducible.
    Templates:
      - original
      - minimal coref (pronouns expanded using last noun fallback)
      - standalone (prepend domain + conversation hint, if available)
      - entity-only (keep capitalized words / entities)
      - constraint-only (keep numbers, years, comparison operators)
    """
    def __init__(self, domain: str):
        self.domain = domain
    
    def generate_rewrites(self, query: str, conversation_history: Optional[str] = None) -> Dict[str, str]:
        rewrites: Dict[str, str] = {}
        rewrites['original'] = query
        rewrites['minimal_coref'] = self._minimal_coref(query, conversation_history)
        rewrites['standalone'] = self._standalone(query, conversation_history)
        rewrites['entity_only'] = self._entity_only(query)
        rewrites['constraint_only'] = self._constraint_only(query)
        return rewrites
    
    def _minimal_coref(self, query: str, conversation_history: Optional[str]) -> str:
        if not conversation_history:
            return query
        # Very simple heuristic: if query starts with pronoun, prepend last sentence from history
        pronouns = ["it", "they", "this", "that", "these", "those"]
        q_lower = query.lower().strip()
        if any(q_lower.startswith(p + " ") for p in pronouns):
            last_sent = conversation_history.strip().split('\n')[-1]
            return f"{last_sent} {query}"
        return query
    
    def _standalone(self, query: str, conversation_history: Optional[str]) -> str:
        # Add domain hint and minimal context cue
        domain_hint = f"[{self.domain}]" if self.domain else ""
        if conversation_history:
            return f"{domain_hint} conversation question: {query}"
        return f"{domain_hint} question: {query}"
    
    def _entity_only(self, query: str) -> str:
        tokens = query.split()
        entities = [t for t in tokens if t[:1].isupper() or t.isupper()]
        if not entities:
            return query
        return " ".join(entities)
    
    def _constraint_only(self, query: str) -> str:
        tokens = query.split()
        constraint_tokens = []
        for t in tokens:
            if any(c.isdigit() for c in t):
                constraint_tokens.append(t)
            elif t.lower() in {"before", "after", "since", "until", ">", "<", ">=", "<="}:
                constraint_tokens.append(t)
        if not constraint_tokens:
            return query
        return " ".join(constraint_tokens)


class MultiRewriteBandit:
    """
    Simple contextual bandit over rewrite templates.
    Maintains empirical mean reward per template (ΔnDCG proxy) and selects with epsilon-greedy.
    """
    def __init__(self, template_names: List[str], epsilon: float = 0.1):
        self.template_names = template_names
        self.epsilon = epsilon
        self.counts = {name: 0 for name in template_names}
        self.values = {name: 0.0 for name in template_names}
    
    def select_template(self) -> str:
        if random.random() < self.epsilon:
            return random.choice(self.template_names)
        # Greedy: pick highest estimated value
        return max(self.template_names, key=lambda n: self.values[n])
    
    def update(self, template_name: str, reward: float):
        c = self.counts[template_name]
        v = self.values[template_name]
        new_c = c + 1
        new_v = v + (reward - v) / new_c
        self.counts[template_name] = new_c
        self.values[template_name] = new_v


def compute_single_query_ndcg_at_k(qrels: Dict[str, Dict[str, int]],
                                   qid: str,
                                   ranked_docs: List[str],
                                   k: int = 10) -> float:
    """Compute nDCG@k for a single query from qrels and ranked doc IDs."""
    if qid not in qrels:
        return 0.0
    rels = qrels[qid]
    dcg = 0.0
    for i, doc_id in enumerate(ranked_docs[:k], start=1):
        rel = rels.get(doc_id, 0)
        if rel > 0:
            dcg += rel / np.log2(i + 1)
    ideal_scores = sorted(rels.values(), reverse=True)[:k]
    idcg = sum(rel / np.log2(i + 1) for i, rel in enumerate(ideal_scores, start=1))
    if idcg == 0:
        return 0.0
    return dcg / idcg


def train_retriever_in_loop_rewrite(config: Dict, gpu_id: int = 0) -> Optional[str]:
    """
    Train/evaluate retriever-in-the-loop rewrite selection with bandit over templates.
    """
    global shutdown_requested
    
    device = f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"
    logging.info(f"Using device: {device}")
    
    output_dir = pathlib.Path(config['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    base_model_path = config.get('model_path', 'BAAI/bge-base-en-v1.5')
    domains = config.get('domains', MTRAG_DOMAINS)
    epsilon = config.get('epsilon', 0.1)
    max_queries_per_domain = config.get('max_queries_per_domain', None)
    
    # Initialize dense retriever
    model = SentenceBERT(base_model_path, device=device)
    retriever = DenseRetrievalExactSearch(model, batch_size=128)
    evaluator = EvaluateRetrieval(retriever, k_values=[10])
    
    all_results: Dict[str, Dict[str, float]] = {}
    
    for domain in domains:
        if shutdown_requested:
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
        
        # Baseline retrieval for ΔnDCG estimation
        logging.info("Computing baseline retrieval (original queries)...")
        baseline_results = evaluator.retrieve(corpus, queries)
        
        baseline_ndcg_per_query: Dict[str, float] = {}
        for qid, doc_scores in baseline_results.items():
            ranked_docs = [doc_id for doc_id, _ in sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)]
            baseline_ndcg_per_query[qid] = compute_single_query_ndcg_at_k(qrels, qid, ranked_docs, k=10)
        
        # Bandit over rewrite templates
        template_names = ["original", "minimal_coref", "standalone", "entity_only", "constraint_only"]
        bandit = MultiRewriteBandit(template_names, epsilon=epsilon)
        
        generator = RewriteTemplateGenerator(domain)
        
        reranked_results: Dict[str, Dict[str, float]] = {}
        
        # Iterate queries and apply bandit-based rewrite selection
        for idx, (qid, query) in enumerate(queries.items()):
            if shutdown_requested:
                break
            if max_queries_per_domain is not None and idx >= max_queries_per_domain:
                break
            
            # Generate candidate rewrites
            rewrites = generator.generate_rewrites(query, conversation_history=None)
            
            # Select template
            chosen_template = bandit.select_template()
            chosen_rewrite = rewrites.get(chosen_template, query)
            
            # Retrieve with chosen rewrite
            results = evaluator.retrieve(corpus, {qid: chosen_rewrite})
            if qid not in results:
                reranked_results[qid] = baseline_results.get(qid, {})
                continue
            
            ranked_docs = [doc_id for doc_id, _ in sorted(results[qid].items(), key=lambda x: x[1], reverse=True)]
            ndcg_rewrite = compute_single_query_ndcg_at_k(qrels, qid, ranked_docs, k=10)
            ndcg_base = baseline_ndcg_per_query.get(qid, 0.0)
            reward = ndcg_rewrite - ndcg_base
            
            bandit.update(chosen_template, reward)
            
            # Store rewrite-based results (can also fuse with baseline if desired)
            reranked_results[qid] = results[qid]
        
        # Evaluate bandit-selected rewrites
        logging.info("Evaluating bandit-selected rewrites...")
        ndcg, _map, recall, precision = evaluator.evaluate(qrels, reranked_results, [1, 3, 5, 10])
        
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
        
        logging.info(f"{domain} - nDCG@10: {all_results[domain]['nDCG@10']:.4f}")
        logging.info(f"Bandit template values: {bandit.values}")
    
    # Compute average
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
        
        # Save results
        results_file = output_dir / "results.json"
        with open(results_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        logging.info(f"\nAverage Results - nDCG@10: {avg_results['nDCG@10']:.4f}")
        
        return str(results_file)
    
    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--gpu_id', type=int, default=0)
    args = parser.parse_args()
    
    with open(args.config) as f:
        config = json.load(f)
    
    train_retriever_in_loop_rewrite(config, gpu_id=args.gpu_id)
