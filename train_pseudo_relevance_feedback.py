"""
Pseudo-Relevance Feedback with LLM Expansion for Multi-Turn RAG Retrieval
Uses initial retrieval results to expand queries using LLM
Medium priority Tier 1 experiment for Task A - Retrieval
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

logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO)

MTRAG_DOMAINS = ["clapnq", "fiqa", "govt", "cloud"]

shutdown_requested = False

def signal_handler(sig, frame):
    """Handle SIGINT (Ctrl+C) gracefully"""
    global shutdown_requested
    logging.info("\n⚠️  Shutdown signal received. Saving checkpoint and exiting gracefully...")
    shutdown_requested = True

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

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
    checkpoint_file = checkpoint_dir / "checkpoint.json"
    
    checkpoint_data = {
        "last_domain": domain,
        "completed_domains": list(results.keys()),
        "results": results,
        "timestamp": datetime.now().isoformat()
    }
    
    with open(checkpoint_file, 'w') as f:
        json.dump(checkpoint_data, f, indent=2)
    
    logging.info(f"✅ Checkpoint saved: {checkpoint_file}")

def expand_query_with_llm(query: str, top_docs: List[str], llm_provider: str = "openai") -> str:
    """Expand query using LLM based on top retrieved documents"""
    # For now, use simple expansion (can be enhanced with actual LLM API)
    # In production, this would call OpenAI/Claude API
    
    if not top_docs:
        return query
    
    # Simple expansion: extract key terms from top documents
    # In production, replace this with actual LLM call
    expanded_terms = []
    for doc in top_docs[:3]:  # Use top 3 documents
        # Simple keyword extraction (can be improved)
        words = doc.lower().split()
        # Filter out common words
        stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        keywords = [w for w in words if len(w) > 3 and w not in stopwords]
        expanded_terms.extend(keywords[:5])  # Top 5 keywords per doc
    
    # Combine original query with expanded terms
    if expanded_terms:
        expanded_query = f"{query} {' '.join(set(expanded_terms[:10]))}"  # Add top 10 unique terms
        return expanded_query
    
    return query

def run_pseudo_relevance_feedback_evaluation(config: Dict, gpu_id: Optional[int] = None) -> Optional[str]:
    """Run pseudo-relevance feedback evaluation with LLM expansion"""
    global shutdown_requested
    
    # Set GPU
    if gpu_id is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        logging.info(f"Using GPU {gpu_id}")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Device: {device}")
    
    output_dir = pathlib.Path(config.get('output_dir', 'experiments/retrieval/pseudo_relevance_feedback'))
    output_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint_dir = output_dir
    resume = config.get('resume', True)
    
    # Load checkpoint if resuming
    all_results = {}
    completed_domains = set()
    if resume:
        checkpoint = load_checkpoint(checkpoint_dir)
        if checkpoint:
            all_results = checkpoint.get('results', {})
            completed_domains = set(checkpoint.get('completed_domains', []))
            logging.info(f"Resuming from checkpoint. Completed domains: {completed_domains}")
    
    domains = config.get('domains', MTRAG_DOMAINS)
    llm_provider = config.get('llm_provider', 'openai')
    top_k_feedback = config.get('top_k_feedback', 10)
    
    # Initialize dense retrieval model
    model_name = config.get('model_path', 'BAAI/bge-base-en-v1.5')
    logging.info(f"Loading base model: {model_name}")
    model = SentenceBERT(model_name, device=device)
    retriever = DenseRetrievalExactSearch(model, batch_size=128)
    k_values = [1, 3, 5, 10]
    evaluator = EvaluateRetrieval(retriever, k_values=k_values)
    
    # Process each domain
    for domain in domains:
        if domain in completed_domains:
            logging.info(f"Skipping {domain} (already completed)")
            continue
        
        if shutdown_requested:
            logging.info("Shutdown requested, saving checkpoint...")
            save_checkpoint(checkpoint_dir, domain, all_results)
            break
        
        logging.info(f"\n{'='*60}")
        logging.info(f"Processing domain: {domain}")
        logging.info(f"{'='*60}")
        
        # Load data
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
        
        # Step 1: Initial retrieval
        logging.info("Step 1: Initial retrieval (top 100)...")
        initial_results = evaluator.retrieve(corpus, queries)
        # Limit to top 100 per query
        initial_results = {qid: dict(list(sorted(docs.items(), key=lambda x: x[1], reverse=True)[:100])) 
                          for qid, docs in initial_results.items()}
        
        # Step 2: Expand queries using pseudo-relevance feedback
        logging.info("Step 2: Expanding queries using pseudo-relevance feedback...")
        expanded_queries = {}
        
        for query_id, query_text in queries.items():
            if query_id not in initial_results:
                expanded_queries[query_id] = query_text
                continue
            
            # Get top documents for feedback
            top_doc_ids = sorted(initial_results[query_id].items(), key=lambda x: x[1], reverse=True)[:top_k_feedback]
            top_docs = [corpus[doc_id].get('text', '') for doc_id, _ in top_doc_ids if doc_id in corpus]
            
            # Expand query using LLM
            expanded_query = expand_query_with_llm(query_text, top_docs, llm_provider)
            expanded_queries[query_id] = expanded_query
        
        # Step 3: Re-retrieve with expanded queries
        logging.info("Step 3: Re-retrieving with expanded queries...")
        # Create temporary query dict for expanded queries
        expanded_query_dict = {qid: expanded_queries[qid] for qid in queries.keys()}
        refined_results = evaluator.retrieve(corpus, expanded_query_dict)
        # Limit to top 100 per query
        refined_results = {qid: dict(list(sorted(docs.items(), key=lambda x: x[1], reverse=True)[:100])) 
                          for qid, docs in refined_results.items()}
        
        # Step 4: Combine initial and refined results
        logging.info("Step 4: Combining initial and refined results...")
        final_results = {}
        
        for query_id in queries.keys():
            initial_scores = initial_results.get(query_id, {})
            refined_scores = refined_results.get(query_id, {})
            
            # Combine using weighted average (can use RRF instead)
            combined_scores = {}
            
            # Add initial results with weight 0.4
            for doc_id, score in initial_scores.items():
                combined_scores[doc_id] = score * 0.4
            
            # Add refined results with weight 0.6
            for doc_id, score in refined_scores.items():
                if doc_id in combined_scores:
                    combined_scores[doc_id] += score * 0.6
                else:
                    combined_scores[doc_id] = score * 0.6
            
            # Sort and take top 100
            final_results[query_id] = dict(sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)[:100])
        
        # Step 5: Evaluate
        logging.info("Step 5: Evaluating results...")
        k_values = [1, 3, 5, 10]
        # Use the existing evaluator (which has the retriever) for evaluation
        ndcg, _map, recall, precision = evaluator.evaluate(qrels, final_results, k_values)
        
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
        
        save_checkpoint(checkpoint_dir, domain, all_results)
        
        if shutdown_requested:
            logging.info("Shutdown requested, exiting after current domain completion.")
            break
    
    # Calculate average results
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
    parser = argparse.ArgumentParser(description='Run Pseudo-Relevance Feedback with LLM Expansion')
    parser.add_argument('--config', type=str, required=True, help='Config JSON file')
    parser.add_argument('--gpu_id', type=int, help='GPU ID to use')
    parser.add_argument('--resume', action='store_true', default=True, help='Resume from checkpoint if available')
    parser.add_argument('--no-resume', dest='resume', action='store_false', help='Do not resume from checkpoint')
    
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    config['resume'] = args.resume
    
    try:
        results_path = run_pseudo_relevance_feedback_evaluation(config, args.gpu_id)
        if results_path:
            logging.info(f"✅ Pseudo-Relevance Feedback completed. Results saved to: {results_path}")
    except KeyboardInterrupt:
        logging.info("Pseudo-Relevance Feedback interrupted by user")
        sys.exit(0)
    except Exception as e:
        logging.error(f"Pseudo-Relevance Feedback failed: {e}", exc_info=True)
        sys.exit(1)

