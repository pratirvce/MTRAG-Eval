"""
Iterative Refinement Retrieval with Feedback
Multiple retrieval rounds with feedback from previous results to refine queries
Novel approach for improving retrieval precision through progressive refinement
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
    logging.info("\n⚠️  Shutdown signal received. Saving progress and exiting gracefully...")
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

def analyze_retrieval_gaps(
    query: str,
    initial_results: Dict[str, float],
    corpus: Dict,
    top_k: int = 10
) -> List[str]:
    """
    Analyze initial retrieval results to identify information gaps.
    Returns list of terms/concepts that might be missing.
    """
    # Get top-k results
    sorted_results = sorted(initial_results.items(), key=lambda x: x[1], reverse=True)[:top_k]
    
    # Extract terms from top results
    retrieved_terms = set()
    for doc_id, score in sorted_results:
        if doc_id in corpus:
            doc_text = corpus[doc_id].get("text", "").lower()
            # Simple term extraction (can be improved)
            terms = doc_text.split()[:50]  # First 50 words
            retrieved_terms.update(terms)
    
    # Extract terms from query
    query_terms = set(query.lower().split())
    
    # Find potential gaps (this is a simple heuristic)
    # In practice, could use more sophisticated methods
    gap_terms = []
    
    # If query has specific terms not in top results, they might be gaps
    # This is a simplified version - can be enhanced with semantic analysis
    return gap_terms

def refine_query_with_feedback(
    original_query: str,
    initial_results: Dict[str, float],
    corpus: Dict,
    model: SentenceBERT,
    device: str
) -> str:
    """
    Refine query based on feedback from initial retrieval results.
    Uses top results to expand/refine the query.
    """
    # Get top-5 results for feedback
    sorted_results = sorted(initial_results.items(), key=lambda x: x[1], reverse=True)[:5]
    
    # Extract key phrases from top results
    feedback_texts = []
    for doc_id, score in sorted_results:
        if doc_id in corpus:
            doc_text = corpus[doc_id].get("text", "")
            # Take first 200 characters as feedback
            feedback_texts.append(doc_text[:200])
    
    # Simple refinement: append key terms from feedback
    # More sophisticated: use semantic similarity to find relevant terms
    if feedback_texts:
        # Combine feedback texts
        feedback_combined = " ".join(feedback_texts)
        # Simple approach: use original query + context from feedback
        # In practice, could use LLM or more sophisticated methods
        refined_query = f"{original_query} {feedback_combined[:100]}"
    else:
        refined_query = original_query
    
    return refined_query

def round1_initial_retrieval(
    model_path: str,
    corpus: Dict,
    queries: Dict,
    top_k: int = 100,
    device: str = "cuda"
) -> Dict:
    """Round 1: Initial dense retrieval"""
    logging.info(f"Round 1: Initial retrieval (top-{top_k})...")
    model = SentenceBERT(model_path, device=device)
    retriever = DenseRetrievalExactSearch(model, batch_size=128)
    evaluator = EvaluateRetrieval(retriever, k_values=[top_k])
    results = evaluator.retrieve(corpus, queries)
    
    # Limit to top_k
    limited_results = {}
    for qid, doc_scores in results.items():
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        limited_results[qid] = {doc_id: score for doc_id, score in sorted_docs}
    
    return limited_results

def round2_refined_retrieval(
    model_path: str,
    corpus: Dict,
    queries: Dict,
    round1_results: Dict,
    top_k: int = 100,
    device: str = "cuda"
) -> Dict:
    """Round 2: Refined retrieval with feedback"""
    logging.info(f"Round 2: Refined retrieval with feedback (top-{top_k})...")
    model = SentenceBERT(model_path, device=device)
    
    # Refine queries based on round 1 results
    refined_queries = {}
    for qid, query_text in queries.items():
        if qid in round1_results:
            refined_query = refine_query_with_feedback(
                query_text,
                round1_results[qid],
                corpus,
                model,
                device
            )
            refined_queries[qid] = refined_query
        else:
            refined_queries[qid] = query_text
    
    # Retrieve with refined queries
    retriever = DenseRetrievalExactSearch(model, batch_size=128)
    evaluator = EvaluateRetrieval(retriever, k_values=[top_k])
    results = evaluator.retrieve(corpus, refined_queries)
    
    # Limit to top_k
    limited_results = {}
    for qid, doc_scores in results.items():
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        limited_results[qid] = {doc_id: score for doc_id, score in sorted_docs}
    
    return limited_results

def combine_and_rerank(
    round1_results: Dict,
    round2_results: Dict,
    final_top_k: int = 100
) -> Dict:
    """
    Combine results from both rounds and rerank.
    Uses reciprocal rank fusion (RRF) to combine rankings.
    """
    logging.info(f"Combining and reranking results (final top-{final_top_k})...")
    
    combined_results = {}
    
    for qid in set(list(round1_results.keys()) + list(round2_results.keys())):
        # Get results from both rounds
        r1_docs = round1_results.get(qid, {})
        r2_docs = round2_results.get(qid, {})
        
        # Reciprocal Rank Fusion (RRF)
        # RRF score = sum(1 / (k + rank)) for each document
        rrf_scores = {}
        k = 60  # RRF constant
        
        # Add scores from round 1
        sorted_r1 = sorted(r1_docs.items(), key=lambda x: x[1], reverse=True)
        for rank, (doc_id, score) in enumerate(sorted_r1, start=1):
            if doc_id not in rrf_scores:
                rrf_scores[doc_id] = 0.0
            rrf_scores[doc_id] += 1.0 / (k + rank)
        
        # Add scores from round 2
        sorted_r2 = sorted(r2_docs.items(), key=lambda x: x[1], reverse=True)
        for rank, (doc_id, score) in enumerate(sorted_r2, start=1):
            if doc_id not in rrf_scores:
                rrf_scores[doc_id] = 0.0
            rrf_scores[doc_id] += 1.0 / (k + rank)
        
        # Sort by RRF score and take top_k
        sorted_final = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:final_top_k]
        combined_results[qid] = {doc_id: float(score) for doc_id, score in sorted_final}
    
    return combined_results

def run_iterative_refinement_evaluation(config: Dict, gpu_id: Optional[int] = None) -> Optional[str]:
    """Run iterative refinement retrieval evaluation"""
    experiment_name = config.get("experiment_name", "iterative_refinement")
    model_path = config.get("model_path", "BAAI/bge-base-en-v1.5")
    domains = config.get("domains", MTRAG_DOMAINS)
    use_data_splits = config.get("use_data_splits", True)
    round1_top_k = config.get("round1_top_k", 100)
    round2_top_k = config.get("round2_top_k", 100)
    final_top_k = config.get("final_top_k", 100)
    resume = config.get("resume", True)
    
    # Setup device
    if gpu_id is not None:
        # Set CUDA_VISIBLE_DEVICES before importing torch operations
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        device = "cuda:0"  # After setting CUDA_VISIBLE_DEVICES, GPU becomes 0
        logging.info(f"Using GPU {gpu_id} (visible as cuda:0)")
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logging.info(f"Using device: {device}")
    
    # Verify device is available
    if device.startswith("cuda") and not torch.cuda.is_available():
        logging.error("CUDA requested but not available. Falling back to CPU.")
        device = "cpu"
    
    logging.info(f"Device confirmed: {device}")
    
    # Setup output directory
    output_dir = pathlib.Path("experiments/retrieval") / experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_dir / "checkpoints"
    
    # Load checkpoint if resuming
    checkpoint_info = None
    if resume:
        checkpoint_info = load_checkpoint(checkpoint_dir)
        if checkpoint_info:
            completed_domains = set(checkpoint_info.keys())
            logging.info(f"Resuming: Completed domains: {completed_domains}")
    
    # Load model
    logging.info(f"Loading model: {model_path}")
    model = SentenceBERT(model_path, device=device)
    
    data_root = pathlib.Path(".")
    all_results = {}
    
    for domain in domains:
        if shutdown_requested:
            break
        
        # Skip if already completed
        if checkpoint_info and domain in checkpoint_info:
            logging.info(f"Skipping {domain} (already completed)")
            all_results[domain] = checkpoint_info[domain]["results"]
            continue
        
        logging.info(f"\n{'='*60}")
        logging.info(f"Evaluating on domain: {domain}")
        logging.info(f"{'='*60}")
        
        # Load data
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
        
        # Round 1: Initial retrieval
        round1_results = round1_initial_retrieval(
            model_path,
            corpus,
            queries,
            top_k=round1_top_k,
            device=device
        )
        
        if shutdown_requested:
            break
        
        # Round 2: Refined retrieval
        round2_results = round2_refined_retrieval(
            model_path,
            corpus,
            queries,
            round1_results,
            top_k=round2_top_k,
            device=device
        )
        
        if shutdown_requested:
            break
        
        # Combine and rerank
        final_results = combine_and_rerank(
            round1_results,
            round2_results,
            final_top_k=final_top_k
        )
        
        # Evaluate
        logging.info("Evaluating final results...")
        k_values = [1, 3, 5, 10]
        evaluator = EvaluateRetrieval(None, k_values=k_values)
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
        
        logging.info(f"Domain {domain} - Recall@10: {recall.get('Recall@10', 0):.4f}, nDCG@10: {ndcg.get('NDCG@10', 0):.4f}")
        save_checkpoint(checkpoint_dir, domain, all_results[domain])
        
        if shutdown_requested:
            logging.info("Shutdown requested, exiting after current domain completion.")
            break
    
    # Calculate average
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
        logging.info("Average Results:")
        logging.info(f"Recall@10: {avg_results['Recall@10']:.4f}")
        logging.info(f"nDCG@10: {avg_results['nDCG@10']:.4f}")
        logging.info(f"{'='*60}")
    
    # Save final results
    results_file = output_dir / "results.json"
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    logging.info(f"Results saved to: {results_file}")
    return str(results_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Iterative Refinement Retrieval with Feedback')
    parser.add_argument('--config', type=str, required=True, help='Config JSON file')
    parser.add_argument('--gpu_id', type=int, help='GPU ID')
    parser.add_argument('--resume', action='store_true', default=True, help='Resume from checkpoint if available')
    parser.add_argument('--no-resume', dest='resume', action='store_false', help='Do not resume from checkpoint')
    
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    config['resume'] = args.resume
    
    try:
        results_file = run_iterative_refinement_evaluation(config, args.gpu_id)
        if results_file:
            logging.info(f"✅ Evaluation completed. Results saved to: {results_file}")
    except KeyboardInterrupt:
        logging.info("Evaluation interrupted by user")
        sys.exit(0)
    except Exception as e:
        logging.error(f"Evaluation failed: {e}", exc_info=True)
        sys.exit(1)

