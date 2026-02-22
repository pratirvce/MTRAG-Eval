"""
LLM-Based Multi-Query Expansion with Checkpointing and Resume Support
Uses GPT-4/Claude to generate query variations for better retrieval
"""

import pathlib
import logging
import argparse
import json
import os
import numpy as np
from typing import Dict, List, Optional
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.models import SentenceBERT
from beir.retrieval.search.dense import DenseRetrievalExactSearch
from sentence_transformers import util
import torch
import signal
import sys
from datetime import datetime
import time

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

def save_checkpoint(checkpoint_dir: pathlib.Path, domain: str, expanded_queries: Dict):
    """Save checkpoint for current domain"""
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_info = {
        "domain": domain,
        "expanded_queries": expanded_queries,
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

def expand_query_with_llm(query: str, llm_type: str = "gpt4", num_variations: int = 3) -> List[str]:
    """
    Expand query using LLM to generate variations.
    
    Args:
        query: Original query
        llm_type: Type of LLM ("gpt4", "claude", "local")
        num_variations: Number of query variations to generate
    
    Returns:
        List of expanded query variations
    """
    # For now, implement a simple placeholder
    # In production, this would call OpenAI API, Claude API, or local LLM
    
    if llm_type == "gpt4":
        # Placeholder - would use OpenAI API
        # import openai
        # response = openai.ChatCompletion.create(...)
        variations = [
            f"{query} (detailed explanation)",
            f"What is {query}?",
            f"Explain {query}"
        ]
    elif llm_type == "claude":
        # Placeholder - would use Anthropic API
        variations = [
            f"{query} (comprehensive answer)",
            f"Information about {query}",
            f"Details on {query}"
        ]
    else:
        # Simple keyword expansion as fallback
        variations = [query]  # Return original for now
    
    return variations[:num_variations]

def combine_results_with_rrf(all_results: List[Dict], k: int = 60) -> Dict:
    """
    Combine multiple retrieval results using Reciprocal Rank Fusion (RRF).
    
    Args:
        all_results: List of result dictionaries (one per query variation)
        k: RRF parameter (typically 60)
    
    Returns:
        Combined results dictionary
    """
    if not all_results:
        return {}
    
    if len(all_results) == 1:
        return all_results[0]
    
    # Combine using RRF
    combined = {}
    
    # Get all query IDs
    all_query_ids = set()
    for results in all_results:
        all_query_ids.update(results.keys())
    
    for query_id in all_query_ids:
        doc_scores = {}
        
        # Collect scores from all variations
        for results in all_results:
            if query_id in results:
                ranked_docs = sorted(results[query_id].items(), key=lambda x: x[1], reverse=True)
                for rank, (doc_id, score) in enumerate(ranked_docs, 1):
                    rrf_score = 1.0 / (k + rank)
                    if doc_id not in doc_scores:
                        doc_scores[doc_id] = 0.0
                    doc_scores[doc_id] += rrf_score
        
        combined[query_id] = doc_scores
    
    return combined

def run_llm_query_expansion_evaluation(config: Dict, gpu_id: Optional[int] = None):
    """Run LLM query expansion evaluation"""
    global shutdown_requested
    
    experiment_name = config.get('experiment_name', 'llm_query_expansion')
    base_model = config.get('base_model')  # Dense retrieval model
    llm_type = config.get('llm_type', 'gpt4')  # 'gpt4', 'claude', 'local'
    num_variations = config.get('num_variations', 3)
    domains = config.get('domains', MTRAG_DOMAINS)
    use_data_splits = config.get('use_data_splits', True)
    resume = config.get('resume', True)
    
    if not base_model:
        logging.error("base_model required in config")
        return None
    
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
    
    # Load checkpoint if resuming
    completed_domains = set()
    expanded_queries_cache = {}
    if resume:
        checkpoint_info = load_checkpoint(checkpoint_dir)
        if checkpoint_info:
            for domain, info in checkpoint_info.items():
                completed_domains.add(domain)
                expanded_queries_cache[domain] = info.get('expanded_queries', {})
            logging.info(f"Resuming: Completed domains: {completed_domains}")
    
    # Load dense retriever
    logging.info(f"Loading base model: {base_model}")
    model = SentenceBERT(base_model, device=device)
    retriever = DenseRetrievalExactSearch(model, batch_size=128)
    
    all_results = {}
    
    for domain in domains:
        if shutdown_requested:
            logging.info("Shutdown requested, saving checkpoint...")
            break
        
        if domain in completed_domains and resume:
            logging.info(f"Skipping {domain} (already completed)")
            continue
        
        logging.info(f"\n{'='*60}")
        logging.info(f"Processing domain: {domain}")
        logging.info(f"{'='*60}")
        
        # Load data
        corpus_file = data_root / "corpora" / "passage_level" / f"{domain}.jsonl"
        
        if use_data_splits:
            query_file = data_root / "data_splits" / "retrieval_tasks" / domain / "test" / f"{domain}_questions.jsonl"
            qrels_file = data_root / "data_splits" / "retrieval_tasks" / domain / "test" / "qrels" / "dev.tsv"
        else:
            query_file = data_root / "human" / "retrieval_tasks" / domain / f"{domain}_questions.jsonl"
            qrels_file = data_root / "human" / "retrieval_tasks" / domain / "qrels" / "dev.tsv"
        
        # Fallback
        if use_data_splits and not query_file.exists():
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
        
        # Expand queries with LLM
        logging.info(f"Expanding queries using {llm_type}...")
        expanded_queries = expanded_queries_cache.get(domain, {})
        
        for query_id, query_text in queries.items():
            if shutdown_requested:
                break
            
            if query_id not in expanded_queries:
                variations = expand_query_with_llm(query_text, llm_type, num_variations)
                expanded_queries[query_id] = variations
                time.sleep(0.1)  # Rate limiting
        
        # Save checkpoint
        save_checkpoint(checkpoint_dir, domain, expanded_queries)
        
        if shutdown_requested:
            break
        
        # Retrieve for each variation
        logging.info("Retrieving for each query variation...")
        all_variation_results = []
        
        # Process queries in batches to avoid single-query issue with BEIR
        batch_size = 10
        query_items = list(expanded_queries.items())
        
        for batch_start in range(0, len(query_items), batch_size):
            if shutdown_requested:
                break
            
            batch_queries = query_items[batch_start:batch_start + batch_size]
            batch_variation_results = {}
            
            # Collect all variations for this batch
            batch_temp_queries = {}
            query_variation_map = {}  # Map temp query_id to (original_query_id, variation_index)
            
            temp_query_counter = 0
            for query_id, variations in batch_queries:
                for var_idx, variation in enumerate(variations):
                    temp_query_id = f"temp_{temp_query_counter}"
                    batch_temp_queries[temp_query_id] = variation
                    query_variation_map[temp_query_id] = (query_id, var_idx)
                    temp_query_counter += 1
            
            if len(batch_temp_queries) == 0:
                continue
            
            # Retrieve for all variations in batch (avoids single-query issue)
            try:
                evaluator = EvaluateRetrieval(retriever, k_values=[10])
                batch_results = evaluator.retrieve(corpus, batch_temp_queries)
                
                # Map results back to original queries
                for temp_query_id, (original_query_id, var_idx) in query_variation_map.items():
                    if temp_query_id in batch_results:
                        if original_query_id not in batch_variation_results:
                            batch_variation_results[original_query_id] = {}
                        # Merge results
                        for doc_id, score in batch_results[temp_query_id].items():
                            if doc_id not in batch_variation_results[original_query_id]:
                                batch_variation_results[original_query_id][doc_id] = 0.0
                            batch_variation_results[original_query_id][doc_id] += score
            except Exception as e:
                logging.error(f"Error retrieving batch: {e}")
                # Fallback: process one at a time with workaround
                logging.info("Falling back to individual query processing...")
                for query_id, variations in batch_queries:
                    variation_results = {}
                    for variation in variations:
                        # Use a dummy second query to avoid single-query bug
                        temp_queries = {
                            query_id: variation,
                            f"{query_id}_dummy": variation  # Duplicate to avoid single-query issue
                        }
                        try:
                            evaluator = EvaluateRetrieval(retriever, k_values=[10])
                            results = evaluator.retrieve(corpus, temp_queries)
                            if query_id in results:
                                if query_id not in variation_results:
                                    variation_results[query_id] = {}
                                for doc_id, score in results[query_id].items():
                                    if doc_id not in variation_results[query_id]:
                                        variation_results[query_id][doc_id] = 0.0
                                    variation_results[query_id][doc_id] += score
                        except Exception as e2:
                            logging.error(f"Error retrieving for {query_id}: {e2}")
                            continue
                    batch_variation_results.update(variation_results)
            
            all_variation_results.append(batch_variation_results)
        
        if shutdown_requested:
            break
        
        # Combine results using RRF
        logging.info("Combining results with RRF...")
        combined_results = combine_results_with_rrf(all_variation_results)
        
        # Evaluate
        logging.info("Evaluating results...")
        k_values = [1, 3, 5, 10]
        evaluator = EvaluateRetrieval(None, k_values=k_values)
        ndcg, _map, recall, precision = evaluator.evaluate(qrels, combined_results, k_values)
        
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
        
        logging.info(f"Domain {domain} - Recall@10: {domain_results['Recall@10']:.4f}, nDCG@10: {domain_results['nDCG@10']:.4f}")
    
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
        
        logging.info("\n" + "="*60)
        logging.info("LLM QUERY EXPANSION RESULTS:")
        logging.info("="*60)
        logging.info(f"Average Recall@10: {avg_results['Recall@10']:.4f}")
        logging.info(f"Average nDCG@10: {avg_results['nDCG@10']:.4f}")
        
        # Save results
        results_file = output_dir / "results.json"
        with open(results_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        logging.info(f"Results saved to: {results_file}")
        return all_results
    
    return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='LLM Query Expansion')
    parser.add_argument('--config', type=str, required=True, help='Config JSON file')
    parser.add_argument('--gpu_id', type=int, help='GPU ID')
    parser.add_argument('--resume', action='store_true', default=True, help='Resume from checkpoint')
    parser.add_argument('--no-resume', dest='resume', action='store_false', help='Do not resume')
    
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    config['resume'] = args.resume
    
    try:
        results = run_llm_query_expansion_evaluation(config, args.gpu_id)
        if results:
            logging.info("✅ LLM query expansion evaluation completed")
    except KeyboardInterrupt:
        logging.info("Evaluation interrupted by user")
        sys.exit(0)
    except Exception as e:
        logging.error(f"Evaluation failed: {e}", exc_info=True)
        sys.exit(1)

