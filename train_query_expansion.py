"""
Query Expansion/Rewriting
Expand queries using LLM or synonym-based methods before retrieval
"""

import pathlib
import logging
import argparse
import json
import os
import numpy as np
from typing import Dict, List
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.models import SentenceBERT
from beir.retrieval.search.dense import DenseRetrievalExactSearch
import torch

logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO)

MTRAG_DOMAINS = ["clapnq", "fiqa", "govt", "cloud"]

def expand_query_simple(query: str, method: str = "synonym") -> str:
    """
    Query expansion using various methods
    
    ⚠️ TASK A COMPLIANCE: Query expansion is preprocessing only,
    NOT text generation. This is compliant with Task A requirements.
    """
    if method == "synonym":
        # Simple synonym expansion
        # Add common synonyms and related terms
        expanded = query
        
        # Common synonym patterns
        synonyms = {
            "what": ["which", "what kind of", "what type of"],
            "how": ["what method", "what way", "what process"],
            "why": ["what reason", "what cause", "what purpose"],
            "when": ["at what time", "what date", "what period"],
            "where": ["in what location", "at what place", "in which"],
        }
        
        query_lower = query.lower()
        for word, syns in synonyms.items():
            if word in query_lower:
                # Add first synonym
                expanded = f"{expanded} {syns[0]}"
                break
        
        return expanded.strip()
    elif method == "paraphrase":
        # Paraphrase expansion - add paraphrased versions
        expanded = query
        
        # Add question variations
        if "?" in query:
            # Remove question mark and add variations
            base = query.replace("?", "").strip()
            expanded = f"{query} {base} explain {base} information about {base}"
        
        return expanded.strip()
    elif method == "keyword":
        # Keyword expansion - extract and expand key terms
        # Simple keyword extraction
        words = query.split()
        # Add important words with variations
        expanded = query
        for word in words[:3]:  # First 3 words
            if len(word) > 3:  # Skip short words
                expanded = f"{expanded} {word} related {word} information"
        
        return expanded.strip()
    else:
        return query

def expand_queries(queries: Dict, method: str = "synonym") -> Dict:
    """Expand all queries"""
    expanded = {}
    for query_id, query_text in queries.items():
        expanded[query_id] = expand_query_simple(query_text, method)
    return expanded

def run_query_expansion_evaluation(config: Dict):
    """Run query expansion evaluation"""
    experiment_name = config.get('experiment_name', 'query_expansion_experiment')
    model_path = config.get('model_path')
    expansion_method = config.get('expansion_method', 'synonym')
    domains = config.get('domains', MTRAG_DOMAINS)
    use_data_splits = config.get('use_data_splits', True)
    
    if not model_path:
        logging.error("model_path required in config")
        return None
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Using device: {device}")
    
    # Load model
    logging.info(f"Loading model: {model_path}")
    model = SentenceBERT(model_path, device=device)
    retriever = DenseRetrievalExactSearch(model, batch_size=128)
    
    data_root = pathlib.Path(".")
    all_results = {}
    
    for domain in domains:
        logging.info(f"\n--- Evaluating query expansion on domain: {domain} ---")
        
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
        
        # Expand queries
        logging.info(f"Expanding queries using method: {expansion_method}")
        expanded_queries = expand_queries(queries, method=expansion_method)
        
        # Retrieve with expanded queries
        logging.info("Running retrieval with expanded queries...")
        evaluator = EvaluateRetrieval(retriever, k_values=[10])
        results = evaluator.retrieve(corpus, expanded_queries)
        
        # Evaluate
        logging.info("Evaluating results...")
        k_values = [1, 3, 5, 10]
        evaluator = EvaluateRetrieval(None, k_values=k_values)
        ndcg, _map, recall, precision = evaluator.evaluate(qrels, results, k_values)
        
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
        
        logging.info(f"Recall@10: {recall.get('Recall@10', 0):.4f}")
        logging.info(f"nDCG@10: {ndcg.get('NDCG@10', 0):.4f}")
    
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
        logging.info("QUERY EXPANSION RESULTS:")
        logging.info("="*60)
        logging.info(f"Average Recall@10: {avg_results['Recall@10']:.4f}")
        logging.info(f"Average nDCG@10: {avg_results['nDCG@10']:.4f}")
        
        # Save results
        output_dir = pathlib.Path("experiments/retrieval") / experiment_name
        output_dir.mkdir(parents=True, exist_ok=True)
        results_file = output_dir / "results.json"
        
        with open(results_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        logging.info(f"Results saved to: {results_file}")
        return all_results
    
    return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Query Expansion Evaluation')
    parser.add_argument('--config', type=str, required=True, help='Config JSON file')
    parser.add_argument('--gpu_id', type=int, help='GPU ID')
    
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    if args.gpu_id is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    
    run_query_expansion_evaluation(config)

