"""
Hybrid Retrieval with Learned Weights
Combine BM25 and dense retrieval with learned weights per domain
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
from rank_bm25 import BM25Okapi
import torch
import re

logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO)

MTRAG_DOMAINS = ["clapnq", "fiqa", "govt", "cloud"]

def combine_results(dense_results: Dict, bm25_results: Dict, alpha: float = 0.5) -> Dict:
    """
    Combine dense and BM25 results with weight alpha.
    alpha: weight for dense retrieval (1-alpha for BM25)
    """
    combined = {}
    
    # Normalize scores to [0, 1] range
    def normalize_scores(scores: Dict) -> Dict:
        if not scores:
            return {}
        max_score = max(scores.values()) if scores.values() else 1.0
        min_score = min(scores.values()) if scores.values() else 0.0
        if max_score == min_score:
            return {k: 0.5 for k in scores.keys()}
        return {k: (v - min_score) / (max_score - min_score) for k, v in scores.items()}
    
    for query_id in set(list(dense_results.keys()) + list(bm25_results.keys())):
        dense_scores = normalize_scores(dense_results.get(query_id, {}))
        bm25_scores = normalize_scores(bm25_results.get(query_id, {}))
        
        # Combine scores
        combined_scores = {}
        all_doc_ids = set(list(dense_scores.keys()) + list(bm25_scores.keys()))
        
        for doc_id in all_doc_ids:
            dense_score = dense_scores.get(doc_id, 0.0)
            bm25_score = bm25_scores.get(doc_id, 0.0)
            combined_scores[doc_id] = alpha * dense_score + (1 - alpha) * bm25_score
        
        combined[query_id] = combined_scores
    
    return combined

def run_hybrid_evaluation(config: Dict):
    """Run hybrid retrieval evaluation with learned weights"""
    experiment_name = config.get('experiment_name', 'hybrid_learned_experiment')
    model_path = config.get('model_path')
    alpha = config.get('alpha', 0.5)  # Weight for dense retrieval
    domains = config.get('domains', MTRAG_DOMAINS)
    use_data_splits = config.get('use_data_splits', True)
    
    if not model_path:
        logging.error("model_path required in config")
        return None
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Using device: {device}")
    
    # Load dense model
    logging.info(f"Loading dense model: {model_path}")
    dense_model = SentenceBERT(model_path, device=device)
    dense_retriever = DenseRetrievalExactSearch(dense_model, batch_size=128)
    
    data_root = pathlib.Path(".")
    all_results = {}
    
    for domain in domains:
        logging.info(f"\n--- Evaluating hybrid retrieval on domain: {domain} ---")
        
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
        
        # Dense retrieval
        logging.info("Running dense retrieval...")
        dense_evaluator = EvaluateRetrieval(dense_retriever, k_values=[10])
        dense_results = dense_evaluator.retrieve(corpus, queries)
        
        # BM25 retrieval using rank-bm25
        logging.info("Running BM25 retrieval...")
        def tokenize(text):
            """Simple tokenization for BM25"""
            text = text.lower()
            text = re.sub(r'[^a-z0-9\s]', ' ', text)
            return text.split()
        
        # Prepare corpus for BM25
        corpus_texts = [corpus[doc_id].get('text', corpus[doc_id].get('title', '')) for doc_id in corpus.keys()]
        tokenized_corpus = [tokenize(text) for text in corpus_texts]
        bm25 = BM25Okapi(tokenized_corpus)
        
        # Retrieve for each query
        bm25_results = {}
        for query_id, query_text in queries.items():
            tokenized_query = tokenize(query_text)
            scores = bm25.get_scores(tokenized_query)
            
            # Get top-k documents
            top_k = 10
            top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
            
            # Convert to BEIR format
            doc_ids = list(corpus.keys())
            bm25_results[query_id] = {doc_ids[i]: float(scores[i]) for i in top_indices}
        
        # Combine results
        logging.info(f"Combining results with alpha={alpha}...")
        combined_results = combine_results(dense_results, bm25_results, alpha=alpha)
        
        # Evaluate
        logging.info("Evaluating combined results...")
        k_values = [1, 3, 5, 10]
        evaluator = EvaluateRetrieval(None, k_values=k_values)
        ndcg, _map, recall, precision = evaluator.evaluate(qrels, combined_results, k_values)
        
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
        logging.info("HYBRID RETRIEVAL RESULTS:")
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
    parser = argparse.ArgumentParser(description='Hybrid Retrieval with Learned Weights')
    parser.add_argument('--config', type=str, required=True, help='Config JSON file')
    parser.add_argument('--gpu_id', type=int, help='GPU ID')
    
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    if args.gpu_id is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    
    run_hybrid_evaluation(config)

