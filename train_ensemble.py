"""
Ensemble Retrieval Models
Combine multiple domain-specific models using weighted voting or reciprocal rank fusion
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

def reciprocal_rank_fusion(results_list: List[Dict], k: int = 60) -> Dict:
    """
    Reciprocal Rank Fusion (RRF) combines multiple ranked lists.
    RRF score = sum(1 / (k + rank)) for each document across all lists
    """
    fused_scores = {}
    
    for results in results_list:
        for query_id, doc_scores in results.items():
            if query_id not in fused_scores:
                fused_scores[query_id] = {}
            
            # Sort documents by score (descending)
            sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
            
            for rank, (doc_id, score) in enumerate(sorted_docs, start=1):
                if doc_id not in fused_scores[query_id]:
                    fused_scores[query_id][doc_id] = 0.0
                fused_scores[query_id][doc_id] += 1.0 / (k + rank)
    
    return fused_scores

def weighted_average(results_list: List[Dict], weights: List[float]) -> Dict:
    """Weighted average of scores from multiple models"""
    if len(results_list) != len(weights):
        raise ValueError("Number of results must match number of weights")
    
    # Normalize weights
    total_weight = sum(weights)
    weights = [w / total_weight for w in weights]
    
    fused_scores = {}
    
    for query_id in results_list[0].keys():
        fused_scores[query_id] = {}
        
        for model_results, weight in zip(results_list, weights):
            if query_id in model_results:
                for doc_id, score in model_results[query_id].items():
                    if doc_id not in fused_scores[query_id]:
                        fused_scores[query_id][doc_id] = 0.0
                    fused_scores[query_id][doc_id] += score * weight
    
    return fused_scores

def run_ensemble_evaluation(config: Dict):
    """Run ensemble evaluation combining multiple models"""
    experiment_name = config.get('experiment_name', 'ensemble_experiment')
    model_paths = config.get('model_paths', [])
    ensemble_method = config.get('ensemble_method', 'rrf')  # 'rrf' or 'weighted'
    weights = config.get('weights', None)
    domains = config.get('domains', MTRAG_DOMAINS)
    use_data_splits = config.get('use_data_splits', True)
    
    if not model_paths:
        logging.error("No model paths provided")
        return None
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Using device: {device}")
    
    # Load all models
    models = []
    retrievers = []
    
    for model_path in model_paths:
        logging.info(f"Loading model: {model_path}")
        try:
            model = SentenceBERT(model_path, device=device)
            retriever = DenseRetrievalExactSearch(model, batch_size=128)
            models.append(model)
            retrievers.append(retriever)
        except Exception as e:
            logging.error(f"Error loading model {model_path}: {e}")
            return None
    
    logging.info(f"Loaded {len(models)} models for ensemble")
    
    data_root = pathlib.Path(".")
    all_results = {}
    
    for domain in domains:
        logging.info(f"\n--- Evaluating ensemble on domain: {domain} ---")
        
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
        
        # Retrieve with each model
        all_model_results = []
        for i, retriever in enumerate(retrievers):
            logging.info(f"Retrieving with model {i+1}/{len(retrievers)}...")
            evaluator = EvaluateRetrieval(retriever, k_values=[10])
            results = evaluator.retrieve(corpus, queries)
            all_model_results.append(results)
        
        # Combine results
        logging.info(f"Combining results using {ensemble_method}...")
        if ensemble_method == 'rrf':
            fused_results = reciprocal_rank_fusion(all_model_results, k=60)
        elif ensemble_method == 'weighted':
            if weights is None:
                weights = [1.0 / len(models)] * len(models)  # Equal weights
            fused_results = weighted_average(all_model_results, weights)
        else:
            logging.error(f"Unknown ensemble method: {ensemble_method}")
            return None
        
        # Evaluate fused results
        logging.info("Evaluating ensemble results...")
        k_values = [1, 3, 5, 10]
        evaluator = EvaluateRetrieval(None, k_values=k_values)
        ndcg, _map, recall, precision = evaluator.evaluate(qrels, fused_results, k_values)
        
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
        logging.info("ENSEMBLE RESULTS:")
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
    parser = argparse.ArgumentParser(description='Ensemble Retrieval Models')
    parser.add_argument('--config', type=str, required=True, help='Config JSON file')
    parser.add_argument('--gpu_id', type=int, help='GPU ID (optional)')
    
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    if args.gpu_id is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    
    run_ensemble_evaluation(config)

