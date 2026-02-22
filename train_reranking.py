"""
Advanced Reranking with Cross-Encoders
Use cross-encoder models to rerank top-K results from dense retrieval
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
from sentence_transformers import CrossEncoder
import torch

logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO)

MTRAG_DOMAINS = ["clapnq", "fiqa", "govt", "cloud"]

def rerank_with_cross_encoder(
    cross_encoder_model: str,
    queries: Dict,
    corpus: Dict,
    initial_results: Dict,
    top_k: int = 100,
    rerank_top_k: int = 20,
    batch_size: int = 32
) -> Dict:
    """
    Rerank initial results using a cross-encoder model.
    
    Args:
        cross_encoder_model: Path to cross-encoder model
        queries: Query dictionary
        corpus: Corpus dictionary
        initial_results: Initial retrieval results
        top_k: Number of top results to rerank
        rerank_top_k: Final number of results after reranking
        batch_size: Batch size for cross-encoder
    """
    logging.info(f"Loading cross-encoder: {cross_encoder_model}")
    model = CrossEncoder(cross_encoder_model)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    
    reranked_results = {}
    
    for query_id, query_text in queries.items():
        if query_id not in initial_results:
            continue
        
        # Get top-K initial results
        doc_scores = initial_results[query_id]
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        
        if len(sorted_docs) == 0:
            reranked_results[query_id] = {}
            continue
        
        # Prepare pairs for cross-encoder
        pairs = []
        doc_ids = []
        
        for doc_id, _ in sorted_docs:
            doc = corpus.get(doc_id)
            if doc:
                title = doc.get("title", "")
                text = doc.get("text", "")
                doc_text = f"{title} {text}".strip() if title else text
                pairs.append([query_text, doc_text])
                doc_ids.append(doc_id)
        
        if len(pairs) == 0:
            reranked_results[query_id] = {}
            continue
        
        # Score pairs with cross-encoder
        scores = model.predict(pairs, batch_size=batch_size, show_progress_bar=False)
        
        # Create reranked results
        reranked_scores = {}
        for doc_id, score in zip(doc_ids, scores):
            reranked_scores[doc_id] = float(score)
        
        # Sort and take top rerank_top_k
        sorted_reranked = sorted(reranked_scores.items(), key=lambda x: x[1], reverse=True)[:rerank_top_k]
        reranked_results[query_id] = {doc_id: score for doc_id, score in sorted_reranked}
    
    return reranked_results

def run_reranking_evaluation(config: Dict):
    """Run reranking evaluation"""
    experiment_name = config.get('experiment_name', 'reranking_experiment')
    base_model_path = config.get('base_model_path')
    cross_encoder_model = config.get('cross_encoder_model', 'cross-encoder/ms-marco-MiniLM-L-12-v2')
    top_k = config.get('top_k', 100)
    rerank_top_k = config.get('rerank_top_k', 20)
    domains = config.get('domains', MTRAG_DOMAINS)
    use_data_splits = config.get('use_data_splits', True)
    
    if not base_model_path:
        logging.error("base_model_path required in config")
        return None
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Using device: {device}")
    
    # Load base retriever
    logging.info(f"Loading base model: {base_model_path}")
    base_model = SentenceBERT(base_model_path, device=device)
    retriever = DenseRetrievalExactSearch(base_model, batch_size=128)
    
    data_root = pathlib.Path(".")
    all_results = {}
    
    for domain in domains:
        logging.info(f"\n--- Evaluating reranking on domain: {domain} ---")
        
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
        
        # Initial retrieval
        logging.info("Running initial retrieval...")
        evaluator = EvaluateRetrieval(retriever, k_values=[top_k])
        initial_results = evaluator.retrieve(corpus, queries)
        
        # Rerank
        logging.info(f"Reranking top-{top_k} results with cross-encoder...")
        reranked_results = rerank_with_cross_encoder(
            cross_encoder_model, queries, corpus, initial_results,
            top_k=top_k, rerank_top_k=rerank_top_k
        )
        
        # Evaluate
        logging.info("Evaluating reranked results...")
        k_values = [1, 3, 5, 10]
        evaluator = EvaluateRetrieval(None, k_values=k_values)
        ndcg, _map, recall, precision = evaluator.evaluate(qrels, reranked_results, k_values)
        
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
        logging.info("RERANKING RESULTS:")
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
    parser = argparse.ArgumentParser(description='Reranking with Cross-Encoders')
    parser.add_argument('--config', type=str, required=True, help='Config JSON file')
    parser.add_argument('--gpu_id', type=int, help='GPU ID')
    
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    if args.gpu_id is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    
    run_reranking_evaluation(config)

