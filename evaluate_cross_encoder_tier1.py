#!/usr/bin/env python3
"""
Tier 1: Evaluate Fine-Tuned Cross-Encoder
Evaluates the trained cross-encoder model for reranking
"""

import sys
import os
import pathlib
import argparse
import json
import logging
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.models import SentenceBERT
from beir.retrieval.search.dense import DenseRetrievalExactSearch
from sentence_transformers import CrossEncoder
import torch
import numpy as np

logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO)

def evaluate_cross_encoder_reranking(model_path: str, gpu_id: int, output_dir: pathlib.Path):
    """Evaluate cross-encoder by reranking initial retrieval results"""
    
    # If CUDA_VISIBLE_DEVICES is set, use device 0 (it's already filtered)
    if 'CUDA_VISIBLE_DEVICES' in os.environ:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    
    # Load fine-tuned cross-encoder
    logging.info(f"Loading cross-encoder from: {model_path}")
    cross_encoder = CrossEncoder(model_path, device=device)
    
    # Use best ensemble results as initial retrieval (or BGE-base as fallback)
    initial_model = SentenceBERT('BAAI/bge-base-en-v1.5', device=device.type)
    retriever = DenseRetrievalExactSearch(initial_model, batch_size=128)
    
    domains = ["clapnq", "fiqa", "govt", "cloud"]
    all_results = {}
    
    for domain in domains:
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
        
        logging.info(f"Corpus: {len(corpus)} docs | Queries: {len(queries)} queries")
        
        # Stage 1: Initial retrieval (top 100)
        logging.info("Stage 1: Initial dense retrieval (top 100)...")
        evaluator = EvaluateRetrieval(retriever, k_values=[100])
        initial_results = evaluator.retrieve(corpus, queries)
        
        # Stage 2: Cross-encoder reranking (top 100 -> top 10)
        logging.info("Stage 2: Cross-encoder reranking...")
        reranked_results = {}
        
        for query_id, query_text in queries.items():
            if query_id not in initial_results:
                continue
            
            # Get top 100 candidates
            candidates = sorted(initial_results[query_id].items(), key=lambda x: x[1], reverse=True)[:100]
            
            # Create query-document pairs
            pairs = []
            doc_ids = []
            for doc_id, _ in candidates:
                doc = corpus.get(doc_id)
                if doc:
                    title = doc.get("title", "")
                    text = doc.get("text", "")
                    doc_text = f"{title} {text}".strip() if title else text
                    pairs.append([query_text, doc_text])
                    doc_ids.append(doc_id)
            
            if not pairs:
                continue
            
            # Rerank with cross-encoder
            scores = cross_encoder.predict(pairs, batch_size=32, show_progress_bar=False)
            
            # Sort by scores and take top 10
            scored_pairs = list(zip(doc_ids, scores))
            scored_pairs.sort(key=lambda x: x[1], reverse=True)
            
            reranked_results[query_id] = {doc_id: float(score) for doc_id, score in scored_pairs[:10]}
        
        # Evaluate reranked results
        logging.info("Evaluating reranked results...")
        k_values = [1, 3, 5, 10]
        evaluator = EvaluateRetrieval(None, k_values=k_values)
        ndcg, _map, recall, precision = evaluator.evaluate(qrels, reranked_results, k_values)
        
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
        logging.info("CROSS-ENCODER RERANKING RESULTS:")
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

def main():
    parser = argparse.ArgumentParser(description='Tier 1: Evaluate Cross-Encoder')
    parser.add_argument('--experiment_name', type=str, required=True)
    parser.add_argument('--gpu', type=int, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--resume', action='store_true', help='Resume from checkpoint')
    parser.add_argument('--no-resume', dest='resume', action='store_false', help='Do not resume')
    parser.set_defaults(resume=True)
    
    args = parser.parse_args()
    
    output_path = pathlib.Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find the trained model
    model_path = pathlib.Path("models/tier1_cross_encoder_finetuned")
    if not model_path.exists():
        # Try alternative location
        model_path = pathlib.Path("experiments/retrieval/tier1_cross_encoder_finetuned")
        config_file = model_path / "config.json"
        if config_file.exists():
            with open(config_file) as f:
                config = json.load(f)
                if 'model_path' in config:
                    model_path = pathlib.Path(config['model_path'])
    
    if not model_path.exists():
        logging.error(f"Model not found at {model_path}. Please check the training completed successfully.")
        return
    
    try:
        results = evaluate_cross_encoder_reranking(str(model_path), args.gpu, output_path)
        logging.info("✅ Cross-encoder evaluation completed successfully")
    except Exception as e:
        logging.error(f"Evaluation failed: {e}", exc_info=True)
        error_info = {
            'experiment': args.experiment_name,
            'status': 'failed',
            'error': str(e)
        }
        results_file = output_path / "results.json"
        with open(results_file, 'w') as f:
            json.dump(error_info, f, indent=2)
        raise

if __name__ == "__main__":
    main()

