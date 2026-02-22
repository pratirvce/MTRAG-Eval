"""
Multi-Stage Retrieval Pipeline with Checkpointing and Resume Support
Stage 1: Fast dense retrieval (top 100)
Stage 2: Cross-encoder reranking (top 50)
Stage 3: Optional final reranking (top 20)
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
from sentence_transformers import CrossEncoder
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

def stage1_dense_retrieval(
    model_path: str,
    corpus: Dict,
    queries: Dict,
    top_k: int = 100,
    device: str = "cuda"
) -> Dict:
    """Stage 1: Fast dense retrieval"""
    logging.info(f"Stage 1: Dense retrieval (top-{top_k})...")
    model = SentenceBERT(model_path, device=device)
    retriever = DenseRetrievalExactSearch(model, batch_size=128)
    evaluator = EvaluateRetrieval(retriever, k_values=[top_k])
    results = evaluator.retrieve(corpus, queries)
    logging.info(f"Stage 1 complete: Retrieved top-{top_k} for {len(results)} queries")
    return results

def stage2_cross_encoder_rerank(
    cross_encoder_path: str,
    queries: Dict,
    corpus: Dict,
    stage1_results: Dict,
    top_k: int = 100,
    rerank_top_k: int = 50,
    batch_size: int = 32,
    device: str = "cuda"
) -> Dict:
    """Stage 2: Cross-encoder reranking"""
    logging.info(f"Stage 2: Cross-encoder reranking (top-{top_k} → top-{rerank_top_k})...")
    
    model = CrossEncoder(cross_encoder_path)
    model.to(device)
    
    reranked_results = {}
    
    for query_id, query_text in queries.items():
        if shutdown_requested:
            break
        
        if query_id not in stage1_results:
            continue
        
        # Get top-K from stage 1
        doc_scores = stage1_results[query_id]
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
    
    logging.info(f"Stage 2 complete: Reranked to top-{rerank_top_k} for {len(reranked_results)} queries")
    return reranked_results

def stage3_final_rerank(
    final_model_path: Optional[str],
    queries: Dict,
    corpus: Dict,
    stage2_results: Dict,
    rerank_top_k: int = 20,
    batch_size: int = 16,
    device: str = "cuda"
) -> Dict:
    """Stage 3: Optional final reranking with larger model"""
    if not final_model_path:
        logging.info("Stage 3: Skipped (no final model specified)")
        return stage2_results
    
    logging.info(f"Stage 3: Final reranking (top-{rerank_top_k})...")
    
    model = CrossEncoder(final_model_path)
    model.to(device)
    
    final_results = {}
    
    for query_id, query_text in queries.items():
        if shutdown_requested:
            break
        
        if query_id not in stage2_results:
            continue
        
        # Get top-K from stage 2
        doc_scores = stage2_results[query_id]
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)[:rerank_top_k * 2]  # Get more candidates
        
        if len(sorted_docs) == 0:
            final_results[query_id] = {}
            continue
        
        # Prepare pairs
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
            final_results[query_id] = {}
            continue
        
        # Score pairs
        scores = model.predict(pairs, batch_size=batch_size, show_progress_bar=False)
        
        # Create final results
        final_scores = {}
        for doc_id, score in zip(doc_ids, scores):
            final_scores[doc_id] = float(score)
        
        # Sort and take top rerank_top_k
        sorted_final = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)[:rerank_top_k]
        final_results[query_id] = {doc_id: score for doc_id, score in sorted_final}
    
    logging.info(f"Stage 3 complete: Final reranking to top-{rerank_top_k} for {len(final_results)} queries")
    return final_results

def run_multistage_evaluation(config: Dict, gpu_id: Optional[int] = None):
    """Run multi-stage retrieval evaluation"""
    global shutdown_requested
    
    experiment_name = config.get('experiment_name', 'multistage_retrieval')
    stage1_model = config.get('stage1_model')  # Dense retrieval model
    stage2_model = config.get('stage2_model')  # Cross-encoder for reranking
    stage3_model = config.get('stage3_model', None)  # Optional final reranking model
    domains = config.get('domains', MTRAG_DOMAINS)
    use_data_splits = config.get('use_data_splits', True)
    stage1_top_k = config.get('stage1_top_k', 100)
    stage2_top_k = config.get('stage2_top_k', 50)
    stage3_top_k = config.get('stage3_top_k', 20)
    resume = config.get('resume', True)
    
    if not stage1_model or not stage2_model:
        logging.error("stage1_model and stage2_model required in config")
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
    if resume:
        checkpoint_info = load_checkpoint(checkpoint_dir)
        if checkpoint_info:
            completed_domains = set(checkpoint_info.keys())
            logging.info(f"Resuming: Completed domains: {completed_domains}")
    
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
        
        # Stage 1: Dense retrieval
        stage1_results = stage1_dense_retrieval(
            stage1_model, corpus, queries, top_k=stage1_top_k, device=device
        )
        
        if shutdown_requested:
            break
        
        # Stage 2: Cross-encoder reranking
        stage2_results = stage2_cross_encoder_rerank(
            stage2_model, queries, corpus, stage1_results,
            top_k=stage1_top_k, rerank_top_k=stage2_top_k, device=device
        )
        
        if shutdown_requested:
            break
        
        # Stage 3: Final reranking (optional)
        if stage3_model:
            final_results = stage3_final_rerank(
                stage3_model, queries, corpus, stage2_results,
                rerank_top_k=stage3_top_k, device=device
            )
        else:
            final_results = stage2_results
        
        if shutdown_requested:
            break
        
        # Evaluate
        logging.info("Evaluating results...")
        k_values = [1, 3, 5, 10]
        evaluator = EvaluateRetrieval(None, k_values=k_values)
        ndcg, _map, recall, precision = evaluator.evaluate(qrels, final_results, k_values)
        
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
        
        # Save checkpoint
        save_checkpoint(checkpoint_dir, domain, domain_results)
    
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
        logging.info("MULTI-STAGE RETRIEVAL RESULTS:")
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
    parser = argparse.ArgumentParser(description='Multi-Stage Retrieval Pipeline')
    parser.add_argument('--config', type=str, required=True, help='Config JSON file')
    parser.add_argument('--gpu_id', type=int, help='GPU ID')
    parser.add_argument('--resume', action='store_true', default=True, help='Resume from checkpoint')
    parser.add_argument('--no-resume', dest='resume', action='store_false', help='Do not resume')
    
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    config['resume'] = args.resume
    
    try:
        results = run_multistage_evaluation(config, args.gpu_id)
        if results:
            logging.info("✅ Multi-stage evaluation completed")
    except KeyboardInterrupt:
        logging.info("Evaluation interrupted by user")
        sys.exit(0)
    except Exception as e:
        logging.error(f"Evaluation failed: {e}", exc_info=True)
        sys.exit(1)

