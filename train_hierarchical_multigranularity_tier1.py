#!/usr/bin/env python3
"""
Tier 1: Hierarchical Multi-Granularity Retrieval
Priority 3 - Good impact experiment
Retrieves at sentence, paragraph, and document levels, then combines results
"""

import sys
import pathlib
import argparse
import json
import logging
import numpy as np
from typing import Dict, List
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.models import SentenceBERT
from beir.retrieval.search.dense import DenseRetrievalExactSearch
import torch
import signal
import re

logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO)

MTRAG_DOMAINS = ["clapnq", "fiqa", "govt", "cloud"]
shutdown_requested = False

def signal_handler(sig, frame):
    global shutdown_requested
    logging.info("\n⚠️  Shutdown signal received. Saving checkpoint...")
    shutdown_requested = True

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

def split_into_sentences(text: str) -> List[str]:
    """Split text into sentences"""
    # Simple sentence splitting (can be improved with nltk)
    sentences = re.split(r'[.!?]+\s+', text)
    return [s.strip() for s in sentences if s.strip()]

def split_into_paragraphs(text: str) -> List[str]:
    """Split text into paragraphs"""
    paragraphs = text.split('\n\n')
    return [p.strip() for p in paragraphs if p.strip()]

def create_granularity_corpus(original_corpus: Dict, granularity: str) -> Dict:
    """Create corpus at different granularities"""
    granular_corpus = {}
    
    for doc_id, doc in original_corpus.items():
        text = doc.get("text", "")
        title = doc.get("title", "")
        
        if granularity == "sentence":
            chunks = split_into_sentences(text)
        elif granularity == "paragraph":
            chunks = split_into_paragraphs(text)
        else:  # document
            chunks = [text]
        
        for idx, chunk in enumerate(chunks):
            chunk_id = f"{doc_id}_granularity_{granularity}_{idx}"
            granular_corpus[chunk_id] = {
                "title": title if idx == 0 else "",
                "text": chunk,
                "original_doc_id": doc_id,
                "granularity": granularity,
                "chunk_index": idx
            }
    
    return granular_corpus

def reciprocal_rank_fusion(results_list: List[Dict], k: int = 60) -> Dict:
    """Combine multiple result sets using Reciprocal Rank Fusion"""
    rrf_scores = {}
    
    for results in results_list:
        for qid, doc_scores in results.items():
            if qid not in rrf_scores:
                rrf_scores[qid] = {}
            
            sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
            for rank, (doc_id, score) in enumerate(sorted_docs, start=1):
                if doc_id not in rrf_scores[qid]:
                    rrf_scores[qid][doc_id] = 0.0
                rrf_scores[qid][doc_id] += 1.0 / (k + rank)
    
    return rrf_scores

def map_to_original_docs(results: Dict, granular_corpus: Dict) -> Dict:
    """Map granular results back to original document IDs"""
    original_results = {}
    
    for qid, doc_scores in results.items():
        original_results[qid] = {}
        
        for chunk_id, score in doc_scores.items():
            if chunk_id in granular_corpus:
                original_doc_id = granular_corpus[chunk_id].get("original_doc_id", chunk_id)
                
                # Take maximum score for each original document
                if original_doc_id not in original_results[qid]:
                    original_results[qid][original_doc_id] = score
                else:
                    original_results[qid][original_doc_id] = max(
                        original_results[qid][original_doc_id], score
                    )
    
    return original_results

def run_hierarchical_retrieval(config: Dict, gpu_id: int, output_dir: pathlib.Path):
    """Run hierarchical multi-granularity retrieval"""
    model_path = config.get('base_model', 'BAAI/bge-base-en-v1.5')
    domains = config.get('domains', MTRAG_DOMAINS)
    use_data_splits = config.get('use_data_splits', True)
    top_k = config.get('top_k', 100)
    resume = config.get('resume', True)
    
    # Set GPU
    if gpu_id is not None:
        import os
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Using device: {device}")
    
    data_root = pathlib.Path(".")
    all_results = {}
    
    # Load checkpoint if resuming
    checkpoint_file = output_dir / "checkpoint.json"
    completed_domains = set()
    if resume and checkpoint_file.exists():
        try:
            with open(checkpoint_file, 'r') as f:
                checkpoint = json.load(f)
                all_results = checkpoint.get("results", {})
                completed_domains = set(all_results.keys())
                logging.info(f"Resuming: Completed domains: {completed_domains}")
        except:
            pass
    
    # Load model
    logging.info(f"Loading model: {model_path}")
    model = SentenceBERT(model_path, device=device)
    retriever = DenseRetrievalExactSearch(model, batch_size=128)
    
    for domain in domains:
        if shutdown_requested:
            break
        
        if domain in completed_domains:
            logging.info(f"Skipping {domain} (already completed)")
            continue
        
        logging.info(f"\n{'='*60}")
        logging.info(f"Evaluating Hierarchical Multi-Granularity on domain: {domain}")
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
        
        # Step 1: Sentence-level retrieval
        logging.info("Step 1: Creating sentence-level corpus...")
        sentence_corpus = create_granularity_corpus(corpus, "sentence")
        logging.info(f"Sentence-level corpus: {len(sentence_corpus)} chunks")
        
        logging.info("Step 1: Retrieving at sentence level...")
        sentence_retriever = DenseRetrievalExactSearch(model, batch_size=128)
        sentence_evaluator = EvaluateRetrieval(sentence_retriever, k_values=[top_k])
        sentence_results = sentence_evaluator.retrieve(sentence_corpus, queries)
        sentence_results_original = map_to_original_docs(sentence_results, sentence_corpus)
        
        if shutdown_requested:
            break
        
        # Step 2: Paragraph-level retrieval
        logging.info("Step 2: Creating paragraph-level corpus...")
        paragraph_corpus = create_granularity_corpus(corpus, "paragraph")
        logging.info(f"Paragraph-level corpus: {len(paragraph_corpus)} chunks")
        
        logging.info("Step 2: Retrieving at paragraph level...")
        paragraph_retriever = DenseRetrievalExactSearch(model, batch_size=128)
        paragraph_evaluator = EvaluateRetrieval(paragraph_retriever, k_values=[top_k])
        paragraph_results = paragraph_evaluator.retrieve(paragraph_corpus, queries)
        paragraph_results_original = map_to_original_docs(paragraph_results, paragraph_corpus)
        
        if shutdown_requested:
            break
        
        # Step 3: Document-level retrieval
        logging.info("Step 3: Retrieving at document level...")
        document_evaluator = EvaluateRetrieval(retriever, k_values=[top_k])
        document_results = document_evaluator.retrieve(corpus, queries)
        
        if shutdown_requested:
            break
        
        # Step 4: Combine using RRF
        logging.info("Step 4: Combining results using Reciprocal Rank Fusion...")
        combined_results = reciprocal_rank_fusion([
            sentence_results_original,
            paragraph_results_original,
            document_results
        ])
        
        # Limit to top_k
        final_results = {}
        for qid, doc_scores in combined_results.items():
            sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
            final_results[qid] = {doc_id: score for doc_id, score in sorted_docs}
        
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
        
        # Save checkpoint
        checkpoint_data = {"results": all_results, "timestamp": str(pathlib.Path(__file__).stat().st_mtime)}
        with open(checkpoint_file, 'w') as f:
            json.dump(checkpoint_data, f, indent=2)
        
        if shutdown_requested:
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
    
    logging.info(f"✅ Results saved to: {results_file}")
    return str(results_file)

def main():
    parser = argparse.ArgumentParser(description='Tier 1: Hierarchical Multi-Granularity')
    parser.add_argument('--experiment_name', type=str, required=True)
    parser.add_argument('--gpu', type=int, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--resume', action='store_true', help='Resume from checkpoint')
    parser.add_argument('--no-resume', dest='resume', action='store_false', help='Do not resume')
    parser.set_defaults(resume=True)
    
    args = parser.parse_args()
    
    output_path = pathlib.Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    config = {
        'base_model': 'BAAI/bge-base-en-v1.5',
        'domains': ['clapnq', 'fiqa', 'govt', 'cloud'],
        'use_data_splits': True,
        'top_k': 100,
        'resume': args.resume
    }
    
    config_file = output_path / "config.json"
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    
    try:
        results_file = run_hierarchical_retrieval(config, args.gpu, output_path)
        logging.info(f"✅ Hierarchical multi-granularity experiment completed")
    except Exception as e:
        logging.error(f"Hierarchical multi-granularity experiment failed: {e}", exc_info=True)
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
