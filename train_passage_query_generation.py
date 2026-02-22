"""
Passage-Query Generation for Retrieval Matching (Structured Query Variants)

Idea:
- For each user query, generate multiple "passage-style" query variants that better match
  documentation-like passages (title-like, definition-like, procedure-like, bullet-like).
- Run retrieval for each variant and fuse results (RRF) to improve matching, especially
  for Cloud/Govt corpora with documentation-style phrasing.

Task A Safe:
- Only rewrites the query; no answer generation.
- Retrieval-only; no disallowed metadata.

Expected: 0.78-0.86 nDCG@10
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

import torch
import signal
import sys
from datetime import datetime
from collections import defaultdict

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


class PassageQueryGenerator:
    """Generate structured passage-style query variants from a base query.

    Variants (all deterministic templates, no LLM required):
    - title_like: short, title-style reformulation
    - definition_like: "Definition of ..." style
    - procedure_like: steps/instructions style
    - problem_like: troubleshooting / issue description style
    """

    def __init__(self, domain: str):
        self.domain = domain

    def generate_variants(self, query: str) -> Dict[str, str]:
        q = query.strip().rstrip('?')
        variants: Dict[str, str] = {}

        # Title-like
        if len(q) > 80:
            short = q[:80] + "..."
        else:
            short = q
        variants["title_like"] = f"{short}"

        # Definition-like
        variants["definition_like"] = f"Definition and overview of {q}"

        # Procedure-like
        variants["procedure_like"] = f"Step-by-step procedure: {q}"

        # Problem/issue-like
        variants["problem_like"] = f"Problem description and resolution: {q}"

        # Original as baseline
        variants["original"] = query

        return variants


def reciprocal_rank_fusion(results_list: List[Dict[str, Dict[str, float]]], k: int = 60) -> Dict[str, Dict[str, float]]:
    """
    Reciprocal Rank Fusion (RRF) to combine multiple retrieval results.
    """
    fused_results = defaultdict(dict)

    for results in results_list:
        for query_id, doc_scores in results.items():
            for rank, (doc_id, score) in enumerate(sorted(doc_scores.items(), key=lambda x: x[1], reverse=True), start=1):
                if doc_id not in fused_results[query_id]:
                    fused_results[query_id][doc_id] = 0.0
                fused_results[query_id][doc_id] += 1.0 / (k + rank)

    return dict(fused_results)


def run_passage_query_generation(config: Dict, gpu_id: Optional[int] = None) -> Optional[str]:
    """Run passage-style query variant retrieval and fusion experiment."""
    global shutdown_requested

    if gpu_id is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        logging.info(f"Using GPU {gpu_id}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Device: {device}")

    output_dir = pathlib.Path(config.get('output_dir', 'experiments/retrieval/passage_query_generation'))
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = config.get('model_path', 'BAAI/bge-base-en-v1.5')
    domains = config.get('domains', MTRAG_DOMAINS)
    top_k = config.get('top_k', 100)

    # Dense retriever
    dense_model = SentenceBERT(model_path, device=device)
    dense_retriever = DenseRetrievalExactSearch(dense_model, batch_size=128)

    all_results: Dict[str, Dict[str, float]] = {}

    for domain in domains:
        if shutdown_requested:
            break

        logging.info(f"\n{'='*60}\nProcessing domain: {domain}\n{'='*60}")

        data_root = pathlib.Path('.')
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

        generator = PassageQueryGenerator(domain)
        evaluator = EvaluateRetrieval(dense_retriever, k_values=[1, 3, 5, 10])

        # For each query, generate variants, retrieve, and fuse
        fused_results: Dict[str, Dict[str, float]] = {}

        for qid, query in queries.items():
            if shutdown_requested:
                break

            variants = generator.generate_variants(query)
            variant_results: List[Dict[str, Dict[str, float]]] = []

            for name, vq in variants.items():
                # Use BEIR's retrieve interface: {qid: query_text}
                res = evaluator.retrieve(corpus, {qid: vq})
                if qid in res:
                    variant_results.append({qid: res[qid]})

            if variant_results:
                fused = reciprocal_rank_fusion(variant_results, k=60)
                fused_results[qid] = fused[qid]
            else:
                # Fallback to original query retrieval
                res = evaluator.retrieve(corpus, {qid: query})
                if qid in res:
                    fused_results[qid] = res[qid]

        # Evaluate fused results
        ndcg, _map, recall, precision = evaluator.evaluate(qrels, fused_results, [1, 3, 5, 10])
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
        logging.info(f"{domain} - nDCG@10: {all_results[domain]['nDCG@10']:.4f}")

    # Aggregate
    if all_results:
        avg = {
            "Recall@1": np.mean([r["Recall@1"] for r in all_results.values()]),
            "Recall@3": np.mean([r["Recall@3"] for r in all_results.values()]),
            "Recall@5": np.mean([r["Recall@5"] for r in all_results.values()]),
            "Recall@10": np.mean([r["Recall@10"] for r in all_results.values()]),
            "nDCG@1": np.mean([r["nDCG@1"] for r in all_results.values()]),
            "nDCG@3": np.mean([r["nDCG@3"] for r in all_results.values()]),
            "nDCG@5": np.mean([r["nDCG@5"] for r in all_results.values()]),
            "nDCG@10": np.mean([r["nDCG@10"] for r in all_results.values()])
        }
        all_results["average"] = avg
        results_file = output_dir / "results.json"
        with open(results_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        logging.info(f"Average nDCG@10: {avg['nDCG@10']:.4f}")
        return str(results_file)

    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--gpu_id', type=int, default=0)
    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)

    run_passage_query_generation(config, gpu_id=args.gpu_id)
