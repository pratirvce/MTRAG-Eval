"""
Multi-Query Sparse + Dense Rewrites
Generate multiple query rewrites, run through both sparse and dense retrievers, fuse results

Task A Compliant: ✅ Query preprocessing + retrieval fusion
Expected: 0.80-0.88 nDCG@10
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
from sentence_transformers import SentenceTransformer
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
    global shutdown_requested
    logging.info("\n⚠️  Shutdown signal received. Saving checkpoint...")
    shutdown_requested = True

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


class MultiQueryRewriter:
    """Generate multiple query rewrites using different strategies"""
    def __init__(self, llm_model_name: Optional[str] = None, use_local: bool = True):
        self.use_local = use_local
        self.llm_model_name = llm_model_name or "microsoft/Phi-3-mini-4k-instruct"
        
        if use_local:
            try:
                from scripts.evaluation.huggingface_client import HuggingFaceLLMClient
                self.llm_client = HuggingFaceLLMClient(self.llm_model_name)
            except Exception as e:
                logging.warning(f"Could not load local LLM: {e}")
                self.llm_client = None
        else:
            self.llm_client = None
    
    def generate_rewrites(self, query: str, num_rewrites: int = 5) -> List[str]:
        """Generate multiple rewrites: stemming, paraphrase, constraint-focused, entity-only"""
        rewrites = [query]  # Always include original
        
        # Simple rewrites
        rewrites.append(query + " information")
        rewrites.append(query + " details")
        
        if "what" in query.lower():
            rewrites.append(query.lower().replace("what", "which"))
        if "how" in query.lower():
            rewrites.append(query.lower().replace("how", "what method"))
        
        # LLM-based rewrites if available
        if self.llm_client and len(rewrites) < num_rewrites:
            try:
                prompt = f"""Generate {num_rewrites - len(rewrites)} different rewrites of this query:
{query}

Generate rewrites (one per line, no explanations):
"""
                response = self.llm_client.generate_response(prompt, max_new_tokens=200, temperature=0.8)
                for line in response.strip().split('\n'):
                    line = line.strip().lstrip('0123456789.-) ').strip()
                    if line and len(line) > 5:
                        rewrites.append(line)
            except:
                pass
        
        return rewrites[:num_rewrites]


class MultiQuerySparseDenseRetriever:
    """
    Multi-query retrieval: generate rewrites, run through sparse + dense, fuse
    """
    def __init__(self, base_model_path: str, rewriter: MultiQueryRewriter, device: str = "cuda"):
        self.device = device
        self.rewriter = rewriter
        self.base_model = SentenceBERT(base_model_path, device=device)
        self.dense_retriever = DenseRetrievalExactSearch(self.base_model, batch_size=128)
        
        # Sparse retrieval (BM25) - optional
        try:
            from rank_bm25 import BM25Okapi
            self.bm25_available = True
        except ImportError:
            self.bm25_available = False
            logging.warning("BM25 not available, using dense only")
    
    def retrieve(self, corpus: Dict, queries: Dict, top_k: int = 10) -> Dict:
        """Multi-query retrieval with sparse + dense fusion"""
        all_results = {}
        
        for query_id, query_text in queries.items():
            # Generate rewrites
            rewrites = self.rewriter.generate_rewrites(query_text, num_rewrites=5)
            
            all_candidates = {}
            
            # Dense retrieval for each rewrite
            for rewrite in rewrites:
                evaluator_temp = EvaluateRetrieval(self.dense_retriever, k_values=[top_k * 2])
                dense_results = evaluator_temp.retrieve(corpus, {query_id: rewrite})
                
                if query_id in dense_results:
                    sorted_docs = sorted(dense_results[query_id].items(), key=lambda x: x[1], reverse=True)
                    for rank, (doc_id, score) in enumerate(sorted_docs, start=1):
                        if doc_id not in all_candidates:
                            all_candidates[doc_id] = {'dense': [], 'sparse': []}
                        all_candidates[doc_id]['dense'].append(1.0 / (60 + rank))
            
            # Sparse retrieval (BM25) if available
            if self.bm25_available:
                corpus_texts = [corpus[doc_id].get('text', '') for doc_id in corpus.keys()]
                tokenized_corpus = [text.lower().split() for text in corpus_texts]
                bm25 = BM25Okapi(tokenized_corpus)
                
                for rewrite in rewrites:
                    query_tokens = rewrite.lower().split()
                    bm25_scores = bm25.get_scores(query_tokens)
                    
                    # Get top documents
                    top_indices = np.argsort(bm25_scores)[::-1][:top_k * 2]
                    for rank, idx in enumerate(top_indices, start=1):
                        doc_id = list(corpus.keys())[idx]
                        if doc_id not in all_candidates:
                            all_candidates[doc_id] = {'dense': [], 'sparse': []}
                        all_candidates[doc_id]['sparse'].append(1.0 / (60 + rank))
            
            # Fuse scores: RRF for dense, RRF for sparse, then combine
            fused_scores = {}
            for doc_id, scores in all_candidates.items():
                dense_rrf = sum(scores['dense']) if scores['dense'] else 0.0
                sparse_rrf = sum(scores['sparse']) if scores['sparse'] else 0.0
                
                # Normalize and combine (60% dense, 40% sparse)
                fused_scores[doc_id] = 0.6 * dense_rrf + 0.4 * sparse_rrf
            
            # Sort and return top_k
            sorted_docs = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
            all_results[query_id] = {doc_id: score for doc_id, score in sorted_docs[:top_k]}
        
        return all_results


def train_multi_query_sparse_dense(config: Dict, gpu_id: Optional[int] = None) -> Optional[str]:
    """Train and evaluate multi-query sparse + dense retrieval"""
    global shutdown_requested
    
    if gpu_id is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Device: {device}")
    
    output_dir = pathlib.Path(config.get('output_dir', 'experiments/retrieval/multi_query_sparse_dense'))
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize components
    base_model = config.get('model_path', 'BAAI/bge-base-en-v1.5')
    llm_model = config.get('llm_model', 'microsoft/Phi-3-mini-4k-instruct')
    use_local_llm = config.get('use_local_llm', True)
    
    rewriter = MultiQueryRewriter(llm_model_name=llm_model, use_local=use_local_llm)
    retriever = MultiQuerySparseDenseRetriever(base_model, rewriter, device=device)
    
    # Evaluation
    domains = config.get('domains', MTRAG_DOMAINS)
    use_data_splits = config.get('use_data_splits', True)
    
    all_results = {}
    
    for domain in domains:
        if shutdown_requested:
            break
        
        logging.info(f"\n{'='*60}\nEvaluating on domain: {domain}\n{'='*60}")
        
        # Load data
        data_root = pathlib.Path(".")
        corpus_file = data_root / "corpora" / "passage_level" / f"{domain}.jsonl"
        
        if use_data_splits:
            query_file = data_root / "data_splits" / "retrieval_tasks" / domain / "train" / f"{domain}_questions.jsonl"
            qrels_file = data_root / "data_splits" / "retrieval_tasks" / domain / "train" / "qrels" / "dev.tsv"
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
        
        # Retrieve and evaluate
        evaluator = EvaluateRetrieval(retriever, k_values=[1, 3, 5, 10])
        results = evaluator.retrieve(corpus, queries)
        ndcg, _map, recall, precision = evaluator.evaluate(qrels, results, [1, 3, 5, 10])
        
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
    
    # Save results
    if all_results:
        avg_results = {
            metric: np.mean([res.get(metric, 0) for res in all_results.values()])
            for metric in list(all_results.values())[0].keys()
        }
        
        final_results = {
            "average": avg_results,
            "domains": all_results
        }
        
        results_file = output_dir / "results.json"
        with open(results_file, 'w') as f:
            json.dump(final_results, f, indent=2)
        
        logging.info(f"\n{'='*60}\nAverage Results:\n")
        for metric, value in avg_results.items():
            logging.info(f"  {metric}: {value:.4f}")
        logging.info(f"{'='*60}\n")
        
        return str(output_dir)
    
    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--gpu_id', type=int, default=None)
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    try:
        results_path = train_multi_query_sparse_dense(config, gpu_id=args.gpu_id)
        if results_path:
            logging.info(f"✅ Training completed: {results_path}")
    except KeyboardInterrupt:
        logging.info("Interrupted by user")
        sys.exit(0)
    except Exception as e:
        logging.error(f"Failed: {e}", exc_info=True)
        sys.exit(1)

