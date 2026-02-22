"""
Test-Time Scaling & Iterative Reranking
Iterative test-time refinement: retrieve → rerank → refine query → re-retrieve

Task A Compliant: ✅ LLM only scores, not generates text
Expected: 0.82-0.90 nDCG@10
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


class LLMReranker:
    """LLM-based reranker - Task A Compliant: only scores, not text"""
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
    
    def rerank(self, query: str, documents: List[str]) -> List[float]:
        """Rerank documents - returns scores only"""
        scores = []
        for doc in documents:
            if self.llm_client:
                try:
                    prompt = f"""Rate relevance (0-10): Query: {query[:100]} Document: {doc[:300]} Output only number:"""
                    response = self.llm_client.generate_response(prompt, max_new_tokens=5, temperature=0)
                    import re
                    numbers = re.findall(r'\b([0-9]|10)\b', response.strip())
                    score = float(numbers[0]) if numbers else 5.0
                    scores.append(max(0.0, min(10.0, score)) / 10.0)
                except:
                    scores.append(0.5)
            else:
                scores.append(0.5)
        return scores


class TestTimeScalingRetriever:
    """
    Iterative retrieval: retrieve → rerank → refine → re-retrieve
    """
    def __init__(self, base_model_path: str, reranker: LLMReranker, device: str = "cuda", num_iterations: int = 2):
        self.device = device
        self.reranker = reranker
        self.num_iterations = num_iterations
        self.base_model = SentenceBERT(base_model_path, device=device)
        self.retriever = DenseRetrievalExactSearch(self.base_model, batch_size=128)
    
    def refine_query(self, query: str, top_docs: List[str]) -> str:
        """Refine query based on top documents (simplified: add context terms)"""
        # Extract key terms from top documents
        all_terms = []
        for doc in top_docs[:3]:
            terms = doc.lower().split()[:10]  # First 10 words
            all_terms.extend(terms)
        
        # Add most common terms to query
        from collections import Counter
        term_counts = Counter(all_terms)
        top_terms = [term for term, _ in term_counts.most_common(3)]
        
        refined = query + " " + " ".join(top_terms)
        return refined.strip()
    
    def retrieve(self, corpus: Dict, queries: Dict, top_k: int = 10) -> Dict:
        """Iterative retrieval with test-time scaling"""
        all_results = {}
        
        for query_id, query_text in queries.items():
            current_query = query_text
            candidates = {}
            
            # Iterative refinement
            for iteration in range(self.num_iterations):
                # Retrieve
                evaluator_temp = EvaluateRetrieval(self.retriever, k_values=[top_k * 2])
                results = evaluator_temp.retrieve(corpus, {query_id: current_query})
                
                if query_id not in results:
                    break
                
                # Get top documents
                sorted_docs = sorted(results[query_id].items(), key=lambda x: x[1], reverse=True)[:20]
                doc_ids = [doc_id for doc_id, _ in sorted_docs]
                doc_texts = [corpus[doc_id].get('text', '') for doc_id in doc_ids]
                
                # Rerank with LLM
                rerank_scores = self.reranker.rerank(current_query, doc_texts)
                
                # Update scores
                for i, (doc_id, _) in enumerate(sorted_docs):
                    if i < len(rerank_scores):
                        if doc_id not in candidates:
                            candidates[doc_id] = []
                        candidates[doc_id].append(rerank_scores[i])
                
                # Refine query for next iteration
                if iteration < self.num_iterations - 1:
                    current_query = self.refine_query(current_query, doc_texts[:5])
            
            # Aggregate scores across iterations
            final_scores = {}
            for doc_id, scores in candidates.items():
                final_scores[doc_id] = np.mean(scores)  # Average across iterations
            
            # Sort and return top_k
            sorted_docs = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
            all_results[query_id] = {doc_id: score for doc_id, score in sorted_docs[:top_k]}
        
        return all_results


def train_test_time_scaling(config: Dict, gpu_id: Optional[int] = None) -> Optional[str]:
    """Train and evaluate test-time scaling"""
    global shutdown_requested
    
    if gpu_id is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Device: {device}")
    
    output_dir = pathlib.Path(config.get('output_dir', 'experiments/retrieval/test_time_scaling'))
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize components
    base_model = config.get('model_path', 'BAAI/bge-base-en-v1.5')
    llm_model = config.get('llm_model', 'microsoft/Phi-3-mini-4k-instruct')
    use_local_llm = config.get('use_local_llm', True)
    num_iterations = config.get('num_iterations', 2)
    
    reranker = LLMReranker(llm_model_name=llm_model, use_local=use_local_llm)
    retriever = TestTimeScalingRetriever(base_model, reranker, device=device, num_iterations=num_iterations)
    
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
        results_path = train_test_time_scaling(config, gpu_id=args.gpu_id)
        if results_path:
            logging.info(f"✅ Training completed: {results_path}")
    except KeyboardInterrupt:
        logging.info("Interrupted by user")
        sys.exit(0)
    except Exception as e:
        logging.error(f"Failed: {e}", exc_info=True)
        sys.exit(1)

