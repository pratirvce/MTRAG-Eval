"""
Iterative Retrieval with Query Expansion Feedback
Initial retrieval → LLM generates expansion terms → Re-retrieve

Task A Compliant: ✅ Feedback is query expansion terms only, not answers
Expected: 0.72-0.80 nDCG@10
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


class IterativeQueryExpander:
    """
    Iterative query expansion using retrieved documents as feedback
    Task A Compliant: Only generates query expansion terms, not answers
    """
    def __init__(self, llm_model_name: Optional[str] = None, use_local: bool = True):
        self.use_local = use_local
        self.llm_model_name = llm_model_name or "microsoft/Phi-3-mini-4k-instruct"
        
        if use_local:
            try:
                from scripts.evaluation.huggingface_client import HuggingFaceLLMClient
                self.llm_client = HuggingFaceLLMClient(self.llm_model_name)
                logging.info(f"Using local LLM for expansion: {self.llm_model_name}")
            except Exception as e:
                logging.warning(f"Could not load local LLM: {e}. Using simple expansion.")
                self.llm_client = None
        else:
            self.llm_client = None
    
    def expand_with_feedback(self, query: str, retrieved_docs: List[str], num_terms: int = 5) -> str:
        """
        Generate expansion terms based on retrieved documents
        Task A Compliant: Only generates terms, not answers
        """
        if self.llm_client is None or not retrieved_docs:
            # Fallback: simple expansion
            return query + " " + " ".join(retrieved_docs[0].split()[:5])
        
        try:
            # Create prompt for term extraction (NOT answer generation)
            doc_texts = "\n".join([doc[:200] for doc in retrieved_docs[:3]])  # Top 3 docs
            
            prompt = f"""Based on these retrieved documents, what key terms should be added to improve retrieval for this query?

Query: {query}

Retrieved Documents:
{doc_texts}

Generate {num_terms} key terms or phrases (comma-separated, no explanations, just the terms):
"""
            
            # Generate expansion terms
            response = self.llm_client.generate_response(prompt, max_new_tokens=50, temperature=0.5)
            
            # Extract terms
            terms = self._extract_terms(response)
            
            # Combine with original query
            expanded_query = f"{query} {' '.join(terms)}"
            return expanded_query.strip()
        
        except Exception as e:
            logging.warning(f"LLM expansion failed: {e}. Using simple expansion.")
            return query
    
    def _extract_terms(self, response: str) -> List[str]:
        """Extract terms from LLM response"""
        # Split by comma or newline
        terms = []
        for part in response.replace('\n', ',').split(','):
            term = part.strip().strip('.-) ')
            if term and len(term) > 2:
                terms.append(term)
        return terms[:5]  # Limit to 5 terms


class IterativeQueryExpansionRetriever:
    """
    Retriever with iterative query expansion
    """
    def __init__(self, base_model_path: str, expander: IterativeQueryExpander, device: str = "cuda"):
        self.device = device
        self.expander = expander
        self.model = SentenceBERT(base_model_path, device=device)
        self.retriever = DenseRetrievalExactSearch(self.model, batch_size=128)
    
    def retrieve(self, corpus: Dict, queries: Dict, top_k: int = 10) -> Dict:
        """
        Iterative retrieval: initial → expansion → re-retrieve
        """
        all_results = {}
        
        for query_id, query_text in queries.items():
            # Initial retrieval (top 20)
            initial_queries = {query_id: query_text}
            # Use BEIR's retrieve method
            evaluator_temp = EvaluateRetrieval(self.retriever, k_values=[20])
            initial_results_dict = evaluator_temp.retrieve(corpus, initial_queries)
            initial_results = initial_results_dict.get(query_id, {})
            
            if query_id not in initial_results or not initial_results[query_id]:
                all_results[query_id] = {}
                continue
            
            # Get top documents for feedback
            top_docs = sorted(initial_results[query_id].items(), key=lambda x: x[1], reverse=True)[:5]
            doc_texts = [corpus[doc_id].get('text', '') for doc_id, _ in top_docs]
            
            # Generate expansion terms from feedback
            expanded_query = self.expander.expand_with_feedback(query_text, doc_texts)
            logging.debug(f"Query {query_id}: Expanded from '{query_text[:50]}...' to '{expanded_query[:50]}...'")
            
            # Re-retrieve with expanded query
            expanded_queries = {query_id: expanded_query}
            # Use BEIR's retrieve method
            evaluator_temp = EvaluateRetrieval(self.retriever, k_values=[top_k])
            final_results_dict = evaluator_temp.retrieve(corpus, expanded_queries)
            final_results = final_results_dict.get(query_id, {})
            
            if query_id in final_results:
                all_results[query_id] = final_results[query_id]
            else:
                # Fallback to initial results
                all_results[query_id] = {doc_id: score for doc_id, score in top_docs[:top_k]}
        
        return all_results


def train_iterative_query_expansion(config: Dict, gpu_id: Optional[int] = None) -> Optional[str]:
    """Train and evaluate iterative query expansion"""
    global shutdown_requested
    
    if gpu_id is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Device: {device}")
    
    output_dir = pathlib.Path(config.get('output_dir', 'experiments/retrieval/iterative_query_expansion'))
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize expander
    llm_model = config.get('llm_model', 'microsoft/Phi-3-mini-4k-instruct')
    use_local_llm = config.get('use_local_llm', True)
    expander = IterativeQueryExpander(llm_model_name=llm_model, use_local=use_local_llm)
    
    # Initialize retriever
    base_model = config.get('model_path', 'BAAI/bge-base-en-v1.5')
    retriever = IterativeQueryExpansionRetriever(base_model, expander, device=device)
    
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
        results_path = train_iterative_query_expansion(config, gpu_id=args.gpu_id)
        if results_path:
            logging.info(f"✅ Training completed: {results_path}")
    except KeyboardInterrupt:
        logging.info("Interrupted by user")
        sys.exit(0)
    except Exception as e:
        logging.error(f"Failed: {e}", exc_info=True)
        sys.exit(1)

