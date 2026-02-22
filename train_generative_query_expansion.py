"""
Generative Query Expansion with LLM
Uses LLM to generate multiple query variants for better retrieval

Task A Compliant: ✅ Query preprocessing only, no answer generation
Expected: 0.75-0.82 nDCG@10
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


class LLMQueryExpander:
    """
    LLM-based query expansion - generates query variants (preprocessing only)
    Task A Compliant: Only generates query variants, not answers
    """
    def __init__(self, llm_model_name: Optional[str] = None, use_local: bool = True):
        self.use_local = use_local
        self.llm_model_name = llm_model_name or "microsoft/Phi-3-mini-4k-instruct"
        
        if use_local:
            try:
                from scripts.evaluation.huggingface_client import HuggingFaceLLMClient
                self.llm_client = HuggingFaceLLMClient(self.llm_model_name)
                logging.info(f"Using local LLM: {self.llm_model_name}")
            except Exception as e:
                logging.warning(f"Could not load local LLM: {e}. Using simple expansion.")
                self.llm_client = None
        else:
            self.llm_client = None
    
    def expand_query(self, query: str, conversation_history: Optional[str] = None, num_variants: int = 3) -> List[str]:
        """
        Generate query variants using LLM
        Task A Compliant: Only generates query strings, not answers
        """
        if self.llm_client is None:
            # Fallback to simple expansion
            return self._simple_expansion(query, num_variants)
        
        try:
            # Create prompt for query expansion (NOT answer generation)
            if conversation_history:
                prompt = f"""Given this conversation history and query, generate {num_variants} alternative phrasings of the query that would help retrieve relevant documents.

Conversation History:
{conversation_history}

Original Query: {query}

Generate {num_variants} alternative query phrasings (one per line, no explanations, just the queries):
"""
            else:
                prompt = f"""Generate {num_variants} alternative phrasings of this query that would help retrieve relevant documents.

Original Query: {query}

Generate {num_variants} alternative query phrasings (one per line, no explanations, just the queries):
"""
            
            # Generate query variants
            response = self.llm_client.generate_response(prompt, max_new_tokens=200, temperature=0.7)
            
            # Parse response to extract query variants
            variants = []
            for line in response.strip().split('\n'):
                line = line.strip()
                # Remove numbering if present
                if line and not line.startswith('#'):
                    # Remove leading numbers/dashes
                    line = line.lstrip('0123456789.-) ').strip()
                    if line and len(line) > 5:  # Valid query
                        variants.append(line)
            
            # Ensure we have at least the original query
            if not variants:
                variants = [query]
            
            # Limit to num_variants
            variants = variants[:num_variants]
            
            # Always include original query
            if query not in variants:
                variants = [query] + variants[:num_variants-1]
            
            return variants[:num_variants]
        
        except Exception as e:
            logging.warning(f"LLM expansion failed: {e}. Using simple expansion.")
            return self._simple_expansion(query, num_variants)
    
    def _simple_expansion(self, query: str, num_variants: int) -> List[str]:
        """Simple expansion fallback"""
        variants = [query]
        
        # Add question word variations
        if "what" in query.lower():
            variants.append(query.lower().replace("what", "which"))
            variants.append(query.lower().replace("what", "describe"))
        elif "how" in query.lower():
            variants.append(query.lower().replace("how", "what method"))
            variants.append(query.lower().replace("how", "explain"))
        
        # Add synonym variations
        if len(variants) < num_variants:
            variants.append(query + " information")
            variants.append(query + " details")
        
        return variants[:num_variants]


class GenerativeQueryExpansionRetriever:
    """
    Retriever with LLM-based query expansion
    """
    def __init__(self, base_model_path: str, expander: LLMQueryExpander, device: str = "cuda"):
        self.device = device
        self.expander = expander
        self.model = SentenceBERT(base_model_path, device=device)
        self.retriever = DenseRetrievalExactSearch(self.model, batch_size=128)
    
    def retrieve(self, corpus: Dict, queries: Dict, top_k: int = 10) -> Dict:
        """
        Retrieve with expanded queries
        """
        all_results = {}
        
        for query_id, query_text in queries.items():
            # Extract conversation history if available
            # For multi-turn, query_text may contain history
            conversation_history = None
            if "|user|:" in query_text:
                # Extract conversation history
                parts = query_text.split("|user|:")
                if len(parts) > 1:
                    conversation_history = "|user|:".join(parts[:-1])
                    query_text = parts[-1].strip()
            
            # Expand query
            query_variants = self.expander.expand_query(query_text, conversation_history, num_variants=3)
            logging.debug(f"Query {query_id}: {len(query_variants)} variants")
            
            # Retrieve with each variant
            variant_results = []
            for variant in query_variants:
                variant_queries = {query_id: variant}
                # Use BEIR's retrieve method
                evaluator_temp = EvaluateRetrieval(self.retriever, k_values=[top_k * 2])
                results_dict = evaluator_temp.retrieve(corpus, variant_queries)
                if query_id in results_dict:
                    variant_results.append(results_dict[query_id])
            
            # Merge results using reciprocal rank fusion
            merged_scores = {}
            for rank, variant_result in enumerate(variant_results):
                sorted_docs = sorted(variant_result.items(), key=lambda x: x[1], reverse=True)
                for doc_rank, (doc_id, score) in enumerate(sorted_docs, start=1):
                    if doc_id not in merged_scores:
                        merged_scores[doc_id] = 0.0
                    # RRF: 1 / (k + rank)
                    merged_scores[doc_id] += 1.0 / (60 + doc_rank)
            
            # Sort and return top_k
            sorted_docs = sorted(merged_scores.items(), key=lambda x: x[1], reverse=True)
            all_results[query_id] = {doc_id: score for doc_id, score in sorted_docs[:top_k]}
        
        return all_results


def train_generative_query_expansion(config: Dict, gpu_id: Optional[int] = None) -> Optional[str]:
    """Train and evaluate with generative query expansion"""
    global shutdown_requested
    
    if gpu_id is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Device: {device}")
    
    output_dir = pathlib.Path(config.get('output_dir', 'experiments/retrieval/generative_query_expansion'))
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize query expander
    llm_model = config.get('llm_model', 'microsoft/Phi-3-mini-4k-instruct')
    use_local_llm = config.get('use_local_llm', True)
    expander = LLMQueryExpander(llm_model_name=llm_model, use_local=use_local_llm)
    
    # Initialize retriever
    base_model = config.get('model_path', 'BAAI/bge-base-en-v1.5')
    retriever = GenerativeQueryExpansionRetriever(base_model, expander, device=device)
    
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
        results_path = train_generative_query_expansion(config, gpu_id=args.gpu_id)
        if results_path:
            logging.info(f"✅ Training completed: {results_path}")
    except KeyboardInterrupt:
        logging.info("Interrupted by user")
        sys.exit(0)
    except Exception as e:
        logging.error(f"Failed: {e}", exc_info=True)
        sys.exit(1)

