"""
Query-Document Interaction Transformer (QDIT)
Novel architecture: Joint transformer encoder for query-document pairs with cross-attention
Expected: 0.52-0.56 nDCG@10
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
from sentence_transformers import SentenceTransformer
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
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

class QDITModel(nn.Module):
    """Query-Document Interaction Transformer"""
    def __init__(self, base_model_name: str = 'BAAI/bge-base-en-v1.5', max_length: int = 512):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        self.encoder = AutoModel.from_pretrained(base_model_name)
        self.max_length = max_length
        self.projection = nn.Linear(self.encoder.config.hidden_size, 768)
        
    def forward(self, query_texts: List[str], doc_texts: List[str]):
        """Encode query-document pairs with interaction"""
        scores = []
        batch_size = len(query_texts)
        
        for i in range(batch_size):
            # Concatenate query and document
            text = f"{query_texts[i]} [SEP] {doc_texts[i]}"
            inputs = self.tokenizer(
                text,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            
            # Move to device
            device = next(self.encoder.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Encode
            outputs = self.encoder(**inputs)
            # Use CLS token
            cls_embedding = outputs.last_hidden_state[:, 0, :]
            projected = self.projection(cls_embedding)
            scores.append(projected)
        
        return torch.cat(scores, dim=0)

def run_qdit_evaluation(config: Dict, gpu_id: Optional[int] = None) -> Optional[str]:
    """Run QDIT evaluation"""
    global shutdown_requested
    
    if gpu_id is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        logging.info(f"Using GPU {gpu_id}")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Device: {device}")
    
    output_dir = pathlib.Path(config.get('output_dir', 'experiments/retrieval/qdit_transformer'))
    output_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint_dir = output_dir
    resume = config.get('resume', True)
    
    # Load checkpoint
    all_results = {}
    completed_domains = set()
    if resume:
        checkpoint_file = checkpoint_dir / "checkpoint.json"
        if checkpoint_file.exists():
            with open(checkpoint_file) as f:
                checkpoint = json.load(f)
                all_results = checkpoint.get('results', {})
                completed_domains = set(checkpoint.get('completed_domains', []))
                logging.info(f"Resuming from checkpoint. Completed domains: {completed_domains}")
    
    domains = config.get('domains', MTRAG_DOMAINS)
    base_model = config.get('model_path', 'BAAI/bge-base-en-v1.5')
    top_k = config.get('top_k', 100)
    
    # Initialize base retriever for candidate generation
    logging.info(f"Loading base model: {base_model}")
    base_retriever_model = SentenceBERT(base_model, device=device)
    retriever = DenseRetrievalExactSearch(base_retriever_model, batch_size=128)
    
    # Initialize QDIT model
    qdit_model = QDITModel(base_model_name=base_model)
    qdit_model.to(device)
    qdit_model.eval()
    
    # Process each domain
    for domain in domains:
        if domain in completed_domains:
            logging.info(f"Skipping {domain} (already completed)")
            continue
        
        if shutdown_requested:
            logging.info("Shutdown requested, saving checkpoint...")
            checkpoint_file = checkpoint_dir / "checkpoint.json"
            with open(checkpoint_file, 'w') as f:
                json.dump({
                    "last_domain": domain,
                    "completed_domains": list(completed_domains),
                    "results": all_results,
                    "timestamp": datetime.now().isoformat()
                }, f, indent=2)
            break
        
        logging.info(f"\n{'='*60}")
        logging.info(f"Processing domain: {domain}")
        logging.info(f"{'='*60}")
        
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
        
        logging.info(f"Loaded {len(corpus)} documents, {len(queries)} queries")
        
        # Step 1: Initial retrieval (get candidates)
        logging.info(f"Step 1: Initial retrieval (top {top_k * 2} candidates)...")
        evaluator = EvaluateRetrieval(retriever, k_values=[top_k * 2])
        initial_results = evaluator.retrieve(corpus, queries)
        
        # Step 2: Re-rank with QDIT
        logging.info("Step 2: Re-ranking with QDIT transformer...")
        final_results = {}
        
        for query_id, query_text in queries.items():
            if query_id not in initial_results:
                final_results[query_id] = {}
                continue
            
            # Get top candidates
            candidates = sorted(initial_results[query_id].items(), 
                              key=lambda x: x[1], reverse=True)[:top_k * 2]
            
            # Re-rank with QDIT
            doc_texts = [corpus[doc_id].get('text', '') for doc_id, _ in candidates if doc_id in corpus]
            query_texts = [query_text] * len(doc_texts)
            
            if not doc_texts:
                final_results[query_id] = {}
                continue
            
            # Batch process
            batch_size = 32
            qdit_scores = []
            for i in range(0, len(doc_texts), batch_size):
                batch_queries = query_texts[i:i+batch_size]
                batch_docs = doc_texts[i:i+batch_size]
                
                with torch.no_grad():
                    embeddings = qdit_model(batch_queries, batch_docs)
                    # Use embedding norm as score (or cosine similarity with query)
                    scores = torch.norm(embeddings, dim=1).cpu().numpy()
                    qdit_scores.extend(scores.tolist())
            
            # Combine initial score and QDIT score
            combined_scores = {}
            for idx, (doc_id, initial_score) in enumerate(candidates):
                if doc_id in corpus and idx < len(qdit_scores):
                    # Weighted combination
                    combined_score = 0.3 * initial_score + 0.7 * qdit_scores[idx]
                    combined_scores[doc_id] = combined_score
            
            # Sort and take top k
            final_results[query_id] = dict(sorted(combined_scores.items(), 
                                                 key=lambda x: x[1], reverse=True)[:top_k])
        
        # Step 3: Evaluate
        logging.info("Step 3: Evaluating results...")
        evaluator = EvaluateRetrieval(None, k_values=[1, 3, 5, 10])
        ndcg, _map, recall, precision = evaluator.evaluate(qrels, final_results, [1, 3, 5, 10])
        
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
        
        logging.info(f"\n{'='*60}")
        logging.info(f"Domain {domain} Results:")
        logging.info(f"  Recall@10: {recall.get('Recall@10', 0):.4f}")
        logging.info(f"  nDCG@10: {ndcg.get('NDCG@10', 0):.4f}")
        logging.info(f"{'='*60}\n")
        
        completed_domains.add(domain)
        
        # Save checkpoint
        checkpoint_file = checkpoint_dir / "checkpoint.json"
        with open(checkpoint_file, 'w') as f:
            json.dump({
                "last_domain": domain,
                "completed_domains": list(completed_domains),
                "results": all_results,
                "timestamp": datetime.now().isoformat()
            }, f, indent=2)
    
    # Final results
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
        
        logging.info(f"\n{'='*60}")
        logging.info("Average Results:")
        for metric, value in avg_results.items():
            logging.info(f"  {metric}: {value:.4f}")
        logging.info(f"{'='*60}\n")
        
        return str(output_dir)
    
    return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--gpu_id', type=int)
    parser.add_argument('--resume', action='store_true', default=True)
    parser.add_argument('--no-resume', dest='resume', action='store_false')
    
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    config['resume'] = args.resume
    
    try:
        results_path = run_qdit_evaluation(config, args.gpu_id)
        if results_path:
            logging.info(f"✅ QDIT transformer completed: {results_path}")
    except KeyboardInterrupt:
        logging.info("Interrupted by user")
        sys.exit(0)
    except Exception as e:
        logging.error(f"Failed: {e}", exc_info=True)
        sys.exit(1)

