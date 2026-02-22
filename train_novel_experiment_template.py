"""
Template for Novel Retrieval Experiments
Task A Compliant: Retrieval only, no text generation

This template can be quickly adapted for new novel experiments.
Replace <EXPERIMENT_NAME> and <NOVEL_COMPONENT> with your specific implementation.
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
from sentence_transformers import SentenceTransformer
import torch
import torch.nn as nn
import torch.nn.functional as F
import signal

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

# ============================================================================
# STEP 1: Define Your Novel Component
# ============================================================================
class NovelComponent(nn.Module):
    """
    Replace this with your novel component:
    - Temporal attention
    - Uncertainty estimator
    - Graph neural network
    - Causal graph builder
    - etc.
    """
    def __init__(self, hidden_dim=768):
        super().__init__()
        self.hidden_dim = hidden_dim
        # Add your novel architecture here
        self.novel_layer = nn.Linear(hidden_dim, hidden_dim)
    
    def forward(self, input_emb):
        """Your novel forward pass"""
        return self.novel_layer(input_emb)

# ============================================================================
# STEP 2: Define Your Retriever Model
# ============================================================================
class NovelRetriever(nn.Module):
    """Your novel retriever that integrates the novel component"""
    def __init__(self, base_model_name='BAAI/bge-large-en-v1.5'):
        super().__init__()
        self.base_encoder = SentenceTransformer(base_model_name)
        self.novel_component = NovelComponent()  # Your novel component
    
    def encode_query(self, query_text: str, **kwargs):
        """Encode query (add novel processing here)"""
        query_emb = self.base_encoder.encode(query_text, convert_to_tensor=True)
        # Apply novel component
        query_emb = self.novel_component(query_emb)
        return query_emb
    
    def encode_documents(self, doc_texts: List[str], **kwargs):
        """Encode documents (add novel processing here)"""
        doc_embs = self.base_encoder.encode(doc_texts, convert_to_tensor=True, batch_size=64)
        # Apply novel component if needed
        return doc_embs

# ============================================================================
# STEP 3: Data Loading Function
# ============================================================================
def load_data(domain: str, data_root: pathlib.Path, use_data_splits: bool = True):
    """Load data for evaluation"""
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
        return corpus, queries, qrels
    except Exception as e:
        logging.error(f"Error loading {domain}: {e}")
        return {}, {}, {}

# ============================================================================
# STEP 4: Main Experiment Function
# ============================================================================
def run_novel_experiment(config: Dict, gpu_id: Optional[int] = None) -> Optional[str]:
    """
    Main experiment function
    Replace 'novel_experiment' with your experiment name
    """
    global shutdown_requested
    
    if gpu_id is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        logging.info(f"Using GPU {gpu_id}")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Device: {device}")
    
    # Update output_dir with your experiment name
    output_dir = pathlib.Path(config.get('output_dir', 'experiments/retrieval/novel_experiment'))
    output_dir.mkdir(parents=True, exist_ok=True)
    
    model_name = config.get('model_path', 'BAAI/bge-large-en-v1.5')
    domains = config.get('domains', MTRAG_DOMAINS)
    use_data_splits = config.get('use_data_splits', True)
    
    # Initialize your novel retriever
    logging.info(f"Initializing Novel Experiment with {model_name}")
    retriever = NovelRetriever(base_model_name=model_name)
    retriever.to(device)
    retriever.eval()
    
    # Evaluation
    logging.info("Evaluating Novel Experiment...")
    all_results = {}
    
    for domain in domains:
        if shutdown_requested:
            break
        
        logging.info(f"\n{'='*60}")
        logging.info(f"Evaluating domain: {domain}")
        logging.info(f"{'='*60}")
        
        data_root = pathlib.Path(".")
        corpus, queries, qrels = load_data(domain, data_root, use_data_splits)
        
        if not corpus or not queries:
            continue
        
        # Encode corpus
        logging.info("Encoding corpus...")
        corpus_texts = [corpus[doc_id].get('text', '') for doc_id in corpus.keys()]
        doc_embs = retriever.encode_documents(corpus_texts)
        doc_embs = doc_embs.to(device)
        doc_ids = list(corpus.keys())
        
        # Retrieve
        results = {}
        
        for query_id, query_text in queries.items():
            # Encode query with novel component
            query_emb = retriever.encode_query(query_text)
            query_emb = query_emb.to(device)
            
            # Compute similarities
            scores = torch.mm(query_emb.unsqueeze(0), doc_embs.t()).squeeze(0)
            scores = scores.cpu().numpy()
            
            # Get top-k
            top_indices = np.argsort(scores)[::-1][:100]
            results[query_id] = {
                doc_ids[idx]: float(scores[idx]) for idx in top_indices
            }
        
        # Evaluate
        evaluator = EvaluateRetrieval(None, k_values=[1, 3, 5, 10])
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
        
        logging.info(f"Domain {domain} - nDCG@10: {ndcg.get('NDCG@10', 0):.4f}")
    
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
        
        logging.info(f"\n✅ Novel Experiment completed!")
        logging.info(f"Average nDCG@10: {avg_results.get('nDCG@10', 0):.4f}")
        logging.info(f"Results saved to: {results_file}")
        
        return str(results_file)
    
    return None

# ============================================================================
# STEP 5: Main Entry Point
# ============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help='Config file path')
    parser.add_argument('--gpu', type=int, help='GPU ID')
    args = parser.parse_args()
    
    if args.config:
        with open(args.config, 'r') as f:
            config = json.load(f)
    else:
        config = {}
    
    run_novel_experiment(config, gpu_id=args.gpu)

# ============================================================================
# USAGE INSTRUCTIONS:
# ============================================================================
"""
1. Copy this template: cp train_novel_experiment_template.py train_<your_experiment>.py
2. Replace <EXPERIMENT_NAME> with your experiment name
3. Implement NovelComponent class with your novel architecture
4. Update NovelRetriever to integrate your component
5. Modify encode_query/encode_documents as needed
6. Create tier1 wrapper: cp train_catr_temporal_retrieval_tier1.py train_<your_experiment>_tier1.py
7. Update tier1 wrapper to import your experiment function
8. Add to auto-runner PENDING_EXPERIMENTS list
9. Verify Task A compliance (no text generation)
"""

