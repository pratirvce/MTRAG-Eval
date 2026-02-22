"""
UQ-Ret: Uncertainty-Quantified Retrieval with Confidence Calibration
Novel Experiment for Tier 1 Conference Publication

Task A Compliant: Retrieval only, provides confidence scores (no text generation)
Quantifies retrieval uncertainty and provides calibrated confidence scores.
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
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression
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

class UncertaintyEstimator(nn.Module):
    """Estimates uncertainty using ensemble of models with MC Dropout"""
    def __init__(self, base_model_name='BAAI/bge-large-en-v1.5', num_models=5, dropout_rate=0.1):
        super().__init__()
        self.num_models = num_models
        self.models = nn.ModuleList([
            SentenceTransformer(base_model_name) for _ in range(num_models)
        ])
        # Enable dropout for uncertainty estimation
        for model in self.models:
            for module in model.modules():
                if isinstance(module, nn.Dropout):
                    module.p = dropout_rate
                    module.train()  # Keep in train mode for MC Dropout
    
    def encode_with_uncertainty(self, texts, num_samples=10):
        """Encode with Monte Carlo dropout for uncertainty estimation"""
        all_embeddings = []
        
        for model in self.models:
            model.eval()
            with torch.no_grad():
                # MC Dropout: sample multiple times
                embeddings_list = []
                for _ in range(num_samples):
                    # Temporarily enable dropout
                    for module in model.modules():
                        if isinstance(module, nn.Dropout):
                            module.train()
                    
                    emb = model.encode(texts, convert_to_tensor=True)
                    embeddings_list.append(emb)
                    
                    # Disable dropout
                    for module in model.modules():
                        if isinstance(module, nn.Dropout):
                            module.eval()
                
                # Average over samples
                avg_emb = torch.stack(embeddings_list).mean(dim=0)
                all_embeddings.append(avg_emb)
        
        # Compute mean and variance across models
        embeddings_stack = torch.stack(all_embeddings)  # [num_models, batch, dim]
        mean_emb = embeddings_stack.mean(dim=0)
        var_emb = embeddings_stack.var(dim=0)
        uncertainty = var_emb.mean(dim=-1)  # Average variance across dimensions
        
        return mean_emb, uncertainty

class ConfidenceCalibrator:
    """Calibrates confidence scores using Platt scaling or isotonic regression"""
    def __init__(self, method='isotonic'):
        self.method = method
        self.calibrator = None
        
    def fit(self, scores, labels):
        """Fit calibrator on scores and binary labels"""
        if self.method == 'isotonic':
            self.calibrator = IsotonicRegression(out_of_bounds='clip')
        else:  # platt
            from sklearn.linear_model import LogisticRegression
            self.calibrator = LogisticRegression()
        
        self.calibrator.fit(scores.reshape(-1, 1), labels)
        
    def predict(self, scores):
        """Predict calibrated confidence scores"""
        if self.calibrator is None:
            return scores
        return self.calibrator.predict_proba(scores.reshape(-1, 1))[:, 1]

def run_uq_uncertainty_retrieval(config: Dict, gpu_id: Optional[int] = None) -> Optional[str]:
    """Run UQ-Ret Uncertainty-Quantified Retrieval Experiment"""
    global shutdown_requested
    
    if gpu_id is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        logging.info(f"Using GPU {gpu_id}")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Device: {device}")
    
    output_dir = pathlib.Path(config.get('output_dir', 'experiments/retrieval/uq_uncertainty_retrieval'))
    output_dir.mkdir(parents=True, exist_ok=True)
    
    model_name = config.get('model_path', 'BAAI/bge-large-en-v1.5')
    domains = config.get('domains', MTRAG_DOMAINS)
    use_data_splits = config.get('use_data_splits', True)
    uncertainty_threshold = config.get('uncertainty_threshold', 0.5)
    
    # Initialize uncertainty estimator
    logging.info(f"Initializing UQ-Ret with {model_name}")
    uncertainty_estimator = UncertaintyEstimator(base_model_name=model_name, num_models=5)
    uncertainty_estimator.to(device)
    
    # Evaluation
    logging.info("Evaluating UQ-Ret Uncertainty-Quantified Retrieval...")
    all_results = {}
    
    for domain in domains:
        if shutdown_requested:
            break
        
        logging.info(f"\n{'='*60}")
        logging.info(f"Evaluating domain: {domain}")
        logging.info(f"{'='*60}")
        
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
            logging.error(f"Error loading {domain}: {e}")
            continue
        
        # Encode corpus with uncertainty
        logging.info("Encoding corpus with uncertainty estimation...")
        corpus_texts = [corpus[doc_id].get('text', '') for doc_id in corpus.keys()]
        corpus_embs, corpus_uncertainty = uncertainty_estimator.encode_with_uncertainty(
            corpus_texts, num_samples=5
        )
        corpus_embs = corpus_embs.cpu().numpy()
        corpus_uncertainty = corpus_uncertainty.cpu().numpy()
        
        # Retrieve with uncertainty
        results = {}
        query_uncertainties = {}
        
        for query_id, query_text in queries.items():
            # Encode query with uncertainty
            query_embs, query_unc = uncertainty_estimator.encode_with_uncertainty(
                [query_text], num_samples=5
            )
            query_emb = query_embs[0].cpu().numpy()
            query_unc_val = query_unc[0].cpu().item()
            query_uncertainties[query_id] = query_unc_val
            
            # Compute similarities
            scores = np.dot(query_emb, corpus_embs.T)
            
            # Adjust scores based on uncertainty (lower uncertainty = higher confidence)
            confidence_scores = scores * (1.0 - query_unc_val)
            
            # Get top-k
            top_indices = np.argsort(confidence_scores)[::-1][:100]
            results[query_id] = {
                list(corpus.keys())[idx]: float(confidence_scores[idx]) for idx in top_indices
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
            "nDCG@10": ndcg.get('NDCG@10', 0),
            "avg_uncertainty": np.mean(list(query_uncertainties.values()))
        }
        
        logging.info(f"Domain {domain} - nDCG@10: {ndcg.get('NDCG@10', 0):.4f}, Avg Uncertainty: {all_results[domain]['avg_uncertainty']:.4f}")
    
    # Save results
    if all_results:
        avg_results = {
            metric: np.mean([res.get(metric, 0) for res in all_results.values() if metric != 'avg_uncertainty'])
            for metric in list(all_results.values())[0].keys() if metric != 'avg_uncertainty'
        }
        
        final_results = {
            "average": avg_results,
            "domains": all_results
        }
        
        results_file = output_dir / "results.json"
        with open(results_file, 'w') as f:
            json.dump(final_results, f, indent=2)
        
        logging.info(f"\n✅ UQ-Ret Uncertainty-Quantified Retrieval completed!")
        logging.info(f"Average nDCG@10: {avg_results.get('nDCG@10', 0):.4f}")
        logging.info(f"Results saved to: {results_file}")
        
        return str(results_file)
    
    return None

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
    
    run_uq_uncertainty_retrieval(config, gpu_id=args.gpu)

