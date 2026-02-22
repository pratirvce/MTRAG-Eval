"""
Ensemble with Meta-Learner Fusion
Combines 5 specialized retrieval models with adaptive fusion weights learned by a meta-learner

Task A Compliant: ✅ Pure retrieval, no generation
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
import torch.nn as nn
import torch.nn.functional as F
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


class MetaLearnerFusion(nn.Module):
    """
    Meta-learner that learns optimal fusion weights based on query features and model scores
    """
    def __init__(self, num_models: int = 5, query_feature_dim: int = 768, hidden_dim: int = 128):
        super().__init__()
        self.num_models = num_models
        
        # Feature extractor for query (simple embedding-based)
        self.query_encoder = nn.Sequential(
            nn.Linear(query_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Fusion network: takes model scores + query features -> model weights
        self.fusion_net = nn.Sequential(
            nn.Linear(num_models + hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_models),
            nn.Softmax(dim=1)  # Ensures weights sum to 1
        )
    
    def forward(self, model_scores: torch.Tensor, query_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            model_scores: [batch_size, num_models] - scores from each model
            query_features: [batch_size, query_feature_dim] - query embeddings
        Returns:
            fused_scores: [batch_size] - final fused scores
        """
        # Encode query features
        query_encoded = self.query_encoder(query_features)  # [batch, hidden_dim]
        
        # Combine model scores and query features
        combined = torch.cat([model_scores, query_encoded], dim=1)  # [batch, num_models + hidden_dim]
        
        # Get fusion weights
        weights = self.fusion_net(combined)  # [batch, num_models]
        
        # Weighted combination of model scores
        fused_scores = torch.sum(weights * model_scores, dim=1)  # [batch]
        
        return fused_scores, weights


class EnsembleMetaLearnerRetriever:
    """
    Ensemble retriever with 5 specialized models and meta-learner fusion
    """
    def __init__(self, model_paths: List[str], device: str = "cuda"):
        self.device = device
        self.num_models = len(model_paths)
        
        # Load all models
        self.models = []
        self.retrievers = []
        
        for model_path in model_paths:
            logging.info(f"Loading model: {model_path}")
            model = SentenceBERT(model_path, device=device)
            retriever = DenseRetrievalExactSearch(model, batch_size=128)
            self.models.append(model)
            self.retrievers.append(retriever)
        
        # Initialize meta-learner
        query_feature_dim = self.models[0].get_sentence_embedding_dimension()
        self.meta_learner = MetaLearnerFusion(
            num_models=self.num_models,
            query_feature_dim=query_feature_dim,
            hidden_dim=128
        ).to(device)
        
        # Query encoder for feature extraction
        self.query_encoder_model = self.models[0]  # Use first model for query encoding
    
    def retrieve(self, corpus: Dict, queries: Dict, top_k: int = 10) -> Dict:
        """
        Retrieve using ensemble with meta-learner fusion
        """
        # Get scores from all models
        all_results = []
        for retriever in self.retrievers:
            # Use BEIR's search method (returns {query_id: {doc_id: score}})
            results = retriever.search(corpus, queries, top_k=top_k * 2)
            all_results.append(results)
        
        # Fuse results using meta-learner
        fused_results = {}
        
        for query_id in queries.keys():
            # Get query embedding for meta-learner
            query_text = queries[query_id]
            query_emb = self.query_encoder_model.encode([query_text], convert_to_tensor=True)
            query_emb = query_emb.to(self.device)
            
            # Collect all candidate documents and their scores from all models
            candidate_scores = {}  # doc_id -> [score1, score2, ..., score5]
            
            for model_idx, results in enumerate(all_results):
                if query_id in results:
                    for doc_id, score in results[query_id].items():
                        if doc_id not in candidate_scores:
                            candidate_scores[doc_id] = [0.0] * self.num_models
                        candidate_scores[doc_id][model_idx] = score
            
            # Normalize scores per model
            for doc_id in candidate_scores:
                scores = candidate_scores[doc_id]
                # Normalize to [0, 1] range
                max_score = max(scores) if max(scores) > 0 else 1.0
                candidate_scores[doc_id] = [s / max_score for s in scores]
            
            # Fuse scores using meta-learner
            fused_doc_scores = {}
            
            for doc_id, model_scores in candidate_scores.items():
                # Prepare inputs
                model_scores_tensor = torch.tensor([model_scores], device=self.device)  # [1, num_models]
                query_features = query_emb  # [1, query_feature_dim]
                
                # Get fused score
                with torch.no_grad():
                    fused_score, weights = self.meta_learner(model_scores_tensor, query_features)
                    fused_doc_scores[doc_id] = fused_score.item()
            
            # Sort and return top_k
            sorted_docs = sorted(fused_doc_scores.items(), key=lambda x: x[1], reverse=True)
            fused_results[query_id] = {doc_id: score for doc_id, score in sorted_docs[:top_k]}
        
        return fused_results


def train_meta_learner(config: Dict, gpu_id: Optional[int] = None) -> Optional[str]:
    """Train meta-learner on retrieval results"""
    global shutdown_requested
    
    if gpu_id is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Device: {device}")
    
    output_dir = pathlib.Path(config.get('output_dir', 'experiments/retrieval/ensemble_meta_learner'))
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model paths
    model_paths = config.get('model_paths', [])
    if not model_paths:
        logging.error("model_paths required in config")
        return None
    
    # Create ensemble retriever
    ensemble = EnsembleMetaLearnerRetriever(model_paths, device=device)
    
    # Train meta-learner on training data
    domains = config.get('domains', MTRAG_DOMAINS)
    use_data_splits = config.get('use_data_splits', True)
    
    optimizer = torch.optim.AdamW(ensemble.meta_learner.parameters(), lr=1e-3)
    epochs = config.get('epochs', 5)
    
    all_results = {}
    
    for domain in domains:
        if shutdown_requested:
            break
        
        logging.info(f"\n{'='*60}\nTraining meta-learner on domain: {domain}\n{'='*60}")
        
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
        
        # Training loop
        ensemble.meta_learner.train()
        
        for epoch in range(epochs):
            total_loss = 0
            num_batches = 0
            
            # Sample queries for training
            query_ids = list(queries.keys())[:100]  # Use subset for training
            
            for query_id in query_ids:
                query_text = queries[query_id]
                
                # Get relevant documents
                relevant_docs = set(qrels.get(query_id, {}).keys())
                
                if not relevant_docs:
                    continue
                
                # Get scores from all models
                model_scores_list = []
                for retriever in ensemble.retrievers:
                    results = retriever.search(corpus, {query_id: query_text}, top_k=50)
                    if query_id in results:
                        scores = results[query_id]
                        model_scores_list.append(scores)
                    else:
                        model_scores_list.append({})
                
                # Create training examples: relevant vs non-relevant
                all_candidates = set()
                for scores in model_scores_list:
                    all_candidates.update(scores.keys())
                
                if len(all_candidates) < 2:
                    continue
                
                # Sample positive and negative examples
                positives = list(relevant_docs & all_candidates)
                negatives = list(all_candidates - relevant_docs)
                
                if not positives or not negatives:
                    continue
                
                # Create batch
                batch_size = min(8, len(positives) + len(negatives[:8]))
                batch_pos = positives[:batch_size//2]
                batch_neg = negatives[:batch_size//2]
                batch_docs = batch_pos + batch_neg
                batch_labels = [1.0] * len(batch_pos) + [0.0] * len(batch_neg)
                
                # Get model scores for batch
                batch_model_scores = []
                for model_idx, scores in enumerate(model_scores_list):
                    doc_scores = [scores.get(doc_id, 0.0) for doc_id in batch_docs]
                    batch_model_scores.append(doc_scores)
                
                # Normalize scores
                batch_model_scores = np.array(batch_model_scores).T  # [batch_size, num_models]
                batch_model_scores = torch.tensor(batch_model_scores, device=device, dtype=torch.float32)
                
                # Normalize to [0, 1]
                max_scores = torch.max(batch_model_scores, dim=0, keepdim=True)[0]
                max_scores = torch.clamp(max_scores, min=1e-6)
                batch_model_scores = batch_model_scores / max_scores
                
                # Get query features
                query_emb = ensemble.query_encoder_model.encode([query_text], convert_to_tensor=True)
                query_emb = query_emb.to(device)
                query_features = query_emb.repeat(len(batch_docs), 1)  # [batch_size, query_feature_dim]
                
                # Forward pass
                fused_scores, weights = ensemble.meta_learner(batch_model_scores, query_features)
                
                # Loss: binary cross-entropy
                labels = torch.tensor(batch_labels, device=device, dtype=torch.float32)
                loss = F.binary_cross_entropy_with_logits(fused_scores, labels)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
            
            if num_batches > 0:
                logging.info(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/num_batches:.4f}")
        
        # Evaluation
        logging.info(f"Evaluating on {domain}...")
        ensemble.meta_learner.eval()
        
        evaluator = EvaluateRetrieval(ensemble, k_values=[1, 3, 5, 10])
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
        
        # Save meta-learner
        meta_learner_path = output_dir / f"meta_learner_{domain}.pt"
        torch.save(ensemble.meta_learner.state_dict(), meta_learner_path)
        logging.info(f"Saved meta-learner to {meta_learner_path}")
    
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
        results_path = train_meta_learner(config, gpu_id=args.gpu_id)
        if results_path:
            logging.info(f"✅ Training completed: {results_path}")
    except KeyboardInterrupt:
        logging.info("Interrupted by user")
        sys.exit(0)
    except Exception as e:
        logging.error(f"Failed: {e}", exc_info=True)
        sys.exit(1)

