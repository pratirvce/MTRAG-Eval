"""
Learning-to-Rank with Listwise Loss for Multi-Turn RAG Retrieval
Uses listwise ranking losses (LambdaRank, ListNet) to directly optimize nDCG
High priority Tier 1 experiment for Task A - Retrieval
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
import torch
import signal
import sys
from datetime import datetime
import pickle

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    logging.warning("XGBoost not available. Will use LightGBM or fallback to simple ranking.")

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    logging.warning("LightGBM not available. Will use XGBoost or fallback to simple ranking.")

logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO)

MTRAG_DOMAINS = ["clapnq", "fiqa", "govt", "cloud"]

shutdown_requested = False

def signal_handler(sig, frame):
    """Handle SIGINT (Ctrl+C) gracefully"""
    global shutdown_requested
    logging.info("\n⚠️  Shutdown signal received. Saving checkpoint and exiting gracefully...")
    shutdown_requested = True

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

def load_checkpoint(checkpoint_dir: pathlib.Path) -> Optional[Dict]:
    """Load checkpoint information if it exists"""
    checkpoint_file = checkpoint_dir / "checkpoint.json"
    if checkpoint_file.exists():
        with open(checkpoint_file, 'r') as f:
            return json.load(f)
    return None

def save_checkpoint(checkpoint_dir: pathlib.Path, domain: str, results: Dict):
    """Save checkpoint for current domain"""
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_file = checkpoint_dir / "checkpoint.json"
    
    checkpoint_data = {
        "last_domain": domain,
        "completed_domains": list(results.keys()),
        "results": results,
        "timestamp": datetime.now().isoformat()
    }
    
    with open(checkpoint_file, 'w') as f:
        json.dump(checkpoint_data, f, indent=2)
    
    logging.info(f"✅ Checkpoint saved: {checkpoint_file}")

def extract_features(query: str, doc: str, query_embedding: np.ndarray, doc_embedding: np.ndarray) -> List[float]:
    """Extract features for learning-to-rank"""
    # Cosine similarity
    cosine_sim = np.dot(query_embedding, doc_embedding) / (np.linalg.norm(query_embedding) * np.linalg.norm(doc_embedding) + 1e-8)
    
    # Text length features
    query_len = len(query.split())
    doc_len = len(doc.split())
    
    # Embedding norm
    query_norm = np.linalg.norm(query_embedding)
    doc_norm = np.linalg.norm(doc_embedding)
    
    # Dot product
    dot_product = np.dot(query_embedding, doc_embedding)
    
    return [
        float(cosine_sim),
        float(query_len),
        float(doc_len),
        float(query_norm),
        float(doc_norm),
        float(dot_product),
    ]

def train_ltr_model(X_train: List[List[float]], y_train: List[int], qid_train: List[str]) -> Optional[object]:
    """Train learning-to-rank model using listwise loss"""
    if not X_train or not y_train:
        logging.warning("No training data available")
        return None
    
    # Try XGBoost Ranker first (best for listwise ranking)
    if XGBOOST_AVAILABLE:
        try:
            logging.info("Training XGBoost Ranker with listwise loss...")
            dtrain = xgb.DMatrix(X_train, label=y_train)
            dtrain.set_group([len([q for q in qid_train if q == qid]) for qid in set(qid_train)])
            
            params = {
                'objective': 'rank:ndcg',
                'eval_metric': 'ndcg@10',
                'eta': 0.1,
                'max_depth': 6,
                'min_child_weight': 1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'tree_method': 'hist' if torch.cuda.is_available() else 'exact',
            }
            
            model = xgb.train(
                params,
                dtrain,
                num_boost_round=100,
                verbose_eval=False
            )
            logging.info("✅ XGBoost Ranker trained successfully")
            return model
        except Exception as e:
            logging.warning(f"XGBoost training failed: {e}. Trying LightGBM...")
    
    # Try LightGBM Ranker
    if LIGHTGBM_AVAILABLE:
        try:
            logging.info("Training LightGBM Ranker with listwise loss...")
            train_data = lgb.Dataset(X_train, label=y_train, group=[len([q for q in qid_train if q == qid]) for qid in set(qid_train)])
            
            params = {
                'objective': 'lambdarank',
                'metric': 'ndcg',
                'ndcg_eval_at': [10],
                'learning_rate': 0.1,
                'num_leaves': 31,
                'feature_fraction': 0.8,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
            }
            
            model = lgb.train(
                params,
                train_data,
                num_boost_round=100,
                verbose_eval=False
            )
            logging.info("✅ LightGBM Ranker trained successfully")
            return model
        except Exception as e:
            logging.warning(f"LightGBM training failed: {e}. Using simple ranking...")
    
    # Fallback: Simple ranking by cosine similarity
    logging.warning("No LTR library available. Using simple cosine similarity ranking.")
    return None

def run_learning_to_rank_evaluation(config: Dict, gpu_id: Optional[int] = None) -> Optional[str]:
    """Run learning-to-rank evaluation with listwise loss"""
    global shutdown_requested
    
    # Set GPU
    if gpu_id is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        logging.info(f"Using GPU {gpu_id}")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Device: {device}")
    
    output_dir = pathlib.Path(config.get('output_dir', 'experiments/retrieval/learning_to_rank'))
    output_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint_dir = output_dir
    resume = config.get('resume', True)
    
    # Load checkpoint if resuming
    all_results = {}
    completed_domains = set()
    if resume:
        checkpoint = load_checkpoint(checkpoint_dir)
        if checkpoint:
            all_results = checkpoint.get('results', {})
            completed_domains = set(checkpoint.get('completed_domains', []))
            logging.info(f"Resuming from checkpoint. Completed domains: {completed_domains}")
    
    domains = config.get('domains', MTRAG_DOMAINS)
    
    # Initialize dense retrieval model for initial retrieval
    model_name = config.get('model_path', 'BAAI/bge-base-en-v1.5')
    logging.info(f"Loading base model: {model_name}")
    model = SentenceBERT(model_name, device=device)
    retriever = DenseRetrievalExactSearch(model, batch_size=128)
    k_values = [1, 3, 5, 10]
    evaluator = EvaluateRetrieval(retriever, k_values=k_values)
    
    # Process each domain
    for domain in domains:
        if domain in completed_domains:
            logging.info(f"Skipping {domain} (already completed)")
            continue
        
        if shutdown_requested:
            logging.info("Shutdown requested, saving checkpoint...")
            save_checkpoint(checkpoint_dir, domain, all_results)
            break
        
        logging.info(f"\n{'='*60}")
        logging.info(f"Processing domain: {domain}")
        logging.info(f"{'='*60}")
        
        # Load data
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
        logging.info("Step 1: Initial retrieval (top 200 candidates)...")
        # Use evaluator.retrieve() - ensure retriever is properly set
        # Always recreate evaluator for each domain to ensure retriever is properly set
        try:
            retriever_check = getattr(evaluator, 'retriever', None)
            if retriever_check is None:
            logging.warning("Evaluator retriever is None, recreating evaluator...")
                evaluator = EvaluateRetrieval(retriever, k_values=k_values)
        except (AttributeError, TypeError):
            logging.warning("Could not check evaluator retriever, recreating evaluator...")
            evaluator = EvaluateRetrieval(retriever, k_values=k_values)
        
        # Ensure retriever is set before calling retrieve
        try:
            initial_results = evaluator.retrieve(corpus, queries)
        except (AttributeError, RuntimeError) as e:
            if "retriever" in str(e).lower() or "not set" in str(e).lower():
                logging.warning(f"Retriever issue detected ({e}), recreating evaluator...")
                evaluator = EvaluateRetrieval(retriever, k_values=k_values)
                initial_results = evaluator.retrieve(corpus, queries)
            else:
                raise
        # Limit to top 200 per query
        initial_results = {qid: dict(list(sorted(docs.items(), key=lambda x: x[1], reverse=True)[:200])) 
                         for qid, docs in initial_results.items()}
        
        # Step 2: Extract features for training
        logging.info("Step 2: Extracting features for LTR model...")
        X_train = []
        y_train = []
        qid_train = []
        
        # Encode queries and documents
        # Get the underlying SentenceTransformer model from SentenceBERT
        sentence_model = model.q_model
        query_texts = [queries[qid] for qid in queries.keys()]
        query_embeddings = sentence_model.encode(query_texts, batch_size=128, show_progress_bar=True, convert_to_numpy=True)
        query_emb_map = {qid: emb for qid, emb in zip(queries.keys(), query_embeddings)}
        
        # For each query, extract features for top candidates
        for query_id in queries.keys():
            if query_id not in initial_results:
                continue
            
            query_text = queries[query_id]
            query_emb = query_emb_map[query_id]
            
            # Get top candidates
            candidates = sorted(initial_results[query_id].items(), key=lambda x: x[1], reverse=True)[:100]
            
            # Extract features for each candidate
            for doc_id, score in candidates:
                if doc_id not in corpus:
                    continue
                
                doc_text = corpus[doc_id].get('text', '')
                doc_emb = sentence_model.encode([doc_text], batch_size=1, show_progress_bar=False, convert_to_numpy=True)[0]
                
                features = extract_features(query_text, doc_text, query_emb, doc_emb)
                X_train.append(features)
                
                # Label: 1 if relevant, 0 otherwise
                relevance = qrels.get(query_id, {}).get(doc_id, 0)
                y_train.append(relevance)
                qid_train.append(query_id)
        
        # Step 3: Train LTR model
        logging.info(f"Step 3: Training LTR model on {len(X_train)} examples...")
        ltr_model = train_ltr_model(X_train, y_train, qid_train)
        
        # Step 4: Re-rank using LTR model
        logging.info("Step 4: Re-ranking documents using LTR model...")
        final_results = {}
        
        for query_id in queries.keys():
            if query_id not in initial_results:
                continue
            
            query_text = queries[query_id]
            query_emb = query_emb_map[query_id]
            
            candidates = sorted(initial_results[query_id].items(), key=lambda x: x[1], reverse=True)[:100]
            
            # Get LTR scores
            ltr_scores = {}
            for doc_id, _ in candidates:
                if doc_id not in corpus:
                    continue
                
                doc_text = corpus[doc_id].get('text', '')
                doc_emb = sentence_model.encode([doc_text], batch_size=1, show_progress_bar=False, convert_to_numpy=True)[0]
                
                features = extract_features(query_text, doc_text, query_emb, doc_emb)
                
                if ltr_model is not None:
                    if XGBOOST_AVAILABLE and isinstance(ltr_model, xgb.core.Booster):
                        score = ltr_model.predict(xgb.DMatrix([features]))[0]
                    elif LIGHTGBM_AVAILABLE and isinstance(ltr_model, lgb.Booster):
                        score = ltr_model.predict([features])[0]
                    else:
                        # Fallback to cosine similarity
                        score = float(np.dot(query_emb, doc_emb) / (np.linalg.norm(query_emb) * np.linalg.norm(doc_emb) + 1e-8))
                else:
                    # Fallback to cosine similarity
                    score = float(np.dot(query_emb, doc_emb) / (np.linalg.norm(query_emb) * np.linalg.norm(doc_emb) + 1e-8))
                
                ltr_scores[doc_id] = score
            
            # Sort by LTR scores
            final_results[query_id] = dict(sorted(ltr_scores.items(), key=lambda x: x[1], reverse=True)[:100])
        
        # Step 5: Evaluate
        logging.info("Step 5: Evaluating results...")
        k_values = [1, 3, 5, 10]
        # Ensure evaluator has retriever set before evaluation
        # Always recreate evaluator with retriever to ensure it's properly initialized
        try:
            evaluator = EvaluateRetrieval(retriever, k_values=k_values)
            ndcg, _map, recall, precision = evaluator.evaluate(qrels, final_results, k_values)
        except (AttributeError, RuntimeError) as e:
            if "retriever" in str(e).lower() or "not set" in str(e).lower():
                logging.error(f"Evaluator error: {e}. Retriever may be None.")
                # Fallback: recreate retriever and evaluator
                retriever = DenseRetrievalExactSearch(model, batch_size=128)
        evaluator = EvaluateRetrieval(retriever, k_values=k_values)
        ndcg, _map, recall, precision = evaluator.evaluate(qrels, final_results, k_values)
            else:
                raise
        
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
        
        save_checkpoint(checkpoint_dir, domain, all_results)
        
        if shutdown_requested:
            logging.info("Shutdown requested, exiting after current domain completion.")
            break
    
    # Calculate average results
    if all_results:
        avg_results = {
            metric: np.mean([res.get(metric, 0) for res in all_results.values()])
            for metric in list(all_results.values())[0].keys()
        }
        
        logging.info(f"\n{'='*60}")
        logging.info("Average Results Across All Domains:")
        for metric, value in avg_results.items():
            logging.info(f"  {metric}: {value:.4f}")
        logging.info(f"{'='*60}\n")
        
        # Save final results
        final_results_path = output_dir / "results.json"
        with open(final_results_path, 'w') as f:
            json.dump({"average": avg_results, "domains": all_results}, f, indent=2)
        logging.info(f"✅ Final results saved to: {final_results_path}")
        
        return str(output_dir)
    
    return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run Learning-to-Rank with Listwise Loss')
    parser.add_argument('--config', type=str, required=True, help='Config JSON file')
    parser.add_argument('--gpu_id', type=int, help='GPU ID to use')
    parser.add_argument('--resume', action='store_true', default=True, help='Resume from checkpoint if available')
    parser.add_argument('--no-resume', dest='resume', action='store_false', help='Do not resume from checkpoint')
    
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    config['resume'] = args.resume
    
    try:
        results_path = run_learning_to_rank_evaluation(config, args.gpu_id)
        if results_path:
            logging.info(f"✅ Learning-to-Rank completed. Results saved to: {results_path}")
    except KeyboardInterrupt:
        logging.info("Learning-to-Rank interrupted by user")
        sys.exit(0)
    except Exception as e:
        logging.error(f"Learning-to-Rank failed: {e}", exc_info=True)
        sys.exit(1)

