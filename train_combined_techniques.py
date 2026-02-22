"""
Combined Techniques: Domain-Specific + Hard Negatives
Train domain-specific models with hard negative mining
"""

import pathlib
import logging
import random
import numpy as np
import argparse
import json
import os
import sys
from datetime import datetime
from typing import List, Dict
from beir.datasets.data_loader import GenericDataLoader
from sentence_transformers import SentenceTransformer, InputExample, losses
from sentence_transformers.evaluation import InformationRetrievalEvaluator
from torch.utils.data import DataLoader
import torch
from sklearn.metrics.pairwise import cosine_similarity

logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO)

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

BASE_MODEL_NAME = "BAAI/bge-base-en-v1.5"
MTRAG_DOMAINS = ["clapnq", "fiqa", "govt", "cloud"]

def load_domain_examples(domain, split="train", use_data_splits=True):
    """Load examples for a specific domain."""
    data_root = pathlib.Path(".")
    
    if use_data_splits:
        corpus_file = data_root / "corpora" / "passage_level" / f"{domain}.jsonl"
        query_file = data_root / "data_splits" / "retrieval_tasks" / domain / split / f"{domain}_questions.jsonl"
        qrels_file = data_root / "data_splits" / "retrieval_tasks" / domain / split / "qrels" / "dev.tsv"
    else:
        corpus_file = data_root / "corpora" / "passage_level" / f"{domain}.jsonl"
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
        return [], {}, {}, {}
    
    return corpus, queries, qrels

def mine_hard_negatives(model, query_text, positive_doc_ids, corpus, corpus_ids, top_k=5, device="cuda"):
    """Mine hard negatives for a query."""
    query_emb = model.encode([query_text], convert_to_numpy=True, show_progress_bar=False)
    
    all_passage_texts = []
    all_passage_ids = []
    for doc_id in corpus_ids:
        if doc_id not in positive_doc_ids:
            doc = corpus.get(doc_id)
            if doc:
                title = doc.get("title", "")
                text = doc.get("text", "")
                doc_text = f"{title} {text}".strip() if title else text
                all_passage_texts.append(doc_text)
                all_passage_ids.append(doc_id)
    
    if len(all_passage_texts) == 0:
        return []
    
    passage_embs = model.encode(
        all_passage_texts,
        batch_size=32,
        convert_to_numpy=True,
        show_progress_bar=False
    )
    
    similarities = cosine_similarity(query_emb, passage_embs)[0]
    top_indices = np.argsort(similarities)[::-1][:top_k]
    hard_negatives = [all_passage_ids[idx] for idx in top_indices]
    
    return hard_negatives

def create_training_examples_with_hard_negatives(
    model, corpus, queries, qrels, num_hard_negatives=3, device="cuda"
):
    """Create training examples with hard negatives."""
    examples = []
    corpus_ids = list(corpus.keys())
    
    for query_id, doc_infos in qrels.items():
        query_text = queries.get(query_id)
        if not query_text:
            continue
        
        positive_doc_ids = [doc_id for doc_id, score in doc_infos.items() if score > 0]
        
        for doc_id in positive_doc_ids:
            doc = corpus.get(doc_id)
            if doc:
                title = doc.get("title", "")
                text = doc.get("text", "")
                doc_text = f"{title} {text}".strip() if title else text
                examples.append(InputExample(texts=[query_text, doc_text], label=1.0))
        
        if len(positive_doc_ids) > 0:
            hard_neg_ids = mine_hard_negatives(
                model, query_text, positive_doc_ids, corpus, corpus_ids,
                top_k=num_hard_negatives, device=device
            )
            
            for neg_doc_id in hard_neg_ids:
                neg_doc = corpus.get(neg_doc_id)
                if neg_doc:
                    title = neg_doc.get("title", "")
                    text = neg_doc.get("text", "")
                    doc_text = f"{title} {text}".strip() if title else text
                    examples.append(InputExample(texts=[query_text, doc_text], label=0.0))
    
    return examples

def train_combined_model(domain, config):
    """Train a domain-specific model with hard negatives."""
    set_seed(config.get('seed', 42))
    
    logging.info("="*60)
    logging.info(f"Training domain-specific + hard negatives model for: {domain}")
    logging.info("="*60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if 'CUDA_VISIBLE_DEVICES' in os.environ:
        device = "cuda:0"
    
    # Load domain data
    corpus, queries, qrels = load_domain_examples(
        domain, split="train", use_data_splits=config.get('use_data_splits', True)
    )
    
    if not corpus:
        logging.error(f"No data for {domain}")
        return None
    
    # Load base model (can start from multi-domain or base)
    base_model = config.get('base_model', BASE_MODEL_NAME)
    if config.get('use_pretrained_multi_domain', False):
        pretrained_path = config.get('pretrained_multi_domain_path', './models/phase1_epochs5')
        try:
            model = SentenceTransformer(pretrained_path, device=device)
            logging.info(f"Starting from: {pretrained_path}")
        except:
            model = SentenceTransformer(base_model, device=device)
            logging.info(f"Using base model: {base_model}")
    else:
        model = SentenceTransformer(base_model, device=device)
    
    # Create training examples with hard negatives
    logging.info("Mining hard negatives...")
    train_examples = create_training_examples_with_hard_negatives(
        model, corpus, queries, qrels,
        num_hard_negatives=config.get('num_hard_negatives', 3),
        device=device
    )
    
    logging.info(f"Created {len(train_examples)} training examples")
    
    # Create dataloader
    train_dataloader = DataLoader(
        train_examples,
        shuffle=True,
        batch_size=config.get('batch_size', 32)
    )
    
    # Loss function
    loss_type = config.get('loss_function', 'CosineSimilarityLoss')
    if loss_type == 'CosineSimilarityLoss':
        train_loss = losses.CosineSimilarityLoss(model=model)
    elif loss_type == 'TripletLoss':
        from sentence_transformers.losses import TripletLoss, TripletDistanceMetric
        train_loss = TripletLoss(
            model=model,
            distance_metric=TripletDistanceMetric.COSINE,
            triplet_margin=1.0
        )
    else:
        train_loss = losses.MultipleNegativesRankingLoss(model=model)
    
    # Validation evaluator
    evaluator = None
    if config.get('use_validation', True):
        try:
            val_corpus, val_queries, val_qrels = load_domain_examples(
                domain, split="val", use_data_splits=config.get('use_data_splits', True)
            )
            if val_corpus and val_queries and val_qrels:
                evaluator = InformationRetrievalEvaluator(
                    queries=val_queries,
                    corpus=val_corpus,
                    relevant_docs=val_qrels,
                    show_progress_bar=True,
                    name=f"{domain}_validation"
                )
        except Exception as e:
            logging.warning(f"Could not create validation evaluator: {e}")
    
    # Output path
    output_path = config.get('output_path', f"./models/domain_specific_{domain}_hard_negatives")
    pathlib.Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Check for resume checkpoint
    resume_from_checkpoint = config.get('resume_from_checkpoint')
    if resume_from_checkpoint and pathlib.Path(resume_from_checkpoint).exists():
        logging.info(f"Resuming from checkpoint: {resume_from_checkpoint}")
        model = SentenceTransformer(str(resume_from_checkpoint), device=device)
    
    # Train
    logging.info(f"Training: Epochs={config.get('epochs', 5)}, Batch Size={config.get('batch_size', 32)}")
    
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=config.get('epochs', 5),
        warmup_steps=config.get('warmup_steps', 100),
        optimizer_params={'lr': config.get('learning_rate', 2e-5)},
        evaluator=evaluator,
        evaluation_steps=config.get('evaluation_steps', 200) if evaluator else None,
        output_path=output_path,
        save_best_model=config.get('save_best_model', True) if evaluator else False,
        show_progress_bar=True,
        checkpoint_save_steps=config.get('checkpoint_steps', 1000),
        checkpoint_path=f"{output_path}-checkpoints" if config.get('save_checkpoints', True) else None
    )
    
    logging.info(f"✅ Model saved to: {output_path}")
    return output_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Domain-Specific + Hard Negatives')
    parser.add_argument('--config', type=str, required=True, help='Config JSON file')
    parser.add_argument('--gpu_id', type=int, help='GPU ID')
    parser.add_argument('--resume_from_checkpoint', type=str, help='Resume from checkpoint')
    
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    if args.gpu_id is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    
    if args.resume_from_checkpoint:
        config['resume_from_checkpoint'] = args.resume_from_checkpoint
    
    domain = config.get('domain')
    if not domain:
        logging.error("Domain not specified in config")
        sys.exit(1)
    
    train_combined_model(domain, config)

