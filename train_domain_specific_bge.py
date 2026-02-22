"""
Domain-Specific BGE Fine-tuning
Train separate models for each domain to maximize domain-specific performance
"""

import pathlib
import logging
import random
import numpy as np
import argparse
import json
import os
from datetime import datetime
from typing import List, Dict
from beir.datasets.data_loader import GenericDataLoader
from sentence_transformers import SentenceTransformer, InputExample, losses
from sentence_transformers.evaluation import InformationRetrievalEvaluator
from torch.utils.data import DataLoader
import torch

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
    
    examples = []
    for query_id, doc_infos in qrels.items():
        query_text = queries.get(query_id)
        if not query_text:
            continue
        for doc_id, score in doc_infos.items():
            if score > 0:
                doc = corpus.get(doc_id)
                if doc:
                    title = doc.get("title", "")
                    text = doc.get("text", "")
                    doc_text = f"{title} {text}".strip() if title else text
                    examples.append(InputExample(texts=[query_text, doc_text]))
    
    return examples, corpus, queries, qrels

def train_domain_model(domain, config):
    """Train a model for a specific domain."""
    set_seed(config.get('seed', 42))
    
    logging.info("="*60)
    logging.info(f"Training domain-specific model for: {domain}")
    logging.info("="*60)
    
    # Support GPU device assignment
    if not torch.cuda.is_available():
        device = "cpu"
    else:
        # If CUDA_VISIBLE_DEVICES is set, use device 0 (it will be the selected GPU)
        if 'CUDA_VISIBLE_DEVICES' in os.environ:
            device = "cuda:0"
            logging.info(f"Using GPU (via CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}) for domain {domain}")
        else:
            gpu_id = config.get('gpu_id')
            if gpu_id is not None:
                if gpu_id >= torch.cuda.device_count():
                    logging.warning(f"GPU {gpu_id} not available, using default GPU")
                    device = "cuda"
                else:
                    device = f"cuda:{gpu_id}"
                    logging.info(f"Using GPU {gpu_id} for domain {domain}")
            else:
                device = "cuda"
    
    # Load domain data
    train_examples, _, _, _ = load_domain_examples(
        domain, split="train", use_data_splits=config.get('use_data_splits', True)
    )
    
    if not train_examples:
        logging.error(f"No training examples for {domain}")
        return None
    
    logging.info(f"Loaded {len(train_examples)} examples for {domain}")
    
    # Load model (can start from multi-domain model or base model)
    base_model = config.get('base_model', BASE_MODEL_NAME)
    
    # Check if we should start from a pre-trained multi-domain model
    if config.get('use_pretrained_multi_domain', False):
        pretrained_path = config.get('pretrained_multi_domain_path', './models/phase1_epochs5')
        try:
            model = SentenceTransformer(pretrained_path, device=device)
            logging.info(f"Starting from pre-trained multi-domain model: {pretrained_path}")
        except:
            model = SentenceTransformer(base_model, device=device)
            logging.info(f"Pre-trained model not found, using base model: {base_model}")
    else:
        model = SentenceTransformer(base_model, device=device)
        logging.info(f"Using base model: {base_model}")
    
    # Create dataloader
    train_dataloader = DataLoader(
        train_examples,
        shuffle=True,
        batch_size=config.get('batch_size', 32)
    )
    
    # Loss function
    train_loss = losses.MultipleNegativesRankingLoss(model=model)
    
    # Validation evaluator
    evaluator = None
    if config.get('use_validation', True):
        try:
            val_examples, val_corpus, val_queries, val_qrels = load_domain_examples(
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
    output_path = config.get('output_path', f"./models/domain_specific_{domain}")
    pathlib.Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Train
    logging.info(f"Training {domain} model: Epochs={config.get('epochs', 5)}, Batch Size={config.get('batch_size', 32)}")
    
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=config.get('epochs', 5),
        warmup_steps=config.get('warmup_steps', 50),
        optimizer_params={'lr': config.get('learning_rate', 1e-5)},  # Lower LR for fine-tuning
        evaluator=evaluator,
        evaluation_steps=config.get('evaluation_steps', 200) if evaluator else None,
        output_path=output_path,
        save_best_model=config.get('save_best_model', True) if evaluator else False,
        show_progress_bar=True
    )
    
    logging.info(f"✅ Domain model for {domain} saved to: {output_path}")
    return output_path

def main():
    parser = argparse.ArgumentParser(description='Train domain-specific BGE models')
    parser.add_argument('--domain', type=str, choices=MTRAG_DOMAINS, help='Domain to train (or "all" for all domains)')
    parser.add_argument('--config', type=str, help='Path to config JSON file')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=1e-5)
    parser.add_argument('--use_pretrained_multi_domain', action='store_true', default=False)
    parser.add_argument('--pretrained_multi_domain_path', type=str, default='./models/phase1_epochs5')
    
    args = parser.parse_args()
    
    if args.config:
        with open(args.config, 'r') as f:
            config = json.load(f)
    else:
        config = {
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'learning_rate': args.learning_rate,
            'use_pretrained_multi_domain': args.use_pretrained_multi_domain,
            'pretrained_multi_domain_path': args.pretrained_multi_domain_path,
            'use_validation': True,
            'use_data_splits': True,
            'warmup_steps': 50,
            'evaluation_steps': 200,
            'save_best_model': True,
            'seed': 42
        }
    
    if args.domain == "all" or args.domain is None:
        # Train all domains
        for domain in MTRAG_DOMAINS:
            config['output_path'] = f"./models/domain_specific_{domain}"
            train_domain_model(domain, config)
    else:
        config['output_path'] = f"./models/domain_specific_{args.domain}"
        train_domain_model(args.domain, config)

if __name__ == "__main__":
    main()

