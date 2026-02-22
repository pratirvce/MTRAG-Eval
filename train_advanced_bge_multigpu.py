"""
Multi-GPU BGE Training with Hard Negative Mining
Optimized for parallel processing across multiple GPUs
"""

import pathlib
import logging
import random
import numpy as np
import argparse
import json
from datetime import datetime
from typing import List, Dict, Tuple, Optional
from beir.datasets.data_loader import GenericDataLoader
from sentence_transformers import SentenceTransformer, InputExample, losses
from sentence_transformers.evaluation import InformationRetrievalEvaluator
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from sklearn.metrics.pairwise import cosine_similarity
from torch.nn.parallel import DataParallel

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

def load_mtrag_examples(domain, split="train", use_data_splits=True):
    """Loads training examples from a domain."""
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

def mine_hard_negatives_parallel(
    model: SentenceTransformer,
    query_text: str,
    positive_doc_ids: List[str],
    corpus: Dict,
    corpus_ids: List[str],
    top_k: int = 5,
    batch_size: int = 512,
    device: str = "cuda"
) -> List[str]:
    """
    Mine hard negatives with optimized batch processing for multi-GPU.
    """
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
    
    # Encode passages in larger batches for multi-GPU efficiency
    passage_embs = model.encode(
        all_passage_texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        show_progress_bar=True
    )
    
    similarities = cosine_similarity(query_emb, passage_embs)[0]
    top_indices = np.argsort(similarities)[::-1][:top_k]
    hard_negatives = [all_passage_ids[idx] for idx in top_indices]
    
    return hard_negatives

def create_training_examples_with_hard_negatives(
    model: SentenceTransformer,
    corpus: Dict,
    queries: Dict,
    qrels: Dict,
    num_hard_negatives: int = 3,
    use_hard_negatives: bool = True,
    device: str = "cuda",
    batch_size: int = 512
) -> List[InputExample]:
    """Create training examples with hard negatives (optimized for multi-GPU)."""
    examples = []
    corpus_ids = list(corpus.keys())
    
    for query_id, doc_infos in qrels.items():
        query_text = queries.get(query_id)
        if not query_text:
            continue
        
        positive_doc_ids = [doc_id for doc_id, score in doc_infos.items() if score > 0]
        
        # Create positive examples
        for doc_id in positive_doc_ids:
            doc = corpus.get(doc_id)
            if doc:
                title = doc.get("title", "")
                text = doc.get("text", "")
                doc_text = f"{title} {text}".strip() if title else text
                examples.append(InputExample(texts=[query_text, doc_text], label=1.0))
        
        # Mine hard negatives
        if use_hard_negatives and len(positive_doc_ids) > 0:
            hard_neg_ids = mine_hard_negatives_parallel(
                model, query_text, positive_doc_ids, corpus, corpus_ids,
                top_k=num_hard_negatives, batch_size=batch_size, device=device
            )
            
            for neg_doc_id in hard_neg_ids:
                neg_doc = corpus.get(neg_doc_id)
                if neg_doc:
                    title = neg_doc.get("title", "")
                    text = neg_doc.get("text", "")
                    doc_text = f"{title} {text}".strip() if title else text
                    examples.append(InputExample(texts=[query_text, doc_text], label=0.0))
    
    return examples

def load_training_data_advanced(
    model: SentenceTransformer,
    domains: List[str],
    split: str = "train",
    use_hard_negatives: bool = True,
    num_hard_negatives: int = 3,
    use_data_splits: bool = True,
    device: str = "cuda",
    batch_size: int = 512
) -> List[InputExample]:
    """Load training data with hard negative mining (multi-GPU optimized)."""
    all_examples = []
    
    for domain in domains:
        corpus, queries, qrels = load_mtrag_examples(domain, split=split, use_data_splits=use_data_splits)
        
        if not corpus or not queries or not qrels:
            continue
        
        logging.info(f"Mining hard negatives for {domain}...")
        domain_examples = create_training_examples_with_hard_negatives(
            model, corpus, queries, qrels,
            num_hard_negatives=num_hard_negatives,
            use_hard_negatives=use_hard_negatives,
            device=device,
            batch_size=batch_size
        )
        
        all_examples.extend(domain_examples)
        logging.info(f"Created {len(domain_examples)} examples for {domain}")
    
    return all_examples

def run_multigpu_training(config: Dict):
    """Run training with multi-GPU support."""
    set_seed(config.get('seed', 42))
    
    logging.info("="*60)
    logging.info(f"🚀 Starting MULTI-GPU ADVANCED training: {config['experiment_name']}")
    logging.info("="*60)
    
    # Multi-GPU setup
    if not torch.cuda.is_available():
        logging.warning("⚠️ CUDA not available.")
        device = "cpu"
        num_gpus = 0
    else:
        gpu_ids = config.get('gpu_ids', None)
        if gpu_ids is not None and isinstance(gpu_ids, list):
            num_gpus = len(gpu_ids)
            device = f"cuda:{gpu_ids[0]}"
            logging.info(f"Using GPUs: {gpu_ids}")
        else:
            num_gpus = torch.cuda.device_count()
            device = "cuda:0"
            logging.info(f"Using all available GPUs: {num_gpus}")
    
    # Load model
    base_model_name = config.get('base_model', BASE_MODEL_NAME)
    model = SentenceTransformer(base_model_name, device=device)
    
    # Wrap model for multi-GPU if available
    if num_gpus > 1 and torch.cuda.is_available():
        if isinstance(gpu_ids, list):
            model._modules['0'].auto_model = DataParallel(
                model._modules['0'].auto_model,
                device_ids=gpu_ids
            )
        else:
            model._modules['0'].auto_model = DataParallel(model._modules['0'].auto_model)
        logging.info(f"Model wrapped for multi-GPU training on {num_gpus} GPUs")
    
    # Load training data
    batch_size_encoding = config.get('encoding_batch_size', 512 * num_gpus if num_gpus > 1 else 512)
    train_examples = load_training_data_advanced(
        model=model,
        domains=MTRAG_DOMAINS,
        split="train",
        use_hard_negatives=config.get('use_hard_negatives', True),
        num_hard_negatives=config.get('num_hard_negatives', 3),
        use_data_splits=config.get('use_data_splits', True),
        device=device,
        batch_size=batch_size_encoding
    )
    
    logging.info(f"Total training examples: {len(train_examples)}")
    
    if not train_examples:
        logging.error("No training examples found.")
        return None
    
    # Create dataloader
    train_dataloader = DataLoader(
        train_examples,
        shuffle=True,
        batch_size=config.get('batch_size', 32) * num_gpus if num_gpus > 1 else config.get('batch_size', 32)
    )
    
    # Loss function
    loss_type = config.get('loss_function', 'CosineSimilarityLoss')
    if loss_type == 'CosineSimilarityLoss':
        train_loss = losses.CosineSimilarityLoss(model=model)
    else:
        train_loss = losses.MultipleNegativesRankingLoss(model=model)
    
    # Training
    output_path = config.get('output_path', f"./models/{config['experiment_name']}")
    pathlib.Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=config.get('epochs', 5),
        warmup_steps=config.get('warmup_steps', 100),
        optimizer_params={'lr': config.get('learning_rate', 2e-5)},
        output_path=output_path,
        show_progress_bar=True
    )
    
    logging.info(f"✅ Training complete. Model saved to: {output_path}")
    return output_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    run_multigpu_training(config)


