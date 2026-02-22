import pathlib
import logging
import random
import numpy as np
import argparse
import json
from datetime import datetime
from beir.datasets.data_loader import GenericDataLoader
from sentence_transformers import SentenceTransformer, InputExample, losses
from sentence_transformers.evaluation import InformationRetrievalEvaluator
from torch.utils.data import DataLoader
import torch

# --- Configuration ---
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO)

def set_seed(seed=42):
    """Sets the seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logging.info(f"🔒 Random seed set to {seed} for reproducibility.")

# Default configuration
BASE_MODEL_NAME = "BAAI/bge-base-en-v1.5"
MTRAG_DOMAINS = ["clapnq", "fiqa", "govt", "cloud"]

def load_mtrag_examples(domain, split="train", use_data_splits=True, use_labels=False):
    """Loads training examples from a domain."""
    logging.info(f"Loading MTRAG domain: {domain} (split: {split})...")
    data_root = pathlib.Path(".")
    
    # Try data_splits first, fall back to original location
    if use_data_splits:
        corpus_file = data_root / "corpora" / "passage_level" / f"{domain}.jsonl"
        query_file = data_root / "data_splits" / "retrieval_tasks" / domain / split / f"{domain}_questions.jsonl"
        qrels_file = data_root / "data_splits" / "retrieval_tasks" / domain / split / "qrels" / "dev.tsv"
    else:
        corpus_file = data_root / "corpora" / "passage_level" / f"{domain}.jsonl"
        query_file = data_root / "human" / "retrieval_tasks" / domain / f"{domain}_questions.jsonl"
        qrels_file = data_root / "human" / "retrieval_tasks" / domain / "qrels" / "dev.tsv"
    
    # Fallback if split files don't exist
    if not query_file.exists() and use_data_splits:
        logging.warning(f"Split file not found: {query_file}, using original location")
        query_file = data_root / "human" / "retrieval_tasks" / domain / f"{domain}_questions.jsonl"
        qrels_file = data_root / "human" / "retrieval_tasks" / domain / "qrels" / "dev.tsv"
    
    try:
        corpus, queries, qrels = GenericDataLoader(
            corpus_file=str(corpus_file),
            query_file=str(query_file),
            qrels_file=str(qrels_file)
        ).load_custom()
    except Exception as e:
        logging.error(f"Error loading {domain}: {e}. Skipping.")
        return [], {}, {}, {}
    
    domain_examples = []
    for query_id, doc_infos in qrels.items():
        query_text = queries.get(query_id)
        if not query_text:
            continue
            
        for doc_id, score in doc_infos.items():
            if score > 0:
                doc = corpus.get(doc_id)
                if doc:
                    # Format passage with title
                    title = doc.get("title", "")
                    text = doc.get("text", "")
                    doc_text = f"{title} {text}".strip() if title else text
                    # For CosineSimilarityLoss, we need labels (similarity scores)
                    # For positive pairs, use label=1.0 (maximum similarity)
                    label = 1.0 if use_labels else None
                    domain_examples.append(InputExample(texts=[query_text, doc_text], label=label))
    
    logging.info(f"Loaded {len(domain_examples)} examples from {domain}")
    return domain_examples, corpus, queries, qrels

def augment_query(query_text):
    """Simple query augmentation."""
    variations = [query_text]  # Original
    
    # Lowercase variation
    if query_text != query_text.lower():
        variations.append(query_text.lower())
    
    # Remove question mark
    if query_text.endswith('?'):
        variations.append(query_text[:-1].strip())
    
    return variations

def augment_passage(title, text):
    """Simple passage augmentation."""
    variations = []
    
    # With title
    if title:
        variations.append(f"{title} {text}".strip())
    
    # Without title
    variations.append(text)
    
    return variations

def load_training_data(domains, split="train", use_augmentation=False, use_data_splits=True, use_labels=False):
    """Load all training data."""
    all_examples = []
    
    for domain in domains:
        examples, _, _, _ = load_mtrag_examples(domain, split=split, use_data_splits=use_data_splits, use_labels=use_labels)
        
        if use_augmentation:
            # Simple augmentation: duplicate examples with variations
            augmented = []
            for ex in examples:
                augmented.append(ex)  # Keep original
                # Add one augmented version
                query_variations = augment_query(ex.texts[0])
                if len(query_variations) > 1:
                    augmented.append(InputExample(texts=[query_variations[1], ex.texts[1]]))
            examples = augmented
        
        all_examples.extend(examples)
    
    return all_examples

def load_validation_data(domains, use_data_splits=True):
    """Load validation data for evaluation during training."""
    val_corpus = {}
    val_queries = {}
    val_qrels = {}
    
    for domain in domains:
        _, corpus, queries, qrels = load_mtrag_examples(domain, split="val", use_data_splits=use_data_splits)
        if corpus:
            # Prefix domain to avoid conflicts
            for k, v in corpus.items():
                val_corpus[f"{domain}_{k}"] = v
            for k, v in queries.items():
                val_queries[f"{domain}_{k}"] = v
            for k, v in qrels.items():
                val_qrels[f"{domain}_{k}"] = {f"{domain}_{dk}": dv for dk, dv in v.items()}
    
    return val_corpus, val_queries, val_qrels

def run_training(config):
    """Run training with given configuration."""
    set_seed(config.get('seed', 42))
    
    logging.info("="*60)
    logging.info(f"Starting training with config: {config['experiment_name']}")
    logging.info("="*60)
    
    # Check GPU
    if not torch.cuda.is_available():
        logging.warning("CUDA not available. Training will be slow.")
    
    # Determine loss function first to know if we need labels
    loss_type = config.get('loss_function', 'MultipleNegativesRankingLoss')
    use_labels = (loss_type == 'CosineSimilarityLoss')
    
    # Load training data
    train_examples = load_training_data(
        MTRAG_DOMAINS,
        split="train",
        use_augmentation=config.get('use_augmentation', False),
        use_data_splits=config.get('use_data_splits', True),
        use_labels=use_labels
    )
    
    logging.info(f"Total training examples: {len(train_examples)}")
    
    if not train_examples:
        logging.error("No training examples found. Exiting.")
        return None
    
    # Load model
    model = SentenceTransformer(config.get('base_model', BASE_MODEL_NAME))
    
    # Create dataloader
    train_dataloader = DataLoader(
        train_examples,
        shuffle=True,
        batch_size=config.get('batch_size', 32)
    )
    
    # Setup loss function
    if loss_type == 'MultipleNegativesRankingLoss':
        train_loss = losses.MultipleNegativesRankingLoss(model=model)
    elif loss_type == 'CosineSimilarityLoss':
        train_loss = losses.CosineSimilarityLoss(model=model)
    else:
        train_loss = losses.MultipleNegativesRankingLoss(model=model)
    
    # Setup validation evaluator if requested
    evaluator = None
    if config.get('use_validation', True):
        try:
            val_corpus, val_queries, val_qrels = load_validation_data(
                MTRAG_DOMAINS,
                use_data_splits=config.get('use_data_splits', True)
            )
            if val_corpus and val_queries and val_qrels:
                evaluator = InformationRetrievalEvaluator(
                    queries=val_queries,
                    corpus=val_corpus,
                    relevant_docs=val_qrels,
                    show_progress_bar=True,
                    name="validation"
                )
                logging.info(f"Validation evaluator created with {len(val_queries)} queries")
        except Exception as e:
            logging.warning(f"Could not create validation evaluator: {e}")
    
    # Training parameters
    output_path = config.get('output_path', f"./models/{config['experiment_name']}")
    pathlib.Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Start training
    logging.info(f"Training config: Epochs={config.get('epochs', 3)}, Batch Size={config.get('batch_size', 32)}")
    
    # Prepare checkpoint path
    checkpoint_path = None
    if config.get('save_checkpoints', True):
        checkpoint_path = f"{output_path}-checkpoints"
        # Ensure checkpoint_path is a valid string (not None)
        if checkpoint_path is None or checkpoint_path == "":
            checkpoint_path = None
    
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=config.get('epochs', 3),
        warmup_steps=config.get('warmup_steps', 100),
        optimizer_params={'lr': config.get('learning_rate', 2e-5)},
        evaluator=evaluator,
        evaluation_steps=config.get('evaluation_steps', 500) if evaluator else None,
        output_path=output_path,
        save_best_model=config.get('save_best_model', True) if evaluator else False,
        show_progress_bar=True,
        checkpoint_save_steps=config.get('checkpoint_steps', 1000) if checkpoint_path else None,
        checkpoint_path=checkpoint_path
    )
    
    logging.info(f"Training complete. Model saved to: {output_path}")
    return output_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help='Path to config JSON file')
    parser.add_argument('--experiment_name', type=str, default='experiment_1')
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=2e-5)
    parser.add_argument('--loss_function', type=str, default='MultipleNegativesRankingLoss')
    parser.add_argument('--use_augmentation', action='store_true')
    parser.add_argument('--use_validation', action='store_true', default=True)
    parser.add_argument('--use_data_splits', action='store_true', default=True)
    
    args = parser.parse_args()
    
    if args.config:
        with open(args.config, 'r') as f:
            config = json.load(f)
    else:
        config = {
            'experiment_name': args.experiment_name,
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'learning_rate': args.learning_rate,
            'loss_function': args.loss_function,
            'use_augmentation': args.use_augmentation,
            'use_validation': args.use_validation,
            'use_data_splits': args.use_data_splits,
            'warmup_steps': 100,
            'evaluation_steps': 500,
            'save_best_model': True,
            'save_checkpoints': True,
            'checkpoint_steps': 1000
        }
    
    run_training(config)

