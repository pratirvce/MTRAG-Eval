#!/usr/bin/env python3
"""
Script to split MT-RAG retrieval data into train, validation, and test sets.

This script splits the retrieval tasks (queries and qrels) for each domain
into train/val/test sets while maintaining the BEIR format structure.
"""

import json
import argparse
import pathlib
import random
from collections import defaultdict
from typing import Dict, List, Set, Tuple
import pandas as pd
import logging

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO
)

# Default split ratios: train, val, test
DEFAULT_SPLIT_RATIOS = (0.7, 0.15, 0.15)
MTRAG_DOMAINS = ["clapnq", "fiqa", "govt", "cloud"]


def load_queries(query_file: pathlib.Path) -> Dict[str, str]:
    """Load queries from JSONL file."""
    queries = {}
    if not query_file.exists():
        logging.warning(f"Query file not found: {query_file}")
        return queries
    
    with open(query_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                queries[item['_id']] = item['text']
    return queries


def load_qrels(qrels_file: pathlib.Path) -> Dict[str, Dict[str, int]]:
    """Load qrels from TSV file."""
    qrels = defaultdict(dict)
    if not qrels_file.exists():
        logging.warning(f"Qrels file not found: {qrels_file}")
        return qrels
    
    with open(qrels_file, 'r', encoding='utf-8') as f:
        next(f)  # Skip header
        for line in f:
            if line.strip():
                parts = line.strip().split('\t')
                if len(parts) >= 3:
                    query_id, corpus_id, score = parts[0], parts[1], int(parts[2])
                    qrels[query_id][corpus_id] = score
    return dict(qrels)


def split_query_ids(query_ids: List[str], train_ratio: float, val_ratio: float, 
                   test_ratio: float, seed: int = 42) -> Tuple[List[str], List[str], List[str]]:
    """
    Split query IDs into train, validation, and test sets.
    
    Args:
        query_ids: List of query IDs to split
        train_ratio: Ratio for training set
        val_ratio: Ratio for validation set
        test_ratio: Ratio for test set (should be 1 - train_ratio - val_ratio)
        seed: Random seed for reproducibility
    
    Returns:
        Tuple of (train_ids, val_ids, test_ids)
    """
    # Validate ratios
    total_ratio = train_ratio + val_ratio + test_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        raise ValueError(f"Split ratios must sum to 1.0, got {total_ratio}")
    
    # Shuffle with seed for reproducibility
    random.seed(seed)
    shuffled_ids = query_ids.copy()
    random.shuffle(shuffled_ids)
    
    n_total = len(shuffled_ids)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    # Test gets the remainder to ensure all queries are used
    
    train_ids = shuffled_ids[:n_train]
    val_ids = shuffled_ids[n_train:n_train + n_val]
    test_ids = shuffled_ids[n_train + n_val:]
    
    return train_ids, val_ids, test_ids


def write_queries(queries: Dict[str, str], query_ids: List[str], output_file: pathlib.Path):
    """Write queries to JSONL file in BEIR format."""
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for query_id in query_ids:
            if query_id in queries:
                item = {
                    '_id': query_id,
                    'text': queries[query_id]
                }
                f.write(json.dumps(item, ensure_ascii=False) + '\n')


def write_qrels(qrels: Dict[str, Dict[str, int]], query_ids: List[str], output_file: pathlib.Path):
    """Write qrels to TSV file in BEIR format."""
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        # Write header
        f.write('query-id\tcorpus-id\tscore\n')
        
        # Write qrels for the specified query IDs
        for query_id in query_ids:
            if query_id in qrels:
                for corpus_id, score in qrels[query_id].items():
                    f.write(f'{query_id}\t{corpus_id}\t{score}\n')


def split_domain_data(domain: str, data_root: pathlib.Path, output_root: pathlib.Path,
                     train_ratio: float, val_ratio: float, test_ratio: float, seed: int):
    """Split data for a single domain."""
    logging.info(f"\n{'='*60}")
    logging.info(f"Splitting data for domain: {domain}")
    logging.info(f"{'='*60}")
    
    # Define input paths
    domain_dir = data_root / "human" / "retrieval_tasks" / domain
    query_file = domain_dir / f"{domain}_questions.jsonl"
    qrels_file = domain_dir / "qrels" / "dev.tsv"
    
    # Load data
    logging.info(f"Loading queries from: {query_file}")
    queries = load_queries(query_file)
    logging.info(f"Loaded {len(queries)} queries")
    
    logging.info(f"Loading qrels from: {qrels_file}")
    qrels = load_qrels(qrels_file)
    logging.info(f"Loaded qrels for {len(qrels)} queries")
    
    # Get query IDs that have both queries and qrels
    query_ids = list(set(queries.keys()) & set(qrels.keys()))
    logging.info(f"Found {len(query_ids)} queries with both query text and qrels")
    
    if len(query_ids) == 0:
        logging.warning(f"No valid query IDs found for domain {domain}. Skipping.")
        return
    
    # Split query IDs
    train_ids, val_ids, test_ids = split_query_ids(
        query_ids, train_ratio, val_ratio, test_ratio, seed
    )
    
    logging.info(f"Split sizes: Train={len(train_ids)}, Val={len(val_ids)}, Test={len(test_ids)}")
    
    # Write splits
    splits = {
        'train': train_ids,
        'val': val_ids,
        'test': test_ids
    }
    
    for split_name, split_ids in splits.items():
        output_dir = output_root / "retrieval_tasks" / domain / split_name
        query_output = output_dir / f"{domain}_questions.jsonl"
        qrels_output = output_dir / "qrels" / "dev.tsv"
        
        logging.info(f"\nWriting {split_name} split:")
        logging.info(f"  Queries: {query_output}")
        logging.info(f"  Qrels: {qrels_output}")
        
        write_queries(queries, split_ids, query_output)
        write_qrels(qrels, split_ids, qrels_output)
        
        logging.info(f"  Written {len(split_ids)} queries and their qrels")


def main():
    parser = argparse.ArgumentParser(
        description='Split MT-RAG retrieval data into train/val/test sets'
    )
    parser.add_argument(
        '--data_root',
        type=str,
        default='.',
        help='Root directory containing the data (default: current directory)'
    )
    parser.add_argument(
        '--output_root',
        type=str,
        default='./data_splits',
        help='Root directory for output splits (default: ./data_splits)'
    )
    parser.add_argument(
        '--train_ratio',
        type=float,
        default=0.7,
        help='Ratio for training set (default: 0.7)'
    )
    parser.add_argument(
        '--val_ratio',
        type=float,
        default=0.15,
        help='Ratio for validation set (default: 0.15)'
    )
    parser.add_argument(
        '--test_ratio',
        type=float,
        default=0.15,
        help='Ratio for test set (default: 0.15)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    parser.add_argument(
        '--domains',
        type=str,
        nargs='+',
        default=MTRAG_DOMAINS,
        help=f'Domains to process (default: {MTRAG_DOMAINS})'
    )
    
    args = parser.parse_args()
    
    # Validate ratios
    total_ratio = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        logging.error(f"Split ratios must sum to 1.0, got {total_ratio}")
        return
    
    data_root = pathlib.Path(args.data_root)
    output_root = pathlib.Path(args.output_root)
    
    logging.info("="*60)
    logging.info("MT-RAG Data Splitting Script")
    logging.info("="*60)
    logging.info(f"Data root: {data_root}")
    logging.info(f"Output root: {output_root}")
    logging.info(f"Split ratios: Train={args.train_ratio}, Val={args.val_ratio}, Test={args.test_ratio}")
    logging.info(f"Random seed: {args.seed}")
    logging.info(f"Domains: {args.domains}")
    
    # Process each domain
    for domain in args.domains:
        try:
            split_domain_data(
                domain=domain,
                data_root=data_root,
                output_root=output_root,
                train_ratio=args.train_ratio,
                val_ratio=args.val_ratio,
                test_ratio=args.test_ratio,
                seed=args.seed
            )
        except Exception as e:
            logging.error(f"Error processing domain {domain}: {e}", exc_info=True)
    
    logging.info("\n" + "="*60)
    logging.info("Data splitting complete!")
    logging.info("="*60)
    logging.info(f"\nOutput structure:")
    logging.info(f"{output_root}/")
    for domain in args.domains:
        for split in ['train', 'val', 'test']:
            logging.info(f"  retrieval_tasks/{domain}/{split}/")
            logging.info(f"    {domain}_questions.jsonl")
            logging.info(f"    qrels/dev.tsv")


if __name__ == "__main__":
    main()

