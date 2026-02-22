"""
Evaluate Advanced Retrieval Models
Supports both multi-domain and domain-specific models
"""

import pathlib
import argparse
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.models import SentenceBERT
from beir.retrieval.search.dense import DenseRetrievalExactSearch
import torch
import numpy as np
import json

def evaluate_model(model_path, domains=["clapnq", "fiqa", "govt", "cloud"], use_data_splits=True):
    """Evaluate a model on all domains."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading model: {model_path} (device: {device})...")
    
    try:
        model = SentenceBERT(model_path, device=device)
    except Exception as e:
        print(f"Error loading model: {e}")
        return None
    
    retriever = DenseRetrievalExactSearch(model, batch_size=128)
    k_values = [1, 3, 5, 10]
    evaluator = EvaluateRetrieval(retriever, k_values=k_values)
    
    data_root = pathlib.Path(".")
    all_results = {}
    
    for domain in domains:
        print(f"\n--- Evaluating on domain: {domain} ---")
        
        # Load test data
        corpus_file = data_root / "corpora" / "passage_level" / f"{domain}.jsonl"
        
        if use_data_splits:
            query_file = data_root / "data_splits" / "retrieval_tasks" / domain / "test" / f"{domain}_questions.jsonl"
            qrels_file = data_root / "data_splits" / "retrieval_tasks" / domain / "test" / "qrels" / "dev.tsv"
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
            print(f"Error loading data for {domain}: {e}")
            continue
        
        print(f"Corpus: {len(corpus)} docs | Queries: {len(queries)} queries")
        
        # Run retrieval
        print(f"Running retrieval...")
        results = evaluator.retrieve(corpus, queries)
        
        # Evaluate
        print("Evaluating results...")
        ndcg, _map, recall, precision = evaluator.evaluate(qrels, results, k_values)
        
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
        
        print(f"Recall@10: {recall.get('Recall@10', 0):.4f}")
        print(f"nDCG@10: {ndcg.get('NDCG@10', 0):.4f}")
    
    # Calculate averages
    if all_results:
        avg_results = {
            "Recall@1": np.mean([r["Recall@1"] for r in all_results.values()]),
            "Recall@3": np.mean([r["Recall@3"] for r in all_results.values()]),
            "Recall@5": np.mean([r["Recall@5"] for r in all_results.values()]),
            "Recall@10": np.mean([r["Recall@10"] for r in all_results.values()]),
            "nDCG@1": np.mean([r["nDCG@1"] for r in all_results.values()]),
            "nDCG@3": np.mean([r["nDCG@3"] for r in all_results.values()]),
            "nDCG@5": np.mean([r["nDCG@5"] for r in all_results.values()]),
            "nDCG@10": np.mean([r["nDCG@10"] for r in all_results.values()])
        }
        all_results["average"] = avg_results
        
        print("\n" + "="*60)
        print("AVERAGE RESULTS ACROSS ALL DOMAINS:")
        print("="*60)
        print(f"Recall@5:  {avg_results['Recall@5']:.4f}")
        print(f"Recall@10: {avg_results['Recall@10']:.4f}")
        print(f"nDCG@5:    {avg_results['nDCG@5']:.4f}")
        print(f"nDCG@10:   {avg_results['nDCG@10']:.4f}")
        print("="*60)
    
    return all_results

def evaluate_domain_specific_models():
    """Evaluate domain-specific models on their respective domains."""
    domains = ["clapnq", "fiqa", "govt", "cloud"]
    results = {}
    
    for domain in domains:
        model_path = f"./models/domain_specific_{domain}"
        print(f"\n{'='*60}")
        print(f"Evaluating domain-specific model for: {domain}")
        print(f"{'='*60}")
        
        domain_results = evaluate_model(model_path, domains=[domain])
        if domain_results:
            results[domain] = domain_results[domain]
    
    # Summary
    if results:
        print("\n" + "="*60)
        print("DOMAIN-SPECIFIC MODELS SUMMARY:")
        print("="*60)
        print(f"{'Domain':<10} | {'R@10':<8} | {'nDCG@10':<8}")
        print("-" * 32)
        for domain, res in results.items():
            print(f"{domain:<10} | {res['Recall@10']:<8.4f} | {res['nDCG@10']:<8.4f}")
        
        avg_r10 = np.mean([r['Recall@10'] for r in results.values()])
        avg_ndcg10 = np.mean([r['nDCG@10'] for r in results.values()])
        print("-" * 32)
        print(f"{'Average':<10} | {avg_r10:<8.4f} | {avg_ndcg10:<8.4f}")
        print("="*60)

def main():
    parser = argparse.ArgumentParser(description='Evaluate advanced retrieval models')
    parser.add_argument('--model_path', type=str, help='Path to model to evaluate')
    parser.add_argument('--domain_specific', action='store_true', help='Evaluate all domain-specific models')
    parser.add_argument('--output', type=str, help='Output JSON file for results')
    parser.add_argument('--domains', nargs='+', default=["clapnq", "fiqa", "govt", "cloud"],
                        help='Domains to evaluate on')
    
    args = parser.parse_args()
    
    if args.domain_specific:
        results = evaluate_domain_specific_models()
    elif args.model_path:
        results = evaluate_model(args.model_path, domains=args.domains)
        
        if args.output and results:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\nResults saved to: {args.output}")
    else:
        parser.print_help()

if __name__ == "__main__":
    main()

