import pathlib
import json
import argparse
import numpy as np
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.models import SentenceBERT
from beir.retrieval.search.dense import DenseRetrievalExactSearch
try:
    from beir.retrieval.search.sparse import BM25Search
    from beir.retrieval.search.hybrid import HybridSearch
    BM25_AVAILABLE = True
except ImportError:
    BM25_AVAILABLE = False
    print("⚠️ BM25Search not available. Hybrid retrieval will be disabled.")
import torch
from sentence_transformers import CrossEncoder

# Setup device
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("✅ Using NVIDIA GPU (cuda)")
else:
    device = torch.device("cpu")
    print("⚠️ Using CPU")

def evaluate_model(model_path, config):
    """Evaluate a model with given configuration."""
    print(f"\n{'='*60}")
    print(f"Evaluating: {config.get('experiment_name', 'model')}")
    print(f"{'='*60}")
    
    # Load model
    model = SentenceBERT(model_path, device=device.type)
    
    # Setup retriever
    retriever_type = config.get('retriever_type', 'dense')
    
    if retriever_type == 'dense':
        retriever = DenseRetrievalExactSearch(model, batch_size=128)
    elif retriever_type == 'hybrid':
        # Dense retriever
        dense_retriever = DenseRetrievalExactSearch(model, batch_size=128)
        
        # Sparse retriever (BM25) - will be initialized per domain
        retriever = None  # Will be set per domain
    else:
        retriever = DenseRetrievalExactSearch(model, batch_size=128)
    
    # Setup reranker if requested
    reranker = None
    if config.get('use_reranking', False):
        reranker_model = config.get('reranker_model', 'cross-encoder/ms-marco-MiniLM-L-6-v2')
        try:
            reranker = CrossEncoder(reranker_model, max_length=512)
            print(f"✅ Loaded reranker: {reranker_model}")
        except Exception as e:
            print(f"⚠️ Could not load reranker: {e}")
            reranker = None
    
    # Evaluation
    k_values = [5, 10]
    evaluator = EvaluateRetrieval(retriever, k_values=k_values) if retriever_type == 'dense' else None
    
    domains = ["clapnq", "fiqa", "govt", "cloud"]
    all_results = {}
    
    for domain in domains:
        print(f"\n--- Processing Domain: {domain} ---")
        
        # Load data
        data_root = pathlib.Path(".")
        corpus_path = data_root / "corpora" / "passage_level"
        corpus_file = corpus_path / f"{domain}.jsonl"
        
        # Try test split first, fall back to original
        use_data_splits = config.get('use_data_splits', True)
        if use_data_splits:
            query_file = data_root / "data_splits" / "retrieval_tasks" / domain / "test" / f"{domain}_questions.jsonl"
            qrels_file = data_root / "data_splits" / "retrieval_tasks" / domain / "test" / "qrels" / "dev.tsv"
            if not query_file.exists():
                query_file = data_root / "human" / "retrieval_tasks" / domain / f"{domain}_questions.jsonl"
                qrels_file = data_root / "human" / "retrieval_tasks" / domain / "qrels" / "dev.tsv"
        else:
            query_file = data_root / "human" / "retrieval_tasks" / domain / f"{domain}_questions.jsonl"
            qrels_file = data_root / "human" / "retrieval_tasks" / domain / "qrels" / "dev.tsv"
        
        try:
            corpus, queries, qrels = GenericDataLoader(
                corpus_file=str(corpus_file),
                query_file=str(query_file),
                qrels_file=str(qrels_file)
            ).load_custom()
        except Exception as e:
            print(f"Error loading {domain}: {e}")
            continue
        
        print(f"Corpus: {len(corpus)} docs | Queries: {len(queries)} queries")
        
        # Run retrieval
        if retriever_type == 'hybrid' and BM25_AVAILABLE:
            # Setup hybrid retriever for this domain
            try:
                sparse_retriever = BM25Search(index_name=f"beir-{domain}")
                sparse_retriever.index(corpus)
                hybrid_retriever = HybridSearch(
                    dense_retriever=DenseRetrievalExactSearch(model, batch_size=128),
                    sparse_retriever=sparse_retriever,
                    alpha=config.get('hybrid_alpha', 0.5)
                )
                evaluator = EvaluateRetrieval(hybrid_retriever, k_values=k_values)
            except Exception as e:
                print(f"⚠️ Hybrid retrieval failed for {domain}, using dense: {e}")
                evaluator = EvaluateRetrieval(DenseRetrievalExactSearch(model, batch_size=128), k_values=k_values)
        elif retriever_type == 'hybrid' and not BM25_AVAILABLE:
            print(f"⚠️ Hybrid retrieval requested but BM25 not available for {domain}, using dense")
            evaluator = EvaluateRetrieval(DenseRetrievalExactSearch(model, batch_size=128), k_values=k_values)
        
        print(f"Running retrieval for {domain}...")
        results = evaluator.retrieve(corpus, queries)
        
        # Reranking if enabled
        if reranker:
            print(f"Reranking top results for {domain}...")
            reranked_results = {}
            top_k_rerank = config.get('top_k_rerank', 100)
            
            for query_id, doc_scores in results.items():
                # Get top K documents
                sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)[:top_k_rerank]
                
                if not sorted_docs:
                    reranked_results[query_id] = doc_scores
                    continue
                
                # Prepare pairs for reranking
                query_text = queries[query_id]
                pairs = []
                doc_ids = []
                for doc_id, _ in sorted_docs:
                    doc_text = corpus[doc_id].get('text', '')
                    if 'title' in corpus[doc_id]:
                        doc_text = f"{corpus[doc_id]['title']} {doc_text}"
                    pairs.append([query_text, doc_text])
                    doc_ids.append(doc_id)
                
                # Rerank
                rerank_scores = reranker.predict(pairs)
                
                # Combine original scores with rerank scores
                combined_scores = {}
                for i, (doc_id, orig_score) in enumerate(sorted_docs):
                    # Weighted combination: 0.3 * original + 0.7 * rerank
                    combined_score = 0.3 * orig_score + 0.7 * rerank_scores[i]
                    combined_scores[doc_id] = combined_score
                
                # Sort by combined score and take top 10
                final_docs = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)[:10]
                reranked_results[query_id] = {doc_id: score for doc_id, score in final_docs}
            
            results = reranked_results
        
        # Evaluate
        print("Evaluating results...")
        ndcg, _map, recall, precision = evaluator.evaluate(qrels, results, k_values)
        
        all_results[domain] = {
            "Recall@5": recall['Recall@5'],
            "Recall@10": recall['Recall@10'],
            "nDCG@5": ndcg['NDCG@5'],
            "nDCG@10": ndcg['NDCG@10']
        }
        
        print(f"--- Results for {domain} ---")
        print(f"Recall@10: {recall['Recall@10']:.4f}")
        print(f"nDCG@10:   {ndcg['NDCG@10']:.4f}")
    
    # Calculate averages
    avg_recall_5 = np.mean([res["Recall@5"] for res in all_results.values()])
    avg_recall_10 = np.mean([res["Recall@10"] for res in all_results.values()])
    avg_ndcg_5 = np.mean([res["nDCG@5"] for res in all_results.values()])
    avg_ndcg_10 = np.mean([res["nDCG@10"] for res in all_results.values()])
    
    # Baseline comparison
    paper_recall_5 = 0.30
    paper_recall_10 = 0.38
    paper_ndcg_5 = 0.27
    paper_ndcg_10 = 0.30
    
    summary = {
        'experiment_name': config.get('experiment_name', 'model'),
        'model_path': model_path,
        'config': config,
        'domain_results': all_results,
        'averages': {
            'Recall@5': float(avg_recall_5),
            'Recall@10': float(avg_recall_10),
            'nDCG@5': float(avg_ndcg_5),
            'nDCG@10': float(avg_ndcg_10)
        },
        'baseline_comparison': {
            'Recall@5': {'ours': float(avg_recall_5), 'baseline': paper_recall_5, 'improvement': float(avg_recall_5 - paper_recall_5)},
            'Recall@10': {'ours': float(avg_recall_10), 'baseline': paper_recall_10, 'improvement': float(avg_recall_10 - paper_recall_10)},
            'nDCG@5': {'ours': float(avg_ndcg_5), 'baseline': paper_ndcg_5, 'improvement': float(avg_ndcg_5 - paper_ndcg_5)},
            'nDCG@10': {'ours': float(avg_ndcg_10), 'baseline': paper_ndcg_10, 'improvement': float(avg_ndcg_10 - paper_ndcg_10)}
        }
    }
    
    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)
    print(f"{'Metric':<12} | {'Ours':<10} | {'Baseline':<10} | {'Improvement':<12}")
    print("-" * 60)
    print(f"{'Recall@5':<12} | {avg_recall_5:<10.4f} | {paper_recall_5:<10.4f} | {avg_recall_5 - paper_recall_5:>+10.4f}")
    print(f"{'Recall@10':<12} | {avg_recall_10:<10.4f} | {paper_recall_10:<10.4f} | {avg_recall_10 - paper_recall_10:>+10.4f}")
    print(f"{'nDCG@5':<12} | {avg_ndcg_5:<10.4f} | {paper_ndcg_5:<10.4f} | {avg_ndcg_5 - paper_ndcg_5:>+10.4f}")
    print(f"{'nDCG@10':<12} | {avg_ndcg_10:<10.4f} | {paper_ndcg_10:<10.4f} | {avg_ndcg_10 - paper_ndcg_10:>+10.4f}")
    
    return summary

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--config', type=str, help='Path to config JSON file')
    parser.add_argument('--experiment_name', type=str, default='evaluation')
    parser.add_argument('--retriever_type', type=str, default='dense', choices=['dense', 'hybrid'])
    parser.add_argument('--use_reranking', action='store_true')
    parser.add_argument('--reranker_model', type=str, default='cross-encoder/ms-marco-MiniLM-L-6-v2')
    parser.add_argument('--hybrid_alpha', type=float, default=0.5)
    parser.add_argument('--top_k_rerank', type=int, default=100)
    parser.add_argument('--use_data_splits', action='store_true', default=True)
    parser.add_argument('--output', type=str, help='Output JSON file for results')
    
    args = parser.parse_args()
    
    if args.config:
        with open(args.config, 'r') as f:
            config = json.load(f)
    else:
        config = {
            'experiment_name': args.experiment_name,
            'retriever_type': args.retriever_type,
            'use_reranking': args.use_reranking,
            'reranker_model': args.reranker_model,
            'hybrid_alpha': args.hybrid_alpha,
            'top_k_rerank': args.top_k_rerank,
            'use_data_splits': args.use_data_splits
        }
    
    summary = evaluate_model(args.model_path, config)
    
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"\n✅ Results saved to: {args.output}")

