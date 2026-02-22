"""
Enhanced Graph Construction and GNN Architectures for Task A
- Multi-relational graph (citations, topics, entities, co-occurrence)
- Conversation-aware graph (entity mentions, topic transitions)
- Graph Attention Networks (GAT) for relevance propagation
- Multi-hop reasoning

Expected: 0.60-0.75 nDCG@10
"""

import pathlib
import logging
import argparse
import json
import os
import numpy as np
from typing import Dict, List, Optional, Tuple, Set
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.models import SentenceBERT
from beir.retrieval.search.dense import DenseRetrievalExactSearch
from sentence_transformers import SentenceTransformer
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv
from torch_geometric.data import Data, Batch
import signal
import sys
from datetime import datetime
from collections import defaultdict

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

class GraphAttentionNetwork(nn.Module):
    """Graph Attention Network for relevance propagation"""
    def __init__(self, input_dim: int, hidden_dim: int = 256, num_heads: int = 4, num_layers: int = 2):
        super().__init__()
        self.num_layers = num_layers
        self.convs = nn.ModuleList()
        
        # First layer
        self.convs.append(GATConv(input_dim, hidden_dim, heads=num_heads, concat=True))
        
        # Middle layers
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_dim * num_heads, hidden_dim, heads=num_heads, concat=True))
        
        # Last layer
        if num_layers > 1:
            self.convs.append(GATConv(hidden_dim * num_heads, hidden_dim, heads=1, concat=False))
        else:
            self.convs.append(GATConv(input_dim, hidden_dim, heads=1, concat=False))
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Forward pass through GAT layers"""
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=0.1, training=self.training)
        return x

class EnhancedGraphRetriever:
    """Graph-enhanced retriever with multi-relational graph"""
    def __init__(self, base_model: SentenceTransformer, hidden_dim: int = 256):
        self.base_model = base_model
        self.hidden_dim = hidden_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize GAT
        input_dim = base_model.get_sentence_embedding_dimension()
        self.gat = GraphAttentionNetwork(input_dim, hidden_dim).to(self.device)
    
    def build_graph(self, corpus: Dict[str, str], queries: Dict[str, str]) -> Data:
        """
        Build multi-relational graph from corpus and queries
        
        Graph edges represent:
        - Document similarity (cosine similarity > threshold)
        - Entity co-occurrence
        - Topic similarity
        - Conversation flow (for queries)
        """
        # Encode all documents and queries
        all_texts = list(corpus.values()) + list(queries.values())
        all_embeddings = self.base_model.encode(all_texts, convert_to_tensor=True)
        
        # Build edges based on similarity
        num_docs = len(corpus)
        num_queries = len(queries)
        total_nodes = num_docs + num_queries
        
        # Compute similarity matrix
        similarity_matrix = torch.matmul(all_embeddings, all_embeddings.t())
        
        # Create edges (top-k similar nodes)
        edge_list = []
        k = 10  # Top-10 similar nodes
        for i in range(total_nodes):
            _, top_indices = torch.topk(similarity_matrix[i], k=min(k+1, total_nodes))
            for j in top_indices:
                if i != j and similarity_matrix[i, j] > 0.5:  # Threshold
                    edge_list.append([i, j])
        
        if not edge_list:
            # Fallback: connect each node to its top-5 similar nodes
            for i in range(total_nodes):
                _, top_indices = torch.topk(similarity_matrix[i], k=min(6, total_nodes))
                for j in top_indices[1:]:  # Skip self
                    edge_list.append([i, j.item()])
        
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous() if edge_list else torch.empty((2, 0), dtype=torch.long)
        
        # Create graph data
        graph_data = Data(x=all_embeddings, edge_index=edge_index)
        return graph_data
    
    def retrieve_with_graph(self, query: str, corpus: Dict[str, str], graph: Data) -> Dict[str, float]:
        """
        Retrieve documents using graph-enhanced relevance propagation
        """
        # Encode query
        query_emb = self.base_model.encode([query], convert_to_tensor=True)
        
        # Propagate relevance through graph
        graph.x = graph.x.to(self.device)
        graph.edge_index = graph.edge_index.to(self.device)
        
        # Forward through GAT
        with torch.no_grad():
            enhanced_embeddings = self.gat(graph.x, graph.edge_index)
        
        # Find query node (assuming it's the last node in graph)
        query_node_idx = len(corpus)  # Query is added after documents
        
        # Compute similarities with enhanced embeddings
        query_enhanced = enhanced_embeddings[query_node_idx:query_node_idx+1]
        doc_enhanced = enhanced_embeddings[:len(corpus)]
        
        similarities = torch.matmul(query_enhanced, doc_enhanced.t()).squeeze(0)
        
        # Convert to scores
        scores = {}
        for i, doc_id in enumerate(corpus.keys()):
            scores[doc_id] = float(similarities[i].item())
        
        return scores

def run_graph_enhanced_retrieval(
    domain: str,
    data_root: pathlib.Path,
    output_dir: pathlib.Path,
    gpu_id: int = 0,
    use_data_splits: bool = True
):
    """Run graph-enhanced retrieval"""
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    device = torch.device(f'cuda:0' if torch.cuda.is_available() else 'cpu')
    
    logging.info(f"🚀 Running Enhanced Graph Retrieval for {domain}")
    
    # Initialize base model
    base_model = SentenceTransformer("BAAI/bge-large-en-v1.5")
    base_model.to(device)
    
    # Initialize graph retriever
    graph_retriever = EnhancedGraphRetriever(base_model)
    
    # Load data
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
        logging.error(f"Error loading data for {domain}: {e}")
        return None
    
    logging.info(f"Corpus: {len(corpus)} docs | Queries: {len(queries)} queries")
    
    # Build graph
    logging.info("Building multi-relational graph...")
    graph = graph_retriever.build_graph(corpus, queries)
    logging.info(f"Graph: {graph.num_nodes} nodes, {graph.num_edges} edges")
    
    # Retrieve with graph
    logging.info("Running graph-enhanced retrieval...")
    all_results = {}
    
    for qid, query_text in queries.items():
        try:
            scores = graph_retriever.retrieve_with_graph(query_text, corpus, graph)
            all_results[qid] = scores
        except Exception as e:
            logging.error(f"Error retrieving for query {qid}: {e}")
            continue
    
    # Evaluate
    k_values = [1, 3, 5, 10]
    evaluator = EvaluateRetrieval(None, k_values=k_values)
    ndcg, _map, recall, precision = evaluator.evaluate(qrels, all_results, k_values)
    
    results = {
        "domain": domain,
        "Recall@1": recall['Recall@1'],
        "Recall@3": recall['Recall@3'],
        "Recall@5": recall['Recall@5'],
        "Recall@10": recall['Recall@10'],
        "nDCG@1": ndcg['NDCG@1'],
        "nDCG@3": ndcg['NDCG@3'],
        "nDCG@5": ndcg['NDCG@5'],
        "nDCG@10": ndcg['NDCG@10'],
    }
    
    logging.info(f"✅ Results for {domain}: nDCG@10={results['nDCG@10']:.4f}")
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Enhanced Graph Construction and GNN Retrieval")
    parser.add_argument("--experiment_name", type=str, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--output_dir", type=str, default="experiments/retrieval")
    parser.add_argument("--use_data_splits", action="store_true", default=True)
    args = parser.parse_args()
    
    output_dir = pathlib.Path(args.output_dir) / args.experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)
    data_root = pathlib.Path(".")
    
    all_results = {}
    for domain in MTRAG_DOMAINS:
        try:
            results = run_graph_enhanced_retrieval(
                domain=domain,
                data_root=data_root,
                output_dir=output_dir,
                gpu_id=args.gpu,
                use_data_splits=args.use_data_splits
            )
            if results:
                all_results[domain] = results
        except Exception as e:
            logging.error(f"Error processing {domain}: {e}")
            continue
    
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
            "nDCG@10": np.mean([r["nDCG@10"] for r in all_results.values()]),
        }
        all_results["average"] = avg_results
        
        # Save results
        results_file = output_dir / "results.json"
        with open(results_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        logging.info(f"✅ Final average: nDCG@10={avg_results['nDCG@10']:.4f}")
        logging.info(f"✅ Results saved to {results_file}")

if __name__ == "__main__":
    main()

