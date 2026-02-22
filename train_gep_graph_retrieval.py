"""
GEP-Ret: Graph-Enhanced Retrieval with Entity Propagation
Novel Experiment for Tier 1 Conference Publication

Task A Compliant: Retrieval only, no text generation
Uses GNNs to propagate entity relationships for better retrieval.

Core Contributions:
1. Entity Extraction: Extract entities from documents
2. Graph Construction: Build domain-specific knowledge graphs
3. GNN Propagation: Use GCN/GraphSAGE for entity propagation
4. Hybrid Fusion: Combine graph signals with dense retrieval
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
from sentence_transformers import SentenceTransformer
import torch
import torch.nn as nn
import torch.nn.functional as F
import signal
import re
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

class SimpleEntityExtractor:
    """Simple entity extraction (can be replaced with NER model)"""
    def __init__(self):
        # Common entity patterns
        self.patterns = [
            r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b',  # Capitalized phrases
            r'\b\d{4}\b',  # Years
            r'\b[A-Z]{2,}\b',  # Acronyms
        ]
    
    def extract(self, text: str) -> Set[str]:
        """Extract entities from text"""
        entities = set()
        for pattern in self.patterns:
            matches = re.findall(pattern, text)
            entities.update([m for m in matches if len(m) > 2])
        return entities

class GraphConvolution(nn.Module):
    """Simple Graph Convolutional Network layer"""
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
    
    def forward(self, node_features, adjacency):
        """Graph convolution: A * X * W"""
        # Normalize adjacency
        degree = torch.sum(adjacency, dim=-1, keepdim=True)
        degree = torch.clamp(degree, min=1.0)
        normalized_adj = adjacency / degree
        
        # Convolution
        aggregated = torch.mm(normalized_adj, node_features)
        output = self.linear(aggregated)
        return output

class GraphEnhancedRetriever(nn.Module):
    """Graph-enhanced retrieval with entity propagation"""
    def __init__(self, base_model_name='BAAI/bge-large-en-v1.5', hidden_dim=768):
        super().__init__()
        self.base_encoder = SentenceTransformer(base_model_name)
        self.entity_extractor = SimpleEntityExtractor()
        self.hidden_dim = hidden_dim
        
        # GNN for entity propagation
        self.gcn1 = GraphConvolution(hidden_dim, hidden_dim)
        self.gcn2 = GraphConvolution(hidden_dim, hidden_dim)
        
        # Fusion layer
        self.fusion = nn.Linear(hidden_dim * 2, hidden_dim)
    
    def build_graph(self, documents: Dict[str, str]) -> Tuple[torch.Tensor, Dict[str, int]]:
        """Build knowledge graph from documents"""
        # Extract entities
        entity_to_docs = defaultdict(set)
        doc_to_entities = {}
        
        for doc_id, doc_text in documents.items():
            entities = self.entity_extractor.extract(doc_text)
            doc_to_entities[doc_id] = entities
            for entity in entities:
                entity_to_docs[entity].add(doc_id)
        
        # Create adjacency matrix (documents connected if they share entities)
        doc_ids = list(documents.keys())
        num_docs = len(doc_ids)
        adjacency = torch.zeros(num_docs, num_docs)
        
        for i, doc_id1 in enumerate(doc_ids):
            entities1 = doc_to_entities[doc_id1]
            for j, doc_id2 in enumerate(doc_ids):
                if i != j:
                    entities2 = doc_to_entities[doc_id2]
                    # Edge weight = Jaccard similarity of entities
                    if entities1 or entities2:
                        intersection = len(entities1 & entities2)
                        union = len(entities1 | entities2)
                        adjacency[i, j] = intersection / union if union > 0 else 0.0
        
        return adjacency, {doc_id: i for i, doc_id in enumerate(doc_ids)}
    
    def encode_with_graph(self, documents: Dict[str, str], doc_embs: torch.Tensor):
        """Encode documents with graph propagation"""
        # Build graph
        adjacency, doc_to_idx = self.build_graph(documents)
        adjacency = adjacency.to(doc_embs.device)
        
        # Graph propagation
        x = self.gcn1(doc_embs, adjacency)
        x = F.relu(x)
        x = self.gcn2(x, adjacency)
        
        # Combine original and graph-enhanced
        combined = torch.cat([doc_embs, x], dim=-1)
        enhanced = self.fusion(combined)
        
        return enhanced

def run_gep_graph_retrieval(config: Dict, gpu_id: Optional[int] = None) -> Optional[str]:
    """Run GEP-Ret Graph-Enhanced Retrieval Experiment"""
    global shutdown_requested
    
    if gpu_id is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        logging.info(f"Using GPU {gpu_id}")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Device: {device}")
    
    output_dir = pathlib.Path(config.get('output_dir', 'experiments/retrieval/gep_graph_retrieval'))
    output_dir.mkdir(parents=True, exist_ok=True)
    
    model_name = config.get('model_path', 'BAAI/bge-large-en-v1.5')
    domains = config.get('domains', MTRAG_DOMAINS)
    use_data_splits = config.get('use_data_splits', True)
    
    # Initialize graph-enhanced retriever
    logging.info(f"Initializing GEP-Ret with {model_name}")
    retriever = GraphEnhancedRetriever(base_model_name=model_name)
    retriever.to(device)
    retriever.eval()
    
    # Evaluation
    logging.info("Evaluating GEP-Ret Graph-Enhanced Retrieval...")
    all_results = {}
    
    for domain in domains:
        if shutdown_requested:
            break
        
        logging.info(f"\n{'='*60}")
        logging.info(f"Evaluating domain: {domain}")
        logging.info(f"{'='*60}")
        
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
            logging.error(f"Error loading {domain}: {e}")
            continue
        
        # Encode corpus with graph enhancement
        logging.info("Encoding corpus with graph enhancement...")
        corpus_texts = {doc_id: corpus[doc_id].get('text', '') for doc_id in corpus.keys()}
        doc_texts_list = [corpus_texts[doc_id] for doc_id in corpus.keys()]
        doc_embs = retriever.base_encoder.encode(doc_texts_list, batch_size=64, convert_to_tensor=True)
        doc_embs = doc_embs.to(device)
        
        # Apply graph propagation
        doc_embs_enhanced = retriever.encode_with_graph(corpus_texts, doc_embs)
        doc_ids = list(corpus.keys())
        
        # Retrieve
        results = {}
        
        for query_id, query_text in queries.items():
            query_emb = retriever.base_encoder.encode(query_text, convert_to_tensor=True)
            query_emb = query_emb.to(device)
            
            # Compute similarities
            scores = torch.mm(query_emb.unsqueeze(0), doc_embs_enhanced.t()).squeeze(0)
            scores = scores.cpu().numpy()
            
            # Get top-k
            top_indices = np.argsort(scores)[::-1][:100]
            results[query_id] = {
                doc_ids[idx]: float(scores[idx]) for idx in top_indices
            }
        
        # Evaluate
        evaluator = EvaluateRetrieval(None, k_values=[1, 3, 5, 10])
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
        
        logging.info(f"Domain {domain} - nDCG@10: {ndcg.get('NDCG@10', 0):.4f}")
    
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
        
        logging.info(f"\n✅ GEP-Ret completed!")
        logging.info(f"Average nDCG@10: {avg_results.get('nDCG@10', 0):.4f}")
        logging.info(f"Results saved to: {results_file}")
        
        return str(results_file)
    
    return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help='Config file path')
    parser.add_argument('--gpu', type=int, help='GPU ID')
    args = parser.parse_args()
    
    if args.config:
        with open(args.config, 'r') as f:
            config = json.load(f)
    else:
        config = {}
    
    run_gep_graph_retrieval(config, gpu_id=args.gpu)

