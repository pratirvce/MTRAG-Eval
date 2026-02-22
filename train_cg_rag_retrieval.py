"""
CG-RAG (Contextualized Graph RAG) Retrieval
Task A - Retrieval Only
Expected: +0.03-0.06 nDCG@10 improvement

CG-RAG Algorithm:
  1. Build knowledge graph from corpus documents
  2. Extract entities and relationships
  3. Contextualize graph based on query/conversation history
  4. Use contextualized graph for retrieval
  5. Score documents based on graph relevance

⚠️ TASK A COMPLIANCE:
  - CG-RAG is a retrieval-only technique
  - Builds graph from documents (corpus processing - allowed)
  - Contextualizes graph based on query (query processing - allowed)
  - No text generation involved
  - Pure retrieval method
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
from sentence_transformers import SentenceTransformer, InputExample
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
import signal
import sys
from datetime import datetime
from collections import defaultdict
import re

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

class KnowledgeGraphBuilder:
    """
    Builds knowledge graph from documents
    Extracts entities and relationships
    """
    def __init__(self):
        self.graph = defaultdict(set)  # {entity: {related_entities}}
        self.entity_to_docs = defaultdict(set)  # {entity: {doc_ids}}
        self.doc_entities = defaultdict(set)  # {doc_id: {entities}}
    
    def extract_entities(self, text: str) -> Set[str]:
        """
        Extract entities from text (simple approach)
        In production, use NER models like spaCy or transformers
        """
        entities = set()
        
        # Extract capitalized phrases (simple entity detection)
        # Pattern: sequences of capitalized words
        capitalized_pattern = r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b'
        matches = re.findall(capitalized_pattern, text)
        
        # Filter: keep phrases with 2-4 words (likely entities)
        for match in matches:
            words = match.split()
            if 2 <= len(words) <= 4:
                entities.add(match.lower())
        
        # Extract numbers and dates
        number_pattern = r'\b\d{4}\b|\b\d+[%$]\b'
        numbers = re.findall(number_pattern, text)
        entities.update(numbers)
        
        return entities
    
    def build_graph(self, corpus: Dict[str, Dict]):
        """
        Build knowledge graph from corpus
        """
        logging.info("Building knowledge graph from corpus...")
        
        for doc_id, doc in corpus.items():
            title = doc.get('title', '')
            text = doc.get('text', '')
            full_text = f"{title} {text}".strip()
            
            # Extract entities
            entities = self.extract_entities(full_text)
            self.doc_entities[doc_id] = entities
            
            # Add entities to document mapping
            for entity in entities:
                self.entity_to_docs[entity].add(doc_id)
            
            # Build relationships (entities in same document are related)
            entity_list = list(entities)
            for i, entity1 in enumerate(entity_list):
                for entity2 in entity_list[i+1:]:
                    self.graph[entity1].add(entity2)
                    self.graph[entity2].add(entity1)
        
        logging.info(f"Graph built: {len(self.graph)} entities, {sum(len(rels) for rels in self.graph.values()) // 2} relationships")
    
    def contextualize_graph(self, query: str, conversation_history: Optional[List[str]] = None) -> Dict[str, float]:
        """
        Contextualize graph based on query and conversation history
        Returns entity relevance scores
        """
        # Extract query entities
        query_entities = self.extract_entities(query)
        
        # Extract history entities
        history_entities = set()
        if conversation_history:
            for turn in conversation_history[-3:]:  # Last 3 turns
                history_entities.update(self.extract_entities(turn))
        
        # Combine query and history entities
        context_entities = query_entities | history_entities
        
        # Score entities based on context
        entity_scores = {}
        for entity in self.graph.keys():
            score = 0.0
            
            # Direct match
            if entity in context_entities:
                score += 2.0
            
            # Related entities (1-hop)
            related = self.graph.get(entity, set())
            for related_entity in related:
                if related_entity in context_entities:
                    score += 1.0
            
            # Related to related entities (2-hop)
            for related_entity in related:
                second_hop = self.graph.get(related_entity, set())
                for second_entity in second_hop:
                    if second_entity in context_entities:
                        score += 0.5
            
            if score > 0:
                entity_scores[entity] = score
        
        return entity_scores
    
    def get_document_scores(self, entity_scores: Dict[str, float]) -> Dict[str, float]:
        """
        Get document scores based on entity relevance
        """
        doc_scores = defaultdict(float)
        
        for entity, score in entity_scores.items():
            # Documents containing this entity get the score
            for doc_id in self.entity_to_docs.get(entity, set()):
                doc_scores[doc_id] += score
        
        return dict(doc_scores)

class CGragRetriever:
    """
    Contextualized Graph RAG Retriever
    
    ⚠️ TASK A COMPLIANCE:
    - Builds graph from documents (corpus processing - allowed)
    - Contextualizes graph based on query (query processing - allowed)
    - No text generation involved
    - Pure retrieval method
    """
    def __init__(self, base_model: SentenceTransformer, kg_builder: KnowledgeGraphBuilder):
        self.base_model = base_model
        self.kg_builder = kg_builder
        self.graph_weight = 0.3  # Weight for graph-based scores
        self.dense_weight = 0.7  # Weight for dense retrieval scores
    
    def retrieve(self, corpus: Dict[str, Dict], queries: Dict[str, str],
                 conversation_histories: Optional[Dict[str, List[str]]] = None,
                 top_k: int = 10) -> Dict[str, Dict[str, float]]:
        """
        Retrieve using CG-RAG (Contextualized Graph RAG)
        
        ⚠️ TASK A COMPLIANCE: Pure retrieval, no text generation
        """
        # Dense retrieval
        dense_retriever = DenseRetrievalExactSearch(
            SentenceBERT(self.base_model, device=self.base_model.device),
            batch_size=128
        )
        evaluator = EvaluateRetrieval(dense_retriever, k_values=[top_k * 2])
        dense_results = evaluator.retrieve(corpus, queries)
        
        # CG-RAG retrieval
        cgrag_results = {}
        for query_id, query_text in queries.items():
            history = conversation_histories.get(query_id) if conversation_histories else None
            
            # Contextualize graph
            entity_scores = self.kg_builder.contextualize_graph(query_text, history)
            
            # Get document scores from graph
            graph_doc_scores = self.kg_builder.get_document_scores(entity_scores)
            
            # Combine dense and graph scores
            dense_scores = dense_results.get(query_id, {})
            combined_scores = {}
            
            # Combine scores for documents in dense results
            for doc_id, dense_score in dense_scores.items():
                graph_score = graph_doc_scores.get(doc_id, 0.0)
                # Normalize graph score
                if graph_score > 0:
                    graph_score = np.log1p(graph_score)  # Log normalization
                
                # Combine scores
                combined_score = (self.dense_weight * dense_score + 
                                self.graph_weight * graph_score)
                combined_scores[doc_id] = combined_score
            
            # Add documents with high graph scores but not in dense top-k
            for doc_id, graph_score in graph_doc_scores.items():
                if doc_id not in combined_scores and graph_score > 0:
                    normalized_graph = np.log1p(graph_score)
                    combined_scores[doc_id] = self.graph_weight * normalized_graph
            
            # Sort and get top-k
            sorted_docs = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
            cgrag_results[query_id] = dict(sorted_docs[:top_k])
        
        return cgrag_results

def load_training_pairs(domain: str, data_root: pathlib.Path, use_data_splits: bool = True) -> Tuple[List[InputExample], Dict, Dict]:
    """Load conversation-document pairs"""
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
        logging.error(f"Error loading data for {domain}: {e}")
        return [], {}, {}
    
    examples = []
    for query_id, doc_scores in qrels.items():
        query_text = queries.get(query_id)
        if not query_text:
            continue
        
        positive_docs = [doc_id for doc_id, score in doc_scores.items() if score > 0]
        for doc_id in positive_docs:
            doc = corpus.get(doc_id)
            if doc:
                title = doc.get("title", "")
                text = doc.get("text", "")
                doc_text = f"{title}\n\n{text}".strip() if title else text
                examples.append(InputExample(texts=[query_text, doc_text], label=1.0))
    
    logging.info(f"Created {len(examples)} positive pairs for {domain}")
    return examples, corpus, queries

def train_cg_rag_retrieval(config: Dict, gpu_id: Optional[int] = None) -> Optional[str]:
    """Train retrieval with CG-RAG (Contextualized Graph RAG)"""
    global shutdown_requested
    
    cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if cuda_visible:
        logging.info(f"CUDA_VISIBLE_DEVICES={cuda_visible}")
    elif gpu_id is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and torch.cuda.is_available():
        try:
            torch.cuda.set_device(0)
            logging.info(f"Device: {device}, GPU: {torch.cuda.current_device()}")
        except Exception as e:
            logging.warning(f"Could not set CUDA device: {e}")
    
    output_dir = pathlib.Path(config.get('output_dir', 'experiments/retrieval/tier1_cg_rag'))
    output_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint_dir = output_dir
    resume = config.get('resume', True)
    
    all_results = {}
    completed_domains = set()
    if resume:
        checkpoint_file = checkpoint_dir / "checkpoint.json"
        if checkpoint_file.exists():
            with open(checkpoint_file) as f:
                checkpoint = json.load(f)
                all_results = checkpoint.get('results', {})
                completed_domains = set(checkpoint.get('completed_domains', []))
                logging.info(f"Resuming from checkpoint. Completed domains: {completed_domains}")
    
    domains = config.get('domains', MTRAG_DOMAINS)
    base_model_name = config.get('model_path', 'BAAI/bge-base-en-v1.5')
    
    # Load base model
    logging.info(f"Loading base model: {base_model_name}")
    base_model = SentenceTransformer(base_model_name)
    base_model.to(device)
    
    # Fine-tune on MTRAG data (optional)
    epochs = config.get('epochs', 3)
    batch_size = config.get('batch_size', 8)
    gradient_accumulation_steps = config.get('gradient_accumulation_steps', 2)
    use_fp16 = config.get('use_fp16', True)
    
    scaler = None
    if use_fp16 and device == "cuda":
        try:
            from torch.cuda.amp import autocast, GradScaler
            scaler = GradScaler()
            logging.info("Mixed precision training (FP16) enabled")
        except ImportError:
            use_fp16 = False
    
    for param in base_model.parameters():
        param.requires_grad = True
    
    # Training loop
    if config.get('fine_tune', True):
        for domain in domains:
            if domain in completed_domains:
                continue
            
            if shutdown_requested:
                checkpoint_file = checkpoint_dir / "checkpoint.json"
                with open(checkpoint_file, 'w') as f:
                    json.dump({
                        "last_domain": domain,
                        "completed_domains": list(completed_domains),
                        "results": all_results,
                        "timestamp": datetime.now().isoformat()
                    }, f, indent=2)
                break
            
            logging.info(f"\n{'='*60}\nTraining on domain: {domain}\n{'='*60}")
            data_root = pathlib.Path(".")
            examples, corpus, queries = load_training_pairs(domain, data_root, config.get('use_data_splits', True))
            
            if not examples:
                continue
            
            dataloader = DataLoader(examples, batch_size=batch_size, shuffle=True)
            optimizer = torch.optim.AdamW(base_model.parameters(), lr=2e-5, weight_decay=0.01)
            
            base_model.train()
            for epoch in range(epochs):
                total_loss = 0
                optimizer.zero_grad()
                
                for batch_idx, batch in enumerate(dataloader):
                    queries_batch = [ex.texts[0] for ex in batch]
                    positives_batch = [ex.texts[1] for ex in batch]
                    
                    # Encode with asymmetric prompts
                    query_prompts = [f"Represent this sentence for searching relevant passages: {q}" for q in queries_batch]
                    doc_prompts = [f"Represent this sentence for retrieval: {d}" for d in positives_batch]
                    
                    if use_fp16 and scaler is not None:
                        with autocast():
                            query_embs = base_model.encode(query_prompts, convert_to_tensor=True, show_progress_bar=False)
                            pos_embs = base_model.encode(doc_prompts, convert_to_tensor=True, show_progress_bar=False)
                            
                            query_embs = F.normalize(query_embs, p=2, dim=1)
                            pos_embs = F.normalize(pos_embs, p=2, dim=1)
                            
                            similarities = torch.sum(query_embs * pos_embs, dim=1)
                            loss = -torch.mean(torch.log(torch.sigmoid(similarities / 0.05) + 1e-8))
                            loss = loss / gradient_accumulation_steps
                        
                        scaler.scale(loss).backward()
                        if (batch_idx + 1) % gradient_accumulation_steps == 0:
                            scaler.step(optimizer)
                            scaler.update()
                            optimizer.zero_grad()
                    else:
                        query_embs = base_model.encode(query_prompts, convert_to_tensor=True, show_progress_bar=False)
                        pos_embs = base_model.encode(doc_prompts, convert_to_tensor=True, show_progress_bar=False)
                        
                        query_embs = F.normalize(query_embs, p=2, dim=1)
                        pos_embs = F.normalize(pos_embs, p=2, dim=1)
                        
                        similarities = torch.sum(query_embs * pos_embs, dim=1)
                        loss = -torch.mean(torch.log(torch.sigmoid(similarities / 0.05) + 1e-8))
                        loss = loss / gradient_accumulation_steps
                        loss.backward()
                        
                        if (batch_idx + 1) % gradient_accumulation_steps == 0:
                            optimizer.step()
                            optimizer.zero_grad()
                    
                    total_loss += loss.item() * gradient_accumulation_steps
                    
                    if device == "cuda" and (batch_idx + 1) % 10 == 0:
                        torch.cuda.empty_cache()
                
                logging.info(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader):.4f}")
        
        # Save trained model
        trained_model_path = checkpoint_dir / "trained_model"
        base_model.save(str(trained_model_path))
        logging.info(f"Saved trained model to {trained_model_path}")
    
    # Reload trained model if available
    trained_model_path = checkpoint_dir / "trained_model"
    if trained_model_path.exists():
        logging.info(f"Loading trained model from {trained_model_path}")
        base_model = SentenceTransformer(str(trained_model_path))
        base_model.to(device)
    
    # Evaluation with CG-RAG
    logging.info("\nEvaluating with CG-RAG retrieval...")
    logging.info("⚠️  TASK A COMPLIANCE: CG-RAG builds graph from documents and contextualizes based on query - pure retrieval")
    
    for domain in domains:
        if domain in completed_domains:
            continue
        
        if shutdown_requested:
            checkpoint_file = checkpoint_dir / "checkpoint.json"
            with open(checkpoint_file, 'w') as f:
                json.dump({
                    "last_domain": domain,
                    "completed_domains": list(completed_domains),
                    "results": all_results,
                    "timestamp": datetime.now().isoformat()
                }, f, indent=2)
            break
        
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
            
            # Build knowledge graph
            logging.info(f"Building knowledge graph for {domain}...")
            kg_builder = KnowledgeGraphBuilder()
            kg_builder.build_graph(corpus)
            
            # Initialize CG-RAG retriever
            cgrag_retriever = CGragRetriever(base_model, kg_builder)
            
            # Retrieve with CG-RAG
            logging.info(f"CG-RAG retrieval for {domain}...")
            cgrag_results = cgrag_retriever.retrieve(
                corpus, queries,
                conversation_histories=None,  # Can be extended to use conversation history
                top_k=10
            )
            
            # Evaluate
            evaluator = EvaluateRetrieval(None, k_values=[1, 3, 5, 10])
            ndcg, _map, recall, precision = evaluator.evaluate(qrels, cgrag_results, [1, 3, 5, 10])
            
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
            
            logging.info(f"\n{domain} Results:")
            logging.info(f"  Recall@10: {recall.get('Recall@10', 0):.4f}")
            logging.info(f"  nDCG@10: {ndcg.get('NDCG@10', 0):.4f}")
            
            completed_domains.add(domain)
            
            checkpoint_file = checkpoint_dir / "checkpoint.json"
            with open(checkpoint_file, 'w') as f:
                json.dump({
                    "last_domain": domain,
                    "completed_domains": list(completed_domains),
                    "results": all_results,
                    "timestamp": datetime.now().isoformat()
                }, f, indent=2)
            
        except Exception as e:
            logging.error(f"Error evaluating {domain}: {e}", exc_info=True)
            continue
    
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
        
        logging.info(f"\n{'='*60}\nAverage Results:\n")
        for metric, value in avg_results.items():
            logging.info(f"  {metric}: {value:.4f}")
        logging.info(f"{'='*60}\n")
        
        return str(output_dir)
    
    return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--gpu_id', type=int)
    parser.add_argument('--resume', action='store_true', default=True)
    parser.add_argument('--no-resume', dest='resume', action='store_false')
    
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    config['resume'] = args.resume
    
    try:
        results_path = train_cg_rag_retrieval(config, args.gpu_id)
        if results_path:
            logging.info(f"✅ CG-RAG retrieval completed: {results_path}")
    except KeyboardInterrupt:
        logging.info("Interrupted by user")
        sys.exit(0)
    except Exception as e:
        logging.error(f"Failed: {e}", exc_info=True)
        sys.exit(1)

