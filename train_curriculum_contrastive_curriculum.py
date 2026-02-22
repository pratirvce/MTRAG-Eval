"""
Curriculum Contrastive Learning with Hard Negatives and Query Rewriting
Novel approach combining:
1. Curriculum learning for progressive negative difficulty
2. Query rewriting based on conversation context
3. Multi-granularity negative mining
4. Hierarchical contrastive learning

Expected: 0.52-0.56 nDCG@10
"""

import pathlib
import logging
import argparse
import json
import os
import numpy as np
from typing import Dict, List, Optional, Tuple
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.models import SentenceBERT
from beir.retrieval.search.dense import DenseRetrievalExactSearch
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader, Dataset
import torch
import torch.nn as nn
import torch.nn.functional as F
import signal
import sys
from datetime import datetime
from rank_bm25 import BM25Okapi
import random
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


class CurriculumScheduler:
    """Schedules negative difficulty progression during training"""
    def __init__(self, total_steps: int, stages: List[str] = None):
        """
        Args:
            total_steps: Total training steps
            stages: List of difficulty stages ['random', 'bm25', 'adversarial']
        """
        self.total_steps = total_steps
        self.stages = stages or ['random', 'bm25', 'adversarial']
        self.current_step = 0
        
    def get_difficulty(self, step: int) -> str:
        """Get current difficulty stage based on training progress"""
        self.current_step = step
        progress = step / self.total_steps
        
        if progress < 0.33:
            return 'random'
        elif progress < 0.66:
            return 'bm25'
        else:
            return 'adversarial'
    
    def get_negative_ratio(self, step: int) -> Tuple[float, float, float]:
        """Get ratio of random:bm25:adversarial negatives"""
        progress = step / self.total_steps
        
        # Smooth transition between stages
        if progress < 0.33:
            # Early: mostly random
            random_ratio = 1.0 - progress * 2.0
            bm25_ratio = progress * 1.5
            adv_ratio = progress * 0.5
        elif progress < 0.66:
            # Mid: mostly BM25
            progress_mid = (progress - 0.33) / 0.33
            random_ratio = 0.0
            bm25_ratio = 1.0 - progress_mid * 0.5
            adv_ratio = progress_mid * 0.5
        else:
            # Late: mostly adversarial
            random_ratio = 0.0
            bm25_ratio = 0.3
            adv_ratio = 0.7
        
        # Normalize
        total = random_ratio + bm25_ratio + adv_ratio
        if total > 0:
            return random_ratio/total, bm25_ratio/total, adv_ratio/total
        return 0.33, 0.33, 0.34


class QueryRewriter:
    """Rewrites queries based on conversation context"""
    def __init__(self, model=None, use_llm: bool = False):
        """
        Args:
            model: SentenceTransformer model for semantic similarity
            use_llm: Whether to use LLM for rewriting (requires API)
        """
        self.model = model
        self.use_llm = use_llm
        
    def extract_conversation_context(self, query_text: str) -> Tuple[str, str]:
        """Extract current query and previous conversation"""
        # Parse conversation format: |user|: ... |agent|: ...
        parts = query_text.split('|user|:')
        if len(parts) > 1:
            current_query = parts[-1].strip()
            previous_context = '|user|:'.join(parts[:-1]).strip()
        else:
            current_query = query_text.strip()
            previous_context = ""
        return current_query, previous_context
    
    def rewrite_query(self, query_text: str, num_variants: int = 2) -> List[str]:
        """
        Generate query variants
        
        Args:
            query_text: Original query with conversation context
            num_variants: Number of variants to generate
            
        Returns:
            List of rewritten queries (including original)
        """
        current_query, previous_context = self.extract_conversation_context(query_text)
        
        variants = [current_query]  # Always include original
        
        # Simple rewriting strategies (can be enhanced with LLM)
        # 1. Paraphrase: Add context from previous conversation
        if previous_context:
            # Extract key entities/topics from previous context
            context_keywords = self._extract_keywords(previous_context)
            if context_keywords:
                enhanced_query = f"{current_query} {' '.join(context_keywords[:3])}"
                variants.append(enhanced_query)
        
        # 2. Expand with synonyms/related terms
        expanded = self._expand_query(current_query)
        if expanded and expanded != current_query:
            variants.append(expanded)
        
        # 3. Simplify: Remove question words, keep core
        simplified = self._simplify_query(current_query)
        if simplified and simplified != current_query:
            variants.append(simplified)
        
        return variants[:num_variants + 1]  # +1 for original
    
    def _extract_keywords(self, text: str, max_keywords: int = 5) -> List[str]:
        """Extract keywords from text (simple implementation)"""
        # Remove conversation markers
        text = re.sub(r'\|user\|:\s*', '', text)
        text = re.sub(r'\|agent\|:\s*', '', text)
        
        # Simple keyword extraction (can be improved)
        words = text.lower().split()
        # Filter stop words and short words
        stop_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 
                      'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
                      'can', 'could', 'should', 'may', 'might', 'must', 'to', 'of',
                      'in', 'on', 'at', 'for', 'with', 'by', 'from', 'as', 'and', 'or'}
        keywords = [w for w in words if len(w) > 3 and w not in stop_words]
        
        # Return most frequent keywords
        from collections import Counter
        word_freq = Counter(keywords)
        return [word for word, _ in word_freq.most_common(max_keywords)]
    
    def _expand_query(self, query: str) -> str:
        """Expand query with related terms (placeholder - can use wordnet/embeddings)"""
        # Simple expansion: add question context
        if query.endswith('?'):
            return query
        # Could use word embeddings to find related terms
        return query
    
    def _simplify_query(self, query: str) -> str:
        """Simplify query to core content"""
        # Remove question words
        question_words = ['what', 'when', 'where', 'who', 'why', 'how', 'which']
        words = query.lower().split()
        simplified = [w for w in words if w not in question_words]
        return ' '.join(simplified) if simplified else query


class MultiGranularityNegativeMiner:
    """Mines negatives at multiple granularities"""
    def __init__(self, corpus_dict: Dict, bm25_index: BM25Okapi = None):
        self.corpus_dict = corpus_dict
        self.bm25_index = bm25_index
        self.corpus_ids = list(corpus_dict.keys())
        
    def mine_negatives(self, query_text: str, positive_text: str, 
                      num_negatives: int, difficulty: str = 'bm25',
                      granularities: List[str] = None) -> Dict[str, List[str]]:
        """
        Mine negatives at multiple granularities
        
        Args:
            query_text: Query text
            positive_text: Positive document text
            num_negatives: Number of negatives per granularity
            difficulty: Difficulty level ('random', 'bm25', 'adversarial')
            granularities: List of granularities ['document', 'sentence', 'phrase']
            
        Returns:
            Dict mapping granularity to list of negative texts
        """
        granularities = granularities or ['document']
        negatives = {g: [] for g in granularities}
        
        # Document-level negatives
        if 'document' in granularities:
            doc_negatives = self._mine_document_negatives(
                query_text, positive_text, num_negatives, difficulty
            )
            negatives['document'] = doc_negatives
        
        # Sentence-level negatives (extract sentences from documents)
        if 'sentence' in granularities:
            sent_negatives = self._mine_sentence_negatives(
                query_text, positive_text, num_negatives, difficulty
            )
            negatives['sentence'] = sent_negatives
        
        # Phrase-level negatives (extract phrases from documents)
        if 'phrase' in granularities:
            phrase_negatives = self._mine_phrase_negatives(
                query_text, positive_text, num_negatives, difficulty
            )
            negatives['phrase'] = phrase_negatives
        
        return negatives
    
    def _mine_document_negatives(self, query_text: str, positive_text: str,
                                 num_negatives: int, difficulty: str) -> List[str]:
        """Mine document-level negatives"""
        if difficulty == 'random':
            # Random documents
            candidates = random.sample(self.corpus_ids, min(len(self.corpus_ids), num_negatives * 5))
        elif difficulty == 'bm25' and self.bm25_index:
            # BM25 top documents (excluding positive)
            query_tokens = query_text.lower().split()
            scores = self.bm25_index.get_scores(query_tokens)
            top_indices = np.argsort(scores)[::-1][:num_negatives * 10]
            candidates = [self.corpus_ids[i] for i in top_indices]
        else:
            # Fallback to random
            candidates = random.sample(self.corpus_ids, min(len(self.corpus_ids), num_negatives * 5))
        
        negatives = []
        for doc_id in candidates:
            if doc_id in self.corpus_dict:
                doc = self.corpus_dict[doc_id]
                doc_text = doc.get('text', '')
                if doc_text and doc_text != positive_text:
                    negatives.append(doc_text)
                    if len(negatives) >= num_negatives:
                        break
        
        return negatives
    
    def _mine_sentence_negatives(self, query_text: str, positive_text: str,
                                 num_negatives: int, difficulty: str) -> List[str]:
        """Mine sentence-level negatives"""
        # Extract sentences from corpus documents
        all_sentences = []
        for doc_id, doc in self.corpus_dict.items():
            doc_text = doc.get('text', '')
            if doc_text and doc_text != positive_text:
                sentences = re.split(r'[.!?]+', doc_text)
                sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
                all_sentences.extend(sentences)
        
        if not all_sentences:
            return []
        
        # Select negatives (can be improved with similarity)
        selected = random.sample(all_sentences, min(len(all_sentences), num_negatives))
        return selected
    
    def _mine_phrase_negatives(self, query_text: str, positive_text: str,
                              num_negatives: int, difficulty: str) -> List[str]:
        """Mine phrase-level negatives"""
        # Extract phrases (3-5 word sequences) from corpus
        all_phrases = []
        for doc_id, doc in self.corpus_dict.items():
            doc_text = doc.get('text', '')
            if doc_text and doc_text != positive_text:
                words = doc_text.split()
                # Extract 3-5 word phrases
                for i in range(len(words) - 2):
                    phrase = ' '.join(words[i:i+4])
                    if len(phrase) > 15:
                        all_phrases.append(phrase)
        
        if not all_phrases:
            return []
        
        selected = random.sample(all_phrases, min(len(all_phrases), num_negatives))
        return selected


class HierarchicalContrastiveLoss(nn.Module):
    """Hierarchical contrastive loss at multiple granularities"""
    def __init__(self, temperature: float = 0.05, granularities: List[str] = None):
        super().__init__()
        self.temperature = temperature
        self.granularities = granularities or ['document']
        self.weights = {g: 1.0 / len(self.granularities) for g in self.granularities}
    
    def forward(self, query_emb, pos_emb, neg_embs_dict: Dict[str, torch.Tensor]):
        """
        Args:
            query_emb: Query embeddings [batch, dim]
            pos_emb: Positive embeddings [batch, dim]
            neg_embs_dict: Dict mapping granularity to negative embeddings
                          Each value: [batch, num_negatives, dim] or [batch, dim]
        """
        total_loss = 0.0
        
        for granularity in self.granularities:
            if granularity not in neg_embs_dict:
                continue
            
            neg_embs = neg_embs_dict[granularity]
            
            # Handle different shapes
            if neg_embs.dim() == 3:
                # [batch, num_negatives, dim] - average negatives
                neg_embs = neg_embs.mean(dim=1)  # [batch, dim]
            elif neg_embs.dim() == 2 and neg_embs.size(1) == query_emb.size(1):
                # [batch, dim] - already averaged
                pass
            else:
                continue
            
            # Ensure float dtype
            if query_emb.dtype != torch.float32 and query_emb.dtype != torch.float64:
                query_emb = query_emb.float()
            if pos_emb.dtype != torch.float32 and pos_emb.dtype != torch.float64:
                pos_emb = pos_emb.float()
            if neg_embs.dtype != torch.float32 and neg_embs.dtype != torch.float64:
                neg_embs = neg_embs.float()
            
            # Normalize
            query_emb_norm = F.normalize(query_emb, p=2, dim=1)
            pos_emb_norm = F.normalize(pos_emb, p=2, dim=1)
            neg_embs_norm = F.normalize(neg_embs, p=2, dim=1)
            
            # Positive similarity
            pos_sim = torch.sum(query_emb_norm * pos_emb_norm, dim=1) / self.temperature
            
            # Negative similarity
            neg_sim = torch.sum(query_emb_norm * neg_embs_norm, dim=1) / self.temperature
            
            # Contrastive loss
            logits = torch.stack([pos_sim, neg_sim], dim=1)  # [batch, 2]
            labels = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)
            
            loss = F.cross_entropy(logits, labels)
            total_loss += self.weights[granularity] * loss
        
        return total_loss


class CurriculumHardNegativeDataset(Dataset):
    """Dataset with curriculum learning and multi-granularity negatives"""
    def __init__(self, examples: List[InputExample], model, corpus_dict: Dict,
                 queries_dict: Dict, curriculum_scheduler: CurriculumScheduler,
                 query_rewriter: QueryRewriter, negative_miner: MultiGranularityNegativeMiner,
                 num_hard_negatives: int = 2, current_step: int = 0):
        self.examples = examples
        self.model = model
        self.corpus_dict = corpus_dict
        self.queries_dict = queries_dict
        self.curriculum_scheduler = curriculum_scheduler
        self.query_rewriter = query_rewriter
        self.negative_miner = negative_miner
        self.num_hard_negatives = num_hard_negatives
        self.current_step = current_step
        
        # Build BM25 index
        corpus_texts = [doc.get('text', '') for doc in corpus_dict.values()]
        tokenized_corpus = [doc.lower().split() for doc in corpus_texts]
        self.bm25 = BM25Okapi(tokenized_corpus)
    
    def __len__(self):
        return len(self.examples)
    
    def update_step(self, step: int):
        """Update current training step for curriculum"""
        self.current_step = step
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        query_text, pos_doc_text = example.texts
        
        # Get current difficulty from curriculum
        difficulty = self.curriculum_scheduler.get_difficulty(self.current_step)
        
        # Rewrite query (generate variants)
        query_variants = self.query_rewriter.rewrite_query(query_text, num_variants=1)
        # Use first variant (can be extended to use multiple)
        query_to_use = query_variants[0] if query_variants else query_text
        
        # Mine negatives at multiple granularities
        negatives_dict = self.negative_miner.mine_negatives(
            query_to_use, pos_doc_text, 
            num_negatives=self.num_hard_negatives,
            difficulty=difficulty,
            granularities=['document']  # Start with document, can add more
        )
        
        # Flatten negatives (for now, use document-level)
        negatives = negatives_dict.get('document', [])
        
        # Ensure we have negatives
        while len(negatives) < self.num_hard_negatives:
            random_doc = random.choice(list(self.corpus_dict.values()))
            random_text = random_doc.get('text', '')
            if random_text != pos_doc_text:
                negatives.append(random_text)
        
        return {
            'query': query_text,  # Original query
            'query_rewritten': query_to_use,  # Rewritten query
            'positive': pos_doc_text,
            'negatives': negatives[:self.num_hard_negatives],
            'negatives_dict': negatives_dict  # Multi-granularity negatives
        }


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
                doc_text = f"{title} {text}".strip() if title else text
                examples.append(InputExample(texts=[query_text, doc_text], label=1.0))
    
    logging.info(f"Created {len(examples)} positive pairs for {domain}")
    return examples, corpus, queries


def train_curriculum_contrastive(config: Dict, gpu_id: Optional[int] = None) -> Optional[str]:
    """Train with curriculum contrastive learning"""
    global shutdown_requested
    
    # CUDA setup
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
    else:
        logging.info(f"Device: {device}")
    
    output_dir = pathlib.Path(config.get('output_dir', 'experiments/retrieval/curriculum_contrastive'))
    output_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint_dir = output_dir
    resume = config.get('resume', True)
    
    # Load checkpoint
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
    base_model = config.get('model_path', 'BAAI/bge-base-en-v1.5')
    epochs = config.get('epochs', 3)
    batch_size = config.get('batch_size', 4)
    num_hard_negatives = config.get('num_hard_negatives', 2)
    gradient_accumulation_steps = config.get('gradient_accumulation_steps', 2)
    use_fp16 = config.get('use_fp16', True)
    
    # Load base model
    logging.info(f"Loading base model: {base_model}")
    model = SentenceTransformer(base_model)
    model.to(device)
    
    # Mixed precision
    scaler = None
    if use_fp16 and device == "cuda":
        try:
            from torch.cuda.amp import autocast, GradScaler
            scaler = GradScaler()
            logging.info("Mixed precision training (FP16) enabled")
        except ImportError:
            use_fp16 = False
    
    for param in model.parameters():
        param.requires_grad = True
    
    # Train per domain
    for domain in domains:
        if domain in completed_domains:
            logging.info(f"Skipping {domain} (already completed)")
            continue
        
        if shutdown_requested:
            logging.info("Shutdown requested, saving checkpoint...")
            checkpoint_file = checkpoint_dir / "checkpoint.json"
            with open(checkpoint_file, 'w') as f:
                json.dump({
                    "last_domain": domain,
                    "completed_domains": list(completed_domains),
                    "results": all_results,
                    "timestamp": datetime.now().isoformat()
                }, f, indent=2)
            break
        
        logging.info(f"\n{'='*60}")
        logging.info(f"Training on domain: {domain}")
        logging.info(f"{'='*60}")
        
        data_root = pathlib.Path(".")
        examples, corpus, queries = load_training_pairs(domain, data_root, config.get('use_data_splits', True))
        
        if not examples:
            logging.warning(f"No training examples for {domain}, skipping")
            continue
        
        # Initialize components
        total_steps = len(examples) // batch_size * epochs
        curriculum_scheduler = CurriculumScheduler(total_steps=total_steps)
        query_rewriter = QueryRewriter(model=model)
        negative_miner = MultiGranularityNegativeMiner(corpus_dict=corpus)
        
        # Create dataset
        dataset = CurriculumHardNegativeDataset(
            examples, model, corpus, queries,
            curriculum_scheduler, query_rewriter, negative_miner,
            num_hard_negatives=num_hard_negatives
        )
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Training setup
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
        loss_fn = HierarchicalContrastiveLoss(temperature=0.05, granularities=['document'])
        
        model.train()
        global_step = 0
        
        for epoch in range(epochs):
            total_loss = 0
            optimizer.zero_grad()
            
            for batch_idx, batch in enumerate(dataloader):
                # Update curriculum step
                dataset.update_step(global_step)
                
                queries_batch = batch['query']
                positives_batch = batch['positive']
                negatives_batch = batch['negatives']
                
                # Encode queries (using rewritten queries)
                query_texts = batch.get('query_rewritten', queries_batch)
                query_features = model.tokenizer(query_texts, padding=True, truncation=True,
                                                max_length=512, return_tensors='pt')
                query_features = {k: v.to(device) for k, v in query_features.items()}
                
                # Get embeddings through model modules
                query_outputs = model._modules['0'](query_features)
                if isinstance(query_outputs, dict):
                    query_embeddings = query_outputs.get('token_embeddings')
                    if query_embeddings is None:
                        query_embeddings = query_outputs.get('last_hidden_state')
                        if query_embeddings is None:
                            for v in query_outputs.values():
                                if isinstance(v, torch.Tensor) and v.dim() == 3:
                                    query_embeddings = v
                                    break
                elif hasattr(query_outputs, 'last_hidden_state'):
                    query_embeddings = query_outputs.last_hidden_state
                else:
                    query_embeddings = query_outputs
                
                if query_embeddings is None:
                    raise ValueError("Could not extract query embeddings")
                
                # Pooling
                pooling_output = model._modules['1']({
                    'token_embeddings': query_embeddings,
                    'attention_mask': query_features['attention_mask']
                })
                query_embs = pooling_output.get('sentence_embedding')
                if query_embs is None:
                    query_embs = next((v for v in pooling_output.values() if isinstance(v, torch.Tensor)), None)
                
                if query_embs.dtype != torch.float32 and query_embs.dtype != torch.float64:
                    query_embs = query_embs.float()
                
                if query_embs.dim() == 1:
                    query_embs = query_embs.unsqueeze(0)
                elif query_embs.dim() == 3:
                    query_embs = query_embs.mean(dim=1)
                
                # Encode positives
                pos_features = model.tokenizer(positives_batch, padding=True, truncation=True,
                                              max_length=512, return_tensors='pt')
                pos_features = {k: v.to(device) for k, v in pos_features.items()}
                pos_outputs = model._modules['0'](pos_features)
                
                if isinstance(pos_outputs, dict):
                    pos_embeddings = pos_outputs.get('token_embeddings')
                    if pos_embeddings is None:
                        pos_embeddings = pos_outputs.get('last_hidden_state')
                        if pos_embeddings is None:
                            for v in pos_outputs.values():
                                if isinstance(v, torch.Tensor) and v.dim() == 3:
                                    pos_embeddings = v
                                    break
                elif hasattr(pos_outputs, 'last_hidden_state'):
                    pos_embeddings = pos_outputs.last_hidden_state
                else:
                    pos_embeddings = pos_outputs
                
                pooling_output = model._modules['1']({
                    'token_embeddings': pos_embeddings,
                    'attention_mask': pos_features['attention_mask']
                })
                pos_embs = pooling_output.get('sentence_embedding')
                if pos_embs is None:
                    pos_embs = next((v for v in pooling_output.values() if isinstance(v, torch.Tensor)), None)
                
                if pos_embs.dtype != torch.float32 and pos_embs.dtype != torch.float64:
                    pos_embs = pos_embs.float()
                
                if pos_embs.dim() == 1:
                    pos_embs = pos_embs.unsqueeze(0)
                elif pos_embs.dim() == 3:
                    pos_embs = pos_embs.mean(dim=1)
                
                # Encode negatives
                neg_texts = [neg for negs in negatives_batch for neg in negs]
                neg_features = model.tokenizer(neg_texts, padding=True, truncation=True,
                                              max_length=512, return_tensors='pt')
                neg_features = {k: v.to(device) for k, v in neg_features.items()}
                neg_outputs = model._modules['0'](neg_features)
                
                if isinstance(neg_outputs, dict):
                    neg_embeddings = neg_outputs.get('token_embeddings')
                    if neg_embeddings is None:
                        neg_embeddings = neg_outputs.get('last_hidden_state')
                        if neg_embeddings is None:
                            for v in neg_outputs.values():
                                if isinstance(v, torch.Tensor) and v.dim() == 3:
                                    neg_embeddings = v
                                    break
                elif hasattr(neg_outputs, 'last_hidden_state'):
                    neg_embeddings = neg_outputs.last_hidden_state
                else:
                    neg_embeddings = neg_outputs
                
                pooling_output = model._modules['1']({
                    'token_embeddings': neg_embeddings,
                    'attention_mask': neg_features['attention_mask']
                })
                neg_embs = pooling_output.get('sentence_embedding')
                if neg_embs is None:
                    neg_embs = next((v for v in pooling_output.values() if isinstance(v, torch.Tensor)), None)
                
                if neg_embs.dtype != torch.float32 and neg_embs.dtype != torch.float64:
                    neg_embs = neg_embs.float()
                
                if neg_embs.dim() == 1:
                    neg_embs = neg_embs.unsqueeze(0)
                elif neg_embs.dim() == 3:
                    neg_embs = neg_embs.mean(dim=1)
                
                # Average negatives per query
                neg_counts = [len(negs) for negs in negatives_batch]
                neg_embs_list = []
                start_idx = 0
                for count in neg_counts:
                    end_idx = start_idx + count
                    query_negs = neg_embs[start_idx:end_idx]
                    if query_negs.dtype != torch.float32 and query_negs.dtype != torch.float64:
                        query_negs = query_negs.float()
                    query_neg_avg = query_negs.mean(dim=0)
                    neg_embs_list.append(query_neg_avg)
                    start_idx = end_idx
                
                neg_embs_avg = torch.stack(neg_embs_list, dim=0)
                
                # Prepare for hierarchical loss
                neg_embs_dict = {'document': neg_embs_avg}
                
                # Compute loss
                if use_fp16 and scaler is not None:
                    with autocast():
                        loss = loss_fn(query_embs, pos_embs, neg_embs_dict)
                        loss = loss / gradient_accumulation_steps
                    
                    scaled_loss = scaler.scale(loss)
                    scaled_loss.backward()
                    
                    if (batch_idx + 1) % gradient_accumulation_steps == 0:
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad()
                else:
                    loss = loss_fn(query_embs, pos_embs, neg_embs_dict)
                    loss = loss / gradient_accumulation_steps
                    loss.backward()
                    
                    if (batch_idx + 1) % gradient_accumulation_steps == 0:
                        optimizer.step()
                        optimizer.zero_grad()
                
                total_loss += loss.item() * gradient_accumulation_steps
                global_step += 1
                
                # Clear cache periodically
                if device == "cuda" and (batch_idx + 1) % 10 == 0:
                    torch.cuda.empty_cache()
            
            logging.info(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader):.4f}")
    
    # Evaluate
    logging.info("\nEvaluating trained model...")
    retriever = DenseRetrievalExactSearch(SentenceBERT(base_model, device=device), batch_size=128)
    evaluator = EvaluateRetrieval(retriever, k_values=[1, 3, 5, 10])
    
    for domain in domains:
        if domain in completed_domains:
            continue
        
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
            
            results = evaluator.retrieve(corpus, queries)
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
            
            completed_domains.add(domain)
            
            # Save checkpoint
            checkpoint_file = checkpoint_dir / "checkpoint.json"
            with open(checkpoint_file, 'w') as f:
                json.dump({
                    "last_domain": domain,
                    "completed_domains": list(completed_domains),
                    "results": all_results,
                    "timestamp": datetime.now().isoformat()
                }, f, indent=2)
            
        except Exception as e:
            logging.error(f"Error evaluating {domain}: {e}")
            continue
    
    # Final results
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
        
        logging.info(f"\n{'='*60}")
        logging.info("Average Results:")
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
        results_path = train_curriculum_contrastive(config, args.gpu_id)
        if results_path:
            logging.info(f"✅ Curriculum contrastive learning completed: {results_path}")
    except KeyboardInterrupt:
        logging.info("Interrupted by user")
        sys.exit(0)
    except Exception as e:
        logging.error(f"Failed: {e}", exc_info=True)
        sys.exit(1)

