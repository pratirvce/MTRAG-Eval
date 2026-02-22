# Retrieval Performance Improvement Suggestions

## Current Performance Analysis

**Your Fine-Tuned Model:**
- Recall@5: **0.2340** (vs baseline 0.30) ❌ -22% worse
- Recall@10: **0.3297** (vs baseline 0.38) ❌ -13% worse
- nDCG@5: **0.2070** (vs baseline 0.27) ❌ -23% worse
- nDCG@10: **0.2465** (vs baseline 0.30) ❌ -18% worse

**Domain Breakdown:**
- **clapnq**: R@10=0.4023, nDCG@10=0.2992 ✅ (best performing)
- **govt**: R@10=0.3892, nDCG@10=0.2895 ✅ (good)
- **cloud**: R@10=0.3056, nDCG@10=0.2305 ⚠️ (below baseline)
- **fiqa**: R@10=0.2214, nDCG@10=0.1666 ❌ (worst performing)

**Key Issue:** The fine-dicating the fine-tuning processtuned model is performing **worse** than the pre-trained baseline, in needs improvement.

---

## 🎯 Priority 1: Training Configuration Improvements

### 1.1 Increase Training Epochs
**Current:** 1 epoch  
**Recommended:** 3-5 epochs

**Why:** 1 epoch is insufficient for the model to learn domain-specific patterns. The model is likely underfitting.

**Implementation:**
```python
EPOCHS = 3  # Start with 3, increase to 5 if needed
```

**Expected Impact:** +5-10% improvement in Recall@10

---

### 1.2 Use Proper Train/Val/Test Split
**Current:** Training on `dev.tsv` (validation set)  
**Recommended:** Use train split for training, val for validation, test for evaluation

**Why:** You're training on the validation set, which can lead to overfitting and poor generalization.

**Implementation:**
```python
# In train_finetuned_bge.py, modify load_mtrag_examples():
qrels_file = data_root / "human" / "retrieval_tasks" / domain / "qrels" / "dev.tsv"
# Change to:
qrels_file = data_root / "data_splits" / "retrieval_tasks" / domain / "train" / "qrels" / "dev.tsv"
```

**Expected Impact:** +3-7% improvement

---

### 1.3 Add Validation During Training
**Current:** No validation monitoring  
**Recommended:** Add validation evaluation during training

**Why:** Monitor overfitting and select best checkpoint.

**Implementation:**
```python
# Add validation set loading
val_examples = load_validation_examples(domain)

# Add evaluator
from sentence_transformers.evaluation import InformationRetrievalEvaluator
evaluator = InformationRetrievalEvaluator(
    queries=val_queries,
    corpus=val_corpus,
    relevant_docs=val_qrels,
    show_progress_bar=True
)

# Add to model.fit()
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=EPOCHS,
    warmup_steps=WARMUP_STEPS,
    evaluator=evaluator,
    evaluation_steps=500,  # Evaluate every 500 steps
    output_path=NEW_MODEL_NAME,
    save_best_model=True,  # Save best model based on validation
    checkpoint_save_steps=1000,
    checkpoint_path=f"{NEW_MODEL_NAME}-checkpoints"
)
```

**Expected Impact:** +2-5% improvement

---

### 1.4 Optimize Learning Rate
**Current:** Default (likely 2e-5)  
**Recommended:** Learning rate schedule with warmup

**Why:** Better learning rate can significantly improve convergence.

**Implementation:**
```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer(BASE_MODEL_NAME)

# Add learning rate to fit()
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=EPOCHS,
    warmup_steps=WARMUP_STEPS,
    optimizer_params={'lr': 2e-5},  # Try: 1e-5, 2e-5, 5e-5
    output_path=NEW_MODEL_NAME
)
```

**Expected Impact:** +3-8% improvement

---

### 1.5 Increase Batch Size
**Current:** 16  
**Recommended:** 32 or 64 (if GPU memory allows)

**Why:** Larger batches provide more stable gradients and better negatives in contrastive learning.

**Implementation:**
```python
TRAIN_BATCH_SIZE = 32  # Increase if GPU memory allows
# Or use gradient accumulation:
# gradient_accumulation_steps = 2  # Effective batch size = 16 * 2 = 32
```

**Expected Impact:** +2-5% improvement

---

## 🎯 Priority 2: Advanced Training Techniques

### 2.1 Hard Negative Mining
**Current:** Random negatives from batch  
**Recommended:** Mine hard negatives (similar but incorrect passages)

**Why:** Hard negatives help the model learn fine-grained distinctions.

**Implementation:**
```python
from sentence_transformers.losses import MultipleNegativesRankingLoss
from sentence_transformers import InputExample

# Use CosineSimilarityLoss with hard negatives
# Or implement hard negative mining:
def mine_hard_negatives(model, query, positive_passage, corpus, top_k=10):
    """Find top-k similar but incorrect passages"""
    query_emb = model.encode(query)
    corpus_embs = model.encode([doc['text'] for doc in corpus])
    similarities = cosine_similarity([query_emb], corpus_embs)[0]
    
    # Get top-k similar (excluding positive)
    top_indices = similarities.argsort()[-top_k:][::-1]
    hard_negatives = [corpus[i] for i in top_indices if corpus[i] != positive_passage]
    return hard_negatives[:5]  # Return top 5 hard negatives
```

**Expected Impact:** +5-10% improvement

---

### 2.2 Use Different Loss Functions
**Current:** MultipleNegativesRankingLoss  
**Recommended:** Try CosineSimilarityLoss or TripletLoss

**Why:** Different losses may work better for your specific data distribution.

**Implementation:**
```python
# Option 1: CosineSimilarityLoss (for explicit scoring)
from sentence_transformers.losses import CosineSimilarityLoss
train_loss = CosineSimilarityLoss(model=model)

# Option 2: TripletLoss (for explicit negatives)
from sentence_transformers.losses import TripletLoss
train_loss = TripletLoss(model=model, distance_metric=TripletDistanceMetric.COSINE)

# Option 3: MarginMSELoss (for ranking)
from sentence_transformers.losses import MarginMSELoss
train_loss = MarginMSELoss(model=model)
```

**Expected Impact:** +2-7% improvement

---

### 2.3 Domain-Specific Fine-Tuning
**Current:** Training on all domains together  
**Recommended:** Fine-tune separate models per domain, or use domain-adaptive training

**Why:** Different domains may benefit from specialized models.

**Implementation:**
```python
# Option 1: Separate models per domain
for domain in MTRAG_DOMAINS:
    domain_examples = load_mtrag_examples(domain)
    model = SentenceTransformer(BASE_MODEL_NAME)
    # Train domain-specific model
    model.fit(...)
    model.save(f"./bge-finetuned-{domain}")

# Option 2: Domain-adaptive (train on all, then fine-tune per domain)
# First train on all domains, then fine-tune each domain model
```

**Expected Impact:** +3-8% improvement per domain

---

### 2.4 Data Augmentation
**Current:** Only using exact query-passage pairs  
**Recommended:** Augment training data

**Why:** More diverse training data improves generalization.

**Implementation:**
```python
# Option 1: Query paraphrasing
def augment_queries(query):
    # Use a paraphrasing model or simple techniques
    paraphrases = [
        query,  # Original
        query.lower(),  # Lowercase
        query.replace("?", ""),  # Remove punctuation
        # Add more variations
    ]
    return paraphrases

# Option 2: Passage chunking variations
def augment_passages(passage, title):
    variations = [
        f"{title} {passage}",  # With title
        passage,  # Without title
        passage[:512],  # Truncated
    ]
    return variations
```

**Expected Impact:** +2-5% improvement

---

## 🎯 Priority 3: Retrieval Strategy Improvements

### 3.1 Hybrid Retrieval (Dense + Sparse)
**Current:** Dense retrieval only  
**Recommended:** Combine dense and sparse (BM25) retrieval

**Why:** Hybrid approaches often outperform single methods.

**Implementation:**
```python
from beir.retrieval.search.dense import DenseRetrievalExactSearch
from beir.retrieval.search.sparse import BM25Search
from beir.retrieval.search.hybrid import HybridSearch

# Dense retriever
dense_retriever = DenseRetrievalExactSearch(model, batch_size=128)

# Sparse retriever (BM25)
sparse_retriever = BM25Search(index_name="beir")

# Hybrid retriever
hybrid_retriever = HybridSearch(
    dense_retriever=dense_retriever,
    sparse_retriever=sparse_retriever,
    alpha=0.5  # Weight: 0.5 dense + 0.5 sparse
)

results = hybrid_retriever.search(corpus, queries, top_k=10)
```

**Expected Impact:** +5-15% improvement

---

### 3.2 Reranking
**Current:** Using top-K directly  
**Recommended:** Rerank top results with a cross-encoder

**Why:** Cross-encoders are more accurate but slower. Use for reranking top results.

**Implementation:**
```python
from sentence_transformers import CrossEncoder

# Load a cross-encoder model
reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

# Retrieve top 100 with bi-encoder
top_100_results = dense_retriever.retrieve(corpus, queries, top_k=100)

# Rerank top 100 to get top 10
reranked_results = {}
for query_id, docs in top_100_results.items():
    query_text = queries[query_id]
    pairs = [(query_text, corpus[doc_id]['text']) for doc_id in docs.keys()]
    scores = reranker.predict(pairs)
    
    # Sort by reranking scores
    sorted_docs = sorted(zip(docs.keys(), scores), key=lambda x: x[1], reverse=True)
    reranked_results[query_id] = {doc_id: score for doc_id, score in sorted_docs[:10]}
```

**Expected Impact:** +10-20% improvement

---

### 3.3 Query Expansion
**Current:** Using queries as-is  
**Recommended:** Expand queries with synonyms or related terms

**Why:** Query expansion can help match relevant documents that use different terminology.

**Implementation:**
```python
# Option 1: Use a query expansion model
from transformers import T5ForConditionalGeneration, T5Tokenizer

def expand_query(query):
    # Use T5 or similar model to expand query
    expanded = f"expand query: {query}"
    # Generate expansion
    return expanded

# Option 2: Use word embeddings for synonym expansion
from sentence_transformers import SentenceTransformer
synonym_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

def expand_with_synonyms(query):
    # Find similar queries or add synonyms
    return query
```

**Expected Impact:** +3-8% improvement

---

### 3.4 Multi-Vector Retrieval
**Current:** Single vector per document  
**Recommended:** Use multiple vectors per document (e.g., sentence-level)

**Why:** Documents may have multiple relevant sections.

**Implementation:**
```python
# Split documents into sentences
def split_into_sentences(text):
    sentences = text.split('. ')
    return [s.strip() for s in sentences if len(s.strip()) > 20]

# Encode each sentence separately
def encode_document_sentences(model, doc):
    sentences = split_into_sentences(doc['text'])
    sentence_embeddings = model.encode(sentences)
    return sentence_embeddings

# During retrieval, use max similarity across all sentences
def retrieve_with_sentences(query_emb, doc_sentence_embs):
    similarities = [cosine_similarity([query_emb], [sent_emb])[0][0] 
                    for sent_emb in doc_sentence_embs]
    return max(similarities)
```

**Expected Impact:** +2-6% improvement

---

## 🎯 Priority 4: Model Architecture Improvements

### 4.1 Use Larger Base Model
**Current:** `bge-base-en-v1.5` (110M parameters)  
**Recommended:** `bge-large-en-v1.5` (335M parameters)

**Why:** Larger models typically perform better but are slower.

**Implementation:**
```python
BASE_MODEL_NAME = "BAAI/bge-large-en-v1.5"  # Instead of bge-base
```

**Expected Impact:** +5-12% improvement (but slower inference)

---

### 4.2 Use Domain-Specific Pre-trained Models
**Current:** General English model  
**Recommended:** Start from domain-specific models if available

**Why:** Starting closer to your domain can improve fine-tuning.

**Examples:**
- Scientific: `allenai/specter2`
- Medical: `pritamdeka/S-PubMedBert-MS-MARCO`
- Legal: `nlpaueb/legal-bert-base-uncased`

**Expected Impact:** +3-10% improvement

---

## 🎯 Priority 5: Data Quality Improvements

### 5.1 Better Query Preprocessing
**Current:** Using queries as-is  
**Recommended:** Clean and normalize queries

**Implementation:**
```python
def preprocess_query(query):
    # Remove conversation markers if present
    query = query.replace("|user|:", "").replace("|agent|:", "")
    query = query.strip()
    
    # Normalize whitespace
    query = " ".join(query.split())
    
    # Handle special characters
    # Add domain-specific preprocessing
    
    return query
```

**Expected Impact:** +1-3% improvement

---

### 5.2 Better Passage Formatting
**Current:** `title + " " + text`  
**Recommended:** Use structured formatting

**Implementation:**
```python
def format_passage(doc):
    title = doc.get("title", "")
    text = doc.get("text", "")
    
    # Option 1: Structured format
    if title:
        formatted = f"Title: {title}\n\n{text}"
    else:
        formatted = text
    
    # Option 2: Add domain-specific formatting
    # For technical docs: "Documentation: {title}\n{text}"
    
    return formatted
```

**Expected Impact:** +1-4% improvement

---

### 5.3 Filter Low-Quality Training Examples
**Current:** Using all qrels  
**Recommended:** Filter out low-confidence or noisy examples

**Implementation:**
```python
# Only use examples with high confidence
for query_id, doc_infos in qrels.items():
    for doc_id, score in doc_infos.items():
        if score > 0:  # Current: any positive
            # Add filtering:
            # - Check if query and passage are too short
            # - Check if they're too similar (might be duplicates)
            # - Check annotation quality
            if is_high_quality(query_text, doc_text):
                domain_examples.append(...)
```

**Expected Impact:** +2-5% improvement

---

## 📊 Recommended Implementation Order

### Phase 1: Quick Wins (1-2 days)
1. ✅ Increase epochs to 3-5
2. ✅ Use proper train/val/test split
3. ✅ Add validation during training
4. ✅ Increase batch size to 32

**Expected Improvement:** +10-20% overall

### Phase 2: Training Improvements (3-5 days)
5. ✅ Optimize learning rate
6. ✅ Implement hard negative mining
7. ✅ Try different loss functions
8. ✅ Add data augmentation

**Expected Improvement:** +10-15% additional

### Phase 3: Retrieval Strategy (2-3 days)
9. ✅ Implement hybrid retrieval (dense + BM25)
10. ✅ Add reranking with cross-encoder
11. ✅ Query expansion

**Expected Improvement:** +15-25% additional

### Phase 4: Advanced (1-2 weeks)
12. ✅ Domain-specific models
13. ✅ Larger base model
14. ✅ Multi-vector retrieval

**Expected Improvement:** +10-20% additional

---

## 🧪 Testing Strategy

After each improvement:
1. **Train** the model with the change
2. **Evaluate** on validation set
3. **Compare** with previous results
4. **Keep** if improvement > 1%, otherwise revert

**Example evaluation script:**
```python
# Quick evaluation on validation set
python evaluate_finetuned_bge.py --split val
```

---

## 📈 Expected Final Performance

With all improvements implemented:
- **Recall@10:** 0.45-0.55 (vs current 0.33, baseline 0.38)
- **nDCG@10:** 0.35-0.45 (vs current 0.25, baseline 0.30)

**Target:** Beat baseline by 15-30%

---

## 🔧 Quick Start: Minimal Changes for Maximum Impact

If you can only make 3 changes, do these:

1. **Increase epochs to 3-5** (5 minutes)
2. **Use proper train split** (10 minutes)
3. **Add hybrid retrieval** (1-2 hours)

**Expected:** +15-25% improvement with minimal effort

---

## 📝 Notes

- **Monitor training loss:** Should decrease steadily
- **Watch for overfitting:** Validation metrics should improve, not degrade
- **Domain differences:** Some domains (fiqa) may need special attention
- **Inference speed:** Some improvements (reranking, larger models) will slow down retrieval

Good luck! 🚀

