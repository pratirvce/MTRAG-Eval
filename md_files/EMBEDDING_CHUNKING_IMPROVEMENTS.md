# Embedding and Chunking Improvements for Better nDCG@10 Retrieval

## Current Strategy Analysis

### Current Chunking (from MTRAG README)
- **Chunk Size:** 512 tokens
- **Overlap:** 100 tokens (fixed)
- **Method:** Fixed-size token-based chunking
- **Issue:** May split sentences/paragraphs mid-way, losing semantic coherence

### Current Embedding (from codebase)
- **Model:** BAAI/bge-base-en-v1.5
- **Dimensions:** 768
- **Encoding:** Single-vector (mean pooling)
- **Query-Document:** Symmetric encoding
- **Max Length:** 512 tokens

---

## 🎯 Embedding Improvements (Expected: +0.05-0.15 nDCG@10)

### 1. **Larger Base Models** ⭐⭐⭐⭐⭐ (Highest Impact)
**Current:** BGE-base (110M params, 768 dim)  
**Improvement:** BGE-v2-large (560M params, 1024 dim)

**Expected Gain:** +0.05-0.08 nDCG@10

**Implementation:**
```python
# Replace in all training scripts
base_model = 'BAAI/bge-large-en-v1.5'  # or 'BAAI/bge-v2-large-en'
# Benefits:
# - Better semantic understanding
# - Higher dimensional embeddings (1024 vs 768)
# - Trained on more data
```

**Trade-offs:**
- ✅ Higher accuracy
- ❌ 5x larger model, slower inference
- ❌ More GPU memory

---

### 2. **Asymmetric Query-Document Encoding** ⭐⭐⭐⭐ (High Impact)
**Current:** Same encoder for queries and documents  
**Improvement:** Separate encoders or query-specific prompts

**Expected Gain:** +0.03-0.06 nDCG@10

**Implementation:**
```python
# BGE models support query/document prompts
model = SentenceTransformer('BAAI/bge-base-en-v1.5')

# Query encoding with prompt
query_emb = model.encode(
    [f"Represent this sentence for searching relevant passages: {query}"],
    prompt_name='query'
)

# Document encoding with prompt
doc_emb = model.encode(
    [f"Represent this sentence for retrieval: {doc}"],
    prompt_name='passage'
)
```

**Why it works:**
- Queries are typically shorter, need different representation
- Documents are longer, need different encoding strategy
- BGE models are trained with this asymmetry

---

### 3. **Multi-Vector Embeddings (ColBERT-style)** ⭐⭐⭐⭐ (High Impact)
**Current:** Single vector per document (mean pooling)  
**Improvement:** Multiple token-level vectors

**Expected Gain:** +0.04-0.08 nDCG@10

**Implementation:**
```python
# Instead of mean pooling, use token-level embeddings
# For each document, get [num_tokens, hidden_dim] embeddings
# For query, get [num_query_tokens, hidden_dim] embeddings
# Compute max similarity across all token pairs

class ColBERTRetriever:
    def encode_query(self, query):
        # Return token-level embeddings
        outputs = model._modules['0'](tokenizer(query))
        return outputs.last_hidden_state  # [1, seq_len, hidden_dim]
    
    def encode_document(self, doc):
        outputs = model._modules['0'](tokenizer(doc))
        return outputs.last_hidden_state  # [1, seq_len, hidden_dim]
    
    def score(self, query_emb, doc_emb):
        # MaxSim: max similarity across all token pairs
        # query_emb: [seq_len_q, hidden_dim]
        # doc_emb: [seq_len_d, hidden_dim]
        scores = torch.matmul(query_emb, doc_emb.T)  # [seq_len_q, seq_len_d]
        return scores.max(dim=1)[0].sum()  # Sum of max similarities
```

**Why it works:**
- Captures fine-grained token-level matching
- Better for long documents
- Handles partial relevance better

**Trade-offs:**
- ✅ Better accuracy
- ❌ Larger index size (N tokens × hidden_dim vs 1 × hidden_dim)
- ❌ Slower retrieval (needs max similarity computation)

---

### 4. **Cross-Encoder Reranking** ⭐⭐⭐⭐⭐ (Highest Impact for Reranking)
**Current:** Bi-encoder only (separate query/doc encoding)  
**Improvement:** Add cross-encoder for top-K reranking

**Expected Gain:** +0.08-0.15 nDCG@10 (when used as reranker)

**Implementation:**
```python
from sentence_transformers import CrossEncoder

# Stage 1: Bi-encoder retrieval (fast, get top 100)
bi_encoder = SentenceTransformer('BAAI/bge-base-en-v1.5')
top_k_docs = retrieve_with_bi_encoder(query, top_k=100)

# Stage 2: Cross-encoder reranking (accurate, rerank top 100)
cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
pairs = [(query, doc) for doc in top_k_docs]
scores = cross_encoder.predict(pairs)
reranked = sorted(zip(top_k_docs, scores), key=lambda x: x[1], reverse=True)
```

**Why it works:**
- Cross-encoder sees query and document together
- Better understanding of relevance
- Typically +0.10-0.15 improvement over bi-encoder alone

**Trade-offs:**
- ✅ Much better accuracy
- ❌ Slower (needs to encode query+doc pairs)
- ❌ Can only rerank small sets (top 100-200)

---

### 5. **Fine-Tuning on MTRAG Data** ⭐⭐⭐⭐ (High Impact)
**Current:** Pre-trained BGE-base  
**Improvement:** Fine-tune on MTRAG training data

**Expected Gain:** +0.03-0.06 nDCG@10

**Implementation:**
- Already implemented in `train_finetuned_bge.py`
- Use contrastive learning with hard negatives
- Train for 3-5 epochs with learning rate 2e-5

---

### 6. **Hybrid Dense-Sparse Embeddings** ⭐⭐⭐ (Medium Impact)
**Current:** Dense embeddings only  
**Improvement:** Combine dense + sparse (BM25/ColBERT)

**Expected Gain:** +0.02-0.05 nDCG@10

**Implementation:**
- Already implemented in `train_hybrid_dynamic_fusion.py`
- Dense: BGE embeddings
- Sparse: BM25 or ColBERT
- Learned fusion weights

---

## 🔪 Chunking Improvements (Expected: +0.02-0.08 nDCG@10)

### 1. **Semantic Chunking** ⭐⭐⭐⭐⭐ (Highest Impact)
**Current:** Fixed 512-token chunks with 100-token overlap  
**Improvement:** Sentence/paragraph-aware chunking

**Expected Gain:** +0.03-0.06 nDCG@10

**Implementation:**
```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

def semantic_chunk(text, chunk_size=512, chunk_overlap=100):
    """
    Chunk by sentences/paragraphs, respecting semantic boundaries
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]  # Try to split at paragraphs first
    )
    chunks = splitter.split_text(text)
    return chunks
```

**Why it works:**
- Preserves semantic coherence
- Avoids splitting mid-sentence
- Better for retrieval (complete thoughts)

**Trade-offs:**
- ✅ Better semantic coherence
- ❌ Variable chunk sizes (need padding/truncation)
- ❌ More complex implementation

---

### 2. **Adaptive Chunk Sizes** ⭐⭐⭐ (Medium Impact)
**Current:** Fixed 512 tokens  
**Improvement:** Domain-specific or content-aware chunk sizes

**Expected Gain:** +0.02-0.04 nDCG@10

**Implementation:**
```python
# Different chunk sizes for different domains
CHUNK_SIZES = {
    'clapnq': 512,   # General knowledge
    'fiqa': 256,     # Financial Q&A (shorter answers)
    'govt': 768,     # Government docs (longer context needed)
    'cloud': 512     # Technical docs
}

# Or content-aware: short chunks for lists, long for paragraphs
def adaptive_chunk(text):
    if is_list_format(text):
        chunk_size = 256
    elif is_paragraph(text):
        chunk_size = 768
    else:
        chunk_size = 512
    return chunk(text, chunk_size)
```

---

### 3. **Hierarchical Chunking** ⭐⭐⭐⭐ (High Impact)
**Current:** Single-level chunks  
**Improvement:** Multi-level chunks (sentence → paragraph → section)

**Expected Gain:** +0.03-0.06 nDCG@10

**Implementation:**
```python
def hierarchical_chunk(text):
    """
    Create chunks at multiple granularities
    """
    # Level 1: Sentences (fine-grained)
    sentence_chunks = split_by_sentences(text, max_tokens=128)
    
    # Level 2: Paragraphs (medium-grained)
    para_chunks = split_by_paragraphs(text, max_tokens=512)
    
    # Level 3: Sections (coarse-grained)
    section_chunks = split_by_sections(text, max_tokens=1024)
    
    # Retrieve at all levels, then merge results
    return {
        'sentences': sentence_chunks,
        'paragraphs': para_chunks,
        'sections': section_chunks
    }
```

**Why it works:**
- Captures information at different granularities
- Some queries need fine-grained (specific facts)
- Some queries need coarse-grained (overview)

---

### 4. **Overlap Strategies** ⭐⭐⭐ (Medium Impact)
**Current:** Fixed 100-token overlap  
**Improvement:** Adaptive or semantic overlap

**Expected Gain:** +0.01-0.03 nDCG@10

**Implementation:**
```python
# Strategy 1: Sentence-aware overlap
def sentence_overlap_chunk(text, chunk_size=512, overlap_sentences=2):
    """
    Overlap by complete sentences, not fixed tokens
    """
    sentences = split_sentences(text)
    chunks = []
    for i in range(0, len(sentences), chunk_size - overlap_sentences):
        chunk = sentences[i:i+chunk_size]
        chunks.append(' '.join(chunk))
    return chunks

# Strategy 2: Context-aware overlap
def context_overlap_chunk(text, chunk_size=512):
    """
    Larger overlap for important sections (headings, first/last paragraphs)
    """
    # Detect important sections
    # Use larger overlap (200 tokens) for these
    # Use smaller overlap (50 tokens) for others
    pass
```

---

### 5. **Query-Aware Chunking** ⭐⭐⭐⭐ (High Impact)
**Current:** Static chunks (same for all queries)  
**Improvement:** Re-chunk or select chunks based on query

**Expected Gain:** +0.03-0.06 nDCG@10

**Implementation:**
```python
def query_aware_chunking(query, document, base_chunks):
    """
    For each query, select or re-chunk documents to maximize relevance
    """
    # Strategy 1: Select most relevant chunks
    query_emb = encode_query(query)
    chunk_embs = [encode_doc(chunk) for chunk in base_chunks]
    similarities = [cosine_sim(query_emb, emb) for emb in chunk_embs]
    top_chunks = [chunk for _, chunk in sorted(zip(similarities, base_chunks), reverse=True)[:5]]
    
    # Strategy 2: Re-chunk around relevant sections
    relevant_sections = find_relevant_sections(query, document)
    new_chunks = create_chunks_around_sections(relevant_sections)
    
    return new_chunks
```

**Why it works:**
- Focuses on query-relevant parts
- Reduces noise from irrelevant chunks
- Better precision

**Trade-offs:**
- ✅ Better relevance
- ❌ More expensive (need to process per query)
- ❌ Can't pre-index (need dynamic chunking)

---

## 📊 Combined Strategy (Maximum Impact)

### Recommended Approach for nDCG@10 > 0.90:

1. **Embedding Improvements:**
   - ✅ Use BGE-v2-large (or BGE-large) as base model
   - ✅ Asymmetric query/document encoding with prompts
   - ✅ Fine-tune on MTRAG data with hard negatives
   - ✅ Add cross-encoder reranking (top 100 → top 10)

2. **Chunking Improvements:**
   - ✅ Semantic chunking (sentence/paragraph-aware)
   - ✅ Adaptive chunk sizes per domain
   - ✅ Hierarchical chunking (multi-granularity)

3. **Combined Pipeline:**
   ```
   Stage 1: Semantic chunking (sentence/paragraph-aware)
   Stage 2: BGE-v2-large embeddings (asymmetric encoding)
   Stage 3: Bi-encoder retrieval (top 200)
   Stage 4: Cross-encoder reranking (top 200 → top 10)
   ```

**Expected Combined Gain:** +0.15-0.25 nDCG@10  
**From baseline 0.51 → Target: 0.66-0.76 nDCG@10**

---

## 🚀 Implementation Priority

### Phase 1: Quick Wins (1-2 days)
1. **Asymmetric encoding** - Easy, uses existing BGE models
2. **Semantic chunking** - Medium effort, high impact
3. **BGE-large/v2-large** - Easy, just change model name

**Expected:** +0.08-0.12 nDCG@10

### Phase 2: Medium Effort (3-5 days)
4. **Cross-encoder reranking** - Medium effort, very high impact
5. **Hierarchical chunking** - Medium effort, high impact
6. **Adaptive chunk sizes** - Low effort, medium impact

**Expected:** +0.10-0.15 nDCG@10

### Phase 3: Advanced (1-2 weeks)
7. **Multi-vector embeddings (ColBERT)** - High effort, high impact
8. **Query-aware chunking** - High effort, high impact
9. **Fine-tuned embeddings** - Medium effort, medium impact

**Expected:** +0.08-0.12 nDCG@10

---

## 📝 Code Examples

### Example 1: Asymmetric Encoding
```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('BAAI/bge-base-en-v1.5')

# Query encoding
query = "What are the side effects of lisinopril?"
query_emb = model.encode(
    [f"Represent this sentence for searching relevant passages: {query}"],
    prompt_name='query'
)

# Document encoding
doc = "Lisinopril is an ACE inhibitor..."
doc_emb = model.encode(
    [f"Represent this sentence for retrieval: {doc}"],
    prompt_name='passage'
)
```

### Example 2: Semantic Chunking
```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

def chunk_document(text, chunk_size=512, chunk_overlap=100):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    return splitter.split_text(text)
```

### Example 3: Cross-Encoder Reranking
```python
from sentence_transformers import CrossEncoder

# After bi-encoder retrieval
top_100_docs = bi_encoder_retrieve(query, top_k=100)

# Rerank with cross-encoder
cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
pairs = [(query, doc['text']) for doc in top_100_docs]
scores = cross_encoder.predict(pairs)

# Sort by scores
reranked = sorted(zip(top_100_docs, scores), key=lambda x: x[1], reverse=True)
final_top_10 = [doc for doc, _ in reranked[:10]]
```

---

## 📈 Expected Performance Gains

| Improvement | Expected nDCG@10 Gain | Effort | Priority |
|------------|----------------------|--------|----------|
| BGE-v2-large | +0.05-0.08 | Low | ⭐⭐⭐⭐⭐ |
| Asymmetric encoding | +0.03-0.06 | Low | ⭐⭐⭐⭐ |
| Cross-encoder reranking | +0.08-0.15 | Medium | ⭐⭐⭐⭐⭐ |
| Semantic chunking | +0.03-0.06 | Medium | ⭐⭐⭐⭐⭐ |
| Hierarchical chunking | +0.03-0.06 | Medium | ⭐⭐⭐⭐ |
| Multi-vector (ColBERT) | +0.04-0.08 | High | ⭐⭐⭐⭐ |
| Query-aware chunking | +0.03-0.06 | High | ⭐⭐⭐ |
| Adaptive chunk sizes | +0.02-0.04 | Low | ⭐⭐⭐ |

**Combined Maximum Gain:** +0.31-0.59 nDCG@10  
**From 0.51 baseline → 0.82-1.10 nDCG@10** (theoretical maximum)

---

## ✅ Recommendations for Your Experiments

1. **Immediate (this week):**
   - Switch to BGE-large or BGE-v2-large
   - Add asymmetric encoding with prompts
   - Implement semantic chunking

2. **Short-term (next week):**
   - Add cross-encoder reranking to multi-stage pipeline
   - Implement hierarchical chunking
   - Fine-tune embeddings on MTRAG data

3. **Long-term (next month):**
   - Implement ColBERT-style multi-vector embeddings
   - Query-aware chunking for complex queries
   - Domain-specific chunking strategies

---

**Last Updated:** 2025-12-20  
**Status:** Ready for Implementation

