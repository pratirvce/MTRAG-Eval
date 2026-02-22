# Example: Evaluating Fine-Tuned BGE Model with `evaluate_finetuned_bge.py`

This document shows a concrete example of how `evaluate_finetuned_bge.py` evaluates a fine-tuned BGE (BAAI General Embedding) model for retrieval tasks.

## Overview

The script:
1. Loads a fine-tuned BGE embedding model
2. Uses it to retrieve relevant documents for queries
3. Evaluates retrieval performance using BEIR framework
4. Compares results against pre-trained BGE baseline

---

## Prerequisites

Before running, you need:
1. **Fine-tuned model**: A BGE model fine-tuned using `train_finetuned_bge.py`, saved at `./bge-finetuned-all-domains`
2. **Corpus files**: Document collections in BEIR format
3. **Query files**: Test queries in BEIR format
4. **Qrels files**: Ground truth relevance labels

---

## Input Data Structure

### 1. Corpus File: `corpora/passage_level/cloud.jsonl`

Each line is a JSON object representing a document passage:

```jsonl
{"_id": "ibmcld_00474-7885-8455", "title": "Cloudant Overview", "text": "IBM Cloudant is a fully managed NoSQL document database service. It provides a scalable, distributed database optimized for web and mobile applications. Cloudant supports JSON documents with flexible schemas..."}
{"_id": "ibmcld_00513-7-2197", "title": "Cloudant Schema", "text": "Cloudant supports JSON documents with flexible schemas. You can store any JSON object without pre-defining fields. This makes it ideal for applications with evolving data structures..."}
{"_id": "ibmcld_00526-7-1750", "title": "Flexible Schema", "text": "Cloudant's flexible schema allows storing JSON documents without predefined structure. You can add new fields to documents dynamically without schema migration..."}
{"_id": "ibmcld_99999-0000-0000", "title": "Watson Services", "text": "IBM Watson provides AI services for various applications including natural language processing, computer vision, and machine learning..."}
...
```

**Format:**
- `_id`: Unique document identifier
- `title`: Document title (optional)
- `text`: Document content/passage text

### 2. Query File: `human/retrieval_tasks/cloud/cloud_questions.jsonl`

Each line is a JSON object representing a query:

```jsonl
{"_id": "user_123_1704067200", "text": "Does IBM offer document databases?"}
{"_id": "user_123_1704067300", "text": "So it can store any random JSON object or I need to specify fields in advance?"}
{"_id": "user_456_1704067400", "text": "What if I want to store an image or PDF with a document?"}
...
```

**Format:**
- `_id`: Unique query identifier
- `text`: Query text

### 3. Qrels File: `human/retrieval_tasks/cloud/qrels/dev.tsv`

Ground truth relevance labels (Tab-separated):

```tsv
query-id	corpus-id	score
user_123_1704067200	ibmcld_00474-7885-8455	1
user_123_1704067200	ibmcld_00513-7-2197	1
user_123_1704067300	ibmcld_00526-7-1750	1
user_123_1704067300	ibmcld_00526-2924-4511	1
```

**Format:**
- `query-id`: Query identifier (matches `_id` in queries file)
- `corpus-id`: Document identifier (matches `_id` in corpus file)
- `score`: Relevance score (1 = relevant, 0 = not relevant)

---

## Running the Evaluation

**Command:**
```bash
python evaluate_finetuned_bge.py
```

**Expected Output:**
```
✅ Using NVIDIA GPU (cuda) for acceleration.
Loading model: ./bge-finetuned-all-domains (this is your fine-tuned model)...

--- Running Full FINE-TUNED BGE Baseline Replication ---

--- Processing Domain: clapnq ---
Loading data: clapnq...
Corpus: 183408 docs | Queries: 150 queries
Running retrieval for clapnq (This will take a while)...
Evaluating results...
--- Results for clapnq ---
Recall@10: 0.4234
nDCG@10:   0.3456

--- Processing Domain: fiqa ---
Loading data: fiqa...
Corpus: 49607 docs | Queries: 120 queries
Running retrieval for fiqa (This will take a while)...
Evaluating results...
--- Results for fiqa ---
Recall@10: 0.3891
nDCG@10:   0.3123

--- Processing Domain: govt ---
Loading data: govt...
Corpus: 72422 docs | Queries: 180 queries
Running retrieval for govt (This will take a while)...
Evaluating results...
--- Results for govt ---
Recall@10: 0.4012
nDCG@10:   0.3289

--- Processing Domain: cloud ---
Loading data: cloud...
Corpus: 61022 docs | Queries: 200 queries
Running retrieval for cloud (This will take a while)...
Evaluating results...
--- Results for cloud ---
Recall@10: 0.4123
nDCG@10:   0.3398

--- Finished All Domains: Final Summary ---
```

---

## How It Works: Step-by-Step Example

### Step 1: Model Loading

```python
MODEL_NAME = "./bge-finetuned-all-domains"
model = SentenceBERT(MODEL_NAME, device="cuda")
```

The script loads your fine-tuned BGE model. This model has been trained on MT-RAG domains to better understand domain-specific language.

### Step 2: Setup Retriever

```python
retriever = DenseRetrievalExactSearch(model, batch_size=128)
evaluator = EvaluateRetrieval(retriever, k_values=[5, 10])
```

- **DenseRetrievalExactSearch**: Uses dense embeddings for retrieval (semantic search)
- **batch_size=128**: Processes 128 queries/documents at once
- **k_values=[5, 10]**: Evaluates top 5 and top 10 results

### Step 3: Load Data for Each Domain

For each domain (clapnq, fiqa, govt, cloud):

```python
corpus, queries, qrels = GenericDataLoader(
    corpus_file="corpora/passage_level/cloud.jsonl",
    query_file="human/retrieval_tasks/cloud/cloud_questions.jsonl",
    qrels_file="human/retrieval_tasks/cloud/qrels/dev.tsv"
).load_custom()
```

**Example loaded data:**

**Corpus** (dictionary):
```python
{
    "ibmcld_00474-7885-8455": {
        "title": "Cloudant Overview",
        "text": "IBM Cloudant is a fully managed NoSQL document database..."
    },
    "ibmcld_00513-7-2197": {
        "title": "Cloudant Schema",
        "text": "Cloudant supports JSON documents with flexible schemas..."
    },
    # ... 61,020 more documents
}
```

**Queries** (dictionary):
```python
{
    "user_123_1704067200": "Does IBM offer document databases?",
    "user_123_1704067300": "So it can store any random JSON object or I need to specify fields in advance?",
    # ... 198 more queries
}
```

**Qrels** (nested dictionary):
```python
{
    "user_123_1704067200": {
        "ibmcld_00474-7885-8455": 1,  # Relevant
        "ibmcld_00513-7-2197": 1      # Relevant
    },
    "user_123_1704067300": {
        "ibmcld_00526-7-1750": 1,     # Relevant
        "ibmcld_00526-2924-4511": 1   # Relevant
    }
}
```

### Step 4: Run Retrieval

```python
results = evaluator.retrieve(corpus, queries)
```

**What happens:**
1. **Encode queries**: Convert all queries to embeddings using the BGE model
2. **Encode documents**: Convert all documents to embeddings (cached for efficiency)
3. **Compute similarity**: For each query, compute cosine similarity with all documents
4. **Rank documents**: Sort documents by similarity score (highest first)
5. **Return top-K**: Keep top 10 documents per query

**Example retrieval result for query "Does IBM offer document databases?":**

```python
results["user_123_1704067200"] = {
    "ibmcld_00474-7885-8455": 0.8923,  # High similarity (relevant!)
    "ibmcld_00513-7-2197": 0.8456,     # High similarity (relevant!)
    "ibmcld_99999-0000-0000": 0.5234,  # Lower similarity (not relevant)
    "ibmcld_88888-1111-2222": 0.4123,  # Lower similarity (not relevant)
    "ibmcld_77777-3333-4444": 0.3456,  # Lower similarity (not relevant)
    # ... top 10 documents
}
```

### Step 5: Evaluate Results

```python
ndcg, _map, recall, precision = evaluator.evaluate(qrels, results, k_values=[5, 10])
```

**For each query, compute metrics:**

#### Query: "Does IBM offer document databases?"

**Retrieved (top 10, ranked by score):**
1. `ibmcld_00474-7885-8455` (0.8923) ✓ Relevant (in qrels)
2. `ibmcld_00513-7-2197` (0.8456) ✓ Relevant (in qrels)
3. `ibmcld_99999-0000-0000` (0.5234) ✗ Not relevant
4. `ibmcld_88888-1111-2222` (0.4123) ✗ Not relevant
5. `ibmcld_77777-3333-4444` (0.3456) ✗ Not relevant
6. ... (top 10)

**Relevant documents (from qrels):** 2 total
- `ibmcld_00474-7885-8455`
- `ibmcld_00513-7-2197`

**Metrics for this query:**
- **Recall@5**: 2/2 = 1.0 (both relevant docs in top 5)
- **Recall@10**: 2/2 = 1.0 (both relevant docs in top 10)
- **nDCG@5**: ~0.95 (high because relevant docs are at positions 1 and 2)
- **nDCG@10**: ~0.95 (same, since both are in top 5)

**Average across all queries in domain:**
- **Recall@5**: 0.4123 (average across all queries)
- **Recall@10**: 0.4123 (average across all queries)
- **nDCG@5**: 0.3398 (average across all queries)
- **nDCG@10**: 0.3398 (average across all queries)

### Step 6: Aggregate Across Domains

After processing all 4 domains, compute average metrics:

```python
avg_recall_5 = np.mean([0.4234, 0.3891, 0.4012, 0.4123])  # = 0.4065
avg_recall_10 = np.mean([0.4234, 0.3891, 0.4012, 0.4123])  # = 0.4065
avg_ndcg_5 = np.mean([0.3456, 0.3123, 0.3289, 0.3398])  # = 0.3317
avg_ndcg_10 = np.mean([0.3456, 0.3123, 0.3289, 0.3398])  # = 0.3317
```

### Step 7: Compare with Baseline

The script compares your fine-tuned model against pre-trained BGE baseline from the paper:

```
--- Your Fine-Tuned BGE vs. Pre-Trained BGE (Paper) ---
Metric       | Your Avg.  | Pre-Trained BGE (Paper)
-------------------------------------------------------
Recall@5     | 0.4065     | 0.30
Recall@10    | 0.4065     | 0.38
nDCG@5       | 0.3317     | 0.27
nDCG@10      | 0.3317     | 0.30
```

**Interpretation:**
- Your fine-tuned model shows **+10.65% improvement** in Recall@5 (0.4065 vs 0.30)
- Your fine-tuned model shows **+2.65% improvement** in Recall@10 (0.4065 vs 0.38)
- Your fine-tuned model shows **+6.17% improvement** in nDCG@5 (0.3317 vs 0.27)
- Your fine-tuned model shows **+3.17% improvement** in nDCG@10 (0.3317 vs 0.30)

---

## Complete Output Example

```
✅ Using NVIDIA GPU (cuda) for acceleration.
Loading model: ./bge-finetuned-all-domains (this is your fine-tuned model)...

--- Running Full FINE-TUNED BGE Baseline Replication ---

--- Processing Domain: clapnq ---
Loading data: clapnq...
Corpus: 183408 docs | Queries: 150 queries
Running retrieval for clapnq (This will take a while)...
Evaluating results...
--- Results for clapnq ---
Recall@10: 0.4234
nDCG@10:   0.3456

--- Processing Domain: fiqa ---
Loading data: fiqa...
Corpus: 49607 docs | Queries: 120 queries
Running retrieval for fiqa (This will take a while)...
Evaluating results...
--- Results for fiqa ---
Recall@10: 0.3891
nDCG@10:   0.3123

--- Processing Domain: govt ---
Loading data: govt...
Corpus: 72422 docs | Queries: 180 queries
Running retrieval for govt (This will take a while)...
Evaluating results...
--- Results for govt ---
Recall@10: 0.4012
nDCG@10:   0.3289

--- Processing Domain: cloud ---
Loading data: cloud...
Corpus: 61022 docs | Queries: 200 queries
Running retrieval for cloud (This will take a while)...
Evaluating results...
--- Results for cloud ---
Recall@10: 0.4123
nDCG@10:   0.3398

--- Finished All Domains: Final Summary ---

--- Your Fine-Tuned BGE vs. Pre-Trained BGE (Paper) ---
Metric       | Your Avg.  | Pre-Trained BGE (Paper)
-------------------------------------------------------
Recall@5     | 0.4065     | 0.30
Recall@10    | 0.4065     | 0.38
nDCG@5       | 0.3317     | 0.27
nDCG@10      | 0.3317     | 0.30

--- Individual Domain Scores (Your Fine-Tuned Model) ---
Domain     | R@10    | nDCG@10
-------------------------------
clapnq     | 0.4234  | 0.3456
fiqa       | 0.3891  | 0.3123
govt       | 0.4012  | 0.3289
cloud      | 0.4123  | 0.3398

Fine-tuned BGE evaluation complete.
```

---

## Key Features Demonstrated

✅ **Fine-tuned model evaluation**: Tests domain-specific improvements  
✅ **BEIR framework**: Uses standard IR evaluation tools  
✅ **Multi-domain testing**: Evaluates on all 4 MT-RAG domains  
✅ **Comprehensive metrics**: Recall@5, Recall@10, nDCG@5, nDCG@10  
✅ **Baseline comparison**: Compares against pre-trained BGE from paper  
✅ **Dense retrieval**: Uses semantic embeddings (not keyword-based)  
✅ **GPU acceleration**: Automatically uses GPU if available  

---

## Understanding the Results

### Why Fine-Tuning Helps

**Pre-trained BGE** (`BAAI/bge-base-en-v1.5`):
- Trained on general web text
- Good at general semantic understanding
- May miss domain-specific terminology

**Fine-tuned BGE** (`./bge-finetuned-all-domains`):
- Trained on MT-RAG domains (clapnq, fiqa, govt, cloud)
- Better understands domain-specific language
- Improved performance on domain queries

### Metric Interpretation

- **Recall@K**: Fraction of relevant documents found in top K
  - Higher = better coverage
  - Example: Recall@10 = 0.41 means 41% of relevant docs are in top 10

- **nDCG@K**: Ranking quality (position matters)
  - Higher = better ranking (relevant docs at top)
  - Example: nDCG@10 = 0.33 means relevant docs are reasonably well-ranked

### Domain Performance

Different domains may show different improvements:
- **ClapNQ** (Wikipedia): May benefit less (general knowledge)
- **Cloud** (Technical docs): May benefit more (domain-specific terms)
- **FiQA** (Finance): May benefit more (specialized vocabulary)
- **Govt** (Government): May benefit more (formal language)

---

## Troubleshooting

### Model Not Found
```
Error: Model ./bge-finetuned-all-domains not found
```
**Solution**: Run `train_finetuned_bge.py` first to create the model

### Out of Memory
```
CUDA out of memory
```
**Solution**: Reduce `batch_size` in the script (line 26) from 128 to 64 or 32

### Slow Performance
```
⚠️ No GPU found. Using CPU (this will be very slow).
```
**Solution**: Use a machine with GPU for faster evaluation

### Missing Data Files
```
Error loading data for cloud: File not found
```
**Solution**: Ensure corpus and query files are extracted and in correct paths

