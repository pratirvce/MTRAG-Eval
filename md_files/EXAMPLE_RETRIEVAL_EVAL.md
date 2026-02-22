# Example: Retrieval Evaluation with `run_retrieval_eval.py`

This document shows a concrete example of how `run_retrieval_eval.py` evaluates retrieval performance using standard IR metrics.

## Overview

The script evaluates retrieval results by:
1. Loading retrieval predictions (from your retrieval system)
2. Loading ground truth relevance labels (qrels)
3. Computing Recall@1, Recall@3, Recall@5, nDCG@1, nDCG@3, nDCG@5
4. Aggregating results across collections
5. Enriching input file with per-query scores

---

## Input Files

### 1. Input JSONL: Retrieval Predictions

**File:** `retrieval_predictions.jsonl`

This file contains retrieval results from your system. Each line is a JSON object with:
- `task_id`: Unique query identifier
- `Collection`: Collection/domain name
- `contexts`: List of retrieved documents with scores

```jsonl
{
  "task_id": "user_123_1704067200",
  "conversation_id": "conv_001",
  "Collection": "mt-rag-ibmcloud-elser-512-100-20240502",
  "contexts": [
    {
      "document_id": "ibmcld_00474-7885-8455",
      "score": 0.95,
      "text": "IBM Cloudant is a fully managed NoSQL document database...",
      "title": "Cloudant Overview"
    },
    {
      "document_id": "ibmcld_00513-7-2197",
      "score": 0.87,
      "text": "Cloudant supports JSON documents with flexible schemas...",
      "title": "Cloudant Schema"
    },
    {
      "document_id": "ibmcld_99999-0000-0000",
      "score": 0.45,
      "text": "IBM Watson provides AI services for various applications...",
      "title": "Watson Services"
    },
    {
      "document_id": "ibmcld_88888-1111-2222",
      "score": 0.32,
      "text": "IBM Cloud storage solutions offer scalable data management...",
      "title": "Cloud Storage"
    },
    {
      "document_id": "ibmcld_77777-3333-4444",
      "score": 0.21,
      "text": "IBM Db2 is a relational database management system...",
      "title": "Db2 Database"
    }
  ]
}
{
  "task_id": "user_123_1704067300",
  "conversation_id": "conv_001",
  "Collection": "mt-rag-ibmcloud-elser-512-100-20240502",
  "contexts": [
    {
      "document_id": "ibmcld_00526-7-1750",
      "score": 0.92,
      "text": "Cloudant's flexible schema allows storing JSON documents without predefined structure...",
      "title": "Flexible Schema"
    },
    {
      "document_id": "ibmcld_00526-2924-4511",
      "score": 0.78,
      "text": "You can add new fields to documents dynamically without schema migration...",
      "title": "Dynamic Fields"
    },
    {
      "document_id": "ibmcld_00474-7885-8455",
      "score": 0.55,
      "text": "IBM Cloudant is a fully managed NoSQL document database...",
      "title": "Cloudant Overview"
    }
  ]
}
{
  "task_id": "user_456_1704067400",
  "conversation_id": "conv_002",
  "Collection": "mt-rag-clapnq-elser-512-100-20240503",
  "contexts": [
    {
      "document_id": "clapnq_12345-6789-0123",
      "score": 0.88,
      "text": "The 2017 Arizona Cardinals season was the team's 98th season...",
      "title": "2017 Cardinals Season"
    },
    {
      "document_id": "clapnq_23456-7890-1234",
      "score": 0.65,
      "text": "The Cardinals played their home games at University of Phoenix Stadium...",
      "title": "Cardinals Stadium"
    }
  ]
}
```

### 2. Qrels File: Ground Truth Relevance

**File:** `human/retrieval_tasks/cloud/qrels/dev.tsv`

This file contains human-annotated relevance judgments. Format: `query-id \t corpus-id \t score`

```tsv
query-id	corpus-id	score
user_123_1704067200	ibmcld_00474-7885-8455	1
user_123_1704067200	ibmcld_00513-7-2197	1
user_123_1704067300	ibmcld_00526-7-1750	1
user_123_1704067300	ibmcld_00526-2924-4511	1
user_456_1704067400	clapnq_12345-6789-0123	1
```

**Note:** 
- `score = 1` means the document is relevant to the query
- `score = 0` means not relevant (usually omitted from qrels)
- Multiple relevant documents per query are allowed

---

## Running the Evaluation

**Command:**
```bash
python scripts/evaluation/run_retrieval_eval.py \
  --input_file retrieval_predictions.jsonl \
  --output_file retrieval_results.jsonl
```

---

## Processing Steps

### Step 1: Load Retrieval Predictions

The script extracts:
- Query ID from `task_id`
- Document scores from `contexts[].score`
- Collection name from `Collection`

**Internal representation:**
```python
retrieval_predictions = {
    "user_123_1704067200": {
        "ibmcld_00474-7885-8455": 0.95,
        "ibmcld_00513-7-2197": 0.87,
        "ibmcld_99999-0000-0000": 0.45,
        "ibmcld_88888-1111-2222": 0.32,
        "ibmcld_77777-3333-4444": 0.21
    },
    "user_123_1704067300": {
        "ibmcld_00526-7-1750": 0.92,
        "ibmcld_00526-2924-4511": 0.78,
        "ibmcld_00474-7885-8455": 0.55
    },
    "user_456_1704067400": {
        "clapnq_12345-6789-0123": 0.88,
        "clapnq_23456-7890-1234": 0.65
    }
}
```

### Step 2: Load Qrels (Ground Truth)

**Internal representation:**
```python
qrels = {
    "user_123_1704067200": {
        "ibmcld_00474-7885-8455": 1,  # Relevant
        "ibmcld_00513-7-2197": 1      # Relevant
    },
    "user_123_1704067300": {
        "ibmcld_00526-7-1750": 1,     # Relevant
        "ibmcld_00526-2924-4511": 1   # Relevant
    },
    "user_456_1704067400": {
        "clapnq_12345-6789-0123": 1   # Relevant
    }
}
```

### Step 3: Group by Collection

The script groups queries by collection:
- **mt-rag-ibmcloud-elser-512-100-20240502**: 2 queries
- **mt-rag-clapnq-elser-512-100-20240503**: 1 query

### Step 4: Compute Metrics per Collection

For each collection, the script uses `pytrec_eval` to compute:

#### For Query: `user_123_1704067200`

**Retrieved documents (ranked by score):**
1. `ibmcld_00474-7885-8455` (score: 0.95) ✓ Relevant
2. `ibmcld_00513-7-2197` (score: 0.87) ✓ Relevant
3. `ibmcld_99999-0000-0000` (score: 0.45) ✗ Not relevant
4. `ibmcld_88888-1111-2222` (score: 0.32) ✗ Not relevant
5. `ibmcld_77777-3333-4444` (score: 0.21) ✗ Not relevant

**Relevant documents (from qrels):** 2 total
- `ibmcld_00474-7885-8455`
- `ibmcld_00513-7-2197`

**Metrics:**
- **Recall@1**: 1/2 = 0.5 (1 relevant found in top 1)
- **Recall@3**: 2/2 = 1.0 (2 relevant found in top 3)
- **Recall@5**: 2/2 = 1.0 (2 relevant found in top 5)
- **nDCG@1**: ~0.72 (normalized discounted cumulative gain)
- **nDCG@3**: ~0.95 (both relevant docs in top 3)
- **nDCG@5**: ~0.95 (both relevant docs in top 5)

#### For Query: `user_123_1704067300`

**Retrieved documents:**
1. `ibmcld_00526-7-1750` (score: 0.92) ✓ Relevant
2. `ibmcld_00526-2924-4511` (score: 0.78) ✓ Relevant
3. `ibmcld_00474-7885-8455` (score: 0.55) ✗ Not relevant

**Relevant documents:** 2 total
- `ibmcld_00526-7-1750`
- `ibmcld_00526-2924-4511`

**Metrics:**
- **Recall@1**: 1/2 = 0.5
- **Recall@3**: 2/2 = 1.0
- **Recall@5**: 2/2 = 1.0
- **nDCG@1**: ~0.72
- **nDCG@3**: ~0.95
- **nDCG@5**: ~0.95

#### For Query: `user_456_1704067400`

**Retrieved documents:**
1. `clapnq_12345-6789-0123` (score: 0.88) ✓ Relevant
2. `clapnq_23456-7890-1234` (score: 0.65) ✗ Not relevant

**Relevant documents:** 1 total
- `clapnq_12345-6789-0123`

**Metrics:**
- **Recall@1**: 1/1 = 1.0
- **Recall@3**: 1/1 = 1.0
- **Recall@5**: 1/1 = 1.0
- **nDCG@1**: 1.0
- **nDCG@3**: 1.0
- **nDCG@5**: 1.0

### Step 5: Aggregate by Collection

**Collection: mt-rag-ibmcloud-elser-512-100-20240502** (2 queries)
- Average Recall@1: (0.5 + 0.5) / 2 = 0.5
- Average Recall@3: (1.0 + 1.0) / 2 = 1.0
- Average Recall@5: (1.0 + 1.0) / 2 = 1.0
- Average nDCG@1: (0.72 + 0.72) / 2 = 0.72
- Average nDCG@3: (0.95 + 0.95) / 2 = 0.95
- Average nDCG@5: (0.95 + 0.95) / 2 = 0.95

**Collection: mt-rag-clapnq-elser-512-100-20240503** (1 query)
- Average Recall@1: 1.0
- Average Recall@3: 1.0
- Average Recall@5: 1.0
- Average nDCG@1: 1.0
- Average nDCG@3: 1.0
- Average nDCG@5: 1.0

### Step 6: Compute Weighted Average Across All Collections

**Total queries:** 3
- Cloud: 2 queries
- ClapNQ: 1 query

**Weighted Average:**
- Recall@1: (0.5 × 2 + 1.0 × 1) / 3 = 0.667
- Recall@3: (1.0 × 2 + 1.0 × 1) / 3 = 1.0
- Recall@5: (1.0 × 2 + 1.0 × 1) / 3 = 1.0
- nDCG@1: (0.72 × 2 + 1.0 × 1) / 3 = 0.813
- nDCG@3: (0.95 × 2 + 1.0 × 1) / 3 = 0.967
- nDCG@5: (0.95 × 2 + 1.0 × 1) / 3 = 0.967

---

## Output Files

### 1. Enriched JSONL: `retrieval_results.jsonl`

Each input line is enriched with a `retriever_scores` field containing per-query metrics:

```jsonl
{
  "task_id": "user_123_1704067200",
  "conversation_id": "conv_001",
  "Collection": "mt-rag-ibmcloud-elser-512-100-20240502",
  "contexts": [
    {
      "document_id": "ibmcld_00474-7885-8455",
      "score": 0.95,
      "text": "IBM Cloudant is a fully managed NoSQL document database...",
      "title": "Cloudant Overview"
    },
    {
      "document_id": "ibmcld_00513-7-2197",
      "score": 0.87,
      "text": "Cloudant supports JSON documents with flexible schemas...",
      "title": "Cloudant Schema"
    },
    {
      "document_id": "ibmcld_99999-0000-0000",
      "score": 0.45,
      "text": "IBM Watson provides AI services for various applications...",
      "title": "Watson Services"
    },
    {
      "document_id": "ibmcld_88888-1111-2222",
      "score": 0.32,
      "text": "IBM Cloud storage solutions offer scalable data management...",
      "title": "Cloud Storage"
    },
    {
      "document_id": "ibmcld_77777-3333-4444",
      "score": 0.21,
      "text": "IBM Db2 is a relational database management system...",
      "title": "Db2 Database"
    }
  ],
  "retriever_scores": {
    "ndcg_cut_1": 0.7213,
    "ndcg_cut_3": 0.9531,
    "ndcg_cut_5": 0.9531,
    "recall_1": 0.5,
    "recall_3": 1.0,
    "recall_5": 1.0
  }
}
{
  "task_id": "user_123_1704067300",
  "conversation_id": "conv_001",
  "Collection": "mt-rag-ibmcloud-elser-512-100-20240502",
  "contexts": [
    {
      "document_id": "ibmcld_00526-7-1750",
      "score": 0.92,
      "text": "Cloudant's flexible schema allows storing JSON documents without predefined structure...",
      "title": "Flexible Schema"
    },
    {
      "document_id": "ibmcld_00526-2924-4511",
      "score": 0.78,
      "text": "You can add new fields to documents dynamically without schema migration...",
      "title": "Dynamic Fields"
    },
    {
      "document_id": "ibmcld_00474-7885-8455",
      "score": 0.55,
      "text": "IBM Cloudant is a fully managed NoSQL document database...",
      "title": "Cloudant Overview"
    }
  ],
  "retriever_scores": {
    "ndcg_cut_1": 0.7213,
    "ndcg_cut_3": 0.9531,
    "ndcg_cut_5": 0.9531,
    "recall_1": 0.5,
    "recall_3": 1.0,
    "recall_5": 1.0
  }
}
{
  "task_id": "user_456_1704067400",
  "conversation_id": "conv_002",
  "Collection": "mt-rag-clapnq-elser-512-100-20240503",
  "contexts": [
    {
      "document_id": "clapnq_12345-6789-0123",
      "score": 0.88,
      "text": "The 2017 Arizona Cardinals season was the team's 98th season...",
      "title": "2017 Cardinals Season"
    },
    {
      "document_id": "clapnq_23456-7890-1234",
      "score": 0.65,
      "text": "The Cardinals played their home games at University of Phoenix Stadium...",
      "title": "Cardinals Stadium"
    }
  ],
  "retriever_scores": {
    "ndcg_cut_1": 1.0,
    "ndcg_cut_3": 1.0,
    "ndcg_cut_5": 1.0,
    "recall_1": 1.0,
    "recall_3": 1.0,
    "recall_5": 1.0
  }
}
```

### 2. Aggregate CSV: `retrieval_results_aggregate.csv`

Summary statistics across all collections:

```csv
nDCG,Recall,collection,count
"[0.72, 0.95, 0.95]","[0.5, 1.0, 1.0]",mt-rag-ibmcloud-elser-512-100-20240502,2
"[1.0, 1.0, 1.0]","[1.0, 1.0, 1.0]",mt-rag-clapnq-elser-512-100-20240503,1
"[0.813, 0.967, 0.967]","[0.667, 1.0, 1.0]",all,3
```

**Column explanation:**
- `nDCG`: List of [nDCG@1, nDCG@3, nDCG@5]
- `Recall`: List of [Recall@1, Recall@3, Recall@5]
- `collection`: Collection name or "all" for weighted average
- `count`: Number of queries in that collection

---

## Console Output

When you run the script, you'll see:

```
collection_name: mt-rag-ibmcloud-elser-512-100-20240502
Retriever Evaluation Aggregate Scores: {
  'nDCG': [0.72, 0.95, 0.95],
  'Recall': [0.5, 1.0, 1.0],
  'collection': 'mt-rag-ibmcloud-elser-512-100-20240502',
  'count': 2
}

collection_name: mt-rag-clapnq-elser-512-100-20240503
Retriever Evaluation Aggregate Scores: {
  'nDCG': [1.0, 1.0, 1.0],
  'Recall': [1.0, 1.0, 1.0],
  'collection': 'mt-rag-clapnq-elser-512-100-20240503',
  'count': 1
}

Weighted average Recall: [0.667, 1.0, 1.0]
Weighted average nDCG: [0.813, 0.967, 0.967]
```

---

## Understanding the Metrics

### Recall@K
- **Definition**: Fraction of relevant documents found in top K results
- **Range**: 0.0 to 1.0 (higher is better)
- **Example**: Recall@3 = 1.0 means all relevant docs are in top 3

### nDCG@K (Normalized Discounted Cumulative Gain)
- **Definition**: Measures ranking quality, giving higher weight to relevant docs at top positions
- **Range**: 0.0 to 1.0 (higher is better)
- **Example**: nDCG@3 = 0.95 means relevant docs are well-ranked in top 3

### Why Both Metrics?
- **Recall**: Measures coverage (did we find all relevant docs?)
- **nDCG**: Measures ranking quality (are relevant docs at the top?)

---

## Key Features Demonstrated

✅ **Multi-collection support**: Handles queries from different domains  
✅ **Per-query metrics**: Each query gets individual scores  
✅ **Collection-level aggregation**: Average metrics per collection  
✅ **Weighted averaging**: Accounts for different collection sizes  
✅ **BEIR compatibility**: Uses standard IR evaluation tools  
✅ **Non-destructive**: Original input data is preserved, scores are added

