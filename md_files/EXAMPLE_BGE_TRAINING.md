# Example: Training Fine-Tuned BGE Model with `train_finetuned_bge.py`

This document shows a concrete example of how `train_finetuned_bge.py` fine-tunes a BGE (BAAI General Embedding) model for retrieval tasks using contrastive learning.

## Overview

The script:
1. Loads training data from all 4 MT-RAG domains
2. Creates (query, positive_passage) pairs from qrels
3. Fine-tunes BGE-base-en-v1.5 using MultipleNegativesRankingLoss
4. Saves checkpoints during training
5. Outputs a fine-tuned model optimized for domain-specific retrieval

---

## Prerequisites

Before running, you need:
1. **Base model**: Pre-trained BGE model (`BAAI/bge-base-en-v1.5`) - downloaded automatically
2. **Training data**: Corpus, queries, and qrels files for all 4 domains
3. **GPU**: Highly recommended (training on CPU is extremely slow)

---

## Input Data Structure

### 1. Corpus File: `corpora/passage_level/cloud.jsonl`

Document passages for training:

```jsonl
{"_id": "ibmcld_00474-7885-8455", "title": "Cloudant Overview", "text": "IBM Cloudant is a fully managed NoSQL document database service..."}
{"_id": "ibmcld_00513-7-2197", "title": "Cloudant Schema", "text": "Cloudant supports JSON documents with flexible schemas..."}
{"_id": "ibmcld_00526-7-1750", "title": "Flexible Schema", "text": "Cloudant's flexible schema allows storing JSON documents..."}
...
```

### 2. Query File: `human/retrieval_tasks/cloud/cloud_questions.jsonl`

Training queries:

```jsonl
{"_id": "user_123_1704067200", "text": "Does IBM offer document databases?"}
{"_id": "user_123_1704067300", "text": "So it can store any random JSON object or I need to specify fields in advance?"}
{"_id": "user_456_1704067400", "text": "What if I want to store an image or PDF with a document?"}
...
```

### 3. Qrels File: `human/retrieval_tasks/cloud/qrels/dev.tsv`

Relevance labels (ground truth):

```tsv
query-id	corpus-id	score
user_123_1704067200	ibmcld_00474-7885-8455	1
user_123_1704067200	ibmcld_00513-7-2197	1
user_123_1704067300	ibmcld_00526-7-1750	1
user_123_1704067300	ibmcld_00526-2924-4511	1
```

---

## Running the Training

**Command:**
```bash
python train_finetuned_bge.py
```

**Expected Output:**
```
2024-01-15 10:30:00 - 🔒 Random seed set to 42 for reproducibility.
2024-01-15 10:30:01 - Starting training process...
2024-01-15 10:30:01 - Loading MTRAG domain: clapnq...
2024-01-15 10:30:05 - Loaded 1250 examples from clapnq
2024-01-15 10:30:05 - Loading MTRAG domain: fiqa...
2024-01-15 10:30:08 - Loaded 980 examples from fiqa
2024-01-15 10:30:08 - Loading MTRAG domain: govt...
2024-01-15 10:30:12 - Loaded 1100 examples from govt
2024-01-15 10:30:12 - Loading MTRAG domain: cloud...
2024-01-15 10:30:15 - Loaded 1350 examples from cloud
2024-01-15 10:30:15 - Loading custom dataset...
2024-01-15 10:30:15 - Loaded 2 custom examples (from placeholder)
2024-01-15 10:30:15 - Total training examples: 4682
2024-01-15 10:30:15 - Loading base model: BAAI/bge-base-en-v1.5...
2024-01-15 10:30:45 - Starting model fine-tuning... (Epochs=1, Batch Size=16)
Epoch: 100%|████████████████████| 293/293 [45:23<00:00,  9.28s/it, Loss=0.234]
2024-01-15 11:16:08 - Training complete. New model saved to: ./bge-finetuned-all-domains
```

---

## Step-by-Step Training Process

### Step 1: Set Random Seed for Reproducibility

```python
set_seed(42)
```

**Purpose**: Ensures reproducible results across runs
- Sets random seed for Python, NumPy, PyTorch
- Disables non-deterministic CUDA operations
- Critical for scientific reproducibility

### Step 2: Load Training Data from MT-RAG Domains

For each domain (clapnq, fiqa, govt, cloud), the script:

#### 2a. Load Corpus, Queries, and Qrels

```python
corpus, queries, qrels = GenericDataLoader(
    corpus_file="corpora/passage_level/cloud.jsonl",
    query_file="human/retrieval_tasks/cloud/cloud_questions.jsonl",
    qrels_file="human/retrieval_tasks/cloud/qrels/dev.tsv"
).load_custom()
```

**Example loaded data:**

**Corpus:**
```python
{
    "ibmcld_00474-7885-8455": {
        "title": "Cloudant Overview",
        "text": "IBM Cloudant is a fully managed NoSQL document database..."
    },
    "ibmcld_00513-7-2197": {
        "title": "Cloudant Schema",
        "text": "Cloudant supports JSON documents with flexible schemas..."
    }
}
```

**Queries:**
```python
{
    "user_123_1704067200": "Does IBM offer document databases?",
    "user_123_1704067300": "So it can store any random JSON object or I need to specify fields in advance?"
}
```

**Qrels:**
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

#### 2b. Create (Query, Positive Passage) Pairs

```python
for query_id, doc_infos in qrels.items():
    query_text = queries.get(query_id)  # "Does IBM offer document databases?"
    
    for doc_id, score in doc_infos.items():
        if score > 0:  # Only relevant passages (score = 1)
            doc = corpus.get(doc_id)
            doc_text = doc.get("title", "") + " " + doc.get("text", "")
            # Create training example
            domain_examples.append(InputExample(texts=[query_text, doc_text]))
```

**Example training pairs created:**

```python
InputExample(texts=[
    "Does IBM offer document databases?",
    "Cloudant Overview IBM Cloudant is a fully managed NoSQL document database service..."
])

InputExample(texts=[
    "Does IBM offer document databases?",
    "Cloudant Schema Cloudant supports JSON documents with flexible schemas..."
])

InputExample(texts=[
    "So it can store any random JSON object or I need to specify fields in advance?",
    "Flexible Schema Cloudant's flexible schema allows storing JSON documents without predefined structure..."
])
```

**Result**: For the cloud domain, this creates ~1,350 training examples (one per relevant query-document pair).

### Step 3: Combine All Training Examples

```python
all_train_examples = []
for domain in ["clapnq", "fiqa", "govt", "cloud"]:
    all_train_examples.extend(load_mtrag_examples(domain))
all_train_examples.extend(load_custom_examples())
```

**Total training examples:**
- ClapNQ: 1,250 examples
- FiQA: 980 examples
- Govt: 1,100 examples
- Cloud: 1,350 examples
- Custom: 2 examples (placeholder)
- **Total: 4,682 examples**

### Step 4: Load Base Model

```python
model = SentenceTransformer("BAAI/bge-base-en-v1.5")
```

**What happens:**
- Downloads pre-trained BGE model (if not cached)
- Loads model weights into memory
- Automatically uses GPU if available
- Model has ~110M parameters

### Step 5: Create DataLoader

```python
train_dataloader = DataLoader(
    all_train_examples, 
    shuffle=True,           # Randomize order each epoch
    batch_size=16          # Process 16 examples at once
)
```

**Batch structure:**
Each batch contains 16 (query, passage) pairs:

```python
Batch 1:
  Example 1: ("Does IBM offer document databases?", "Cloudant Overview IBM Cloudant...")
  Example 2: ("What is photosynthesis?", "Photosynthesis is the process...")
  Example 3: ("How do I file taxes?", "Tax filing requires...")
  ...
  Example 16: ("What is machine learning?", "Machine learning is a subset...")
```

### Step 6: Define Loss Function (MultipleNegativesRankingLoss)

```python
train_loss = losses.MultipleNegativesRankingLoss(model=model)
```

**How MultipleNegativesRankingLoss Works:**

This is a **contrastive learning** approach. For each batch:

1. **Encode all queries and passages** in the batch:
   ```
   Query embeddings: [q1, q2, q3, ..., q16]
   Passage embeddings: [p1, p2, p3, ..., p16]
   ```

2. **Compute similarity matrix** (cosine similarity):
   ```
         p1    p2    p3  ...  p16
   q1  [0.92, 0.15, 0.23, ..., 0.31]  ← q1 should match p1 (high score)
   q2  [0.18, 0.89, 0.12, ..., 0.22]  ← q2 should match p2 (high score)
   q3  [0.25, 0.11, 0.91, ..., 0.19]  ← q3 should match p3 (high score)
   ...
   ```

3. **Loss calculation**: For each query, maximize similarity with its positive passage while minimizing similarity with all other passages (negatives):
   ```
   Loss for q1 = -log(exp(sim(q1, p1)) / (exp(sim(q1, p1)) + exp(sim(q1, p2)) + ... + exp(sim(q1, p16))))
   ```

4. **Learning objective**: 
   - **Maximize**: Similarity between matching (query, passage) pairs
   - **Minimize**: Similarity between non-matching pairs
   - **Automatic negatives**: Other examples in the batch serve as negatives (no need to explicitly label negatives)

**Example with batch size 3:**

```python
Batch:
  (q1, p1)  ← Positive pair
  (q2, p2)  ← Positive pair
  (q3, p3)  ← Positive pair

Similarity matrix:
         p1    p2    p3
  q1  [0.92, 0.15, 0.23]  ← q1 should have high score with p1
  q2  [0.18, 0.89, 0.12]  ← q2 should have high score with p2
  q3  [0.25, 0.11, 0.91]  ← q3 should have high score with p3

Loss encourages:
  - q1-p1 similarity ↑ (currently 0.92, good!)
  - q1-p2 similarity ↓ (currently 0.15, good!)
  - q1-p3 similarity ↓ (currently 0.23, could be lower)
  - q2-p2 similarity ↑ (currently 0.89, good!)
  - q2-p1 similarity ↓ (currently 0.18, good!)
  - etc.
```

### Step 7: Training Loop

```python
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=1,                    # Number of passes through data
    warmup_steps=100,            # Gradual learning rate increase
    output_path="./bge-finetuned-all-domains",
    show_progress_bar=True,
    checkpoint_save_steps=1000,  # Save checkpoint every 1000 steps
    checkpoint_path="./bge-finetuned-all-domains-checkpoints"
)
```

**Training process:**

1. **Warmup phase** (first 100 steps):
   - Learning rate gradually increases from 0
   - Helps stabilize training

2. **Main training** (remaining steps):
   - For each batch:
     - Forward pass: Encode queries and passages
     - Compute loss: MultipleNegativesRankingLoss
     - Backward pass: Compute gradients
     - Update weights: Adjust model parameters
   - Progress: `Epoch: 100%|████████| 293/293 [45:23<00:00, Loss=0.234]`

3. **Checkpointing** (every 1000 steps):
   - Saves model state to `./bge-finetuned-all-domains-checkpoints/checkpoint-1000/`
   - Allows resuming training if interrupted

4. **Final save**:
   - Saves final model to `./bge-finetuned-all-domains/`

---

## What the Model Learns

### Before Fine-Tuning (Pre-trained BGE)

The pre-trained model understands general semantic similarity:
- "car" is similar to "vehicle"
- "happy" is similar to "joyful"
- But may not understand domain-specific terms well

### After Fine-Tuning

The fine-tuned model learns domain-specific patterns:

**Example 1: Technical Terminology**
- Query: "Does IBM offer document databases?"
- Passage: "IBM Cloudant is a fully managed NoSQL document database..."
- **Learned**: "document databases" should match "NoSQL document database"

**Example 2: Domain-Specific Language**
- Query: "What is a flexible schema?"
- Passage: "Cloudant's flexible schema allows storing JSON documents..."
- **Learned**: Domain-specific terms get higher similarity scores

**Example 3: Multi-turn Context**
- Query: "So it can store any random JSON object?"
- Passage: "Cloudant supports JSON documents with flexible schemas..."
- **Learned**: Conversational queries match relevant technical passages

---

## Training Output Structure

After training completes:

```
./bge-finetuned-all-domains/
├── config.json                          # Model configuration
├── config_sentence_transformers.json    # Sentence Transformers config
├── modules.json                         # Module structure
├── pytorch_model.bin                    # Model weights (or model.safetensors)
├── sentence_bert_config.json           # Sentence-BERT config
├── tokenizer_config.json               # Tokenizer config
├── tokenizer.json                      # Tokenizer files
├── vocab.txt                           # Vocabulary
└── README.md                           # Model card

./bge-finetuned-all-domains-checkpoints/
├── checkpoint-1000/
│   ├── config.json
│   ├── pytorch_model.bin
│   ├── optimizer.pt                    # Optimizer state
│   ├── scheduler.pt                    # Learning rate scheduler
│   └── trainer_state.json              # Training state
├── checkpoint-2000/
│   └── ...
└── checkpoint-3000/
    └── ...
```

---

## Training Metrics and Progress

During training, you'll see:

```
Epoch: 100%|████████████████████| 293/293 [45:23<00:00,  9.28s/it, Loss=0.234]
```

**Interpretation:**
- **293 steps**: Total number of batches (4,682 examples ÷ 16 batch size = 293 batches)
- **45:23**: Total training time (45 minutes, 23 seconds)
- **9.28s/it**: Average time per batch (9.28 seconds)
- **Loss=0.234**: Final loss value (lower is better)

**Loss progression** (typical):
- Start: ~0.8-1.2 (model is confused)
- Middle: ~0.4-0.6 (learning domain patterns)
- End: ~0.2-0.3 (well-trained on domains)

---

## Key Features Demonstrated

✅ **Multi-domain training**: Combines data from all 4 MT-RAG domains  
✅ **Contrastive learning**: Uses MultipleNegativesRankingLoss for efficient training  
✅ **Automatic negatives**: No need to explicitly create negative examples  
✅ **Reproducibility**: Random seed ensures consistent results  
✅ **Checkpointing**: Saves progress during training  
✅ **GPU acceleration**: Automatically uses GPU if available  
✅ **Custom data support**: Placeholder function for adding your own data  

---

## Understanding Contrastive Learning

### Why MultipleNegativesRankingLoss?

**Traditional approach** (explicit negatives):
- Need to manually create negative examples
- Requires labeling: (query, positive_passage, negative_passage)
- More data preparation work

**MultipleNegativesRankingLoss** (automatic negatives):
- Only need (query, positive_passage) pairs
- Other examples in batch automatically become negatives
- More efficient and scalable

### How It Improves Retrieval

**Before training:**
```
Query: "Does IBM offer document databases?"
Top results:
  1. "IBM offers various cloud services..." (score: 0.65)  ← Not ideal
  2. "Document databases are NoSQL systems..." (score: 0.62)
  3. "IBM Cloudant is a document database..." (score: 0.58)  ← Should be #1
```

**After training:**
```
Query: "Does IBM offer document databases?"
Top results:
  1. "IBM Cloudant is a document database..." (score: 0.89)  ← Correct!
  2. "Cloudant supports JSON documents..." (score: 0.85)
  3. "IBM offers various cloud services..." (score: 0.45)  ← Lower now
```

---

## Custom Dataset Integration

The script includes a placeholder function for adding your own data:

```python
def load_custom_examples():
    my_examples = []
    
    # Add your own (query, passage) pairs
    my_examples.append(InputExample(texts=[
        "Your query text here",
        "Your relevant passage text here"
    ]))
    
    return my_examples
```

**Example usage:**
```python
def load_custom_examples():
    my_examples = []
    
    # Add domain-specific examples
    my_examples.append(InputExample(texts=[
        "What is the return policy?",
        "Our return policy allows returns within 30 days of purchase..."
    ]))
    
    my_examples.append(InputExample(texts=[
        "How do I reset my password?",
        "To reset your password, click the 'Forgot Password' link..."
    ]))
    
    return my_examples
```

---

## Troubleshooting

### Out of Memory (OOM) Error
```
RuntimeError: CUDA out of memory
```
**Solution**: Reduce `TRAIN_BATCH_SIZE` from 16 to 8 or 4

### Slow Training
```
⚠️ No GPU found. Using CPU (this will be extremely slow).
```
**Solution**: Use a machine with GPU. Training on CPU can take days.

### Model Not Improving
**Possible causes:**
- Too few training examples
- Learning rate too high/low
- Need more epochs

**Solution**: 
- Increase `EPOCHS` from 1 to 3-5
- Add more training data
- Adjust learning rate (advanced)

### Checkpoint Recovery
If training is interrupted, you can resume from a checkpoint:
```python
# Load from checkpoint
model = SentenceTransformer("./bge-finetuned-all-domains-checkpoints/checkpoint-1000")
# Continue training...
```

---

## Next Steps

After training completes:

1. **Evaluate the model**: Run `evaluate_finetuned_bge.py` to see performance improvements
2. **Use for retrieval**: Load the model in your retrieval pipeline
3. **Compare results**: Check if fine-tuning improved Recall@10 and nDCG@10

The fine-tuned model should show improved performance on MT-RAG domains compared to the pre-trained baseline!

