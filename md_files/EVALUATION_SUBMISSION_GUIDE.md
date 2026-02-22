# MTRAGEval Task A Evaluation Submission Guide

This guide ensures proper submission format for MTRAGEval Task A (Retrieval Only).

**Official Task Page**: [MTRAGEval](https://ibm.github.io/mt-rag-benchmark/MTRAGEval/)
**Repository**: [IBM/mt-rag-benchmark](https://github.com/IBM/mt-rag-benchmark)

---

## 📋 Submission Requirements

### Evaluation Period
- **Start**: January 10, 2026
- **End**: January 20, 2026
- **Ranking Metric**: nDCG

### What Will Be Provided
- ✅ Query text
- ✅ Corpus domain (e.g., ClapNQ, Govt, Cloud, FiQA)
- ✅ Corpus documents
- ✅ Task IDs

### What Will NOT Be Provided
- ❌ Question type (e.g., factoid)
- ❌ Answerability (e.g., unanswerable, answerable)
- ❌ Multi-turn type (e.g., follow-up, clarification)
- ❌ Conversation history details
- ❌ Any metadata beyond domain

---

## 📝 Required Output Format

According to [evaluation README](scripts/evaluation/README.md), your submission must be a JSONL file with this format:

```json
{
  "task_id": "unique_task_id",
  "Collection": "mt-rag-clapnq-elser-512-100-20240503",
  "contexts": [
    {
      "document_id": "822086267_6698-7277-0-579",
      "score": 18.759138,
      "text": "...",  // Optional
      "title": "...", // Optional
      "source": ""   // Optional
    }
  ]
}
```

### Required Fields
- **`task_id`**: Unique task identifier (provided in evaluation data)
- **`Collection`**: Collection name (one of):
  - `mt-rag-clapnq-elser-512-100-20240503`
  - `mt-rag-govt-elser-512-100-20240611`
  - `mt-rag-fiqa-beir-elser-512-100-20240501`
  - `mt-rag-ibmcloud-elser-512-100-20240502`
- **`contexts`**: List of retrieval results
  - **`document_id`**: Must match corpus document IDs exactly
  - **`score`**: Retrieval score (higher = more relevant)

### Optional Fields
- `text`: Document text (not required for evaluation)
- `title`: Document title (not required for evaluation)
- `source`: Source information (not required for evaluation)

---

## 🔧 Creating Submission File

### Step 1: Load Evaluation Data
```python
# Evaluation data will be provided in this format
# You'll receive queries and domain information
evaluation_data = load_evaluation_data()  # Format TBD by organizers
```

### Step 2: Run Retrieval
```python
# Use your best model(s) based on domain
for task in evaluation_data:
    domain = task["Collection"]  # Extract domain from Collection name
    
    # Select appropriate model
    if domain == "mt-rag-clapnq-elser-512-100-20240503":
        model = load_model("models/domain_specific_clapnq")
    elif domain == "mt-rag-govt-elser-512-100-20240611":
        model = load_model("models/domain_specific_govt")
    # ... etc
    
    # Run retrieval
    results = retrieve(model, task["query"], corpus)
    
    # Format results
    contexts = [
        {"document_id": doc_id, "score": score}
        for doc_id, score in results.items()
    ]
    
    submission_item = {
        "task_id": task["task_id"],
        "Collection": task["Collection"],
        "contexts": contexts
    }
```

### Step 3: Save Submission
```python
# Save as JSONL file
with open("submission_task_a.jsonl", "w") as f:
    for item in submission_data:
        f.write(json.dumps(item) + "\n")
```

---

## ✅ Compliance Checklist

Before submitting:

- [ ] **No Metadata Usage**: Verify no code uses question_type, answerability, or multi-turn type
- [ ] **Domain-Only**: Only use domain information (will be provided)
- [ ] **Correct Format**: Output matches required JSONL format
- [ ] **Document IDs**: Match corpus document IDs exactly
- [ ] **Scores**: Properly ranked (higher = more relevant)
- [ ] **Collection Names**: Use exact collection names from requirements
- [ ] **Test Locally**: Test with official evaluation script before submission

---

## 🎯 Recommended Submission Strategy

### Option 1: Domain-Specific Models (Recommended)
```python
# Use best domain-specific model for each domain
domain_models = {
    "mt-rag-clapnq-elser-512-100-20240503": "models/domain_specific_clapnq",
    "mt-rag-govt-elser-512-100-20240611": "models/domain_specific_govt",
    "mt-rag-fiqa-beir-elser-512-100-20240501": "models/domain_specific_fiqa",
    "mt-rag-ibmcloud-elser-512-100-20240502": "models/domain_specific_cloud"
}
```

### Option 2: Ensemble (Alternative)
```python
# Use ensemble of domain-specific models
# Combine results using RRF or weighted average
ensemble_models = [
    "models/domain_specific_clapnq",
    "models/domain_specific_govt",
    "models/domain_specific_cloud"
]
results = ensemble_retrieve(ensemble_models, query, corpus)
```

### Option 3: Best Single Model
```python
# Use Phase 5 ensemble model (if performs well)
model = "models/phase5_ensemble_domain_specific"
```

---

## 📊 Evaluation Script Usage

### Test Your Submission Locally
```bash
# Use official evaluation script
python scripts/evaluation/run_retrieval_eval.py \
    --input_file submission_task_a.jsonl \
    --output_file evaluation_results.jsonl
```

This will:
- Compute Recall@1, Recall@3, Recall@5
- Compute nDCG@1, nDCG@3, nDCG@5
- Generate aggregate CSV file
- Add `retriever_scores` to output file

**Note**: Task A ranking uses **nDCG** metric.

---

## ⚠️ Important Notes

1. **Document ID Format**: 
   - Must match corpus IDs exactly
   - Format: `822086267_6698-7277-0-579` (with offsets)
   - Check corpus files for exact format

2. **Score Normalization**:
   - Scores should be properly ranked
   - Higher scores = more relevant
   - Can be similarity scores, logits, or any ranking score

3. **Top-K Results**:
   - Submit top-K results per query (K typically 10-100)
   - More results may help recall but slow evaluation

4. **Collection Name Mapping**:
   - Map domain to collection name:
     - ClapNQ → `mt-rag-clapnq-elser-512-100-20240503`
     - Govt → `mt-rag-govt-elser-512-100-20240611`
     - FiQA → `mt-rag-fiqa-beir-elser-512-100-20240501`
     - Cloud → `mt-rag-ibmcloud-elser-512-100-20240502`

---

## 🔍 Pre-Submission Testing

### Test with Training Data
```python
# Test your submission format with training data
test_queries = load_training_queries()
test_results = run_retrieval(test_queries)
submission = format_submission(test_results)

# Validate format
validate_submission_format(submission)

# Test with evaluation script
run_evaluation_script(submission)
```

### Verify Compliance
```bash
# Check for any metadata usage
grep -r "answerability\|question_type\|multi.turn" your_submission_script.py

# Should return nothing (or only in comments)
```

---

## 📝 Submission Process

1. **Prepare Submission File**:
   - Format: JSONL file
   - Name: `submission_task_a.jsonl` (or as specified)
   - Format: As described above

2. **Submit via Google Form**:
   - Form will be provided during evaluation period
   - Upload your JSONL file
   - May need team information

3. **Wait for Results**:
   - Results will be posted on leaderboard
   - Ranking based on nDCG metric

---

## 🎯 Best Practices

1. **Use Domain-Specific Models**: Best performance so far
2. **Test Locally First**: Use evaluation script before submission
3. **Verify Format**: Ensure JSONL format is correct
4. **Check Document IDs**: Must match corpus exactly
5. **Submit Early**: Allows time for fixes if needed
6. **Document Approach**: Keep notes for paper submission

---

## 📚 References

- [MTRAGEval Official Page](https://ibm.github.io/mt-rag-benchmark/MTRAGEval/)
- [Evaluation README](scripts/evaluation/README.md)
- [Compliance Document](MTRAGEVAL_TASK_A_COMPLIANCE.md)
- [IBM/mt-rag-benchmark](https://github.com/IBM/mt-rag-benchmark)

---

*Last Updated: 2025-12-13*
*Ready for Evaluation Period: January 10-20, 2026*

