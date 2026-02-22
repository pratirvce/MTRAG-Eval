# How to Run the MT-RAG Benchmark Project

This guide explains how to set up and run different components of the MT-RAG benchmark project.

## Prerequisites

- Python 3.8+ (recommended: 3.9 or 3.10)
- CUDA-capable GPU (recommended for training and evaluation)
- Git

## Step 1: Clone and Navigate to Project

```bash
cd /home/prevanka/prati/su-mt-rag/mt-rag-benchmark
```

## Step 2: Set Up Python Environment

### Option A: Using Conda (Recommended)

```bash
# Create a new conda environment
conda create -n mt-rag python=3.10 -y
conda activate mt-rag

# Install PyTorch (adjust CUDA version as needed)
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

### Option B: Using Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

## Step 3: Install Dependencies

### For Evaluation Scripts

The evaluation scripts have specific requirements:

```bash
# First install PyTorch (if not using conda)
pip install torch==2.8.0

# Then install evaluation dependencies
pip install -c scripts/evaluation/constraints.txt -r scripts/evaluation/requirements.txt
```

**Note:** If `flash_attn` installation fails, install torch first:
```bash
pip install torch==2.8.0
pip install -c scripts/evaluation/constraints.txt -r scripts/evaluation/requirements.txt
```

### For Training/Evaluation Scripts (BGE Fine-tuning)

```bash
# Install BEIR and Sentence Transformers
pip install beir sentence-transformers

# Additional dependencies
pip install numpy pandas tqdm
```

### Complete Installation (All Components)

```bash
# Core dependencies
pip install torch==2.8.0
pip install beir sentence-transformers
pip install numpy pandas tqdm

# Evaluation dependencies
pip install -c scripts/evaluation/constraints.txt -r scripts/evaluation/requirements.txt
```

## Step 4: Prepare Data

### Extract Corpora (if needed)

The corpora files are zipped. Extract them if you need to use them:

```bash
cd corpora/passage_level
unzip clapnq.jsonl.zip
unzip cloud.jsonl.zip
unzip fiqa.jsonl.zip
unzip govt.jsonl.zip
cd ../..
```

## Step 5: Running Different Components

### A. Convert Conversations to Retrieval Format

Convert conversation data to BEIR retrieval format:

```bash
python scripts/conversations2retrieval.py \
    -i human/conversations/conversations.json \
    -o output_dir \
    -t -1  # -1 = last turn only, -3 = current + previous Q+A, 0 = full conversation
```

Options:
- `-i, --input`: Input conversations JSON file
- `-o, --output`: Output directory
- `-t, --turns_to_keep`: Which turns to keep (-1 = last turn, -3 = current + previous, 0 = full)
- `-q, --q_only`: Only use questions (exclude agent responses)

### B. Evaluate Retrieval Performance

Evaluate retrieval results:

```bash
python scripts/evaluation/run_retrieval_eval.py \
    --input_file human/generation_tasks/RAG.jsonl \
    --output_file results/retrieval_results.jsonl
```

This will:
- Compute Recall@1, Recall@3, Recall@5 and nDCG@1, nDCG@3, nDCG@5
- Generate aggregate CSV file: `results/retrieval_results_aggregate.csv`
- Add `retriever_scores` to each task in the output file

### C. Evaluate Generation Performance

#### Using OpenAI (Azure) as Judge

```bash
export AZURE_OPENAI_API_KEY="your-api-key"
export OPENAI_AZURE_HOST="https://your-endpoint.openai.azure.com/"

python scripts/evaluation/run_generation_eval.py \
    -i scripts/evaluation/responses-10.jsonl \
    -o results/generation_results.jsonl \
    -e scripts/evaluation/config.yaml \
    --provider openai \
    --openai_key $AZURE_OPENAI_API_KEY \
    --azure_host $OPENAI_AZURE_HOST
```

#### Using HuggingFace Model as Judge

```bash
python scripts/evaluation/run_generation_eval.py \
    -i scripts/evaluation/responses-10.jsonl \
    -o results/generation_results.jsonl \
    -e scripts/evaluation/config.yaml \
    --provider hf \
    --judge_model ibm-granite/granite-3.3-8b-instruct
```

This evaluates:
- **Algorithmic metrics**: BERTScore, ROUGE-L, Recall, Extractiveness
- **LLM-as-judge metrics**: IDK detection, RAGAS faithfulness, RadBench score
- **IDK-conditioned metrics**: Adjusted scores based on answerability

### D. Train Fine-tuned BGE Model

Fine-tune BGE model on MT-RAG domains:

```bash
python train_finetuned_bge.py
```

This will:
- Load data from all 4 domains (clapnq, fiqa, govt, cloud)
- Fine-tune BAAI/bge-base-en-v1.5 model
- Save model to `./bge-finetuned-all-domains`

**Configuration** (edit in `train_finetuned_bge.py`):
- `TRAIN_BATCH_SIZE`: 16 (reduce to 8 if OOM)
- `EPOCHS`: 1 (increase to 3-5 for better results)
- `WARMUP_STEPS`: 100

### E. Evaluate Fine-tuned BGE Model

Evaluate your fine-tuned model:

```bash
python evaluate_finetuned_bge.py
```

This will:
- Load the fine-tuned model from `./bge-finetuned-all-domains`
- Evaluate on all 4 domains
- Compare against pre-trained BGE baseline
- Print Recall@5, Recall@10, nDCG@5, nDCG@10

## Step 6: Input File Formats

### For Retrieval Evaluation

Input JSONL file should have:
```json
{
  "task_id": "unique_id",
  "Collection": "mt-rag-clapnq-elser-512-100-20240503",
  "contexts": [
    {
      "document_id": "822086267_6698-7277-0-579",
      "score": 18.759138,
      "text": "..."
    }
  ]
}
```

### For Generation Evaluation

Input JSONL file should have:
```json
{
  "task_id": "unique_id",
  "input": [...],
  "contexts": [...],
  "targets": [...],
  "predictions": [
    {
      "text": "Generated answer text here"
    }
  ]
}
```

## Troubleshooting

### GPU Issues

If CUDA is not available:
- Check: `python -c "import torch; print(torch.cuda.is_available())"`
- Install appropriate PyTorch version with CUDA support
- For training, GPU is highly recommended (CPU will be very slow)

### Memory Issues

- Reduce `TRAIN_BATCH_SIZE` in training scripts
- Use smaller models for evaluation
- Enable gradient checkpointing if needed

### Flash Attention Installation

If `flash_attn` fails to install:
```bash
pip install torch==2.8.0
# Then retry evaluation requirements
pip install -c scripts/evaluation/constraints.txt -r scripts/evaluation/requirements.txt
```

### Missing Dependencies

If you get import errors:
```bash
# Install missing packages individually
pip install beir sentence-transformers pytrec_eval
pip install evaluate bert_score rouge-score
pip install ragas langchain langchain-community
```

## Example Workflow

1. **Extract corpora** (if needed):
   ```bash
   cd corpora/passage_level && unzip *.zip && cd ../..
   ```

2. **Train a model**:
   ```bash
   python train_finetuned_bge.py
   ```

3. **Evaluate retrieval**:
   ```bash
   python evaluate_finetuned_bge.py
   ```

4. **Evaluate generation** (if you have predictions):
   ```bash
   python scripts/evaluation/run_generation_eval.py \
       -i your_predictions.jsonl \
       -o results.jsonl \
       --provider hf \
       --judge_model ibm-granite/granite-3.3-8b-instruct
   ```

## Additional Resources

- Evaluation README: `scripts/evaluation/README.md`
- Retrieval tasks README: `human/retrieval_tasks/README.md`
- Generation tasks README: `human/generation_tasks/README.md`
- Corpora README: `corpora/README.md`

