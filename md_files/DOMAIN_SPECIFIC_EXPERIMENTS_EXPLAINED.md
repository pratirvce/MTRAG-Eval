# Domain-Specific Experiments Explained

## 🎯 What Are Domain-Specific Experiments?

Domain-specific experiments train **separate, specialized models** for each of the four domains (ClapNQ, FiQA, Govt, Cloud) instead of using one general model for all domains. The idea is that a model trained specifically on one domain will understand that domain's unique characteristics better than a general model.

---

## 🤔 Why Domain-Specific Models?

### The Problem with Multi-Domain Models

A single model trained on all domains must learn:
- **ClapNQ**: Wikipedia-style factual queries, entity relationships
- **FiQA**: Financial terminology, market data, investment concepts
- **Govt**: Bureaucratic language, regulations, policy documents
- **Cloud**: Technical documentation, API references, code snippets

This is like asking one person to be an expert in four completely different fields. While possible, specialization usually leads to better performance.

### The Solution: Specialization

By training separate models for each domain, each model can:
1. **Learn domain-specific terminology** (e.g., "ROI" in finance vs "API" in cloud)
2. **Understand domain-specific query patterns** (e.g., Wikipedia queries vs technical docs)
3. **Focus on domain-specific relevance criteria** (what makes a passage relevant differs by domain)

---

## 🔄 How Domain-Specific Training Works

### The Two-Stage Transfer Learning Strategy

The domain-specific experiments use a **two-stage transfer learning approach**:

```
Stage 1: Multi-Domain Pre-Training
┌─────────────────────────────────────┐
│ BGE-base-en-v1.5 (Pre-trained)      │
│         ↓                            │
│ Train on ALL domains together        │
│ (ClapNQ + FiQA + Govt + Cloud)      │
│         ↓                            │
│ Phase 1 Epochs 5 Model              │
│ (General multi-domain knowledge)     │
└─────────────────────────────────────┘
              ↓
              ↓ (Transfer Learning)
              ↓
┌─────────────────────────────────────┐
│ Stage 2: Domain-Specific Fine-Tuning│
│                                     │
│ For ClapNQ:                         │
│   Phase 1 Model → Train on ClapNQ  │
│   → domain_specific_clapnq          │
│                                     │
│ For FiQA:                           │
│   Phase 1 Model → Train on FiQA    │
│   → domain_specific_fiqa            │
│                                     │
│ For Govt:                           │
│   Phase 1 Model → Train on Govt    │
│   → domain_specific_govt           │
│                                     │
│ For Cloud:                          │
│   Phase 1 Model → Train on Cloud   │
│   → domain_specific_cloud           │
└─────────────────────────────────────┘
```

### Why This Two-Stage Approach?

1. **Stage 1 (Multi-Domain)**: Provides a strong foundation
   - Model learns general semantic understanding
   - Understands relationships between queries and documents
   - Has knowledge from all domains

2. **Stage 2 (Domain-Specific)**: Adds specialization
   - Takes the general knowledge from Stage 1
   - Fine-tunes it on one specific domain
   - Learns domain-specific nuances

This is like:
- **Stage 1**: Learning to be a general doctor (knows medicine basics)
- **Stage 2**: Specializing in cardiology (deep expertise in one area)

---

## 📋 Training Process Details

### Configuration Used

For each domain-specific model:

```json
{
  "domain": "clapnq",  // or "fiqa", "govt", "cloud"
  "base_model": "BAAI/bge-base-en-v1.5",
  "epochs": 7,
  "batch_size": 32,
  "learning_rate": 1e-05,  // Lower LR for fine-tuning
  "use_pretrained_multi_domain": true,
  "pretrained_multi_domain_path": "./models/phase1_epochs5",
  "use_validation": true,
  "use_data_splits": true,
  "warmup_steps": 50,
  "evaluation_steps": 200,
  "save_best_model": true
}
```

### Step-by-Step Training Process

1. **Load Starting Model**
   - Start from `phase1_epochs5` (multi-domain pre-trained model)
   - This model already knows general retrieval patterns

2. **Load Domain-Specific Data**
   - Load only training data from ONE domain (e.g., only ClapNQ)
   - Use train/val/test splits (70/15/15)
   - Create (query, relevant_passage) pairs

3. **Fine-Tune on Domain Data**
   - Train for 7 epochs on domain-specific data
   - Use lower learning rate (1e-5) to make small adjustments
   - Use MultipleNegativesRankingLoss (contrastive learning)

4. **Validation During Training**
   - Evaluate on validation set every 200 steps
   - Save best model based on validation performance
   - Prevents overfitting

5. **Save Domain-Specific Model**
   - Save to `models/domain_specific_{domain}/`
   - This model is now specialized for that domain

### Code Flow

```python
# 1. Load pre-trained multi-domain model
model = SentenceTransformer("./models/phase1_epochs5")

# 2. Load domain-specific training data
train_examples = load_domain_examples("clapnq", split="train")

# 3. Create training dataloader
train_dataloader = DataLoader(train_examples, batch_size=32)

# 4. Define loss function
train_loss = losses.MultipleNegativesRankingLoss(model=model)

# 5. Fine-tune for 7 epochs
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=7,
    learning_rate=1e-5,  # Lower LR for fine-tuning
    output_path="./models/domain_specific_clapnq"
)
```

---

## 📊 Results Comparison

### Multi-Domain vs Domain-Specific

| Approach | Model | Recall@10 | nDCG@10 | Improvement |
|----------|-------|-----------|---------|--------------|
| **Multi-Domain** | Phase 1 Epochs 5 | 0.4693 | 0.3671 | Baseline |
| **Domain-Specific ClapNQ** | domain_specific_clapnq | **0.6016** | **0.4981** | +28.2% / +35.7% |
| **Domain-Specific Govt** | domain_specific_govt | **0.5511** | **0.4628** | +17.4% / +26.1% |
| **Domain-Specific Cloud** | domain_specific_cloud | **0.5293** | **0.4104** | +12.8% / +11.8% |
| **Domain-Specific FiQA** | domain_specific_fiqa | **0.5119** | **0.4026** | +9.1% / +9.7% |
| **Average Domain-Specific** | - | **0.5485** | **0.4435** | **+16.9% / +20.8%** |

### Why Domain-Specific Works Better

1. **ClapNQ**: Wikipedia queries benefit from entity-focused understanding
   - Domain model: 0.6016 Recall@10
   - Multi-domain: 0.5529 Recall@10
   - **+8.8% improvement**

2. **Govt**: Government documents have specific bureaucratic language
   - Domain model: 0.5511 Recall@10
   - Multi-domain: 0.5108 Recall@10
   - **+7.9% improvement**

3. **Cloud**: Technical documentation needs code/API understanding
   - Domain model: 0.5293 Recall@10
   - Multi-domain: 0.3851 Recall@10
   - **+37.4% improvement** (biggest gain!)

4. **FiQA**: Financial terminology requires domain expertise
   - Domain model: 0.5119 Recall@10
   - Multi-domain: 0.4286 Recall@10
   - **+19.4% improvement**

---

## 🔍 Key Differences from Multi-Domain Training

### Multi-Domain Training (Phase 1)

```python
# Train on ALL domains together
all_examples = []
for domain in ["clapnq", "fiqa", "govt", "cloud"]:
    all_examples.extend(load_domain_examples(domain))

# One model for all domains
model.fit(all_examples)  # Mixed domain data
```

**Characteristics:**
- ✅ One model handles all domains
- ✅ Simpler deployment (one model file)
- ❌ Must balance learning across domains
- ❌ May not optimize for any single domain

### Domain-Specific Training (Phase 4)

```python
# Train separate model for EACH domain
for domain in ["clapnq", "fiqa", "govt", "cloud"]:
    domain_examples = load_domain_examples(domain)
    model.fit(domain_examples)  # Only one domain
    model.save(f"domain_specific_{domain}")
```

**Characteristics:**
- ✅ Optimized for each domain
- ✅ Better performance per domain
- ❌ Need 4 separate models
- ❌ More complex deployment

---

## 💡 Why This Strategy Works

### 1. Transfer Learning Benefits

Starting from a multi-domain model gives:
- **Strong foundation**: General semantic understanding
- **Better initialization**: Better than starting from scratch
- **Faster convergence**: Less training needed
- **Stable training**: Pre-trained weights prevent instability

### 2. Domain Specialization Benefits

Fine-tuning on one domain allows:
- **Focused learning**: Model only needs to learn one domain's patterns
- **Terminology mastery**: Deep understanding of domain-specific terms
- **Query pattern recognition**: Understands domain-specific query styles
- **Relevance optimization**: Learns what makes passages relevant in that domain

### 3. The Best of Both Worlds

```
General Knowledge (Multi-Domain) + Specialized Knowledge (Domain-Specific)
= Better Performance Than Either Alone
```

---

## 🎯 When to Use Domain-Specific Models

### Use Domain-Specific Models When:

✅ **You know the domain in advance**
   - Can route queries to the right model
   - Example: "This is a finance query → use FiQA model"

✅ **Performance is critical**
   - Domain-specific models consistently outperform multi-domain
   - Worth the complexity if performance matters

✅ **You have domain-specific data**
   - Enough training data per domain
   - Can afford to train multiple models

### Use Multi-Domain Models When:

✅ **Domain is unknown at query time**
   - Can't route to specific model
   - Need one model for all queries

✅ **Simpler deployment is needed**
   - One model is easier to manage
   - Less storage/compute overhead

✅ **Limited resources**
   - Can't train/maintain multiple models
   - Multi-domain is "good enough"

---

## 📈 Performance Insights

### Biggest Gains

1. **Cloud Domain**: +37.4% improvement
   - Technical documentation benefits most from specialization
   - Code/API understanding requires focused training

2. **FiQA Domain**: +19.4% improvement
   - Financial terminology needs domain expertise
   - Market data queries require specialized understanding

3. **ClapNQ Domain**: +8.8% improvement
   - Already performed well with multi-domain
   - Still gains from specialization

4. **Govt Domain**: +7.9% improvement
   - Government language patterns benefit from specialization

### Why Cloud and FiQA Benefit Most

- **Cloud**: Technical terms (APIs, endpoints, code) are very different from general language
- **FiQA**: Financial jargon (ROI, derivatives, portfolios) requires specialized knowledge
- **ClapNQ/Govt**: More similar to general language, so smaller gains

---

## 🔧 Technical Details

### Training Hyperparameters

- **Epochs**: 7 (enough to specialize, not overfit)
- **Learning Rate**: 1e-5 (lower than initial training, for fine-tuning)
- **Batch Size**: 32 (standard for this model size)
- **Loss Function**: MultipleNegativesRankingLoss (contrastive learning)
- **Validation**: Every 200 steps (monitor for overfitting)

### Data Splits

- **Train**: 70% (for learning)
- **Val**: 15% (for monitoring)
- **Test**: 15% (for final evaluation)

### Model Architecture

- **Base**: BGE-base-en-v1.5 (110M parameters)
- **Starting Point**: Phase 1 Epochs 5 (multi-domain pre-trained)
- **Output**: Domain-specific fine-tuned model

---

## 🚀 How to Use Domain-Specific Models

### In Production

```python
# Load the appropriate model based on domain
domain = detect_query_domain(query)  # Your domain detection logic

if domain == "clapnq":
    model = SentenceTransformer("./models/domain_specific_clapnq")
elif domain == "fiqa":
    model = SentenceTransformer("./models/domain_specific_fiqa")
elif domain == "govt":
    model = SentenceTransformer("./models/domain_specific_govt")
elif domain == "cloud":
    model = SentenceTransformer("./models/domain_specific_cloud")
else:
    model = SentenceTransformer("./models/phase1_epochs5")  # Fallback

# Use model for retrieval
embeddings = model.encode([query, *documents])
```

### Evaluation

```bash
# Evaluate all domain-specific models
python evaluate_advanced_models.py --domain_specific

# Evaluate specific domain model
python evaluate_advanced_models.py \
    --model_path ./models/domain_specific_clapnq \
    --domains clapnq
```

---

## 📝 Summary

**Domain-specific experiments** train separate models for each domain, starting from a multi-domain pre-trained model. This two-stage transfer learning approach combines:

1. **General knowledge** from multi-domain training
2. **Specialized knowledge** from domain-specific fine-tuning

**Results**: Average +16.9% Recall@10 and +20.8% nDCG@10 improvement over multi-domain models.

**Best for**: Scenarios where you know the domain in advance and performance is critical.

**Trade-off**: Better performance vs. more complex deployment (4 models instead of 1).

---

*For more details, see `COMPREHENSIVE_EXPERIMENT_RESULTS.md`*

