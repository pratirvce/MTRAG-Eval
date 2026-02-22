# Example: Converting Conversations to BEIR Retrieval Format

This document shows a concrete example of how `conversations2retrieval.py` transforms conversation data into BEIR format.

## Input: Conversation JSON Format

Here's an example input conversation file (`conversations.json`):

```json
[
  {
    "author": "user_123",
    "editor": "editor_456",
    "retriever": {
      "collection": {
        "name": "mt-rag-cloud-elser-512-100-20240502"
      }
    },
    "messages": [
      {
        "speaker": "user",
        "text": "Does IBM offer document databases?",
        "timestamp": 1704067200
      },
      {
        "speaker": "agent",
        "text": "Yes, IBM offers Cloudant, which is a document database service.",
        "contexts": [
          {
            "document_id": "ibmcld_00474-7885-8455",
            "text": "IBM Cloudant is a fully managed NoSQL document database...",
            "feedback": {
              "relevant": {
                "user_123": {
                  "value": "yes",
                  "timestamp": 1704067201
                },
                "editor_456": {
                  "value": "yes",
                  "timestamp": 1704067202
                }
              }
            }
          },
          {
            "document_id": "ibmcld_00513-7-2197",
            "text": "Cloudant supports JSON documents with flexible schemas...",
            "feedback": {
              "relevant": {
                "user_123": {
                  "value": "yes",
                  "timestamp": 1704067203
                }
              }
            }
          }
        ]
      },
      {
        "speaker": "user",
        "text": "So it can store any random JSON object or I need to specify fields in advance?",
        "timestamp": 1704067300
      },
      {
        "speaker": "agent",
        "text": "Cloudant uses a flexible schema, so you can store any JSON object without pre-defining fields.",
        "contexts": [
          {
            "document_id": "ibmcld_00526-7-1750",
            "text": "Cloudant's flexible schema allows storing JSON documents without predefined structure...",
            "feedback": {
              "relevant": {
                "user_123": {
                  "value": "yes",
                  "timestamp": 1704067301
                }
              }
            }
          }
        ]
      },
      {
        "speaker": "user",
        "text": "What if I want to store an image or PDF with a document?",
        "timestamp": 1704067400
      },
      {
        "speaker": "agent",
        "text": "I don't have information about storing images or PDFs in Cloudant.",
        "contexts": [
          {
            "document_id": "ibmcld_00510-7123-9213",
            "text": "Cloudant is designed for JSON documents...",
            "feedback": {
              "relevant": {
                "user_123": {
                  "value": "no",
                  "timestamp": 1704067401
                }
              }
            }
          }
        ]
      }
    ]
  }
]
```

## Processing with Different `-t` Flag Values

### Example 1: `-t -1` (Last Turn Only)

**Command:**
```bash
python scripts/conversations2retrieval.py \
  -i conversations.json \
  -o output_dir \
  -t -1
```

**How it works:**
- `-t -1` means: keep only the last turn (slice from index -1)
- For each user question, only that question is used as the query

**Output: `queries.jsonl`**
```jsonl
{"_id": "user_123_1704067200", "text": "|user|: Does IBM offer document databases?"}
{"_id": "user_123_1704067300", "text": "|user|: So it can store any random JSON object or I need to specify fields in advance?"}
```

**Output: `qrels/dev.tsv`**
```tsv
query-id	corpus-id	score
user_123_1704067200	ibmcld_00474-7885-8455	1
user_123_1704067200	ibmcld_00513-7-2197	1
user_123_1704067300	ibmcld_00526-7-1750	1
```

**Note:** The third question (about images/PDFs) is **filtered out** because all its contexts have `"value": "no"` feedback (unanswerable).

---

### Example 2: `-t -3` (Current Question + Previous Q+A)

**Command:**
```bash
python scripts/conversations2retrieval.py \
  -i conversations.json \
  -o output_dir \
  -t -3
```

**How it works:**
- `-t -3` means: keep the last 3 turns (current question + previous user question + previous agent response)
- This provides conversation context for better retrieval

**Output: `queries.jsonl`**
```jsonl
{"_id": "user_123_1704067200", "text": "|user|: Does IBM offer document databases?"}
{"_id": "user_123_1704067300", "text": "|user|: Does IBM offer document databases?\n|agent|: Yes, IBM offers Cloudant, which is a document database service.\n|user|: So it can store any random JSON object or I need to specify fields in advance?"}
```

**Output: `qrels/dev.tsv`**
```tsv
query-id	corpus-id	score
user_123_1704067200	ibmcld_00474-7885-8455	1
user_123_1704067200	ibmcld_00513-7-2197	1
user_123_1704067300	ibmcld_00526-7-1750	1
```

---

### Example 3: `-t 0` (Full Conversation)

**Command:**
```bash
python scripts/conversations2retrieval.py \
  -i conversations.json \
  -o output_dir \
  -t 0
```

**How it works:**
- `-t 0` means: keep all turns from the beginning (slice from index 0)
- Includes the entire conversation history

**Output: `queries.jsonl`**
```jsonl
{"_id": "user_123_1704067200", "text": "|user|: Does IBM offer document databases?"}
{"_id": "user_123_1704067300", "text": "|user|: Does IBM offer document databases?\n|agent|: Yes, IBM offers Cloudant, which is a document database service.\n|user|: So it can store any random JSON object or I need to specify fields in advance?"}
```

**Output: `qrels/dev.tsv`**
```tsv
query-id	corpus-id	score
user_123_1704067200	ibmcld_00474-7885-8455	1
user_123_1704067200	ibmcld_00513-7-2197	1
user_123_1704067300	ibmcld_00526-7-1750	1
```

---

### Example 4: `-t -1 -q` (Last Turn, Questions Only)

**Command:**
```bash
python scripts/conversations2retrieval.py \
  -i conversations.json \
  -o output_dir \
  -t -1 \
  -q
```

**How it works:**
- `-q` flag: Only include user questions, exclude agent responses
- `-t -1`: Last turn only

**Output: `queries.jsonl`**
```jsonl
{"_id": "user_123_1704067200", "text": "|user|: Does IBM offer document databases?"}
{"_id": "user_123_1704067300", "text": "|user|: So it can store any random JSON object or I need to specify fields in advance?"}
```

**Note:** Agent responses are excluded from the query text.

---

## Key Processing Steps

1. **Extract Collection Name**: From `retriever.collection.name` → groups queries by domain
2. **Build Conversation History**: Accumulates turns as `|speaker|: text` format
3. **Generate Query ID**: `{author}_{timestamp}` for each user question
4. **Apply Turn Selection**: Uses Python list slicing `conversation[turns_to_keep:]`
5. **Extract Qrels**: From agent's `contexts` where `feedback.relevant.{author}.value == "yes"`
6. **Filter Unanswerable**: Removes queries where no contexts have positive feedback

## Output File Structure

```
output_dir/
└── retrieval/
    └── mt-rag-cloud-elser-512-100-20240502/
        ├── queries_turns-1_qonlyFalse.jsonl
        └── qrels/
            └── dev.tsv
```

## BEIR Format Compliance

The output follows BEIR (Benchmarking IR) format:

- **queries.jsonl**: One JSON object per line with `_id` and `text` fields
- **qrels/dev.tsv**: Tab-separated values with columns: `query-id`, `corpus-id`, `score` (1 = relevant, 0 = not relevant)

This format is compatible with BEIR evaluation tools and can be used with various retrieval systems (BM25, dense retrievers, etc.).

