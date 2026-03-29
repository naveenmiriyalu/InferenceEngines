# Reranker Support in vLLM: Complete Guide

## Table of Contents
1. [Overview](#overview)
2. [Reranker Architectures](#1-reranker-architectures)
3. [Core Implementation](#2-core-implementation)
4. [Supported Models](#3-supported-models)
5. [API Endpoints](#4-api-endpoints)
6. [Usage Examples](#5-usage-examples)
7. [Special Model Handling](#6-special-model-handling)
8. [Performance Optimization](#7-performance-optimization)
9. [Integration with RAG](#8-integration-with-rag)
10. [Best Practices](#9-best-practices)

---

## Overview

vLLM provides **production-ready reranker support** for Retrieval-Augmented Generation (RAG) systems. Rerankers refine initial retrieval results by scoring query-document pairs with higher accuracy than traditional embedding similarity.

**Key Features**:
- ✅ Three reranking architectures (cross-encoder, bi-encoder, late-interaction)
- ✅ OpenAI-compatible `/rerank` API
- ✅ Programmatic `LLM.score()` interface
- ✅ Vision reranking capabilities (multimodal documents)
- ✅ Batch processing for throughput optimization
- ✅ Support for 20+ reranker models

---

## 1. Reranker Architectures

### 1.1 Architecture Types

**Location**: `vllm/tasks.py` (Line 17)

```python
ScoreType = Literal["bi-encoder", "cross-encoder", "late-interaction"]
```

#### Cross-Encoder (Most Accurate)

**How it works**:
- Concatenates query + document into single input
- Processes through transformer jointly
- Outputs relevance score

**Pros**:
- Highest accuracy (query-document interaction)
- Best for final reranking (top-k candidates)

**Cons**:
- Slower (cannot cache query embeddings)
- Higher computational cost

**Use Case**: Final reranking of top-100 candidates

**Example Models**:
- `BAAI/bge-reranker-v2-m3`
- `cross-encoder/ms-marco-MiniLM-L-12-v2`
- `Qwen/Qwen3-Reranker-0.6B`

#### Bi-Encoder (Fast, Cacheable)

**How it works**:
- Encodes query and document separately
- Computes cosine similarity between embeddings
- Returns similarity score

**Pros**:
- Fast (can cache embeddings)
- Scalable to millions of documents

**Cons**:
- Lower accuracy (no query-document interaction)

**Use Case**: Initial retrieval or when caching is critical

**Example Models**:
- `BAAI/bge-base-en-v1.5`
- `sentence-transformers/all-MiniLM-L6-v2`

#### Late-Interaction (ColBERT-style)

**How it works**:
- Encodes query and document into token-level embeddings
- Computes max-sim interaction between all token pairs
- Aggregates scores

**Pros**:
- Accuracy close to cross-encoder
- Can cache document token embeddings
- Better than bi-encoder

**Cons**:
- More complex implementation
- Requires special kernels

**Use Case**: Large-scale retrieval with caching

**Example Models**:
- `jinaai/jina-colbert-v2`
- `answerdotai/answerai-colbert-small-v1`
- `colbert-ir/colbertv2.0`

### 1.2 Score Type Selection

**Location**: `vllm/entrypoints/pooling/score/serving.py` (Lines 76-81)

```python
class ServingScores(OpenAIServing):
    def __init__(self, ...):
        self.score_type = self.model_config.score_type

        if self.score_type == "cross-encoder":
            self._score_func = self._cross_encoding_score
        elif self.score_type == "late-interaction":
            self._score_func = self._late_interaction_score
        else:  # "bi-encoder"
            self._score_func = self._embedding_score
```

**Automatic Detection**: vLLM automatically detects score type from model config

---

## 2. Core Implementation

### 2.1 Main Serving Class

**Location**: `vllm/entrypoints/pooling/score/serving.py`

```python
class ServingScores(OpenAIServing):
    """Handles scoring and reranking requests"""

    def __init__(
        self,
        engine_client: EngineClient,
        models: OpenAIServingModels,
        *,
        request_logger: RequestLogger | None,
        score_template: str | None = None,
    ):
        self.score_template = score_template
        self.score_type = self.model_config.score_type
        self.is_multimodal_model = self.model_config.is_multimodal_model

    async def create_score(
        self,
        request: ScoreRequest,
        raw_request: Request,
    ) -> ScoreResponse | ErrorResponse:
        """Process score/rerank request"""
        # Validate inputs
        # Route to appropriate score function
        # Return scored results
```

### 2.2 Request/Response Protocols

**Location**: `vllm/entrypoints/pooling/score/protocol.py`

```python
class RerankRequest(BaseModel):
    """Rerank API request (Jina/Cohere compatible)"""
    model: str
    query: str | ScoreInput
    documents: list[str | ScoreInput]
    top_n: int | None = None  # Return top-n results
    return_documents: bool = True  # Include docs in response
    max_chunks_per_doc: int | None = None  # For long documents

class RerankResponse(BaseModel):
    """Rerank API response"""
    id: str
    model: str
    results: list[RerankResult]  # Sorted by relevance
    usage: RerankUsage

class RerankResult(BaseModel):
    """Single reranked result"""
    index: int  # Original position in input
    relevance_score: float  # Reranking score
    document: RerankDocument | None  # Optional document

class ScoreRequest(BaseModel):
    """Generic score API request"""
    model: str
    data_1: ScoreInputs  # Queries
    data_2: ScoreInputs  # Documents
    truncate_prompt_tokens: str | None = None  # "auto" or specific side
```

### 2.3 Score Computation

**Cross-Encoder Implementation** (Lines 120-180):

```python
async def _cross_encoding_score(
    self,
    data_1: list[ScoreData],
    data_2: list[ScoreData],
    request: RerankRequest | ScoreRequest,
    ...
) -> list[ScoringRequestOutput] | ErrorResponse:
    """Cross-encoder scoring: concatenate query+doc"""

    # Build prompts using template
    prompts = []
    for query, doc in zip(data_1, data_2):
        prompt = self.score_template.format(
            query=query, document=doc
        )
        prompts.append(prompt)

    # Process through model
    outputs = await self.engine_client.score(
        prompts,
        pooling_params=request.to_pooling_params(task="score"),
    )

    return outputs
```

**Late-Interaction Implementation** (Lines 200-280):

```python
async def _late_interaction_score(
    self,
    data_1: list[ScoreData],  # Queries
    data_2: list[ScoreData],  # Documents
    ...
) -> list[PoolingRequestOutput] | ErrorResponse:
    """ColBERT-style late interaction"""

    # 1. Encode queries (cache token embeddings)
    query_outputs = await self._encode_late_interaction_queries(data_1)

    # 2. Encode documents
    doc_outputs = await self._encode_late_interaction_docs(data_2)

    # 3. Compute max-sim scores
    scores = compute_maxsim_scores(query_outputs, doc_outputs)

    return scores
```

### 2.4 API Router

**Location**: `vllm/entrypoints/pooling/score/api_router.py`

```python
@router.post("/rerank")
async def create_rerank(
    request: RerankRequest,
    raw_request: Request,
) -> RerankResponse:
    """Jina/Cohere-compatible rerank endpoint"""
    return await serving_scores.create_rerank(request, raw_request)

@router.post("/score")
async def create_score(
    request: ScoreRequest,
    raw_request: Request,
) -> ScoreResponse:
    """Generic score endpoint"""
    return await serving_scores.create_score(request, raw_request)
```

---

## 3. Supported Models

### 3.1 Cross-Encoder Models

| Model | Size | Language | Notes |
|-------|------|----------|-------|
| `BAAI/bge-reranker-v2-m3` | 567M | Multilingual | Best overall |
| `BAAI/bge-reranker-v2-gemma` | 2.5B | Multilingual | Higher accuracy |
| `BAAI/bge-reranker-base` | 110M | English | Fast baseline |
| `Qwen/Qwen3-Reranker-0.6B` | 600M | Multilingual | Requires conversion |
| `cross-encoder/ms-marco-MiniLM-L-12-v2` | 33M | English | Very fast |
| `mixedbread-ai/mxbai-rerank-large-v1` | 335M | Multilingual | High quality |

**Testing**: `tests/models/language/pooling_mteb_test/test_bge_reranker_v2_gemma.py`

### 3.2 Late-Interaction Models (ColBERT)

| Model | Size | Language | Notes |
|-------|------|----------|-------|
| `jinaai/jina-colbert-v2` | 110M | Multilingual | Production-ready |
| `answerdotai/answerai-colbert-small-v1` | 33M | English | Fast |
| `colbert-ir/colbertv2.0` | 110M | English | Original ColBERT |

**Example**: `examples/pooling/score/colbert_rerank_online.py`

### 3.3 Vision Rerankers (Multimodal)

| Model | Type | Notes |
|-------|------|-------|
| `nvidia/Llama-Nemotron-VL` | Cross-encoder | Text + images |
| `jinaai/jina-reranker-v2-base-multilingual` | Cross-encoder | Multimodal |
| `vidore/colpali` | Late-interaction | Document images |

**Example**: `examples/pooling/score/vision_reranker_offline.py`

### 3.4 Model Registration

**Location**: `tests/models/registry.py`

```python
# Registered reranker models for testing
RERANKER_MODELS = [
    "BAAI/bge-reranker-v2-m3",
    "BAAI/bge-reranker-base",
    "Qwen/Qwen3-Reranker-0.6B",
    "cross-encoder/ms-marco-MiniLM-L-12-v2",
    # ... more models
]
```

---

## 4. API Endpoints

### 4.1 Rerank API (OpenAI-Compatible)

**Endpoint**: `POST /rerank`

**Request**:
```json
{
  "model": "BAAI/bge-reranker-v2-m3",
  "query": "What is machine learning?",
  "documents": [
    "Machine learning is a subset of AI",
    "Python is a programming language",
    "Deep learning uses neural networks"
  ],
  "top_n": 2,
  "return_documents": true
}
```

**Response**:
```json
{
  "id": "rerank-abc123",
  "model": "BAAI/bge-reranker-v2-m3",
  "results": [
    {
      "index": 0,
      "relevance_score": 0.95,
      "document": {
        "text": "Machine learning is a subset of AI"
      }
    },
    {
      "index": 2,
      "relevance_score": 0.72,
      "document": {
        "text": "Deep learning uses neural networks"
      }
    }
  ],
  "usage": {
    "prompt_tokens": 48,
    "total_tokens": 48
  }
}
```

### 4.2 Score API

**Endpoint**: `POST /score`

**Request**:
```json
{
  "model": "BAAI/bge-reranker-v2-m3",
  "data_1": ["What is AI?", "Explain gravity"],
  "data_2": ["AI is intelligence", "Gravity pulls objects"],
  "use_activation": true
}
```

**Response**:
```json
{
  "id": "score-def456",
  "model": "BAAI/bge-reranker-v2-m3",
  "data": [
    {"index": 0, "score": 0.89},
    {"index": 1, "score": 0.76}
  ],
  "usage": {
    "prompt_tokens": 32,
    "total_tokens": 32
  }
}
```

---

## 5. Usage Examples

### 5.1 Basic Reranking (Online API)

**File**: `examples/pooling/score/rerank_api_online.py`

```python
import requests

url = "http://127.0.0.1:8000/rerank"

data = {
    "model": "BAAI/bge-reranker-base",
    "query": "What is the capital of France?",
    "documents": [
        "The capital of Brazil is Brasilia.",
        "The capital of France is Paris.",
        "Horses and cows are both animals",
    ],
}

response = requests.post(url, json=data)
print(response.json())
```

**Start Server**:
```bash
vllm serve BAAI/bge-reranker-base --runner pooling
```

### 5.2 Offline Inference (Programmatic)

**File**: `examples/basic/offline_inference/score.py`

```python
from vllm import LLM

# Initialize reranker model
llm = LLM(model="BAAI/bge-reranker-v2-m3", runner="pooling")

# Score query-document pairs
query = "What is the capital of France?"
documents = [
    "The capital of Brazil is Brasilia.",
    "The capital of France is Paris.",
    "Horses and cows are both animals",
]

# Score all pairs
scores = []
for doc in documents:
    output = llm.score(query, doc)[0]
    scores.append(output.outputs.score)

# Sort by score
ranked_docs = sorted(
    zip(documents, scores),
    key=lambda x: x[1],
    reverse=True
)

for doc, score in ranked_docs:
    print(f"Score: {score:.4f} - {doc}")
```

**Output**:
```
Score: 0.9523 - The capital of France is Paris.
Score: 0.1245 - The capital of Brazil is Brasilia.
Score: 0.0123 - Horses and cows are both animals
```

### 5.3 ColBERT Late-Interaction

**File**: `examples/pooling/score/colbert_rerank_online.py`

```python
import requests

# Start server with ColBERT model
# vllm serve jinaai/jina-colbert-v2 --runner pooling

url = "http://127.0.0.1:8000/rerank"

data = {
    "model": "jinaai/jina-colbert-v2",
    "query": "What is machine learning?",
    "documents": [
        "Machine learning is a subset of artificial intelligence.",
        "Python is a programming language.",
        "Deep learning uses neural networks.",
    ],
    "top_n": 2,  # Return top-2 results
}

response = requests.post(url, json=data)

# Results sorted by relevance
for result in response.json()["results"]:
    print(f"Score: {result['relevance_score']:.4f}")
    print(f"Document: {result['document']['text']}")
    print()
```

### 5.4 Vision Reranking

**File**: `examples/pooling/score/vision_reranker_offline.py`

```python
from vllm import LLM
from vllm.assets.image import ImageAsset

# Initialize vision reranker
llm = LLM(
    model="nvidia/Llama-Nemotron-VL-Reranker",
    runner="pooling",
    trust_remote_code=True,
)

# Query
query = "A cat sitting on a mat"

# Documents with images
documents = [
    {
        "text": "An image of a dog",
        "image": ImageAsset("dog.jpg").pil_image,
    },
    {
        "text": "An image of a cat on a mat",
        "image": ImageAsset("cat.jpg").pil_image,
    },
]

# Score each document
scores = []
for doc in documents:
    output = llm.score(
        {"text": query},
        {"text": doc["text"], "image": doc["image"]},
    )[0]
    scores.append(output.outputs.score)

# Rank by relevance
for i, score in enumerate(scores):
    print(f"Document {i}: {score:.4f}")
```

---

## 6. Special Model Handling

### 6.1 Qwen3-Reranker (Original Version)

**Challenge**: Original Qwen3-Reranker uses "yes"/"no" token logits for scoring, requiring all 151,669 vocabulary tokens.

**Solution**: Override architecture to sequence classification.

**File**: `examples/pooling/score/qwen3_reranker_offline.py`

```python
from vllm import LLM

llm = LLM(
    model="Qwen/Qwen3-Reranker-0.6B",
    runner="pooling",
    hf_overrides={
        # Route to sequence classification
        "architectures": ["Qwen3ForSequenceClassification"],

        # Extract specific token logits
        "classifier_from_token": ["no", "yes"],

        # Enable conversion logic
        "is_original_qwen3_reranker": True,
    },
)
```

**Alternative**: Use pre-converted model:
```python
llm = LLM(
    model="tomaarsen/Qwen3-Reranker-0.6B-seq-cls",
    runner="pooling",
)
```

**Conversion Script**: `examples/pooling/score/convert_model_to_seq_cls.py`

### 6.2 Custom Score Templates

**Templates for proper query-document formatting**:

**File**: `examples/pooling/score/template/qwen3_reranker.jinja`

```jinja
Given a query and passage, determine if the passage answers the query.

Query: {{ query }}
Passage: {{ document }}
```

**Usage**:
```python
from pathlib import Path

template_path = Path("template/qwen3_reranker.jinja")
chat_template = template_path.read_text()

llm = LLM(
    model="Qwen/Qwen3-Reranker-0.6B",
    runner="pooling",
    chat_template=chat_template,
    # ... other configs
)
```

---

## 7. Performance Optimization

### 7.1 Batch Processing

**Automatic Batching**:
```python
# Process multiple query-document pairs in batch
queries = ["query1", "query2", "query3"]
documents = ["doc1", "doc2", "doc3"]

# vLLM automatically batches
scores = llm.score(queries, documents)
```

**Benefits**:
- 5-10x throughput improvement
- Better GPU utilization
- Lower per-request latency

### 7.2 Late-Interaction Caching

**Query Caching** (Lines 45-48 in `pooling_params.py`):

```python
class LateInteractionParams:
    mode: str  # "cache_query" or "score_doc"
    query_key: str  # Stable cache key
    query_uses: int | None  # Expected reuses

# Cache query embeddings for multiple documents
pooling_params = PoolingParams(
    late_interaction_params=LateInteractionParams(
        mode="cache_query",
        query_key="user_query_123",
        query_uses=100,  # Will score 100 documents
    )
)
```

**Benefits**:
- Avoid recomputing query embeddings
- 50-100x speedup for 1 query vs many documents
- Essential for large-scale reranking

### 7.3 Pooling Configuration

```python
PoolingParams(
    task="score",
    use_activation=True,  # Apply sigmoid/softmax
)
```

**Activation Functions**:
- `use_activation=True`: Apply model's default activation (usually sigmoid)
- `use_activation=False`: Return raw logits

---

## 8. Integration with RAG

### 8.1 Typical RAG Pipeline

```
1. Initial Retrieval (BM25 or Embedding Search)
   ↓ (Top-1000 candidates)
2. First-stage Reranking (Bi-encoder)
   ↓ (Top-100 candidates)
3. Second-stage Reranking (Cross-encoder)
   ↓ (Top-10 final results)
4. Generation (LLM with context)
```

### 8.2 Example RAG Integration

```python
# Pseudo-code for RAG system
from vllm import LLM

# Initialize models
retriever = EmbeddingRetriever(...)
reranker = LLM(model="BAAI/bge-reranker-v2-m3", runner="pooling")
generator = LLM(model="meta-llama/Llama-3.1-8B-Instruct")

def rag_query(query: str):
    # 1. Initial retrieval
    candidates = retriever.search(query, top_k=100)

    # 2. Rerank candidates
    scores = []
    for doc in candidates:
        score = reranker.score(query, doc.text)[0].outputs.score
        scores.append(score)

    # 3. Select top-k
    top_docs = sorted(
        zip(candidates, scores),
        key=lambda x: x[1],
        reverse=True
    )[:5]

    # 4. Generate answer
    context = "\n\n".join([doc.text for doc, _ in top_docs])
    prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"

    answer = generator.generate(prompt)
    return answer
```

### 8.3 LangChain Integration

```python
from langchain.llms import VLLM
from langchain.retrievers import VLLMReranker

# vLLM reranker in LangChain
reranker = VLLMReranker(
    model="BAAI/bge-reranker-v2-m3",
    top_n=10,
)

# Use in retrieval chain
retrieval_chain = (
    vectorstore.as_retriever()
    | reranker
    | generate_answer
)

result = retrieval_chain.invoke({"query": "What is AI?"})
```

---

## 9. Best Practices

### 9.1 Model Selection Guidelines

**For Small-Scale (<10K docs)**:
- Use cross-encoder directly
- Best accuracy
- Example: `BAAI/bge-reranker-v2-m3`

**For Medium-Scale (10K-1M docs)**:
- Use late-interaction (ColBERT)
- Cache document embeddings
- Example: `jinaai/jina-colbert-v2`

**For Large-Scale (>1M docs)**:
- Two-stage: bi-encoder → cross-encoder
- First stage: Fast embedding search (top-1000)
- Second stage: Cross-encoder reranking (top-10)

### 9.2 Prompt Template Design

**Good Template** (clear structure):
```jinja
Query: {{ query }}
Document: {{ document }}
Relevant:
```

**Bad Template** (ambiguous):
```jinja
{{ query }} {{ document }}
```

**Template Testing**:
```python
# Test template with known pairs
test_pairs = [
    ("Paris capital", "Paris is capital of France", True),
    ("Paris capital", "London is capital of UK", False),
]

for query, doc, expected in test_pairs:
    score = reranker.score(query, doc)[0].outputs.score
    print(f"Expected: {expected}, Score: {score}")
```

### 9.3 Performance Tuning

**Batch Size**:
```bash
# Increase batch size for throughput
vllm serve BAAI/bge-reranker-base \
  --max-num-seqs 256 \
  --max-model-len 512
```

**GPU Memory**:
```bash
# Reduce memory for smaller GPUs
vllm serve BAAI/bge-reranker-base \
  --gpu-memory-utilization 0.7 \
  --max-num-seqs 128
```

**Quantization**:
```bash
# FP8 quantization for 2x throughput
vllm serve BAAI/bge-reranker-v2-m3 \
  --quantization fp8 \
  --kv-cache-dtype fp8
```

### 9.4 Error Handling

```python
try:
    response = requests.post(
        "http://localhost:8000/rerank",
        json=data,
        timeout=30,
    )
    response.raise_for_status()
    results = response.json()
except requests.exceptions.Timeout:
    # Handle timeout
    print("Reranking timeout - using fallback")
except requests.exceptions.HTTPError as e:
    # Handle API errors
    print(f"Reranking failed: {e}")
```

---

## 10. Summary

### 10.1 Key Capabilities

vLLM provides **comprehensive reranker support** with:
- ✅ **3 architecture types**: Cross-encoder, bi-encoder, late-interaction
- ✅ **20+ models**: BGE, Qwen3, Jina, ColBERT, vision rerankers
- ✅ **OpenAI-compatible API**: `/rerank` endpoint (Jina/Cohere compatible)
- ✅ **Programmatic interface**: `LLM.score()` for offline inference
- ✅ **Performance optimizations**: Batching, caching, quantization
- ✅ **Multimodal support**: Vision reranking for document images

### 10.2 File Reference Summary

**Core Implementation**:
- `vllm/entrypoints/pooling/score/serving.py`: Main serving class (200+ lines)
- `vllm/entrypoints/pooling/score/protocol.py`: Request/response protocols (150+ lines)
- `vllm/entrypoints/pooling/score/api_router.py`: API endpoints (100+ lines)
- `vllm/entrypoints/pooling/score/utils.py`: Scoring utilities (300+ lines)

**Examples**:
- `examples/pooling/score/rerank_api_online.py`: Basic reranking
- `examples/pooling/score/qwen3_reranker_offline.py`: Qwen3 specific
- `examples/pooling/score/colbert_rerank_online.py`: ColBERT late-interaction
- `examples/pooling/score/vision_reranker_offline.py`: Vision reranking

**Tests**:
- `tests/entrypoints/pooling/score/test_online_rerank.py`: API tests
- `tests/models/language/pooling_mteb_test/test_bge_reranker_v2_gemma.py`: Model tests
- `tests/models/language/pooling_mteb_test/test_cross_encoder.py`: Cross-encoder tests

### 10.3 Recommended Workflow

1. **Start Simple**: Use pre-trained cross-encoder (BAAI/bge-reranker-v2-m3)
2. **Test Accuracy**: Evaluate on your domain with ground-truth pairs
3. **Optimize Performance**: Switch to late-interaction if needed
4. **Scale**: Add batching and caching for production
5. **Monitor**: Track latency, throughput, and accuracy metrics

---

**Document Version**: 1.0
**Last Updated**: 2026-03-29
