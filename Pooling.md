# Pooling Implementation Comparison

Comprehensive comparison of pooling support for embedding, classification, and reward models across vLLM, SGLang, and TensorRT-LLM.

**Last Updated:** 2026-03-28

---

## Table of Contents

1. [Overview & Architecture Comparison](#overview--architecture-comparison)
2. [Pooling Methods](#pooling-methods)
3. [Embedding Model Support](#embedding-model-support)
4. [Classification Model Support](#classification-model-support)
5. [Reward Model Support](#reward-model-support)
6. [Late Interaction Models (ColBERT)](#late-interaction-models-colbert)
7. [Pooling Configuration](#pooling-configuration)
8. [API and Endpoints](#api-and-endpoints)
9. [Code Sources & Implementation Details](#code-sources--implementation-details)
10. [Feature Comparison Matrix](#feature-comparison-matrix)

---

## Overview & Architecture Comparison

### vLLM

**Architecture:** Comprehensive pooling framework with sequence and token-level methods

**Core Components:**
- **Sequence Pooling:** CLS, LAST, MEAN (aggregate to single vector per sequence)
- **Token Pooling:** ALL, STEP (per-token embeddings/classifications)
- **Pooler Hierarchy:** Abstract base → Seq/Tok poolers → Dispatch pooler

**Key Files:**
- `/vllm/config/pooler.py` (146 lines) - PoolerConfig
- `/vllm/model_executor/layers/pooler/` - Pooler implementations
- `/vllm/pooling_params.py` (225 lines) - PoolingParams API

**Design Philosophy:**
- Task-based routing (embed, classify, score, token_embed, token_classify)
- Matryoshka representation support
- Multiple activation functions
- Chunked prefill handling

---

### SGLang

**Architecture:** Two-method pooling system with sparse embedding support

**Core Components:**
- **Pooling Types:** LAST, CLS (IntEnum: 0, 1)
- **Standard Pooler:** Regular dense embeddings
- **Sparse Pooler:** SPLADE-style sparse embeddings

**Key Files:**
- `/sglang/python/sglang/srt/layers/pooler.py` (131 lines) - Pooler, CrossEncodingPooler
- `/sglang/python/sglang/srt/layers/sparse_pooler.py` (93 lines) - SparsePooler

**Design Philosophy:**
- Minimal, focused implementation
- Per-request Matryoshka dimension truncation
- Optional L2 normalization
- Classification head support

---

### TensorRT-LLM

**Architecture:** Vision-focused pooling with BERT support

**Core Components:**
- **2D Spatial Pooling:** AvgPool2d for vision models
- **Token Pooling:** First token (CLS) extraction for BERT
- **Vision Global Pooling:** Multiple strategies (avg, avgmax, max, token, map)
- **Multimodal Pooling:** Spatial interpolation for vision-language

**Key Files:**
- `/TensorRT-LLM/tensorrt_llm/layers/pooling.py` (38 lines) - AvgPool2d
- `/TensorRT-LLM/tensorrt_llm/functional.py` (3512-3543) - avg_pool2d
- `/TensorRT-LLM/tensorrt_llm/models/bert/model.py` (414-445) - BertPooler

**Design Philosophy:**
- TensorRT-optimized pooling operations
- Vision transformer support
- Multimodal model integration
- Minimal text embedding pooling

---

## Pooling Methods

### Sequence-Level Pooling (Seqwise)

#### vLLM Implementation

**File:** `/vllm/model_executor/layers/pooler/seqwise/methods.py`

**1. CLS Pooling** (Lines 34-45)
```python
class CLSPool(SequencePoolingMethod):
    def forward(self, hidden_states: torch.Tensor,
                pooling_cursor: PoolingCursor) -> torch.Tensor:
        # Extract first token (CLS token)
        return hidden_states[pooling_cursor.first_token_indices_gpu]
```
- **Use Case:** BERT-style models
- **Output:** `[batch_size, hidden_size]`

**2. LAST Pooling** (Lines 48-55)
```python
class LastPool(SequencePoolingMethod):
    def forward(self, hidden_states: torch.Tensor,
                pooling_cursor: PoolingCursor) -> torch.Tensor:
        # Extract last token
        return hidden_states[pooling_cursor.last_token_indices_gpu]
```
- **Use Case:** Decoder-only models (Llama, Qwen)
- **Output:** `[batch_size, hidden_size]`

**3. MEAN Pooling** (Lines 58-82)
```python
class MeanPool(SequencePoolingMethod):
    def forward(self, hidden_states: torch.Tensor,
                pooling_cursor: PoolingCursor) -> torch.Tensor:
        # Efficient mean using cumsum (O(n) complexity)
        cumsum = torch.cumsum(hidden_states, dim=0, dtype=torch.float32)
        start_indices = pooling_cursor.first_token_indices_gpu
        end_indices = pooling_cursor.last_token_indices_gpu
        prompt_lens = pooling_cursor.prompt_lens_cpu.to(device, non_blocking=True)

        return (cumsum[end_indices] - cumsum[start_indices] +
                hidden_states[start_indices]) / prompt_lens.unsqueeze(1)
```
- **Use Case:** Sentence transformers
- **Precision:** Uses float32 for accuracy
- **Output:** `[batch_size, hidden_size]`

---

#### SGLang Implementation

**File:** `/sglang/python/sglang/srt/layers/pooler.py` (Lines 28-78)

**Pooler Class:**
```python
class Pooler(nn.Module):
    def __init__(self, pooling_type: PoolingType, normalize: bool = False):
        self.pooling_type = pooling_type
        self.normalize = normalize

    def forward(self, hidden_states: torch.Tensor,
                forward_batch: ForwardBatch) -> EmbeddingPoolerOutput:
        seq_lens = forward_batch.extend_seq_lens
        cu_seq_lens = torch.cumsum(torch.cat([torch.tensor([0]), seq_lens]), dim=0)

        # LAST pooling (Lines 48-50)
        if self.pooling_type == PoolingType.LAST:
            last_token_flat_indices = cu_seq_lens[1:] - 1
            pooled = hidden_states[last_token_flat_indices]

        # CLS pooling (Lines 51-55)
        elif self.pooling_type == PoolingType.CLS:
            first_token_flat_indices = cu_seq_lens[:-1]
            pooled = hidden_states[first_token_flat_indices]

        # Matryoshka dimension truncation (Lines 59-67)
        if forward_batch.dimensions is not None:
            outputs = []
            for i, dim in enumerate(forward_batch.dimensions):
                outputs.append(pooled[i, :dim])
            return EmbeddingPoolerOutput(embeddings=outputs)

        # L2 normalization (Lines 69-76)
        if self.normalize:
            pooled = F.normalize(pooled, p=2, dim=1)

        return EmbeddingPoolerOutput(embeddings=pooled)
```

**Pooling Types:**
```python
class PoolingType(IntEnum):
    LAST = 0
    CLS = 1
```

---

#### TensorRT-LLM Implementation

**File:** `/TensorRT-LLM/tensorrt_llm/models/bert/model.py` (Lines 414-445)

**BertPooler Class:**
```python
class BertPooler(Module):
    def __init__(self, hidden_size, dtype):
        super().__init__()
        self.dense = Linear(hidden_size, hidden_size, dtype=dtype)
        self.activation = ACT2FN['tanh']

    def forward(self, hidden_states, input_lengths, remove_input_padding):
        if not remove_input_padding:
            # Extract first token (CLS token at position 0)
            first_token_tensor = select(hidden_states, 1, 0)
        else:
            # Calculate first token indices using cumulative sum
            first_token_indices = cumsum(
                concat([
                    0,
                    slice(input_lengths,
                          starts=[0],
                          sizes=(shape(input_lengths) - constant(np.array([1], dtype=np.int32))))
                ]), 0)
            first_token_tensor = index_select(hidden_states, 0, first_token_indices)

        # Apply dense layer and activation
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output
```

**Features:**
- **CLS token extraction** with padding support
- **Dense projection** followed by Tanh activation
- **Use Case:** BERT classification models

---

### Token-Level Pooling (Tokwise)

#### vLLM Implementation

**File:** `/vllm/model_executor/layers/pooler/tokwise/methods.py`

**1. ALL Pooling** (Lines 35-81)
```python
class AllPool(TokenPoolingMethod):
    def forward(self, hidden_states: torch.Tensor,
                pooling_metadata: PoolingMetadata) -> list[torch.Tensor]:
        # Returns all tokens from each sequence
        # Handles chunked prefill with caching
        outputs = []
        for req_idx, (start, end) in enumerate(zip(starts, ends)):
            # Extract sequence tokens
            seq_hidden = hidden_states[start:end]

            # Handle chunked prefill
            if pooling_metadata.pooling_states[req_idx].hidden_states_cache:
                cached = pooling_metadata.pooling_states[req_idx].hidden_states_cache
                seq_hidden = torch.cat([cached, seq_hidden], dim=0)

            outputs.append(seq_hidden)

        return outputs
```
- **Output:** List of tensors (variable length per sequence)
- **Use Case:** Token-level embeddings

**2. STEP Pooling** (Lines 84-116)
```python
class StepPool(AllPool):
    def forward(self, hidden_states: torch.Tensor,
                pooling_metadata: PoolingMetadata) -> list[torch.Tensor]:
        # Get all tokens first
        all_tokens = super().forward(hidden_states, pooling_metadata)

        # Filter by step_tag_id and returned_token_ids
        outputs = []
        for i, data in enumerate(all_tokens):
            token_id = pooling_metadata.get_prompt_token_ids(i)

            if returned_token_ids is not None and len(returned_token_ids) > 0:
                data = data[:, returned_token_ids]

            if step_tag_id is not None:
                data = data[token_id == step_tag_id]

            outputs.append(data)

        return outputs
```
- **Use Case:** Reward models (Qwen2RewardModel)
- **Requires:** Token IDs for filtering

---

### Vision Pooling (2D Spatial)

#### TensorRT-LLM - AvgPool2d

**File:** `/TensorRT-LLM/tensorrt_llm/layers/pooling.py` (Lines 21-38)

```python
class AvgPool2d(Module):
    def __init__(self,
                 kernel_size: Tuple[int],
                 stride: Optional[Tuple[int]] = None,
                 padding: Optional[Tuple[int]] = (0, 0),
                 ceil_mode: bool = False,
                 count_include_pad: bool = True) -> None:
        super().__init__()
        self.kernel_szie = kernel_size
        self.stride = stride
        self.padding = padding
        self.ceil_mode = ceil_mode
        self.count_include_pad = count_include_pad

    def forward(self, input):
        return avg_pool2d(input, self.kernel_szie, self.stride,
                          self.padding, self.ceil_mode, self.count_include_pad)
```

**Functional Implementation:**
```python
# File: /TensorRT-LLM/tensorrt_llm/functional.py (Lines 3512-3543)
def avg_pool2d(input: Tensor,
               kernel_size: Tuple[int],
               stride: Optional[Tuple[int]] = None,
               padding: Optional[Tuple[int]] = (0, 0),
               ceil_mode: bool = False,
               count_include_pad: bool = True) -> Tensor:

    # Uses TensorRT's add_pooling_nd with PoolingType.AVERAGE
    layer = default_trtnet().add_pooling_nd(input.trt_tensor,
                                            trt.PoolingType.AVERAGE,
                                            kernel_size)
    if stride is None:
        stride = kernel_size
    layer.stride_nd = stride
    # ... output processing
```

**Use Cases:**
- Vision transformers (RADIO)
- Multimodal models (LLaVA)
- Spatial downsampling

---

## Embedding Model Support

### vLLM - EmbeddingMixin

**File:** `/vllm/model_executor/models/transformers/pooling.py` (Lines 33-45)

```python
class EmbeddingMixin:
    default_seq_pooling_type: ClassVar[SequencePoolingType] = "CLS"

    def __init_pooler__(self, **kwargs):
        self.pooler = DispatchPooler.for_embedding(
            pooler_config=self.config.get_pooler_config(),
            **kwargs
        )
```

**Supported Models:**
- intfloat/multilingual-e5-small
- Snowflake/snowflake-arctic-embed-m-v1.5
- Alibaba-NLP/gte-Qwen2-1.5B-instruct
- BAAI/bge-base-en-v1.5

**Pooler Configuration:**
```python
# File: /vllm/config/pooler.py
pooling_type: Optional[str] = None  # Resolved to seq_pooling_type or tok_pooling_type
seq_pooling_type: SequencePoolingType = "LAST"  # CLS, LAST, MEAN
use_activation: bool = True  # Apply L2 normalization
dimensions: Optional[int] = None  # Matryoshka dimension reduction
```

**Example Usage:**
```python
from vllm import LLM

llm = LLM(
    model="intfloat/multilingual-e5-small",
    task="embed",
    max_model_len=512,
)

outputs = llm.encode([
    "What is the capital of France?",
    "Paris is the capital of France."
])
```

---

### SGLang - Embedding Models

**File:** `/sglang/docs/supported_models/retrieval_ranking/embedding_models.md`

**Supported Models:**
- Qwen3-Embedding (0.6B, 4B)
- E5 (Llama/Mistral based)
- GTE-Qwen2
- BGE
- CLIP (multimodal)
- Granite, Arcee, Apertus, Exaone4

**Configuration:**
```bash
# Launch server
python3 -m sglang.launch_server \
  --model-path Qwen/Qwen3-Embedding-4B \
  --is-embedding

# Matryoshka dimensions
python3 -m sglang.launch_server \
  --model-path Qwen/Qwen3-Embedding-0.6B \
  --is-embedding \
  --json-model-override-args '{"matryoshka_dimensions": [128, 256, 512, 1024, 1536]}'
```

**API Usage:**
```python
import requests

response = requests.post(
    "http://localhost:30000/v1/embeddings",
    json={
        "model": "Qwen/Qwen3-Embedding-4B",
        "input": "What is the capital of France?",
        "dimensions": 512  # Optional Matryoshka truncation
    }
)
```

**Pooling Implementation:**
```python
# File: /sglang/python/sglang/srt/models/llama_embedding.py (Line 25)
self.pooler = Pooler(pooling_type=PoolingType.LAST, normalize=True)
```

---

### TensorRT-LLM - BERT Embedding

**Model:** BERTModel

**File:** `/TensorRT-LLM/tensorrt_llm/models/bert/model.py` (Lines 350-403)

```python
class BertModel(nn.Module):
    def __init__(self, ...):
        self.embeddings = BertEmbedding(config)
        self.encoder = BertEncoder(...)

        # Pooler configuration
        pooling_type = (PoolingType.CLS if is_embedding else PoolingType.LAST)
        self.pooler = Pooler(pooling_type=pooling_type, normalize=True)

    def forward(self, input_ids, positions, forward_batch, ...):
        hidden_states = self.embeddings(input_ids, positions, forward_batch)
        hidden_states = self.encoder(hidden_states, forward_batch)
        return self.pooler(hidden_states, forward_batch)
```

**Supported Models:**
- bert-base-uncased
- bert-large-uncased
- roberta-base
- roberta-large

---

## Classification Model Support

### vLLM - SequenceClassificationMixin

**File:** `/vllm/model_executor/models/transformers/pooling.py` (Lines 48-102)

```python
class SequenceClassificationMixin:
    default_seq_pooling_type: ClassVar[SequencePoolingType] = "CLS"

    def __init_pooler__(self, **kwargs):
        self.pooler = DispatchPooler.for_seq_cls(
            pooler_config=self.config.get_pooler_config(),
            **kwargs
        )
```

**Supported Models:**
- BertForSequenceClassification (`bert.py:791`)
- RobertaForSequenceClassification (`roberta.py:260`)
- jason9693/Qwen2.5-1.5B-apeach
- papluca/xlm-roberta-base-language-detection

**Classifier Head:**
```python
# File: /vllm/model_executor/layers/pooler/seqwise/heads.py (Lines 100-151)
class ClassifierPoolerHead(SequencePoolerHead):
    def __init__(self, num_labels: int, hidden_size: int, logit_bias: torch.Tensor | None):
        self.classifier = ReplicatedLinear(hidden_size, num_labels, bias=False)
        self.logit_bias = logit_bias  # Subtracted from logits
        self.activation = self._get_activation(num_labels)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        logits = self.classifier(hidden_states)

        if self.logit_bias is not None:
            logits = logits - self.logit_bias

        return self.activation(logits)
```

**Activation Functions:**
- **Multi-class:** Softmax (num_labels > 1)
- **Binary:** Sigmoid (num_labels == 1)

---

### SGLang - Classification Models

**File:** `/sglang/docs/supported_models/retrieval_ranking/classify_models.md`

**Supported Models:**
- LlamaForSequenceClassification
- Qwen2ForSequenceClassification
- Qwen3ForSequenceClassification
- BertForSequenceClassification
- Gemma2ForSequenceClassification

**Implementation Example:**
```python
# File: /sglang/python/sglang/srt/models/llama_classification.py (Lines 29-65)
class LlamaForClassification(nn.Module):
    def __init__(self, config: LlamaConfig, ...):
        self.model = LlamaModel(...)
        self.classification_head = nn.Linear(
            config.hidden_size,
            config.classification_out_size
        )
        self.pooler = Pooler(pooling_type=PoolingType.LAST, normalize=False)

    def forward(self, input_ids, positions, forward_batch, ...):
        hidden_states = self.model(input_ids, positions, forward_batch, ...)
        logits = self.classification_head(hidden_states)
        pooled_logits = self.pooler(logits, forward_batch).embeddings
        return EmbeddingPoolerOutput(pooled_logits)
```

**API Endpoint:**
```
POST /v1/classify

Request:
{
  "model": "model_name",
  "input": "text to classify"
}

Response:
{
  "data": [{
    "label": "Default",
    "probs": [0.565..., 0.434...],
    "num_classes": 2
  }]
}
```

---

### TensorRT-LLM - RoBERTa Classification

**File:** `/TensorRT-LLM/tensorrt_llm/models/bert/model.py` (Lines 448-483)

```python
class RobertaForSequenceClassification(RobertaPreTrainedModel):
    def __init__(self, config: BERTConfig, ...):
        self.roberta = RobertaModel(config, ...)
        self.pooler = BertPooler(config.hidden_size, config.dtype)
        self.classifier = Linear(
            config.hidden_size,
            config.num_labels,
            dtype=config.dtype
        )

    def forward(self, input_ids, input_lengths, ...):
        hidden_states = self.roberta(input_ids, input_lengths, ...)
        pooled = self.pooler(hidden_states, input_lengths, remove_input_padding)
        logits = self.classifier(pooled)
        return logits
```

---

## Reward Model Support

### vLLM - Qwen2RewardModel

**File:** `/vllm/model_executor/models/qwen2_rm.py` (Lines 27-75)

```python
class Qwen2RewardBaseModel(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        self.model = Qwen2Model(vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model"))

        # Score head: Linear → ReLU → Linear
        self.score = nn.Sequential(
            ReplicatedLinear(config.hidden_size, config.hidden_size, bias=False),
            nn.ReLU(),
            ReplicatedLinear(config.hidden_size, 1, bias=False)
        )

        # Token pooling with STEP method
        self.pooler = DispatchPooler(
            {
                "token_embed": TokenEmbeddingPoolerHead(...),
                "token_classify": TokenClassifierPoolerHead(...)
            },
            pooler_config=vllm_config.pooler_config,
        )
```

**Usage:**
```python
# Step pooling filters tokens by step_tag_id
pooling_params = PoolingParams(
    task="token_classify",
    step_tag_id=151668,  # Filter token ID
    returned_token_ids=[0, 1, 2]  # Vocabulary dimensions
)
```

**Example Model:** internlm/internlm2-1_8b-reward

---

### SGLang - Reward Models

**Supported Models:**
- Qwen2ForRewardModel
- Qwen3RewardModel
- LlamaForSequenceClassificationWithNormal_Weights
- InternLM2ForRewardModel

**Implementation:**
```python
# File: /sglang/python/sglang/srt/models/qwen2_rm.py (Line 47)
self.pooler = Pooler(pooling_type=PoolingType.LAST, normalize=False)
```

---

## Late Interaction Models (ColBERT)

### vLLM - ColBERT Support

**File:** `/vllm/model_executor/models/colbert.py` (Lines 34-100)

**ColBERTMixin:**
```python
class ColBERTMixin:
    score_type: ClassVar[ScoreType] = "late-interaction"

    def _init_colbert_components(self):
        # Linear projection for token embeddings
        self.linear = ReplicatedLinear(
            self.config.hidden_size,
            self.config.ColBERT.get('linear_dim', self.config.hidden_size),
            bias=False
        )

    def _build_colbert_pooler(self, **kwargs):
        # Token-level embeddings with linear projection
        return DispatchPooler.for_embedding(
            pooler_config=self.config.get_pooler_config(),
            **kwargs
        )
```

**Supported Models:**
- ColBERTModel
- ColBERTModernBertModel
- ColBERTJinaRobertaModel
- ColQwen3Model

**Late Interaction Scoring:**
- Per-token embeddings
- MaxSim scoring: `max(cosine_similarity(query_token, doc_tokens))`
- Aggregated across all query tokens

---

## Pooling Configuration

### vLLM - PoolerConfig

**File:** `/vllm/config/pooler.py` (Lines 19-146)

```python
@config
class PoolerConfig:
    # Pooling type (resolved to seq or tok)
    pooling_type: Optional[str] = None

    # Sequence pooling method
    seq_pooling_type: SequencePoolingType = "LAST"  # CLS, LAST, MEAN

    # Token-wise pooling method
    tok_pooling_type: TokenPoolingType = "ALL"  # ALL, STEP

    # Apply activation function
    use_activation: bool = True

    # Matryoshka dimension reduction
    dimensions: Optional[int] = None

    # Chunked processing for long sequences
    enable_chunked_processing: bool = True
    max_embed_len: Optional[int] = None

    # Classification logit biases
    logit_bias: Optional[torch.Tensor] = None

    # Step pooling for reward models
    step_tag_id: Optional[int] = None
    returned_token_ids: Optional[list[int]] = None
```

**Valid Parameters by Task:**
```python
{
    "embed": ["dimensions", "use_activation"],
    "classify": ["use_activation"],
    "score": ["use_activation"],
    "token_embed": ["dimensions", "use_activation"],
    "token_classify": ["use_activation"],
}
```

---

### SGLang - Pooling Configuration

**Server Launch:**
```bash
python3 -m sglang.launch_server \
  --model-path <model> \
  --is-embedding  # Enable embedding mode
```

**Pooling Type Selection:**
```python
# File: /sglang/python/sglang/srt/models/bert.py (Lines 369-378)
pooling_type = (
    PoolingType.CLS
    if get_global_server_args().is_embedding
    else PoolingType.LAST
)
```

**Matryoshka Configuration:**
```bash
--json-model-override-args '{"matryoshka_dimensions": [128, 256, 512, 1024, 1536]}'
```

---

### TensorRT-LLM - Pooling Configuration

**BERT Configuration:**
```python
# File: /TensorRT-LLM/tensorrt_llm/models/bert/config.py
class BERTConfig(PretrainedConfig):
    def __init__(self,
                 is_roberta: bool = False,
                 type_vocab_size,
                 pad_token_id=None,
                 num_labels=None,
                 **kwargs):
        self.is_roberta = is_roberta
        self.num_labels = num_labels
```

**Vision Pooling:**
```python
# RADIO Vision Transformer
global_pool: Literal['', 'avg', 'avgmax', 'max', 'token', 'map'] = 'token'
```

---

## API and Endpoints

### vLLM - Embedding API

**File:** `/vllm/entrypoints/pooling/embed/protocol.py`

**Request:**
```python
class EmbeddingCompletionRequest(BaseModel):
    model: str
    input: Union[str, List[str]]
    dimensions: Optional[int] = None  # Matryoshka
    use_activation: Optional[bool] = None
```

**Response:**
```python
class EmbeddingCompletionResponse(BaseModel):
    data: List[EmbeddingData]
    model: str
    usage: UsageInfo

class EmbeddingData(BaseModel):
    embedding: List[float]
    index: int
```

**Example:**
```python
import requests

response = requests.post(
    "http://localhost:8000/v1/embeddings",
    json={
        "model": "intfloat/e5-small",
        "input": "Hello, world!",
        "dimensions": 256
    }
)
```

---

### vLLM - Pooling API

**File:** `/vllm/entrypoints/pooling/pooling/protocol.py`

**Request:**
```python
class PoolingCompletionRequest(BaseModel):
    model: str
    input: Union[str, List[str]]
    task: PoolingTask  # embed, classify, score, token_embed, token_classify
    use_activation: Optional[bool] = None
    dimensions: Optional[int] = None
```

**Example:**
```bash
# Start server
vllm serve <model> --runner pooling

# Classification request
curl -X POST http://localhost:8000/pooling \
  -H "Content-Type: application/json" \
  -d '{
    "model": "model_name",
    "input": "text to classify",
    "task": "classify"
  }'
```

---

### SGLang - Embeddings API

**Endpoint:** `POST /v1/embeddings`

**Request:**
```json
{
  "model": "Qwen/Qwen3-Embedding-4B",
  "input": "What is the capital of France?",
  "dimensions": 512
}
```

**Response:**
```json
{
  "data": [{"embedding": [...]}]
}
```

---

### SGLang - Classification API

**Endpoint:** `POST /v1/classify`

**Request:**
```json
{
  "model": "model_name",
  "input": "text to classify"
}
```

**Response:**
```json
{
  "data": [{
    "label": "Default",
    "probs": [0.565, 0.434],
    "num_classes": 2
  }]
}
```

---

## Code Sources & Implementation Details

### vLLM Key Files

| Component | File Path | Lines | Key Classes |
|-----------|-----------|-------|-------------|
| **Pooler Config** | `config/pooler.py` | 146 | PoolerConfig |
| **Pooling Params** | `pooling_params.py` | 225 | PoolingParams |
| **Seq Methods** | `model_executor/layers/pooler/seqwise/methods.py` | 93 | CLSPool, LastPool, MeanPool |
| **Tok Methods** | `model_executor/layers/pooler/tokwise/methods.py` | 125 | AllPool, StepPool |
| **Seq Heads** | `model_executor/layers/pooler/seqwise/heads.py` | 151 | EmbeddingPoolerHead, ClassifierPoolerHead |
| **Tok Heads** | `model_executor/layers/pooler/tokwise/heads.py` | 133 | TokenEmbeddingPoolerHead, TokenClassifierPoolerHead |
| **Seq Poolers** | `model_executor/layers/pooler/seqwise/poolers.py` | 89 | SequencePooler |
| **Tok Poolers** | `model_executor/layers/pooler/tokwise/poolers.py` | 93 | TokenPooler |
| **Dispatch** | `model_executor/layers/pooler/special.py` | 170 | DispatchPooler, IdentityPooler |
| **Activations** | `model_executor/layers/pooler/activations.py` | 162 | PoolerNormalize, PoolerClassify |
| **Pooling Mixin** | `model_executor/models/transformers/pooling.py` | 102 | EmbeddingMixin, SequenceClassificationMixin |
| **ColBERT** | `model_executor/models/colbert.py` | 100 | ColBERTMixin |
| **Qwen2 RM** | `model_executor/models/qwen2_rm.py` | 75 | Qwen2RewardBaseModel |
| **Metadata** | `v1/pool/metadata.py` | 125 | PoolingCursor, PoolingMetadata |

### SGLang Key Files

| Component | File Path | Lines | Key Classes |
|-----------|-----------|-------|-------------|
| **Pooler** | `srt/layers/pooler.py` | 131 | Pooler, CrossEncodingPooler, PoolingType |
| **Sparse Pooler** | `srt/layers/sparse_pooler.py` | 93 | SparsePooler |
| **Server Args** | `srt/server_args.py` | 2967-2970 | is_embedding flag |
| **ForwardBatch** | `srt/model_executor/forward_batch_info.py` | 230-375 | ForwardBatch (dimensions field) |
| **BERT** | `srt/models/bert.py` | 350-502 | BertModel |
| **Llama Embed** | `srt/models/llama_embedding.py` | 87 | LlamaEmbeddingModel |
| **Classification** | `srt/models/llama_classification.py` | 81 | LlamaForClassification |
| **Qwen2 RM** | `srt/models/qwen2_rm.py` | - | Qwen2ForRewardModel |
| **RoBERTa Sparse** | `srt/models/roberta.py` | 204-282 | RobertaModel (sparse pooling) |

### TensorRT-LLM Key Files

| Component | File Path | Lines | Key Classes |
|-----------|-----------|-------|-------------|
| **AvgPool2d Layer** | `layers/pooling.py` | 21-38 | AvgPool2d |
| **avg_pool2d** | `functional.py` | 3512-3543 | avg_pool2d function |
| **BertPooler** | `models/bert/model.py` | 414-445 | BertPooler |
| **BERT Model** | `models/bert/model.py` | 350-403 | BertModel |
| **RoBERTa** | `models/bert/model.py` | 448-483 | RobertaForSequenceClassification |
| **RADIO** | `_torch/models/modeling_radio.py` | 417-418, 504, 670 | Vision pooling |
| **LLaVA Pooling** | `runtime/multimodal_model_runner.py` | 269-286 | apply_pooling (spatial) |

---

## Feature Comparison Matrix

| Feature | vLLM | SGLang | TensorRT-LLM |
|---------|------|--------|--------------|
| **CLS Pooling** | ✅ SequencePoolingMethod | ✅ PoolingType.CLS | ✅ BertPooler |
| **LAST Pooling** | ✅ SequencePoolingMethod | ✅ PoolingType.LAST | ❌ No |
| **MEAN Pooling** | ✅ SequencePoolingMethod | ❌ No | ❌ No |
| **ALL Token Pooling** | ✅ TokenPoolingMethod | ❌ No | ❌ No |
| **STEP Token Pooling** | ✅ TokenPoolingMethod | ❌ No | ❌ No |
| **2D Spatial Pooling** | ❌ No | ❌ No | ✅ AvgPool2d |
| **Sparse Pooling** | ❌ No | ✅ SparsePooler | ❌ No |
| **Vision Global Pooling** | ❌ No | ❌ No | ✅ RADIO (5 modes) |
| **Multimodal Pooling** | ❌ No | ❌ No | ✅ Spatial interpolation |
| **Matryoshka Support** | ✅ dimensions param | ✅ Per-request truncation | ❌ No |
| **L2 Normalization** | ✅ PoolerNormalize | ✅ normalize param | ❌ No (Tanh) |
| **Classification Head** | ✅ ClassifierPoolerHead | ✅ CrossEncodingPooler | ✅ Linear classifier |
| **Activation Functions** | ✅ 5 types | ✅ Sigmoid/Softmax | ✅ Tanh |
| **Chunked Prefill** | ✅ Hidden states caching | ❌ No | ❌ No |
| **Late Interaction** | ✅ ColBERT support | ❌ No | ❌ No |
| **Reward Models** | ✅ STEP pooling | ✅ Standard pooling | ❌ No |
| **Task-Based Routing** | ✅ 6 tasks | ❌ No | ❌ No |
| **Embedding API** | ✅ /v1/embeddings | ✅ /v1/embeddings | ❌ No |
| **Classification API** | ✅ /pooling | ✅ /v1/classify | ❌ No |
| **Score API** | ✅ /score | ❌ No | ❌ No |
| **Pooler Config** | ✅ PoolerConfig (11 fields) | ✅ PoolingType + normalize | ✅ BERTConfig |
| **Dynamic Dimensions** | ✅ Per-request | ✅ Per-request | ❌ No |
| **Prefix Cache Skip** | ✅ For token tasks | ❌ No | ❌ No |
| **BERT Support** | ✅ Full | ✅ Full | ✅ Full |
| **Llama Support** | ✅ Full | ✅ Full | ❌ No |
| **Qwen Support** | ✅ Full | ✅ Full | ❌ No |
| **Vision Transformers** | ❌ No | ❌ No | ✅ RADIO, CLIP |

---

## Best Practices & Recommendations

### Model Selection

**Use vLLM when:**
- Need comprehensive pooling methods (MEAN, ALL, STEP)
- Matryoshka embeddings required
- Late interaction models (ColBERT)
- Reward models with step pooling
- Task-based API routing

**Use SGLang when:**
- Simple LAST/CLS pooling sufficient
- Per-request dimension truncation needed
- Sparse embeddings (SPLADE)
- Classification models

**Use TensorRT-LLM when:**
- Vision transformers (RADIO)
- Multimodal models (LLaVA)
- BERT classification
- TensorRT-optimized inference

### Configuration Tips

**vLLM - Embedding Model:**
```python
llm = LLM(
    model="intfloat/multilingual-e5-small",
    task="embed",
    pooling_config={
        "seq_pooling_type": "MEAN",  # or CLS, LAST
        "use_activation": True,       # L2 normalization
        "dimensions": 384,            # Matryoshka truncation
    }
)
```

**vLLM - Classification Model:**
```python
llm = LLM(
    model="jason9693/Qwen2.5-1.5B-apeach",
    task="classify",
    pooling_config={
        "seq_pooling_type": "CLS",
        "use_activation": True,  # Softmax
    }
)
```

**SGLang - Matryoshka Embedding:**
```bash
python3 -m sglang.launch_server \
  --model-path Qwen/Qwen3-Embedding-0.6B \
  --is-embedding \
  --json-model-override-args '{"matryoshka_dimensions": [128, 256, 512]}'
```

---

## Conclusion

All three systems provide pooling support with different focuses:

**vLLM** excels in:
- Comprehensive pooling methods
- Task-based routing
- Token-level pooling for reward models
- Matryoshka embeddings
- Late interaction support

**SGLang** excels in:
- Simplicity and efficiency
- Per-request configuration
- Sparse embeddings
- Classification models

**TensorRT-LLM** excels in:
- Vision transformer pooling
- Multimodal models
- TensorRT optimization
- 2D spatial pooling

The choice depends on:
1. **Model type:** Text embedding, classification, reward, or vision
2. **Pooling method:** Simple (CLS/LAST) vs advanced (MEAN, token-level)
3. **Use case:** Retrieval, classification, scoring, or multimodal
4. **Performance:** Throughput vs specialized optimization
5. **API requirements:** Standard embeddings vs task-specific routing
