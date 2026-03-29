# Speculative Decoding Implementation Comparison

Comprehensive comparison of speculative decoding and Multi-Token Prediction (MTP) implementations across vLLM, SGLang, and TensorRT-LLM.

**Last Updated:** 2026-03-28

---

## Table of Contents

1. [Overview & Architecture Comparison](#overview--architecture-comparison)
2. [Speculative Methods Supported](#speculative-methods-supported)
3. [Draft Model Support](#draft-model-support)
4. [Medusa Implementation](#medusa-implementation)
5. [EAGLE/EAGLE3 Implementation](#eagleeagle3-implementation)
6. [Multi-Token Prediction (MTP)](#multi-token-prediction-mtp)
7. [N-Gram Prompt Lookup](#n-gram-prompt-lookup)
8. [Verification & Acceptance](#verification--acceptance)
9. [Configuration & Command-Line Options](#configuration--command-line-options)
10. [Performance Characteristics](#performance-characteristics)
11. [Code Sources & Implementation Details](#code-sources--implementation-details)
12. [Feature Comparison Matrix](#feature-comparison-matrix)

---

## Overview & Architecture Comparison

### vLLM

**Architecture:** Comprehensive speculative decoding framework with 10 methods

**Core Configuration:**
- **File:** `/vllm/config/speculative.py` (863 lines)
- **Class:** `SpeculativeConfig`
- **Design Philosophy:** Unified configuration covering all speculative methods

**Supported Methods:**
```python
SpeculativeMethod = Literal[
    "ngram",                  # N-gram prompt lookup
    "medusa",                 # Medusa heads
    "mlp_speculator",         # MLP Speculator
    "draft_model",            # Full draft model
    "suffix",                 # Suffix decoding (Arctic Inference)
    "eagle",                  # EAGLE fast inference
    "eagle3",                 # EAGLE3 with auxiliary hidden states
    "extract_hidden_states",  # Extract hidden states for EAGLE
    "mtp",                    # Multi-Token Prediction (MTP)
    "ngram_gpu"               # GPU-accelerated N-gram
]
```

**Key Features:**
- Automatic method detection from model name
- Draft model tensor parallelism configuration
- Token tree generation for tree-based speculation
- Probabilistic and strict rejection sampling
- Parallel drafting support

---

### SGLang

**Architecture:** Three-algorithm system with V2 overlap workers

**Core Configuration:**
- **File:** `/sglang/python/sglang/srt/speculative/spec_info.py` (Lines 15-105)
- **Enum:** `SpeculativeAlgorithm`

**Supported Algorithms:**
```python
class SpeculativeAlgorithm(Enum):
    EAGLE = auto()       # Feature-based tree drafting with topk branching
    EAGLE3 = auto()      # Improved EAGLE without feature prediction
    STANDALONE = auto()  # Token-level drafting with separate draft model
    NGRAM = auto()       # Pattern-based drafting from ngram cache
    NONE = auto()        # Disabled
```

**Key Features:**
- V1/V2 worker architecture (overlap scheduling)
- Multi-layer EAGLE support
- Tree-based verification with rejection sampling
- Configurable acceptance thresholds
- Draft model quantization independent from target

---

### TensorRT-LLM

**Architecture:** Multi-mode speculative system with TensorRT optimization

**Core Interface:**
- **File:** `/TensorRT-LLM/tensorrt_llm/_torch/speculative/interface.py` (Lines 41-52)
- **Enum:** `SpeculativeDecodingMode`

**Supported Modes:**
```python
class SpeculativeDecodingMode(IntEnum):
    MTP = auto()                    # Multi-Token Prediction (vanilla)
    MTP_EAGLE = auto()              # MTP with EAGLE heads (2-model)
    MTP_EAGLE_ONE_MODEL = auto()    # MTP with EAGLE heads (1-model)
    EAGLE3 = auto()                 # EAGLE3 (2-model)
    EAGLE3_ONE_MODEL = auto()       # EAGLE3 (1-model)
    NGRAM = auto()                  # N-gram based drafting
    DRAFT_TARGET = auto()           # User-provided draft model
    USER_PROVIDED = auto()          # User-provided drafter
    SAVE_HIDDEN_STATES = auto()     # Hidden state capture mode
    NONE = auto()                   # Disabled
    AUTO = auto()                   # Automatic selection
```

**Key Features:**
- TensorRT-optimized execution
- One-model and two-model variants
- KV cache rewind for 1-model modes
- Lookahead decoding support
- Hidden state capture for training

---

## Speculative Methods Supported

### Methods Comparison Table

| Method | vLLM | SGLang | TensorRT-LLM | Description |
|--------|------|--------|--------------|-------------|
| **Draft Model** | ✅ draft_model | ✅ STANDALONE | ✅ DRAFT_TARGET | Separate smaller draft model |
| **EAGLE** | ✅ eagle | ✅ EAGLE | ✅ EAGLE3 (2-model) | Feature-based speculation |
| **EAGLE3** | ✅ eagle3 | ✅ EAGLE3 | ✅ EAGLE3_ONE_MODEL | Improved EAGLE without feature prediction |
| **Medusa** | ✅ medusa | ❌ No | ✅ MEDUSA | Multiple prediction heads |
| **MTP** | ✅ mtp | ❌ No | ✅ MTP variants | Multi-token prediction |
| **N-Gram** | ✅ ngram, ngram_gpu | ✅ NGRAM | ✅ NGRAM | Prompt lookup |
| **MLP Speculator** | ✅ mlp_speculator | ❌ No | ❌ No | Embedding-based speculation |
| **Suffix Decoding** | ✅ suffix | ❌ No | ❌ No | Arctic Inference tree caching |
| **Lookahead** | ❌ No | ❌ No | ✅ LOOKAHEAD | N-gram based lookahead |
| **Custom** | ✅ extract_hidden_states | ❌ No | ✅ USER_PROVIDED | Custom drafter |

---

## Draft Model Support

### vLLM - Draft Model Configuration

**File:** `/vllm/config/speculative.py` (Lines 338-604)

**Configuration Example:**
```python
speculative_config = {
    "method": "draft_model",
    "model": "Qwen/Qwen2.5-1.5B-Instruct",  # Draft model
    "num_speculative_tokens": 4,
    "draft_tensor_parallel_size": 1,
    "quantization": "fp8",  # Draft model quantization
    "rejection_sample_method": "strict"  # or "probabilistic"
}

llm = LLM(
    model="Qwen/Qwen2.5-7B-Instruct",  # Target model
    speculative_config=speculative_config
)
```

**Key Features:**
- Draft model can have independent TP size (line 681-713)
- Separate quantization for draft model
- Vocab size validation between draft and target
- Parallel drafting mode (line 338-604)

**Draft Model Creation:**
```python
def create_draft_parallel_config(self) -> ParallelConfig:
    """Creates draft parallel config (Lines 733-752)"""
    draft_config = copy.deepcopy(self.target_parallel_config)
    draft_config.tensor_parallel_size = self.draft_tensor_parallel_size
    return draft_config
```

---

### SGLang - STANDALONE Worker

**File:** `/sglang/python/sglang/srt/speculative/standalone_worker.py` (Lines 1-185)

**Configuration Example:**
```bash
python3 -m sglang.launch_server \
    --model Qwen/Qwen2.5-7B-Instruct \
    --speculative-algorithm STANDALONE \
    --speculative-draft-model-path Qwen/Qwen2.5-1.5B-Instruct \
    --speculative-num-steps 4 \
    --speculative-eagle-topk 2 \
    --speculative-num-draft-tokens 7
```

**Key Components:**
- **StandaloneWorker** (Lines 1-109): V1 worker without overlap
- **StandaloneWorkerV2** (Lines 111-185): V2 worker with overlap scheduler
- **Draft model loading:** Lines 32-45 in standalone_worker.py
- **Verification:** Uses same tree sampling as EAGLE

**Memory Reservation:**
```python
# server_args.py Lines 1051-1057
if self.speculative_algorithm == "STANDALONE":
    reserved_mem += 6 * 1024  # 6 GB for draft model + CUDA graphs
```

---

### TensorRT-LLM - Draft-Target Mode

**File:** `/TensorRT-LLM/tensorrt_llm/llmapi/llm_args.py` (Lines 1094-1108)

**Configuration Example:**
```python
from tensorrt_llm.llmapi import DraftTargetDecodingConfig

spec_config = DraftTargetDecodingConfig(
    max_draft_len=4,
    speculative_model="Qwen/Qwen2.5-1.5B-Instruct"
)

llm = LLM(
    model="Qwen/Qwen2.5-7B-Instruct",
    speculative_config=spec_config
)
```

**Key Features:**
- PyTorch backend only (no TensorRT engine for draft)
- Draft model loaded separately
- KV cache management for draft tokens
- Linear drafting loop by default

---

## Medusa Implementation

### vLLM - Medusa Support

**Model Implementation:**
- **File:** `/vllm/model_executor/models/medusa.py` (Lines 1-150+)
- **Class:** `Medusa`

**Architecture:**
```python
class Medusa(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        # Multiple residual blocks (one per head)
        self.blocks = nn.ModuleList([
            ResidualBlock(config, hidden_size, num_hidden_layers)
            for _ in range(config.num_heads)
        ])
        # LM heads for each prediction head
        self.lm_heads = nn.ModuleList([
            ParallelLMHead(vocab_size, hidden_size)
            for i in range(config.num_heads)
        ])
        self.logits_processor = LogitsProcessor(vocab_size, truncated_vocab_size)
```

**Configuration:**
- **File:** `/vllm/transformers_utils/configs/medusa.py` (Lines 1-68)

**Parameters:**
```python
class MedusaConfig(PretrainedConfig):
    model_type = "medusa"

    num_heads: int = 5           # Number of speculation heads
    num_hidden_layers: int = 1   # Layers per head
    max_paths: int = 64
    topk: int = 10
    truncated_vocab_size: int | None = None
```

**Usage Example:**
```python
speculative_config = {
    "method": "medusa",
    "model": "FasterDecoding/medusa-vicuna-7b-v1.3",
    "num_speculative_tokens": 5
}
```

**Medusa Proposer:**
- **File:** `/vllm/v1/spec_decode/medusa.py` (Lines 1-79)
- Each head predicts one token using argmax
- Top-1 selection from each Medusa head
- Tree structure determined by max_paths

---

### TensorRT-LLM - Medusa Support

**Model Implementation:**
- **File:** `/TensorRT-LLM/tensorrt_llm/models/medusa/model.py` (Lines 37-90)

**Architecture:**
```python
class MedusaLayer(Module):
    def __init__(self, hidden_size, hidden_act="silu", dtype=None, mapping=...):
        self.linear = ColumnLinear(hidden_size, hidden_size, ...)
        self.hidden_act = hidden_act

    def forward(self, x):
        return x + ACT2FN[self.hidden_act](self.linear(x))

class MedusaHead(Module):
    def __init__(self, num_layers, hidden_size, vocab_size, ...):
        self.medusa_layers = ModuleList([
            MedusaLayer(...) for _ in range(num_layers)
        ])
        self.lm_head = ColumnLinear(hidden_size, vocab_size, ...)
```

**Configuration:**
- **File:** `/TensorRT-LLM/tensorrt_llm/llmapi/llm_args.py` (Lines 832-847)

```python
class MedusaDecodingConfig(DecodingBaseConfig):
    medusa_choices: Optional[List[List[int]]] = None  # Tree structure
    num_medusa_heads: Optional[int] = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Linear tree only
        self.max_total_draft_tokens = self.max_draft_len
```

**Usage Example:**
```python
speculative_config = MedusaDecodingConfig(
    speculative_model="FasterDecoding/medusa-vicuna-7b-v1.3",
    max_draft_len=63,
    num_medusa_heads=4,
    medusa_choices=[[0], [0, 0], [1], [0, 1], [2], ...]  # Tree paths
)
```

**Medusa Utilities:**
- **File:** `/TensorRT-LLM/tensorrt_llm/runtime/medusa_utils.py` (Lines 1-150)
- `choices_2_paths()` - Converts choices to path format
- `get_medusa_tree()` - Builds tree structure
- `get_medusa_mask()` - Creates acceptance mask

---

## EAGLE/EAGLE3 Implementation

### vLLM - EAGLE Support

**Model Variants:**
- `llama_eagle.py` - LlamaForCausalLM with EAGLE
- `llama_eagle3.py` - Llama with EAGLE3
- `deepseek_eagle.py` - DeepSeekV2/V3 with EAGLE
- `deepseek_eagle3.py` - DeepSeekV2/V3 with EAGLE3
- `minicpm_eagle.py` - MiniCPM with EAGLE
- `mistral_large_3_eagle.py` - Mistral Large 3 with EAGLE3
- `eagle2_5_vl.py` - Vision-Language variant
- `llama4_eagle.py` - Llama4 with EAGLE

**Configuration:**
```python
speculative_config = {
    "method": "eagle",
    "model": "yuhuili/EAGLE-LLaMA3.1-Instruct-8B",
    "num_speculative_tokens": 3,
    "disable_padded_drafter_batch": False,
    "parallel_drafting": False,
}
```

**EAGLE Model Architecture:**
- **File:** `/vllm/model_executor/models/llama_eagle.py` (Lines 1-100+)

```python
class LlamaModel(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        self.embed_tokens = VocabParallelEmbedding(vocab_size, hidden_size)
        self.layers = nn.ModuleList([
            LlamaDecoderLayer(vllm_config, i == 0, ...)  # Skip input_layernorm for layer 0
            for i in range(num_hidden_layers)
        ])
        # Concatenates embeddings + hidden states (2x input for first layer)
        self.fc = ReplicatedLinear(
            input_size=hidden_size * 2,
            output_size=hidden_size,
        )
```

**GPU-Accelerated EAGLE Speculator:**
- **File:** `/vllm/v1/worker/gpu/spec_decode/eagle/speculator.py` (Lines 1-150+)
- Autoregressive draft token generation
- CUDA graph manager for efficient execution
- Handles hidden state passing

---

### SGLang - EAGLE/EAGLE3 Support

**Worker Implementation:**
- **File:** `/sglang/python/sglang/srt/speculative/eagle_worker.py` (Lines 78-200)

**Configuration:**
```bash
# EAGLE-2
python3 -m sglang.launch_server \
    --model meta-llama/Llama-2-7b-chat-hf \
    --speculative-algorithm EAGLE \
    --speculative-draft-model-path lmsys/sglang-EAGLE-llama2-chat-7B \
    --speculative-num-steps 3 \
    --speculative-eagle-topk 4 \
    --speculative-num-draft-tokens 16

# EAGLE-3
python3 -m sglang.launch_server \
    --model meta-llama/Meta-Llama-3.1-8B-Instruct \
    --speculative-algorithm EAGLE3 \
    --speculative-draft-model-path jamesliu1/sglang-EAGLE3-Llama-3.1-Instruct-8B \
    --speculative-num-steps 3 \
    --speculative-eagle-topk 4 \
    --speculative-num-draft-tokens 16
```

**Key Components:**

1. **Draft Phase** (`eagle_draft_cuda_graph_runner.py` - 396 lines):
   - CUDA graph execution for draft model
   - Tree building with topk branching
   - Draft token sampling

2. **Verification Phase** (`eagle_info.py` Lines 216-540):
   ```python
   def verify(self, target_logits: torch.Tensor) -> EagleVerifyOutput:
       # Apply temperature and top-k/top-p filtering
       # Run tree speculative sampling kernel
       # Verify draft tokens against target predictions
       # Return accepted tokens and hidden states
   ```

3. **Tree Building** (`eagle_utils.py` Lines 47-158):
   ```python
   def build_tree_kernel_efficient(
       verified_id: torch.Tensor,
       parent_list: List[torch.Tensor],
       top_scores_index: torch.Tensor,
       draft_tokens: torch.Tensor,
       topk: int,
       spec_steps: int,
       ...
   ):
       # Returns: tree_mask, positions, retrive_index,
       #          retrive_next_token, retrive_next_sibling
   ```

**Multi-Layer EAGLE:**
- **File:** `/sglang/python/sglang/srt/speculative/multi_layer_eagle_worker.py` (748 lines)
- Captures hidden states from multiple layers
- Improved draft quality with richer features

---

### TensorRT-LLM - EAGLE3 Support

**Configuration:**
- **File:** `/TensorRT-LLM/tensorrt_llm/llmapi/llm_args.py` (Lines 850-990)

```python
class EagleDecodingConfig(DecodingBaseConfig):
    eagle_choices: Optional[List[List[int]]] = None  # Static tree structure
    greedy_sampling: Optional[bool] = True
    posterior_threshold: Optional[float] = None
    use_dynamic_tree: Optional[bool] = False
    dynamic_tree_max_topK: Optional[int] = None
    num_eagle_layers: Optional[int] = None
    max_non_leaves_per_layer: Optional[int] = None
    eagle3_one_model: Optional[bool] = True  # 1-model vs 2-model
    eagle3_layers_to_capture: Optional[Set[int]] = None
    eagle3_model_arch: str = "llama3"  # "llama3" or "mistral_large3"
```

**Tree Modes:**
1. **Linear Tree:** One token per layer (default)
2. **Static Tree:** Pre-defined branching via `eagle_choices`
3. **Dynamic Tree:** Runtime-generated with `dynamic_tree_max_topK`

**Usage Example:**
```python
# EAGLE3 One-Model (Linear)
spec_config = EagleDecodingConfig(
    max_draft_len=3,
    speculative_model="yuhuili/EAGLE3-LLaMA3.1-Instruct-8B",
    eagle3_one_model=True
)

# EAGLE3 Static Tree
spec_config = EagleDecodingConfig(
    speculative_model="yuhuili/EAGLE-Vicuna-7B-v1.3",
    max_draft_len=63,
    num_eagle_layers=4,
    max_non_leaves_per_layer=10,
    eagle_choices=[[0], [0, 0], [1], [0, 1], [2], ...]  # 56+ paths
)
```

**Model Architecture:**
- **File:** `/TensorRT-LLM/tensorrt_llm/_torch/models/modeling_speculative.py` (Lines 27-150)

**Eagle3Attention:**
```python
class Eagle3Attention(Attention):
    # First layer: 2x hidden_size input (concat embeddings + hidden states)
    # Subsequent layers: 1x hidden_size (regular)

    if not next_layer_regular:  # First layer
        qkv_proj = Linear(
            2 * hidden_size,  # Double input
            tp_size * q_size + 2 * tp_size * kv_size,
        )
```

**Resource Manager:**
- **File:** `/TensorRT-LLM/tensorrt_llm/_torch/speculative/eagle3.py` (Lines 22-108)
- Manages hidden states across layers
- Tree manager for static/dynamic trees
- Handles one-model KV cache rewind

---

## Multi-Token Prediction (MTP)

### vLLM - MTP Support

**Configuration:**
```python
speculative_config = {
    "method": "mtp",
    "num_speculative_tokens": 4,
    # Uses same model as target
}
```

**Supported MTP Models:**
```python
MTPModelTypes = Literal[
    "deepseek_mtp",           # DeepSeekV3 MTP variant
    "mimo_mtp",               # MiMo MTP
    "glm4_moe_mtp",          # GLM-4 MoE MTP
    "glm4_moe_lite_mtp",     # GLM-4 Lite MTP
    "glm_ocr_mtp",           # GLM OCR MTP
    "ernie_mtp",             # ERNIE MTP
    "nemotron_h_mtp",        # Nemotron-H MTP
    "exaone_moe_mtp",        # Exaone MoE MTP
    "qwen3_next_mtp",        # Qwen3 Next MTP
    "qwen3_5_mtp",           # Qwen3.5 MTP
    "longcat_flash_mtp",     # LongCat Flash MTP
    "pangu_ultra_moe_mtp",   # Pangu Ultra MoE MTP
    "step3p5_mtp",           # Step 3.5 MTP
    "mtp",                   # Generic MTP
]
```

**HF Config Override:**
- **File:** `/vllm/config/speculative.py` (Lines 213-336)
- Automatically sets `n_predict` parameter for MTP models
- Enforces eager execution for DeepSeekV32

---

### TensorRT-LLM - MTP Support

**Configuration:**
- **File:** `/TensorRT-LLM/tensorrt_llm/llmapi/llm_args.py` (Lines 1110-1168)

```python
class MTPDecodingConfig(DecodingBaseConfig):
    num_nextn_predict_layers: int = 1
    use_relaxed_acceptance_for_thinking: bool = False
    relaxed_topk: int = 1
    relaxed_delta: float = 0.0
    use_mtp_vanilla: bool = False
    mtp_eagle_one_model: bool = True

    # Thinking phase tokens (for R1 models)
    begin_thinking_phase_token: int = 128798  # <think>
    end_thinking_phase_token: int = 128799    # </think>
```

**Usage Example:**
```python
spec_config = MTPDecodingConfig(
    num_nextn_predict_layers=1,
    use_relaxed_acceptance_for_thinking=True,
    relaxed_topk=10,
    relaxed_delta=0.01
)

llm = LLM(
    model="nvidia/DeepSeek-R1-FP4",
    speculative_config=spec_config,
)
```

**MTP Modes:**
1. **MTP Vanilla:** Classic multi-token prediction
2. **MTP EAGLE (2-model):** EAGLE heads with separate model
3. **MTP EAGLE (1-model):** EAGLE heads integrated in target model

**Hidden States Manager:**
- **File:** `/TensorRT-LLM/tensorrt_llm/_torch/speculative/mtp.py` (Lines 36-98)
- Manages hidden states pool for predictions
- Relaxed acceptance for thinking phase
- Delta-based threshold tuning

---

## N-Gram Prompt Lookup

### vLLM - N-Gram Implementation

**CPU N-Gram (Numba JIT):**
- **File:** `/vllm/v1/spec_decode/ngram_proposer.py` (Lines 1-120+)

```python
class NgramProposer:
    def __init__(self, vllm_config: VllmConfig):
        self.min_n = vllm_config.speculative_config.prompt_lookup_min  # e.g., 5
        self.max_n = vllm_config.speculative_config.prompt_lookup_max  # e.g., 5
        self.k = vllm_config.speculative_config.num_speculative_tokens
```

**Algorithm:**
1. Search prompt history for longest N-gram match
2. Extract `k` tokens following the match
3. Return as draft tokens

**GPU N-Gram:**
- **File:** `/vllm/v1/spec_decode/ngram_proposer_gpu.py` (Lines 1-100+)

```python
class NgramGPUKernel(nn.Module):
    def _find_first_and_extract_all_n_parallel(
        self,
        token_ids: torch.Tensor,
        seq_lengths: torch.Tensor,
        min_ngram_len: int,
        max_ngram_len: int,
        num_draft_tokens: int,
    ) -> torch.Tensor:
        # Vectorized N-gram search using torch.unfold
        # Parallel matching across all ngram sizes
```

**Configuration:**
```python
speculative_config = {
    "method": "ngram_gpu",
    "num_speculative_tokens": 4,
    "prompt_lookup_max": 8,
    "prompt_lookup_min": 4,
}
```

---

### SGLang - N-Gram Implementation

**File:** `/sglang/python/sglang/srt/speculative/ngram_worker.py` (286 lines)

**Configuration:**
```bash
python3 -m sglang.launch_server \
    --model meta-llama/Meta-Llama-3.1-8B-Instruct \
    --speculative-algorithm NGRAM \
    --speculative-ngram-min-match-window-size 1 \
    --speculative-ngram-max-match-window-size 12 \
    --speculative-ngram-min-bfs-breadth 1 \
    --speculative-ngram-max-bfs-breadth 10 \
    --speculative-ngram-match-type BFS \
    --speculative-ngram-branch-length 18 \
    --speculative-ngram-capacity 10000000
```

**Key Parameters:**
- `min_match_window_size` / `max_match_window_size`: N-gram size range
- `min_bfs_breadth` / `max_bfs_breadth`: Breadth-first search limits
- `match_type`: "BFS" or "PROB" (probability-based)
- `branch_length`: Maximum draft length
- `capacity`: N-gram cache capacity (tokens)

**Environment Variables:**
```bash
SGLANG_NGRAM_FORCE_GREEDY_VERIFY=True/False  # Force greedy verification
```

---

### TensorRT-LLM - N-Gram Implementation

**Configuration:**
- **File:** `/TensorRT-LLM/tensorrt_llm/llmapi/llm_args.py` (Lines 1055-1093)

```python
class NGramDecodingConfig(DecodingBaseConfig):
    max_matching_ngram_size: int = 3
    is_keep_all: bool = True
    is_use_oldest: bool = True
    is_public_pool: bool = True
```

**Usage Example:**
```python
spec_config = NGramDecodingConfig(
    max_draft_len=3,
    max_matching_ngram_size=3,
    is_keep_all=True,
    is_use_oldest=True,
    is_public_pool=True,
)

llm = LLM(
    model="meta-llama/Llama-3.1-8B-Instruct",
    speculative_config=spec_config,
    disable_overlap_scheduler=True,  # NGram incompatible with overlap
)
```

**Pool Manager:**
- **File:** `/TensorRT-LLM/tensorrt_llm/_torch/speculative/ngram.py` (Lines 15-150)
- Maintains pattern-match pairs
- Pattern: ["I", "love"] → Matches: [["apple", ...], ["banana", ...]]
- Keep-all vs keep-oldest strategies

---

## Verification & Acceptance

### vLLM - Rejection Sampling

**File:** `/vllm/v1/worker/gpu/spec_decode/rejection_sampler.py` (Lines 1-150+)

**Strict Rejection Sampling:**
```python
def strict_rejection_sample(
    target_sampled: torch.Tensor,
    draft_sampled: torch.Tensor,
    cu_num_logits: torch.Tensor,
    num_speculative_steps,
) -> tuple[torch.Tensor, torch.Tensor]:
    # Accept draft tokens if all consecutive tokens match target samples
    # Once mismatch found, reject rest and use target sample
```

**Probabilistic Rejection Sampling:**
```python
def _probabilistic_rejection_sample_kernel(...):
    # For each draft token at position i:
    # - Sample u ~ Uniform[0,1]
    # - Accept if: target_prob[i] > u * draft_prob[i]
```

**Configuration:**
```python
speculative_config = {
    ...
    "rejection_sample_method": "strict",  # or "probabilistic"
}
```

---

### SGLang - Tree Speculative Sampling

**File:** `/sglang/sgl-kernel/python/sgl_kernel/speculative.py` (Lines 4-35)

**Kernel:**
```python
def tree_speculative_sampling_target_only(
    predicts: torch.Tensor,           # Output: accepted token IDs
    accept_index: torch.Tensor,       # Output: indices of accepted tokens
    accept_token_num: torch.Tensor,   # Output: number accepted per request
    candidates: torch.Tensor,         # Draft token candidates
    retrive_index: torch.Tensor,      # Tree: parent indices
    retrive_next_token: torch.Tensor, # Tree: next token pointers
    retrive_next_sibling: torch.Tensor,  # Tree: sibling pointers
    uniform_samples: torch.Tensor,    # Random samples for rejection
    target_probs: torch.Tensor,       # Target model probabilities
    draft_probs: torch.Tensor,        # Draft model probabilities
    threshold_single: float = 1.0,    # Single-token acceptance threshold
    threshold_acc: float = 1.0,       # Accumulated threshold
    deterministic: bool = True,
)
```

**Acceptance Logic:**
```python
# Accept if: target_prob ≥ threshold_single × draft_prob
# Accumulated acceptance: product of acceptance ratios ≥ threshold_acc
```

**Configuration:**
```bash
--speculative-accept-threshold-single 1.0  # Default: strict matching
--speculative-accept-threshold-acc 1.0     # Default: strict accumulation
```

---

### TensorRT-LLM - Verification Methods

**Spec Worker Base:**
- **File:** `/TensorRT-LLM/tensorrt_llm/_torch/speculative/interface.py` (Lines 363-627)

**Sample and Accept:**
```python
def _sample_and_accept_draft_tokens_base(
    self,
    target_logits: torch.Tensor,
    draft_tokens: torch.Tensor,
    spec_metadata: SpecMetadata,
) -> tuple[torch.Tensor, torch.Tensor]:
    # Greedy or probabilistic sampling
    # Comparison against draft tokens
    # Returns: (accepted_tokens, num_accepted_tokens)
```

**Force-Accepted Tokens (Testing):**
```python
# Environment variable: TLLM_SPEC_DECODE_FORCE_NUM_ACCEPTED_TOKENS
# Forces fixed acceptance count for debugging
```

---

## Configuration & Command-Line Options

### vLLM Configuration

**Python API:**
```python
from vllm import LLM

speculative_config = {
    "method": "eagle",  # or medusa, ngram, draft_model, etc.
    "model": "yuhuili/EAGLE-LLaMA3.1-Instruct-8B",
    "num_speculative_tokens": 3,
    "disable_padded_drafter_batch": False,
    "parallel_drafting": False,
    "rejection_sample_method": "strict",
    "draft_tensor_parallel_size": 1,
    "quantization": None,
}

llm = LLM(
    model="meta-llama/Llama-3.1-8B-Instruct",
    speculative_config=speculative_config,
)
```

**Command-Line:**
```bash
vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --speculative-config '{"method": "eagle", "model": "...", "num_speculative_tokens": 3}'
```

**Key Parameters:**
- `method`: Speculative method
- `model`: Draft model path or EAGLE weights
- `num_speculative_tokens`: Draft depth
- `rejection_sample_method`: "strict" or "probabilistic"
- `parallel_drafting`: Enable parallel token generation
- `disable_padded_drafter_batch`: Disable batch padding optimization

---

### SGLang Configuration

**Command-Line Arguments:**
```bash
# Core arguments
--speculative-algorithm {EAGLE|EAGLE3|STANDALONE|NGRAM}
--speculative-draft-model-path <path>
--speculative-draft-model-revision <revision>
--speculative-num-steps <int>           # Autoregressive depth
--speculative-eagle-topk <int>          # Branching factor
--speculative-num-draft-tokens <int>    # Max verification capacity
--speculative-accept-threshold-single <float>  # Default: 1.0
--speculative-accept-threshold-acc <float>     # Default: 1.0
--speculative-token-map <path>          # FR-Spec token map
--speculative-attention-mode {prefill|decode}
--enable-multi-layer-eagle              # Multi-layer EAGLE

# NGRAM arguments
--speculative-ngram-min-match-window-size <int>  # Default: 1
--speculative-ngram-max-match-window-size <int>  # Default: 12
--speculative-ngram-min-bfs-breadth <int>        # Default: 1
--speculative-ngram-max-bfs-breadth <int>        # Default: 10
--speculative-ngram-match-type {BFS|PROB}        # Default: BFS
--speculative-ngram-branch-length <int>          # Default: 18
--speculative-ngram-capacity <int>               # Default: 10,000,000
```

**Environment Variables:**
```bash
SGLANG_ENABLE_SPEC_V2=True/False        # Enable V2 overlap scheduler
SGLANG_NGRAM_FORCE_GREEDY_VERIFY=True/False
SGLANG_SIMULATE_ACC_LEN=<float>         # Simulate acceptance
```

---

### TensorRT-LLM Configuration

**Python API:**
```python
from tensorrt_llm.llmapi import (
    EagleDecodingConfig,
    MedusaDecodingConfig,
    MTPDecodingConfig,
    NGramDecodingConfig,
    LookaheadDecodingConfig,
)

# EAGLE3
eagle_config = EagleDecodingConfig(
    max_draft_len=3,
    speculative_model="yuhuili/EAGLE3-LLaMA3.1-Instruct-8B",
    eagle3_one_model=True,
)

# Medusa
medusa_config = MedusaDecodingConfig(
    speculative_model="FasterDecoding/medusa-vicuna-7b-v1.3",
    max_draft_len=63,
    num_medusa_heads=4,
    medusa_choices=[[0], [0, 0], ...],
)

# MTP
mtp_config = MTPDecodingConfig(
    num_nextn_predict_layers=1,
    use_relaxed_acceptance_for_thinking=True,
)

# Lookahead
lookahead_config = LookaheadDecodingConfig(
    max_window_size=4,
    max_ngram_size=4,
    max_verification_set_size=4,
)

llm = LLM(model=..., speculative_config=config)
```

**Base Configuration Parameters:**
```python
class DecodingBaseConfig:
    max_draft_len: Optional[int] = None
    max_total_draft_tokens: Optional[int] = None
    speculative_model: Optional[Union[str, Path]] = None
    max_concurrency: Optional[int] = None  # Disable above batch size
    acceptance_window: Optional[int] = None
    acceptance_length_threshold: Optional[float] = None
    allow_advanced_sampling: bool = False
    draft_len_schedule: Optional[dict[int, int]] = None  # {batch_size: draft_len}
```

---

## Performance Characteristics

### Acceptance Rate Metrics

**vLLM Metrics:**
```python
metrics = llm.get_metrics()
# Available metrics:
# - vllm:spec_decode_num_drafts
# - vllm:spec_decode_num_draft_tokens
# - vllm:spec_decode_num_accepted_tokens
# - vllm:spec_decode_num_accepted_tokens_per_pos
```

**SGLang Metrics:**
- Per-request tracking via `req.spec_accepted_tokens`
- Acceptance tracking in `EagleVerifyOutput`
- `accept_length_per_req_cpu: List[int]`

**TensorRT-LLM Metrics:**
- Acceptance window statistics
- Rolling average over configurable window
- Automatic disable below threshold

### Memory Overhead

**vLLM:**
- Draft model: Full model memory
- EAGLE: Draft model + hidden state buffers
- Medusa: 5 heads × (residual block + lm_head)
- NGram: Minimal (CPU-based cache)

**SGLang:**
- STANDALONE: 6 GB reserved (draft + CUDA graphs)
- EAGLE: 2 GB reserved (draft + CUDA graphs)
- NGRAM: Capacity × token size (default: 10M tokens)

**TensorRT-LLM:**
- Model-dependent: Draft model memory
- KV cache: Additional cache for draft tokens
- Hidden states: For EAGLE3 (num_layers × hidden_size)

### Throughput Improvements

**Typical Speedup Ranges:**

| Method | Speedup | Conditions |
|--------|---------|------------|
| **EAGLE** | 1.5-2.5x | High acceptance rate (>60%) |
| **EAGLE3** | 1.8-3x | One-model efficiency gains |
| **Medusa** | 1.3-2x | Tree-based parallelism |
| **Draft Model** | 1.2-2x | Depends on draft quality |
| **N-Gram** | 1.1-1.5x | Repetitive content |
| **MTP** | 1.2-2x | Model-specific |

---

## Code Sources & Implementation Details

### vLLM Key Files

| Component | File Path | Lines | Key Classes/Functions |
|-----------|-----------|-------|----------------------|
| **Core Config** | `config/speculative.py` | 863 | SpeculativeConfig, SpeculativeMethod |
| **Medusa Model** | `model_executor/models/medusa.py` | 150+ | Medusa, ResidualBlock |
| **Medusa Config** | `transformers_utils/configs/medusa.py` | 68 | MedusaConfig |
| **EAGLE Models** | `model_executor/models/*_eagle*.py` | 100+ each | LlamaModel, DeepseekV2Eagle3DecoderLayer |
| **EAGLE Config** | `transformers_utils/configs/eagle.py` | 93 | EAGLEConfig |
| **EAGLE Proposer** | `v1/spec_decode/eagle.py` | 150+ | SpecDecodeBaseProposer |
| **EAGLE GPU** | `v1/worker/gpu/spec_decode/eagle/speculator.py` | 150+ | EagleSpeculator |
| **MLPSpeculator** | `model_executor/models/mlp_speculator.py` | 100+ | MLPSpeculator |
| **NGram CPU** | `v1/spec_decode/ngram_proposer.py` | 120+ | NgramProposer |
| **NGram GPU** | `v1/spec_decode/ngram_proposer_gpu.py` | 100+ | NgramGPUKernel |
| **Suffix** | `v1/spec_decode/suffix_decoding.py` | 100+ | SuffixDecodingProposer |
| **Rejection** | `v1/worker/gpu/spec_decode/rejection_sampler.py` | 150+ | strict/probabilistic sampling |
| **Metadata** | `v1/spec_decode/metadata.py` | 67 | SpecDecodeMetadata |
| **Metrics** | `v1/spec_decode/metrics.py` | - | Metrics collection |

### SGLang Key Files

| Component | File Path | Lines | Key Classes/Functions |
|-----------|-----------|-------|----------------------|
| **Spec Info** | `srt/speculative/spec_info.py` | 143 | SpeculativeAlgorithm, SpecInputType |
| **EAGLE Worker** | `srt/speculative/eagle_worker.py` | 1032 | EAGLEWorker |
| **EAGLE Worker V2** | `srt/speculative/eagle_worker_v2.py` | 878 | EAGLEWorkerV2 |
| **EAGLE Info** | `srt/speculative/eagle_info.py` | 821 | EagleVerifyInput, EagleDraftInput |
| **EAGLE Utils** | `srt/speculative/eagle_utils.py` | 199 | build_tree_kernel_efficient |
| **Draft Runner** | `srt/speculative/eagle_draft_cuda_graph_runner.py` | 396 | EAGLEDraftCudaGraphRunner |
| **Multi-Layer** | `srt/speculative/multi_layer_eagle_worker.py` | 748 | MultiLayerEagleWorker |
| **NGRAM Worker** | `srt/speculative/ngram_worker.py` | 286 | NGRAMWorker |
| **NGRAM Info** | `srt/speculative/ngram_info.py` | 452 | NGramVerifyInput |
| **Standalone** | `srt/speculative/standalone_worker.py` | 109 | StandaloneWorker |
| **Spec Kernel** | `sgl-kernel/python/sgl_kernel/speculative.py` | 35 | tree_speculative_sampling_target_only |
| **Server Args** | `srt/server_args.py` | 3923-4074 | Speculative arguments |

### TensorRT-LLM Key Files

| Component | File Path | Lines | Key Classes/Functions |
|-----------|-----------|-------|----------------------|
| **Interface** | `_torch/speculative/interface.py` | 41-627 | SpeculativeDecodingMode, SpecMetadata, SpecWorkerBase |
| **EAGLE3 Metadata** | `_torch/speculative/eagle3.py` | 22-200 | Eagle3SpecMetadata, Eagle3ResourceManager |
| **EAGLE3 Model** | `_torch/models/modeling_speculative.py` | 27-150 | Eagle3Attention, Eagle3DecoderLayer |
| **MTP Config** | `llmapi/llm_args.py` | 1110-1168 | MTPDecodingConfig |
| **MTP Manager** | `_torch/speculative/mtp.py` | 36-176 | MTPHiddenStatesManager, MTPSpecMetadata |
| **Medusa Config** | `llmapi/llm_args.py` | 832-847 | MedusaDecodingConfig |
| **Medusa Model** | `models/medusa/model.py` | 37-150 | MedusaLayer, MedusaHead |
| **Medusa Utils** | `runtime/medusa_utils.py` | 1-150 | _medusa_setup, get_medusa_tree |
| **EAGLE Config** | `llmapi/llm_args.py` | 850-990 | EagleDecodingConfig |
| **EAGLE Model Config** | `models/eagle/config.py` | 27-221 | EagleConfig |
| **NGram Config** | `llmapi/llm_args.py` | 1055-1093 | NGramDecodingConfig |
| **NGram Pool** | `_torch/speculative/ngram.py` | 15-150 | NGramPoolManager |
| **Lookahead** | `llmapi/llm_args.py` | 1558-1605 | LookaheadDecodingConfig |
| **Drafting Loops** | `_torch/speculative/drafting_loops.py` | 1-150 | BaseDraftingLoopWrapper, LinearDraftingLoopWrapper |
| **Base Config** | `llmapi/llm_args.py` | 645-756 | DecodingBaseConfig |

---

## Feature Comparison Matrix

| Feature | vLLM | SGLang | TensorRT-LLM |
|---------|------|--------|--------------|
| **Number of Methods** | 10 | 4 | 10+ |
| **Draft Model** | ✅ Yes | ✅ STANDALONE | ✅ DRAFT_TARGET |
| **EAGLE** | ✅ Multiple variants | ✅ V1/V2 | ✅ EAGLE3 |
| **Medusa** | ✅ Full support | ❌ No | ✅ Full support |
| **MTP** | ✅ 13 model types | ❌ No | ✅ 3 variants |
| **N-Gram** | ✅ CPU + GPU | ✅ BFS/PROB | ✅ Pool-based |
| **Lookahead** | ❌ No | ❌ No | ✅ Yes |
| **Suffix Decoding** | ✅ Arctic | ❌ No | ❌ No |
| **MLP Speculator** | ✅ Yes | ❌ No | ❌ No |
| **Tree-Based** | ✅ Token tree | ✅ Topk tree | ✅ Static/dynamic tree |
| **Rejection Sampling** | ✅ Strict + Probabilistic | ✅ Threshold-based | ✅ Greedy + Probabilistic |
| **Parallel Drafting** | ✅ Yes | ❌ No | ❌ No |
| **Draft TP** | ✅ Independent TP size | ✅ Follows target | ✅ Separate config |
| **Draft Quantization** | ✅ Independent | ✅ Independent | ✅ Same as target |
| **One-Model Mode** | ✅ MTP variants | ❌ No | ✅ EAGLE3, MTP |
| **Two-Model Mode** | ✅ All methods | ✅ All methods | ✅ Most methods |
| **CUDA Graphs** | ✅ Per-method | ✅ Draft + Verify | ✅ Automatic capture |
| **Overlap Scheduler** | ❌ No | ✅ SpecV2 | ✅ Yes |
| **Multi-Layer Capture** | ✅ EAGLE variants | ✅ Multi-layer EAGLE | ✅ eagle3_layers_to_capture |
| **Acceptance Tracking** | ✅ Metrics API | ✅ Per-request | ✅ Window-based |
| **Auto-Detection** | ✅ From model name | ❌ Manual | ✅ AUTO mode |
| **KV Cache Rewind** | ✅ Per-method | ✅ For one-model | ✅ For one-model |
| **Hidden State Capture** | ✅ extract_hidden_states | ❌ No | ✅ SAVE_HIDDEN_STATES |
| **Custom Drafter** | ✅ Via plugins | ❌ No | ✅ USER_PROVIDED |
| **Batch Size Adaptation** | ✅ Via config | ✅ Auto-tuning | ✅ draft_len_schedule |
| **Backend Support** | ✅ V1 engine | ✅ V1/V2 workers | ✅ PyTorch/TensorRT |

---

## Best Practices & Recommendations

### Method Selection Guide

**Use EAGLE/EAGLE3 when:**
- High-quality draft model available
- Target model supports hidden state extraction
- Acceptance rate >60% expected
- Memory budget allows draft model

**Use Medusa when:**
- Model has Medusa heads trained
- Tree-based speculation beneficial
- Fixed branching structure acceptable

**Use Draft Model when:**
- Smaller draft model available (e.g., 1.5B for 7B target)
- Vocabulary matches target model
- Straightforward token-level drafting sufficient

**Use N-Gram when:**
- Repetitive content expected
- Minimal memory overhead required
- Simple prompt lookup acceptable

**Use MTP when:**
- Model architecture supports MTP
- One-model efficiency critical
- Thinking phase (R1 models)

### Configuration Tips

**vLLM:**
```python
# High acceptance rate scenario
speculative_config = {
    "method": "eagle",
    "num_speculative_tokens": 5,  # Higher for better speedup
    "parallel_drafting": True,    # Enable parallel generation
}

# Memory-constrained scenario
speculative_config = {
    "method": "ngram_gpu",
    "num_speculative_tokens": 2,  # Lower draft depth
    "prompt_lookup_max": 5,
}
```

**SGLang:**
```bash
# High throughput
SGLANG_ENABLE_SPEC_V2=True python3 -m sglang.launch_server \
    --speculative-algorithm STANDALONE \
    --speculative-num-steps 4 \
    --speculative-eagle-topk 1  # Linear tree for V2

# Quality over speed
python3 -m sglang.launch_server \
    --speculative-algorithm EAGLE \
    --enable-multi-layer-eagle \
    --speculative-eagle-topk 4
```

**TensorRT-LLM:**
```python
# One-model efficiency
eagle_config = EagleDecodingConfig(
    eagle3_one_model=True,
    max_draft_len=3,
)

# Maximum acceptance
eagle_config = EagleDecodingConfig(
    use_dynamic_tree=True,
    dynamic_tree_max_topK=10,
    max_total_draft_tokens=64,
)
```

---

## Conclusion

All three systems provide robust speculative decoding support with different strengths:

**vLLM** excels in:
- Breadth of methods (10 variants)
- Flexibility (parallel drafting, custom drafters)
- Advanced methods (suffix decoding, MLP speculator)

**SGLang** excels in:
- Overlap scheduling (SpecV2)
- Multi-layer EAGLE
- Tree-based verification efficiency

**TensorRT-LLM** excels in:
- TensorRT optimization
- One-model modes (efficiency)
- Lookahead decoding
- Production deployment

The choice depends on:
1. **Model availability:** EAGLE/Medusa heads, draft models
2. **Performance requirements:** Throughput vs latency
3. **Memory constraints:** Full draft model vs lightweight methods
4. **Deployment environment:** Cloud vs edge, single-GPU vs multi-GPU
5. **Inference framework:** Integration with existing infrastructure
