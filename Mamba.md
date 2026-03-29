# Mamba Architecture Support: vLLM, SGLang, and TensorRT-LLM

## Table of Contents
1. [Executive Summary](#executive-summary)
2. [vLLM Mamba Support](#1-vllm-mamba-support)
3. [SGLang Mamba Support](#2-sglang-mamba-support)
4. [TensorRT-LLM Mamba Support](#3-tensorrt-llm-mamba-support)
5. [Comparative Analysis](#4-comparative-analysis)
6. [Code References](#5-code-references)

---

## Executive Summary

Mamba is a State Space Model (SSM) architecture that provides an alternative to Transformer attention mechanisms. This document compares Mamba support across vLLM, SGLang, and TensorRT-LLM.

**Key Findings:**
- **vLLM** has the most comprehensive Mamba support: Mamba1, Mamba2, hybrid models (Jamba, Zamba2), pipeline parallelism, and LoRA (Mamba1 only)
- **SGLang** provides sophisticated hybrid cache management with separate LRU lists for Mamba and attention blocks
- **TensorRT-LLM** offers production-ready support with limitations (Mamba1 lacks tensor parallelism)

**Architecture Comparison:**
- **State Caching (Mamba)**: O(state_size) memory, not dependent on sequence length
- **KV Caching (Transformer)**: O(seq_len) memory, linear growth with context

---

## 1. vLLM Mamba Support

### 1.1 Mamba1 Implementation

#### Core Files
- **Model**: `vllm/model_executor/models/mamba.py` (283 lines)
- **Mixer**: `vllm/model_executor/layers/mamba/mamba_mixer.py` (400+ lines)

#### Class Structure (mamba.py, Lines 49-99)

```python
class MambaDecoderLayer(nn.Module):
    """Wraps MambaMixer with RMSNorm"""
    def __init__(self, config: MambaConfig):
        self.mixer = MambaMixer(config)
        self.norm = RMSNorm(config.hidden_size, eps=config.layer_norm_epsilon)

class MambaModel(nn.Module):
    """Embedding layer + stacked decoder layers"""
    def __init__(self, config: MambaConfig):
        self.embeddings = VocabParallelEmbedding(...)
        self.layers = nn.ModuleList([
            MambaDecoderLayer(config) for _ in range(config.num_hidden_layers)
        ])

class MambaForCausalLM(nn.Module):
    """Full model with logits processor"""
    def __init__(self, config: MambaConfig):
        self.backbone = MambaModel(config)
        self.lm_head = ParallelLMHead(...)
```

#### MambaMixer Implementation (mamba_mixer.py, Lines 44-100+)

```python
class MambaMixer(nn.Module):
    def __init__(self, config: MambaConfig):
        # Conv1d projection
        self.conv1d = ColumnParallelLinear(
            intermediate_size, intermediate_size, bias=True
        )

        # Input projection (x and z branches)
        self.in_proj = MergedColumnParallelLinear(
            hidden_size, [intermediate_size] * 2, bias=False
        )

        # Time-step projection
        self.dt_proj = ColumnParallelLinear(
            dt_rank, intermediate_size, bias=True
        )

        # Selective projection for dt, B, C
        self.x_proj = RowParallelLinear(
            intermediate_size, dt_rank + state_size * 2, bias=False
        )
```

#### SSM Parameters
- **A matrix** (state transition): Initialized from A_log via weight loader (Lines 131-145)
- **D matrix** (skip connection): Scalar per head (Line 142)
- **Configurable**: state size (`ssm_state_size`), conv kernel size, intermediate size

---

### 1.2 Mamba2 Implementation

#### Core Files
- **Model**: `vllm/model_executor/models/mamba2.py` (296 lines)
- **Mixer**: `vllm/model_executor/layers/mamba/mamba_mixer2.py` (400+ lines)

#### Key Differences from Mamba1 (mamba2.py, Lines 47-95)

```python
class Mamba2DecoderLayer(nn.Module):
    def __init__(self, config: Mamba2Config):
        self.mixer = MambaMixer2(config)  # Uses Mamba2 mixer

        # NO LoRA support (Line 107)
        assert not is_lora_enabled, "LoRA not supported for Mamba2"

        # Additional parameters
        self.n_groups = config.n_groups
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim

        # Dynamic intermediate size
        self.intermediate_size = config.expand * config.hidden_size
```

#### Mixer2 Implementation (mamba_mixer2.py, Lines 1-100+)

```python
class MambaMixer2(nn.Module):
    """Mamba2 mixer with gated RMS norm and group-based computation"""

    def __init__(self, config: Mamba2Config):
        # Gated RMS Norm (custom op)
        self.norm = Mixer2RMSNormGated(config.hidden_size, eps=1e-5)

        # Group-based computation
        self.n_groups = config.n_groups

        # Head dimension awareness for multi-head SSM
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim
```

---

### 1.3 State Management

#### Cache Configuration
**File:** `vllm/config/cache.py` (Lines 91-102)

```python
@dataclass
class CacheConfig:
    mamba_block_size: int = 256         # Block size (multiple of 8)
    mamba_page_size_padded: int | None  # Override for hybrid models
    mamba_cache_dtype: str = "auto"     # Conv and SSM state dtype
    mamba_ssm_cache_dtype: str = "auto" # Separate dtype for temporal state
    mamba_cache_mode: str = "all"       # "all", "align", or "none"
```

#### State Shape Calculation
**File:** `vllm/model_executor/layers/mamba/mamba_utils.py` (Lines 99-150)

**Mamba1 State Shapes:**
```python
# Conv state (transposed)
conv_state_shape = (conv_kernel - 1, intermediate_size // tp_size)

# Temporal state
temporal_state_shape = (intermediate_size // tp_size, state_size)
```

**Mamba2 State Shapes:**
```python
# Conv dimension includes groups
conv_dim = intermediate_size + 2 * n_groups * state_size
conv_state_shape = (conv_kernel - 1 + num_spec, conv_dim // tp_size)

# Multi-head temporal state
temporal_state_shape = (num_heads // tp_size, head_dim, state_size)
```

#### State Dtype Calculator (mamba_utils.py, Lines 19-97)

```python
def mamba1_state_dtype(config) -> tuple[torch.dtype, torch.dtype]:
    """Returns (conv_state_dtype, temporal_state_dtype)"""
    # Auto-detection and customizable precision
    return (conv_dtype, temporal_dtype)

def mamba2_state_dtype(config) -> tuple[torch.dtype, torch.dtype]:
    """Same tuple structure as Mamba1"""
    return (conv_dtype, temporal_dtype)
```

---

### 1.4 SSM Kernels

#### Kernel Locations
`vllm/model_executor/layers/mamba/ops/`

#### 1. Selective Scan Update
**File:** `mamba_ssm.py`

```python
@triton.jit
def _selective_scan_update_kernel(
    state_ptr,      # State tensor
    x_ptr,          # Input
    dt_ptr,         # Time-step
    dt_bias_ptr,    # Time-step bias
    A_ptr,          # State transition matrix
    B_ptr,          # Input matrix
    C_ptr,          # Output matrix
    D_ptr,          # Skip connection
    z_ptr,          # Gate
    # ... additional parameters
):
    """Selective scan update kernel (Triton)"""
    # Supports speculative decoding
    # Variable-length sequences
    # State batch indices for proper routing
```

#### 2. Causal Conv1d
**File:** `causal_conv1d.py`

```python
def causal_conv1d_fn(x, weight, bias=None):
    """Forward: full sequence convolution"""
    # Triton kernel with padding support
    # Speculative decoding aware
    # Block-oriented computation with APC

def causal_conv1d_update(x, conv_state, weight, bias=None):
    """Update: single token update"""
    # Updates conv state with new token
    # Returns output for current token
```

#### 3. SSD (State Space Duality)
**File:** `ssd_combined.py`

**Five-Stage Pipeline (Lines 81-100):**

1. **Chunk cumsum**: Cumulative product of A*dt per chunk
2. **Chunk scan**: Within-chunk computation
3. **State passing**: State updates between chunks
4. **BMM operations**: Chunk-wise attention-like operation
5. **Output combination**: Final output assembly

```python
def ssd_chunk_scan_combined(
    x, dt, A, B, C,
    chunk_size: int,
    D: Optional[Tensor] = None,
    z: Optional[Tensor] = None,
):
    """Combined SSD chunk scan with state passing"""
    # Stage 1: Chunk cumsum
    # Stage 2: Chunk scan
    # Stage 3: State passing
    # Stage 4: BMM
    # Stage 5: Output combination
```

---

### 1.5 Hybrid Architectures

#### Jamba (AI21Labs - Mamba + Transformer)
**File:** `vllm/model_executor/models/jamba.py`

**Architecture (Lines 116-250+):**
```python
class JambaMambaDecoderLayer(nn.Module):
    """Mamba layer with optional MoE"""
    def __init__(self, config, layer_idx):
        self.mamba = MambaMixer(config)
        self.feed_forward = JambaMoE(config) if has_moe else None

class JambaAttentionDecoderLayer(nn.Module):
    """Multi-head attention layer with MoE"""
    def __init__(self, config, layer_idx):
        self.self_attn = Attention(config)
        self.feed_forward = JambaMoE(config)

# Alternating Mamba and Attention layers
class JambaModel(nn.Module):
    def __init__(self, config):
        self.layers = nn.ModuleList([
            JambaMambaDecoderLayer(...) if is_mamba_layer
            else JambaAttentionDecoderLayer(...)
            for layer_idx in range(config.num_hidden_layers)
        ])
```

**Features:**
- LoRA support for Mamba layers (Line 130)
- MoE integration (Lines 59-113)
- Interfaces: `HasInnerState`, `IsHybrid`, `SupportsLoRA`, `SupportsMambaPrefixCaching`

---

#### Zamba2 (Zyphra - Mamba2 + Transformer)
**File:** `vllm/model_executor/models/zamba2.py`

**Architecture:**
```python
class Zamba2LoRA(nn.Module):
    """LoRA layer for shared attention/gated MLP"""
    # Custom LoRA implementation for Zamba2

class Zamba2Attention(nn.Module):
    """Shared attention blocks"""

class Zamba2MLP(nn.Module):
    """Gated MLP blocks"""

class Zamba2DecoderLayer(nn.Module):
    """Hybrid layer with Mamba2 or Attention"""
    def __init__(self, config, layer_idx):
        if is_mamba_layer:
            self.mamba = MambaMixer2(config)
        else:
            self.attn = Zamba2Attention(config)
            self.mlp = Zamba2MLP(config)
```

**Features:**
- Uses `MambaMixer2` for SSM blocks
- Standard Transformer attention for attention blocks
- Interfaces: `HasInnerState`, `IsHybrid`, `SupportsMambaPrefixCaching`

---

### 1.6 Pipeline Parallelism Support

**Both Mamba1 and Mamba2 support PP**

**Implementation (mamba.py, Lines 150-168):**
```python
from vllm.distributed.parallel_state import get_pp_group

class MambaModel(nn.Module):
    def forward(self, input_ids, positions):
        if get_pp_group().is_first_rank:
            # Embedding computation
            hidden_states = self.embeddings(input_ids)
        else:
            # Receive from previous stage
            hidden_states = intermediate_tensors["hidden_states"]

        # Layer computation
        for layer in self.layers:
            hidden_states = layer(hidden_states)

        if not get_pp_group().is_last_rank:
            # Pass to next stage
            return IntermediateTensors({"hidden_states": hidden_states})

        return hidden_states
```

---

### 1.7 LoRA Support

**Mamba1:**
- ✅ LoRA enabled (Line 56 parameter `is_lora_enabled`)
- Passed to MambaMixer (Line 76)
- Affects conv1d weight contiguity (Line 199)

**Mamba2:**
- ❌ NOT supported (Line 107: `assert not is_lora_enabled`)
- Significant limitation vs Mamba1

**Jamba:**
- ✅ Mamba layers: LoRA supported (Lines 130, 142)
- ✅ Attention layers: Standard transformer LoRA

**Zamba2:**
- ✅ Dedicated LoRA layer: `Zamba2LoRA` (Lines 53-100+)
- For both attention and MLP blocks

---

## 2. SGLang Mamba Support

### 2.1 Implementation Overview

**Files:**
- `sglang/python/sglang/srt/layers/attention/mamba/mamba.py`
- `sglang/python/sglang/srt/layers/attention/mamba/mamba2_metadata.py`
- `sglang/python/sglang/srt/mem_cache/mamba_radix_cache.py`

---

### 2.2 Mamba2 Mixer (mamba.py, Lines 1-100+)

```python
class MambaMixer:
    def __init__(self, config):
        # Supports both CUDA and NPU backends (Lines 34-51)
        if is_cuda():
            from sglang.srt.layers.attention.mamba.causal_conv1d import (...)
            from sglang.srt.layers.attention.mamba.causal_conv1d_triton import (...)
        elif is_npu():
            from sgl_kernel_npu.mamba.causal_conv1d import (...)
```

**Custom Weight Loader (Lines 56-130+):**
```python
def mamba_v2_sharded_weight_loader(
    param: nn.Parameter,
    loaded_weight: torch.Tensor,
    shard_id: int,
    tp_size: int,
):
    """Handles Mamba v2 sharding with group replication"""
    # Manages split of x, B, C projections
    # Handles CPU padding for odd TP sizes
```

---

### 2.3 Radix Cache for Mamba

**File:** `sglang/python/sglang/srt/mem_cache/mamba_radix_cache.py`

#### Dual-Mode KV Cache (Lines 1-150+)

```python
class RadixCache:
    """Separate tracking for full and Mamba states"""

    def __init__(self):
        # Line 73: Mamba state value
        self.mamba_value: Optional[torch.Tensor] = None

        # Line 72: Attention KV cache value
        self.value: Optional[torch.Tensor] = None
```

#### Dual LRU Lists (Lines 118-148)

```python
# LRU for attention KV cache
attention_lru = LRUList(mamba=False)

# LRU for Mamba state cache
mamba_lru = LRUList(mamba=True)
```

#### Lock Reference System (Lines 74-79)

```python
# Locks for attention layers
self.full_lock_ref: int = 0

# Locks for Mamba layers
self.mamba_lock_ref: int = 0

# Invariant: full_lock_ref >= mamba_lock_ref
```

---

### 2.4 Differences from vLLM

| Feature | vLLM | SGLang |
|---------|------|--------|
| **Backend Support** | Primarily CUDA | CUDA + NPU |
| **Cache Architecture** | Unified interface | Separate LRU lists |
| **Weight Loading** | Standard sharding | Sophisticated group management |
| **Kernel Selection** | Triton default | Triton or CUDA (flexible) |

---

## 3. TensorRT-LLM Mamba Support

### 3.1 Supported Models

**Implementation Location:**
- `tensorrt_llm/models/mamba/`
- `tensorrt_llm/_torch/modules/mamba/`
- `tensorrt_llm/_torch/pyexecutor/`

**Supported:**
- Mamba1 (basic support)
- Mamba2 (full support)
- Checkpoint conversion from Hugging Face

---

### 3.2 Model Architecture

**File:** `tensorrt_llm/models/mamba/model.py` (Lines 1-100+)

```python
class MambaLayer(Module):
    def __init__(self, config: MambaConfig):
        if config.mamba_version == 'Mamba1':
            # Line 46: Mamba1 disables tensor parallelism
            assert config.mapping.tp_size == 1, "Mamba1 requires TP=1"
            self.ssm = Mamba(config)
        elif config.mamba_version == 'Mamba2':
            # Full TP support
            self.ssm = Mamba2(config)
```

**Key Difference:** Mamba1 explicitly disables tensor parallelism

---

### 3.3 Mamba2Mixer

**File:** `tensorrt_llm/_torch/modules/mamba/mamba2_mixer.py` (Lines 1-100+)

**Parameters:**
- `d_model`: Hidden size
- `d_state`: State size
- `d_conv`: Convolution kernel size
- `nheads`: Number of heads
- `n_groups`: Number of groups
- `head_dim`: Per-head dimension
- `chunk_size`: Chunk size for SSD

**Layout Calculation (Lines 76-79):**
```python
d_inner = head_dim * nheads
d_in_proj = 2 * d_inner + 2 * n_groups * d_state + nheads
conv_dim = d_inner + 2 * n_groups * d_state
```

---

### 3.4 Cache Management

**File:** `tensorrt_llm/_torch/pyexecutor/mamba_cache_manager.py` (Lines 28-100+)

```python
class MambaCacheManager:
    def __init__(self, config, mapping):
        # Manages both conv and SSM states

        # Tensor parallelism sharding
        conv_dim = conv_dim // tp_size      # Line 60
        nheads = nheads // tp_size          # Line 61

        # Allocate cache on CUDA device
        self.cache = torch.zeros(..., device="cuda")  # Line 64

        # Track layer offsets for PP
        self.layer_offsets = [...]  # Lines 72-75

        # Separate dtypes for conv and SSM cache
        self.conv_dtype = ...  # Line 45
        self.ssm_dtype = ...
```

---

### 3.5 Checkpoint Conversion

**File:** `tensorrt_llm/examples/models/core/mamba/convert_checkpoint.py`

**Supported Checkpoint Types:**
- `CheckpointType.hf`: Hugging Face transformers models
- `CheckpointType.state_spaces`: State-spaces/mamba official checkpoints
- `CheckpointType.mistral_inference`: Mistral inference format

**Features (Lines 73-96):**
- Automatic dtype inference
- Weight-only quantization (INT4/INT8)
- Multi-worker parallel conversion
- Pipeline and tensor parallelism support

---

## 4. Comparative Analysis

### 4.1 State Caching vs KV Caching

#### State Caching (Mamba)

| Aspect | Details |
|--------|---------|
| **Memory Usage** | `(batch, seq_len, state_size)` per layer |
| | Much smaller: state_size ≈ 16-128 vs seq_len*head_dim |
| **State Shape** | Mamba1: `(conv_kernel-1, intermediate_size)` + `(intermediate_size, state_size)` |
| | Mamba2: `(num_heads, head_dim, state_size)` |
| **Update Mechanism** | Selective state update via SSM equations |
| | **Not linear in sequence length** |
| **Recurrence** | Stateful: Previous states affect current computation |

#### KV Caching (Transformer Attention)

| Aspect | Details |
|--------|---------|
| **Memory Usage** | `(batch, seq_len, num_heads, head_dim)` per layer |
| | **Linear growth with sequence length** |
| **State Shape** | Key: `(seq_len, num_heads, head_dim)` |
| | Value: `(seq_len, num_heads, head_dim)` |
| **Update Mechanism** | Simple concatenation of new K, V |
| **Recurrence** | Stateless: All previous tokens stored |

#### Hybrid Caching (Jamba/Zamba)
- **Separate caches** for Mamba and attention blocks
- **vLLM**: Unified cache interface with dual LRU tracking
- **SGLang**: Explicit `mamba_value` and `value` fields with separate locks

---

### 4.2 Memory Efficiency Comparison

**Mamba:** O(state_size) - not dependent on sequence length
**Transformer:** O(seq_len) - linear growth

**Factor:** ~100-1000x reduction for long sequences (state_size=16 vs seq_len=16384)

---

### 4.3 Computation Patterns

| Model | Computation | Kernel Type |
|-------|-------------|-------------|
| **Mamba1** | Selective scan over sequence | Triton selective_state_update |
| **Mamba2** | Chunk scan + state passing | SSD pipeline (5-stage) |
| | Intra-chunk attention | BMM operations |
| **Transformer** | QK^T @ V | Flash Attention |

---

### 4.4 Feature Parity

| Feature | Mamba1 | Mamba2 | Jamba | Zamba2 | TRT M1 | TRT M2 |
|---------|--------|--------|-------|--------|--------|--------|
| **Basic Inference** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| **Tensor Parallelism** | Limited | ✅ | ✅ | ✅ | ❌ | ✅ |
| **Pipeline Parallelism** | ✅ | ✅ | ✅ | ✅ | ? | ? |
| **LoRA** | ✅ | ❌ | ✅ | ✅ | ? | ? |
| **Prefix Caching** | ✅ | ✅ | ✅ | ✅ | ? | ? |
| **Speculative Decoding** | ✅ | ✅ | ✅ | ✅ | ? | ? |
| **Quantization** | W8A8 | W8A8 | Partial | Partial | INT4/8 | INT4/8 |

---

### 4.5 Limitations

#### vLLM Limitations
1. **Mamba2 LoRA Gap**: Only Mamba1 and Jamba support LoRA
2. **State Shape Complexity**: Mamba2 requires head-dimension awareness
3. **Hybrid Model Caching**: Config mode complexity ("align" vs "all")

#### TensorRT-LLM Limitations
1. **Mamba1 TP Support**: Mamba1 cannot use tensor parallelism (TP=1 required)
2. **Checkpoint Support**: Limited to three source formats

#### SGLang Limitations
1. **Backend Coverage**: NPU support is separate code path, less mature
2. **Caching Complexity**: Dual LRU system adds complexity

---

### 4.6 Opportunities for Optimization

#### vLLM Opportunities
1. **Mamba2 LoRA**: Similar pattern to Zamba2's approach
2. **State Caching**: Shared state blocks for prefix caching
3. **Graph Capture**: CUDA graph support for conv + selective scan

#### TensorRT-LLM Opportunities
1. **Unified TP Support**: Mamba1 TP would unlock larger models
2. **Custom Kernels**: Extended for quantization awareness

#### Cross-Framework Opportunities
1. **State Persistence**: Pre-compute and cache stable states
2. **Adaptive Chunk Sizes**: Dynamic chunking based on memory
3. **State Compression**: Low-rank factorization of state transitions

---

## 5. Code References

### vLLM Core Mamba Files

| File | Lines | Purpose |
|------|-------|---------|
| `model_executor/models/mamba.py` | 283 | Mamba1 model definition |
| `model_executor/models/mamba2.py` | 296 | Mamba2 model definition |
| `layers/mamba/mamba_mixer.py` | 400+ | Mamba1 mixer/SSM layer |
| `layers/mamba/mamba_mixer2.py` | 400+ | Mamba2 mixer with gated RMS |
| `layers/mamba/mamba_utils.py` | 400+ | State shape/dtype calculators |
| `layers/mamba/abstract.py` | 63 | Base class for Mamba layers |
| `config/cache.py` | 150+ | Cache configuration |

### vLLM Hybrid Models

| File | Lines | Purpose |
|------|-------|---------|
| `model_executor/models/jamba.py` | 500+ | Jamba (Mamba+Attention) |
| `model_executor/models/zamba2.py` | 600+ | Zamba2 (Mamba2+Attention) |

### vLLM Kernels

| File | Purpose |
|------|---------|
| `layers/mamba/ops/mamba_ssm.py` | Selective scan update (Triton) |
| `layers/mamba/ops/causal_conv1d.py` | Causal conv1d (Triton) |
| `layers/mamba/ops/ssd_combined.py` | SSD pipeline orchestration |
| `layers/mamba/ops/ssd_chunk_scan.py` | Chunk scan kernel |
| `layers/mamba/ops/ssd_chunk_state.py` | Chunk state kernel |
| `layers/mamba/ops/ssd_state_passing.py` | State passing kernel |
| `layers/mamba/ops/ssd_bmm.py` | BMM operations kernel |

### SGLang Mamba Files

| File | Lines | Purpose |
|------|-------|---------|
| `srt/layers/attention/mamba/mamba.py` | 300+ | Mamba2 mixer implementation |
| `srt/mem_cache/mamba_radix_cache.py` | 600+ | Hybrid cache management |

### TensorRT-LLM Mamba Files

| File | Lines | Purpose |
|------|-------|---------|
| `models/mamba/model.py` | 200+ | Model definition |
| `models/mamba/config.py` | 150+ | Configuration |
| `_torch/modules/mamba/mamba2_mixer.py` | 300+ | Mamba2 mixer |
| `_torch/pyexecutor/mamba_cache_manager.py` | 200+ | Cache management |

---

## Summary

**vLLM** provides the most comprehensive Mamba support with Mamba1, Mamba2, hybrid architectures (Jamba, Zamba2), pipeline parallelism, and LoRA for Mamba1. **SGLang** excels at hybrid cache management with sophisticated radix cache and dual LRU lists. **TensorRT-LLM** offers production-ready inference but with trade-offs like Mamba1 lacking tensor parallelism.

The key opportunity for all frameworks is extending Mamba2 LoRA support and optimizing state compression for ultra-long context scenarios where Mamba's O(state_size) memory advantage shines.

---

**Document Version:** 1.0
**Last Updated:** 2026-03-29
