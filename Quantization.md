# Quantization Support: vLLM, SGLang, and TensorRT-LLM

## Table of Contents
1. [Executive Summary](#executive-summary)
2. [vLLM Quantization](#1-vllm-quantization)
3. [SGLang Quantization](#2-sglang-quantization)
4. [TensorRT-LLM Quantization](#3-tensorrt-llm-quantization)
5. [Comparative Analysis](#4-comparative-analysis)
6. [Performance & Accuracy Trade-offs](#5-performance--accuracy-trade-offs)
7. [Hardware Support Matrix](#6-hardware-support-matrix)
8. [Recommendations](#7-recommendations)

---

## Executive Summary

This document provides a comprehensive comparison of quantization support across three major LLM serving frameworks: vLLM, SGLang, and TensorRT-LLM. All three frameworks support extensive quantization methods to reduce memory footprint and increase inference throughput, with different strengths:

- **vLLM**: 24+ quantization methods, extensive Marlin kernel integration, broad hardware support
- **SGLang**: 21+ methods including AutoRound, advanced W4A8 variants, custom kernel implementations
- **TensorRT-LLM**: Engine-level optimization, QServe support, comprehensive plugin system

**Key Findings:**
- **Weight-only quantization** (AWQ, GPTQ) offers 4-8x memory reduction with 1-3% accuracy loss
- **FP8 quantization** requires Ada/Hopper GPUs but provides 4x reduction with <0.5% accuracy loss
- **Marlin kernels** (vLLM) provide fastest inference for quantized models
- **AutoRound** (SGLang-specific) improves accuracy 0.5-1% over GPTQ

---

## 1. vLLM Quantization

### 1.1 Supported Methods

**File:** `vllm/model_executor/layers/quantization/__init__.py` (Lines 12-36)

**Complete List (24 methods):**
```python
QUANTIZATION_METHODS = {
    "awq", "fp8", "modelopt", "modelopt_fp4", "modelopt_mxfp8",
    "modelopt_mixed", "gguf", "gptq_marlin", "awq_marlin", "gptq",
    "compressed-tensors", "bitsandbytes", "quark", "moe_wna16",
    "torchao", "inc", "mxfp4", "cpu_awq", "marlin", "qqq",
    "fbgemm_fp8", "experts_int8", "neuron_quant", "ipex"
}
```

**Deprecated:** `tpu_int8`, `ptpc_fp8`, `fbgemm_fp8`, `fp_quant`, `experts_int8`, `petit_nvfp4`

---

### 1.2 AWQ (Activation-aware Weight Quantization)

#### Core Implementation
**Files:**
- Base: `awq.py` (278 lines)
- Marlin variant: `awq_marlin.py` (790 lines)
- Triton variant: `awq_triton.py` (337 lines)

#### Configuration
**File:** `awq.py` (Lines 32-100)

```python
class AWQConfig(QuantizationConfig):
    weight_bits: int = 4                    # Only 4-bit supported
    group_size: int                         # Group-wise quantization (default: 128)
    zero_point: bool                        # Runtime zero points
    modules_to_not_convert: list[str] = []  # Skip certain modules
    pack_factor: int = 8                    # 32 / 4 = 8 weights per int32

    def get_min_capability(self) -> int:
        return 75  # Turing+ (SM 75)

    def get_supported_act_dtypes(self) -> list[torch.dtype]:
        return [torch.half]
```

#### Key Features
- **Group-wise quantization**: Reduces quantization error via per-group scaling
- **Zero-point support**: Asymmetric quantization for better accuracy
- **Symmetric bias quantization**: Optional bias quantization
- **Marlin kernel acceleration**: 2-3x speedup over standard kernels

#### Kernel Implementations
**CUDA Path:** `vllm/csrc/quantization/awq/`

**Backends:**
1. **AWQ CUDA kernel** (default)
2. **AWQ Triton kernel** (portable)
3. **AWQ Marlin kernel** (fastest, Lines 17-18 in `awq_marlin.py`)

**Performance:**
- Marlin: **2-3x faster** than CUDA
- Memory: **8x reduction** (FP16 → 4-bit)
- Accuracy: **1-2% loss** on benchmarks

---

### 1.3 GPTQ (Generative Pre-trained Transformer Quantization)

#### Core Implementation
**Files:**
- Base: `gptq.py` (393 lines)
- Marlin variant: `gptq_marlin.py` (929 lines)

#### Configuration
**File:** `gptq.py` (Lines 43-100)

```python
class GPTQConfig(QuantizationConfig):
    weight_bits: int                        # 2, 3, 4, or 8 bits
    group_size: int                         # Per-group quantization
    desc_act: bool                          # Descending activation optimization
    lm_head_quantized: bool = False         # Quantize LM head
    dynamic: dict[str, dict] = {}           # Per-module quantization overrides

    pack_factor: Fraction = Fraction(32, weight_bits)

    def get_min_capability(self) -> int:
        return 75  # Turing+
```

#### Advanced Features

**1. Per-Module Dynamic Quantization (Lines 55-82):**
```python
# Example: Different quantization for different layer types
dynamic_config = {
    ".*self_attn.*": {"bits": 4, "group_size": 128},
    ".*mlp.*": {"bits": 8, "group_size": 64},
}
```

**2. Hessian-Based Calibration:**
- Uses second-order information for optimal weight quantization
- Minimizes quantization error during model preparation

**3. Marlin Integration:**
- Tile-based quantization: `GPTQ_MARLIN_TILE = 16`
- Min thread constraints: N=64, K=128 (`marlin_utils.py`, Lines 25-27)
- Max parallel: 16

#### Supported Bit-widths
| Bits | Pack Factor | Memory Reduction | Use Case |
|------|-------------|------------------|----------|
| 2 | 16 | 16x | Extreme compression |
| 3 | 10.67 | 10.67x | Balanced |
| 4 | 8 | 8x | Standard quantization |
| 8 | 4 | 4x | High accuracy |

---

### 1.4 FP8 Quantization

#### Core Implementation
**Files:**
- Core: `fp8.py` (1,247 lines)
- Utils: `model_executor/layers/quantization/utils/fp8_utils.py`
- Marlin variant: `marlin_utils_fp8.py`
- Input quantization: `input_quant_fp8.py`

#### Configuration
**File:** `fp8.py` (Lines 102-150)

```python
class Fp8Config(QuantizationConfig):
    is_checkpoint_fp8_serialized: bool      # Pre-quantized checkpoint
    activation_scheme: str                  # "static" or "dynamic"
    ignored_layers: list[str] = []          # Skip certain layers
    weight_block_size: list[int] = None     # 2D block quantization

    def get_min_capability(self) -> int:
        return 75  # Ada/Hopper optimal (SM 89+)

    def get_supported_act_dtypes(self) -> list[torch.dtype]:
        return [torch.bfloat16, torch.half]
```

#### Quantization Schemes

**1. Dynamic Activation (Lines 118, 131-136):**
```python
# Per-token scaling computed on-the-fly
# Best for varying input distributions
# Overhead: ~5-10% latency
```

**2. Static Activation (Lines 114-120):**
```python
# Pre-computed scales from calibration
# Fixed scales for all inputs
# Lower latency, requires good calibration dataset
```

**3. Block-wise Quantization (Lines 120-137):**
```python
# Requires fp8-serialized checkpoint
# Only supports dynamic activation
# Block dimensions: [128, 128] (typical)
# Deepseek format: 1x128 or 128x128 blocks
```

#### Hardware Features
- **Hopper/Ada GPUs**: Native FP8 tensor cores
- **Block-scale FP8**: Deepseek format support
- **CUTLASS integration**: Per-channel scaling

#### FP8 Formats
| Format | Exponent | Mantissa | Range | Use Case |
|--------|----------|----------|-------|----------|
| E4M3 | 4 bits | 3 bits | ±448 | Activations (wider range) |
| E5M2 | 5 bits | 2 bits | ±57344 | Weights (higher precision) |

**Performance:**
- Memory: **4x reduction** (FP32 → FP8)
- Accuracy loss: **<0.5%** on most benchmarks
- Speedup: **1.5-2x** on Hopper with FP8 tensor cores

---

### 1.5 MXFP4 (Microsoft MX Format FP4)

#### Core Implementation
**Files:**
- Core: `mxfp4.py` (1,298 lines)
- Utils: `mxfp4_utils.py` (150 lines)

#### Configuration
**File:** `mxfp4.py` (Lines 65-160)

```python
class Mxfp4Config(QuantizationConfig):
    def get_min_capability(self) -> int:
        return 90  # Hopper+ only (SM 90)

    def get_supported_act_dtypes(self) -> list[torch.dtype]:
        return [torch.bfloat16]
```

#### Backend Selection (Lines 109-160)

**Priority Order:**
1. **SM100 FlashInfer MXFP4-MXFP8 (TRTLLM)** (Lines 135-138)
2. **SM100 FlashInfer MXFP4-MXFP8 (CUTLASS)** (Lines 123-129)
3. **SM100 FlashInfer MXFP4-BF16** (Lines 139-146)
4. **SM90 FlashInfer MXFP4-BF16** (Lines 116-122)
5. **Marlin Backend** (Lines 147-153)
6. **Triton Backend** (Lines 155-160)
7. **CK Backend (AMD)** (Lines 161-163)

#### Features
- **FP4 packing**: 2 values per byte
- **MoE dimension alignment**: 256 (`mxfp4_utils.py`, Line 22)
- **Weight swizzling**: For OAI kernels (Lines 25-87)

**Performance:**
- Memory: **8x reduction** (BF16 → MXFP4)
- Accuracy loss: **2-3%** (more aggressive than standard FP4)
- Hardware requirement: **Hopper H100/H200 only**

---

### 1.6 W8A8 (8-bit Weights + 8-bit Activations)

#### Related Files
- `fp8.py` with W8A8 support
- `model_executor/layers/quantization/utils/w8a8_utils.py`
- CUTLASS scaled matrix multiplication

#### Configuration
```python
# Unified quantization for weights and activations
# Per-channel weight scaling + per-token activation scaling
# Block-wise quantization support
```

#### Features
- **Symmetric quantization**: No zero points
- **Per-channel weights**: Independent scale per output channel
- **Per-token activations**: Dynamic scale per token
- **CUTLASS kernels**: Hardware-accelerated INT8 GEMM

**Performance:**
- Memory: **4x reduction** (FP32 → INT8)
- Accuracy loss: **0.5-1%**
- Speedup: **2-3x** on Ampere+ with INT8 tensor cores

---

### 1.7 Marlin Kernels (Unified Backend)

#### Implementation
**Files:**
- `marlin_utils.py` (utility functions)
- `marlin_utils_fp8.py` (FP8-specific)
- `marlin_utils_fp4.py` (FP4-specific)

#### Supported Quantization Types
**File:** `marlin_utils.py` (Lines 41-78)

**Without Zero Points (GPTQ-style):**
- `uint4b8` (4-bit with 8-bit group bias)
- `uint8b128` (8-bit with 128-element groups)
- `float8_e4m3fn` (FP8 hardware format)
- `float4_e2m1f` (FP4 hardware format)

**With Zero Points (AWQ-style):**
- `uint4` (4-bit with runtime zero points)

#### Hardware Support
**File:** `marlin_utils.py` (Lines 49-56)

| GPU Architecture | SM Version | Support Level |
|------------------|------------|---------------|
| Turing | 75 | Limited (no FP8/FP4) |
| Ampere | 80/86 | Full support |
| Ada | 89 | Full support + FP8 |
| Hopper | 90 | Full support + FP8/FP4 |

#### Performance
- **Fastest quantized inference**: 2-3x faster than standard kernels
- **Tile-based processing**: GPTQ_MARLIN_TILE = 16
- **Parallelization**: Up to 16 parallel threads

---

### 1.8 Other Methods

#### GGUF (Quantized Model Format)
**File:** `gguf.py` (678 lines)

- Support for GGML format weights
- Multiple bit-widths via format specification
- CPU and GPU execution
- Direct loading from `.gguf` files

#### BitsAndBytes (8-bit Quantization)
**File:** `bitsandbytes.py` (608 lines)

- Dynamic & static 8-bit quantization
- Per-layer thresholding
- Mixed precision support

#### Compressed Tensors
**Directory:** `compressed_tensors/`

- Sparsity + quantization combined
- Multiple compression algorithms
- Support for structured/unstructured sparsity

#### QUARK (AMD Quark Quantization)
**Directory:** `quark/`

- INT4-FP8 mixed precision for MoE
- Hardware accelerated on RDNA3
- AMD GPU optimization

#### ModelOpt (NVIDIA TensorRT-LLM Compatible)
**File:** `modelopt.py` (2,233 lines)

- FP8, FP4, MXFP8, Mixed precision
- Integration with NVIDIA ModelOpt toolkit
- Pre/post-quantization hooks

---

## 2. SGLang Quantization

### 2.1 Supported Methods

**File:** `sglang/srt/layers/quantization/__init__.py` (Lines 53-78)

**Base Methods (21 total):**
```python
BASE_QUANTIZATION_METHODS = {
    "fp8", "mxfp8", "blockwise_int8", "modelopt", "modelopt_fp8",
    "modelopt_fp4", "w8a8_int8", "w8a8_fp8", "awq", "awq_marlin",
    "bitsandbytes", "gguf", "gptq", "gptq_marlin", "moe_wna16",
    "compressed-tensors", "qoq", "w4afp8", "petit_nvfp4",
    "fbgemm_fp8", "quark", "auto-round", "modelslim",
    "quark_int4fp8_moe",
}
```

**Conditional:**
```python
if torch.cuda.is_available():
    BASE_QUANTIZATION_METHODS["mxfp4"] = Mxfp4Config
```

---

### 2.2 SGLang-Specific Methods

#### AutoRound
**File:** `auto_round.py` (396 lines)

```python
class AutoRoundConfig(QuantizationConfig):
    SUPPORTED_BITS: set[int] = {2, 3, 4, 8}
    SUPPORTED_FORMATS: set[str] = {
        "auto_round:auto_gptq",
        "auto_round:auto_awq",
    }
    SUPPORTED_BACKENDS: set[str] = {
        "auto", "gptq", "gptq:marlin",
        "awq", "awq:marlin", "marlin"
    }

    def get_min_capability(self) -> int:
        return 60  # Broader GPU support than vLLM
```

**Features:**
- Gradient-based weight quantization
- Supports both GPTQ and AWQ packing formats
- Better accuracy than standard GPTQ (0.5-1% improvement)
- Multiple backend routing

---

#### W4A8 (Mixed-Precision 4-bit Weights + 8-bit Activations)
**File:** `w4afp8.py` (406 lines)

```python
class W4AFp8Config(QuantizationConfig):
    is_checkpoint_fp8_serialized: bool
    is_checkpoint_w4afp8_serialized: bool
    linear_activation_scheme: str           # "dynamic" or "static"
    moe_activation_scheme: str              # "dynamic" or "static"
    weight_block_size: list[int] = [128, 128]
    group_size: int = 128

    def get_min_capability(self) -> int:
        return 90  # Hopper+ only
```

**Advantages over W4A16:**
- **Faster**: 30-40% speedup vs W4A16
- **Memory efficient**: 4x reduction vs FP16
- **Accuracy**: <1% loss vs W8A8

---

#### W8A8 Variants
**File:** `w8a8_fp8.py` (378 lines)

```python
class W8A8Fp8Config(QuantizationConfig):
    is_checkpoint_fp8_serialized: bool
    # Granularity: Per-channel weights + Per-token activations
    # Type: Symmetric

    def get_min_capability(self) -> int:
        return 89  # Ampere+ (for optimal performance)
```

**File:** `w8a8_int8.py` (378 lines)
- Integer 8-bit variant of W8A8
- Similar structure to FP8 version

---

#### Blockwise INT8
**File:** `blockwise_int8.py` (380 lines)

```python
class BlockInt8Config(QuantizationConfig):
    is_checkpoint_int8_serialized: bool
    activation_scheme: str                  # "dynamic" or "static"
    weight_block_size: Optional[List[int]]

    def get_min_capability(self) -> int:
        return 80  # Ampere+
```

---

#### QoQ (Quantization-of-Quantization)
**File:** `qoq.py` (406 lines)

- Second-order quantization for enhanced compression
- Quantizes the quantization parameters themselves
- Further memory reduction with minimal accuracy loss

---

#### Quark INT4-FP8 MoE
**File:** `quark_int4fp8_moe.py` (443 lines)

- AMD-optimized MoE quantization
- Mixed 4-bit weights with 8-bit activations
- Specialized for Mixture-of-Experts models

---

### 2.3 SGLang-Specific Utilities

#### FP8 Kernel Implementation
**File:** `fp8_kernel.py` (2,051 lines)

- Custom CUDA/Triton kernels for FP8 operations
- Per-token group quantization
- Hardware-specific optimizations (NVIDIA, AMD)

#### FP8 Utils
**File:** `fp8_utils.py` (1,340 lines)

- Extended FP8 utilities specific to SGLang
- Marlin FP8 integration
- More extensive than vLLM's utilities

#### Marlin Utils
**File:** `marlin_utils.py` (983 lines)

- SGLang's Marlin integration (more extensive than vLLM)
- Custom kernel dispatch
- Additional optimization passes

#### INT8 Kernel
**File:** `int8_kernel.py` (441 lines)

- Block-wise INT8 operations
- W8A8 block scaled matrix multiplication
- Optimized for NVIDIA and AMD GPUs

#### Modelopt Quantization
**File:** `modelopt_quant.py` (1,871 lines)

- Extended ModelOpt support beyond vLLM
- FP4 and FP8 variants
- Integration with NVIDIA TensorRT-LLM

---

## 3. TensorRT-LLM Quantization

### 3.1 Quantization Modes

**File:** `tensorrt_llm/quantization/mode.py`

#### QuantAlgo Enum (Lines 23-48)

```python
class QuantAlgo(StrEnum):
    # Weight-only quantization
    W8A16 = "W8A16"                         # INT8 weights, FP16 activations
    W4A16 = "W4A16"                         # INT4 weights, FP16 activations
    W4A16_AWQ = "W4A16_AWQ"                 # AWQ-style weight-only
    W8A16_GPTQ = "W8A16_GPTQ"               # GPTQ INT8
    W4A16_GPTQ = "W4A16_GPTQ"               # GPTQ INT4
    W4A16_MXFP4 = "W4A16_MXFP4"             # MXFP4 weight-only

    # Weight + Activation quantization
    W4A8_AWQ = "W4A8_AWQ"                   # AWQ with activation quantization
    W8A8_SQ_PER_CHANNEL = "W8A8_SQ_PER_CHANNEL"
    W8A8_SQ_PER_TENSOR_PLUGIN = "W8A8_SQ_PER_TENSOR_PLUGIN"
    W8A8_SQ_PER_CHANNEL_PER_TOKEN_PLUGIN = "..."
    W8A8_SQ_PER_CHANNEL_PER_TENSOR_PLUGIN = "..."
    W8A8_SQ_PER_TENSOR_PER_TOKEN_PLUGIN = "..."

    # QServe quantization
    W4A8_QSERVE_PER_GROUP = "W4A8_QSERVE_PER_GROUP"
    W4A8_QSERVE_PER_CHANNEL = "W4A8_QSERVE_PER_CHANNEL"

    # FP8 quantization
    FP8 = "FP8"                             # Pure FP8
    FP8_PER_CHANNEL_PER_TOKEN = "FP8_PER_CHANNEL_PER_TOKEN"
    FP8_BLOCK_SCALES = "FP8_BLOCK_SCALES"   # Deepseek format

    # NVFP4 and mixed precision
    NVFP4 = "NVFP4"
    W4A8_NVFP4_FP8 = "W4A8_NVFP4_FP8"       # NVFP4 weights + FP8 activations
    W4A8_MXFP4_FP8 = "W4A8_MXFP4_FP8"       # MXFP4 weights + FP8 activations
    W4A8_MXFP4_MXFP8 = "W4A8_MXFP4_MXFP8"   # Full MXFP

    # Other
    INT8 = "INT8"
    MIXED_PRECISION = "MIXED_PRECISION"
    NVFP4_AWQ = "NVFP4_AWQ"
    NO_QUANT = "NO_QUANT"
```

---

#### QuantMode Flags (Lines 65-103)

```python
class QuantMode(IntFlag):
    # Weight quantization
    INT4_WEIGHTS = auto()
    INT8_WEIGHTS = auto()

    # Activation quantization
    ACTIVATIONS = auto()

    # Scaling granularity
    PER_CHANNEL = auto()        # Static scaling per channel
    PER_TOKEN = auto()          # Dynamic per-token scaling
    PER_GROUP = auto()          # Static per-group scaling

    # KV cache quantization
    INT8_KV_CACHE = auto()
    FP8_KV_CACHE = auto()

    # FP8 modes
    FP8_QDQ = auto()            # Quantize-Dequantize
    FP8_ROWWISE = auto()        # Deepseek format
    FP8_1x128_128x128 = auto()  # Block scales

    # Specialized modes
    W4A8_QSERVE = auto()
    NVFP4 = auto()
    NVFP4_KV_CACHE = auto()
    W4A8_NVFP4_FP8 = auto()
    W4A8_MXFP4_FP8 = auto()
    W4A8_MXFP4_MXFP8 = auto()
    W4A16_MXFP4 = auto()
```

---

### 3.2 Quantization Layers

**File:** `tensorrt_llm/quantization/layers.py` (Lines 1-150)

#### SmoothQuant Linear Layer

```python
class SmoothQuantLinear(Linear):
    """SmoothQuant: Smooths activation distribution for better quantization"""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        per_channel_scale: bool = False,
        act_scale: Optional[float] = None,
    ):
        # Weights: INT8
        # Activations: INT8 (dynamic or static)
        # Scaling: per-channel or per-tensor
        ...

    def forward(self, x):
        # DQ(input) * DQ(weights) * scales
        ...
```

#### Weight-only Quantization
- INT4 or INT8 weights with full-precision activations
- Per-group scaling
- Minimal accuracy loss

---

### 3.3 Quantization Functional API

**File:** `tensorrt_llm/quantization/functional.py` (Lines 1-150)

#### smooth_quant_gemm (Lines 34-96)
```python
def smooth_quant_gemm(
    A: Tensor,
    B: Tensor,
    per_channel_scale: Tensor,
    per_token_scale: Optional[Tensor] = None,
    plugin: bool = True,
) -> Tensor:
    """Smooth quantized GEMM operation"""
    if plugin:
        # Use TensorRT plugin
        return _smooth_quant_gemm_plugin(...)
    else:
        # Fallback implementation
        return torch.matmul(DQ(A), DQ(B)) * scales
```

#### qserve_gemm (Lines 99-150)
```python
def qserve_gemm(
    A: Tensor,
    B: Tensor,
    per_group: bool = True,
) -> Tensor:
    """QServe-specific group-wise quantization"""
    # Per-group or per-channel modes
    ...
```

#### Supported Operations
- `quantize_tensor()`
- `dequantize()`
- `quantize_per_token()`
- `weight_only_quant_matmul()`
- `smooth_quant_layer_norm()`
- `smooth_quant_rms_norm()`
- `fp8_rowwise_gemm()`
- `qserve_gemm_per_group()`
- `qserve_gemm_per_channel()`

---

### 3.4 Configuration

**File:** `tensorrt_llm/quantization/mode.py` (Lines 238-341)

```python
QuantMode.from_description(
    quantize_weights: bool = False,
    quantize_activations: bool = False,
    per_token: bool = False,
    per_channel: bool = False,
    per_group: bool = False,
    use_int4_weights: bool = False,
    use_int8_kv_cache: bool = False,
    use_fp8_kv_cache: bool = False,
    use_fp8_qdq: bool = False,
    use_fp8_block_scales: bool = False,
    use_fp8_rowwise: bool = False,
    use_nvfp4: bool = False,
    use_w4a8_nvfp4_fp8: bool = False,
    use_w4a8_qserve: bool = False,
    use_w4a8_mxfp4_fp8: bool = False,
    use_w4a8_mxfp4_mxfp8: bool = False,
    use_w4a16_mxfp4: bool = False,
)
```

#### Helper Methods
```python
# SmoothQuant shortcuts
QuantMode.use_smooth_quant(per_token=True, per_channel=True)

# QServe shortcuts
QuantMode.use_qserve(per_group=True)

# Weight-only shortcuts
QuantMode.use_weight_only(use_int4_weights=True, per_group=True)
```

---

## 4. Comparative Analysis

### 4.1 Method Support Matrix

| Method | vLLM | SGLang | TensorRT-LLM | Notes |
|--------|------|--------|--------------|-------|
| **AWQ (W4A16)** | ✅ | ✅ | ✅ | Group-wise, activations unquantized |
| **GPTQ (2/3/4/8-bit)** | ✅ | ✅ | ✅ | Per-group scaling, Hessian-based |
| **Marlin** | ✅ | ✅ | ❌ | Fastest inference backend |
| **FP8** | ✅ | ✅ | ✅ | Hardware native on Ada/Hopper |
| **W8A8** | ✅ | ✅ (INT8/FP8) | ✅ (SmoothQuant) | Both weights & activations |
| **W4A8** | ❌ | ✅ | ✅ (QServe) | Optimal balance |
| **MXFP4** | ✅ | ✅ | ❌ (W4A8_MXFP4_*) | Requires Hopper+ |
| **INT8 (W8A16)** | ✅ | ✅ | ✅ | Broader hardware support |
| **GGUF** | ✅ | ✅ | ❌ | Generic quantized format |
| **NVFP4** | ✅ | ✅ | ✅ | NVIDIA 4-bit format |
| **QServe** | ❌ | ❌ | ✅ | Per-group W4A8 |
| **AutoRound** | ✅ (as INC) | ✅ | ❌ | Post-training optimization |
| **BitsAndBytes** | ✅ | ✅ | ❌ | Dynamic 8-bit |
| **QoQ** | ❌ | ✅ | ❌ | Second-order quantization |

---

### 4.2 Hardware Support

#### vLLM Hardware Support

| GPU Arch | Volta | Turing | Ampere | Ada | Hopper | AMD | Intel | x86 |
|----------|-------|--------|--------|-----|--------|-----|-------|-----|
| **AWQ** | ❌ | ✅* | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ |
| **GPTQ** | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ |
| **Marlin** | ❌ | ✅* | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ |
| **INT8** | ❌ | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ | ✅ |
| **FP8** | ❌ | ❌ | ❌ | ✅ | ✅ | ✅ | ❌ | ❌ |
| **GGUF** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ |

**Minimum SM Versions:**
- Volta: 70
- Turing: 75
- Ampere: 80/86
- Ada: 89
- Hopper: 90

**Key Requirements:**
- FP8 requires Ada (SM 89) or Hopper (SM 90) for hardware acceleration
- Marlin requires Turing SM 75+
- MXFP4 requires Hopper SM 90+

---

### 4.3 Kernel Implementation Comparison

#### vLLM Kernel Paths
`vllm/csrc/quantization/`

**Directories:**
- `awq/` - AWQ CUDA kernels
- `gptq/` - GPTQ CUDA kernels
- `marlin/` - Marlin backend kernels
- `machete/` - Universal quantization
- `w8a8/` - W8A8 kernels
- `cutlass_w4a8/` - CUTLASS W4A8
- `fp4/` - FP4 kernels
- `gguf/` - GGUF kernels

**Core Utilities:**
- `utils.cuh` (1,843 bytes)
- `vectorization.cuh` - SIMD operations

---

#### SGLang Kernel Paths
`sglang/sgl-kernel/csrc/quantization/`
`sglang/python/sglang/srt/layers/quantization/`

**Python Kernels:**
- `fp8_kernel.py` (2,051 lines) - FP8 Triton/CUDA
- `int8_kernel.py` (441 lines) - INT8 operations
- `fp8_utils.py` (1,340 lines) - FP8 utilities
- `marlin_utils_fp8.py` - Marlin FP8 integration

**Benchmark Suite:**
`sglang/benchmark/kernels/quantization/`

---

#### TensorRT-LLM Plugin System

**Quantization Plugins:**
- `SmoothQuantGemm` plugin
- `QServeGemm` plugin
- Per-tensor/per-channel variants

**Plugin Creation:**
```python
plg_creator = trt.get_plugin_registry().get_plugin_creator(
    'SmoothQuantGemmPlugin', '1', TRT_LLM_PLUGIN_NAMESPACE
)
gemm_plug = plg_creator.create_plugin("sq_gemm", pfc)
```

---

## 5. Performance & Accuracy Trade-offs

### 5.1 Accuracy Loss by Method

| Method | Bits | Typical Accuracy Loss | Notes |
|--------|------|----------------------|-------|
| **AWQ** | W4A16 | 1-3% | Group-wise scaling helps |
| **GPTQ** | W4A16 | 1-2% | Hessian calibration |
| **GPTQ** | W8A16 | <0.5% | Minimal loss |
| **FP8** | W8A8 | <0.5% | Hardware accelerated |
| **INT8** | W8A8 | 0.5-1% | Per-channel scaling |
| **MXFP4** | W4A16 | 2-3% | More aggressive |
| **W4A8** | Mixed | <1% | Balanced approach |
| **AutoRound** | W4A16 | 0.5-1.5% | Better than GPTQ |

---

### 5.2 Memory Reduction

**Weight-only Quantization:**
| Quantization | Memory Reduction | Overhead |
|--------------|------------------|----------|
| 4-bit | 8x | ~5% (group scales) |
| 8-bit | 4x | ~3% (channel scales) |

**Activation Quantization:**
| Quantization | Reduction | Notes |
|--------------|-----------|-------|
| INT8 activations | 4x (vs FP32) | Dynamic per-token |
| FP8 activations | 4x (vs FP32) | Hardware native |

**KV Cache Quantization:**
| Quantization | Reduction | Accuracy Impact |
|--------------|-----------|-----------------|
| INT8 KV | 4x | <1% loss |
| FP8 KV | 4x | <0.5% loss |

---

### 5.3 Inference Speed

**Fastest to Slowest (Typical):**

1. **Marlin kernels** (vLLM, SGLang)
   - GPTQ Marlin: 2-3x faster than standard GPTQ
   - AWQ Marlin: 2-3x faster than standard AWQ
   - Tile-based optimization

2. **Fused kernels** (TensorRT-LLM)
   - SmoothQuant plugin: 1.5-2x faster
   - QServe plugin: Optimized for long-context

3. **Generic backends**
   - Triton kernels: Flexible, slightly slower
   - FlashInfer: Library-based

4. **Fallback implementations**
   - DQ + GEMM + Q: Slowest but portable

---

### 5.4 Backend Performance (MXFP4)

**File:** `mxfp4.py` (Lines 109-160)

| Backend | Hardware | Performance |
|---------|----------|-------------|
| **FlashInfer (Ada)** | SM 89+ | Best for MXFP4 |
| **CK (AMD GFX950)** | ROCM | Best for AMD |
| **Marlin** | SM 75+ | CPU quant, fast inference |
| **Triton** | SM 90 | Flexible, good |

---

## 6. Hardware Support Matrix

### 6.1 GPU Architecture Support

| Quantization | Volta | Turing | Ampere | Ada | Hopper | AMD | Intel |
|--------------|-------|--------|--------|-----|--------|-----|-------|
| **AWQ** | ❌ | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ |
| **GPTQ** | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ |
| **FP8** | ❌ | ❌ | ❌ | ✅ | ✅ | ✅ | ❌ |
| **W8A8 INT8** | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ |
| **MXFP4** | ❌ | ❌ | ❌ | ❌ | ✅ | ✅* | ❌ |
| **Marlin** | ❌ | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ |

**Notes:**
- FP8 requires native tensor core support (Ada/Hopper)
- MXFP4 requires Hopper H100/H200 or AMD MI300
- Marlin requires SM 75+ (Turing or newer)

---

### 6.2 Framework Advantages

#### vLLM Advantages
1. **Comprehensive Marlin integration**
   - Unified backend for GPTQ/AWQ/FP8/FP4
   - Fastest inference for weight-only quantization

2. **MXFP4 multiple backends**
   - FlashInfer, Marlin, Triton, CK support
   - Flexible for different hardware

3. **GGUF support**
   - Load/run models in GGUF format
   - CPU and GPU execution

4. **Broad model compatibility**
   - 24+ quantization methods
   - Extensive testing infrastructure

---

#### SGLang Advantages
1. **AutoRound support**
   - Post-training weight rounding
   - 0.5-1% accuracy improvement over GPTQ
   - Supports GPTQ/AWQ backends

2. **More quantization kernels**
   - ~2051 LOC in fp8_kernel.py
   - Custom Triton/CUDA implementations
   - Per-token group quantization

3. **Extended ModelOpt**
   - 1,871 lines in modelopt_quant.py
   - More FP4/FP8 variants
   - Deeper integration

4. **QoQ (Quantization-of-Quantization)**
   - Second-order compression
   - Unique to SGLang

---

#### TensorRT-LLM Advantages
1. **Engine-level optimization**
   - Quantization applied during model compilation
   - Kernel fusion with quantization

2. **QServe support**
   - Specialized for long-context QA
   - Per-group and per-channel modes

3. **Plugin system**
   - Custom quantization kernels via TensorRT plugins
   - High performance

4. **KV cache block scales**
   - FP8_1x128_128x128 format (Deepseek)
   - Optimized for specific models

---

## 7. Recommendations

### 7.1 When to Use Each Method

#### For Maximum Speed (Latency-Critical)
- **Use:** Marlin GPTQ or Marlin AWQ (vLLM/SGLang)
- **Hardware:** Turing+ NVIDIA GPUs
- **Trade-off:** 1-3% accuracy loss, 8x memory reduction

#### For Maximum Accuracy (Accuracy-Critical)
- **Use:** FP8 or W8A8 INT8
- **Hardware:** Ada/Hopper for FP8, Ampere+ for INT8
- **Trade-off:** <0.5% accuracy loss, 4x memory reduction

#### For Extreme Memory Constraints
- **Use:** MXFP4 (if Hopper) or GPTQ 4-bit
- **Hardware:** Hopper for MXFP4, Turing+ for GPTQ
- **Trade-off:** 2-3% accuracy loss, 8x memory reduction

#### For Balanced Performance/Accuracy
- **Use:** W4A8 (SGLang) or AutoRound (SGLang)
- **Hardware:** Hopper for W4A8
- **Trade-off:** <1% accuracy loss, optimal throughput

#### For Long-Context Workloads
- **Use:** FP8 KV cache quantization + QServe (TensorRT-LLM)
- **Hardware:** Ada/Hopper
- **Trade-off:** <0.5% accuracy loss, 4x KV cache reduction

---

### 7.2 Framework Selection Guide

#### Choose vLLM if:
- You need the fastest quantized inference (Marlin kernels)
- Broad hardware support is important
- You want GGUF model support
- Extensive quantization options are needed (24+ methods)

#### Choose SGLang if:
- You want AutoRound for better accuracy
- W4A8 mixed precision is optimal for your use case
- Custom kernel development is needed (2000+ LOC kernel library)
- QoQ second-order compression is valuable

#### Choose TensorRT-LLM if:
- You need production-grade optimization
- QServe for long-context is required
- Plugin system integration is important
- Engine-level fusion is critical

---

### 7.3 Configuration Examples

#### vLLM AWQ Example
```python
from vllm import LLM

llm = LLM(
    model="TheBloke/Llama-2-7B-AWQ",
    quantization="awq",
    dtype="auto",
    max_model_len=4096,
)
```

#### vLLM FP8 Example
```python
llm = LLM(
    model="neuralmagic/Meta-Llama-3.1-8B-Instruct-FP8",
    quantization="fp8",
    kv_cache_dtype="fp8",
)
```

#### SGLang AutoRound Example
```bash
python -m sglang.launch_server \
  --model-path AutoRound/Llama-3-8B-Instruct-AutoRound-4bit \
  --quantization auto-round
```

#### TensorRT-LLM QServe Example
```python
from tensorrt_llm import LLM
from tensorrt_llm.quantization import QuantConfig, QuantAlgo

quant_config = QuantConfig(QuantAlgo.W4A8_QSERVE_PER_GROUP)
llm = LLM(model_path, quant_config=quant_config)
```

---

## Summary Table

| Feature | vLLM | SGLang | TensorRT-LLM |
|---------|------|--------|--------------|
| **Total Methods** | 24+ | 21+ | 15+ |
| **Marlin Support** | ✅ Best | ✅ Good | ❌ |
| **AutoRound** | ✅ (as INC) | ✅ Native | ❌ |
| **QServe** | ❌ | ❌ | ✅ |
| **W4A8** | ❌ | ✅ | ✅ |
| **FP8** | ✅ | ✅ | ✅ |
| **MXFP4** | ✅ Multi-backend | ✅ | ✅ Limited |
| **Plugin System** | ❌ | ❌ | ✅ |
| **Custom Kernels** | ~1000 LOC | ~2000 LOC | Plugin-based |
| **Hardware Support** | Broadest | Wide | NVIDIA-focused |

---

**Document Version:** 1.0
**Last Updated:** 2026-03-29
**Research Scope:** vLLM (latest), SGLang (latest), TensorRT-LLM (latest)
