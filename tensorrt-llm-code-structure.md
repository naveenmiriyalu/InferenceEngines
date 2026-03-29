# TensorRT-LLM Code Structure Overview

## Quick Facts
- **Architecture**: Hybrid Python + C++ system
- **Core Technology**: NVIDIA TensorRT for inference optimization
- **Python Files**: ~173 core files
- **C++ Files**: ~355 source + header files
- **Supported Models**: 40+ architectures
- **Location**: `/home/nmiriyal/Documents/UnderstandingVLLM/LatestVLLM/TensorRT-LLM/`

---

## 1. Directory Structure

```
TensorRT-LLM/
├── tensorrt_llm/           # Main Python package
│   ├── llmapi/             # High-level user API
│   ├── models/             # Model definitions (40+ architectures)
│   ├── layers/             # Reusable layer components
│   ├── runtime/            # Inference runtime
│   ├── executor/           # Request execution framework
│   ├── quantization/       # Quantization support
│   ├── _torch/             # PyTorch backend
│   └── inputs/             # Input preprocessing
├── cpp/                    # C++ implementation
│   ├── tensorrt_llm/
│   │   ├── executor/       # C++ executor engine
│   │   ├── batch_manager/  # Batching & scheduling
│   │   ├── kernels/        # CUDA kernels (50+)
│   │   ├── plugins/        # TensorRT plugins
│   │   └── nanobind/       # Python-C++ bindings
├── triton_kernels/         # Vendored Triton kernels
├── examples/               # Usage examples
├── tests/                  # Test suite
├── benchmarks/             # Performance benchmarks
└── docs/                   # Documentation
```

---

## 2. Python Package (`tensorrt_llm/`)

### **High-Level API** (`llmapi/`)

User-facing interface for simple deployment.

- **`llm.py`** (58KB) - **Main entry point**
  - `LLM` class: High-level API for inference
  - Automatic model building
  - Tokenization and output processing
  - Distributed session handling

- **`llm_args.py`** (137KB) - Comprehensive configuration
  - Model parameters
  - Quantization settings
  - Parallel execution options
  - Build and runtime settings

- **`llm_utils.py`** (38KB) - Utilities
  - Build cache management
  - Model loading helpers
  - Configuration validation

- **`mpi_session.py`** (21KB) - Distributed sessions
  - MPI-based multi-GPU coordination
  - Session lifecycle management

### **Build & Compilation Layer**

Core of TensorRT engine creation.

- **`builder.py`** (59KB) - **TensorRT engine builder**
  - `Builder` class: Compiles models to TensorRT engines
  - Optimization configuration
  - Engine serialization
  - Plugin registration

- **`network.py`** (37KB) - Network definition
  - Network graph management
  - Layer connections
  - Input/output specification

- **`functional.py`** (292KB) - **Largest file**
  - Functional API for tensor operations
  - Layer construction primitives
  - Operations: attention, linear, normalization, activation
  - Quantization operations

- **`module.py`** (10KB) - PyTorch-like Module base
  - Base class for model components
  - Parameter management
  - Forward method interface

- **`graph_rewriting.py`** (24KB) - Graph optimization
  - Pattern matching and rewriting
  - Fusion opportunities
  - Performance optimizations

### **Runtime & Execution**

#### **Runtime** (`runtime/`)
- **`generation.py`** (228KB) - **Core generation logic**
  - Text generation and decoding
  - Sampling strategies
  - Beam search implementation
  - Streaming support

- **`model_runner.py`** (44KB) - TensorRT model execution wrapper
- **`model_runner_cpp.py`** (56KB) - C++ binding wrapper
- **`multimodal_model_runner.py`** (132KB) - Multimodal handling
- **`kv_cache_manager.py`** (18KB) - KV cache management

#### **Executor** (`executor/`)
Request execution framework for production serving.

- **`executor.py`** (24KB) - Base executor interface
  - Request submission and tracking
  - Result retrieval
  - Abstract interface for different execution modes

- **`base_worker.py`** (45KB) - Worker implementation
  - Model loading and initialization
  - Forward pass execution
  - Batch processing

- **`ray_executor.py`** (18KB) - Ray-based distributed execution
- **`rpc_proxy.py`**, **`rpc_worker.py`** - RPC-based execution
- **`result.py`** (43KB) - Result and output definitions

### **Model Definitions** (`models/`)

40+ model architectures:

- **Standard Models**: llama, gpt, qwen, deepseek_v1/v2, phi, gemma, mistral, falcon, baichuan
- **Special Architectures**: mamba (SSMs), eagle, medusa (speculative decoding)
- **Multimodal**: clip, cogvlm, mllama, fuyu
- **Each model file implements**:
  - `prepare_inputs()` - Input preparation
  - Attention backend selection
  - Custom layer implementations

### **Layers** (`layers/`)

Reusable components for model construction:

- **`attention.py`** - Multi-head, MQA, GQA attention
- **`linear.py`** - Linear layers (with quantization support)
- **`mlp.py`** - MLP blocks
- **`moe.py`** - Mixture of Experts
- **`normalization.py`** - LayerNorm, RMSNorm
- **`embedding.py`** - Token and position embeddings
- **`activation.py`** - GELU, SiLU, Swish, etc.
- **`ssm.py`**, **`recurrent.py`** - State space models
- **`lora.py`** - LoRA adapters

### **Quantization** (`quantization/`)

Comprehensive quantization support:

- **`functional.py`** (63KB) - Quantization operations
- **`layers.py`** (140KB) - Quantized layer implementations
- **`quantize_by_modelopt.py`** - NVIDIA ModelOpt integration
- **Supported formats**:
  - INT4, INT8 (weights)
  - FP8 (weights and activations)
  - W8A8 (8-bit weights, 8-bit activations)
  - Per-layer and per-group quantization

### **PyTorch Backend** (`_torch/`)

Alternative PyTorch-based execution path:

- **`models/`** - PyTorch model implementations
- **`auto_deploy/`** - Automatic JIT deployment
- **`custom_ops/`** - Custom CUDA kernels for PyTorch
- **`attention_backend/`** - Alternative attention implementations
- **`distributed/`**, **`peft/`**, **`speculative/`** - Specialized features

### **Input Processing** (`inputs/`)

Multimodal input handling:

- **`multimodal.py`** (30KB) - Image/video embedding
- **`registry.py`** (30KB) - Plugin registry for custom processors

---

## 3. C++ Source Code (`cpp/tensorrt_llm/`)

### **Executor & Runtime**

#### **Executor** (`executor/`)
Main C++ execution engine:

- **`executor.cpp/h`** - C++ executor implementation
- **`executorImpl.cpp/h`** - Implementation details
- **`dynamicBatchTuner.cpp/h`** - Auto-tuning for batch sizes
- 60+ configuration and utility files

#### **Batch Manager** (`batch_manager/`)
Request batching and scheduling:

- **40+ header files** for:
  - KV cache management
  - Scheduling policies
  - Buffer management
  - Sequence tracking
- **`llmRequest.h`** - Request representation
- **`kvCacheManager.h`** - Cache allocation and tracking
- **`sequenceSlotManager.h`** - Sequence management

### **CUDA Kernels** (`kernels/`)

50+ custom CUDA kernels for performance-critical operations:

#### **Attention**
- **`attentionMask.cu`** - Attention mask computation
- **`fmha_v2/`** - Flash Attention v2 implementation
- Multi-head, grouped-query attention variants

#### **Decoding**
- **`decodingKernels.cu`** - Token decoding
- **`beamSearchKernels.cu`** - Beam search
- **`samplingKernels.cu`** - Top-k, top-p, temperature

#### **Quantization**
- INT8, FP8, W8A8 specific kernels
- Dequantization during inference
- Quantized GEMM operations

#### **Mixture-of-Experts**
- **`customMoeRoutingKernels.cu`** - Expert routing
- **`moePerm.cu`** - Token permutation
- Load balancing kernels

#### **Communication**
- **`customAllReduceKernels.cu`** - Custom collective operations
- **`allgatherKernels.cu`** - All-gather for TP

#### **Utilities**
- **`banBadWords.cu`** - Bad word filtering
- **`cumsumLastDim.cu`** - Cumulative sum
- **`gatherTreeKernel.cu`** - Beam search tree gathering
- **`loraKernels.cu`** - LoRA adapter application

### **Python Bindings**

#### **Nanobind** (`nanobind/`) - **Modern, primary**
- **`bindings.cpp`** - Main module entry point
- **`batch_manager/bindings.cpp`** - Batch manager API
- **`executor/bindings.cpp`** - Executor API
- **`runtime/bindings.cpp`** - Runtime utilities

Creates Python module: `tensorrt_llm.bindings`
- Submodules: `executor`, `batch_manager`, `runtime`, `internal`

#### **PyBind11** (`pybind/`) - Legacy (being phased out)

### **Plugins & Utilities**

- **`plugins/`** - Custom TensorRT plugins
  - Attention plugins
  - Normalization plugins
  - Custom operations not in TensorRT
- **`common/`** - Shared utilities
  - Quantization definitions
  - CUDA utilities
  - Data type conversions
- **`thop/`** - Tensor operations
  - AllReduce
  - Communication primitives

---

## 4. Architecture Patterns

### **Build Pipeline**

```
Python Model Definition (Module-based)
    ↓
builder.py → Create TensorRT Network
    ↓
functional.py → Add Layers and Operations
    ↓
TensorRT Optimization
    ├── Layer fusion
    ├── Kernel selection
    ├── Quantization
    └── Graph optimization
    ↓
Serialized Engine (.plan file)
```

### **Execution Pipeline**

```
Input Text (Python)
    ↓
Tokenization (Python - Transformers)
    ↓
Embeddings (Python forward or C++ engine)
    ↓
TensorRT Engine Execution (C++)
    ├── CUDA Kernels
    ├── Flash Attention
    ├── Quantized GEMM
    └── Custom Operations
    ↓
Logits → Sampling (C++ kernels)
    ↓
Output Tokens → Detokenization (Python)
    ↓
Generated Text
```

### **Distributed Execution**

```
Python LLM API
    ↓
GenerationExecutor (abstract)
    ├── BaseWorker (local execution)
    ├── RayGPUWorker (Ray-based distributed)
    └── RpcWorker (RPC-based distributed)
    ↓
C++ Executor (via nanobind)
    ├── Batch Manager (scheduling)
    ├── KV Cache Manager (memory)
    └── TensorRT Engine Runner
    ↓
Multi-GPU Execution
```

---

## 5. Python-C++ Integration

### **Binding Architecture**

#### **Primary: nanobind**
Modern C++ to Python bindings:
```python
import tensorrt_llm.bindings as tllm_bindings

executor = tllm_bindings.Executor(config)
request_id = executor.enqueue_request(request)
result = executor.await_response(request_id)
```

Compiled module: `tensorrt_llm.bindings`
- `executor` - Request execution
- `batch_manager` - Scheduling and batching
- `runtime` - CUDA operations, memory
- `internal` - Internal utilities

### **Key Interfaces**

1. **Executor Interface** - Request submission and result retrieval
2. **Batch Manager** - Scheduling and cache allocation
3. **Runtime** - CUDA operations, memory management
4. **Model Loading** - Weight loading, config serialization

### **Data Exchange**

- **DLPack** for zero-copy tensor passing (PyTorch ↔ TensorRT)
- **NumPy arrays** for CPU-side data
- **Protocol Buffers** for configuration (optional)

---

## 6. Key Entry Points

### **For Users (Python)**

#### 1. High-Level LLM Class
```python
from tensorrt_llm import LLM

llm = LLM(model="meta-llama/Llama-2-7b-hf")
outputs = llm.generate(["Hello, how are you?"])
print(outputs[0].text)
```

#### 2. AutoModel (Lower-Level)
```python
from tensorrt_llm import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
# Build engine, then use for inference
```

#### 3. Builder (Expert API)
```python
from tensorrt_llm import Builder

builder = Builder()
network = builder.create_network()
# Manually construct network
engine = builder.build_engine(network, config)
```

### **For Developers (C++)**

#### 1. Executor C++ Class
```cpp
#include "tensorrt_llm/executor/executor.h"

Executor executor(config);
auto request_id = executor.enqueueRequest(request);
auto result = executor.awaitResponse(request_id);
```

#### 2. ModelRunner
```cpp
// Engine loading and inference
// Weight loading
// Forward pass execution
```

#### 3. BatchManager
```cpp
// Scheduling and resource allocation
// KV cache management
```

---

## 7. CUDA/C++ Components Role

### **Custom CUDA Kernels Implement**:

1. **Flash Attention (FMHA)** - Optimized attention computation
2. **Quantized Operations** - INT8, FP8, W8A8 inference
3. **MoE Routing** - Expert selection and permutation
4. **Decoding Utilities** - Beam search, sampling
5. **AllReduce** - Custom collective operations for TP
6. **Kernel Fusion** - Linear + activation, attention fusions

### **TensorRT Integration**:

- Automatic graph optimization
- Kernel selection and tuning
- Plugin deployment for custom ops
- Dynamic shape support
- Mixed precision execution

---

## 8. Specialized Subsystems

### **Quantization**
- **Supported formats**: INT4, INT8, FP8, W8A8
- **Per-layer configuration**: Different layers can use different quantization
- **ModelOpt integration**: Automatic quantization calibration
- **Files**: `quantization/`, kernels in `cpp/kernels/`

### **Multi-Modal**
- Image/video encoders
- Vision transformers (CLIP, etc.)
- Unified input preprocessing
- **Files**: `inputs/`, multimodal model implementations

### **Advanced Decoding**
- **Speculative decoding**: With draft models
- **Medusa**: Parallel decoding heads
- **Lookahead decoding**: Future token prediction
- **Beam search**: Exhaustive search
- **Sampling variants**: Top-k, nucleus, temperature
- **Files**: `runtime/generation.py`, decoding kernels

### **Distributed Inference**
- **Tensor Parallelism (TP)**: Split model across GPUs
- **Pipeline Parallelism (PP)**: Split layers across GPUs
- **Multi-node support**: Across multiple machines
- **Disaggregated serving**: Separate prefill and decode
- **Files**: MPI session, distributed communicators

### **Memory Optimization**
- **KV cache strategies**: Retention and reuse
- **Cache sharing**: Between requests
- **GPU/CPU offloading**: For large models
- **Pre-allocated pools**: Reduce allocation overhead
- **Files**: `runtime/kv_cache_manager.py`, batch manager

---

## 9. Important Files by Size and Role

| File | Size | Purpose |
|------|------|---------|
| `functional.py` | 292KB | Tensor ops, layer building blocks |
| `runtime/generation.py` | 228KB | Core generation logic |
| `quantization/layers.py` | 140KB | Quantized layer implementations |
| `llmapi/llm_args.py` | 137KB | Comprehensive configuration |
| `runtime/multimodal_model_runner.py` | 132KB | Multimodal model handling |
| `quantization/functional.py` | 63KB | Quantization operations |
| `builder.py` | 59KB | TensorRT engine compilation |
| `llmapi/llm.py` | 58KB | User-facing API |
| `llmapi/llm_utils.py` | 38KB | Build cache, utilities |
| `network.py` | 37KB | Network graph definition |

---

## 10. Build System

### **Python Build** (`setup.py`, `pyproject.toml`)
- Detects precompiled bindings or builds from source
- CMake-based C++ compilation
- Nanobind binding generation
- Wheel packaging

### **C++ Build** (CMake)
- TensorRT plugin compilation
- CUDA kernel compilation
- Nanobind extension building
- Dependency management (TensorRT, CUDA, cuBLAS, cuDNN)

### **Dependencies**
- **TensorRT**: Core optimization engine (≥8.6)
- **CUDA**: GPU execution (≥11.8)
- **PyTorch**: Python model definitions
- **Transformers**: Model loading
- **MPI**: Multi-GPU communication (optional)

---

## 11. Design Patterns

1. **Layered Architecture**: Python (convenience) → Functional (declarative) → C++ (performance) → CUDA (hardware)
2. **Builder Pattern**: Separate construction (Builder) from representation (Network)
3. **Module System**: PyTorch-like interface for composability
4. **Plugin Architecture**: Extend TensorRT with custom operations
5. **Executor Abstraction**: Multiple execution backends (local, Ray, RPC)
6. **Zero-Copy**: DLPack for efficient tensor transfer
7. **Ahead-of-Time Compilation**: Models compiled to optimized engines
8. **Dynamic Batching**: Efficient batch processing in C++
9. **Graph Optimization**: Automatic fusion and kernel selection
10. **Hardware-Specific Tuning**: CUTLASS templates, Tensor Cores

---

## 12. Key Optimizations

1. **TensorRT Optimizations**: Layer fusion, kernel auto-tuning, graph optimization
2. **Flash Attention**: Memory-efficient attention computation
3. **Quantization**: INT4/INT8/FP8 for reduced memory and faster inference
4. **CUTLASS GEMM**: High-performance matrix multiplication
5. **KV Cache Management**: Efficient memory reuse
6. **CUDA Graphs**: Reduced kernel launch overhead
7. **Tensor Parallelism**: Scale across multiple GPUs
8. **Custom Kernels**: Hand-optimized for specific operations
9. **Plugin System**: Extend TensorRT for LLM-specific ops
10. **Dynamic Shapes**: Handle variable-length sequences efficiently

---

## 13. Workflow Examples

### **Simple Inference**
```python
from tensorrt_llm import LLM

llm = LLM(model="meta-llama/Llama-2-7b-hf")
outputs = llm.generate(["Hello world"])
```

### **Quantized Inference**
```python
llm = LLM(
    model="meta-llama/Llama-2-7b-hf",
    quant_config=QuantConfig(quant_algo="W8A8")
)
```

### **Multi-GPU Inference**
```python
llm = LLM(
    model="meta-llama/Llama-70b-hf",
    tensor_parallel_size=4
)
```

### **Custom Build**
```python
from tensorrt_llm import Builder, AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
builder = Builder()
engine = builder.build(model, config)
```

---

## Summary

TensorRT-LLM is a **production-optimized** LLM inference engine that:

1. **Combines Python and C++**: High-level API (Python) + Performance (C++/CUDA)
2. **Leverages TensorRT**: Automatic optimization, kernel selection, graph fusion
3. **Supports 40+ Models**: Standard and specialized architectures
4. **Extensive Quantization**: INT4, INT8, FP8, W8A8 for efficiency
5. **Distributed Inference**: TP, PP, multi-node support
6. **Advanced Decoding**: Speculative, Medusa, beam search, sampling
7. **Multimodal**: Image/video processing pipelines
8. **Production-Ready**: Robust executor framework, batching, caching

**Architecture Philosophy**:
- **AOT Compilation**: Models compiled to optimized engines (vs JIT)
- **Maximum Performance**: NVIDIA hardware-specific optimizations
- **Flexibility**: Multiple APIs (high-level LLM, mid-level Builder, low-level C++)
- **Enterprise Focus**: Stability, performance, scalability

The codebase achieves peak performance through tight integration with NVIDIA's software stack (TensorRT, CUTLASS, cuBLAS, cuDNN) while maintaining a user-friendly Python interface.
