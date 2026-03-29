# SGLang Code Structure Overview

## Quick Facts
- **Main Focus**: High-performance LLM serving with Domain-Specific Language (DSL)
- **Architecture**: Three-process design (HTTP Server, Scheduler, Detokenizer)
- **Key Innovation**: RadixAttention for prefix caching
- **Location**: `/home/nmiriyal/Documents/UnderstandingVLLM/LatestVLLM/sglang/`

---

## 1. Directory Structure

```
sglang/
├── python/sglang/              # Main Python package
│   ├── lang/                   # SGLang DSL (frontend language)
│   ├── srt/                    # SGLang Runtime (SRT) - serving engine
│   ├── jit_kernel/             # JIT-compiled Triton kernels
│   ├── multimodal_gen/         # Multimodal generation
│   ├── eval/                   # Evaluation frameworks
│   └── cli/                    # Command-line interface
├── sgl-kernel/                 # C++/CUDA AOT kernels
│   └── csrc/                   # C++ source code
├── sgl-model-gateway/          # Model gateway service
├── benchmark/                  # Performance benchmarks
├── test/                       # Integration tests
├── examples/                   # Example usage
├── docs/                       # Documentation
├── docker/                     # Docker configurations
└── scripts/                    # Build scripts
```

---

## 2. Python Package Structure

### **Frontend: SGLang DSL** (`/python/sglang/lang/`)

The Domain-Specific Language for structured LLM programming.

#### Core API (`api.py`)
- `gen()` - Generate text
- `select()` - Choose from options
- `function()` - Define SGLang functions
- `system()` / `user()` / `assistant()` - Chat roles
- `image()` / `video()` - Multimodal inputs
- `gen_int()` / `gen_string()` - Type-constrained generation

#### Interpreter (`interpreter.py`)
- Executes SGLang programs against runtime backends
- Manages program state and control flow
- Example:
```python
@sgl.function
def multi_turn_chat(s, question):
    s += sgl.system("You are a helpful assistant.")
    s += sgl.user(question)
    s += sgl.assistant(sgl.gen("answer", max_tokens=100))
```

#### Backend Implementations (`backend/`)
- `runtime_endpoint.py` - Local SGLang Runtime backend
- `openai.py` - OpenAI API backend
- `anthropic.py` - Anthropic API backend
- `litellm.py` - LiteLLM backend
- `vertexai.py` - Google VertexAI backend
- `base_backend.py` - Base backend interface

#### Other Components
- `ir.py` - Intermediate representation for DSL expressions
- `chat_template.py` - Chat template management
- `choices.py` - Choice sampling methods
- `tracer.py` - Program execution tracing

---

### **Backend: SGLang Runtime (SRT)** (`/python/sglang/srt/`)

High-performance serving engine with three-process architecture.

#### **Entry Points** (`/entrypoints/`)
- **`engine.py`** - Main Python API
  - `Engine` class: Orchestrates entire system
  - Spawns scheduler and detokenizer processes
  - Manages inter-process communication (ZMQ)

- **`http_server.py`** - HTTP API server
  - FastAPI-based OpenAI-compatible API
  - Endpoints: `/v1/completions`, `/v1/chat/completions`, `/v1/embeddings`

- **`grpc_server.py`** - gRPC server
- **`context.py`** - Request context management
- **`EngineBase.py`** - Base engine class

#### **Scheduling & Batch Management** (`/managers/`)
Core of the zero-overhead CPU scheduler:

- **`scheduler.py`** (132KB) - **Central scheduler**
  - Continuous batching
  - Prefill-decode disaggregation
  - Request queueing and batch formation

- **`schedule_batch.py`** - Batch construction and validation
- **`schedule_policy.py`** - Scheduling policies
- **`tokenizer_manager.py`** - Input tokenization
- **`detokenizer_manager.py`** - Output detokenization
- **`cache_controller.py`** - KV-cache management
- **`io_struct.py`** - Request/response data structures
- **`template_manager.py`** - Chat template handling

#### **Model Execution** (`/model_executor/`)
- **`model_runner.py`** (106KB) - Main model forward pass
- **`cuda_graph_runner.py`** - CUDA graph optimization
- **`cpu_graph_runner.py`** - CPU graph runner
- **`piecewise_cuda_graph_runner.py`** - Memory-efficient CUDA graphs
- **`forward_batch_info.py`** - Batch info for forward passes

#### **Model Implementations** (`/models/`)
50+ model architectures:
- **Standard**: Llama, Qwen, DeepSeek (v2, v3, MLA), Gemma, Mistral, GLM, GPT, Baichuan, Falcon
- **MoE Models**: bailing_moe, afmoe, exaone_moe
- **Multimodal**: LLaVA variants, DeepSeek VL, Janus, etc.

#### **Attention Mechanisms** (`/layers/attention/`)
Multiple backend implementations:
- **`flashattention_backend.py`** - FlashAttention v2/v3
- **`flashinfer_backend.py`** - FlashInfer integration
- **`flashinfer_mla_backend.py`** - Multi-Head Latent Attention (MLA)
- **`triton_backend.py`** - Triton kernel-based attention
- **`hybrid_linear_attn_backend.py`** - Linear attention (Mamba-style)
- **`vision.py`** - Vision transformer attention
- Others: TensorRT-LLM MHA/MLA, xpu_backend, torch_native, wave, aiter, nsa

#### **Memory & KV-Cache** (`/mem_cache/`)
RadixAttention and memory management:
- **`radix_cache.py`** - Radix tree-based prefix caching (RadixAttention)
- **`hiradix_cache.py`** - Hierarchical RadixAttention
- **`mamba_radix_cache.py`** - Cache for state-space models
- **`memory_pool.py`** (74KB) - GPU memory pool allocation
- **`allocator.py`** - Memory allocator
- **`hicache_storage.py`** - Hi-Cache storage backend

#### **Model Layers** (`/layers/`)
- **`linear.py`** - Quantized linear layers (AWQ, GPTQ, FP8, etc.)
- **`rotary_embedding.py`** - RoPE (Rotary Position Embeddings)
- **`layernorm.py`** - Layer normalization (RMSNorm, etc.)
- **`sampler.py`** - Token sampling strategies
- **`logits_processor.py`** - Logit processing and constraints
- **`elementwise.py`** - Element-wise operations
- **`communicator.py`** - Distributed communication primitives

#### **Constrained Generation** (`/constrained/`)
Grammar-based structured output:
- **`grammar_manager.py`** - FSM-based grammar constraints
- **`xgrammar_backend.py`** - XGrammar backend
- **`llguidance_backend.py`** - LLGuidance integration
- **`outlines_backend.py`** - Outlines library integration
- **`outlines_jump_forward.py`** - Jump-forward optimization

#### **Distributed Training** (`/distributed/`)
- **`parallel_state.py`** - TP, PP, EP state management
- **`device_communicators/`** - NCCL, Gloo, etc.

#### **Advanced Features**
- **`/lora/`** - LoRA adapter support
- **`/speculative/`** - Speculative decoding
- **`/disaggregation/`** - Prefill-decode disaggregation (PD)
- **`/dllm/`** - Distributed LLM
- **`/elastic_ep/`** - Elastic expert parallelism
- **`/eplb/`** - Expert parallelism load balancer
- **`/compilation/`** - PyTorch torch.compile integration
- **`/function_call/`** - Function calling/tool use

#### **Configuration & Utilities**
- **`server_args.py`** (249KB) - 100+ configuration options
- **`environ.py`** - Environment variable management
- **`utils/`** - Utility functions

---

### **JIT Kernels** (`/python/sglang/jit_kernel/`)

Just-In-Time compiled CUDA/Triton kernels:
- **`flash_attention_v4.py`** - FlashAttention implementation
- **`rope.py`** - RoPE positional encoding
- **`hicache.py`** - Hi-Cache implementation
- **`gptq_marlin.py`** - GPTQ quantization
- **`fused_metadata_copy.py`** - Fused copy operations
- **`norm.py`**, **`pos_enc.py`**, **`timestep_embedding.py`** - Element-wise ops
- Diffusion-related kernels

---

### **Multimodal Generation** (`/python/sglang/multimodal_gen/`)
- **`registry.py`** - Model registry for multimodal models
- **`runtime/`** - Multimodal generation runtime
- **`configs/`** - Model configurations

---

## 3. C++ Source Code (`sgl-kernel/csrc/`)

Ahead-Of-Time compiled CUDA kernels for production.

### **Main Extension**
- **`common_extension.cc`** - Python C++ extension bindings

### **Kernel Categories**

#### **AllReduce** (`allreduce/`)
- Custom all-reduce operations
- MSCCL++ integration
- Distributed training primitives

#### **Attention** (`attention/`)
- `cascade/` - Cascaded attention
- `cutlass_mla/` - CUTLASS-based Multi-Head Latent Attention
- `merge_attn_states/` - Attention state merging

#### **Element-wise** (`elementwise/`)
- Fused activation functions
- RoPE implementation
- Position encoding
- Copy operations

#### **GEMM** (`gemm/`)
- AWQ quantization
- GPTQ quantization
- Marlin kernels
- DeepSeek-specific optimizations

#### **Mixture-of-Experts** (`moe/`)
- Token routing
- Load balancing
- Expert selection

#### **Quantization** (`quantization/`)
- Various quantization operations
- INT4, INT8, FP8 support

#### **CPU** (`cpu/`)
- CPU kernels
- Intel AMX support

---

## 4. Three-Process Architecture

SGLang Runtime uses three separate processes communicating via ZMQ:

```
┌─────────────────────────────────────────────────────┐
│  Process 1: HTTP/gRPC Server + TokenizerManager     │
│  - Handle user requests                             │
│  - Tokenize inputs                                  │
│  - Apply chat templates                             │
│  - Return responses to clients                      │
└────────────────┬────────────────────────────────────┘
                 │ (ZMQ IPC)
                 ↓
┌─────────────────────────────────────────────────────┐
│  Process 2: Scheduler                               │
│  - Receive tokenized requests                       │
│  - Continuous batching                              │
│  - KV-cache management (RadixAttention)            │
│  - Call ModelRunner for forward passes              │
│  - Prefill-decode disaggregation                    │
│  - Send outputs to Detokenizer                      │
└────────────────┬────────────────────────────────────┘
                 │ (ZMQ IPC)
                 ↓
┌─────────────────────────────────────────────────────┐
│  Process 3: DetokenizerManager                      │
│  - Convert token IDs to text                        │
│  - Handle streaming/non-streaming modes             │
│  - Return to HTTP server                            │
└─────────────────────────────────────────────────────┘
```

---

## 5. Request Flow

```
User Request
    ↓
HTTP Server (FastAPI)
    ↓
TokenizerManager → Tokenize input
    ↓ (ZMQ)
Scheduler
    ├── RadixCache lookup (prefix caching)
    ├── Batch formation (continuous batching)
    ├── KV-cache allocation
    └── ModelRunner → Forward pass
        ├── Attention computation
        ├── Sampling
        └── Token generation
    ↓ (ZMQ)
DetokenizerManager → Tokens to text
    ↓
HTTP Response (streaming or complete)
```

---

## 6. Key Architectural Innovations

### **1. RadixAttention**
- **Hierarchical radix tree** for prefix caching
- Automatically shares KV cache for common prefixes
- Significantly reduces memory usage and computation
- Files: `mem_cache/radix_cache.py`, `mem_cache/hiradix_cache.py`

### **2. Continuous Batching**
- Interleave prefill and decode stages
- Dynamic batch size adjustment
- File: `managers/scheduler.py`

### **3. Prefill-Decode Disaggregation (PD)**
- Separate physical streams for prefill vs decode
- Optimizes GPU utilization
- Directory: `srt/disaggregation/`

### **4. Zero-Overhead Scheduling**
- Minimal CPU intervention during inference
- CUDA graphs for reduced overhead
- Files: `model_executor/cuda_graph_runner.py`

### **5. Paged Attention**
- Flexible memory allocation for sequences
- Similar to vLLM's PagedAttention
- Files: Integrated in attention backends

### **6. Speculative Decoding**
- Generate multiple tokens with draft model
- Verify with full model
- Directory: `srt/speculative/`

---

## 7. Key Entry Points

### **User-Facing APIs**

#### 1. SGLang DSL (Python)
```python
import sglang as sgl

@sgl.function
def chat(s, question):
    s += sgl.system("You are helpful.")
    s += sgl.user(question)
    s += sgl.assistant(sgl.gen("answer"))

state = chat.run(question="What is AI?", backend="openai")
print(state["answer"])
```

#### 2. Runtime Engine (Python)
```python
from sglang import Engine

engine = Engine(model="meta-llama/Llama-2-7b-hf")
# Engine spawns scheduler and detokenizer processes
```

#### 3. HTTP Server (CLI)
```bash
sglang launch-server --model meta-llama/Llama-2-7b-hf --port 8000
```

#### 4. OpenAI-Compatible API (HTTP)
```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "meta-llama/Llama-2-7b-hf", "messages": [...]}'
```

### **Core Configuration**
- **File**: `srt/server_args.py` (249KB)
- **100+ options** for:
  - Model loading (format, quantization, dtype)
  - Performance tuning (batch size, prefill/decode settings)
  - Distributed training (TP, PP, EP)
  - Memory management (KV-cache, swap)
  - Hardware selection

---

## 8. Important Files by Size

| File | Size | Purpose |
|------|------|---------|
| `srt/server_args.py` | 249KB | Comprehensive configuration |
| `srt/managers/scheduler.py` | 132KB | Core batch scheduler |
| `srt/model_executor/model_runner.py` | 106KB | Model execution |
| `srt/mem_cache/memory_pool.py` | 74KB | GPU memory management |
| `lang/interpreter.py` | - | DSL interpreter |
| `srt/entrypoints/engine.py` | - | Main engine orchestrator |

---

## 9. Python-to-C++/CUDA Integration

### **Two-Tier Kernel System**

#### **JIT Kernels** (`python/sglang/jit_kernel/`)
- Triton-based kernels
- Compiled at runtime
- Fast iteration for development
- Example: `flash_attention_v4.py`

#### **AOT Kernels** (`sgl-kernel/csrc/`)
- Pre-compiled CUDA kernels
- Production optimization
- Hardware-specific tuning
- Example: `attention/cutlass_mla/`

### **Binding Mechanism**
1. **Direct PyTorch ops**: `torch.ops.sglang_kernels.*`
2. **Python C++ extensions**: Via `common_extension.cc`
3. **Triton JIT**: Dynamic compilation

---

## 10. Advanced Features

### **Multimodal Support**
- Image/video processing pipelines
- Encoder-decoder architectures
- Vision transformer integrations
- Directory: `multimodal_gen/`

### **Quantization Ecosystem**
- AWQ, GPTQ, GPTQ-Marlin, FP8, FP4, INT4
- Per-tensor and per-group quantization
- TorchAO integration

### **Grammar-Constrained Generation**
- Multiple backends: XGrammar, LLGuidance, Outlines
- FSM-based token filtering
- JSON schema constraints
- Directory: `srt/constrained/`

### **Distributed Training**
- Data parallelism
- Tensor parallelism (TP)
- Pipeline parallelism (PP)
- Expert parallelism for MoE
- Expert load balancing

---

## 11. Build System

### **Python Build**
- `pyproject.toml` - Python package configuration
- Key dependencies:
  - PyTorch 2.9.1
  - Transformers 4.57.1
  - FlashInfer 0.6.3
  - XGrammar 0.1.27
  - FastAPI/Uvicorn
  - ZMQ for IPC

### **C++ Build**
- CMake-based build system
- Multi-backend: CUDA, ROCm, Intel MUSA, CPU
- CUTLASS templates

---

## 12. Key Design Patterns

1. **Three-Process Architecture**: Separation of HTTP, scheduling, detokenization
2. **ZMQ IPC**: High-performance inter-process communication
3. **RadixAttention**: Automatic prefix caching via radix trees
4. **Continuous Batching**: Dynamic request batching
5. **DSL Frontend**: User-friendly structured generation
6. **Multiple Backends**: Flexibility in deployment (local, OpenAI, Anthropic)
7. **CUDA Graphs**: Reduced CPU overhead
8. **Modular Attention**: Multiple backend implementations
9. **Grammar Constraints**: FSM-based structured output
10. **JIT + AOT Kernels**: Development speed + production performance

---

## Summary

SGLang is a **dual-purpose** system:

1. **Frontend (DSL)**: User-friendly language for structured LLM programming
   - Intuitive API with `gen()`, `select()`, roles
   - Multi-backend support
   - Type-constrained generation

2. **Backend (SRT)**: High-performance serving engine
   - Three-process architecture
   - RadixAttention for prefix caching
   - Continuous batching
   - 50+ model architectures
   - Advanced features: speculative decoding, grammar constraints, multimodal

**Key Strengths**:
- Efficient prefix caching (RadixAttention)
- Clean separation of concerns (3 processes)
- Fast iteration (JIT kernels) + production performance (AOT kernels)
- Flexible deployment (local runtime or cloud APIs)
- Strong support for constrained generation (JSON, grammar)

The codebase emphasizes both **developer experience** (DSL) and **production performance** (optimized runtime).
