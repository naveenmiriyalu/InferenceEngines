# Learning Plan for TensorRT-LLM Codebase

This guide outlines the Python and C++ skills needed to understand and contribute to TensorRT-LLM effectively.

---

## Part A: Python Skills

### Level 1: Python Fundamentals (Prerequisites)

#### Core Language Features
- [x] **Object-Oriented Programming**
  - Classes, inheritance, abstract base classes
  - Properties, class methods, static methods
  - Example: `tensorrt_llm/module.py` - Module base class

- [x] **Type Annotations**
  - Type hints for functions and variables
  - Generic types, Optional, Union
  - Example: Extensive typing throughout codebase

- [x] **Decorators**
  - Function and class decorators
  - `@property`, `@staticmethod`
  - Example: Model layer decorators

- [x] **Context Managers**
  - Resource management
  - `with` statements
  - Example: Engine lifecycle management

---

### Level 2: Advanced Python

#### PyTorch Fundamentals
- [ ] **Tensor Operations**
  - Creating and manipulating tensors
  - Device placement (`.to(device)`, `.cuda()`)
  - Example: All model definitions in `models/`

- [ ] **nn.Module**
  - Creating custom modules
  - `forward()` method
  - Parameter registration
  - Example: `module.py`, layer implementations in `layers/`

- [ ] **Model Loading**
  - Loading pretrained weights
  - State dict manipulation
  - Example: `AutoModelForCausalLM`

#### Hugging Face Ecosystem
- [ ] **Transformers Library**
  - Loading models with `from_pretrained()`
  - Tokenizers
  - Model configurations
  - Example: Integration in model loading

- [ ] **SafeTensors**
  - Efficient weight loading
  - Security advantages
  - Example: Weight loading utilities

---

### Level 3: TensorRT-LLM Specific Patterns

#### Module System
- [ ] **PyTorch-Like Module Pattern**
  - Inheriting from `Module` base class
  - Building models compositionally
  - Example: All files in `tensorrt_llm/layers/`

```python
from tensorrt_llm import Module
from tensorrt_llm.functional import *

class MyLayer(Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.linear = Linear(hidden_size, hidden_size)

    def forward(self, x):
        return self.linear(x)
```

#### Functional API
- [ ] **Functional Layer Construction**
  - Using `functional.py` for operations
  - Declarative network building
  - Example: `functional.py` (292KB) - Comprehensive operations

```python
from tensorrt_llm import functional as F

output = F.gelu(F.linear(input, weight))
```

#### Builder Pattern
- [ ] **Network Construction**
  - Using `Builder` to create networks
  - Engine compilation
  - Example: `builder.py` - TensorRT engine builder

```python
from tensorrt_llm import Builder

builder = Builder()
network = builder.create_network()
# Add layers to network
engine = builder.build_engine(network, config)
```

---

### Level 4: Configuration & API Design

#### Configuration Classes
- [ ] **Dataclasses and Configs**
  - Configuration management
  - Validation
  - Example: `llmapi/llm_args.py` (137KB) - Comprehensive config

#### High-Level API
- [ ] **LLM Class Design**
  - Unified user interface
  - Automatic building and caching
  - Example: `llmapi/llm.py` (58KB)

```python
from tensorrt_llm import LLM

llm = LLM(
    model="meta-llama/Llama-2-7b-hf",
    quant_config=...,
    tensor_parallel_size=2
)
outputs = llm.generate(prompts)
```

---

### Level 5: Runtime & Execution

#### Generation Runtime
- [ ] **Generation Logic**
  - Autoregressive generation
  - Sampling strategies
  - Beam search
  - Example: `runtime/generation.py` (228KB)

- [ ] **Model Runner**
  - TensorRT engine execution
  - Batch processing
  - Example: `runtime/model_runner.py`

#### Executor Pattern
- [ ] **Abstract Executors**
  - Multiple execution backends
  - Request tracking
  - Example: `executor/executor.py`

- [ ] **Distributed Execution**
  - Ray integration
  - RPC-based distribution
  - Example: `executor/ray_executor.py`, `executor/rpc_proxy.py`

---

### Level 6: Quantization

#### Quantization Concepts
- [ ] **Quantization Fundamentals**
  - INT4, INT8, FP8 quantization
  - Weight-only vs activation quantization
  - Example: `quantization/functional.py`, `quantization/layers.py`

- [ ] **ModelOpt Integration**
  - Automatic quantization
  - Calibration
  - Example: `quantization/quantize_by_modelopt.py`

---

### Level 7: Python-C++ Integration

#### Understanding Bindings
- [ ] **Nanobind Concepts**
  - How C++ classes are exposed to Python
  - Calling C++ from Python
  - Example: Using `tensorrt_llm.bindings`

```python
import tensorrt_llm.bindings as tllm_bindings

# C++ Executor exposed to Python
executor = tllm_bindings.Executor(config)
```

#### DLPack
- [ ] **Zero-Copy Tensor Transfer**
  - DLPack protocol
  - PyTorch ↔ TensorRT tensor sharing
  - Example: Tensor passing to C++ engine

---

## Part B: C++ Skills

### Level 1: C++ Fundamentals

#### Core Language Features
- [ ] **Object-Oriented C++**
  - Classes, inheritance, virtual functions
  - Constructors, destructors
  - RAII (Resource Acquisition Is Initialization)
  - Example: `cpp/tensorrt_llm/executor/executor.h`

- [ ] **Templates**
  - Function and class templates
  - Template specialization
  - Example: Kernel templates in `cpp/tensorrt_llm/kernels/`

- [ ] **Smart Pointers**
  - `std::unique_ptr`, `std::shared_ptr`
  - Memory management
  - Example: Throughout C++ codebase

- [ ] **Move Semantics**
  - Rvalue references
  - `std::move`
  - Efficient resource transfer

---

### Level 2: CUDA Programming

#### CUDA Basics
- [ ] **CUDA Kernel Fundamentals**
  - `__global__`, `__device__`, `__host__`
  - Thread hierarchy (blocks, grids, threads)
  - Memory hierarchy (global, shared, registers)
  - Example: All `.cu` files in `cpp/tensorrt_llm/kernels/`

```cuda
__global__ void myKernel(float* output, const float* input, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = input[idx] * 2.0f;
    }
}
```

- [ ] **Memory Management**
  - `cudaMalloc`, `cudaFree`
  - `cudaMemcpy`
  - Memory coalescing
  - Example: Kernel memory operations

- [ ] **Synchronization**
  - `__syncthreads()`
  - `cudaDeviceSynchronize()`
  - CUDA streams and events
  - Example: Multi-stream execution

#### Advanced CUDA
- [ ] **Shared Memory**
  - Shared memory allocation
  - Bank conflicts
  - Tiling for performance
  - Example: Flash Attention kernels in `kernels/fmha_v2/`

- [ ] **Warp-Level Primitives**
  - `__shfl_*` functions
  - Warp reductions
  - Example: Attention kernels

- [ ] **CUDA Streams**
  - Concurrent kernel execution
  - Overlapping compute and memory
  - Example: Model execution pipelines

---

### Level 3: TensorRT Integration

#### TensorRT Basics
- [ ] **TensorRT Concepts**
  - Network definition
  - Builder and engine
  - Inference runtime
  - Documentation: [TensorRT Developer Guide](https://docs.nvidia.com/deeplearning/tensorrt/)

- [ ] **TensorRT Plugins**
  - Creating custom plugins
  - Plugin serialization
  - Example: `cpp/tensorrt_llm/plugins/`

```cpp
class MyPlugin : public nvinfer1::IPluginV2DynamicExt {
    // Implement plugin interface
    int enqueue(...) override;
    size_t getWorkspaceSize(...) override;
    // ...
};
```

- [ ] **TensorRT Optimization**
  - Layer fusion
  - Kernel auto-tuning
  - Mixed precision

---

### Level 4: Performance Optimization

#### CUDA Performance
- [ ] **Occupancy Optimization**
  - Thread block sizing
  - Register usage
  - Shared memory usage
  - Tools: NVIDIA Nsight Compute

- [ ] **Memory Optimization**
  - Coalesced memory access
  - Shared memory bank conflicts
  - Texture/cache utilization

- [ ] **Compute Optimization**
  - Tensor Core utilization
  - Instruction-level parallelism
  - Warp divergence minimization

#### CUTLASS
- [ ] **CUTLASS Templates**
  - High-performance GEMM
  - Template-based kernel generation
  - Example: CUTLASS usage in MLA attention

---

### Level 5: Distributed Computing

#### MPI Basics
- [ ] **MPI Fundamentals**
  - `MPI_Init`, `MPI_Finalize`
  - Point-to-point communication
  - Collective operations
  - Example: Multi-GPU coordination

#### NCCL
- [ ] **NCCL for GPU Communication**
  - AllReduce, AllGather, etc.
  - Multi-GPU communication
  - Example: Tensor parallelism implementation

---

### Level 6: Python-C++ Bindings

#### Nanobind
- [ ] **Nanobind Fundamentals**
  - Exposing C++ classes to Python
  - Function binding
  - Type conversions
  - Example: `cpp/tensorrt_llm/nanobind/bindings.cpp`

```cpp
#include <nanobind/nanobind.h>

namespace nb = nanobind;

NB_MODULE(my_module, m) {
    nb::class_<MyClass>(m, "MyClass")
        .def(nb::init<int>())
        .def("my_method", &MyClass::my_method);
}
```

- [ ] **Type Conversions**
  - C++ ↔ Python type mapping
  - Custom converters
  - DLPack integration

---

## Recommended Learning Path

### Weeks 1-2: Python Foundations
1. Study PyTorch basics (tensors, nn.Module)
2. Understand Hugging Face Transformers
3. Read `tensorrt_llm/module.py` and `tensorrt_llm/layers/linear.py`

### Weeks 3-4: TensorRT-LLM API
1. Study `llmapi/llm.py` - High-level API
2. Trace `LLM.generate()` execution flow
3. Understand `builder.py` - Engine compilation

### Weeks 5-6: C++ Basics
1. Review C++ OOP, templates, smart pointers
2. Understand RAII and move semantics
3. Read `cpp/tensorrt_llm/executor/executor.h`

### Weeks 7-8: CUDA Programming
1. Learn CUDA kernel basics
2. Study memory hierarchy and synchronization
3. Read simple kernels in `cpp/tensorrt_llm/kernels/`
4. Example: `decodingKernels.cu`

### Weeks 9-10: Advanced Topics
1. Study Flash Attention implementation (`kernels/fmha_v2/`)
2. Understand quantization kernels
3. Learn TensorRT plugin system
4. Explore nanobind bindings

---

## Key Files to Study (In Order)

### Python
1. **`llmapi/llm.py`** (58KB) - Start here for high-level understanding
2. **`builder.py`** (59KB) - Engine compilation process
3. **`functional.py`** (292KB) - All available operations
4. **`runtime/generation.py`** (228KB) - Generation logic
5. **`models/llama.py`** - Example model implementation
6. **`layers/attention.py`** - Attention layer
7. **`quantization/layers.py`** (140KB) - Quantized layers

### C++
1. **`cpp/tensorrt_llm/executor/executor.h`** - Executor interface
2. **`cpp/tensorrt_llm/batch_manager/kvCacheManager.h`** - KV cache
3. **`cpp/tensorrt_llm/kernels/decodingKernels.cu`** - Simple kernel example
4. **`cpp/tensorrt_llm/kernels/samplingKernels.cu`** - Sampling implementation
5. **`cpp/tensorrt_llm/kernels/fmha_v2/`** - Flash Attention (advanced)
6. **`cpp/tensorrt_llm/nanobind/bindings.cpp`** - Python bindings

---

## Practical Exercises

### Exercise 1: Use High-Level API
```python
from tensorrt_llm import LLM

llm = LLM(model="gpt2")
outputs = llm.generate(["Hello world"])
print(outputs[0].text)

# Trace through:
# 1. How LLM loads the model
# 2. How engine is built and cached
# 3. How generation executes
```

### Exercise 2: Build Custom Model
```python
from tensorrt_llm import Module
from tensorrt_llm.functional import *

class CustomLayer(Module):
    def forward(self, x):
        # Implement custom logic
        return gelu(linear(x, self.weight))

# Build and compile to TensorRT engine
```

### Exercise 3: Read a CUDA Kernel
Study `cpp/tensorrt_llm/kernels/samplingKernels.cu`:
- Understand thread indexing
- See how sampling is implemented
- Learn memory access patterns

### Exercise 4: Understand Quantization
Trace INT8 quantization:
1. Python: `quantization/layers.py`
2. C++: Quantized GEMM kernels
3. See how weights are quantized and used

---

## Essential Tools

### Development Tools
- **CUDA Toolkit**: NVIDIA CUDA compiler and libraries
- **TensorRT**: NVIDIA TensorRT SDK
- **CMake**: Build system
- **GCC/Clang**: C++ compiler

### Profiling Tools
- **NVIDIA Nsight Compute**: CUDA kernel profiling
- **NVIDIA Nsight Systems**: System-level profiling
- **PyTorch Profiler**: Python-level profiling

### Debugging Tools
- **cuda-gdb**: CUDA debugging
- **pdb**: Python debugging
- **Valgrind**: Memory debugging (CPU)

---

## Libraries to Master

### Python Libraries
- **PyTorch**: Deep learning framework
- **Transformers**: Hugging Face models
- **NumPy**: Numerical computing
- **typing**: Type annotations

### C++ Libraries
- **TensorRT**: Inference optimization engine
- **CUDA**: GPU programming
- **cuBLAS**: GPU linear algebra
- **cuDNN**: GPU deep learning primitives
- **NCCL**: Multi-GPU communication
- **MPI**: Distributed computing (optional)

### Tools
- **CUTLASS**: High-performance CUDA templates
- **Triton**: JIT CUDA kernel generation (in `triton_kernels/`)
- **nanobind**: Python-C++ bindings

---

## Common Patterns in TensorRT-LLM

### Pattern 1: Module Definition
```python
class Attention(Module):
    def __init__(self, hidden_size, num_heads):
        super().__init__()
        self.qkv = Linear(hidden_size, 3 * hidden_size)

    def forward(self, hidden_states):
        qkv = self.qkv(hidden_states)
        return attention(qkv, ...)
```

### Pattern 2: Functional API
```python
from tensorrt_llm import functional as F

output = F.gelu(F.linear(input, weight, bias))
```

### Pattern 3: Builder Usage
```python
builder = Builder()
network = builder.create_network()
# Add layers
engine = builder.build_engine(network, config)
engine.save("model.engine")
```

### Pattern 4: CUDA Kernel
```cuda
__global__ void sampling_kernel(
    int* output_ids,
    const float* logits,
    const float* temperatures,
    int vocab_size
) {
    int batch_idx = blockIdx.x;
    // Sampling logic
}
```

---

## Resources

### Documentation
- [TensorRT Documentation](https://docs.nvidia.com/deeplearning/tensorrt/)
- [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [PyTorch Documentation](https://pytorch.org/docs/)
- [Nanobind Documentation](https://nanobind.readthedocs.io/)

### Books
- "Professional CUDA C Programming" by John Cheng
- "CUDA by Example" by Sanders & Kandrot
- "Effective Modern C++" by Scott Meyers
- "C++ Templates: The Complete Guide" by Vandevoorde & Josuttis

### Courses
- NVIDIA DLI: CUDA Programming
- NVIDIA DLI: Accelerating Inference with TensorRT
- Online: CUDA at Scale for the Enterprise (Coursera)

### Papers
- "FlashAttention: Fast and Memory-Efficient Exact Attention" (Dao et al.)
- "LLM.int8(): 8-bit Matrix Multiplication for Transformers" (Dettmers et al.)
- TensorRT optimization techniques (NVIDIA blogs)

---

## Understanding Build and Execution Flow

### Build Time
```
Python Model Definition
    ↓
Builder.create_network()
    ↓
Add layers via functional API
    ↓
TensorRT optimization
    ├── Layer fusion
    ├── Precision calibration
    ├── Kernel selection
    └── Graph optimization
    ↓
Serialized .engine file
```

### Runtime
```
Load .engine file
    ↓
Create TensorRT runtime (C++)
    ↓
Allocate GPU memory
    ↓
Execute inference (C++ executor)
    ├── CUDA kernels
    ├── TensorRT plugins
    └── Custom operations
    ↓
Return results to Python
```

---

## Summary

To effectively work with TensorRT-LLM:

### Python Skills Priority:
1. **PyTorch**: Essential for model definitions
2. **Hugging Face**: Model loading and tokenization
3. **Module System**: Understanding layer composition
4. **Functional API**: Declarative network building
5. **Builder Pattern**: Engine compilation

### C++ Skills Priority:
1. **CUDA Programming**: Critical for performance
2. **TensorRT API**: Core optimization engine
3. **Nanobind**: Python-C++ interface
4. **Templates**: Generic kernel development
5. **Smart Pointers**: Memory management

### Learning Strategy:
1. Start with Python high-level API (`LLM` class)
2. Understand model definitions and layers
3. Learn engine building process
4. Study C++ executor and runtime
5. Dive into CUDA kernels for performance understanding
6. Master quantization for deployment

The key differentiator of TensorRT-LLM is the tight integration between Python (user interface) and C++/CUDA (performance), with TensorRT providing automatic optimization. Understanding this layered architecture is crucial for effective development.
