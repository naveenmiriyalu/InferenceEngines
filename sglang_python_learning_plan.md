# Python Learning Plan for SGLang Codebase

This guide outlines the Python concepts and skills needed to understand and contribute to the SGLang codebase effectively.

---

## Level 1: Python Fundamentals (Prerequisites)

### Core Language Features
- [x] **Object-Oriented Programming**
  - Classes, inheritance, abstract base classes
  - Properties, class methods, static methods
  - Multiple inheritance and MRO
  - Example: `sglang/lang/backend/base_backend.py` - Backend interface

- [x] **Type Annotations**
  - Type hints for functions and variables
  - Generic types, Optional, Union
  - Protocol classes
  - Example: Extensive use throughout codebase

- [x] **Context Managers**
  - `__enter__` and `__exit__` methods
  - Resource management patterns
  - Example: Memory pool management in `srt/mem_cache/`

- [x] **Decorators**
  - Function and method decorators
  - `@property`, `@staticmethod`, `@classmethod`
  - Custom decorators for DSL
  - Example: `@sgl.function` decorator in `lang/api.py`

---

## Level 2: Advanced Python for DSL Design

### Metaprogramming & DSL Patterns
- [ ] **Function Decorators for DSL**
  - Creating domain-specific decorators
  - Capturing function context
  - Example: `@sgl.function` in `lang/api.py`

- [ ] **Operator Overloading**
  - `__add__`, `__iadd__` for DSL syntax
  - `s += sgl.user(...)` pattern
  - Example: `lang/interpreter.py` - State class

- [ ] **Magic Methods**
  - `__call__` for callable objects
  - `__getitem__`, `__setitem__` for state access
  - Example: DSL state management

### Intermediate Representations
- [ ] **AST and IR Design**
  - Building intermediate representations
  - Program analysis and transformation
  - Example: `lang/ir.py` - SGLang IR classes (SglGen, SglSelect, etc.)

- [ ] **Visitor Pattern**
  - Traversing IR trees
  - Code generation from IR
  - Example: `lang/interpreter.py` - Executing IR

---

## Level 3: Async Programming & IPC

### Async Fundamentals
- [ ] **asyncio Basics**
  - `async`/`await` syntax
  - Event loops and coroutines
  - `asyncio.Queue`, `asyncio.Task`
  - Example: `srt/entrypoints/http_server.py` - FastAPI endpoints

- [ ] **Async Generators**
  - `async for` and `yield`
  - Streaming responses
  - Example: Streaming chat completions in HTTP server

- [ ] **Async Context Managers**
  - `async with` statements
  - Async resource cleanup
  - Example: HTTP client connections

### Inter-Process Communication
- [ ] **ZeroMQ (ZMQ)**
  - Socket types: REQ/REP, PUSH/PULL, PUB/SUB
  - High-performance message passing
  - Example: `srt/entrypoints/engine.py` - Three-process communication
    - TokenizerManager ↔ Scheduler ↔ DetokenizerManager

- [ ] **Multiprocessing Module**
  - `Process`, `Queue`, `Pipe`
  - Shared memory considerations
  - Process lifecycle management
  - Example: Spawning scheduler and detokenizer processes

- [ ] **Serialization for IPC**
  - Pickle for Python objects
  - MessagePack for efficiency
  - Custom serialization
  - Example: Passing requests between processes

---

## Level 4: PyTorch & Deep Learning

### PyTorch Basics
- [ ] **Tensor Operations**
  - Creating and manipulating tensors
  - GPU memory management (`torch.cuda`)
  - Device placement (`.to(device)`, `.cuda()`)
  - Example: All model execution in `srt/model_executor/model_runner.py`

- [ ] **nn.Module**
  - Creating custom modules
  - `forward()` method
  - Parameter registration and initialization
  - Example: Model implementations in `srt/models/`

- [ ] **Custom CUDA Operations**
  - `torch.ops` registry
  - Registering custom ops from C++
  - Example: `torch.ops.sglang_kernels.*` from `sgl-kernel/`

### Advanced PyTorch
- [ ] **Torch.compile**
  - PyTorch 2.0+ compilation
  - Graph optimization
  - Example: `srt/compilation/` - torch.compile integration

- [ ] **CUDA Events and Streams**
  - `torch.cuda.Event()` for synchronization
  - `torch.cuda.Stream()` for concurrent execution
  - Example: CUDA graph execution

- [ ] **Distributed Primitives**
  - `torch.distributed` module
  - NCCL backend for GPU communication
  - Process groups
  - Example: `srt/distributed/parallel_state.py`

---

## Level 5: SGLang-Specific Patterns

### DSL Implementation
- [ ] **State Management**
  - Tracking execution state
  - Variable scoping
  - Example: `lang/interpreter.py` - State class

- [ ] **Backend Abstraction**
  - Multiple backend support
  - Backend interface design
  - Example: `lang/backend/` - OpenAI, Anthropic, Runtime backends

- [ ] **Template Processing**
  - Chat template handling
  - Jinja2 templates
  - Example: `srt/managers/template_manager.py`

### Request Processing
- [ ] **Request Lifecycle**
  - From HTTP request to response
  - State tracking through pipeline
  - Example: `srt/managers/io_struct.py` - Data structures

- [ ] **Batch Construction**
  - Dynamic batching
  - Request prioritization
  - Example: `srt/managers/schedule_batch.py`

### Memory Management
- [ ] **Radix Trees**
  - Prefix tree data structure
  - Automatic sharing of common prefixes
  - Example: `srt/mem_cache/radix_cache.py` - RadixAttention

- [ ] **Memory Pooling**
  - GPU memory allocation strategies
  - Block management
  - Example: `srt/mem_cache/memory_pool.py`

---

## Level 6: Performance & Profiling

### CUDA Graphs
- [ ] **CUDA Graph Basics**
  - Graph capture and replay
  - Reducing CPU overhead
  - Example: `srt/model_executor/cuda_graph_runner.py`

- [ ] **Piecewise CUDA Graphs**
  - Memory-efficient graph execution
  - Example: `srt/model_executor/piecewise_cuda_graph_runner.py`

### Profiling Tools
- [ ] **Python Profilers**
  - `cProfile`, `line_profiler`
  - Memory profilers
  - Bottleneck identification

- [ ] **PyTorch Profiler**
  - `torch.profiler`
  - CUDA kernel profiling
  - Memory usage tracking

### Environment Management
- [ ] **Environment Variables**
  - Reading and validating env vars
  - Configuration via environment
  - Example: `srt/environ.py`

---

## Level 7: Library Integrations

### HTTP Servers
- [ ] **FastAPI**
  - Route definition and decorators
  - Request/response models (Pydantic)
  - Async endpoints
  - Dependency injection
  - Example: `srt/entrypoints/http_server.py` - OpenAI-compatible API

- [ ] **Uvicorn/Gunicorn**
  - ASGI server deployment
  - Worker management
  - Example: HTTP server deployment

### gRPC
- [ ] **gRPC Basics**
  - Protocol Buffers
  - Service definition
  - Async gRPC
  - Example: `srt/entrypoints/grpc_server.py`

### Hugging Face Ecosystem
- [ ] **Transformers Library**
  - Loading models and configs
  - Tokenizers (Fast tokenizers)
  - AutoModel classes
  - Example: Model loading in `srt/model_executor/`

- [ ] **SafeTensors**
  - Efficient weight loading
  - Security advantages
  - Example: Weight loading utilities

### Constrained Generation
- [ ] **XGrammar**
  - Grammar-based constraints
  - FSM construction
  - Example: `srt/constrained/xgrammar_backend.py`

- [ ] **LLGuidance**
  - Structured output generation
  - Example: `srt/constrained/llguidance_backend.py`

- [ ] **Outlines**
  - JSON schema constraints
  - Regex patterns
  - Example: `srt/constrained/outlines_backend.py`

---

## Level 8: Testing & Development

### Testing Frameworks
- [ ] **pytest**
  - Test discovery and fixtures
  - Parametrized tests
  - Async test support
  - Example: `test/` directory

- [ ] **Mocking**
  - `unittest.mock`
  - Patching for isolation
  - Example: Testing without GPU

### Code Quality
- [ ] **Type Checking**
  - `mypy` for static analysis
  - Type stubs

- [ ] **Linting & Formatting**
  - `ruff` for fast linting
  - Code formatting

---

## Recommended Learning Path

### Week 1-2: Foundations
1. Review Python OOP, decorators, type hints
2. Study DSL patterns and operator overloading
3. Understand `@sgl.function` decorator in `lang/api.py`

### Week 3-4: Architecture Understanding
1. Read `srt/entrypoints/engine.py` - Three-process architecture
2. Trace ZMQ communication flow
3. Study `lang/interpreter.py` - DSL interpreter

### Week 5-6: Deep Dive
1. Explore scheduler logic in `srt/managers/scheduler.py`
2. Understand RadixAttention in `srt/mem_cache/radix_cache.py`
3. Study continuous batching implementation

### Week 7-8: Advanced Topics
1. CUDA graph optimization
2. Constrained generation backends
3. Distributed inference patterns

---

## Key Files to Study (In Order)

1. **Start Here**: `python/sglang/lang/api.py`
   - DSL entry points
   - High-level API design

2. **DSL Interpreter**: `python/sglang/lang/interpreter.py`
   - How DSL programs execute
   - State management

3. **Engine Entry**: `python/sglang/srt/entrypoints/engine.py`
   - Three-process orchestration
   - ZMQ communication setup

4. **HTTP Server**: `python/sglang/srt/entrypoints/http_server.py`
   - FastAPI endpoints
   - Request handling

5. **Scheduler**: `python/sglang/srt/managers/scheduler.py` (132KB)
   - Core scheduling logic
   - Continuous batching
   - Prefill-decode disaggregation

6. **Model Runner**: `python/sglang/srt/model_executor/model_runner.py` (106KB)
   - Model execution
   - Forward pass orchestration

7. **RadixCache**: `python/sglang/srt/mem_cache/radix_cache.py`
   - Prefix caching implementation
   - Radix tree data structure

8. **Memory Pool**: `python/sglang/srt/mem_cache/memory_pool.py` (74KB)
   - GPU memory management
   - Block allocation

---

## Practical Exercises

### Exercise 1: Create a Simple DSL Function
```python
import sglang as sgl

@sgl.function
def simple_qa(s, question):
    s += sgl.user(question)
    s += sgl.assistant(sgl.gen("answer", max_tokens=50))

# Trace through:
# 1. How @sgl.function decorator works
# 2. How s += ... modifies state
# 3. How backend executes the function
```

### Exercise 2: Trace a Request Through Three Processes
Follow a single request:
```
HTTP Request → TokenizerManager (Process 1)
              ↓ (ZMQ)
         Scheduler (Process 2)
              ↓ (ZMQ)
    DetokenizerManager (Process 3)
              ↓
         HTTP Response
```

### Exercise 3: Understand RadixAttention
Study `srt/mem_cache/radix_cache.py`:
- How radix tree stores KV cache
- Prefix matching algorithm
- Cache reuse mechanism

### Exercise 4: Add a Custom Sampling Method
Modify `srt/layers/sampler.py` to implement a custom sampling strategy.

---

## Python Libraries to Master

### Essential
- **PyTorch**: Deep learning framework
- **asyncio**: Async programming
- **zmq (pyzmq)**: Inter-process communication
- **multiprocessing**: Process management
- **typing**: Type annotations

### Important
- **FastAPI**: HTTP API framework
- **Pydantic**: Data validation
- **transformers**: Hugging Face models
- **numpy**: Numerical computing

### SGLang-Specific
- **FlashInfer**: Fast attention kernels
- **XGrammar**: Grammar constraints
- **LLGuidance**: Structured output
- **Outlines**: JSON schema generation
- **Triton**: JIT CUDA kernels (for `jit_kernel/`)

---

## Common Patterns in SGLang

### Pattern 1: DSL Function Decorator
```python
@sgl.function
def my_function(s, arg1, arg2):
    # s is the state object
    s += sgl.user(f"Question: {arg1}")
    s += sgl.assistant(sgl.gen("response"))
```

### Pattern 2: State Modification
```python
# += operator is overloaded to append to state
s += sgl.user("Hello")
s += sgl.assistant(sgl.gen("answer"))

# Access generated content
print(s["answer"])
```

### Pattern 3: Backend Abstraction
```python
# Same DSL code works with different backends
state = my_function.run(
    backend="runtime",  # or "openai", "anthropic"
    arg1="test"
)
```

### Pattern 4: ZMQ Communication
```python
import zmq

context = zmq.Context()
socket = context.socket(zmq.PUSH)
socket.connect("ipc:///tmp/sglang-scheduler")
socket.send_pyobj(request)
```

### Pattern 5: Radix Tree Cache Lookup
```python
# Automatic prefix caching in RadixAttention
# If two requests share prefix "Hello, how are",
# they reuse the same KV cache blocks
```

---

## Understanding the Three-Process Architecture

### Process 1: TokenizerManager
```python
# In srt/entrypoints/engine.py or http_server.py
# Receives HTTP request
# Tokenizes input text
# Sends tokenized request via ZMQ to Scheduler
```

### Process 2: Scheduler
```python
# In srt/managers/scheduler.py
# Receives tokenized requests
# Performs continuous batching
# Manages KV cache with RadixAttention
# Calls ModelRunner for execution
# Sends token IDs to Detokenizer
```

### Process 3: DetokenizerManager
```python
# In srt/managers/detokenizer_manager.py
# Receives token IDs
# Converts to text
# Sends back to HTTP server
```

---

## Resources

### Documentation
- [PyTorch Documentation](https://pytorch.org/docs/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [ZeroMQ Guide](https://zguide.zeromq.org/)
- [Python asyncio](https://docs.python.org/3/library/asyncio.html)

### SGLang-Specific
- SGLang GitHub: Issues and PRs
- SGLang documentation
- Research papers on RadixAttention

### Books
- "Fluent Python" by Luciano Ramalho
- "High Performance Python" by Gorelick & Ozsvald
- "Python Concurrency with asyncio" by Matthew Fowler
- "Crafting Interpreters" by Robert Nystrom (for DSL design)

---

## Unique SGLang Concepts to Master

### 1. RadixAttention
- Understand radix tree data structure
- Prefix matching for cache reuse
- Automatic memory optimization
- Files: `srt/mem_cache/radix_cache.py`, `srt/mem_cache/hiradix_cache.py`

### 2. Continuous Batching
- Dynamic batch formation
- Prefill-decode interleaving
- Request prioritization
- Files: `srt/managers/scheduler.py`, `srt/managers/schedule_batch.py`

### 3. Prefill-Decode Disaggregation
- Separate streams for prefill vs decode
- Resource optimization
- Files: `srt/disaggregation/`

### 4. Grammar-Constrained Generation
- FSM-based token filtering
- Multiple backend support (XGrammar, LLGuidance, Outlines)
- JSON schema constraints
- Files: `srt/constrained/`

### 5. JIT vs AOT Kernels
- **JIT** (`jit_kernel/`): Triton-based, runtime compilation
- **AOT** (`sgl-kernel/`): Pre-compiled CUDA kernels
- Understanding when to use each

---

## Summary

To effectively work with SGLang, focus on:

1. **DSL Design**: Understand decorators, operator overloading, metaprogramming
2. **Async Python**: Essential for HTTP server and IPC
3. **ZeroMQ**: Three-process architecture communication
4. **PyTorch**: Model execution and custom ops
5. **FastAPI**: HTTP API implementation
6. **RadixAttention**: Unique prefix caching mechanism
7. **Continuous Batching**: Dynamic request scheduling
8. **Constrained Generation**: Grammar-based structured output

Start with the DSL (`lang/api.py`, `lang/interpreter.py`), understand the three-process architecture (`srt/entrypoints/engine.py`), then dive into scheduling and memory management.

The codebase emphasizes both **user experience** (DSL) and **performance** (optimized runtime), so understanding both frontend and backend is crucial.
