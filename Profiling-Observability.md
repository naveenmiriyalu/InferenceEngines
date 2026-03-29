# Profiling and Observability Support: vLLM, SGLang, and TensorRT-LLM

## Table of Contents
1. [Overview](#overview)
2. [vLLM Profiling & Observability](#1-vllm-profiling--observability)
3. [SGLang Profiling & Observability](#2-sglang-profiling--observability)
4. [TensorRT-LLM Profiling & Observability](#3-tensorrt-llm-profiling--observability)
5. [Comparative Analysis](#4-comparative-analysis)
6. [Best Practices](#5-best-practices)
7. [Summary](#6-summary)

---

## Overview

Production LLM serving requires comprehensive profiling and observability to:
- **Identify performance bottlenecks** (compute vs memory vs communication)
- **Debug accuracy issues** (numerical stability, quantization errors)
- **Monitor system health** (memory leaks, request failures)
- **Optimize resource utilization** (GPU, CPU, network)

This document analyzes profiling and observability capabilities across three major frameworks.

---

## 1. vLLM Profiling & Observability

### 1.1 Profiling Tools

#### PyTorch Profiler Integration

**Location**: `vllm/config/profiler.py` (148 lines), `vllm/utils/profiling.py` (57 lines)

```python
@dataclass
class ProfilerConfig:
    """PyTorch profiler configuration"""

    profiler: str = 'torch'  # 'torch' or 'cuda'
    torch_profiler_dir: str = '/tmp/vllm_profiles'

    # Profiling options
    torch_profiler_with_stack: bool = False  # Python stack traces
    torch_profiler_with_flops: bool = False  # FLOP counting
    torch_profiler_with_memory: bool = False  # Memory profiling
    torch_profiler_record_shapes: bool = False  # Tensor shapes
    torch_profiler_use_gzip: bool = True  # Compress traces

    # Scheduling
    delay_iterations: int = 5  # Skip initial iterations (warmup)
    max_iterations: int = 100  # Maximum profiling duration
    warmup_iterations: int = 2  # Discard JIT compilation
    active_iterations: int = 5  # Data collection iterations
    wait_iterations: int = 0  # Zero-overhead wait period
```

**Usage Example**:
```bash
# Start vLLM with profiling
vllm serve meta-llama/Llama-2-7b-hf \
  --profiler torch \
  --torch-profiler-dir /tmp/profiles \
  --torch-profiler-with-stack \
  --torch-profiler-with-memory
```

**Output**: JSON traces viewable in Perfetto UI (`ui.perfetto.dev`)

#### CUDA Profiler (Nsight Systems)

**Documentation**: `vllm/docs/contributing/profiling.md` (Lines 84-174)

```bash
# Profile with Nsight Systems
nsys profile \
  --trace-fork-before-exec=true \
  --cuda-graph-trace=node \
  --output=vllm_profile.nsys-rep \
  python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-2-7b-hf

# Dynamic profiling with CUDA Profiler API
nsys profile \
  --capture-range=cudaProfilerApi \
  --capture-range-end=stop \
  python script.py
```

**Features**:
- CUDA kernel timeline visualization
- Memory transfer tracking
- CUDA graph node tracing
- Python stack correlation

#### Python cProfile

**Location**: `vllm/utils/profiling.py` (Lines 12-57)

```python
from vllm.utils.profiling import cprofile, cprofile_context

# Decorator for function profiling
@cprofile("results.prof")
def expensive_function():
    # Profile this function
    pass

# Context manager for block profiling
with cprofile_context("block.prof"):
    function_to_profile()
```

**Benefits**:
- No GPU profiling overhead
- Python-level hotspot identification
- Integrates with `py-spy`, `scalene`

### 1.2 Observability Features

#### Hierarchical Logging

**Location**: `vllm/logger.py` (318 lines)

```python
from vllm.logger import init_logger

logger = init_logger(__name__)

# De-duplication utilities
logger.debug_once("message", scope="process")  # Once per process
logger.info_once("message", scope="global")    # Once globally
logger.warning_once("message", scope="local")  # Once per rank

# Custom formatters
# - ColoredFormatter: Terminal output with ANSI colors
# - NewLineFormatter: Multi-line log entries
```

**Environment Variables**:
```bash
VLLM_LOGGING_LEVEL=DEBUG  # Control verbosity
VLLM_LOGGING_COLOR=1      # Enable colored output
VLLM_LOGGING_STREAM=stdout  # stdout or stderr
VLLM_TRACE_FUNCTION=1     # Enable function call tracing
```

#### Function Call Tracing

**Location**: `vllm/logger.py` (Lines 254-317)

```python
def enable_trace_function_call(log_file_path: str, root_dir: str):
    """Enable thread-level function call tracing"""
    # Captures every function call/return with timestamps
    # Use case: Debug complex execution flows

# Usage:
VLLM_TRACE_FUNCTION=1 VLLM_RPC_TIMEOUT=1800000 python script.py
```

**Output**:
```
[Thread-1] CALL   module.function (file.py:123)
[Thread-1] RETURN module.function → result (elapsed: 0.05s)
```

#### OpenTelemetry Integration

**Location**: `vllm/config/observability.py` (153 lines), `vllm/tracing/otel.py` (200+ lines)

```python
# Configuration
@dataclass
class ObservabilityConfig:
    otlp_traces_endpoint: str = ""  # Trace exporter endpoint
    collect_detailed_traces: str = "disabled"  # "model", "worker", "all"

# Span creation decorator
from vllm.tracing.otel import instrument_otel

@instrument_otel
def my_function(arg1, arg2):
    # Automatically wrapped in OpenTelemetry span
    pass

# Context propagation
from vllm.tracing.otel import extract_trace_context, init_otel_tracer

tracer = init_otel_tracer()
context = extract_trace_context(http_headers)
with tracer.start_as_current_span("operation", context=context):
    # Execute operation with trace context
    pass
```

**Supported Protocols**:
- gRPC export (OTLP/gRPC)
- HTTP/Protobuf export (OTLP/HTTP)

**Span Attributes**:
- Code location: module, function, file, line number
- Resource attributes: process ID, process kind
- Custom attributes: request IDs, batch sizes

#### Prometheus Metrics

**Location**: `vllm/v1/metrics/prometheus.py` (83 lines)

```python
from vllm.v1.metrics.prometheus import setup_multiprocess_prometheus

# Setup multiprocess mode
registry = setup_multiprocess_prometheus()

# Metrics endpoint
# GET http://localhost:8000/metrics

# Key metrics:
# - vllm:time_to_first_token_seconds (Histogram)
# - vllm:inter_token_latency_seconds (Histogram)
# - vllm:kv_cache_usage_perc (Gauge)
# - vllm:prefix_cache_hits / vllm:prefix_cache_queries (Counter)
```

**See `Metrics.md` for comprehensive metric catalog**

#### NVTX Tracing for Nsight

**Location**: `vllm/utils/nvtx_pytorch_hooks.py` (200+ lines)

```python
# Enable layerwise NVTX markers
from vllm.config.observability import ObservabilityConfig

config = ObservabilityConfig(enable_layerwise_nvtx_tracing=True)

# Automatically inserts NVTX markers for:
# - Conv layers
# - Pooling layers
# - Attention layers
# - Tensor shapes and layer parameters
```

**Visualization**:
- View in NVIDIA Nsight Systems
- Timeline shows layer execution with parameters
- Correlate with CUDA kernels

#### Debug Modes

```bash
# Iteration details logging
--enable-logging-iteration-details

# KV cache metrics (sampled)
--kv-cache-metrics-sample=0.05  # Sample 5% of blocks

# CUDA graph metrics
--cudagraph-metrics

# Multimodal processor stats
--enable-mm-processor-stats
```

### 1.3 Memory Profiling

**Garbage Collection Debugging**:
```bash
# Track GC collections
VLLM_GC_DEBUG=1 python script.py

# Track top objects
VLLM_GC_DEBUG='{"top_objects":5}' python script.py
```

**Documentation**: `vllm/docs/contributing/profiling.md` (Lines 250-256)

### 1.4 Configuration Files

**Key Files**:
- `vllm/config/profiler.py`: Profiler configuration (148 lines)
- `vllm/config/observability.py`: Observability settings (153 lines)
- `vllm/docs/contributing/profiling.md`: Complete guide (256 lines)

---

## 2. SGLang Profiling & Observability

### 2.1 Profiling Tools

#### PyTorch Profiler API

**Location**: `sglang/profiler.py` (100+ lines)

```python
# HTTP endpoints for profiling control
import requests

# Start profiling
response = requests.post(
    "http://localhost:30000/start_profile",
    json={
        "output_dir": "/tmp/profiles",
        "num_steps": 5,
        "activities": ["CPU", "GPU"],
        "profile_by_stage": True,  # Separate prefill/decode traces
        "merge_profiles": False,
        "profile_prefix": "llm"
    }
)

# Stop profiling
response = requests.post("http://localhost:30000/stop_profile")
```

**Features**:
- Stage-based profiling (prefill vs decode)
- Multi-worker profile merging
- AMD NPU support via `torch_npu` patches

**Profile Manager**:

**Location**: `sglang/srt/utils/profile_utils.py` (Lines 30-100)

```python
class ProfileManager:
    """Manages stage-triggered profiling"""

    def __init__(self, output_dir, activities, num_steps):
        self.profiler = torch.profiler.profile(
            activities=activities,
            record_shapes=True,
            with_stack=True,
            profile_memory=False,
        )
```

#### RPD Profiler (AMD ROCm)

**Location**: `sglang/3rdparty/amd/profiling/PROFILING.md` (426 lines)

**RPD (ROCm Profile Data)**:
- Low-overhead cross-platform profiler
- Works on both AMD and NVIDIA GPUs
- Python + CPU activity tracing
- Output: RPD binary → JSON via `rpd2tracing.py`

```python
from rpdTracerControl import rpdTracerControl

rpd = rpdTracerControl()
rpd.setPythonTrace(True)  # Include Python calls
rpd.start()

# Named profiling regions
rpd.rangePush("", "my_kernel", "")
# ... code to profile ...
rpd.rangePop()

rpd.stop()
rpd.flush()

# Convert to Perfetto JSON
# python rpd2tracing.py output.rpd
```

**Visualization**: Perfetto UI (streaming mode for large files)

### 2.2 Observability Features

#### Request Logging

**Location**: `sglang/srt/utils/request_logger.py` (100+ lines)

```python
# Configuration flags
--log-requests  # Enable request logging
--log-request-level=info  # Verbosity
--log-requests-format=json  # json or text

# Log targets
logger.log_received_request(obj, tokenizer, request)
logger.log_completed_request(obj, output)
logger.log_request_output(obj, output)

# Header extraction for tracing
# Whitelisted headers: X-Request-ID, traceparent, tracestate
```

**Exceeded Time Logging**:
```bash
SGLANG_LOG_REQUEST_EXCEEDED_MS=1000  # Log requests >1s
```

#### OpenTelemetry Tracing

**Location**: `sglang/srt/tracing/trace.py` (100+ lines)

```python
from sglang.srt.tracing.trace import (
    is_tracing_enabled,
    extract_trace_headers,
    trace_slice,
    trace_slice_end,
    trace_event_batch
)

# Check OTEL availability
if is_tracing_enabled():
    # Extract trace context
    context = extract_trace_headers(http_headers)

    # Create span
    trace_slice("operation_name", context)
    # ... operation ...
    trace_slice_end("operation_name")
```

**Thread Context Management**:
- `SglangTraceThreadInfo`: Thread metadata
- `SglangTraceSliceContext`: Span context
- `TraceContextTextMapPropagator`: Context propagation

#### Prometheus Metrics

**Location**: `sglang/docs/references/production_metrics.md`

```bash
# Enable metrics
python -m sglang.launch_server \
  --model-path meta-llama/Llama-3.1-8B \
  --enable-metrics

# Access metrics
curl http://localhost:30000/metrics
```

**Key Metrics** (See `Metrics.md` for full list):
- `sglang:time_to_first_token_seconds` (Histogram)
- `sglang:cache_hit_rate` (Gauge)
- `sglang:func_latency_seconds` (Histogram with function labels)

#### Crash Dump & Replay

**Location**: `sglang/docs/advanced_features/observability.md` (Lines 30-35)

```bash
# Crash dump (last 5 minutes of requests)
python -m sglang.launch_server \
  --crash-dump-folder /tmp/crash_dump

# Request recording
python3 -m sglang.srt.managers.configure_logging \
  --url http://localhost:30000 \
  --dump-requests-folder /tmp/sglang_request_dump

# Replay
python scripts/playground/replay_request_dump.py \
  --dump-folder /tmp/sglang_request_dump
```

### 2.3 Configuration Files

**Key Files**:
- `sglang/profiler.py`: HTTP API for profiling (100+ lines)
- `sglang/srt/utils/profile_utils.py`: Profile manager (100+ lines)
- `sglang/srt/tracing/trace.py`: OTEL integration (200+ lines)
- `sglang/3rdparty/amd/profiling/PROFILING.md`: RPD guide (426 lines)

---

## 3. TensorRT-LLM Profiling & Observability

### 3.1 Profiling Tools

#### Timer-Based Profiling

**Location**: `tensorrt_llm/profiler.py` (Lines 53-107)

```python
from tensorrt_llm.profiler import Timer

timer = Timer()

# Start timer
timer.start("forward_pass")

# ... execute operation ...

# Stop and record
elapsed = timer.stop("forward_pass")

# Get results
total_time = timer.elapsed_time_in_sec("forward_pass")

# Print summary
timer.summary()
# Output:
# forward_pass: 0.123456s
```

**Default Instance**: `_default_timer` (module-level singleton)

#### Memory Profiling

**Location**: `tensorrt_llm/profiler.py` (Lines 131-150)

```python
from tensorrt_llm.profiler import host_memory_info, device_memory_info

# Host memory (CPU)
alloc, free, total = host_memory_info(pid=os.getpid())
print(f"Host: {alloc/1e9:.2f}GB / {total/1e9:.2f}GB")

# Device memory (GPU)
dev_alloc, dev_free, dev_total = device_memory_info(device=0)
print(f"GPU: {dev_alloc/1e9:.2f}GB / {dev_total/1e9:.2f}GB")
```

**Dependencies**:
- Host: `psutil` (USS - Unique Set Size)
- Device: `pynvml` (NVIDIA Management Library)

#### NVTX Layer Tracing

**Location**: `tensorrt_llm/_torch/pyexecutor/layerwise_nvtx_marker.py` (200+ lines)

```python
from tensorrt_llm._torch.pyexecutor.layerwise_nvtx_marker import (
    LayerwiseNvtxMarker
)

marker = LayerwiseNvtxMarker()
marker.register_hooks(model)

# Run with Nsight Systems
# nsys profile python script.py
```

**Features**:
- Pre/post forward hooks on all layers
- Tensor shape extraction
- Layer parameter capture (Conv, Pooling, Attention)
- Iteration counting

### 3.2 Observability Features

#### Unified Logger

**Location**: `tensorrt_llm/logger.py` (150+ lines)

```python
from tensorrt_llm.logger import logger

# Environment variable
# TLLM_LOG_LEVEL=verbose

# Logging methods
logger.error("Critical error message")
logger.warning("Warning message")
logger.info("Informational message")
logger.verbose("Verbose message")
logger.debug("Debug message")

# Singleton pattern - one log per occurrence
logger.log_once("Message shown once", level="warning")

# Distributed logging
logger.set_rank(rank)  # Prefix logs with rank
```

**Severity Levels**:
- INTERNAL_ERROR → ERROR → WARNING → INFO → VERBOSE → DEBUG

**Integration**:
- TensorRT logger: `trt.Logger`
- Polygraphy logger: `G_LOGGER`
- Python logging: `logging.getLogger()`

#### Benchmark Infrastructure

**Location**:
- `tensorrt_llm/commands/bench.py`: CLI benchmarking
- `tensorrt_llm/scaffolding/benchmark.py`: Utilities
- `tensorrt_llm/serve/scripts/benchmark_serving.py`: Serving benchmark

**Metrics Collected**:
- Latency: min, max, mean, median, P95, P99
- Throughput: requests/sec, tokens/sec
- Memory: peak device/host usage
- Time breakdown: prefill vs decode
- Cache statistics: hit rates, evictions

### 3.3 Configuration Files

**Key Files**:
- `tensorrt_llm/profiler.py`: Timer + memory profiling (200+ lines)
- `tensorrt_llm/logger.py`: Unified logging (150+ lines)
- `tensorrt_llm/_torch/pyexecutor/layerwise_nvtx_marker.py`: NVTX (200+ lines)

---

## 4. Comparative Analysis

### 4.1 Feature Matrix

| Feature | vLLM | SGLang | TensorRT-LLM |
|---------|------|--------|--------------|
| **PyTorch Profiler** | ✅ Full (stack, shapes, memory, FLOPs) | ✅ Basic (stage-based) | ❌ (via wrapper) |
| **CUDA Profiler** | ✅ (documented, Nsight) | ⚠️ (manual setup) | ✅ (NVTX + Nsight) |
| **OpenTelemetry** | ✅ Full (gRPC/HTTP) | ✅ Full (gRPC/HTTP) | ❌ |
| **Prometheus** | ✅ Comprehensive | ✅ Comprehensive | ❌ (custom only) |
| **Request Tracing** | ✅ (OpenTelemetry) | ✅ (OTEL + dump/replay) | ❌ |
| **Memory Profiling** | ✅ (GC tracking) | ✅ (OTEL) | ✅ (host/device) |
| **NVTX Tracing** | ✅ (layerwise) | ⚠️ (hooks in utils) | ✅ (comprehensive) |
| **Visualization** | Perfetto, Chrome | Perfetto, RPD | Nsight Systems |
| **Output Formats** | JSON (gzip), Protobuf | JSON, RPD binary | CSV, JSON |
| **Function Profiling** | cProfile, trace | Function timer | Timer class |
| **Logging Levels** | 4 levels | 5+ levels | 6 levels |
| **Request Logging** | ⚠️ (basic) | ✅ (comprehensive) | ❌ |
| **Crash Debugging** | ⚠️ (trace enabled) | ✅ (crash dump + replay) | ❌ |
| **Distributed Support** | ✅ (scope: global/local) | ✅ (TP rank tracking) | ✅ (rank ID) |
| **GC Debugging** | ✅ | ❌ | ❌ |

### 4.2 Ease of Use Comparison

**vLLM**:
- ✅ Most comprehensive out-of-the-box profiling
- ✅ Well-documented with examples
- ✅ Integrated metrics and tracing
- ❌ Complex configuration for advanced features

**SGLang**:
- ✅ HTTP API for profiling control (easy integration)
- ✅ Crash dump/replay for debugging
- ✅ Multi-backend support (CUDA, AMD, NPU)
- ❌ Fewer built-in profiling tools

**TensorRT-LLM**:
- ✅ Simple timer-based profiling
- ✅ NVTX integration for Nsight
- ❌ Limited observability compared to others
- ❌ No distributed tracing support

### 4.3 Performance Overhead

| Tool | vLLM | SGLang | TensorRT-LLM |
|------|------|--------|--------------|
| **PyTorch Profiler** | ~10-30% overhead | ~10-30% | N/A |
| **CUDA Profiler** | ~5-15% | ~5-15% | ~5-15% |
| **NVTX Markers** | <5% | <5% | <5% |
| **cProfile** | ~5-10% | ~5-10% (via timer) | ~5-10% |
| **Metrics Collection** | <1% (default) | <1% | N/A |
| **OpenTelemetry** | <2% (span creation) | <2% | N/A |

---

## 5. Best Practices

### 5.1 Profiling Workflow

**Initial Diagnosis**:
1. Enable metrics collection to identify high-level bottlenecks
2. Use timer-based profiling for coarse-grained measurement
3. Narrow down to specific code paths

**Detailed Profiling**:
1. PyTorch profiler for Python/GPU correlation
2. CUDA profiler (Nsight) for kernel-level analysis
3. NVTX markers for layer-level granularity

**Iterative Optimization**:
1. Profile → Identify hotspot → Optimize → Verify
2. Compare before/after metrics
3. Test with production workload

### 5.2 Observability Configuration

**Development**:
```bash
# vLLM
VLLM_LOGGING_LEVEL=DEBUG \
VLLM_TRACE_FUNCTION=1 \
vllm serve ... --enable-logging-iteration-details

# SGLang
--log-requests \
--log-request-level=debug \
--crash-dump-folder=/tmp/crash
```

**Production**:
```bash
# vLLM
--otlp-traces-endpoint=http://jaeger:4317 \
--collect-detailed-traces=worker

# SGLang
--enable-metrics

# Prometheus scraping both on /metrics endpoint
```

### 5.3 Common Issues & Solutions

**Issue**: High memory usage not tracked by profiler
- **Solution**: Enable GC debugging (`VLLM_GC_DEBUG=1`)

**Issue**: Profiler overhead too high for production
- **Solution**: Use sampling-based profiling or metrics only

**Issue**: Cannot correlate Python code with CUDA kernels
- **Solution**: Enable stack traces in PyTorch profiler (`torch_profiler_with_stack=True`)

**Issue**: Distributed tracing spans disconnected
- **Solution**: Ensure trace context propagation in HTTP headers (`traceparent`)

---

## 6. Summary

### 6.1 Framework Recommendations

**Choose vLLM for**:
- Comprehensive observability requirements
- OpenTelemetry integration
- Advanced profiling (MFU, cache residency)
- Well-documented profiling workflows

**Choose SGLang for**:
- Crash debugging and request replay
- AMD hardware support (RPD profiler)
- HTTP API-driven profiling
- Function-level timing

**Choose TensorRT-LLM for**:
- Simple timer-based profiling needs
- NVTX + Nsight workflows
- Minimal overhead production deployment
- C++ kernel optimization

### 6.2 Key Takeaways

1. **vLLM has the most comprehensive observability stack**: PyTorch profiler, OpenTelemetry, Prometheus, NVTX, GC debugging
2. **SGLang excels at debugging workflows**: Crash dumps, request replay, function timing
3. **TensorRT-LLM focuses on low-overhead profiling**: Timer-based measurement, NVTX for Nsight
4. **All frameworks support CUDA profiling via Nsight Systems**
5. **OpenTelemetry is only available in vLLM and SGLang**
6. **Metrics export is mature in vLLM and SGLang, limited in TensorRT-LLM**

### 6.3 File Reference Summary

**vLLM**:
- `vllm/config/profiler.py`: Profiler configuration (148 lines)
- `vllm/config/observability.py`: Observability settings (153 lines)
- `vllm/logger.py`: Logging system (318 lines)
- `vllm/tracing/otel.py`: OpenTelemetry (200+ lines)
- `vllm/docs/contributing/profiling.md`: Guide (256 lines)

**SGLang**:
- `sglang/profiler.py`: Profiler API (100+ lines)
- `sglang/srt/utils/profile_utils.py`: Profile manager (100+ lines)
- `sglang/srt/tracing/trace.py`: OTEL tracing (200+ lines)
- `sglang/srt/utils/request_logger.py`: Request logging (100+ lines)
- `sglang/3rdparty/amd/profiling/PROFILING.md`: RPD guide (426 lines)

**TensorRT-LLM**:
- `tensorrt_llm/profiler.py`: Timer + memory (200+ lines)
- `tensorrt_llm/logger.py`: Unified logger (150+ lines)
- `tensorrt_llm/_torch/pyexecutor/layerwise_nvtx_marker.py`: NVTX (200+ lines)

---

**Document Version**: 1.0
**Last Updated**: 2026-03-29
