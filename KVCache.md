# KV Cache Implementation Analysis: vLLM, SGLang, and TensorRT-LLM

## Table of Contents
1. [Overview](#overview)
2. [vLLM KV Cache](#1-vllm-kv-cache-implementation)
3. [SGLang KV Cache](#2-sglang-kv-cache-implementation)
4. [TensorRT-LLM KV Cache](#3-tensorrt-llm-kv-cache-implementation)
5. [Performance Comparison](#4-performance-optimization-comparisons)
6. [Pros & Cons Analysis](#5-pros--cons-analysis)
7. [Improvement Opportunities](#6-improvement-opportunities)
8. [Summary](#7-summary)

---

## Overview

KV (Key-Value) cache is critical for transformer inference efficiency:
- **Memory consumption**: Dominates memory usage in long-context scenarios
- **Sharing**: Prefix caching enables cross-request KV reuse
- **Quantization**: Reduces memory at cost of accuracy
- **Offloading**: CPU/disk storage for inactive sequences

This document analyzes KV cache architectures across three frameworks with detailed code references.

---

## 1. vLLM KV Cache Implementation

### 1.1 Paged Architecture

**Core Concept**: Fixed-size blocks (like OS virtual memory paging)

**Location**: `vllm/v1/core/kv_cache_manager.py` (514 lines)

**Memory Layout**:
```python
# K cache: [num_blocks, num_kv_heads, head_size/x, block_size, x]
# V cache: [num_blocks, num_kv_heads, head_size, block_size]

# Block size: 16 tokens (default), configurable to 32/64/128
# Advantages:
# 1. Eliminates memory fragmentation
# 2. Enables efficient prefix sharing
# 3. Supports non-contiguous allocation
```

### 1.2 Block Pool Management

**File**: `vllm/v1/core/block_pool.py` (400+ lines)

**FreeKVCacheBlockQueue** (Lines 158-366):
```python
class FreeKVCacheBlockQueue:
    """Doubly-linked list for O(1) LRU eviction"""

    # Operations:
    # - popleft(): O(1) allocation
    # - append(): O(1) return to free pool
    # - remove(): O(1) specific block eviction

    # Eviction priority:
    # 1. Least recently used block first
    # 2. Tail of block chain for tie-breaking
```

**BlockHashToBlockMap** (Lines 33-126):
```python
class BlockHashToBlockMap:
    """Hash table: block_hash → KVCacheBlock(s)"""

    # Design: Union type to reduce GC overhead
    # - Single block: Direct KVCacheBlock
    # - Multiple blocks: dict[block_id, KVCacheBlock]

    def get_one_block(self, key) -> KVCacheBlock | None:
        # O(1) lookup for prefix caching
```

### 1.3 Prefix Caching Integration

**Hash Computation** (`vllm/v1/core/kv_cache_utils.py:532-559`):
```python
def hash_block_tokens(
    hash_function,
    parent_block_hash,  # Chained hashing for position-awareness
    curr_block_token_ids,
    extra_keys  # LoRA, multimodal, cache salt
) -> BlockHash:
    """Incremental hash for efficient prefix matching"""
```

**Supported Algorithms**:
- SHA256 (default, secure)
- SHA256_CBOR (reproducible, cross-language)
- xxHash (faster, non-cryptographic)

### 1.4 CPU Offloading

**Location**: `vllm/v1/kv_offload/`

**Features**:
- Asynchronous KV transfer to CPU for inactive sequences
- Prefetch mechanism for reactivation
- Trade-off: CPU bandwidth vs GPU memory savings

**Implementation**:
```python
# Offload inactive sequence KV cache to CPU
# Reduces GPU memory pressure
# ~100-200ms latency to reload on reactivation
```

### 1.5 Quantization Support

**File**: `vllm/docs/features/quantization/quantized_kvcache.md`

**FP8 Quantization**:
```python
# Per-tensor scaling
scale = 1.0  # Default, or computed from calibration

# Per-attention-head scaling
k_scale: Tensor[num_kv_heads]
v_scale: Tensor[num_kv_heads]

# Calibration options:
# 1. No calibration (default scale)
# 2. Random token on-the-fly calibration
# 3. Dataset-based calibration (llm-compressor)
```

**Supported Formats**:
- `fp8_e4m3`: 1 sign + 4 exponent + 3 mantissa (±448 range)
- `fp8_e5m2`: 1 sign + 5 exponent + 2 mantissa (±57344 range)

### 1.6 Code Locations

| Component | File | Lines |
|-----------|------|-------|
| KV Cache Manager | `v1/core/kv_cache_manager.py` | 514 |
| Block Pool | `v1/core/block_pool.py` | 400+ |
| Utils | `v1/core/kv_cache_utils.py` | 1689 |
| Coordinator | `v1/core/kv_cache_coordinator.py` | 200+ |
| Encoder Cache | `v1/core/encoder_cache_manager.py` | 200+ |

---

## 2. SGLang KV Cache Implementation

### 2.1 Radix Tree Architecture

**Location**: `sglang/srt/mem_cache/radix_cache.py`

**TreeNode Structure** (Lines 97-158):
```python
class TreeNode:
    children = defaultdict(TreeNode)  # Token/page → child
    parent: TreeNode  # Bottom-up traversal
    key: RadixKey  # Token sequence
    value: torch.Tensor  # KV cache data

    # Eviction metadata
    last_access_time: float  # For LRU
    creation_time: float  # For FIFO
    hit_count: int  # For LFU

    # Host offloading
    host_value: torch.Tensor  # CPU-offloaded cache
    hash_value: List[str]  # SHA256 per page
```

**Radix Tree Benefits**:
- Automatic prefix compression
- O(m) lookup where m = matched length
- Copy-on-write semantics for sharing

### 2.2 Hybrid GPU/CPU Caching (HiCache)

**File**: `sglang/srt/mem_cache/hiradix_cache.py`

**Architecture**:
```
GPU Memory (Hot) ←→ CPU Memory (Cold)
     ↓                     ↓
Radix Tree Nodes    Radix Tree Nodes
     ↓                     ↓
KV Cache Pool       KV Cache Host Pool
                    (mmap/file backend)
```

**Features**:
- Direct I/O backend for async transfers
- NUMA-aware memory binding
- Prefetch operations
- Page-first memory layout optimization

### 2.3 Token-to-KV Pool Allocators

**Location**: `sglang/srt/mem_cache/`

**Multiple Implementations**:
```python
# MHATokenToKVPool (Multi-Head Attention)
# - Layout: [num_tokens, num_kv_heads, head_dim]
# - Optimized for standard attention

# MLATokenToKVPool (Multi-Head Latent Attention)  
# - Compressed representations
# - Lower memory footprint

# NSATokenToKVPool (Specialized Attention)
# - Custom attention patterns
# - Layer-specific optimizations
```

### 2.4 KV Cache Quantization

**File**: `sglang/srt/layers/quantization/kv_cache.py`

**Per-Token Quantization**:
```python
# K-cache quantization kernels
# - nsa/quant_k_cache.py: Quantize on store
# - nsa/dequant_k_cache.py: Dequantize on load

# Strategy: Per-token scales for flexibility
# Reduces memory while maintaining accuracy
```

### 2.5 Code Locations

| Component | File |
|-----------|------|
| Radix Cache | `srt/mem_cache/radix_cache.py` |
| HiCache | `srt/mem_cache/hiradix_cache.py` |
| Memory Pool | `srt/mem_cache/memory_pool.py` |
| Host Pool | `srt/mem_cache/memory_pool_host.py` |
| Quantization | `srt/layers/quantization/kv_cache.py` |
| Store Kernel | `jit_kernel/kvcache.py` |

---

## 3. TensorRT-LLM KV Cache Implementation

### 3.1 Block-Based Management

**File**: `tensorrt_llm/runtime/kv_cache_manager.py`

**Block Class** (Lines 21-38):
```python
class Block:
    idx: int  # Physical block index
    ref_count: int  # Reference counter for sharing

    def add_link(self):
        self.ref_count += 1  # Used by another sequence

    def remove_link(self):
        self.ref_count -= 1

    def is_shared(self) -> bool:
        return self.ref_count > 1  # Multiple sequences
```

**BlocksManager** (Lines 66-150):
```python
class BlocksManager:
    # Structure
    free_blocks: List[Block]
    allocated_blocks: dict[owner][beam_idx] = [[Block, ...]]

    def allocate(self, owner, share_across_beam=False):
        # FIFO allocation from free pool
        # Optional sharing across beam width

    def replace_shared_block(self, owner, block_idx):
        # Copy-on-write for beam search branching
```

### 3.2 Memory Layout

**Homogeneous Layers**:
```cpp
// Shape: [num_blocks, num_layers, 2, block_size]
// - 2 dimensions: K and V
// - All layers share same structure
```

**Heterogeneous Layers**:
```cpp
// Shape: [num_blocks, 2, block_size]
// - More flexible for variable layer configs
// - Slightly slower indexing
```

### 3.3 Paged Attention Kernel

**File**: `tensorrt_llm/cpp/kernels/fmha_v2/src/fmha/paged_kv_cache.h`

```cpp
struct Kv_block_array {
    int32_t mMaxSeqs;          // Batch size
    int32_t mMaxBlocksPerSeq;  // Max blocks per request
    int32_t mTokensPerBlock;   // Power of 2 for fast modulo
    int32_t mTokensPerBlockLog2;  // log2(block_size)
    int32_t mBytesPerBlock;    // Memory per block
    void* mPoolPtr;            // Start of block pool
    int32_t* mBlockOffsets;    // Logical → physical mapping
};
```

**Optimization**: Power-of-2 block size eliminates modulo operation

### 3.4 Beam Search Support

**Copy-on-Write Semantics**:
```python
# Before update: Seq→Block0(shared, ref_cnt=2)
blocks_mgr.replace_shared_block(owner=seq, block_idx=0)
# After update:
#   Seq1→Block0(shared, ref_cnt=1)  
#   Seq2→Block8(unique, ref_cnt=1)
```

### 3.5 Code Locations

| Component | File |
|-----------|------|
| KV Manager | `runtime/kv_cache_manager.py` |
| Paged KV | `cpp/kernels/fmha_v2/src/fmha/paged_kv_cache.h` |
| KV Type | `llmapi/kv_cache_type.py` |
| Memory Pools | `runtime/memory_pools/pools_kv_cache_manager.py` |

---

## 4. Performance Optimization Comparisons

### 4.1 Memory Efficiency

| Framework | Strategy | Compression | CPU Offload | Quantization |
|-----------|----------|-------------|-------------|--------------|
| **vLLM** | Paged + prefix cache | Hash dedup | Yes (v1) | FP8 per-tensor/head |
| **SGLang** | Radix tree hybrid | Radix CoW | Yes (HiCache) | Per-token K-cache |
| **TensorRT-LLM** | Paged + beam CoW | Beam sharing | Limited | INT8/FP8 |

### 4.2 Cache Locality

**vLLM**:
- Memory coalescing in paged attention
- Thread-group access patterns
- Warp-level reductions

**SGLang**:
- Radix tree prefetching
- NUMA-aware binding
- Direct I/O for CPU offload

**TensorRT-LLM**:
- Block offset table for fast indexing
- Power-of-2 block size
- Shared blocks within beam width

### 4.3 Kernel Fusion

**vLLM**:
- Reshape-and-cache kernels
- Fused RoPE + KV cache (MLA)
- CUDA graph integration

**SGLang**:
- Store cache kernel with streams
- Triton-based operations
- CUDA graph compatible

**TensorRT-LLM**:
- FMHA v2 with paged attention
- Optimized scatter/gather

---

## 5. Pros & Cons Analysis

### 5.1 vLLM

**Pros**:
- ✅ True paging with automatic prefix caching
- ✅ Sophisticated hybrid KV cache (multi-attention)
- ✅ FP8 with per-head granularity
- ✅ Encoder output caching (multimodal)
- ✅ Comprehensive block management (LRU, ref counting)

**Cons**:
- ❌ Complexity in multi-group scheduling
- ❌ Hash computation overhead
- ❌ Block recomputation for misaligned requests

### 5.2 SGLang

**Pros**:
- ✅ Unified radix tree for prefix matching
- ✅ Native hybrid GPU/CPU (HiCache)
- ✅ Fine-grained copy-on-write
- ✅ Async prefetch mechanism
- ✅ NUMA-aware allocation

**Cons**:
- ❌ Complex radix tree operations
- ❌ CPU storage adds latency
- ❌ Prefetch timing critical

### 5.3 TensorRT-LLM

**Pros**:
- ✅ Simple block offset table
- ✅ Efficient beam search with sharing
- ✅ Low management overhead
- ✅ Tight FMHA integration

**Cons**:
- ❌ No prefix caching
- ❌ No CPU offloading
- ❌ Limited quantization flexibility

---

## 6. Improvement Opportunities

### 6.1 For vLLM

**Reduce Recomputation Overhead**:
```python
# Current: Always recompute last token
# Opportunity: Support partial block caching
# - Use rotary embeddings to enable last-token-only recompute
# - Save computation for long cached prefixes
```

**Hybrid Memory Tiers**:
```python
# Add disk-based cache for very inactive sequences
# Async prefetch like SGLang's HiCache
# Priority-based eviction (keep high-priority in GPU)
```

**Quantization Enhancements**:
```python
# Dynamic quantization scale adaptation
# Per-layer quantization granularity
# Mixed precision KV cache (K in FP8, V in FP16)
```

### 6.2 For SGLang

**Cross-Request Prefix Deduplication**:
```python
# Radix tree already supports prefix sharing
# Add hash-based duplicate detection across requests
# Reduces tree size and memory usage
```

**CPU Storage Optimization**:
```python
# Predictive prefetching based on access patterns
# Compression for CPU-side storage (LZ4/Snappy)
# Intelligent page swapping
```

### 6.3 For TensorRT-LLM

**Prefix Caching**:
```cpp
// Add hash-based block caching
// Integrate with existing block pool
// Similar to vLLM's approach
```

**CPU Offload**:
```cpp
// Implement CPU offload for completed prefixes
// Pipelined GPU/CPU compute
// RDMA support for remote KV cache
```

---

## 7. Summary

### 7.1 Key Takeaways

1. **vLLM**: Most comprehensive KV cache with paged architecture + prefix caching + CPU offload
2. **SGLang**: Radix tree approach excellent for long common prefixes, native GPU/CPU hybrid
3. **TensorRT-LLM**: Simple and efficient for beam search, minimal overhead, no prefix caching

### 7.2 Recommendation by Use Case

**Use vLLM when**:
- Need automatic prefix caching
- Multi-turn conversations with common context
- LoRA/multimodal workloads requiring isolation

**Use SGLang when**:
- Very long common prefixes (large system prompts)
- Custom eviction policies needed
- CPU offloading required

**Use TensorRT-LLM when**:
- Beam search is primary use case
- Explicit control over sharing preferred
- Minimal overhead critical

---

**Document Version**: 1.0
**Last Updated**: 2026-03-29
