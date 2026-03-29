# Prefix Caching Implementation Analysis: vLLM, SGLang, and TensorRT-LLM

## Table of Contents
1. [Overview](#overview)
2. [vLLM Implementation](#1-vllm-prefix-caching-implementation)
3. [SGLang Implementation](#2-sglang-radix-cache-implementation)
4. [TensorRT-LLM Implementation](#3-tensorrt-llm-block-management)
5. [Comparative Analysis](#4-comparative-analysis)
6. [Configuration Examples](#5-configuration-examples)
7. [Integration with Scheduler](#6-integration-with-scheduler)
8. [Key Insights and Optimizations](#7-key-insights-and-optimizations)
9. [Summary](#8-summary)

---

## Overview

This document provides a comprehensive analysis of prefix caching implementations across three major LLM serving frameworks: vLLM, SGLang, and TensorRT-LLM. Prefix caching (also known as KV cache sharing) is a crucial optimization that reuses key-value cache for common prompt prefixes across different requests, dramatically reducing computational costs and improving latency.

**Key Benefits of Prefix Caching:**
- Reduced computation: Avoid reprocessing common prompt prefixes
- Lower latency: TTFT (Time To First Token) improvement for cached prefixes
- Higher throughput: More requests served with same hardware
- Memory efficiency: Share KV cache blocks across multiple requests

---

## 1. vLLM Prefix Caching Implementation

### 1.1 Core Implementation Files

| File | Lines | Purpose |
|------|-------|---------|
| `vllm/v1/core/kv_cache_manager.py` | 514 | Main KV cache manager orchestrating prefix caching |
| `vllm/v1/core/kv_cache_coordinator.py` | ~600 | Coordinates cache lookups across different attention types |
| `vllm/v1/core/single_type_kv_cache_manager.py` | ~1200 | Implements cache lookup for specific attention types |
| `vllm/v1/core/block_pool.py` | ~500 | Block allocation and caching with hash-based lookup |
| `vllm/v1/core/kv_cache_utils.py` | 1800+ | Hashing, block structures, and utility functions |
| `vllm/config/cache.py` | 219 | Cache configuration parameters |

### 1.2 Key Algorithms

#### A. Hash-Based Block Lookup

**Location**: `vllm/v1/core/kv_cache_utils.py:532-559`

```python
def hash_block_tokens(
    hash_function: Callable[[Any], bytes],
    parent_block_hash: BlockHash | None,
    curr_block_token_ids: Sequence[int],
    extra_keys: tuple[Any, ...] | None = None,
) -> BlockHash:
    """Computes hash value for a KV-cache block with parent chaining.

    This enables position-aware prefix matching by including the parent
    block's hash in the current block's hash computation.
    """
    if not parent_block_hash:
        parent_block_hash = NONE_HASH
    curr_block_token_ids_tuple = tuple(curr_block_token_ids)
    return BlockHash(
        hash_function((parent_block_hash, curr_block_token_ids_tuple, extra_keys))
    )
```

**Key Features:**
- **Chained hashing**: Each block hash includes the parent block's hash
- **Position-aware**: Same tokens at different positions have different hashes
- **Deterministic**: Same token sequence always produces same hash
- **Extra keys support**: LoRA adapters, multimodal inputs, cache salts can be included
- **Multiple algorithms**: SHA256, SHA256_CBOR, xxHash supported

**Algorithm Flow:**
1. Start with parent block hash (or NONE_HASH for first block)
2. Combine parent_hash + current_tokens + extra_keys
3. Hash the tuple to create unique block identifier
4. This hash becomes parent_hash for next block

#### B. Longest Cache Hit Finding

**Location**: `vllm/v1/core/single_type_kv_cache_manager.py:421-460`

```python
@classmethod
def find_longest_cache_hit(
    cls,
    block_hashes: BlockHashList,
    max_length: int,
    kv_cache_group_ids: list[int],
    block_pool: BlockPool,
    kv_cache_spec: KVCacheSpec,
    use_eagle: bool,
    alignment_tokens: int,
    dcp_world_size: int = 1,
    pcp_world_size: int = 1,
) -> tuple[list[KVCacheBlock], ...]:
    """Find longest cache hit prefix by sequential block lookup.

    Returns:
        Tuple of cached block lists (one per KV cache group)
    """
    computed_blocks: tuple[list[KVCacheBlock], ...] = tuple(
        [] for _ in range(len(kv_cache_group_ids))
    )
    block_size = kv_cache_spec.block_size
    max_num_blocks = max_length // block_size

    # Sequential scan through blocks
    for block_hash in itertools.islice(block_hashes, max_num_blocks):
        if cached_block := block_pool.get_cached_block(
            block_hash, kv_cache_group_ids
        ):
            for computed, cached in zip(computed_blocks, cached_block):
                computed.append(cached)
        else:
            break  # Stop at first cache miss

    # EAGLE speculative decoding: drop last block
    if use_eagle and computed_blocks[0]:
        for computed in computed_blocks:
            computed.pop()

    return computed_blocks
```

**Algorithm Characteristics:**
- **Linear scan**: O(n) where n = number of blocks matched
- **Early termination**: Stops at first cache miss
- **Cache locality**: Sequential access pattern is CPU cache-friendly
- **Group support**: Handles multiple KV cache groups (for distributed settings)
- **EAGLE integration**: Removes last block for speculative decoding requirements

#### C. Hybrid Attention Cache Hit (Iterative Fixed-Point)

**Location**: `vllm/v1/core/kv_cache_coordinator.py:453-498`

For models with multiple attention types (e.g., hybrid full + sliding window attention), vLLM uses an iterative convergence algorithm:

```python
def find_longest_cache_hit(
    self,
    block_hashes: list[BlockHash],
    max_cache_hit_length: int,
) -> tuple[tuple[list[KVCacheBlock], ...], int]:
    """Iterative fixed-point algorithm for hybrid attention.

    Ensures cache hit respects block size requirements of all attention types.
    Converges because hit length monotonically decreases.
    """
    hit_length = max_cache_hit_length
    hit_blocks_by_group = [None] * num_groups

    while True:
        curr_hit_length = hit_length

        # Check each attention type
        for attn_type, manager in self._managers.items():
            blocks, length = manager.find_longest_cache_hit(
                block_hashes, curr_hit_length
            )

            if length < hit_length:
                hit_length = length  # Reduce to satisfy this constraint
                break

        # Converged when no manager reduces the length
        if hit_length == curr_hit_length:
            break

    return hit_blocks_by_group, hit_length
```

**Why Needed:**
- Different attention mechanisms may have different block size constraints
- Full attention might use 16-token blocks
- Sliding window attention might use different granularity
- Algorithm finds largest common prefix respecting all constraints

### 1.3 Core Data Structures

#### BlockHash and BlockHashWithGroupId

**Location**: `vllm/v1/core/kv_cache_utils.py:33-68`

```python
BlockHash = NewType("BlockHash", bytes)
BlockHashWithGroupId = NewType("BlockHashWithGroupId", bytes)

def make_block_hash_with_group_id(
    block_hash: BlockHash, group_id: int
) -> BlockHashWithGroupId:
    """Pack hash and group_id into single key.

    Format: [hash_bytes (variable length) | group_id (4 bytes)]
    """
    return BlockHashWithGroupId(
        block_hash + group_id.to_bytes(4, "big", signed=False)
    )

def get_block_hash(key: BlockHashWithGroupId) -> BlockHash:
    """Extract hash from key (last 4 bytes are group_id)."""
    return BlockHash(key[:-4])
```

**Design Rationale:**
- Single key type reduces dictionary overhead
- Group ID enables distributed KV cache (tensor parallelism)
- Big-endian encoding for cross-platform consistency

#### KVCacheBlock

**Location**: `vllm/v1/core/kv_cache_utils.py:109-156`

```python
@dataclass(slots=True)
class KVCacheBlock:
    """Represents a single block in KV cache."""

    block_id: int                                # Physical block index
    ref_cnt: int = 0                             # Reference count for sharing
    _block_hash: BlockHashWithGroupId | None = None  # Hash when cached
    prev_free_block: "KVCacheBlock | None" = None    # LRU linked list
    next_free_block: "KVCacheBlock | None" = None
    is_null: bool = False                        # Placeholder block

    def incr_ref(self) -> None:
        """Increment reference count (block is shared)."""
        self.ref_cnt += 1

    def decr_ref(self) -> None:
        """Decrement reference count."""
        assert self.ref_cnt > 0
        self.ref_cnt -= 1

    @property
    def is_cached(self) -> bool:
        """Block is in hash table and can be reused."""
        return self._block_hash is not None
```

**Key Properties:**
- **Reference counting**: Supports shared blocks across requests
- **Doubly-linked list**: For LRU eviction policy
- **Hash storage**: Block hash stored only when fully populated
- **Null blocks**: Placeholder for lazy allocation
- **Slots**: Memory optimization via `__slots__`

#### BlockHashToBlockMap

**Location**: `vllm/v1/core/block_pool.py:33-127`

```python
class BlockHashToBlockMap:
    """Hash table mapping block_hash -> KVCacheBlock(s).

    Optimized to minimize GC overhead by using union types instead
    of always allocating dict for each entry.
    """

    def __init__(self):
        self._cache: dict[
            BlockHashWithGroupId,
            KVCacheBlock | dict[int, KVCacheBlock]
        ] = {}

    def get_one_block(
        self, key: BlockHashWithGroupId
    ) -> KVCacheBlock | None:
        """Get single block or first from dict."""
        blocks = self._cache.get(key)
        if isinstance(blocks, KVCacheBlock):
            return blocks
        if isinstance(blocks, dict):
            return next(iter(blocks.values()))
        return None

    def put(
        self,
        key: BlockHashWithGroupId,
        block: KVCacheBlock,
    ) -> None:
        """Insert block into cache."""
        existing = self._cache.get(key)

        if existing is None:
            self._cache[key] = block
        elif isinstance(existing, KVCacheBlock):
            # Promote to dict
            self._cache[key] = {existing.block_id: existing, block.block_id: block}
        else:
            # Already a dict
            existing[block.block_id] = block
```

**Design Optimizations:**
- **Union type**: Single block or dict reduces allocations
- **Lazy promotion**: Only allocate dict when multiple blocks share same hash
- **O(1) lookup**: Hash table provides constant-time access
- **Block ID indexing**: Supports multiple physical blocks with same content

#### FreeKVCacheBlockQueue

**Location**: `vllm/v1/core/kv_cache_utils.py:158-367`

```python
class FreeKVCacheBlockQueue:
    """Doubly-linked list for LRU eviction.

    Eviction priority:
    1. Least recently used block first
    2. If same access time, tail of block chain first
    """

    def __init__(self, blocks: list[KVCacheBlock]):
        self.fake_free_list_head = KVCacheBlock(block_id=-1)
        self.fake_free_list_tail = KVCacheBlock(block_id=-1)

        # Initialize as doubly-linked list
        self.fake_free_list_head.next_free_block = blocks[0] if blocks else self.fake_free_list_tail
        for i in range(len(blocks)):
            blocks[i].prev_free_block = blocks[i-1] if i > 0 else self.fake_free_list_head
            blocks[i].next_free_block = blocks[i+1] if i < len(blocks)-1 else self.fake_free_list_tail
        self.fake_free_list_tail.prev_free_block = blocks[-1] if blocks else self.fake_free_list_head

    def popleft(self) -> KVCacheBlock:
        """O(1) allocation from head."""
        block = self.fake_free_list_head.next_free_block
        self._remove(block)
        return block

    def append(self, block: KVCacheBlock) -> None:
        """O(1) return to tail for eviction."""
        self._insert_before(self.fake_free_list_tail, block)

    def remove(self, block: KVCacheBlock) -> None:
        """O(1) removal from middle."""
        self._remove(block)
```

**Operations:**
- **popleft()**: O(1) - Allocate least recently used block
- **append()**: O(1) - Return block to eviction queue
- **remove()**: O(1) - Remove specific block (when referenced)
- **Sentinel nodes**: Simplify edge cases with fake head/tail

### 1.4 Configuration Options

**Location**: `vllm/config/cache.py:26-84`

```python
@dataclass
class CacheConfig:
    """Configuration for KV cache and prefix caching."""

    enable_prefix_caching: bool = True
    """Whether to enable prefix caching."""

    prefix_caching_hash_algo: PrefixCachingHashAlgo = "sha256"
    """Hash algorithm for prefix caching.

    Options:
    - "sha256": Cryptographic hash, default, most secure
    - "sha256_cbor": Reproducible CBOR serialization + SHA256
    - "xxhash": Faster non-cryptographic hash (requires xxhash package)
    - "xxhash_cbor": CBOR + xxHash for speed and reproducibility
    """

    block_size: int = 16
    """Number of tokens per KV cache block."""

    gpu_memory_utilization: float = 0.9
    """Fraction of GPU memory for KV cache."""

    cache_dtype: str = "auto"
    """Data type for KV cache (auto, fp16, fp8, etc.)."""
```

**User Example**: `vllm/examples/offline_inference/prefix_caching.py`

```python
from vllm import LLM, SamplingParams

# Enable prefix caching with default SHA256
llm = LLM(
    model="facebook/opt-125m",
    enable_prefix_caching=True,
    gpu_memory_utilization=0.4,
)

# Common prefix
prefix = "You are a helpful assistant. "
prompts = [prefix + "What is 2+2?", prefix + "What is the sky?"]

# Automatic cache sharing
outputs = llm.generate(prompts, SamplingParams(temperature=0.0))
```

### 1.5 Integration with Scheduler

**Location**: `vllm/v1/core/kv_cache_manager.py:176-216`

```python
def get_computed_blocks(self, request: Request) -> tuple[KVCacheBlocks, int]:
    """Get cached blocks for request using prefix matching.

    Returns:
        (cached_blocks, num_tokens_hit)
    """

    # Skip cache if disabled or request opts out
    if not self.enable_caching or request.skip_reading_prefix_cache:
        return self.empty_kv_cache_blocks, 0

    # Calculate maximum possible cache hit
    max_cache_hit_length = (
        request.num_tokens - request.num_computed_tokens
    )

    # Find longest cache hit
    computed_blocks, num_new_computed_tokens = (
        self.coordinator.find_longest_cache_hit(
            request.block_hashes, max_cache_hit_length
        )
    )

    # Record cache hit statistics
    if self.log_stats:
        self.prefix_cache_stats.record(
            num_tokens=request.num_tokens,
            num_hits=num_new_computed_tokens,
            preempted=request.num_preemptions > 0,
        )

    return self.create_kv_cache_blocks(computed_blocks), num_new_computed_tokens
```

**Statistics Tracking** (`PrefixCacheStats`):
- Total queries to prefix cache
- Total cache hits
- Cache hit rate calculation
- Per-request hit metrics
- Preemption impact tracking

### 1.6 Performance Characteristics

| Aspect | Implementation Detail |
|--------|----------------------|
| **Hash Computation** | O(1) per block via incremental chaining |
| **Lookup Performance** | O(1) per block via hash table |
| **Cache Hit Finding** | O(k) where k = blocks matched |
| **Eviction Policy** | LRU with O(1) operations |
| **Block Granularity** | 16 tokens default (configurable) |
| **Memory Overhead** | ~32 bytes hash + 48 bytes metadata per block |
| **Reference Counting** | Supports unlimited sharing |

---

## 2. SGLang Radix Cache Implementation

### 2.1 Core Implementation Files

| File | Purpose |
|------|---------|
| `sglang/srt/mem_cache/radix_cache.py` | Main radix tree implementation |
| `sglang/srt/mem_cache/base_prefix_cache.py` | Abstract interface and data structures |
| `sglang/srt/mem_cache/evict_policy.py` | Eviction strategies (LRU, LFU, FIFO, etc.) |
| `sglang/srt/mem_cache/hiradix_cache.py` | Hybrid in-memory/host cache |
| `sglang/srt/mem_cache/hicache_storage.py` | Storage with SHA256-based hashing |

### 2.2 Key Algorithms

#### A. Radix Tree Structure

**Location**: `sglang/srt/mem_cache/radix_cache.py:97-158`

```python
class TreeNode:
    """Node in the radix tree for prefix caching.

    Unlike a trie, radix tree compresses common prefixes into single nodes.
    """

    def __init__(self, id: Optional[int] = None, priority: int = 0):
        self.children = defaultdict(TreeNode)  # Child nodes by token/page
        self.parent: TreeNode = None           # Parent pointer
        self.key: RadixKey = None              # Token sequence for this node
        self.value: Optional[torch.Tensor] = None  # KV cache tensor

        # Eviction metadata
        self.lock_ref = 0                      # Lock reference count
        self.last_access_time = time.monotonic()  # For LRU
        self.creation_time = time.monotonic()     # For FIFO
        self.hit_count = 0                     # For LFU

        # Host cache offloading
        self.host_ref_counter = 0              # Host value protection
        self.host_value: Optional[torch.Tensor] = None  # CPU offloaded cache

        # Position-aware hashing
        self.hash_value: Optional[List[str]] = None  # SHA256 hashes per page

        # Priority-based eviction
        self.priority = priority
```

**Data Structure Properties:**
- **Radix tree** (not trie): Compresses common prefixes into single nodes
- **Children dictionary**: Fast O(1) child lookup by token/page ID
- **Parent pointers**: Enables bottom-up traversal for eviction
- **Direct tensor storage**: KV cache stored in tree nodes
- **Metadata-rich**: Supports multiple eviction policies

#### B. Prefix Matching

**Location**: `sglang/srt/mem_cache/radix_cache.py:352-422`

```python
def match_prefix(self, params: MatchPrefixParams) -> MatchResult:
    """Find longest cached prefix in radix tree.

    Traverses tree from root, matching tokens/pages until mismatch or end.
    """
    key = params.key  # RadixKey with token_ids and optional extra_key

    if self.disable or len(key) == 0:
        return empty_match_result()

    # Page alignment for paged KV cache
    if self.page_size != 1:
        page_aligned_len = len(key) // self.page_size * self.page_size
        key = key[:page_aligned_len]

    # Traverse tree from root
    value, last_node = self._match_prefix_helper(self.root_node, key)

    if value:
        value = torch.cat(value)  # Concatenate matched KV tensors
    else:
        value = torch.empty((0,), dtype=torch.int64, device=self.device)

    return MatchResult(
        device_indices=value,
        last_device_node=last_node,
        last_host_node=last_node,
    )

def _match_prefix_helper(
    self, node: TreeNode, key: RadixKey
) -> tuple[list, TreeNode]:
    """Recursive helper for prefix matching."""
    if len(key) == 0:
        return [], node

    # Try to find child matching first token/page
    first_token_or_page = self._get_first_token_or_page(key)
    if first_token_or_page not in node.children:
        return [], node  # No match

    child = node.children[first_token_or_page]

    # Match as many tokens as possible with this child
    match_len = self.key_match_fn(key, child.key)

    if match_len < len(child.key):
        # Partial match - stop here
        return [], node

    # Full match - continue to children
    if child.value is not None:
        values = [child.value]
    else:
        values = []

    # Recurse with remaining key
    child_values, last_node = self._match_prefix_helper(
        child, key[match_len:]
    )

    return values + child_values, last_node
```

**Algorithm Characteristics:**
- **Tree traversal**: O(m) where m = matched prefix length
- **Greedy matching**: Matches longest prefix at each level
- **Page-aware**: Aligns to page boundaries for paged KV cache
- **Tensor accumulation**: Collects KV tensors along matched path

#### C. Node Splitting for Position-Aware Hashing

**Location**: `sglang/srt/mem_cache/radix_cache.py:201-258`

```python
def compute_node_hash_values(
    node: "TreeNode", page_size: int
) -> List[str]:
    """Compute SHA256 hash values with position awareness.

    Hashing is position-aware via parent hash chaining.
    Each page's hash includes the previous page's hash.
    """
    hash_values = []

    # Get parent's last hash if parent exists
    parent_hash = None
    if node.parent is not None and node.parent.hash_value is not None:
        if len(node.parent.key) > 0 and len(node.parent.hash_value) > 0:
            parent_hash = node.parent.hash_value[-1]

    # Iterate through node's pages
    for start in range(0, len(node.key), page_size):
        page_tokens = node.key.token_ids[start : start + page_size]
        if not page_tokens:
            continue

        # SHA256-based chaining
        hash_val = get_hash_str(page_tokens, prior_hash=parent_hash)
        hash_values.append(hash_val)
        parent_hash = hash_val  # Chain to next page

    return hash_values

def split_node_hash_value(
    child_hash_value: Optional[List[str]],
    split_len: int,
    page_size: int
) -> tuple[Optional[List[str]], Optional[List[str]]]:
    """Split hash values when splitting node during insertion.

    When a node is split (partial match during insert), hash values
    must also be split to maintain position-aware chaining.
    """
    split_pages = split_len // page_size if page_size > 1 else split_len
    new_node_hash = child_hash_value[:split_pages]
    child_hash = child_hash_value[split_pages:]
    return new_node_hash, child_hash
```

**Why Position-Aware Hashing:**
- Same tokens at different positions should have different hashes
- Enables cache reuse verification across different contexts
- Supports distributed cache with consistent hashing
- HiCache (host offloading) uses hashes for cache lookup

#### D. Tree Key Matching

**Location**: `sglang/srt/mem_cache/radix_cache.py:167-188`

```python
def _key_match_page_size1(key0: RadixKey, key1: RadixKey):
    """Token-by-token matching for page_size=1."""
    i = 0
    for k0, k1 in zip(key0.token_ids, key1.token_ids):
        if k0 != k1:
            break
        i += 1
    return i

def _key_match_paged(key0: RadixKey, key1: RadixKey, page_size: int):
    """Page-by-page matching for larger pages.

    Compares entire pages at once for efficiency.
    """
    min_len = min(len(key0), len(key1))
    i = 0
    while i < min_len:
        # Compare page slices
        if key0.token_ids[i : i + page_size] != key1.token_ids[i : i + page_size]:
            break
        i += page_size
    return i
```

**Optimization:**
- Different matching functions for different page sizes
- Page-level matching reduces comparisons
- Selected at initialization based on configuration

### 2.3 Core Data Structures

#### RadixKey

**Location**: `sglang/srt/mem_cache/radix_cache.py:67-95`

```python
class RadixKey:
    """Key for radix cache lookup.

    Encapsulates token sequence and optional metadata for cache namespacing.
    """

    def __init__(
        self,
        token_ids: List[int],
        extra_key: Optional[str] = None,
        is_bigram: bool = False,
    ):
        self.token_ids = token_ids  # Token ID sequence
        self.extra_key = extra_key  # LoRA ID, cache salt, etc.
        self.is_bigram = is_bigram  # For EAGLE speculative decoding

    def __len__(self):
        return len(self.token_ids)

    def __getitem__(self, key):
        if isinstance(key, slice):
            return RadixKey(
                self.token_ids[key],
                self.extra_key,
                self.is_bigram,
            )
        return self.token_ids[key]
```

**Features:**
- Token ID sequence as primary key
- Extra key for namespace isolation (LoRA adapters, different models, etc.)
- Bigram support for EAGLE speculative decoding
- Slice support for recursive matching

#### Base Prefix Cache Interface

**Location**: `sglang/srt/mem_cache/base_prefix_cache.py:35-112`

```python
@dataclasses.dataclass
class MatchPrefixParams:
    """Parameters for prefix matching operation."""
    key: RadixKey
    cow_mamba: bool = False  # Copy-on-write for Mamba models
    req: Optional[Req] = None

@dataclasses.dataclass
class InsertParams:
    """Parameters for cache insertion."""
    key: RadixKey
    value: Optional[torch.Tensor] = None
    mamba_value: Optional[torch.Tensor] = None
    prev_prefix_len: int = 0
    swa_evicted_seqlen: int = 0  # Sliding window attention
    chunked: bool = False
    priority: int = 0

class MatchResult(NamedTuple):
    """Result of prefix matching operation."""
    device_indices: torch.Tensor       # Cached KV indices
    last_device_node: Any              # Terminal tree node
    last_host_node: Any                # For HiCache
    host_hit_length: int = 0           # HiCache hit length
    mamba_branching_seqlen: Optional[int] = None  # Mamba model support

class BasePrefixCache(ABC):
    """Abstract base class for prefix caches."""

    @abstractmethod
    def match_prefix(self, params: MatchPrefixParams) -> MatchResult:
        """Find longest cached prefix."""
        pass

    @abstractmethod
    def insert(self, params: InsertParams) -> None:
        """Insert completed request into cache."""
        pass
```

### 2.4 Eviction Policies

**Location**: `sglang/srt/mem_cache/evict_policy.py`

SGLang supports pluggable eviction strategies:

```python
class EvictionStrategy(ABC):
    """Abstract eviction strategy."""

    @abstractmethod
    def select_evict_node(self, candidates: List[TreeNode]) -> TreeNode:
        """Select node to evict from candidates."""
        pass

class LRUStrategy(EvictionStrategy):
    """Least Recently Used - evict oldest access time."""
    def select_evict_node(self, candidates: List[TreeNode]) -> TreeNode:
        return min(candidates, key=lambda x: x.last_access_time)

class LFUStrategy(EvictionStrategy):
    """Least Frequently Used - evict lowest hit count."""
    def select_evict_node(self, candidates: List[TreeNode]) -> TreeNode:
        return min(candidates, key=lambda x: x.hit_count)

class FIFOStrategy(EvictionStrategy):
    """First In First Out - evict oldest creation time."""
    def select_evict_node(self, candidates: List[TreeNode]) -> TreeNode:
        return min(candidates, key=lambda x: x.creation_time)

class MRUStrategy(EvictionStrategy):
    """Most Recently Used - evict newest access time."""
    def select_evict_node(self, candidates: List[TreeNode]) -> TreeNode:
        return max(candidates, key=lambda x: x.last_access_time)

class FILOStrategy(EvictionStrategy):
    """First In Last Out - evict newest creation time."""
    def select_evict_node(self, candidates: List[TreeNode]) -> TreeNode:
        return max(candidates, key=lambda x: x.creation_time)

class PriorityStrategy(EvictionStrategy):
    """Priority-based - evict lowest priority."""
    def select_evict_node(self, candidates: List[TreeNode]) -> TreeNode:
        return min(candidates, key=lambda x: x.priority)
```

**Usage:**
```python
cache = RadixCache(
    CacheInitParams(eviction_policy="lru")  # or "lfu", "fifo", etc.
)
```

### 2.5 Initialization and Configuration

**Location**: `sglang/srt/mem_cache/radix_cache.py:262-307`

```python
class RadixCache(BasePrefixCache):
    """Radix tree-based prefix cache with pluggable eviction."""

    def __init__(self, params: CacheInitParams):
        self.disable = params.disable
        self.req_to_token_pool = params.req_to_token_pool
        self.token_to_kv_pool_allocator = params.token_to_kv_pool_allocator
        self.page_size = params.page_size
        self.enable_kv_cache_events = params.enable_kv_cache_events
        self.is_eagle = params.is_eagle
        self.disable_finished_insert = params.disable_finished_insert
        self.eviction_policy = params.eviction_policy.lower()

        # Setup key matching function based on page size
        if self.page_size == 1:
            self.key_match_fn = _key_match_page_size1
        else:
            self.key_match_fn = partial(_key_match_paged, page_size=self.page_size)

        # Setup eviction strategy
        if self.eviction_policy == "lru":
            self.eviction_strategy = LRUStrategy()
        elif self.eviction_policy == "lfu":
            self.eviction_strategy = LFUStrategy()
        elif self.eviction_policy == "fifo":
            self.eviction_strategy = FIFOStrategy()
        elif self.eviction_policy == "mru":
            self.eviction_strategy = MRUStrategy()
        elif self.eviction_policy == "filo":
            self.eviction_strategy = FILOStrategy()
        elif self.eviction_policy == "priority":
            self.eviction_strategy = PriorityStrategy()
        else:
            raise ValueError(f"Unknown eviction policy: {self.eviction_policy}")

        # Initialize root node
        self.root_node = TreeNode()
```

### 2.6 Performance Characteristics

| Aspect | Implementation Detail |
|--------|----------------------|
| **Tree Structure** | Radix tree with node splitting |
| **Lookup Time** | O(m) where m = matched prefix length |
| **Node Operations** | O(1) child lookup via dict |
| **Memory Overhead** | Full KV tensors stored in nodes + metadata |
| **Eviction** | Pluggable strategies (6 options) |
| **Position-Aware Hashing** | SHA256 chaining per page |
| **Page Size** | Configurable (1 = token-level, >1 = pages) |
| **Insert Complexity** | O(m) + O(log n) for eviction candidate search |

---

## 3. TensorRT-LLM Block Management

### 3.1 Core Implementation Files

| File | Purpose |
|------|---------|
| `tensorrt_llm/runtime/kv_cache_manager.py` | Main block pool and reference counting |
| `tensorrt_llm/runtime/memory_pools/pools_kv_cache_manager.py` | Multi-pool manager |
| `tensorrt_llm/llmapi/build_cache.py` | Cache initialization |

### 3.2 Block Management Architecture

#### A. Block Class with Reference Counting

**Location**: `tensorrt_llm/runtime/kv_cache_manager.py:21-38`

```python
class Block(object):
    """Single KV cache block with reference counting for sharing."""

    def __init__(self, block_idx: int):
        self.idx = block_idx      # Physical block index in pool
        self.ref_count = 0        # Number of sequences using this block

    def add_link(self):
        """Increment reference count - block is shared by another sequence."""
        self.ref_count += 1

    def remove_link(self):
        """Decrement reference count - sequence no longer uses this block."""
        self.ref_count -= 1

    def has_link(self) -> bool:
        """Check if block is still in use."""
        return self.ref_count > 0

    def is_shared(self) -> bool:
        """Check if block is shared by multiple sequences."""
        return self.ref_count > 1
```

**Design Philosophy:**
- Simple explicit reference counting
- No automatic cache lookup - manual sharing required
- Clear ownership model for beam search
- Minimal metadata overhead

#### B. BlocksManager with Beam Search Support

**Location**: `tensorrt_llm/runtime/kv_cache_manager.py:66-180`

```python
class BlocksManager(object):
    """Manages KV cache blocks with beam search support.

    Supports prefix caching via manual block sharing and copy-on-write
    semantics for beam search branching.
    """

    def __init__(
        self,
        *,
        num_layers: int,
        num_blocks: int,
        block_size: int,
        max_blocks_per_seq: int = 128,
        beam_width: int = 1,
    ):
        self.max_blocks_per_seq = max_blocks_per_seq
        self.num_layers = num_layers
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.beam_width = beam_width

        # Initialize free block pool
        self.free_blocks = []
        for bi in range(num_blocks):
            self.free_blocks.append(Block(bi))

        # Track allocated blocks per sequence
        # allocated_blocks[owner][beam_idx] = [Block, Block, ...]
        self.allocated_blocks = defaultdict(
            lambda: [[] for _ in range(beam_width)]
        )

    def has_free_block(self) -> bool:
        """Check if free blocks available."""
        return len(self.free_blocks) > 0

    def allocate(
        self,
        owner: GenerationSequence,
        share_across_beam: bool = False
    ):
        """Allocate new block for a sequence.

        Args:
            owner: Sequence owning the block
            share_across_beam: If True, same block used for all beams
        """
        block = None
        for bi in range(self.beam_width):
            if not self.has_free_block():
                raise RuntimeError("Can't allocate new block for KV cache")

            # Reuse same block across beams or allocate new
            if block is None or not share_across_beam:
                block = self.free_blocks.pop(0)  # FIFO allocation

            block.add_link()  # Increment reference count
            self.allocated_blocks[owner][bi].append(block)

    def free(self, owner: GenerationSequence):
        """Free all blocks for a sequence."""
        for beam_blocks in self.allocated_blocks[owner]:
            for block in beam_blocks:
                block.remove_link()
                if not block.has_link():
                    self.free_blocks.append(block)

        del self.allocated_blocks[owner]

    def replace_shared_block(
        self,
        owner: GenerationSequence,
        block_idx: int
    ):
        """Replace shared block with private copies (copy-on-write).

        Used when beam search branches and beams diverge - each beam
        needs its own copy of the block.
        """
        if not self.allocated_blocks[owner][0][block_idx].is_shared():
            return  # Already private

        # Free shared block
        for bi in range(self.beam_width):
            block = self.allocated_blocks[owner][bi][block_idx]
            block.remove_link()
            if not block.has_link():
                self.free_blocks.append(block)

        # Allocate new private blocks for each beam
        for bi in range(self.beam_width):
            block = self.free_blocks.pop(0)
            block.add_link()
            self.allocated_blocks[owner][bi][block_idx] = block
```

**Key Features:**
- **FIFO allocation**: Blocks allocated in order from free pool
- **Copy-on-write**: Shared blocks replaced when beams diverge
- **Beam width support**: Each sequence can have multiple beams
- **Explicit sharing**: Application must manage sharing manually

#### C. Sequence Management

**Location**: `tensorrt_llm/runtime/kv_cache_manager.py:40-64`

```python
class GenerationSequence(object):
    """Represents a generation sequence (request)."""

    def __init__(self, seq_idx, batch_idx):
        self.seq_idx = seq_idx      # Unique sequence identifier
        self.batch_idx = batch_idx  # Position in batch

    def get_batch_idx(self) -> int:
        """Returns index of sequence in batch."""
        return self.batch_idx

    def get_seq_idx(self) -> int:
        """Returns sequence index."""
        return self.seq_idx
```

### 3.3 Core Data Structures

#### Block Offsets Tensor

**Location**: `tensorrt_llm/runtime/kv_cache_manager.py:181-215`

```python
def get_offset_array(self, beam_width: int) -> torch.Tensor:
    """Returns block offset array for attention kernels.

    Returns:
        Tensor of shape [batch_size, beam_width, 2, max_blocks_per_seq]
        where dimension 2 contains [key_offsets, value_offsets]
    """
    offset_array = create_nested_list(
        (len(self.allocated_blocks), beam_width, 2, self.max_blocks_per_seq)
    )

    k_idx = 0  # Key cache index
    v_idx = 1  # Value cache index

    # Populate offset array
    for owner, beams_blocks in self.allocated_blocks.items():
        for bi, blocks in enumerate(beams_blocks):
            for block_idx, block in enumerate(blocks):
                # Key cache offset
                offset_array[owner.batch_idx][bi][k_idx][block_idx] = (
                    self.get_k_or_v_block_offset(block.idx, k_idx)
                )
                # Value cache offset
                offset_array[owner.batch_idx][bi][v_idx][block_idx] = (
                    self.get_k_or_v_block_offset(block.idx, v_idx)
                )

    return torch.tensor(offset_array, dtype=torch.int32)

def get_k_or_v_block_offset(self, block_idx: int, k_or_v: int) -> int:
    """Calculate physical memory offset for block.

    Args:
        block_idx: Logical block index
        k_or_v: 0 for key cache, 1 for value cache

    Returns:
        Physical offset in KV cache memory
    """
    return (block_idx * self.num_layers + k_or_v) * self.block_size
```

**Usage:**
- Attention kernels receive offset arrays to locate KV cache blocks
- Supports paged attention with non-contiguous memory
- Efficient for beam search with separate beam offsets

### 3.4 Integration with Memory Pools

**Location**: `tensorrt_llm/runtime/memory_pools/pools_kv_cache_manager.py:10-60`

```python
class PoolsKVCacheManager(object):
    """Manages multiple KV cache managers for different model layers.

    Supports models with different attention types in different layers.
    """

    def __init__(
        self,
        pools_metadata: List[Pool],
        max_blocks_per_seq,
        num_blocks,
        tokens_per_block,
        head_size,
        max_attention_window_size,
        beam_width,
        sink_token_len
    ):
        self._num_pools = len(pools_metadata)
        self._kv_cache_managers = []

        # Create separate manager for each pool
        for pool in pools_metadata:
            block_size = pool.num_kv_heads * tokens_per_block * head_size
            self._kv_cache_managers.append(
                KVCacheManager(
                    num_layers=pool.num_layers,
                    num_blocks=num_blocks,
                    block_size=block_size,
                    tokens_per_block=tokens_per_block,
                    max_blocks_per_seq=max_blocks_per_seq,
                    max_attention_window_size=max_attention_window_size,
                    sink_token_len=sink_token_len,
                    beam_width=beam_width,
                )
            )

    def add_sequence(self, seq: GenerationSequence, num_blocks: int):
        """Add sequence to all pools."""
        for manager in self._kv_cache_managers:
            manager.add_sequence(seq, num_blocks)
```

### 3.5 Key Features

| Feature | Implementation |
|---------|-----------------|
| **Sharing Model** | Explicit reference counting |
| **Allocation Strategy** | FIFO (simple queue) |
| **Beam Search** | Copy-on-write block replacement |
| **Memory Layout** | Contiguous block pools per layer |
| **Multi-Pool Support** | Separate managers for model sections |
| **Prefix Caching** | Manual via block sharing (no automatic lookup) |

---

## 4. Comparative Analysis

### 4.1 Prefix Caching Approach

| Aspect | vLLM | SGLang | TensorRT-LLM |
|--------|------|--------|--------------|
| **Data Structure** | Hash map + linked list | Radix tree | Block array + ref counting |
| **Lookup Method** | Hash-based O(1) per block | Tree traversal O(m) | Manual sharing |
| **Automatic Cache Hit** | ✓ Yes | ✓ Yes | ✗ No |
| **Block Sharing** | Hash table + LRU | Radix tree nodes | Explicit ref counting |
| **Hash Algorithm** | SHA256, xxHash | SHA256 position-aware | None |
| **Granularity** | Block-level (16 tokens) | Page-level (configurable) | Block-level |

### 4.2 Algorithm Complexity

| Operation | vLLM | SGLang | TensorRT-LLM |
|-----------|------|--------|--------------|
| **Prefix Match** | O(k) hash lookups | O(m) tree traversal | O(1) check |
| **Cache Insert** | O(1) hash insert | O(m) tree insert | O(1) add_link |
| **Block Eviction** | O(1) LRU pop | O(n) find candidate | O(1) free |
| **Memory Overhead** | ~32B hash + metadata | Full tensor + metadata | ~8B ref count |

**Legend:**
- k = number of blocks matched
- m = matched prefix length (tokens)
- n = number of candidate nodes

### 4.3 Cache Hit Detection

**vLLM:**
```
1. Compute block hashes (incremental, cached)
2. For each block hash:
   a. Lookup in hash table (O(1))
   b. If found, add to matched blocks
   c. If not found, stop
3. Return matched blocks
```

**SGLang:**
```
1. Start at root node
2. While tokens remain:
   a. Find child matching next token/page (O(1))
   b. If no child, stop
   c. Match as many tokens as possible (O(p))
   d. If partial match, stop
   e. Continue to next child
3. Return accumulated KV tensors
```

**TensorRT-LLM:**
```
1. Application decides which blocks to share
2. Call allocate(share_across_beam=True)
3. Manually increment ref_count
4. On completion, decrement ref_count
```

### 4.4 Memory Efficiency Comparison

| Framework | Strategy | Overhead per Block |
|-----------|----------|-------------------|
| **vLLM** | Hash-based sharing | ~80 bytes (hash + metadata + pointers) |
| **SGLang** | Tree node storage | ~200+ bytes (node + tensor ptr + metadata) |
| **TensorRT-LLM** | Dense array | ~8 bytes (ref count only) |

### 4.5 Eviction Policies

| Framework | Eviction Strategy | Implementation |
|-----------|------------------|----------------|
| **vLLM** | LRU (implicit) | Doubly-linked free list |
| **SGLang** | Pluggable (6 options) | Strategy pattern with metadata |
| **TensorRT-LLM** | FIFO / Manual | Simple list append |

**SGLang Eviction Options:**
1. **LRU** - Best for temporal locality (most common)
2. **LFU** - Best for frequency-based patterns
3. **FIFO** - Simple, predictable
4. **MRU** - For anti-caching patterns
5. **FILO** - Stack-like access
6. **Priority** - Application-controlled importance

### 4.6 Special Features

**vLLM:**
- ✓ Multiple hash algorithms (SHA256, xxHash)
- ✓ CBOR serialization for reproducibility
- ✓ Hybrid attention support (fixed-point algorithm)
- ✓ EAGLE speculative decoding integration
- ✓ Automatic LoRA/multimodal hash inclusion
- ✓ Prefix cache statistics tracking

**SGLang:**
- ✓ Position-aware hashing (SHA256 chaining)
- ✓ 6 pluggable eviction policies
- ✓ HiCache for CPU offloading
- ✓ EAGLE speculative decoding support
- ✓ Priority-based eviction
- ✓ Mamba model support (COW semantics)

**TensorRT-LLM:**
- ✓ Explicit copy-on-write for beam search
- ✓ Multi-pool architecture for heterogeneous layers
- ✓ Simple reference counting model
- ✓ Beam width multiplexing
- ✓ Sink token support for long context
- ✗ No automatic prefix cache lookup

---

## 5. Configuration Examples

### 5.1 vLLM Configuration

```python
from vllm import LLM, SamplingParams

# Basic prefix caching with SHA256
llm = LLM(
    model="meta-llama/Llama-2-7b-hf",
    enable_prefix_caching=True,
    gpu_memory_utilization=0.9,
)

# Use xxHash for faster hashing
llm = LLM(
    model="meta-llama/Llama-2-7b-hf",
    enable_prefix_caching=True,
    prefix_caching_hash_algo="xxhash",  # Requires: pip install xxhash
    block_size=16,
)

# CBOR serialization for reproducibility
llm = LLM(
    model="meta-llama/Llama-2-7b-hf",
    enable_prefix_caching=True,
    prefix_caching_hash_algo="sha256_cbor",
)

# Usage example with common prefix
prefix = "You are an expert Python programmer. "
prompts = [
    prefix + "Write a function to compute fibonacci.",
    prefix + "Write a function to sort a list.",
    prefix + "Write a function to reverse a string.",
]

# First request processes prefix, others reuse cache
outputs = llm.generate(prompts, SamplingParams(temperature=0.0))
```

### 5.2 SGLang Configuration

```python
from sglang.srt.mem_cache.radix_cache import RadixCache
from sglang.srt.mem_cache.base_prefix_cache import CacheInitParams

# Configure RadixCache with LRU eviction
cache = RadixCache(
    CacheInitParams(
        disable=False,
        req_to_token_pool=token_pool,
        token_to_kv_pool_allocator=allocator,
        page_size=4,           # 4-token pages
        eviction_policy="lru", # Options: lru, lfu, fifo, mru, filo, priority
    )
)

# Match prefix
from sglang.srt.mem_cache.base_prefix_cache import MatchPrefixParams, RadixKey

result = cache.match_prefix(
    MatchPrefixParams(
        key=RadixKey(
            token_ids=[1, 2, 3, 4, 5],
            extra_key="lora_adapter_1",  # Namespace isolation
        )
    )
)

print(f"Matched tokens: {len(result.device_indices)}")

# Server launch with automatic prefix caching
# $ python -m sglang.launch_server \
#     --model-path meta-llama/Llama-3.1-8B \
#     --port 30000
```

### 5.3 TensorRT-LLM Configuration

```python
from tensorrt_llm.runtime.kv_cache_manager import BlocksManager, GenerationSequence

# Create block manager
blocks_mgr = BlocksManager(
    num_layers=32,
    num_blocks=256,
    block_size=128,  # tokens per block * head_dim
    max_blocks_per_seq=128,
    beam_width=4,
)

# Allocate blocks for sequence
seq = GenerationSequence(seq_idx=0, batch_idx=0)
for _ in range(10):
    blocks_mgr.allocate(owner=seq, share_across_beam=True)

# On beam search branch, replace shared blocks
blocks_mgr.replace_shared_block(owner=seq, block_idx=5)

# Get offset array for attention kernels
offset_array = blocks_mgr.get_offset_array(beam_width=4)

# Free sequence blocks
blocks_mgr.free(owner=seq)
```

---

## 6. Integration with Scheduler

### 6.1 vLLM Scheduler Integration

**File**: `vllm/v1/core/kv_cache_manager.py`

**Flow:**
```
1. Request arrives with token sequence
2. Scheduler calls KVCacheManager.get_computed_blocks(request)
3. Manager:
   a. Computes block hashes from token sequence
   b. Calls coordinator.find_longest_cache_hit(block_hashes)
   c. Returns (cached_blocks, num_tokens_hit)
4. Scheduler allocates only remaining blocks
5. Prefill skips cached tokens, continues from num_tokens_hit
6. On completion, blocks are cached for future requests
```

**Statistics Tracking:**
- Total cache queries
- Total cache hits
- Hit rate per 1000 queries
- Impact of preemption on cache hits

**Code Example:**
```python
# In scheduler
computed_blocks, num_hit_tokens = kv_cache_mgr.get_computed_blocks(request)

# Adjust prefill range
prefill_start = num_hit_tokens
prefill_end = request.num_tokens

# Allocate only missing blocks
num_blocks_needed = (prefill_end - prefill_start + block_size - 1) // block_size
```

### 6.2 SGLang Scheduler Integration

**File**: `sglang/srt/mem_cache/radix_cache.py`

**Flow:**
```
1. Request arrives with token IDs
2. Create RadixKey from token_ids + extra metadata
3. Call radix_cache.match_prefix(params)
4. RadixCache:
   a. Traverses tree from root
   b. Matches longest prefix
   c. Returns matched KV indices and terminal node
5. Scheduler uses matched indices for zero-copy prefix
6. On request finish, call cache_finished_req() to insert into tree
```

**Key Integration Points:**
- `forward()` - Checks cache before execution
- `schedule()` - Allocates tokens for uncached portion only
- `finish_request()` - Inserts completed KV cache into tree

**Code Example:**
```python
# In scheduler forward pass
match_result = self.radix_cache.match_prefix(
    MatchPrefixParams(
        key=RadixKey(req.input_ids),
        req=req,
    )
)

if len(match_result.device_indices) > 0:
    # Reuse cached KV
    req.kv_indices = match_result.device_indices
    req.prefix_len = len(match_result.device_indices)
```

### 6.3 TensorRT-LLM Scheduler Integration

**File**: `tensorrt_llm/runtime/kv_cache_manager.py`

**Flow:**
```
1. Request arrives
2. Application manually identifies shared prefix
3. For sequences sharing prefix:
   a. Allocate blocks with share_across_beam=True
   b. Blocks have ref_count > 1
4. On beam search divergence:
   a. Call replace_shared_block()
   b. Private copies allocated
5. On completion, decrement ref_count
```

**No Automatic Cache Lookup:**
- Application must manage sharing explicitly
- Useful for known patterns (beam search, multi-turn chat)
- Lower overhead but requires manual management

---

## 7. Key Insights and Optimizations

### 7.1 Cache Hit Optimization Strategies

**vLLM Optimizations:**
1. **Incremental hashing**: Block hashes computed once and cached
2. **Linear scan with early exit**: CPU cache-friendly, stops at first miss
3. **EAGLE integration**: Automatically drops last block for speculative decoding
4. **Chained hashing**: Position-aware without recomputing full sequence
5. **Multiple hash algorithms**: Trade security vs speed (SHA256 vs xxHash)

**SGLang Optimizations:**
1. **Radix tree compression**: Common prefixes stored once
2. **Page-aligned matching**: Reduces comparisons for large page sizes
3. **Node splitting**: Refines boundaries for future cache hits
4. **Pluggable eviction**: Adapt to workload characteristics
5. **Position-aware hashing**: Enables distributed cache consistency

**TensorRT-LLM Optimizations:**
1. **Simple ref counting**: Minimal overhead, predictable behavior
2. **FIFO allocation**: Good cache locality
3. **Copy-on-write**: Lazy duplication for beam search
4. **Multi-pool**: Different strategies for different layer types

### 7.2 Performance Bottlenecks

| Bottleneck | vLLM | SGLang | TensorRT-LLM |
|------------|------|--------|--------------|
| **Hash Computation** | Amortized via incremental | Per-page SHA256 | N/A |
| **Lookup Latency** | O(1) per block | O(m) traversal | O(1) manual |
| **Eviction Cost** | O(1) LRU pop | O(n) heap/scan | O(1) FIFO |
| **Memory Fragmentation** | Low (hash-based) | Medium (tree) | Low (array) |

### 7.3 Memory Efficiency Analysis

**vLLM:**
- Hash table overhead: ~32 bytes per block hash
- Metadata overhead: ~48 bytes per block
- Total: ~80 bytes per 16-token block = 5 bytes/token
- Hash computation: Amortized O(1) per block

**SGLang:**
- Tree node overhead: ~200 bytes
- Full tensor pointer storage
- Position-aware hashing: SHA256 per page
- Total: ~200+ bytes per node
- Better for very long common prefixes

**TensorRT-LLM:**
- Reference count: 4 bytes
- Block index: 4 bytes
- Total: ~8 bytes per block = 0.5 bytes/token (128 tokens/block)
- No automatic lookup overhead

### 7.4 Workload Recommendations

**Use vLLM when:**
- Automatic prefix caching is critical
- Diverse workloads with varying prefixes
- Need hash algorithm flexibility
- Multi-turn conversations with common context
- LoRA adapters or multimodal inputs need isolation

**Use SGLang when:**
- Very long common prefixes (e.g., large system prompts)
- Need custom eviction policies
- CPU offloading required (HiCache)
- Workload has priority tiers
- Fine-grained control over caching behavior

**Use TensorRT-LLM when:**
- Beam search is primary use case
- Explicit control over sharing preferred
- Minimal overhead is critical
- Known prefix patterns (manual management acceptable)
- Multi-pool architecture needed

### 7.5 Cache Hit Rate Factors

**Common Factors:**
1. **Prefix length**: Longer common prefixes → higher hit rate
2. **Block/page size**: Larger blocks reduce hit granularity
3. **Workload pattern**: Few unique prefixes → higher hit rate
4. **Cache capacity**: More memory → more cached blocks
5. **Eviction policy**: Matches access pattern → higher hit rate

**Example Hit Rates:**
- Multi-turn chat with system prompt: 80-95%
- Batch inference with common prefix: 70-90%
- Diverse user queries: 10-30%
- RAG with fixed context: 60-80%

---

## 8. Summary

### 8.1 Quick Comparison Table

| Property | vLLM | SGLang | TensorRT-LLM |
|----------|------|--------|--------------|
| **Core Data Structure** | Hash map + LRU list | Radix tree | Block array |
| **Lookup Complexity** | O(1) per block | O(m) traversal | Manual |
| **Automatic Cache** | ✓ Yes | ✓ Yes | ✗ No |
| **Eviction Policy** | LRU (implicit) | 6 pluggable strategies | FIFO/manual |
| **Hash Algorithm** | SHA256/xxHash/CBOR | SHA256 position-aware | None |
| **Beam Search Support** | Via groups | Via COW | Explicit CoW |
| **Memory Overhead** | ~5 bytes/token | ~variable | ~0.5 bytes/token |
| **Position Awareness** | Block chaining | SHA256 chaining | N/A |
| **LoRA/Multimodal** | Integrated hash | Extra key namespace | Manual |
| **File Count** | 6 core files | 5 core files | 3 core files |
| **Configuration** | CacheConfig | CacheInitParams | BlocksManager init |
| **Best For** | General purpose | Long prefixes | Beam search |

### 8.2 Key Takeaways

1. **vLLM** provides the most comprehensive automatic prefix caching with flexible hashing and seamless integration
2. **SGLang** excels at very long common prefixes with rich eviction policy options and CPU offloading
3. **TensorRT-LLM** focuses on explicit control with minimal overhead, ideal for beam search and known patterns

### 8.3 File Reference Summary

**vLLM Core Files:**
- `vllm/v1/core/kv_cache_manager.py` (lines 176-216) - Main manager
- `vllm/v1/core/kv_cache_coordinator.py` (lines 349-498) - Hybrid attention
- `vllm/v1/core/single_type_kv_cache_manager.py` (lines 421-460) - Cache hit finding
- `vllm/v1/core/block_pool.py` (lines 33-208) - Hash table implementation
- `vllm/v1/core/kv_cache_utils.py` (lines 532-613) - Hashing algorithms
- `vllm/config/cache.py` (lines 26-84) - Configuration

**SGLang Core Files:**
- `sglang/srt/mem_cache/radix_cache.py` (lines 97-438) - Tree implementation
- `sglang/srt/mem_cache/base_prefix_cache.py` (lines 27-143) - Interface
- `sglang/srt/mem_cache/evict_policy.py` - Eviction strategies
- `sglang/srt/mem_cache/hiradix_cache.py` - CPU offloading
- `sglang/srt/mem_cache/hicache_storage.py` (lines 201-232) - Position-aware hashing

**TensorRT-LLM Core Files:**
- `tensorrt_llm/runtime/kv_cache_manager.py` (lines 21-215) - Block management
- `tensorrt_llm/runtime/memory_pools/pools_kv_cache_manager.py` (lines 10-60) - Multi-pool
- `tensorrt_llm/llmapi/build_cache.py` - Cache initialization

---

**Document Version**: 1.0
**Last Updated**: 2026-03-29
