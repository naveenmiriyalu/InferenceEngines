# LLM Inference Scheduling Algorithms

A comprehensive comparison of scheduling implementations in vLLM, SGLang, and TensorRT-LLM.

---

## Table of Contents
1. [Overview](#1-overview)
2. [vLLM Scheduling](#2-vllm-scheduling)
3. [SGLang Scheduling](#3-sglang-scheduling)
4. [TensorRT-LLM Scheduling](#4-tensorrt-llm-scheduling)
5. [Algorithmic Comparison](#5-algorithmic-comparison)
6. [Prefill/Decode Optimizations](#6-prefilldecode-optimizations)
7. [Configuration Guide](#7-configuration-guide)
8. [Best Practices](#8-best-practices)

---

## 1. Overview

### What is LLM Scheduling?

LLM scheduling decides:
1. **Which requests** to execute in each iteration
2. **How many tokens** to process per request
3. **Resource allocation** (GPU memory, KV cache blocks)
4. **Request prioritization** and preemption

### Key Challenges

- **Variable sequence lengths**: Requests have different prompt/output sizes
- **Memory constraints**: Limited KV cache capacity
- **Fairness vs throughput**: Balance individual latency with system throughput
- **Prefill-decode asymmetry**: Prefill processes N tokens, decode generates 1 token

### Scheduling Approaches

| System | Core Model | Primary Constraint |
|--------|------------|-------------------|
| **vLLM** | Token budget accounting | max_num_batched_tokens + max_num_seqs |
| **SGLang** | Token budget + continuous batching | max_total_tokens + max_prefill_tokens |
| **TensorRT-LLM** | Two-tier capacity + micro-batch | Capacity scheduler → Micro-batch scheduler |

---

## 2. vLLM Scheduling

### 2.1 Core Architecture

**Location**: `/home/nmiriyal/Documents/UnderstandingVLLM/LatestVLLM/vllm/vllm/v1/core/sched/scheduler.py`

**Key Insight**: No separate "prefill" and "decode" phases. Unified token accounting model.

**Design Philosophy** (Lines 339-348):
```python
# NOTE(woosuk) on the scheduling algorithm:
# There's no "decoding phase" nor "prefill phase" in the scheduler.
# Each request just has the num_computed_tokens and
# num_tokens_with_spec. num_tokens_with_spec =
# len(prompt_token_ids) + len(output_token_ids) + len(spec_token_ids).
# At each step, the scheduler tries to assign tokens to the requests
# so that each request's num_computed_tokens can catch up its
# num_tokens_with_spec.
```

### 2.2 Understanding max_num_batched_tokens

**Definition** (from `vllm/config/scheduler.py`, lines 42-53):

```python
max_num_batched_tokens: int = Field(default=2048)
    # Maximum number of tokens to be processed in a single iteration.
    # This is the token budget for scheduling.
    # In each iteration, the scheduler will try to schedule requests
    # up to max_num_batched_tokens tokens in total.
```

**How It Works** (scheduler.py, lines 357-360):

```python
def schedule(self) -> SchedulerOutput:
    token_budget = self.max_num_scheduled_tokens  # Derived from max_num_batched_tokens
    # ... scheduler loops consume from token_budget
```

**Token Budget Consumption** (Lines 506-507, 801-802):

```python
# For each scheduled request:
num_scheduled_tokens[request_id] = num_new_tokens
token_budget -= num_new_tokens  # Deduct from budget
```

**Validation** (Lines 829-833):

```python
total_num_scheduled_tokens = sum(num_scheduled_tokens.values())
assert total_num_scheduled_tokens <= self.max_num_scheduled_tokens
assert token_budget >= 0
```

**Key Properties**:
- Token budget is **shared** between prefill and decode
- Large prefill requests consume more budget
- Multiple decode requests (1 token each) consume less budget
- Budget depleted when `token_budget <= 0` or no more requests

### 2.3 Understanding max_num_seqs

**Definition** (from `vllm/config/scheduler.py`, lines 62-67):

```python
max_num_seqs: int = Field(default=128)
    # Maximum number of sequences to be processed in a single iteration.
    # This is the sequence budget for scheduling.
```

**How It Works** (scheduler.py, lines 104-110):

```python
self.max_num_running_reqs = self.scheduler_config.max_num_seqs
```

**Enforcement** (Lines 558-559):

```python
while (self.waiting or self.skipped_waiting) and token_budget > 0:
    if len(self.running) == self.max_num_running_reqs:
        break  # Stop scheduling new requests - sequence limit reached
```

**Key Properties**:
- Independent constraint from token budget
- Limits **concurrent sequences**, not total tokens
- Prevents excessive context switching overhead
- Typically set to balance throughput and latency

### 2.4 Two-Phase Scheduling Algorithm

**Phase 1: Schedule RUNNING Requests** (Lines 373-508)

```python
req_index = 0
while req_index < len(self.running) and token_budget > 0:
    request = self.running[req_index]

    # 1. Calculate tokens needed (lines 394-401)
    num_new_tokens = (
        request.num_tokens_with_spec          # Target tokens
        + request.num_output_placeholders     # Async placeholders
        - request.num_computed_tokens         # Already computed
    )

    # 2. Apply chunked prefill threshold (lines 402-405)
    threshold = self.scheduler_config.long_prefill_token_threshold
    if 0 < threshold < num_new_tokens:
        num_new_tokens = threshold

    # 3. Cap by token budget (line 411)
    num_new_tokens = min(num_new_tokens, token_budget)

    # 4. Try to allocate KV cache (lines 451-497)
    while True:
        new_blocks = self.kv_cache_manager.allocate_slots(
            request, num_new_tokens, ...
        )
        if new_blocks is not None:
            break  # Success

        # PREEMPTION: Free memory by preempting lowest-priority request
        if self.policy == SchedulingPolicy.PRIORITY:
            preempted_req = max(
                self.running,
                key=lambda r: (r.priority, r.arrival_time)  # Higher = lower priority
            )
        else:  # FCFS
            preempted_req = self.running.pop()  # Last scheduled

        self._preempt_request(preempted_req, timestamp)
        preempted_reqs.append(preempted_req)

    # 5. Schedule the request
    scheduled_running_reqs.append(request)
    num_scheduled_tokens[request_id] = num_new_tokens
    token_budget -= num_new_tokens
    req_index += 1
```

**Key Features**:
- **Running requests scheduled first** (continue existing work)
- **Preemption on KV cache OOM** (free space for current request)
- **Token budget shared** across all running requests

**Phase 2: Schedule WAITING Requests** (Lines 553-827)

```python
# Only schedule waiting if NO preemptions occurred (line 554)
if not preempted_reqs and self._pause_state == PauseState.UNPAUSED:
    while (self.waiting or self.skipped_waiting) and token_budget > 0:
        # Check sequence limit (lines 558-559)
        if len(self.running) == self.max_num_running_reqs:
            break

        request = self.waiting.pop_request()

        # 1. Compute prefix cache hits (lines 600-640)
        if request.num_computed_tokens == 0:
            # Local cache
            new_computed_blocks, num_new_local_computed_tokens = (
                self.kv_cache_manager.get_computed_blocks(request)
            )
            # Remote cache (external KV connector)
            if self.connector is not None:
                ext_tokens, load_kv_async = (
                    self.connector.get_num_new_matched_tokens(request, ...)
                )
            num_computed_tokens = num_new_local + num_external

        # 2. Calculate tokens to schedule (lines 651-670)
        num_new_tokens = request.num_tokens - num_computed_tokens
        if self.scheduler_config.enable_chunked_prefill:
            num_new_tokens = min(num_new_tokens, token_budget)
        elif num_new_tokens > token_budget:
            break  # Can't schedule without chunking

        # 3. Allocate KV cache (lines 722-731)
        new_blocks = self.kv_cache_manager.allocate_slots(
            request, num_new_tokens,
            new_computed_blocks=new_computed_blocks,
            num_external_computed_tokens=num_external_computed_tokens,
        )
        if new_blocks is None:
            break  # Out of memory

        # 4. Schedule the request (lines 784-804)
        self.running.append(request)
        request.status = RequestStatus.RUNNING
        request.num_computed_tokens = num_computed_tokens
        num_scheduled_tokens[request_id] = num_new_tokens
        token_budget -= num_new_tokens
```

**Key Features**:
- **Only runs if no preemptions** (fairness)
- **Prefix cache automatically applied**
- **Chunked prefill enables partial scheduling**
- **Stops on first unschedulable request** (FCFS ordering)

### 2.5 Request State Machine

```
WAITING
  ↓ (scheduled)
RUNNING
  ↓ (preempted)
PREEMPTED → back to WAITING (prepended for quick re-schedule)
  ↓ (finished)
FINISHED_STOPPED / FINISHED_LENGTH_CAPPED / FINISHED_ABORTED
```

**State Transitions** (Lines 929-949):

```python
def _preempt_request(self, request: Request, timestamp: float) -> None:
    self.kv_cache_manager.free(request)              # Free KV cache
    request.status = RequestStatus.PREEMPTED
    request.num_computed_tokens = 0                  # Reset
    request.num_preemptions += 1
    self.waiting.prepend_request(request)            # High priority for re-schedule
```

### 2.6 Chunked Prefill

**Configuration** (scheduler.py, lines 83-89):

```python
enable_chunked_prefill: bool = True           # Enable chunking
long_prefill_token_threshold: int = 0         # Max tokens per prefill chunk
max_num_partial_prefills: int = 1             # Max concurrent partial prefills
max_long_partial_prefills: int = 1            # Max concurrent long prefills
```

**Algorithm** (Lines 656-668):

```python
threshold = self.scheduler_config.long_prefill_token_threshold
if 0 < threshold < num_new_tokens:
    num_new_tokens = threshold  # Cap prefill chunk size

if not self.scheduler_config.enable_chunked_prefill:
    if num_new_tokens > token_budget:
        break  # Can't schedule - entire prefill must fit
```

**Benefits**:
- Large prefills don't block all other requests
- Better fairness in mixed workloads
- Prefill can be split across multiple iterations

### 2.7 Code Sources

**Primary Files**:
- `vllm/v1/core/sched/scheduler.py` (1086 lines) - Main scheduling algorithm
- `vllm/config/scheduler.py` (280 lines) - Configuration definitions
- `vllm/v1/core/sched/request_queue.py` (150 lines) - Queue implementations
- `vllm/v1/core/kv_cache_manager.py` (800+ lines) - KV cache allocation
- `vllm/v1/request.py` (400+ lines) - Request state machine

**Key Methods**:
- `Scheduler.schedule()` - Line 338 (main algorithm)
- `Scheduler._preempt_request()` - Line 929 (preemption logic)
- `KVCacheManager.allocate_slots()` - KV cache allocation
- `KVCacheManager.get_computed_blocks()` - Prefix cache lookup

---

## 3. SGLang Scheduling

### 3.1 Core Architecture

**Location**: `/home/nmiriyal/Documents/UnderstandingVLLM/LatestVLLM/sglang/python/sglang/srt/managers/scheduler.py`

**Key Insight**: Continuous batching with explicit prefill/decode separation and cache-aware scheduling.

### 3.2 Token Budget Management

**Three Token Budgets** (Lines 574-580):

```python
(
    self.max_total_num_tokens,        # Total KV cache capacity
    self.max_prefill_tokens,          # Max tokens for prefill batches
    self.max_running_requests,        # Max concurrent requests
    ...
) = self.tp_worker.get_worker_info()
```

#### **1. max_total_tokens**

**Definition**: Total KV cache available for all requests

**Usage in PrefillAdder** (schedule_policy.py, lines 449-467):

```python
@property
def rem_total_tokens(self):
    """Total remaining tokens (available + evictable)"""
    available_and_evictable = (
        self.token_to_kv_pool_allocator.available_size()
        + self.tree_cache.evictable_size()
    )
    return available_and_evictable - self.rem_total_token_offset
```

**Conservative Estimation** (Lines 510-526):

```python
def _update_prefill_budget(self, prefix_len, extend_input_len, max_new_tokens):
    # Reserve space for both prefill AND future decode
    self.rem_total_token_offset += extend_input_len + max_new_tokens

    # Use new_token_ratio to estimate actual decode space needed
    estimated_decode_tokens = int(max_new_tokens * self.new_token_ratio)
```

**Token Ratio** (Lines 811-823):

```python
# Starts conservative, decays with successful decodes
self.new_token_ratio = 0.7  # Default: assume 70% of max_new_tokens needed
# Decays on each successful decode step
self.new_token_ratio = max(
    self.new_token_ratio - self.new_token_ratio_decay,
    self.min_new_token_ratio
)
```

#### **2. max_prefill_tokens**

**Definition**: Maximum tokens in a single prefill batch forward pass

**Purpose**: Prevent unfair prefill dominance over decode

**Usage** (scheduler.py, line 2036):

```python
adder = PrefillAdder(
    ...
    rem_input_tokens=self.max_prefill_tokens,  # Prefill token budget
    ...
)
```

**Budget Check** (schedule_policy.py, lines 747-748):

```python
if real_input_tokens >= self.rem_input_tokens and len(can_run_list) != 0:
    return AddReqResult.OTHER  # Prefill budget exhausted
```

#### **3. max_running_requests**

**Definition**: Maximum concurrent requests in running batch

**Enforcement** (scheduler.py, lines 2071-2072):

```python
if len(adder.can_run_list) >= self.get_num_allocatable_reqs(running_bs):
    self.running_batch.batch_is_full = True
```

### 3.3 PrefillAdder Algorithm

**Location**: `sglang/srt/managers/schedule_policy.py` (Lines 372-898)

**Purpose**: Build a prefill batch from waiting queue while respecting budgets

**Core Addition Loop** (scheduler.py, lines 2052-2097):

```python
for req in self.waiting_queue:
    # 1. Check LoRA compatibility
    if self.enable_lora and req.lora_id not in running_loras:
        if not self._check_lora_batch(...):
            continue

    # 2. Check if batch is full
    if len(adder.can_run_list) >= self.get_num_allocatable_reqs(running_bs):
        self.running_batch.batch_is_full = True

    # 3. Try preemption if batch full
    if self.running_batch.batch_is_full:
        if not self.try_preemption or not adder.preempt_to_schedule(req):
            break

    # 4. Add request to batch
    res = adder.add_one_req(req, has_chunked_req=(self.chunked_req is not None))

    if res != AddReqResult.CONTINUE:
        if res == AddReqResult.NO_TOKEN:
            self.running_batch.batch_is_full = True
        break
```

**Individual Request Addition** (schedule_policy.py, lines 719-827):

```python
def add_one_req(self, req, has_chunked_req, truncation_align_size):
    # 1. Check total token budget
    total_tokens = req.extend_input_len + max_new_tokens
    if total_tokens >= self.rem_total_tokens:
        return AddReqResult.NO_TOKEN

    # 2. Check prefill token budget
    if real_input_tokens >= self.rem_input_tokens and len(can_run_list) != 0:
        return AddReqResult.OTHER

    # 3. Lock cache node for thread safety
    with self._lock_node(req.last_node):
        # 4. Load back host cache if available
        if req.host_hit_length > 0:
            new_indices = self.tree_cache.init_load_back(
                req.last_host_node, req.host_hit_length
            )

        # 5. Handle chunked prefill
        if self.rem_chunk_tokens is None or input_tokens <= self.rem_chunk_tokens:
            # Full prefill
            self.can_run_list.append(req)
            self._update_prefill_budget(prefix_len, input_tokens, max_new_tokens)
        else:
            # Chunked: truncate to rem_chunk_tokens
            trunc_len = self.rem_chunk_tokens // self.page_size * self.page_size
            req.set_extend_input_len(trunc_len)
            self.can_run_list.append(req)
            self.new_chunked_req = req
            self._update_prefill_budget(prefix_len, trunc_len, 0)

    return self.budget_state()
```

### 3.4 Cache-Aware Scheduling Policies

**Location**: `sglang/srt/managers/schedule_policy.py` (Lines 93-363)

**Available Policies**:

| Policy | Type | Algorithm | Line |
|--------|------|-----------|------|
| **LPM** | Cache-aware | Longest prefix match | 133 |
| **DFS_WEIGHT** | Cache-aware | Depth-first search weighting | 137 |
| **FCFS** | Cache-agnostic | First come first serve | 117 |
| **LOF** | Cache-agnostic | Longest output first | 144 |
| **RANDOM** | Cache-agnostic | Random shuffle | 150 |
| **ROUTING_KEY** | Cache-agnostic | Routing key matching | 152 |

**LPM (Longest Prefix Match)** - Most Important (Lines 182-240):

```python
def _compute_prefix_matches(self, waiting_queue, policy):
    for req in waiting_queue:
        prefix_ids = req.origin_input_ids + req.output_ids

        # Match against RadixCache
        match_result = self.tree_cache.match_prefix(
            MatchPrefixParams(key=RadixKey(token_ids=prefix_ids, ...))
        )

        # Store matched prefix
        req.prefix_indices = match_result.device_indices
        req.last_node = match_result.last_device_node
        req.host_hit_length = match_result.host_hit_length

    # Sort by longest prefix match
    waiting_queue.sort(key=lambda r: -len(r.prefix_indices))

    # In-batch prefix caching: deprioritize if better to wait
    for req in waiting_queue:
        if len(req.prefix_indices) <= IN_BATCH_PREFIX_CACHING_CHECK_THRESHOLD:
            in_batch_matches = self.waiting_queue_radix_tree.match_prefix(...)
            if len(in_batch_matches) >= IN_BATCH_PREFIX_CACHING_DEPRIORITIZE_THRESHOLD:
                temporary_deprioritized.add(req.rid)
```

**Why LPM Matters**:
- Prioritizes requests with high cache hit rates
- Reduces actual compute needed
- Better GPU utilization
- Lower latency for cached requests

### 3.5 Continuous Batching

**Main Loop** (scheduler.py, line 1875):

```python
def get_next_batch_to_run(self):
    # 1. Filter finished requests (line 1899)
    self.filter_finished_req()

    # 2. Handle chunked prefill continuation (line 2045)
    if self.chunked_req is not None:
        self.chunked_req.init_next_round_input()
        self.chunked_req = adder.add_chunked_req(self.chunked_req)

    # 3. Check if can schedule new prefill (lines 2013)
    if self.running_batch.is_empty() or not self.running_batch.batch_is_full:
        new_batch = self.get_new_batch_prefill()
        if new_batch is not None:
            return new_batch  # Prefill batch

    # 4. Otherwise, continue decode (line 1935)
    if not self.running_batch.is_empty():
        return self.running_batch  # Decode batch
```

**Mixed Batching** (Lines 2193, schedule_batch.py:1770-1799):

```python
if self.is_mixed_chunk:
    # Mix prefill with decode in same batch
    self.running_batch.prepare_for_decode()
    new_batch.mix_with_running(self.running_batch)
    new_batch.forward_mode = ForwardMode.MIXED
    return new_batch
```

### 3.6 Decode Retraction (OOM Handling)

**Memory Check** (scheduler.py, lines 2213-2220):

```python
def update_running_batch(self, batch):
    if not batch.check_decode_mem():
        # Decode would OOM - retract requests
        retracted_reqs, new_token_ratio, reqs_to_abort = batch.retract_decode()

        # Update token ratio estimate
        self.new_token_ratio = new_token_ratio

        # Return retracted requests to waiting queue
        for req in retracted_reqs:
            self._add_request_to_queue(req, is_retracted=True)
```

**Retraction Algorithm** (schedule_batch.py, lines 1847-1905):

```python
def retract_decode(self):
    # Sort by output length (remove short sequences first)
    sorted_indices.sort(
        key=lambda i: (
            len(self.reqs[i].output_ids),          # Ascending
            -len(self.reqs[i].origin_input_ids)    # Descending
        ),
        reverse=True
    )

    # Remove requests until memory sufficient
    while not self.check_decode_mem(selected_indices):
        idx = sorted_indices.pop()
        retracted_reqs.append(self.reqs[idx])
        self.release_req(idx, len(sorted_indices))

    # Compute new token ratio
    total_decoded = sum(len(r.output_ids) for r in self.reqs)
    total_max = sum(r.sampling_params.max_new_tokens for r in self.reqs)
    new_ratio = (total_decoded + STEPS * len(reqs)) / (total_max + 1)

    return retracted_reqs, min(1.0, new_ratio), reqs_to_abort
```

### 3.7 Chunked Prefill

**Initialization** (scheduler.py, lines 764-771):

```python
self.chunked_prefill_size = server_args.chunked_prefill_size  # e.g., 512
self.chunked_req = None  # Current chunked request
self.is_mixed_chunk = (
    chunked_prefill_size is not None and enable_mixed_chunk
)
```

**Chunking Logic** (schedule_policy.py, lines 591-616):

```python
def add_chunked_req(self, req):
    _rem_tokens = min(self.rem_chunk_tokens, int(self.rem_total_tokens))

    # Truncate to available chunk tokens
    truncated = req.extend_input_len > _rem_tokens
    req.set_extend_input_len(min(req.extend_input_len, _rem_tokens))
    req.fill_ids = req.fill_ids[:len(req.prefix_indices) + req.extend_input_len]

    self.can_run_list.append(req)
    self._update_prefill_budget(0, req.extend_input_len,
                               0 if truncated else max_new_tokens)

    return req if truncated else None  # Return if still needs chunking
```

### 3.8 Code Sources

**Primary Files**:
- `sglang/srt/managers/scheduler.py` (3500+ lines) - Main scheduling loop
- `sglang/srt/managers/schedule_policy.py` (900 lines) - Policies and PrefillAdder
- `sglang/srt/managers/schedule_batch.py` (2100 lines) - Batch construction
- `sglang/srt/mem_cache/radix_cache.py` (1800+ lines) - RadixAttention cache
- `sglang/srt/managers/io_struct.py` (200 lines) - Request data structures

**Key Methods**:
- `Scheduler.get_next_batch_to_run()` - Line 1875 (main orchestration)
- `Scheduler._get_new_batch_prefill_raw()` - Line 1977 (prefill batch formation)
- `PrefillAdder.add_one_req()` - Line 719 (individual request addition)
- `SchedulePolicy.calc_priority()` - Line 93 (cache-aware sorting)
- `ScheduleBatch.retract_decode()` - Line 1847 (OOM handling)

---

## 4. TensorRT-LLM Scheduling

### 4.1 Core Architecture

**Location**: `/home/nmiriyal/Documents/UnderstandingVLLM/LatestVLLM/TensorRT-LLM/cpp/tensorrt_llm/batch_manager/`

**Key Insight**: Two-tier scheduling with capacity constraints and micro-batch organization.

### 4.2 Two-Tier Scheduler Design

```
Request Pool
    ↓
[Tier 1] CapacityScheduler
    ├─ Filters by resource availability
    ├─ Applies capacity policy (MAX_UTILIZATION, GUARANTEED_NO_EVICT, etc.)
    └─ Returns: fitting requests + paused requests
    ↓
[Tier 2] MicroBatchScheduler
    ├─ Splits into context (prefill) and generation (decode)
    ├─ Applies context chunking
    └─ Returns: context requests + generation requests
    ↓
Execution
```

### 4.3 Capacity Scheduler

**Location**: `cpp/tensorrt_llm/batch_manager/capacityScheduler.cpp`

**Four Scheduling Policies**:

#### **1. MaxRequestsScheduler** (Lines 154-175)

**Purpose**: Simple count-based limit

```cpp
for (auto const& req : activeRequests) {
    if (scheduledRequests.size() >= maxNumRequests) break;

    if (req->isEncoderInitState() || req->isContextInitState() ||
        req->isGenerationInProgressState()) {
        scheduledRequests.emplace_back(req);
    }
}
```

**When to Use**: No KV cache manager available, simple batch size limiting

#### **2. MaxUtilizationScheduler** (Lines 341-425)

**Purpose**: Maximum GPU utilization with optional eviction

**Algorithm**:
```cpp
for (auto const& req : activeRequests) {
    // Try to schedule request
    if (canSchedule(req, kvCacheManager, peftCacheManager)) {
        scheduledRequests.push_back(req);
    } else {
        // Resource exhausted - try pausing last scheduled request
        if (!scheduledRequests.empty()) {
            auto lastReq = scheduledRequests.back();
            scheduledRequests.pop_back();
            pausedRequests.push_back(lastReq);

            // Retry current request with freed resources
            if (canSchedule(req, ...)) {
                scheduledRequests.push_back(req);
            }
        }
    }
}
```

**KV Cache Reuse Optimization** (Lines 34-123):

```cpp
// Skip scheduling if another request already created needed blocks
auto newContextBlockOpt = kvCacheManager.findNewContextBlock(uniqueTokens, *req);
if (newContextBlockOpt.has_value()) {
    auto newContextBlock = newContextBlockOpt.value();
    if (newlyContributedContextBlocks.count(newContextBlock) > 0) {
        // Better to skip - we'll reuse that block later
        return true;  // Skip this request
    }
    newlyContributedContextBlocks.insert(newContextBlock);
}
```

**Benefits**:
- Identifies duplicate context work
- Skips requests that would duplicate blocks
- Improves cache hit rate

#### **3. GuaranteedNoEvictScheduler** (Lines 195-332)

**Purpose**: Once started, never paused - exclusive scheduling

**Algorithm**:
```cpp
// Pre-allocate all required resources
for (auto const& req : activeRequests) {
    // Check if resources available BEFORE scheduling
    if (!hasEnoughKvCacheBlocks(req) ||
        !hasEnoughCrossAttentionBlocks(req) ||
        !hasEnoughPeftCachePages(req)) {
        break;  // Stop scheduling - insufficient resources
    }

    scheduledRequests.push_back(req);
    reserveResources(req);  // Reserve for entire duration
}
```

**When to Use**: Low-latency applications where pausing is unacceptable

#### **4. StaticBatchScheduler** (Template Specialization)

**Purpose**: Static batching - only add new requests when batch empty

**Constraint**:
```cpp
if (!StaticBatchScheduling || scheduledRequests.size() == 0) {
    // Can only add if batch is empty
}
```

**When to Use**: Fixed batch size TensorRT engines

### 4.4 Micro-Batch Scheduler

**Location**: `cpp/tensorrt_llm/batch_manager/microBatchScheduler.cpp`

**Purpose**: Organize requests into context (prefill) and generation (decode) with chunking

**Main Function** (Lines 171-325):

```cpp
std::tuple<RequestVector, RequestVector> operator()(
    RequestVector& activeRequests,
    ReqIdsSet const& inflightReqIds,
    SizeType32 maxBatchSizeRuntime,
    std::optional<SizeType32> maxNumTokensRuntime)
{
    RequestVector contextRequests, generationRequests;

    // Phase 1: Select generation requests first (lines 189-285)
    SizeType32 scheduledBeamWidth = 0;
    for (auto& req : activeRequests) {
        if (inflightReqIds.count(req->getReqId())) continue;  // Skip inflight

        if (!req->isContextInitState()) {
            // Generation request
            auto reqBeamWidth = req->getSamplingConfig().beamWidth;

            // Enforce beam width uniformity
            if (scheduledBeamWidth == 0) {
                scheduledBeamWidth = reqBeamWidth;
            } else if (scheduledBeamWidth != reqBeamWidth) {
                continue;  // Skip - different beam width
            }

            generationRequests.push_back(req);
        } else {
            // Context request
            contextRequests.push_back(req);
        }
    }

    // Phase 2: Apply context chunking if needed (lines 287-310)
    if (!allContextRequestsFit) {
        setCtxRequestsChunkSize(contextRequests, ...);
        fitDraftTokens(contextRequests, ...);
    }

    // Phase 3: Sort requests (line 312)
    sortRequests(contextRequests, generationRequests, chunksPresent);

    return {contextRequests, generationRequests};
}
```

### 4.5 Context Chunking Policies

**Two Strategies**:

#### **FIRST_COME_FIRST_SERVED** (Lines 118-144)

**Algorithm**:
```cpp
// Sequential completion - complete first request fully before next
for (auto& llmReq : contextsToBeChunked) {
    SizeType32 suggestedChunkSize = llmReq->getContextRemainingLength();

    // Limit by available capacity and max context length
    SizeType32 actualChunkSize = min({
        ctxTokensCapacity.value_or(infinity),
        maxContextLength.value_or(infinity),
        suggestedChunkSize
    });

    // Align to chunk unit size (KV cache block size)
    actualChunkSize = (actualChunkSize / chunkUnitSize) * chunkUnitSize;
    llmReq->setContextChunkSize(actualChunkSize);

    // Deduct from capacity for next request
    if (ctxTokensCapacity) {
        ctxTokensCapacity -= actualChunkSize;
    }
}
```

**Characteristics**:
- First request gets all available capacity
- Subsequent requests get remaining capacity
- Optimizes for low individual latency
- May starve later requests

#### **EQUAL_PROGRESS** (Lines 85-115)

**Algorithm**:
```cpp
// Balanced round-robin chunking
SizeType32 numTokensSingleLoop = 1;
while ((!ctxTokensCapacity || numCtxTokens < ctxTokensCapacity) &&
       numTokensSingleLoop) {
    numTokensSingleLoop = 0;

    for (auto& llmReq : contextsToBeChunked) {
        SizeType32 pastChunkSize = llmReq->getContextChunkSize();
        SizeType32 suggestedChunkSize = pastChunkSize + chunkUnitSize;

        llmReq->setContextChunkSize(suggestedChunkSize);
        SizeType32 actualChunkSize = llmReq->getContextChunkSize();
        SizeType32 actualIncrement = actualChunkSize - pastChunkSize;

        // Check if fits within limits
        if ((ctxTokensCapacity && numCtxTokens + actualIncrement > limit) ||
            (maxContextLength && actualChunkSize > limit)) {
            llmReq->setContextChunkSize(pastChunkSize);  // Revert
            continue;
        }

        numCtxTokens += actualIncrement;
        numTokensSingleLoop += actualIncrement;
    }
}
```

**Characteristics**:
- All requests advance by one chunk unit per loop
- Fair progress across all requests
- Optimizes for fairness
- May increase overall latency

### 4.6 Request State Machine

**Location**: `cpp/include/tensorrt_llm/batch_manager/llmRequest.h`

**Complete State Flow**:

```
kUNKNOWN (0)
    ↓
kENCODER_INIT (1)  [for encoder-decoder models]
    ↓
kDISAGG_GENERATION_INIT (8)  [disaggregated setup]
    ↓
[SCHEDULABLE STATES]
kCONTEXT_INIT (10)  ← START (prefill/context phase)
    ↓
kGENERATION_IN_PROGRESS (13)  ← DECODE phase
    ↓
kGENERATION_TO_COMPLETE (14)
    ↓
kGENERATION_COMPLETE (20)  ← DONE
```

**Key State Queries** (Lines 1453-1599):

```cpp
bool isContextInitState()           // In prefill
bool isGenerationInProgressState()  // In decode
bool isContextFinished()            // Transitioned to decode
bool isLastContextChunk()           // Last chunk of prefill
```

**Chunking Position Tracking**:

```cpp
SizeType32 mContextCurrentPosition;  // Current position in context
SizeType32 mContextChunkSize;        // Current chunk token count

SizeType32 getContextRemainingLength() const {
    return mPromptLen - getContextCurrentPosition();
}

void moveToNextContextChunk() {
    mContextCurrentPosition += mContextChunkSize;
    mContextChunkSize = 0;  // Reset for next iteration
}
```

### 4.7 Configuration Parameters

**Capacity Scheduler Config**:

```cpp
struct CapacitySchedulerConfig {
    SizeType32 maxNumRequests;
    executor::CapacitySchedulerPolicy policy;
    // kMAX_UTILIZATION, kGUARANTEED_NO_EVICT, kSTATIC_BATCH
    bool hasKvCacheManager;
    bool twoStepsLookAhead;
};
```

**Micro-Batch Scheduler Config**:

```cpp
struct ContextChunkingConfig {
    executor::ContextChunkingPolicy chunkingPolicy;
    // kFIRST_COME_FIRST_SERVED or kEQUAL_PROGRESS
    SizeType32 chunkUnitSize;  // Typically KV cache block size
};
```

### 4.8 Code Sources

**Primary Files**:
- `cpp/tensorrt_llm/batch_manager/capacityScheduler.cpp` (542 lines)
- `cpp/tensorrt_llm/batch_manager/microBatchScheduler.cpp` (328 lines)
- `cpp/include/tensorrt_llm/batch_manager/llmRequest.h` (1800+ lines)
- `cpp/tensorrt_llm/batch_manager/trtGptModelInflightBatching.cpp` (2000+ lines)
- `cpp/tensorrt_llm/batch_manager/utils/inflightBatchingUtils.cpp` (250+ lines)

**Key Methods**:
- `CapacityScheduler::operator()()` - Request filtering by capacity
- `MicroBatchScheduler::operator()()` - Context/generation split
- `setCtxRequestsChunkSize()` - Context chunking application
- `moveFinishedContextRequestsToGeneration()` - State transition helper

---

## 5. Algorithmic Comparison

### 5.1 Scheduling Model

| System | Core Approach | Primary Constraint |
|--------|---------------|-------------------|
| **vLLM** | Unified token accounting | Token budget + sequence count |
| **SGLang** | Continuous batching | Token budget + cache-aware |
| **TensorRT-LLM** | Two-tier filtering | Capacity → Micro-batch |

### 5.2 Token Budget Enforcement

| System | Budget Type | Enforcement | Shared? |
|--------|-------------|-------------|---------|
| **vLLM** | `max_num_batched_tokens` | Hard cap per iteration | Yes (prefill + decode) |
| **SGLang** | `max_prefill_tokens` | Separate prefill limit | Separate budgets |
|  | `max_total_tokens` | Total KV cache limit | Shared |
| **TensorRT-LLM** | `maxNumTokensRuntime` | Optional micro-batch limit | Context-only |

### 5.3 Request Prioritization

| System | Queue Type | Priority Support | Ordering |
|--------|------------|------------------|----------|
| **vLLM** | Deque or Heap | Yes (optional) | (priority, arrival_time) |
| **SGLang** | List with sorting | Yes (optional) | Cache-aware or FCFS |
| **TensorRT-LLM** | Vector | No | FCFS with beam width grouping |

### 5.4 Preemption/Eviction

| System | Trigger | Victim Selection | Recovery |
|--------|---------|------------------|----------|
| **vLLM** | KV cache OOM | Lowest priority or last scheduled | Prepend to waiting queue |
| **SGLang** | Decode OOM | Shortest output length | Return to waiting queue |
| **TensorRT-LLM** | Capacity exhausted | Last scheduled (MAX_UTIL only) | Paused requests vector |

### 5.5 Prefill vs Decode Handling

| System | Separation | Strategy |
|--------|------------|----------|
| **vLLM** | **Unified** | No explicit separation, token accounting handles both |
| **SGLang** | **Explicit** | Separate prefill batch vs decode batch |
| **TensorRT-LLM** | **Explicit** | Context requests vs generation requests |

---

## 6. Prefill/Decode Optimizations

### 6.1 vLLM Optimizations

#### **Chunked Prefill**

**Configuration**:
```python
enable_chunked_prefill: bool = True
long_prefill_token_threshold: int = 0  # Max tokens per chunk
```

**Benefits**:
- Large prefills don't monopolize GPU
- Better fairness in mixed workloads
- Lower latency for decode requests

**Trade-off**: Increases prefill latency but improves overall throughput

#### **Prefix Caching**

**Automatic Application** (Lines 600-640):

```python
new_computed_blocks, num_new_local_computed_tokens = (
    self.kv_cache_manager.get_computed_blocks(request)
)
# Reduces num_new_tokens by cached tokens
num_new_tokens = request.num_tokens - num_computed_tokens
```

**Benefits**:
- Skip computation for cached tokens
- Reduces effective prefill length
- Better token budget utilization

#### **Encoder Budget Separation**

**Multimodal Support** (Lines 409-426):

```python
encoder_compute_budget = self.max_num_encoder_input_tokens
# Separate budget for vision/audio encoders
# Prevents multimodal from blocking text-only requests
```

### 6.2 SGLang Optimizations

#### **Cache-Aware Scheduling**

**LPM Policy** (schedule_policy.py, lines 133-156):

```python
# Sort waiting queue by prefix match length
waiting_queue.sort(key=lambda r: -len(r.prefix_indices))
```

**Benefits**:
- High cache-hit requests scheduled first
- Reduces actual compute needed
- Better GPU utilization

**Impact**: 20-40% latency reduction in chat workloads

#### **Mixed Chunking**

**Configuration**:
```python
enable_mixed_chunk: bool = True
chunked_prefill_size: int = 512
```

**Algorithm** (scheduler.py, lines 2193):

```python
if self.is_mixed_chunk:
    # Mix prefill chunk with decode requests
    self.running_batch.prepare_for_decode()
    new_batch.mix_with_running(self.running_batch)
    new_batch.forward_mode = ForwardMode.MIXED
```

**Benefits**:
- Prefill and decode in same forward pass
- Better GPU utilization
- Reduced iteration overhead

#### **Retraction Mechanism**

**Graceful Degradation** (schedule_batch.py, lines 1847-1905):

```python
# On decode OOM: remove shortest sequences
sorted_indices.sort(key=lambda i: len(self.reqs[i].output_ids))
while not self.check_decode_mem():
    retracted_reqs.append(self.reqs[sorted_indices.pop()])
```

**Benefits**:
- No request failures on OOM
- Automatically adjusts to available memory
- Updates token ratio for better future estimates

### 6.3 TensorRT-LLM Optimizations

#### **Context Chunking**

**EQUAL_PROGRESS Policy**:

**Benefits**:
- All prefills advance together
- Fair progress across requests
- Prevents starvation

**FIRST_COME_FIRST_SERVED Policy**:

**Benefits**:
- Lower individual latency
- Complete high-priority requests first
- Simpler logic

#### **KV Cache Block Reuse**

**Optimization** (capacityScheduler.cpp, lines 34-123):

```cpp
// Skip requests that would duplicate context blocks
if (newlyContributedContextBlocks.count(newContextBlock) > 0) {
    return true;  // Skip - we'll reuse that block
}
```

**Benefits**:
- Identifies duplicate work
- Better cache hit rate
- Reduced context computation

#### **Beam Width Grouping**

**Constraint** (microBatchScheduler.cpp, lines 265-275):

```cpp
if (scheduledBeamWidth != reqBeamWidth) {
    continue;  // Skip - different beam width
}
```

**Benefits**:
- Uniform computation patterns
- Better CUDA kernel efficiency
- Simplified batch management

**Trade-off**: May delay some decode requests

---

## 7. Configuration Guide

### 7.1 vLLM Configuration

**Recommended Settings**:

```python
from vllm import LLM
from vllm.config import SchedulerConfig

# For throughput-optimized workloads
scheduler_config = SchedulerConfig(
    max_num_batched_tokens=2048,      # Higher = better throughput
    max_num_seqs=128,                 # Balance batch size
    enable_chunked_prefill=True,      # Fairness
    long_prefill_token_threshold=512, # Chunk size
)

# For latency-optimized workloads
scheduler_config = SchedulerConfig(
    max_num_batched_tokens=512,       # Lower = better latency
    max_num_seqs=32,                  # Fewer concurrent requests
    enable_chunked_prefill=False,     # Complete prefills faster
)
```

**Tuning Guidelines**:

| Workload | max_num_batched_tokens | max_num_seqs | chunked_prefill |
|----------|----------------------|--------------|-----------------|
| **Chat (short prompts)** | 1024-2048 | 64-128 | True |
| **Long documents** | 4096-8192 | 32-64 | True |
| **Code generation** | 2048-4096 | 64-128 | True |
| **Low latency** | 512-1024 | 16-32 | False |

### 7.2 SGLang Configuration

**Recommended Settings**:

```bash
# Throughput-optimized
python -m sglang.launch_server \
    --model meta-llama/Llama-3.1-8B \
    --max-total-tokens 8192 \
    --max-prefill-tokens 4096 \
    --max-running-requests 128 \
    --schedule-policy lpm \
    --enable-mixed-chunk \
    --chunked-prefill-size 512

# Latency-optimized
python -m sglang.launch_server \
    --model meta-llama/Llama-3.1-8B \
    --max-total-tokens 4096 \
    --max-prefill-tokens 2048 \
    --max-running-requests 32 \
    --schedule-policy fcfs \
    --disable-mixed-chunk
```

**Tuning Guidelines**:

| Workload | max_total_tokens | max_prefill_tokens | schedule_policy |
|----------|-----------------|-------------------|-----------------|
| **Chat** | 8192-16384 | 4096-8192 | lpm |
| **Completion** | 4096-8192 | 2048-4096 | fcfs |
| **Multi-turn** | 16384-32768 | 8192-16384 | dfs-weight |

### 7.3 TensorRT-LLM Configuration

**Recommended Settings**:

```python
from tensorrt_llm import LLM
from tensorrt_llm.executor import CapacitySchedulerPolicy, ContextChunkingPolicy

# Throughput-optimized
llm = LLM(
    model="meta-llama/Llama-3.1-8B",
    capacity_scheduler_policy=CapacitySchedulerPolicy.MAX_UTILIZATION,
    context_chunking_policy=ContextChunkingPolicy.EQUAL_PROGRESS,
    max_num_tokens=4096,
)

# Latency-optimized
llm = LLM(
    model="meta-llama/Llama-3.1-8B",
    capacity_scheduler_policy=CapacitySchedulerPolicy.GUARANTEED_NO_EVICT,
    context_chunking_policy=ContextChunkingPolicy.FIRST_COME_FIRST_SERVED,
    max_num_tokens=2048,
)
```

---

## 8. Best Practices

### 8.1 Token Budget Sizing

**vLLM**:
- Set `max_num_batched_tokens` to ~50-75% of available KV cache
- Ensure `max_num_batched_tokens >= max_num_seqs` (validated)
- For long prompts: increase `max_num_batched_tokens`, enable chunked prefill

**SGLang**:
- Set `max_total_tokens` to total KV cache capacity
- Set `max_prefill_tokens` to 25-50% of `max_total_tokens`
- Use cache-aware policies (LPM) for chat workloads

**TensorRT-LLM**:
- Use `MAX_UTILIZATION` for maximum throughput
- Use `GUARANTEED_NO_EVICT` for predictable latency
- Set `maxNumTokens` conservatively for chunking

### 8.2 Sequence Budget Sizing

**vLLM**:
- Start with `max_num_seqs = 128` for balanced workloads
- Reduce to 32-64 for low-latency applications
- Increase to 256+ for high-throughput batch inference

**SGLang**:
- Set `max_running_requests` based on average output length
- Higher for chat (128-256), lower for long generation (32-64)

**TensorRT-LLM**:
- Controlled by capacity scheduler automatically
- Override with `maxNumRequests` if needed

### 8.3 Prefill Strategy Selection

**When to Use Chunked Prefill**:
- Mixed workloads (prefill + decode)
- Long prompts (>1K tokens)
- Fairness requirements
- Chat applications

**When to Disable Chunked Prefill**:
- Prefill-only workloads
- Low-latency requirements
- Short prompts (<512 tokens)
- Batch inference

### 8.4 Monitoring Metrics

**vLLM**:
```bash
# Enable metrics
--cudagraph-metrics

# Key metrics to watch:
# - num_preemptions (should be low)
# - token_budget_utilization (should be high)
# - avg_prefill_tokens_per_request
```

**SGLang**:
```bash
# Enable metrics
--enable-metrics

# Key metrics:
# - cache_hit_rate (LPM should be >50%)
# - retraction_count (should be low)
# - mixed_batch_ratio
```

**TensorRT-LLM**:
- Monitor `numCtxRequests` vs `numGenRequests`
- Track `pausedRequests` count (MAX_UTIL only)
- Measure chunking efficiency

---

## 9. Summary

### 9.1 Key Differences

| Aspect | vLLM | SGLang | TensorRT-LLM |
|--------|------|--------|--------------|
| **Model** | Unified token accounting | Continuous batching | Two-tier capacity |
| **Constraints** | Token + sequence budget | Token + prefill budget | Capacity + micro-batch |
| **Prefill/Decode** | Unified | Explicit separation | Explicit separation |
| **Priority** | Optional | Optional (cache-aware) | No |
| **Preemption** | On KV cache OOM | On decode OOM | On capacity (optional) |
| **Chunking** | Token threshold | Fixed chunk size | Policy-based |
| **Cache-Aware** | Automatic prefix cache | Scheduling policy (LPM) | Block reuse skip |

### 9.2 Best Use Cases

**Choose vLLM when**:
- Need unified prefill/decode handling
- Want automatic prefix caching
- Require priority-based scheduling
- OpenAI-compatible API desired

**Choose SGLang when**:
- Heavy prefix sharing (chat, multi-turn)
- Need cache-aware scheduling
- Want maximum flexibility
- Willing to tune configuration

**Choose TensorRT-LLM when**:
- Maximum performance on NVIDIA GPUs
- Production deployment with TensorRT
- Prefer two-tier resource management
- Need predictable latency (GUARANTEED_NO_EVICT)

All three systems demonstrate sophisticated scheduling with different trade-offs optimized for their target use cases.
