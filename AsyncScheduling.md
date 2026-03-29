# Async Scheduling in LLM Inference Systems

A comprehensive comparison of async scheduling implementations in vLLM, SGLang, and TensorRT-LLM.

---

## Table of Contents
1. [vLLM Async Scheduling](#1-vllm-async-scheduling)
2. [SGLang Async Scheduling](#2-sglang-async-scheduling)
3. [TensorRT-LLM Async Scheduling](#3-tensorrt-llm-async-scheduling)
4. [Algorithmic Differences](#4-algorithmic-differences)
5. [Comparative Summary](#5-comparative-summary)

---

## 1. vLLM Async Scheduling

### 1.1 Architecture Overview

vLLM uses a **multi-process architecture** with async coordination:

```
AsyncLLM (Python API)
    ↓ (ZMQ IPC)
EngineCore (Background Process)
    ├── Scheduler → Makes scheduling decisions
    ├── KVCacheManager → Memory allocation
    └── ModelExecutor → GPU execution
    ↓ (ZMQ IPC)
AsyncLLM output_handler → Distributes results
```

### 1.2 Key Components

#### **Code Locations**
- **Async API**: `vllm/v1/engine/async_llm.py`
- **Engine Core**: `vllm/v1/engine/core.py`
- **Scheduler**: `vllm/v1/core/sched/scheduler.py`
- **Async Scheduler**: `vllm/v1/core/sched/async_scheduler.py`
- **Request Queue**: `vllm/v1/core/sched/request_queue.py`

#### **Class Hierarchy**
```python
AsyncLLM                          # User-facing async API
  ↓
EngineCoreClient                  # IPC client to EngineCore
  ↓ (ZMQ DEALER socket)
EngineCore                        # Background process
  ↓
AsyncScheduler / Scheduler        # Scheduling logic
```

### 1.3 Request Queueing

**Two Queue Implementations** (`request_queue.py`):

#### FCFS (First-Come-First-Served)
```python
class FCFSRequestQueue(deque[Request], RequestQueue):
    def add_request(self, request: Request) -> None:
        self.append(request)        # O(1) tail append

    def pop_request(self) -> Request:
        return self.popleft()       # O(1) head pop
```

#### Priority Queue
```python
class PriorityRequestQueue(RequestQueue):
    def __init__(self) -> None:
        self._heap: list[Request] = []

    def add_request(self, request: Request) -> None:
        heapq.heappush(self._heap, request)  # Min-heap

    def pop_request(self) -> Request:
        return heapq.heappop(self._heap)
```

**Request Ordering**: `(priority, arrival_time)` - lower priority value = higher urgency

### 1.4 Scheduling Decisions

#### **Token Budget Model** (scheduler.py)

The core algorithm uses **token budgets** rather than batch counts:

```python
def schedule(self) -> SchedulerOutput:
    token_budget = self.max_num_scheduled_tokens

    # Step 1: Schedule RUNNING requests (continuation)
    for request in self.running:
        num_new_tokens = min(
            request.num_tokens_with_spec - request.num_computed_tokens,
            token_budget
        )

        # Allocate KV cache blocks
        new_blocks = self.kv_cache_manager.allocate_slots(
            request, num_new_tokens, num_lookahead_tokens
        )

        if new_blocks is None:
            # OUT OF MEMORY: Preempt lowest-priority running request
            preempted = max(
                self.running,
                key=lambda r: (r.priority, r.arrival_time)
            )
            self._preempt_request(preempted)
            continue

        token_budget -= num_new_tokens

    # Step 2: Schedule WAITING requests (new work)
    while self.waiting and token_budget > 0:
        request = self.waiting.pop_request()

        # Check prefix cache for KV reuse
        new_computed_blocks, num_cached_tokens = (
            self.kv_cache_manager.get_computed_blocks(request)
        )

        # Allocate remaining slots
        num_new_tokens = min(
            request.num_prompt_tokens - num_cached_tokens,
            token_budget
        )
        new_blocks = self.kv_cache_manager.allocate_slots(...)

        if new_blocks:
            self.running.append(request)
            token_budget -= num_new_tokens
        else:
            self.waiting.prepend_request(request)
            break

    return scheduler_output
```

#### **Scheduling Constraints**

1. **Token Budget**: `max_num_scheduled_tokens` per iteration
2. **Batch Size**: `max_num_running_reqs` concurrent requests
3. **Model Length**: Cannot exceed `max_model_len - 1`
4. **Long Prefill Threshold**: Cap prefill tokens to allow decode progress

### 1.5 Async-Specific Features

#### **Output Placeholders** (async_scheduler.py)

The key async optimization: pre-allocate space for future tokens.

```python
class AsyncScheduler(Scheduler):
    def _update_after_schedule(self, scheduler_output: SchedulerOutput):
        for req_id in scheduler_output.num_scheduled_tokens:
            request = self.requests[req_id]

            # Get speculative decode tokens for this request
            spec_tokens = scheduler_output.scheduled_spec_decode_tokens.get(
                req_id, ()
            )

            # Add placeholders: 1 main token + N spec tokens
            num_placeholders = 1 + len(spec_tokens)
            request.num_output_placeholders += num_placeholders

            # Reusable placeholder IDs (avoid allocating new lists)
            request.spec_token_ids = self._spec_token_placeholders

    def _update_request_with_output(self, request, new_token_ids):
        # Reduce placeholders as tokens materialize
        request.num_output_placeholders -= len(new_token_ids)

        # Only cache if request wasn't preempted
        if status_before == RequestStatus.RUNNING:
            self.kv_cache_manager.cache_blocks(
                request,
                request.num_computed_tokens - request.num_output_placeholders
            )

        return new_token_ids, stopped
```

**Why Output Placeholders?**
- Sync scheduling: Must schedule again after each token
- Async scheduling: Pre-allocate N tokens → avoid re-scheduling overhead
- Enables worker to proceed independently without waiting for scheduler

#### **Deferred Scheduling**

```python
# Skip scheduling if we've already allocated enough placeholders
if (request.num_output_placeholders > 0
    and request.num_computed_tokens + 2 - request.num_output_placeholders
        >= request.num_prompt_tokens + request.max_tokens):
    continue  # Don't schedule this request yet
```

### 1.6 Async API Flow

#### **AsyncLLM.generate()** (async_llm.py)

```python
async def generate(
    self, prompt, sampling_params, request_id
) -> AsyncGenerator[RequestOutput, None]:

    # Add request to EngineCore via ZMQ
    request_queue = await self.add_request(request_id, prompt, sampling_params)

    # Yield outputs as they arrive
    while not finished:
        # Try non-blocking first (avoid asyncio overhead)
        output = request_queue.get_nowait()
        if output is None:
            # Blocking wait if nothing available
            output = await request_queue.get()

        yield output

        if output.finished:
            finished = True
            break
```

#### **Background Output Handler**

```python
async def output_handler():
    """Continuously processes outputs from EngineCore."""
    while True:
        # Receive batch of outputs from EngineCore (via ZMQ)
        outputs = await self.engine_core_client.get_output_async()

        # Process on CPU (detokenization, formatting)
        processed_outputs = self.output_processor.process_outputs(outputs)

        # Split large batches to avoid blocking event loop
        for chunk_start in range(0, len(processed_outputs), chunk_size):
            chunk = processed_outputs[chunk_start:chunk_start + chunk_size]

            # Distribute to per-request queues
            for output in chunk:
                request_queue = self.request_outputs[output.request_id]
                request_queue.put_nowait(output)

            # Yield to event loop between chunks
            if chunk_start + chunk_size < len(processed_outputs):
                await asyncio.sleep(0)
```

### 1.7 Preemption Strategy

When KV cache is full:

```python
while allocation_failed:
    if self.policy == SchedulingPolicy.PRIORITY:
        # Preempt LOWEST priority running request
        preempted = max(
            self.running,
            key=lambda r: (r.priority, r.arrival_time)  # Higher = worse
        )
    else:  # FCFS
        # Preempt most recent request
        preempted = self.running.pop()

    # Free KV cache blocks
    self.kv_cache_manager.free(preempted.req_id)
    preempted.status = RequestStatus.PREEMPTED

    # Prepend to waiting queue for resumption
    self.waiting.prepend_request(preempted)

    # Retry allocation
    new_blocks = self.kv_cache_manager.allocate_slots(...)
```

### 1.8 Configuration Parameters

```python
class SchedulerConfig:
    # Async mode
    async_scheduling: bool = True

    # Capacity limits
    max_num_seqs: int = 256                    # Max concurrent requests
    max_num_batched_tokens: int = 512          # Total token budget
    max_num_scheduled_tokens: Optional[int]    # Per-step token budget

    # Prefill control
    long_prefill_token_threshold: int = 0      # Cap prefill per step
    enable_chunked_prefill: bool = False       # Chunk large prompts

    # Memory
    enable_prefix_caching: bool = False        # KV cache reuse

    # Policy
    policy: str = "fcfs"                       # "fcfs" or "priority"
```

---

## 2. SGLang Async Scheduling

### 2.1 Architecture Overview

SGLang uses a **three-process architecture** with ZMQ IPC:

```
HTTP Server + TokenizerManager (Process 1)
    ↓ (ZMQ PUSH)
Scheduler (Process 2)
    ├── Waiting Queue
    ├── Running Batch
    ├── RadixCache (Prefix Tree)
    └── GPU Workers (TP/PP ranks)
    ↓ (ZMQ PUSH)
DetokenizerManager (Process 1)
```

### 2.2 Key Components

#### **Code Locations**
- **Scheduler**: `sglang/srt/managers/scheduler.py`
- **Schedule Policy**: `sglang/srt/managers/schedule_policy.py`
- **Schedule Batch**: `sglang/srt/managers/schedule_batch.py`
- **RadixCache**: `sglang/srt/mem_cache/radix_cache.py`
- **Prefill Disaggregation**: `sglang/srt/disaggregation/prefill.py`
- **Decode Disaggregation**: `sglang/srt/disaggregation/decode.py`

### 2.3 Request Queueing

#### **Request Reception** (scheduler.py)

```python
def recv_requests(self) -> List[TokenizedGenerateReqInput]:
    """Non-blocking ZMQ reception from TokenizerManager."""

    # Only rank 0 receives
    if self.pp_rank == 0 and self.attn_tp_rank == 0:
        recv_reqs = []
        while True:
            try:
                # Non-blocking receive
                if self.recv_limit_reached(len(recv_reqs)):
                    break
                recv_req = self.recv_from_tokenizer.recv_pyobj(zmq.NOBLOCK)
                recv_reqs.append(recv_req)
            except zmq.ZMQError:
                break  # No more messages

        # Broadcast to other TP ranks
        if self.tp_size > 1:
            broadcast_pyobj(recv_reqs, self.tp_group)

        return recv_reqs
```

**Configurable receive limit**: `SGLANG_SCHEDULER_MAX_RECV_PER_POLL` (default 100)

#### **Queue Management** (scheduler.py:1679-1701)

```python
def _add_request_to_queue(self, req: Req, is_retracted: bool = False):
    # Normal mode: Add to waiting queue
    if self.disaggregation_mode == DisaggregationMode.NULL:
        # Validate/set priority
        if not self._set_or_validate_priority(req):
            return

        # Abort if queue full and lower priority
        if self._abort_on_queued_limit(req):
            return

        # Prefetch from hierarchical cache storage
        self._prefetch_kvcache(req)

        # Simple append (FIFO with priority sorting)
        self.waiting_queue.append(req)
        req.time_stats.wait_queue_entry_time = time.perf_counter()

    # Disaggregation modes have specialized queues
    elif self.disaggregation_mode == DisaggregationMode.PREFILL:
        self.disagg_prefill_bootstrap_queue.add(req, ...)
    elif self.disaggregation_mode == DisaggregationMode.DECODE:
        self.disagg_decode_prealloc_queue.add(req, is_retracted)
```

### 2.4 Scheduling Decisions

#### **Cache-Aware Priority Calculation** (schedule_policy.py)

```python
class SchedulePolicy:
    class CacheAwarePolicy(Enum):
        LPM = "lpm"                    # Longest Prefix Match
        DFS_WEIGHT = "dfs-weight"      # Depth-First Search weighting

    class CacheAgnosticPolicy(Enum):
        FCFS = "fcfs"                  # First Come First Serve
        LOF = "lof"                    # Longest Output First
        RANDOM = "random"
        ROUTING_KEY = "routing-key"    # Match routing keys

    def calc_priority(self, waiting_queue, running_batch):
        """Calculate and sort waiting queue by priority."""

        if self.policy == CacheAwarePolicy.LPM:
            # Match prefixes against RadixCache
            self._compute_prefix_matches(waiting_queue)

            # Sort by longest prefix match
            waiting_queue.sort(
                key=lambda r: -len(r.prefix_indices)
                    if r.rid not in temporary_deprioritized
                    else float("inf")
            )

        elif self.policy == CacheAwarePolicy.DFS_WEIGHT:
            # Build radix tree from waiting queue
            for req in waiting_queue:
                self.waiting_queue_radix_tree.insert(req.origin_input_ids)

            # Sort by DFS weight (number of descendants)
            waiting_queue.sort(key=lambda r: -r.dfs_weight)

        elif self.policy == CacheAgnosticPolicy.LOF:
            # Sort by longest output first
            waiting_queue.sort(key=lambda r: -r.sampling_params.max_new_tokens)
```

#### **Prefix Matching Algorithm** (schedule_policy.py:76-156)

```python
def _compute_prefix_matches(self, waiting_queue, policy):
    """Match request prefixes against RadixCache."""
    temporary_deprioritized = set()

    for req in waiting_queue:
        prefix_ids = req.origin_input_ids + req.output_ids
        extra_key = req.extra_key  # LoRA ID, cache salt

        # Match against persistent RadixCache
        match_result = self.tree_cache.match_prefix(
            MatchPrefixParams(
                key=RadixKey(token_ids=prefix_ids, extra_key=extra_key)
            )
        )

        # Store matched indices and nodes
        req.prefix_indices = match_result.device_indices
        req.last_node = match_result.last_device_node
        req.last_host_node = match_result.last_host_node
        req.host_hit_length = match_result.host_hit_length

        # In-batch prefix caching: deprioritize if better to wait
        if len(req.prefix_indices) <= IN_BATCH_PREFIX_CACHING_CHECK_THRESHOLD:
            # Check if waiting queue has many similar requests
            in_batch_matches = self.waiting_queue_radix_tree.match_prefix(...)

            if len(in_batch_matches) >= DEPRIORITIZE_THRESHOLD:
                # Better to let other requests with same prefix go first
                temporary_deprioritized.add(req.rid)

    return temporary_deprioritized
```

**In-batch Prefix Caching**: If multiple waiting requests share a long prefix, schedule one first so others can reuse its KV cache.

### 2.5 Batch Formation

#### **PrefillAdder Algorithm** (schedule_policy.py:371-828)

The core batching logic:

```python
class PrefillAdder:
    def __init__(
        self,
        page_size,
        tree_cache,
        token_to_kv_pool_allocator,
        running_batch,
        new_token_ratio,         # Estimated ratio of new tokens
        max_prefill_tokens,      # Input token budget
        chunked_prefill_size,    # Max tokens per chunk
        running_bs,              # Current batch size
        priority_threshold,      # Preemption threshold
    ):
        # Token budgets
        self.rem_input_tokens = max_prefill_tokens
        self.rem_chunk_tokens = chunked_prefill_size

        # Calculate total available tokens (conservative estimate)
        available_and_evictable = (
            token_to_kv_pool_allocator.available_size()
            + tree_cache.evictable_size()
        )
        self.rem_total_tokens = available_and_evictable - offset

    def add_one_req(self, req: Req, has_chunked_req: bool) -> AddReqResult:
        """Try to add one request to the batch."""

        # Check if batch is full
        if len(self.can_run_list) >= max_running_requests:
            return AddReqResult.NO_SLOT

        # Calculate tokens needed
        req_extend_len = req.extend_input_len  # Tokens to process this step
        req_total_len = req.input_len + req.max_new_tokens  # Total

        # Conservative estimation: assume some tokens are new
        estimated_new_tokens = int(req_total_len * self.new_token_ratio)

        # Check token budgets
        if estimated_new_tokens > self.rem_total_tokens:
            return AddReqResult.NO_TOKEN

        if req_extend_len > self.rem_input_tokens:
            return AddReqResult.NO_TOKEN

        # For chunked prefill: may truncate request
        if req_extend_len > self.rem_chunk_tokens:
            if has_chunked_req:
                return AddReqResult.NO_TOKEN

            # Truncate to chunk boundary
            req.set_extend_input_len(self.rem_chunk_tokens)
            req.fill_ids = req.fill_ids[:len(req.prefix_indices) + self.rem_chunk_tokens]
            self.new_chunked_req = req
            req_extend_len = self.rem_chunk_tokens

        # Add to batch
        self.can_run_list.append(req)

        # Update budgets
        self.rem_input_tokens -= req_extend_len
        self.rem_total_tokens -= estimated_new_tokens
        if req_extend_len > 0:
            self.rem_chunk_tokens -= req_extend_len

        return AddReqResult.CONTINUE

    def preempt_to_schedule(self, req: Req) -> bool:
        """Try to preempt a running request to make room."""

        if not self.priority_scheduling_preemption:
            return False

        # Find lowest-priority running request
        victim = min(
            self.running_batch.reqs,
            key=lambda r: (r.priority, -r.arrival_time)
        )

        # Only preempt if new request has higher priority
        if req.priority + self.priority_threshold > victim.priority:
            # Retract victim from running batch
            self.running_batch.retract_decode_req(victim)
            return True

        return False
```

#### **Batch Formation Entry Point** (scheduler.py:1960-2202)

```python
def _get_new_batch_prefill_raw(self) -> Optional[ScheduleBatch]:
    """Form a new prefill batch from waiting queue."""

    # Calculate priority for all waiting requests
    self.policy.calc_priority(self.waiting_queue, self.running_batch)

    # Create PrefillAdder with token budgets
    adder = PrefillAdder(
        self.page_size,
        self.tree_cache,
        self.token_to_kv_pool_allocator,
        self.running_batch,
        self.new_token_ratio,
        self.max_prefill_tokens,
        chunked_prefill_size,
        running_bs,
        self.priority_scheduling_preemption_threshold,
    )

    # Greedily add requests from waiting queue (already sorted)
    for req in self.waiting_queue:
        # Check LoRA compatibility
        if self.enable_lora and req.lora_id not in running_loras:
            if not self._check_lora_batch(...):
                continue  # Skip this request

        # Try to add request
        res = adder.add_one_req(req, has_chunked_req=(self.chunked_req is not None))

        if res == AddReqResult.CONTINUE:
            continue  # Successfully added
        elif res == AddReqResult.NO_TOKEN:
            self.running_batch.batch_is_full = True
            break
        elif res == AddReqResult.NO_SLOT:
            # Try preemption
            if self.try_preemption and adder.preempt_to_schedule(req):
                res = adder.add_one_req(req, ...)
                if res == AddReqResult.CONTINUE:
                    continue
            break

    # Remove scheduled requests from waiting queue
    self.waiting_queue = [r for r in self.waiting_queue if r not in adder.can_run_list]

    # Create ScheduleBatch
    if adder.can_run_list:
        return ScheduleBatch.init_new(adder.can_run_list, ...)
    return None
```

### 2.6 Event Loop Variants

#### **Normal Event Loop** (scheduler.py:1107-1145)

```python
@DynamicGradMode()
def event_loop_normal(self):
    """Standard synchronous event loop."""
    while True:
        # 1. Receive new requests (non-blocking ZMQ)
        recv_reqs = self.recv_requests()
        self.process_input_requests(recv_reqs)

        # 2. Get next batch to run
        batch = self.get_next_batch_to_run()
        self.cur_batch = batch

        # 3. Execute batch and process results immediately
        if batch:
            result = self.run_batch(batch)
            self.process_batch_result(batch, result)
        else:
            self.self_check_during_idle()

        self.last_batch = batch
```

#### **Overlap Event Loop** (scheduler.py:1147-1211)

```python
@DynamicGradMode()
def event_loop_overlap(self):
    """Event loop that overlaps CPU and GPU work."""
    self.result_queue = deque()  # Queue of (batch, result) pairs

    while True:
        # 1. Receive requests (CPU)
        recv_reqs = self.recv_requests()
        self.process_input_requests(recv_reqs)

        # 2. Get next batch (CPU)
        batch = self.get_next_batch_to_run()
        self.cur_batch = batch

        # 3. Determine if overlap should be disabled for this batch
        disable_overlap = self.is_disable_overlap_for_batch(batch)

        # 4. If disabling, process last result NOW
        if disable_overlap and self.last_batch:
            tmp_batch, tmp_result = self.result_queue.popleft()
            self.process_batch_result(tmp_batch, tmp_result)

        # 5. Launch current batch on GPU (async)
        if batch:
            batch_result = self.run_batch(batch)  # Non-blocking
            self.result_queue.append((batch.copy(), batch_result))

        # 6. Process LAST batch result while GPU runs (CPU)
        if self.last_batch:
            if not disable_overlap:
                tmp_batch, tmp_result = self.result_queue.popleft()
                self.process_batch_result(tmp_batch, tmp_result)
        elif batch is None:
            self.self_check_during_idle()

        # 7. Launch delayed sampling if needed
        if self.is_generation:
            self.launch_batch_sample_if_needed(batch_result)

        self.last_batch = batch
```

**Overlap Pattern**:
```
Iteration N:   [CPU: Process batch N-1] [GPU: Run batch N]
Iteration N+1: [CPU: Process batch N]   [GPU: Run batch N+1]
```

### 2.7 Prefill-Decode Disaggregation

When disaggregation is enabled, SGLang separates prefill and decode workers.

#### **Prefill Worker Lifecycle** (disaggregation/prefill.py:238-319)

```python
class PrefillBootstrapQueue:
    """
    Three-stage lifecycle:
    1. Bootstrap Queue: Initialize KV sender, handshake with decode worker
    2. Waiting Queue: Regular prefill scheduling
    3. Inflight Queue: Poll for KV transfer completion
    """

    def add(self, req: Req, num_kv_heads: int):
        # Create KV sender for this request
        req.disagg_kv_sender = kv_sender_class(
            mgr=self.kv_manager,
            bootstrap_addr=f"{req.bootstrap_host}:{self.bootstrap_port}",
            bootstrap_room=req.bootstrap_room,
        )

        # Set max_new_tokens = 1 (prefill outputs one token)
        self._process_req(req)

        self.queue.append(req)

    def pop_bootstrapped(self) -> List[Req]:
        """Pop requests that finished bootstrap handshake."""

        # Poll all senders (with all_reduce for coordination)
        polls = poll_and_all_reduce(
            [req.disagg_kv_sender for req in self.queue],
            self.gloo_group
        )

        bootstrapped_reqs = []
        for req, poll in zip(self.queue, polls):
            if poll == KVPoll.Bootstrapping:
                continue  # Still handshaking
            elif poll == KVPoll.Failed:
                prepare_abort(req, error_message)
                continue

            # KVPoll.WaitingForInput: Ready for prefill
            req.metadata_buffer_index = allocator.alloc()
            bootstrapped_reqs.append(req)

        # Remove bootstrapped from queue
        self.queue = [r for r in self.queue if r not in bootstrapped_reqs]
        return bootstrapped_reqs
```

#### **Decode Worker Lifecycle** (disaggregation/decode.py:62-168)

```python
class DecodePreallocQueue:
    """
    Four-stage lifecycle:
    1. PreallocQueue: Initialize KV receiver, preallocate cache
    2. TransferQueue: Poll for KV transfer completion
    3. WaitingQueue: Construct PrebuiltExtendBatch
    4. RunningBatch: Merge into running batch for decoding
    """

    def add(self, req: Req, is_retracted: bool = False):
        # Create KV receiver
        kv_receiver = kv_receiver_class(
            mgr=self.kv_manager,
            bootstrap_addr=f"{req.bootstrap_host}:{req.bootstrap_port}",
            bootstrap_room=req.bootstrap_room,
            prefill_dp_rank=req.data_parallel_rank,
        )

        self.queue.append(DecodeRequest(req=req, kv_receiver=kv_receiver))

    def pop_preallocated(self) -> List[DecodeRequest]:
        """Pop requests that finished cache preallocation."""

        preallocated_reqs = []
        for decode_req in self.queue:
            # Try to preallocate KV cache
            success = self._try_preallocate_kv_cache(decode_req)

            if success:
                preallocated_reqs.append(decode_req)

        # Move to transfer queue
        self.transfer_queue.extend(preallocated_reqs)
        self.queue = [r for r in self.queue if r not in preallocated_reqs]

        return preallocated_reqs
```

#### **Disaggregation Event Loops**

**Prefill Worker** (disaggregation/prefill.py:361-398):
```python
def event_loop_normal_disagg_prefill(self: Scheduler):
    while True:
        # Receive requests
        recv_reqs = self.recv_requests()
        self.process_input_requests(recv_reqs)

        # Move bootstrapped requests to waiting queue
        self.waiting_queue.extend(
            self.disagg_prefill_bootstrap_queue.pop_bootstrapped()
        )

        # Get and run prefill batch
        batch = self.get_next_disagg_prefill_batch_to_run()
        if batch:
            result = self.run_batch(batch)
            self.process_batch_result_disagg_prefill(batch, result)

        # Process inflight KV transfers
        self.process_disagg_prefill_inflight_queue()
```

**Decode Worker** (disaggregation/decode.py:413-446):
```python
def event_loop_normal_disagg_decode(self: Scheduler):
    while True:
        # Receive requests
        recv_reqs = self.recv_requests()
        self.process_input_requests(recv_reqs)

        # Poll prealloc and transfer queues
        prebuilt_batch = self.get_prebuilt_batch()
        if prebuilt_batch:
            self.waiting_queue.extend(prebuilt_batch.reqs)

        # Get and run decode batch (merged)
        batch = self.get_next_batch_to_run()
        if batch:
            result = self.run_batch(batch)
            self.process_batch_result(batch, result)
```

### 2.8 Configuration Parameters

```python
# Schedule policy
schedule_policy: str = "lpm"  # "lpm", "fcfs", "lof", "dfs-weight"

# Capacity limits
max_running_requests: int = 1024
max_total_tokens: int = 8192
max_prefill_tokens: int = 4096
chunked_prefill_size: int = 512

# Cache
enable_prefix_cache: bool = True
disable_radix_cache: bool = False
new_token_ratio: float = 0.7  # Conservative estimate for unknown tokens

# Priority scheduling
enable_priority_scheduling: bool = False
priority_scheduling_preemption: bool = False
priority_scheduling_preemption_threshold: float = 0.0

# Disaggregation
disaggregation_mode: str = "null"  # "null", "prefill", "decode"

# Event loop
enable_overlap_schedule: bool = True
```

---

## 3. TensorRT-LLM Async Scheduling

### 3.1 Architecture Overview

TensorRT-LLM uses a **multi-tier async queue architecture**:

```
Python LLM API (asyncio)
    ↓
GenerationExecutor (request/result queues)
    ↓ (IPC Queues)
Worker Processes
    ↓
C++ Batch Manager
    ├── CapacityScheduler → Select fitting requests
    ├── MicroBatchScheduler → Context/generation split
    └── TrtGptModelInflightBatching → Async execution
```

### 3.2 Key Components

#### **Code Locations**
- **Python API**: `tensorrt_llm/llmapi/llm.py`
- **Executor**: `tensorrt_llm/executor/executor.py`, `proxy.py`
- **AsyncQueue**: `tensorrt_llm/llmapi/utils.py:364`
- **Batch Manager**: `cpp/tensorrt_llm/batch_manager/`
- **Capacity Scheduler**: `cpp/tensorrt_llm/batch_manager/capacityScheduler.h`
- **Micro Batch Scheduler**: `cpp/tensorrt_llm/batch_manager/microBatchScheduler.h`
- **Inflight Batching**: `cpp/tensorrt_llm/batch_manager/trtGptModelInflightBatching.cpp`

### 3.3 Request Queueing

#### **Python-Level Async API** (llmapi/llm.py)

```python
class LLM:
    def generate_async(
        self,
        inputs: Union[str, List[str]],
        sampling_params: Optional[SamplingParams] = None,
        scheduling_params: Optional[SchedulingParams] = None,
    ) -> Union[GenerationResult, List[GenerationResult]]:
        """
        Non-blocking async submission.
        Returns GenerationResult "future" immediately.
        """

        # Create request with ID
        request = GenerationRequest(
            inputs=inputs,
            sampling_params=sampling_params or self.default_sampling_params,
            scheduling_params=scheduling_params,
        )

        # Submit to executor (non-blocking)
        result = self._executor.submit(request)

        # Returns immediately
        return result
```

#### **Executor Request Submission** (executor/executor.py)

```python
class GenerationExecutor:
    def submit(self, request: GenerationRequest) -> GenerationResult:
        """Submit request and return result future."""

        # Assign unique ID
        request.set_id(self._get_next_client_id())

        # Create result future
        result = GenerationResult(
            request=request,
            background_error_handler=self._error_handler,
            streaming=request.streaming,
        )

        # Store result for later lookup
        self._results[request.id] = result

        # Return immediately (non-blocking)
        return result
```

#### **IPC Queue Architecture** (executor/proxy.py)

```python
class GenerationExecutorProxy(GenerationExecutor):
    def __init__(self, ...):
        # Three IPC queues
        self.request_queue = IpcQueue(...)      # Main → Workers
        self.result_queue = FusedIpcQueue(...)  # Workers → Main
        self.worker_init_status_queue = IpcQueue(...)

        # Start background dispatch thread
        self._start_dispatch_threads()

    def _start_dispatch_threads(self):
        """Launch background thread to process results."""

        def dispatch_result_task():
            while not self._shutdown:
                # Block until result available
                res = self.result_queue.get(timeout=0.1)

                if res is None:
                    continue

                # Look up GenerationResult for this request
                request_id = res.client_id
                gen_result = self._results.get(request_id)

                if gen_result:
                    # Put into result's queue (wakes awaiter)
                    gen_result.queue.put(res)

                # Handle completion
                if res.is_final:
                    del self._results[request_id]

        thread = Thread(target=dispatch_result_task, daemon=True)
        thread.start()
```

### 3.4 AsyncQueue Implementation

#### **Bridge Between Sync and Async** (llmapi/utils.py:364)

```python
class AsyncQueue:
    """Queue that works with both sync and async code."""

    def __init__(self):
        self._q = collections.deque()       # Data store
        self._event = asyncio.Event()       # Async notification
        self._sync_q = _SyncQueue(self)     # Sync wrapper

    @property
    def sync_q(self):
        """Get sync-compatible wrapper."""
        return self._sync_q

    def put(self, item):
        """Thread-safe put (called from worker threads)."""
        self._q.append(item)
        self._event.set()

    async def get(self, timeout: Optional[float] = None) -> Any:
        """Async get with timeout."""
        start_time = time.monotonic()

        while True:
            # Check if data available
            if self._q:
                self._event.clear()
                return self._q.popleft()

            # Calculate remaining timeout
            if timeout is not None:
                elapsed = time.monotonic() - start_time
                remaining = timeout - elapsed
                if remaining <= 0:
                    raise asyncio.TimeoutError()
            else:
                remaining = None

            # Wait for data (async)
            try:
                await asyncio.wait_for(self._event.wait(), timeout=remaining)
            except asyncio.TimeoutError:
                if timeout is not None:
                    raise

class _SyncQueue:
    """Synchronous wrapper for AsyncQueue."""

    def __init__(self, async_queue: AsyncQueue):
        self._async_queue = async_queue
        self._pending_items = []

    def put_nowait(self, item):
        """Store for batched notification."""
        self._pending_items.append(item)

    @staticmethod
    def notify_many(event_loop, sync_queues: List['_SyncQueue']):
        """Notify multiple queues in one event loop call."""

        def notify_callback():
            for sq in sync_queues:
                for item in sq._pending_items:
                    sq._async_queue.put(item)
                sq._pending_items.clear()

        event_loop.call_soon_threadsafe(notify_callback)
```

**Design Benefits**:
- Worker threads (no event loop access) can call `put()`
- Async code can `await get()`
- Batched notifications reduce event loop overhead

### 3.5 C++ Batch Manager Scheduling

#### **Two-Phase Async Model**

TensorRT-LLM uses **asynchronous GPU execution** with overlapped scheduling:

**Phase 1: forwardAsync()** - Non-blocking scheduling and dispatch

**Phase 2: forwardSync()** - Blocking wait for completion

#### **Capacity Scheduling** (capacityScheduler.h)

Four scheduler variants:

```cpp
// 1. MaxRequestsScheduler: Simple count limit
class MaxRequestsScheduler : public CapacityScheduler {
    std::tuple<ReqList, ReqList> operator()(
        RequestList const& activeRequests,
        KvCacheManager const& kvCacheManager
    ) override {
        ReqList fittingRequests, requestsToPause;

        for (auto const& req : activeRequests) {
            if (fittingRequests.size() >= maxNumRequests) {
                requestsToPause.push_back(req);
            } else {
                fittingRequests.push_back(req);
            }
        }

        return {fittingRequests, requestsToPause};
    }
};

// 2. MaxUtilizationScheduler: Resource-aware
class MaxUtilizationScheduler : public CapacityScheduler {
    // Considers:
    // - KV cache blocks available
    // - Request token counts
    // - May pause requests if resources exhausted
};

// 3. GuaranteedNoEvictScheduler: Never evict
class GuaranteedNoEvictScheduler : public CapacityScheduler {
    // Only schedules if guaranteed not to evict
    // More conservative than MaxUtilization
};

// 4. StaticBatchScheduler: Fixed batch compilation
class StaticBatchScheduler : public CapacityScheduler {
    // For static batch size TRT engines
    // Pads or truncates to fixed size
};
```

#### **Micro-Batch Scheduling** (microBatchScheduler.h)

Splits batch into **context** (prefill) and **generation** (decode) phases:

```cpp
class MicroBatchScheduler {
    enum ContextChunkingPolicy {
        kEQUAL_PROGRESS,           // All requests advance equally
        kFIRST_COME_FIRST_SERVED   // Requests finish in order
    };

    std::tuple<ReqList, ReqList> operator()(
        ReqList const& fittingRequests,
        std::set<uint64_t> const& inflightReqIds,
        int maxContextRequests,
        int maxNumTokens,
        int maxNumGenerationRequests,
        ContextChunkingPolicy policy
    ) {
        ReqList contextRequests, generationRequests;
        int remainingContextTokens = maxNumTokens;

        for (auto const& req : fittingRequests) {
            // Skip if already processing (inflight)
            if (inflightReqIds.count(req->getReqId())) {
                continue;
            }

            if (req->isContextPhase()) {
                // Context (prefill) request
                int reqContextTokens = req->getNumContextTokens();

                if (policy == kEQUAL_PROGRESS) {
                    // Chunk equally across requests
                    int chunkSize = remainingContextTokens / (maxContextRequests - contextRequests.size());
                    reqContextTokens = std::min(reqContextTokens, chunkSize);
                }

                if (reqContextTokens <= remainingContextTokens) {
                    req->setMaxContextLen(reqContextTokens);
                    contextRequests.push_back(req);
                    remainingContextTokens -= reqContextTokens;
                }
            } else {
                // Generation (decode) request
                if (generationRequests.size() < maxNumGenerationRequests) {
                    generationRequests.push_back(req);
                }
            }
        }

        return {contextRequests, generationRequests};
    }
};
```

#### **Async Execution Model** (trtGptModelInflightBatching.cpp)

```cpp
void TrtGptModelInflightBatching::forwardAsync(
    RequestList const& activeRequests
) {
    // Step 1: CAPACITY SCHEDULING
    // Select requests that fit in memory
    auto [fittingRequests, fittingDisaggGenInitRequests, requestsToPause]
        = (*mCapacityScheduler)(
            activeRequests,
            *mKvCacheManager,
            *mPeftCacheManager
        );

    // Pause requests that don't fit
    for (auto const& req : requestsToPause) {
        req->pause();
    }

    // Step 2: MICRO BATCH SCHEDULING
    // Split into context and generation phases
    std::tie(currRequests.contextRequests, currRequests.generationRequests)
        = (*mMicroBatchScheduler)(
            fittingRequests,
            mInflightReqIds,  // Exclude already-processing requests
            maxContextRequests,
            maxNumTokens,
            maxGenerationRequests,
            mContextChunkingPolicy
        );

    // Step 3: RESOURCE ALLOCATION
    // Assign sequence slots
    (*mAssignReqSeqSlots)(
        currRequests.contextRequests,
        currRequests.generationRequests,
        mSeqSlotManager
    );

    // Allocate KV cache blocks
    (*mAllocateKvCache)(
        currRequests.contextRequests,
        currRequests.generationRequests,
        mKvCacheManager
    );

    // Step 4: BATCH EXECUTION (NON-BLOCKING)
    executeBatch(currRequests);  // Launches CUDA kernels

    // Step 5: ASYNC DECODER (for speculative decoding)
    // Returns CUDA event for async completion tracking
    mDecoderFinishedEvents[mMicroBatchId] = decoderStepAsync(currRequests);

    // Add to inflight set (excluded from next scheduling)
    for (auto const& req : currRequests.allRequests()) {
        mInflightReqIds.insert(req->getReqId());
    }

    // Returns immediately (no GPU sync)
}

void TrtGptModelInflightBatching::forwardSync() {
    // Step 1: WAIT FOR DECODER EXECUTION
    auto& decoderEvent = mDecoderFinishedEvents.at(mMicroBatchId);
    mDecStepAsyncSndHdls = decoderSync(currRequests, decoderEvent);

    // Step 2: HANDLE REQUEST STATE TRANSITIONS
    // Context → Generation if context phase complete
    for (auto& req : currRequests.contextRequests) {
        if (req->isContextPhaseComplete()) {
            req->transitionToGenerationPhase();
        }
    }

    // Generation → Complete if done
    for (auto& req : currRequests.generationRequests) {
        if (req->isGenerationComplete()) {
            req->markComplete();
            mInflightReqIds.erase(req->getReqId());
        }
    }

    // Step 3: PAUSE REQUESTS IF NEEDED
    (*mPauseRequests)(
        currRequests.generationRequests,
        mKvCacheManager,
        mPeftCacheManager
    );

    // Step 4: TERMINATE COMPLETED REQUESTS
    for (auto& req : currRequests.allRequests()) {
        if (req->isTerminated()) {
            mInflightReqIds.erase(req->getReqId());
        }
    }
}
```

**Async Execution Pattern**:
```
Iteration N:
  forwardAsync(batch_N)  → Launch kernels, return immediately

Iteration N+1:
  forwardSync()          → Wait for batch_N completion
  forwardAsync(batch_N+1) → Launch next batch
```

This overlaps CPU scheduling with GPU execution.

### 3.6 Request Lifecycle

```
1. User: llm.generate_async(prompt)
   ↓
2. GenerationExecutor.submit(request)
   → Returns GenerationResult immediately
   ↓
3. Request queued: request_queue.put(request)
   ↓
4. Worker process receives request
   ↓
5. C++ Batch Manager:
   forwardAsync() → Schedule and launch
   ↓
6. GPU executes batch (async)
   ↓
7. Next iteration: forwardSync() → Wait
   ↓
8. Worker sends Response: result_queue.put(response)
   ↓
9. Background thread: dispatch_result_task()
   → result.queue.put(response)
   ↓
10. User: await result.aresult() or result.result()
    → Unblocked, receives output
```

### 3.7 Scheduling Parameters

```python
@dataclass
class SchedulingParams:
    attention_dp_rank: Optional[int] = None       # Distributed attention rank
    attention_dp_relax: Optional[bool] = None     # Allow flexible scheduling

# Batch Manager Config (C++)
struct SchedulerConfig {
    CapacitySchedulerPolicy capacitySchedulerPolicy;
    ContextChunkingPolicy contextChunkingPolicy;
    int maxNumRequests;
    int maxNumTokens;
    int maxContextRequests;
    int maxGenerationRequests;
};
```

---

## 4. Algorithmic Differences

### 4.1 Queue Management

| System | Queue Type | Priority Support | Ordering |
|--------|------------|------------------|----------|
| **vLLM** | Deque (FCFS) or Heap (Priority) | Yes | `(priority, arrival_time)` |
| **SGLang** | List with sorting | Yes (optional) | Cache-aware or FCFS |
| **TensorRT-LLM** | IPC Queue | No | FIFO |

### 4.2 Batching Strategy

| System | Metric | Policy |
|--------|--------|--------|
| **vLLM** | **Token budget** | Schedule up to N tokens per step |
| **SGLang** | **Token budget + memory** | Conservative estimation with `new_token_ratio` |
| **TensorRT-LLM** | **Request count + tokens** | Capacity scheduler + micro-batch scheduler |

### 4.3 Preemption/Eviction

| System | Trigger | Victim Selection | Recovery |
|--------|---------|------------------|----------|
| **vLLM** | KV cache OOM | Lowest priority running request | Prepend to waiting queue |
| **SGLang** | Token budget exceeded | Lowest priority (configurable threshold) | Retract to waiting queue |
| **TensorRT-LLM** | Capacity scheduler | No automatic preemption | Pause until resources available |

### 4.4 Prefix Caching Integration

| System | Mechanism | Scheduling Impact |
|--------|-----------|-------------------|
| **vLLM** | Prefix matching before allocation | Reduces token budget consumption |
| **SGLang** | **RadixAttention with cache-aware scheduling** | **LPM or DFS-weight sorting** |
| **TensorRT-LLM** | KV cache manager | Transparent to scheduler |

**SGLang is unique**: Scheduling policy directly optimizes for cache hits (LPM, DFS-weight).

### 4.5 Async Execution Model

| System | CPU-GPU Overlap | Mechanism |
|--------|----------------|-----------|
| **vLLM** | Separate processes | EngineCore in subprocess, ZMQ IPC |
| **SGLang** | Event loop variants | Overlap event loop: process batch N-1 while running batch N |
| **TensorRT-LLM** | Async kernel dispatch | forwardAsync() + forwardSync() pattern |

### 4.6 Chunked Prefill

| System | Implementation | Scheduling |
|--------|----------------|-----------|
| **vLLM** | `long_prefill_token_threshold` | Cap prefill tokens per step |
| **SGLang** | `chunked_prefill_size` | Truncate request to chunk boundary, resume later |
| **TensorRT-LLM** | Context chunking policy | EQUAL_PROGRESS or FCFS chunking |

### 4.7 Disaggregation Support

| System | Prefill-Decode Separation | Implementation |
|--------|---------------------------|----------------|
| **vLLM** | Experimental | Not in v1 engine |
| **SGLang** | **Full support** | **Separate event loops, KV transfer queues** |
| **TensorRT-LLM** | Partial | Disaggregated generation init requests |

**SGLang has the most mature disaggregation**: Separate prefill/decode workers with async KV transfer.

### 4.8 Speculative Decoding

| System | Integration | Scheduling |
|--------|-------------|-----------|
| **vLLM** | Output placeholders | Pre-allocate for draft tokens |
| **SGLang** | Draft model support | Separate draft token handling |
| **TensorRT-LLM** | Async decoder step | decoderStepAsync() with CUDA event |

---

## 5. Comparative Summary

### 5.1 Design Philosophy

| System | Focus | Strength |
|--------|-------|----------|
| **vLLM** | **Throughput-first** | Token budget model, efficient KV cache (PagedAttention) |
| **SGLang** | **Cache-aware + flexibility** | RadixAttention, cache-aware scheduling, DSL frontend |
| **TensorRT-LLM** | **Maximum performance** | TensorRT optimization, async kernel dispatch, production-ready |

### 5.2 Scheduling Complexity

**vLLM**:
- Token budget scheduling
- Priority-based preemption
- Output placeholders for async
- ⭐ **Complexity: Medium**

**SGLang**:
- Cache-aware priority calculation
- RadixAttention prefix matching
- Multiple event loop variants
- Prefill-decode disaggregation
- ⭐ **Complexity: High**

**TensorRT-LLM**:
- Multi-tier queue architecture
- Capacity + micro-batch scheduling
- Async kernel dispatch
- ⭐ **Complexity: Medium-High**

### 5.3 Async Implementation

| Aspect | vLLM | SGLang | TensorRT-LLM |
|--------|------|--------|--------------|
| **Process Model** | Multi-process (ZMQ) | Three-process (ZMQ) | Multi-process (IPC queues) |
| **Output Handling** | Background output_handler | Detokenizer process | Background dispatch thread |
| **Event Loop** | asyncio | Custom event loops | asyncio + threads |
| **GPU Overlap** | Separate process | Overlap event loop | forwardAsync/Sync |

### 5.4 Feature Matrix

| Feature | vLLM | SGLang | TensorRT-LLM |
|---------|------|--------|--------------|
| Priority Scheduling | ✅ | ✅ (optional) | ❌ |
| Prefix Caching | ✅ | ✅ (RadixAttention) | ✅ |
| Cache-Aware Scheduling | ❌ | ✅ (LPM, DFS) | ❌ |
| Preemption | ✅ | ✅ | ⚠️ (pause only) |
| Chunked Prefill | ✅ | ✅ | ✅ |
| Disaggregation | ⚠️ (experimental) | ✅ (full) | ⚠️ (partial) |
| Async Kernel Dispatch | ❌ | ⚠️ (in overlap mode) | ✅ |
| Output Placeholders | ✅ | ❌ | ❌ |

### 5.5 Code Locations Reference

#### **vLLM**
```
vllm/v1/engine/async_llm.py           - Async API
vllm/v1/engine/core.py                - Engine core process
vllm/v1/core/sched/scheduler.py       - Scheduler logic
vllm/v1/core/sched/async_scheduler.py - Async scheduler
vllm/v1/core/sched/request_queue.py   - Queue implementations
vllm/v1/core/kv_cache_manager.py      - KV cache management
```

#### **SGLang**
```
sglang/srt/managers/scheduler.py         - Main scheduler
sglang/srt/managers/schedule_policy.py   - Scheduling policies
sglang/srt/managers/schedule_batch.py    - Batch construction
sglang/srt/mem_cache/radix_cache.py      - RadixAttention
sglang/srt/disaggregation/prefill.py     - Prefill worker
sglang/srt/disaggregation/decode.py      - Decode worker
```

#### **TensorRT-LLM**
```
tensorrt_llm/llmapi/llm.py                              - Async API
tensorrt_llm/executor/executor.py                       - Executor
tensorrt_llm/executor/proxy.py                          - IPC queues
tensorrt_llm/llmapi/utils.py                            - AsyncQueue
cpp/tensorrt_llm/batch_manager/capacityScheduler.h     - Capacity scheduling
cpp/tensorrt_llm/batch_manager/microBatchScheduler.h   - Micro-batch scheduling
cpp/tensorrt_llm/batch_manager/trtGptModelInflightBatching.cpp - Async execution
```

---

## 6. Key Takeaways

### 6.1 When to Use Each System

**vLLM**:
- Need high throughput with efficient memory usage
- PagedAttention for dynamic batching
- Priority-based request handling
- OpenAI-compatible API

**SGLang**:
- Heavy prefix sharing workloads (chatbots, multi-turn)
- Need cache-aware scheduling
- DSL for structured generation
- Prefill-decode disaggregation for large-scale serving

**TensorRT-LLM**:
- Maximum single-request latency performance
- NVIDIA hardware optimization
- Production deployment with TensorRT
- Quantization and optimization focus

### 6.2 Algorithmic Innovations

**vLLM**:
- ✨ **Output placeholders** - Reduce scheduling overhead in async mode
- ✨ **Token budget model** - Better resource utilization than batch count

**SGLang**:
- ✨ **Cache-aware scheduling** - LPM and DFS-weight sorting for maximum cache reuse
- ✨ **In-batch prefix caching** - Deprioritize requests that would benefit from waiting
- ✨ **Overlap event loop** - CPU-GPU pipelining

**TensorRT-LLM**:
- ✨ **Async kernel dispatch** - forwardAsync/forwardSync pattern
- ✨ **AsyncQueue** - Bridges sync worker threads and async event loop
- ✨ **Micro-batch scheduling** - Optimal context/generation split

---

## Conclusion

All three systems implement sophisticated async scheduling with different trade-offs:

- **vLLM**: Token budget + output placeholders for throughput
- **SGLang**: Cache-aware scheduling + disaggregation for flexibility
- **TensorRT-LLM**: Async kernel dispatch + TensorRT optimization for performance

The choice depends on workload characteristics (prefix sharing, latency vs throughput, hardware) and deployment requirements.
