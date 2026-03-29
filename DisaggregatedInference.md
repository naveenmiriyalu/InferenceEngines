# Disaggregated Inference: vLLM, SGLang, and TensorRT-LLM

## Table of Contents
1. [Executive Summary](#executive-summary)
2. [vLLM Disaggregated Inference](#1-vllm-disaggregated-inference)
3. [SGLang Disaggregated Inference](#2-sglang-disaggregated-inference)
4. [TensorRT-LLM Disaggregated Inference](#3-tensorrt-llm-disaggregated-inference)
5. [Comparative Analysis](#4-comparative-analysis)
6. [Performance Characteristics](#5-performance-characteristics)
7. [Deployment Examples](#6-deployment-examples)

---

## Executive Summary

Disaggregated inference separates LLM inference into **Prefill** (context encoding) and **Decode** (token generation) phases, running them on different compute resources. This architecture enables independent optimization of Time-To-First-Token (TTFT) and Inter-Token-Latency (ITL).

**Key Benefits:**
- **Resource Optimization**: Prefill uses high compute, Decode uses high memory bandwidth
- **Throughput**: Eliminate interference between prefill and decode batches
- **Scalability**: Scale prefill and decode independently based on workload

**Framework Support:**
- **vLLM**: Modular 3-tier architecture with multiple transfer protocols (NIXL, Mooncake, P2P NCCL)
- **SGLang**: Queue-based lifecycle with Mooncake/NIXL backends, router integration
- **TensorRT-LLM**: Executor API integration with UCX/NIXL/MPI backends

---

## 1. vLLM Disaggregated Inference

### 1.1 Architecture Overview

vLLM implements a modular three-layer abstraction:

```
┌─────────────────────────────────────────────────────┐
│            KV Connector Layer (Top)                 │
│  - send_kv_caches_and_hidden_states()              │
│  - Factory pattern for connector selection          │
└──────────────────┬──────────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────────┐
│         KV Lookup Buffer (Middle)                   │
│  - insert(token, kv_cache)                         │
│  - drop_select() - SQL-like semantics              │
│  - Handles request ordering divergence             │
└──────────────────┬──────────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────────┐
│        KV Pipe (Bottom - FIFO)                      │
│  - send_tensor(), recv_tensor()                    │
│  - Single-direction tensor transmission            │
└─────────────────────────────────────────────────────┘
```

---

### 1.2 File Structure

```
vllm/distributed/kv_transfer/
├── kv_transfer_state.py          # State management
├── kv_events.py                  # Event tracking
├── __init__.py                   # Global initialization
├── README.md                      # Documentation
└── kv_connector/
    ├── base.py                   # Base connector type
    ├── factory.py                # Connector factory & registration
    └── v1/
        ├── base.py               # KVConnectorBase_V1 (Lines 170-663)
        ├── __init__.py           # V1 exports & enums
        ├── metrics.py            # Stats aggregation (Lines 18-150)
        ├── utils.py              # Shared utilities
        │
        # Transfer Protocols
        ├── nixl_connector.py      # NIXL backend
        ├── mooncake/
        │   ├── mooncake_connector.py
        │   └── mooncake_utils.py
        ├── p2p/
        │   ├── p2p_nccl_connector.py
        │   ├── p2p_nccl_engine.py
        │   └── tensor_memory_pool.py
        ├── moriio/
        │   ├── moriio_connector.py
        │   ├── moriio_engine.py
        │   └── moriio_common.py
        │
        # Specialized Connectors
        ├── lmcache_connector.py   # LMCache integration
        ├── lmcache_mp_connector.py
        ├── offloading_connector.py # CPU offloading
        ├── flexkv_connector.py     # FlexKV distributed KV store
        ├── multi_connector.py      # Multi-connector routing
        │
        # Examples & Benchmarks
        ├── example_connector.py
        ├── example_hidden_states_connector.py
        └── decode_bench_connector.py
```

---

### 1.3 Configuration

**File:** `vllm/config/kv_transfer.py`

```python
@config
class KVTransferConfig:
    kv_connector: str | None = None           # Connector type
    engine_id: str | None = None              # Unique engine ID
    kv_buffer_device: str = "cuda"            # Buffer device (cuda/cpu/xpu)
    kv_buffer_size: float = 1e9               # Buffer size in bytes
    kv_role: KVRole | None = None             # "kv_producer"/"kv_consumer"/"kv_both"
    kv_rank: int | None = None                # Instance rank (0=prefill, 1=decode)
    kv_parallel_size: int = 1                 # Number of parallel instances
    kv_ip: str = "127.0.0.1"                  # IP for distributed connection
    kv_port: int = 14579                      # Port for KV communication
    kv_connector_extra_config: dict = {}      # Backend-specific config
    kv_connector_module_path: str | None = None
    enable_permute_local_kv: bool = False     # HND to NHD conversion
    kv_load_failure_policy: str = "fail"      # "recompute" or "fail"
```

---

### 1.4 KV Connector Base Class

**File:** `v1/base.py` (Lines 170-663)

#### Key Interfaces

**Worker-Side Methods:**
```python
class KVConnectorBase_V1:
    def bind_connector_metadata(self, metadata):
        """Bind metadata for KV load/save"""

    def start_load_kv(self):
        """Begin async KV loading from connector"""

    def wait_for_layer_load(self, layer_name):
        """Block until layer KV is loaded"""

    def save_kv_layer(self, layer_idx):
        """Save layer KV asynchronously"""

    def wait_for_save(self):
        """Block until all saves complete"""

    def get_finished(self):
        """Track completed transfers"""
```

**Scheduler-Side Methods:**
```python
    def get_num_new_matched_tokens(self):
        """Query available KV cache tokens"""

    def update_state_after_alloc(self):
        """Handle block allocation"""

    def build_connector_meta(self):
        """Build metadata for transfers"""

    def request_finished(self):
        """Signal request completion"""
```

---

### 1.5 Registered Connectors

**File:** `factory.py` (Lines 142-215)

```python
KVConnectorFactory.register_connector("ExampleConnector", ...)
KVConnectorFactory.register_connector("P2pNcclConnector", ...)
KVConnectorFactory.register_connector("LMCacheConnectorV1", ...)
KVConnectorFactory.register_connector("NixlConnector", ...)
KVConnectorFactory.register_connector("MooncakeConnector", ...)
KVConnectorFactory.register_connector("MultiConnector", ...)
KVConnectorFactory.register_connector("MoRIIOConnector", ...)
KVConnectorFactory.register_connector("OffloadingConnector", ...)
KVConnectorFactory.register_connector("FlexKVConnectorV1", ...)
```

---

### 1.6 Transfer Protocols

#### 1.6.1 NIXL Connector

**File:** `nixl_connector.py`

**Version:** 2 (Line 93)
**Transport:** ZMQ + NIXL wrapper

```python
NIXL_CONNECTOR_VERSION: int = 2

class NixlAgentMetadata:
    """Compatible changes tracked via version"""
    remote_request_id_mapping: dict  # v2 addition
    tp_topology: dict                 # TP awareness
    kv_layout: str                    # HND/NHD

# Protocol: ZMQ message passing + NIXL binary transfer
# Features:
# - Fully async send/recv
# - Host buffer support for KV transfer
# - Copy operation registration via set_host_xfer_buffer_ops()
# - Compatibility checking between prefill/decode versions
```

---

#### 1.6.2 Mooncake Connector

**File:** `mooncake/mooncake_connector.py`

**Dependencies:** Mooncake transfer engine
**Protocol:** ZMQ + HTTP + Mooncake async transfer

```python
class MooncakeXferMetadata(msgspec.Struct):
    remote_hostname: str
    remote_port: int
    remote_tp_size: int
    remote_tp_rank: int
    req_blocks: dict[ReqId, tuple[TransferId, list[int]]]
    kv_caches_base_addr: list[int]

class MooncakeXferResponseStatus(IntEnum):
    FINISH = 0
    CONTINUE = 1
    ERROR = 2
```

**Features:**
- Session-based KV transfers
- Health checking via HTTP
- Mooncake TransferEngine integration
- NVLink transport support (`SGLANG_MOONCAKE_CUSTOM_MEM_POOL=NVLINK`)

---

#### 1.6.3 P2P NCCL Connector

**File:** `p2p/p2p_nccl_connector.py`

**Topology:** Local peer-to-peer with NCCL communication
**Engine:** P2pNcclEngine

```python
@dataclass
class P2pNcclConnectorMetadata(KVConnectorMetadata):
    requests: list[ReqMeta]

    def add_request(
        self,
        request_id: str,
        token_ids: list[int],
        block_ids: list[int],
        block_size: int,
    ) -> None:
        """Add request metadata for P2P transfer"""
```

**Use Case:** Single-node multi-GPU disaggregation

---

### 1.7 Deployment Example

**Script:** `examples/online_serving/disaggregated_prefill.sh`

```bash
# Prefill instance (KV producer)
CUDA_VISIBLE_DEVICES=0 vllm serve "$MODEL_NAME" \
    --port 8100 \
    --kv-transfer-config '{
        "kv_connector":"P2pNcclConnector",
        "kv_role":"kv_producer",
        "kv_rank":0,
        "kv_parallel_size":2,
        "kv_buffer_size":"1e9"
    }'

# Decode instance (KV consumer)
CUDA_VISIBLE_DEVICES=1 vllm serve "$MODEL_NAME" \
    --port 8200 \
    --kv-transfer-config '{
        "kv_connector":"P2pNcclConnector",
        "kv_role":"kv_consumer",
        "kv_rank":1,
        "kv_parallel_size":2
    }'

# Proxy server bridges both instances
python3 disagg_prefill_proxy_server.py
```

---

### 1.8 Benchmarks

**Location:** `benchmarks/disagg_benchmarks/`

**Scripts:**
- `disagg_performance_benchmark.sh` - QPS scaling
- `disagg_overhead_benchmark.sh` - Overhead profiling
- `visualize_benchmark_results.py` - Result analysis

**Typical Setup:**
- Model: Meta-Llama-3.1-8B-Instruct
- Input: 1024 tokens, Output: 6 tokens
- QPS: 2/4/6/8 requests/sec
- Resource: 2x A100 GPUs

---

## 2. SGLang Disaggregated Inference

### 2.1 Architecture Overview

SGLang implements **Prefill-Decode (PD) disaggregation**:

```
┌────────────────────────────────────────────────────┐
│         Prefill Server (Encoding Phase)            │
├────────────────────────────────────────────────────┤
│  Bootstrap Queue → Waiting Queue → Inflight Queue │
│  ├─ Sender initialization & handshake             │
│  ├─ Forward pass execution                        │
│  └─ Async KV transfer to decode                   │
└────────────────┬───────────────────────────────────┘
                 │
        ┌────────▼────────┐
        │  KV Transfer    │
        │  (Mooncake/     │
        │   NIXL/Mori/    │
        │   Ascend)       │
        └────────┬────────┘
                 │
┌────────────────▼─────────────────────────────────┐
│         Decode Server (Generation Phase)          │
├────────────────────────────────────────────────────┤
│  PreallocQueue → TransferQueue → Running Batch   │
│  ├─ Receiver initialization & handshake          │
│  ├─ KV pre-allocation on arrival                 │
│  ├─ Async KV reception                           │
│  └─ Generation execution                         │
└────────────────────────────────────────────────────┘
```

---

### 2.2 File Structure

```
sglang/python/sglang/srt/disaggregation/
├── prefill.py                    # Prefill server logic
├── decode.py                     # Decode server logic
├── encode_server.py              # Encoder server (EPD)
├── encode_receiver.py            # Encoder receiver
├── utils.py                      # Utilities
├── kv_events.py                  # Event tracking
├── decode_kvcache_offload_manager.py
├── decode_schedule_batch_mixin.py
│
├── base/
│   └── conn.py                   # Abstract KV interfaces
│
├── common/
│   ├── conn.py                   # Shared connection logic
│   └── utils.py                  # FastQueue, utilities
│
├── mooncake/
│   ├── conn.py                   # Mooncake implementation (150+ lines)
│   └── utils.py
│
├── nixl/
│   └── conn.py                   # NIXL implementation
│
├── mori/
│   └── conn.py
│
├── ascend/
│   └── conn.py
│
└── fake/
    └── conn.py                   # Mock for testing
```

---

### 2.3 Base Abstractions

**File:** `base/conn.py`

```python
class KVArgs:
    """Shared metadata for KV transfer"""
    engine_rank: int
    kv_data_ptrs: List[int]           # Pointers to KV tensors
    kv_data_lens: List[int]           # Sizes of KV data
    kv_item_lens: List[int]           # Individual item sizes
    aux_data_ptrs: List[int]          # Auxiliary data (output IDs, logprobs)
    aux_data_lens: List[int]
    state_data_ptrs: List[int]        # Mamba/SWA state pointers
    state_type: str                   # "none", "mamba", "swa"
    decode_tp_size: int               # Decode-side TP parallelism
    kv_head_num: int
    page_size: int

class KVPoll(enum.Enum):
    Failed = 0
    Bootstrapping = 1
    WaitingForInput = 2
    Transferring = 3
    Success = 4

class BaseKVManager(ABC):
    """Manages transfer lifecycle"""
    def __init__(self, args: KVArgs, disaggregation_mode, server_args):
        ...

class BaseKVSender(ABC):
    """Prefill: sends KV to decode"""
    def init(self, num_kv_indices: int, aux_index: Optional[int]):
        ...

    def send(self, kv_indices: np.ndarray, state_indices: List[int]):
        ...

    def poll(self) -> KVPoll:
        ...

class BaseKVReceiver(ABC):
    """Decode: receives KV from prefill"""
    def init(self, kv_indices: np.ndarray, aux_index, state_indices):
        ...

    def poll(self) -> KVPoll:
        ...
```

---

### 2.4 Utilities

**File:** `utils.py`

```python
class DisaggregationMode(Enum):
    NULL = "null"
    PREFILL = "prefill"
    DECODE = "decode"

class TransferBackend(Enum):
    MOONCAKE = "mooncake"
    MORI = "mori"
    NIXL = "nixl"
    ASCEND = "ascend"
    FAKE = "fake"

class MetadataBuffers:
    """CPU/GPU buffers for transfer metadata"""
    output_ids: torch.Tensor              # Shape: (batch, 16)
    cached_tokens: torch.Tensor
    output_token_logprobs_val: torch.Tensor
    output_token_logprobs_idx: torch.Tensor
    output_top_logprobs_val: torch.Tensor
    output_top_logprobs_idx: torch.Tensor
    output_hidden_states: torch.Tensor    # For spec decode
    bootstrap_room: torch.Tensor          # Validation field

def poll_and_all_reduce(pollers, gloo_group):
    """AllReduce across TP workers to synchronize transfer state"""
    polls = [int(poller.poll()) for poller in pollers]
    tensor_to_reduce = torch.tensor(polls, dtype=torch.uint8, device="cpu")
    dist.all_reduce(tensor_to_reduce, op=dist.ReduceOp.MIN, group=gloo_group)
    return tensor_to_reduce.tolist()
```

---

### 2.5 Mooncake Implementation

**File:** `mooncake/conn.py` (Lines 1-150+)

```python
@dataclasses.dataclass
class TransferKVChunk:
    """Prefill → Decode: KV chunk with metadata"""
    room: int                                    # Bootstrap room ID
    prefill_kv_indices: npt.NDArray[np.int32]   # Which KV cache pages
    index_slice: slice                           # Chunk offset
    is_last: bool                                # End-of-request marker
    prefill_aux_index: Optional[int]             # Aux data location
    state_indices: Optional[List[int]]           # Mamba/SWA state

@dataclasses.dataclass
class TransferInfo:
    """Decode → Prefill: transfer destination metadata"""
    room: int
    endpoint: str                                # Prefill server IP:port
    dst_port: int
    mooncake_session_id: str                     # Mooncake session handle
    dst_kv_indices: npt.NDArray[np.int32]        # Where to put KV
    dst_aux_index: int                           # Aux destination
    dst_state_indices: List[int]
    required_dst_info_num: int                   # Expected responses
    is_dummy: bool                               # Placeholder flag

class AuxDataCodec:
    """Serialization of output tokens, logprobs, hidden states"""
    @staticmethod
    def encode_aux_data(...):
        ...

    @staticmethod
    def decode_aux_data(...):
        ...
```

---

### 2.6 Queue Lifecycles

#### Prefill Server (prefill.py)

```
1. Bootstrap Queue
   ├─ Initialize sender for each request
   ├─ Poll senders for bootstrap completion
   └─ Move to Waiting Queue when ready

2. Waiting Queue
   ├─ PrefillAdder: pop requests for forward pass
   ├─ Model forward execution
   └─ Add request to Inflight Queue

3. Inflight Queue
   ├─ Poll sender non-blocking
   ├─ Check transfer status
   └─ Return when transfer complete
```

#### Decode Server (decode.py)

```
1. PreallocQueue
   ├─ Initialize receiver for each request
   ├─ Receiver handshake with prefill
   └─ Pre-allocate KV space when ready

2. TransferQueue
   ├─ Poll receiver for transfer status
   ├─ Track async KV reception
   └─ Move to Waiting Queue when done

3. Waiting Queue
   ├─ Build PrebuiltExtendBatch
   ├─ Metadata-only forward (skip prefill)
   └─ Merge into RunningBatch

4. RunningBatch
   └─ Execute generation phase
```

---

### 2.7 Configuration & Deployment

```bash
# Single-node example (Mooncake backend):
python -m sglang.launch_server \
  --model-path meta-llama/Llama-3.1-8B-Instruct \
  --disaggregation-mode prefill \
  --port 30000 \
  --disaggregation-ib-device mlx5_roce0

python -m sglang.launch_server \
  --model-path meta-llama/Llama-3.1-8B-Instruct \
  --disaggregation-mode decode \
  --port 30001 \
  --base-gpu-id 1 \
  --disaggregation-ib-device mlx5_roce0

# Launch router
python -m sglang_router.launch_router \
  --pd-disaggregation \
  --prefill http://127.0.0.1:30000 \
  --decode http://127.0.0.1:30001 \
  --port 8000
```

---

### 2.8 Environment Variables

**Prefill Server:**
```bash
SGLANG_DISAGGREGATION_THREAD_POOL_SIZE    # Worker threads (default: dynamic)
SGLANG_DISAGGREGATION_QUEUE_SIZE          # Parallel transfer queues (default: 4)
SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT   # Request init timeout (default: 300s)
SGLANG_MOONCAKE_CUSTOM_MEM_POOL           # NVLink/BAREX/INTRA_NODE_NVLINK
MC_FORCE_MNNVL                            # Force Mooncake NVLink
```

**Decode Server:**
```bash
SGLANG_DISAGGREGATION_HEARTBEAT_INTERVAL  # Health check (default: 5.0s)
SGLANG_DISAGGREGATION_HEARTBEAT_MAX_FAILURE # Offline threshold (default: 2)
SGLANG_DISAGGREGATION_WAITING_TIMEOUT     # KV reception timeout (default: 300s)
```

**NIXL Backend:**
```bash
SGLANG_DISAGGREGATION_NIXL_BACKEND        # "UCX" (default) or "LIBFABRIC"
```

---

## 3. TensorRT-LLM Disaggregated Inference

### 3.1 Architecture Overview

TensorRT-LLM uses the **Executor API** with a pluggable KV cache transceiver:

```
┌────────────────────────────┐
│   Client Requests          │
└────────────┬───────────────┘
             │
   ┌─────────▼─────────────────────────────┐
   │  Disaggregated Server (Router)         │
   │  - Routes context-only to prefill      │
   │  - Routes generation-only to decode    │
   │  - Orchestrates context_params         │
   └────────────┬──────────────┬────────────┘
                │              │
    ┌───────────▼──┐   ┌──────▼──────────┐
    │Context Engines│   │ Generation      │
    │(Prefill)      │   │ Engines (Decode)│
    │               │   │                 │
    │Executor API:  │   │Executor API:    │
    │- context_only │   │- gen_only       │
    │- return ctx   │   │- use ctx_params │
    │  params       │   │                 │
    └───────────────┘   └────────┬────────┘
            │                    │
            └────────┬───────────┘
                     │
    ┌────────────────▼──────────────┐
    │  KV Cache Transceiver (UCX/   │
    │  NIXL/MPI backends)           │
    │  - src: context GPU memory    │
    │  - dst: generation GPU memory │
    │  - RDMA or NVLink transfer    │
    └───────────────────────────────┘
```

---

### 3.2 Configuration Files

#### DisaggregatedParams

**File:** `tensorrt_llm/disaggregated_params.py`

```python
@dataclass(slots=True, kw_only=True)
class DisaggregatedParams:
    # Prefill-Decode disaggregation
    request_type: Optional[str]           # "context_only" | "generation_only"
                                          # | "context_and_generation"
    first_gen_tokens: Optional[List[int]] = None
    ctx_request_id: Optional[int] = None  # Link between P and D
    opaque_state: Optional[bytes] = None  # Custom state exchange
    draft_tokens: Optional[List[int]] = None

    # Encoder-Prefill disaggregation
    multimodal_embedding_handles: Optional[List[Dict[str, Any]]] = None
    multimodal_hashes: Optional[List[List[int]]] = None
    mrope_position_ids_handle: Optional[Dict[str, Any]] = None

    def get_context_phase_params(self) -> tllme.ContextPhaseParams:
        ...

    def get_request_type(self) -> tllme.RequestType:
        ...
```

---

#### DisaggServerConfig

**File:** `tensorrt_llm/llmapi/disagg_utils.py`

```python
@dataclass
class CtxGenServerConfig():
    type: Literal['ctx', 'gen']
    hostname: Optional[str] = None
    port: Optional[int] = None
    instance_num_ranks: int = 1            # MPI ranks per instance
    other_args: dict = field(default_factory=dict)

@dataclass
class DisaggServerConfig():
    server_configs: List[CtxGenServerConfig]
    hostname: str = "localhost"
    port: int = 8000                       # Main service port
    ctx_router_config: Optional[RouterConfig] = None
    gen_router_config: Optional[RouterConfig] = None
    conditional_disagg_config: Optional[ConditionalDisaggConfig] = None
    max_retries: int = 1
    disagg_cluster_config: Optional[DisaggClusterConfig] = None
```

---

### 3.3 KV Cache Transceiver

**File:** `tensorrt_llm/_torch/pyexecutor/kv_cache_transceiver.py`

```python
class CacheTransceiverConfig:
    backend: str = "NIXL"                 # "NIXL", "UCX", "MPI", "DEFAULT"
    max_tokens_in_buffer: int             # >= max ISL for optimal performance

class KVCacheTransceiver:
    """Exchanges KV blocks between context and generation engines"""

    # Prefill side
    def register_kv_cache(self, kv_tensor):
        """Register KV tensor for RDMA"""

    def send_kv_cache_blocks(self, block_ids):
        """Send KV blocks via RDMA/NVLink"""

    # Decode side
    def allocate_recv_buffer(self, num_tokens):
        """Allocate buffer for receiving KV blocks"""

    def recv_kv_cache_blocks(self, block_ids):
        """Receive KV blocks via RDMA/NVLink"""
```

---

### 3.4 Supported Backends

| Backend | Protocol | Features | Use Case |
|---------|----------|----------|----------|
| **NIXL** | UCX/LibFabric | High-speed, dynamic scaling | Recommended default |
| **UCX** | RDMA/NVLink | Low-latency, multi-node | High-performance |
| **MPI** | MPI alltoall | Portable, stable | Legacy deployments |
| **DEFAULT** | Platform-specific | Auto-detection | Single-node |

---

### 3.5 Deployment Example

**Configuration Files:**

```yaml
# context_config.yml
disable_overlap_scheduler: True
cache_transceiver_config:
  backend: UCX
  max_tokens_in_buffer: 2048

# gen_config.yml
cache_transceiver_config:
  backend: UCX
  max_tokens_in_buffer: 2048

# disagg_config.yaml
hostname: localhost
port: 8000
backend: pytorch

context_servers:
  num_instances: 2
  urls:
    - "localhost:8001"
    - "localhost:8002"

generation_servers:
  num_instances: 1
  urls:
    - "localhost:8003"
```

**Launch Commands:**

```bash
# Launch context servers
CUDA_VISIBLE_DEVICES=0 trtllm-serve TinyLlama-1.1B \
    --host localhost --port 8001 \
    --config ./context_config.yml &

CUDA_VISIBLE_DEVICES=1 trtllm-serve TinyLlama-1.1B \
    --host localhost --port 8002 \
    --config ./context_config.yml &

# Launch generation servers
CUDA_VISIBLE_DEVICES=2 trtllm-serve TinyLlama-1.1B \
    --host localhost --port 8003 \
    --config ./gen_config.yml &

# Launch orchestrator
trtllm-serve disaggregated -c disagg_config.yaml
```

---

## 4. Comparative Analysis

### 4.1 Architecture Comparison

| Aspect | vLLM | SGLang | TensorRT-LLM |
|--------|------|--------|--------------|
| **Abstraction Level** | 3-tier (Connector, Lookup, Pipe) | 2-tier (Manager, Sender/Receiver) | Executor API + Transceiver |
| **Request Routing** | Proxy server | Router service | Disaggregated server |
| **State Sync** | Global connector state | AllReduce across TP | Context params + opaque_state |
| **Queue Model** | FIFO + SQL-like lookup | Bootstrap → Waiting → Running | Pre-allocation + streaming |
| **Async Support** | Layer-by-layer | Thread pool queued | Event-based |

---

### 4.2 Transfer Protocol Comparison

| Protocol | Framework(s) | Transport | Key Feature |
|----------|--------------|-----------|-------------|
| **NIXL** | vLLM, SGLang | ZMQ + RDMA/NVLink | Version compatibility checking |
| **Mooncake** | vLLM, SGLang | ZMQ + HTTP + async engine | NVLink support, session-based |
| **P2P NCCL** | vLLM | NCCL collectives | Single-node, tight integration |
| **UCX** | TensorRT-LLM | RDMA/NVLink | Multi-node, standard HPC |
| **MPI** | TensorRT-LLM | MPI alltoall | Portable, stable |

---

### 4.3 Configuration Complexity

**vLLM:**
- **Minimal**: 3-5 config parameters per instance
- Example: `--kv-transfer-config '{"kv_connector":"P2pNcclConnector",...}'`

**SGLang:**
- **Moderate**: Backend selection + environment variables
- Example: `--disaggregation-transfer-backend nixl --disaggregation-mode prefill`

**TensorRT-LLM:**
- **Complex**: YAML config files + Executor API parameters
- Example: 3 separate server configs + disagg orchestrator

---

## 5. Performance Characteristics

### 5.1 Network Requirements

| Scenario | Bandwidth | Latency | Recommended Protocol |
|----------|-----------|---------|----------------------|
| Single-node, 2 GPUs | PCIe/NVLink | <1us | P2P NCCL (vLLM) |
| Multi-node, high-perf | InfiniBand | <10us | NIXL/UCX (all) |
| Multi-node, standard | Ethernet | 100us+ | NIXL/MPI |

---

### 5.2 KV Transfer Overhead

**Data Volume (per sequence):**
```
KV Cache = 2 * num_layers * batch_size * seq_len * hidden_dim * dtype_size
         = 2 * 32 * 1 * 1024 * 4096 * 2 bytes  (Llama-3.1-8B)
         ≈ 512 MB per sequence
```

**Transfer Time:**
- @ 1 TB/s (NVLink): ~0.5ms
- @ 100 Gbps (Ethernet): ~40ms

---

### 5.3 Theoretical Limits

**GPU-to-GPU (Same node, NVLink 4):**
```
Peak BW: 900 GB/s
Latency: <100ns
Effective BW: 800+ GB/s
Transfer time (512 MB): ~0.6ms
```

**GPU-to-GPU (Multi-node, InfiniBand HDR):**
```
Peak BW: 200 Gbps = 25 GB/s
Latency: 1-2 μs
Effective BW: 20+ GB/s
Transfer time (512 MB): ~25ms
```

---

### 5.4 Metadata Overhead

**vLLM NixlConnector:**
- Metadata: block IDs, TP topology (<1 KB per request)
- Protocol: ZMQ (TCP-based)

**SGLang Mooncake:**
- Metadata: TransferInfo struct (~100 bytes)
- Auxiliary data: 64+ bytes per token (output IDs, logprobs)
- Protocol: ZMQ + HTTP + in-band

**TensorRT-LLM:**
- Metadata: ContextPhaseParams (<100 bytes)
- Protocol: Executor API structured passing

---

### 5.5 Failure Handling

| Framework | Load Failure | Network Failure | Timeout |
|-----------|--------------|-----------------|---------|
| **vLLM** | Policy: "recompute"/"fail" | Connector error propagation | No explicit timeout |
| **SGLang** | Heartbeat-based detection | Error callback + cleanup | 300s (configurable) |
| **TensorRT-LLM** | Executor exception | Retry mechanism (max_retries=1) | Implicit in executor |

---

## 6. Deployment Examples

### 6.1 vLLM Example

```python
# Prefill instance
vllm_args = [
    "--model", "meta-llama/Llama-3.1-8B-Instruct",
    "--port", "8100",
    "--kv-transfer-config", json.dumps({
        "kv_connector": "P2pNcclConnector",
        "kv_role": "kv_producer",
        "kv_rank": 0,
        "kv_parallel_size": 2,
        "kv_buffer_size": "1e9"
    })
]

# Decode instance
vllm_args = [
    "--model", "meta-llama/Llama-3.1-8B-Instruct",
    "--port", "8200",
    "--kv-transfer-config", json.dumps({
        "kv_connector": "P2pNcclConnector",
        "kv_role": "kv_consumer",
        "kv_rank": 1,
        "kv_parallel_size": 2
    })
]
```

---

### 6.2 SGLang Example

```bash
# Prefill server
python -m sglang.launch_server \
  --model-path meta-llama/Llama-3.1-8B-Instruct \
  --disaggregation-mode prefill \
  --disaggregation-transfer-backend mooncake \
  --port 30000

# Decode server
python -m sglang.launch_server \
  --model-path meta-llama/Llama-3.1-8B-Instruct \
  --disaggregation-mode decode \
  --disaggregation-transfer-backend mooncake \
  --port 30001

# Router
python -m sglang_router.launch_router \
  --pd-disaggregation \
  --prefill http://127.0.0.1:30000 \
  --decode http://127.0.0.1:30001 \
  --port 8000
```

---

### 6.3 TensorRT-LLM Example

```yaml
# disagg_config.yaml
hostname: localhost
port: 8000

context_servers:
  urls: ["localhost:8001", "localhost:8002"]

generation_servers:
  urls: ["localhost:8003"]
```

```bash
trtllm-serve disaggregated -c disagg_config.yaml
```

---

## Summary

### Strengths by Framework

**vLLM:**
- ✅ Modular 3-tier abstraction
- ✅ Multiple protocol backends (9+ connectors)
- ✅ Layer-by-layer async transfer
- ✅ Comprehensive examples

**SGLang:**
- ✅ Clean queue-based lifecycle
- ✅ Strong Mooncake + NIXL integration
- ✅ Fine-grained environment variable tuning
- ✅ Router service for load balancing
- ✅ Auxiliary data transfer (logprobs, hidden states)

**TensorRT-LLM:**
- ✅ Integrated Executor API
- ✅ Multiple backends (NIXL, UCX, MPI)
- ✅ Overlap optimization with multi-request batching
- ✅ Production-ready with Dynamo integration

### Recommendations

**For Prototyping:** vLLM - Simplest entry with ExampleConnector

**For Production (High-Performance):** SGLang with Mooncake/NIXL - Clean model, fine-tuned control

**For Data Center Scale:** TensorRT-LLM with Dynamo - Enterprise orchestration, dynamic scaling

---

**Document Version:** 1.0
**Last Updated:** 2026-03-29
