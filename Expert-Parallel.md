# Expert Parallelism and Load Balancing Comparison

Comprehensive comparison of Expert Parallelism (EP) and load balancing for Mixture-of-Experts (MoE) models across vLLM, SGLang, and TensorRT-LLM.

**Last Updated:** 2026-03-28

---

## Table of Contents

1. [Overview & Architecture Comparison](#overview--architecture-comparison)
2. [MoE Layer Implementations](#moe-layer-implementations)
3. [Expert Routing & Top-K Gating](#expert-routing--top-k-gating)
4. [Expert Parallelism Strategies](#expert-parallelism-strategies)
5. [Load Balancing Algorithms](#load-balancing-algorithms)
6. [Communication Backends](#communication-backends)
7. [Supported MoE Models](#supported-moe-models)
8. [Configuration & Command-Line Options](#configuration--command-line-options)
9. [Code Sources & Implementation Details](#code-sources--implementation-details)
10. [Feature Comparison Matrix](#feature-comparison-matrix)

---

## Overview & Architecture Comparison

### vLLM

**Architecture:** Modular expert parallelism with EPLB (Expert Parallel Load Balancing)

**Core Components:**
- **FusedMoE:** Main MoE layer with router integration
- **Router Hierarchy:** Base → Topk → Grouped → Biased variants
- **EPLB State:** Physical-to-logical expert mapping with redundancy
- **Expert Placement:** Linear or round-robin strategies

**Key Features:**
- Dynamic expert rearrangement based on load
- Redundant expert replicas for popular experts
- Multiple routing methods (8 types)
- All2All communication with 8 backends
- Sequence parallel for MoE

**Files:**
- `/vllm/model_executor/layers/fused_moe/` - MoE layer implementation
- `/vllm/distributed/eplb/` - Expert parallel load balancing
- `/vllm/config/parallel.py` - Parallel configuration

---

### SGLang

**Architecture:** Three-backend system with EPLB and overlap scheduling

**Core Components:**
- **MoE Router:** Softmax + Topk with softcapping
- **Expert Location:** Physical-to-logical mapping with locality-aware dispatch
- **EPLB Manager:** Utilization-based rebalancing
- **All2All Backends:** 6 communication options

**Key Features:**
- Static/dynamic dispatch algorithms
- Nearest-expert selection (GPU > Node > Remote)
- Expert distribution metrics
- Two-batch overlap (TBO) and single-batch overlap (SBO)
- DeepEP/FlashInfer/Mooncake backends

**Files:**
- `/sglang/python/sglang/srt/layers/moe/` - MoE implementation
- `/sglang/python/sglang/srt/eplb/` - EPLB and location management
- `/sglang/python/sglang/srt/server_args.py` - Configuration

---

### TensorRT-LLM

**Architecture:** Two-tier routing with host memory sharing

**Core Components:**
- **Routing Methods:** 7 types (Default, Renormalize, DeepSeekV3, Llama4, MiniMax2)
- **Load Balancer:** C++ implementation with shared memory
- **HostMoeTensorSharer:** Shared memory for expert weights
- **SingleLayerMoeLoadBalancer:** Per-layer load tracking

**Key Features:**
- Static and dynamic routing modes
- GPU-aware expert assignment
- Round-robin load balancing
- MPI-based weight migration
- Host memory tensor sharing across local ranks

**Files:**
- `/TensorRT-LLM/tensorrt_llm/_torch/modules/fused_moe/routing.py` - Routing methods
- `/TensorRT-LLM/tensorrt_llm/_torch/modules/fused_moe/moe_load_balancer.py` - Load balancing
- `/TensorRT-LLM/tensorrt_llm/llmapi/llm_args.py` - Configuration

---

## MoE Layer Implementations

### vLLM - FusedMoE

**File:** `/vllm/model_executor/layers/fused_moe/layer.py`

**Main Class:**
```python
class FusedMoE(CustomOp.register("fused_moe")):
    def __init__(
        self,
        num_experts: int,
        top_k: int,
        hidden_size: int,
        intermediate_size: int,
        reduce_results: bool = False,
        renormalize: bool = True,
        use_grouped_topk: bool = False,
        num_expert_group: Optional[int] = None,
        topk_group: Optional[int] = None,
        ...
    ):
        self.num_experts = num_experts
        self.top_k = top_k
        self.hidden_size = hidden_size
        self.intermediate_size_per_partition = intermediate_size

        # Router selection via factory
        self.router = create_fused_moe_router(
            routing_method=routing_method,
            num_experts=num_experts,
            top_k=top_k,
            ...
        )
```

**Expert Kernels:**
- `TritonExperts` - Triton-based implementation
- `CutlassExpertsFp8Base` - CUTLASS FP8
- `CutlassExpertsW4A8Fp8` - CUTLASS W4A8 FP8
- `DeepGemmExperts` - DeepGEMM MoE
- `BatchedDeepGemmExperts` - Batched DeepGEMM
- `FlashInferExperts` - FlashInfer CUTLASS
- `BatchedTritonExperts` - Batched Triton
- `MarlinExpertsBase` - Marlin MoE

**Expert Placement:**
```python
# File: layer.py Lines 67-153
def determine_expert_map(
    ep_size: int,
    ep_rank: int,
    global_num_experts: int,
    expert_placement_strategy: ExpertPlacementStrategy = "linear",
    ...
) -> tuple[int, torch.Tensor | None, torch.Tensor | None]:

    # Linear placement: contiguous distribution
    # ep_size=2, global_num_experts=4
    # rank 0: experts [0,1], rank 1: experts [2,3]

    # Round-robin placement: scattered distribution
    # ep_size=2, global_num_experts=4
    # rank 0: experts [0,2], rank 1: experts [1,3]
```

---

### SGLang - MoE Router Implementation

**File:** `/sglang/python/sglang/srt/layers/moe/router.py` (Lines 1-429)

**FusedMoeRouter Class:**
```python
class FusedMoeRouter:
    def __init__(self, config):
        self.top_k = config.num_experts_per_tok
        self.num_experts = config.n_routed_experts
        self.scoring_func = getattr(config, "scoring_func", "softmax")
        self.correction_bias = getattr(config, "e_score_correction_bias", None)

    def forward(self, hidden_states: torch.Tensor) -> tuple:
        # Router forward pass
        # Returns: (topk_weights, topk_ids, token_expert_indices)
```

**Router Variants:**
- `fused_moe_router_cudacore()` (Lines 117-156) - CUDA core implementation
- `fused_moe_router_tensorcore()` (Lines 286-337) - Tensor core implementation
- `fused_moe_router_shim()` (Lines 340-390) - Selection shim

**Softcapping:**
```python
# Lines 48-56 in CUDA core kernel
if softcap is not None:
    logits = softcap * torch.tanh(logits / softcap)
```

**Correction Bias:**
```python
# Lines 58-60, 224-230
if correction_bias is not None:
    logits = logits + correction_bias
```

---

### TensorRT-LLM - Routing Methods

**File:** `/TensorRT-LLM/tensorrt_llm/_torch/modules/fused_moe/routing.py` (Lines 1-805)

**RoutingMethodType Enum (Lines 147-162):**
```python
class RoutingMethodType(IntEnum):
    Default = 0           # Softmax → TopK
    Renormalize = 1       # TopK → Softmax
    DeepSeekV3 = 2        # Sigmoid → RoutingBiasAdd → Top2 in group → Top4 groups
    Llama4 = 3            # Top1 → Sigmoid
    RenormalizeNaive = 4  # Softmax → TopK → Renormalize
    MiniMax2 = 5          # Sigmoid → RoutingBiasAdd → TopK → Renormalize
```

**BaseMoeRoutingMethod (Lines 164-188):**
```python
class BaseMoeRoutingMethod(ABC):
    @abstractmethod
    def apply(
        self,
        routing_logits: torch.Tensor,
        expert_indices: Optional[torch.Tensor] = None,
        **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Returns: (token_selected_experts: int32, token_final_scales: float32)
        raise NotImplementedError
```

**DeepSeekV3 Routing (Lines 344-382):**
```python
class DeepSeekV3MoeRoutingMethod(BaseMoeRoutingMethod):
    def apply(self, routing_logits, **kwargs):
        # Sigmoid → RoutingBiasAdd
        # Top2 experts per group → Top4 groups → Top8 experts total
        # Complex grouped routing with bias correction
```

**Load Balanced Routing (Lines 580-609):**
```python
class LoadBalancedMoeRoutingMethod(BaseMoeRoutingMethod):
    def apply(self, routing_logits, **kwargs):
        # Round-robin expert assignment for perfect balance
        # Generates cyclic indices across GPUs
```

---

## Expert Routing & Top-K Gating

### vLLM - Router Hierarchy

**File:** `/vllm/model_executor/layers/fused_moe/router/base_router.py` (Lines 99-250)

**BaseRouter Class:**
```python
class BaseRouter(FusedMoERouter):
    def select_experts(
        self,
        hidden_states: torch.Tensor,
        topk_group: int | None = None,
        token_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        # Template method:
        # 1. EPLB state validation
        # 2. Indices dtype conversion
        # 3. Delegate to _compute_routing() (abstract)
        # 4. EPLB mapping (logical to physical expert IDs)
        # 5. Indices dtype conversion

        # Returns: (topk_weights, topk_ids, token_expert_indices)
```

**Routing Method Types (config.py Lines 100-121):**
```python
class RoutingMethodType(IntEnum):
    Default = 0           # Softmax → TopK
    Renormalize = 1       # TopK → Softmax/Sigmoid
    DeepSeekV3 = 2        # Sigmoid → RoutingBiasAdd → Grouped Topk
    Llama4 = 3            # Top1 → Sigmoid
    RenormalizeNaive = 4  # Softmax/Sigmoid → TopK → Renormalize
    TopK = 5              # TopK (no softmax)
    Custom = 6
    Simulated = 7
```

**Concrete Routers:**

1. **FusedTopKRouter** (`router/fused_topk_router.py` Lines 116-166)
   - Standard fused top-k routing with softmax/sigmoid
   - Function: `fused_topk()` (Lines 69-114)
   - Supports: `scoring_func` in ["softmax", "sigmoid"]

2. **FusedTopKBiasRouter** (`router/fused_topk_bias_router.py` Lines 173-200+)
   - Key Function: `fused_topk_bias()` (Lines 74-171)
   - Special Feature: `e_score_correction_bias` parameter
   - Use Case: DeepSeek models with routing bias

3. **GroupedTopKRouter** (`router/grouped_topk_router.py` Lines 250+)
   - Function: `grouped_topk()` (Lines 84-166)
   - Parameters:
     - `num_expert_group`: Number of expert groups
     - `topk_group`: Top-k groups to select from
   - Used by: DeepSeek-V2, DeepSeek-V3

**GateLinear (Router Network):**
- **File:** `router/gate_linear.py` (Lines 12-118)
- **Three-Tier GEMM Dispatch:**
  1. Tier 1: DSV3 specialized kernel (SM90+, batch≤16)
  2. Tier 2: cuBLAS bf16→fp32 (SM90+ + bf16 + fp32)
  3. Tier 3: F.linear via ReplicatedLinear (fallback)

---

### SGLang - Top-K Selection

**File:** `/sglang/python/sglang/srt/layers/moe/topk.py` (Lines 1-1100)

**TopK Class:**
```python
class TopK:
    """Multi-platform expert selection"""
```

**Output Formats:**
- **StandardTopKOutput** (Lines 161-170)
- **TritonKernelTopKOutput** (Lines 173-182)
- **BypassedTopKOutput** (Lines 185-196)

**Routing Variants:**

1. **fused_topk()** (Lines 450-490)
   - Standard fused topk

2. **grouped_topk()** (Lines 494-559)
   - Grouped expert selection
   - Multi-stage routing

3. **biased_grouped_topk()** (Lines 730-872)
   - Bias-aware grouped selection
   - For models with routing bias

4. **kimi_k2_biased_topk_impl()** (Lines 590-628)
   - Specialized for 384-expert models
   - Optimized kernel for large expert counts

---

### TensorRT-LLM - Router Selection

**Function:** `create_renormalize_expert_load_balanced_logits()` (routing.py Lines 648-804)

**GPU-Aware Expert Assignment:**
```python
def create_renormalize_expert_load_balanced_logits(
    num_experts: int,
    experts_per_token: int,
    num_tokens: int,
    dtype: torch.dtype,
    device: torch.device,
    moe_ep_size: int,
    moe_ep_rank: int,
) -> torch.Tensor:
    # Ensures balanced work distribution across expert parallel GPUs
    # Creates logits with high values (7.5) for selected experts
    # Low values (0.5) for others
    # Applies softmax for probability distribution
```

**Expert Assignment Pattern:**
```python
# Round-robin across GPUs
# moe_ep_size=4, experts_per_token=2, num_experts=16
# GPU 0 gets experts [0, 4, 8, 12]
# GPU 1 gets experts [1, 5, 9, 13]
# etc.
```

---

## Expert Parallelism Strategies

### vLLM - ParallelConfig

**File:** `/vllm/config/parallel.py` (Lines 97-883)

**Core Fields:**
```python
class ParallelConfig:
    pipeline_parallel_size: int = 1
    tensor_parallel_size: int = 1
    data_parallel_size: int = 1

    # Expert Parallelism
    enable_expert_parallel: bool = False
    enable_ep_weight_filter: bool = False
    enable_eplb: bool = False
    eplb_config: EPLBConfig = Field(...)
    expert_placement_strategy: ExpertPlacementStrategy = "linear"
    all2all_backend: All2AllBackend = "allgather_reducescatter"
```

**Expert Placement Strategies:**
- **"linear":** Contiguous placement
  - 4 experts, 2 ranks → rank0:[0,1], rank1:[2,3]
- **"round_robin":** Interleaved placement
  - 4 experts, 2 ranks → rank0:[0,2], rank1:[1,3]

**EPLBConfig (Lines 55-94):**
```python
@config
class EPLBConfig:
    window_size: int = 1000              # Expert load recording window
    step_interval: int = 3000            # Rearrangement interval (tokens)
    num_redundant_experts: int = 0       # Extra expert replicas
    log_balancedness: bool = False       # Performance logging
    log_balancedness_interval: int = 1   # Logging frequency
    use_async: bool = False              # Non-blocking EPLB
    policy: EPLBPolicyOption = "default"
```

**FusedMoEParallelConfig (config.py Lines 925-1131):**
```python
@dataclass
class FusedMoEParallelConfig:
    tp_size, pcp_size, dp_size, ep_size: int  # Parallelism sizes
    tp_rank, pcp_rank, dp_rank, ep_rank: int  # Current ranks
    use_ep: bool                              # EP enabled
    all2all_backend: str                      # Communication strategy
    enable_eplb: bool                         # Load balancing
    sp_size: int                              # Sequence parallel size

    @staticmethod
    def make(vllm_config, vllm_parallel_config) -> "FusedMoEParallelConfig":
        # Determines EP configuration based on parallelism setup
        # With EP: flattens TP across DP/PCP, sets ep_size and ep_rank
```

**Sequence Parallel for MoE (Lines 586-600):**
- Enabled when:
  - Using certain all2all backends
  - Expert parallelism enabled
  - TP size > 1 AND DP size > 1
- Purpose: Avoid duplicate computation from attention all_reduce

---

### SGLang - Expert Parallelism Configuration

**File:** `/sglang/python/sglang/srt/server_args.py` (Lines 494-518)

**Server Arguments:**
```python
# Expert parallelism settings
ep_size: int = 1
moe_a2a_backend: Literal[...] = "none"
moe_runner_backend: str = "auto"
deepep_mode: Literal["auto", "normal", "low_latency"] = "auto"
ep_num_redundant_experts: int = 0
ep_dispatch_algorithm: Optional[Literal["static", "dynamic", "fake"]] = None
init_expert_location: str = "trivial"

# EPLB settings
enable_eplb: bool = False
eplb_algorithm: str = "auto"
eplb_rebalance_num_iterations: int = 1000
eplb_rebalance_layers_per_chunk: Optional[int] = None
eplb_min_rebalancing_utilization_threshold: float = 1.0
expert_distribution_recorder_mode: Optional[...] = None
expert_distribution_recorder_buffer_size: Optional[int] = None
enable_expert_distribution_metrics: bool = False
```

**All2All Backends (Lines 495-497):**
- `none` (default): All-Reduce/All-Gather, bypass dispatch
- `deepep`: DeepEP optimized token shuffling
- `mooncake`: Elastic inference with RDMA
- `mori`: AMD ROCm native all-to-all
- `flashinfer`: FlashInfer implementation
- `ascend_fuseep`: Ascend NPU fused operator

**MoE Runner Backends (Lines 498):**
- `auto` (default): Hardware/model-aware auto-selection
- `triton`: Triton-based grouped GEMMs
- `deep_gemm`: DeepGEMM optimized with FP8
- `cutlass`: CUTLASS-based GEMMs
- `flashinfer_trtllm`: FlashInfer + TensorRT-LLM
- `flashinfer_cutclass`: FlashInfer + CUTLASS
- `flashinfer_mxfp4`: MXFP4 quantization variant

**DeepEP Mode (Lines 501):**
- `auto`: Automatic dispatch mode switching
- `normal`: Optimized for prefill (high throughput)
- `low_latency`: Optimized for decode (CUDA Graph compatible)

---

### TensorRT-LLM - No Native EP

TensorRT-LLM does not have a native expert parallelism implementation like vLLM or SGLang. Instead:
- Uses standard tensor parallelism for MoE layers
- Load balancing through routing method selection
- Host memory sharing for expert weights
- Static and dynamic routing modes

---

## Load Balancing Algorithms

### vLLM - EPLB (Expert Parallel Load Balancing)

**File:** `/vllm/distributed/eplb/eplb_state.py`

**EplbModelState (Lines 89-199):**
```python
@dataclass
class EplbModelState:
    # Mapping structures
    physical_to_logical_map: torch.Tensor  # (num_moe_layers, num_physical_experts)
    logical_to_physical_map: torch.Tensor  # (num_moe_layers, num_logical_experts, num_redundant_experts+1)
    logical_replica_count: torch.Tensor    # (num_moe_layers, num_logical_experts)

    # Load tracking
    expert_load_pass: torch.Tensor         # Current forward pass load
    expert_load_window: torch.Tensor       # Sliding window of loads
    expert_buffer: List[torch.Tensor]      # Weight transfer buffers
```

**Example Mapping:**
```python
# 2-layer model, 6 physical experts, 4 logical experts, 3 EP ranks
physical_to_logical_map = [
    [0, 1, 2, 3, 0, 1],  # Layer 0
    [0, 2, 0, 1, 0, 3]   # Layer 1
]

logical_to_physical_map = [
    [[0, 4, -1], [1, 5, -1], [2, -1, -1], [3, -1, -1]],  # Layer 0
    [[0, 2, 4], [3, -1, -1], [1, -1, -1], [5, -1, -1]]   # Layer 1
]

logical_replica_count = [
    [2, 2, 1, 1],  # Layer 0
    [3, 1, 1, 1]   # Layer 1
]
```

**EPLB Mapping Function:**
- **File:** `router/base_router.py` (Lines 17-96)
- **Function:** `eplb_map_to_physical_and_record()`
- **Algorithm:**
  1. Convert logical expert IDs to physical IDs
  2. Use position-based modulo for pseudo-random replica selection
  3. Record expert load metrics
  4. Returns physical expert IDs

**DefaultEplbPolicy:**
- **File:** `/vllm/distributed/eplb/policy/default.py` (Lines 21-101+)
- **balanced_packing():** Greedy bin packing for balanced load
- **replicate_experts():** Creates redundant experts for popular experts

---

### SGLang - EPLB Manager

**File:** `/sglang/python/sglang/srt/eplb/eplb_manager.py` (Lines 16-119)

**EPLBManager Class:**
```python
class EPLBManager:
    def __init__(self, model_runner):
        self.model_runner = model_runner
        self.rebalance_num_iterations = server_args.eplb_rebalance_num_iterations
        self.min_utilization_threshold = server_args.eplb_min_rebalancing_utilization_threshold

    def on_forward_pass_end(self):
        # Trigger rebalancing check

    def rebalance(self):
        # Execute rebalancing operation
        # 1. Check if rebalance needed (utilization-based)
        # 2. Compute updated layer IDs and chunks
        # 3. Apply new expert location mapping

    def _check_rebalance_needed(self) -> bool:
        # Utilization-based decision
        # Checks mean/max ratio against threshold
```

**Expert Distribution Recording:**
- **File:** `/sglang/python/sglang/srt/eplb/expert_distribution.py` (Lines 1-1049)
- **ExpertDistributionRecorder** (Lines 55-279)
  - `start_record()`: Begin tracking
  - `stop_record()`: Stop recording
  - `dump_record()`: Export data
  - Hooks for expert selection and dispatch events

**Load Balancing Metrics (Lines 1020-1048):**
```python
def compute_utilization_rate(gpu_physical_count):
    mean_gpu_count = gpu_physical_count.mean()
    max_gpu_count = gpu_physical_count.max()
    balancedness = (mean_gpu_count + 1e-5) / (max_gpu_count + 1e-5)
    return balancedness
```

**Expert Location Management:**
- **File:** `/sglang/python/sglang/srt/eplb/expert_location.py` (Lines 1-574)

**ExpertLocationMetadata (Lines 39-82):**
```python
@dataclass
class ExpertLocationMetadata:
    physical_to_logical_map: torch.Tensor
    logical_to_all_physical_map: torch.Tensor
    logical_to_rank_dispatch_physical_map: torch.Tensor
```

**Location Initialization:**
- `init_trivial()`: Default 1:1 mapping
- `init_by_mapping()`: Custom mapping
- `init_by_eplb()`: EPLB-computed optimal arrangement

**Location Dispatch Algorithm:**
- **File:** `expert_location_dispatch.py` (Lines 76-109)
- **Static:** Direct mapping via rank dispatch map
- **Dynamic:** Random selection from candidates with locality preference
- **Nearest Expert:** Same-GPU > Same-node > Any-node

---

### TensorRT-LLM - MoeLoadBalancer

**File:** `/TensorRT-LLM/tensorrt_llm/_torch/modules/fused_moe/moe_load_balancer.py` (Lines 1-1202)

**SingleLayerMoeLoadBalancer (Lines 374-832):**
```python
class SingleLayerMoeLoadBalancer:
    def __init__(self, num_experts, ep_size, ep_rank, updates_enabled=True):
        # Wraps C++ implementation for single MoE layer
        # updates_enabled: Static (False) vs Dynamic (True) routing

    def start_wait_gpu_stage(self):
        # Initiates GPU stage with async stream support

    def done_wait_gpu_stage(self):
        # Completes GPU wait stage

    def update_local_statistic(self, expert_ids):
        # Records expert statistics per token

    def get_local_statistic_tensor(self):
        # Retrieves statistics for all-reduce

    def route(self, expert_ids):
        # Routes tokens to experts using load balancer
```

**MoeLoadBalancer (Lines 839-1177):**
```python
class MoeLoadBalancer:
    def __init__(self, ep_rank, ep_size, num_layers_to_update_per_iter=1):
        # Main coordinator for all MoE layers

    def add_layer(self, layer_id, num_experts, updates_enabled=True):
        # Creates SingleLayerMoeLoadBalancer for each layer

    def finalize_model(self):
        # Synchronizes shared memory after all layers added

    def start_iter(self):
        # Iteration start boundary

    def end_iter(self):
        # Iteration end boundary

    def shutdown(self):
        # Cleanup and MPI barrier
```

**HostMoeTensorSharer (Lines 127-372):**
```python
class HostMoeTensorSharer:
    # Manages shared memory for expert weights across local ranks

    def share_host_tensor_with_shape(self, name, shape, dtype):
        # Shares weights from loader

    def finalize_layer_weights(self, layer_id):
        # Creates shared memory buffer
        # Name pattern: {base_name}_l{layer_id}_lr{rank}_all

    def finalize_host_tensor_sharing(self):
        # Maps remote shared memory
```

**Load Balancing Strategy:**
- Track expert utilization per layer
- Periodically migrate expert weights via shared memory
- Uses MPI for coordination
- Supports both static (no updates) and dynamic (periodic updates) modes

---

## Communication Backends

### vLLM - All2All Backends

**Configuration:** `all2all_backend` in ParallelConfig (Lines 162-171)

**Available Backends:**
1. **"naive":** Basic broadcasts
2. **"allgather_reducescatter":** Default, AllGather + ReduceScatter
3. **"deepep_high_throughput":** DeepEP high-throughput kernels
4. **"deepep_low_latency":** DeepEP low-latency kernels
5. **"mori":** MORI kernels
6. **"nixl_ep":** NIXL-EP kernels
7. **"flashinfer_nvlink_two_sided":** FlashInfer two-sided
8. **"flashinfer_nvlink_one_sided":** FlashInfer one-sided (high throughput)

**DeepEP Prepare/Finalize:**
- `DeepEPHTPrepareAndFinalize` - High throughput
- `DeepEPLLPrepareAndFinalize` - Low latency

**FlashInfer Communication:**
- `FlashInferNVLinkTwoSidedPrepareAndFinalize`
- `FlashInferNVLinkOneSidedPrepareAndFinalize`

---

### SGLang - All2All Backends

**Configuration:** `--moe-a2a-backend` (server_args.py Lines 495-497)

**Available Backends:**
1. **"none"** (default): All-Reduce/All-Gather for EP
2. **"deepep":** DeepEP optimized token shuffling
3. **"mooncake":** Elastic inference with RDMA
4. **"mori":** AMD ROCm native all-to-all
5. **"flashinfer":** FlashInfer implementation
6. **"ascend_fuseep":** Ascend NPU fused operator

**Computation Overlap Strategies:**

**Two-Batch Overlap (TBO):**
- Splits requests into micro-batches
- Interleaves attention computation with dispatch/combine
- Uses `YieldOperation()` for pause points
- Enable: `--enable-two-batch-overlap`
- Potential: 2x throughput increase

**Single-Batch Overlap (SBO):**
- Dispatcher-hook system for within-batch overlap
- Shared expert computation with communication
- Enable: `--enable-single-batch-overlap`

---

### TensorRT-LLM - No Dedicated All2All

TensorRT-LLM does not have specialized All2All backends for expert parallelism. Instead:
- Uses standard collective operations
- MPI-based weight migration
- Host memory sharing for local ranks
- Focus on routing method optimization

---

## Supported MoE Models

### vLLM - MoE Models

**File:** `/vllm/model_executor/models/` (various `*moe*.py` files)

**Supported Models:**

1. **DeepSeek Family** (`deepseek_v2.py`):
   - DeepSeek-V2, DeepSeek-V3
   - Uses DeepSeekV3 routing (sigmoid + groups)
   - Multi-head latent attention (MLA)

2. **Qwen Family:**
   - `qwen2_moe.py`: Standard topk routing
   - `qwen3_moe.py`: Advanced routing with bias
   - `qwen3_vl_moe.py`: Vision-language variant
   - `qwen3_omni_moe_thinker.py`: Extended reasoning

3. **GLM Family:**
   - `glm4_moe.py`, `glm4_moe_lite.py`
   - `glm4_moe_lite_mtp.py`, `glm4_moe_mtp.py`: With MTP

4. **Others:**
   - Mixtral (`mixtral.py`)
   - PhiMoE (`phimoe.py`)
   - OLMoE (`olmoe.py`)
   - Exaone MoE (`exaone_moe.py`)
   - ERNIE45 MoE (`ernie45_moe.py`)
   - AFMoE (`afmoe.py`)
   - LFM2 MoE (`lfm2_moe.py`)
   - GraniteMoE variants (`granitemoe*.py`)
   - Bailing MoE (`bailing_moe*.py`)

---

### SGLang - MoE Models

**Supported Models** (have `get_model_config_for_expert_location()` method):
- Bailing MOE
- Deepseek V2/V3/R1
- Exaone MOE
- GLM4 MOE (multiple variants)
- GPT-OSS
- Llada2
- Longcat Flash
- MIMO V2 Flash
- Qwen2 MOE
- Qwen3 MOE (multiple variants)
- Granite MOE (regular + hybrid)
- LFM2 MOE
- Phimoe
- Olmoe
- SDAR MOE
- Xverse MOE
- Ernie45 MOE VL

---

### TensorRT-LLM - Routing Method Support

**Models by Routing Method:**

**Default (Softmax → TopK):**
- Mixtral
- Generic MoE models

**DeepSeekV3 (Sigmoid → Grouped):**
- DeepSeek-V2
- DeepSeek-V3

**Llama4 (Top1 → Sigmoid):**
- Llama4 MoE variants

**MiniMax2 (Sigmoid → Bias → TopK):**
- MiniMax-M2

---

## Configuration & Command-Line Options

### vLLM Configuration

**Command-Line Arguments:**
```bash
# Expert Parallelism
--enable-expert-parallel         # Enable EP instead of TP for MoE
--enable-ep-weight-filter        # Skip non-local expert weights during loading
--all2all-backend                # Select all2all kernel
--expert-placement-strategy      # "linear" or "round_robin"

# Load Balancing
--enable-eplb                    # Enable expert load balancing
--enable-elastic-ep              # Enable elastic expert parallelism

# EPLB Configuration
--eplb-window-size 1000          # Expert load recording window
--eplb-step-interval 3000        # Rearrangement interval (tokens)
--eplb-num-redundant-experts 0   # Extra expert replicas
--eplb-log-balancedness          # Performance logging

# Standard Parallelism
--tensor-parallel-size           # TP size
--pipeline-parallel-size         # PP size
--data-parallel-size             # DP size
```

**Python API:**
```python
from vllm import LLM
from vllm.config import ParallelConfig, EPLBConfig

parallel_config = ParallelConfig(
    tensor_parallel_size=1,
    data_parallel_size=8,
    enable_expert_parallel=True,
    expert_placement_strategy="round_robin",
    all2all_backend="deepep_high_throughput",
    enable_eplb=True,
    eplb_config=EPLBConfig(
        window_size=1000,
        step_interval=3000,
        num_redundant_experts=2,
    )
)

llm = LLM(
    model="deepseek-ai/DeepSeek-V3",
    parallel_config=parallel_config,
)
```

---

### SGLang Configuration

**Command-Line Arguments:**
```bash
# Expert Parallelism
--ep-size 8                              # EP group size
--moe-a2a-backend deepep                 # All2All backend
--moe-runner-backend deep_gemm           # MoE computation backend
--deepep-mode auto                       # DeepEP dispatch mode

# EPLB
--enable-eplb                            # Enable load balancing
--eplb-algorithm elasticity_aware        # EPLB algorithm
--eplb-rebalance-num-iterations 1000     # Rebalancing interval
--eplb-min-rebalancing-utilization-threshold 1.0

# Expert Distribution
--ep-num-redundant-experts 2             # Redundant experts
--ep-dispatch-algorithm dynamic          # static/dynamic/fake
--init-expert-location trivial           # Initial location strategy
--enable-expert-distribution-metrics     # Track metrics

# Overlap Strategies
--enable-two-batch-overlap               # TBO
--enable-single-batch-overlap            # SBO
```

**Example Usage:**
```bash
# DeepSeek-V3 with DeepEP and EPLB
python -m sglang.launch_server \
  --model-path deepseek-ai/DeepSeek-V3 \
  --moe-a2a-backend deepep \
  --moe-runner-backend deep_gemm \
  --tp 8 \
  --ep-size 8 \
  --enable-eplb \
  --eplb-algorithm elasticity_aware

# With speculative decoding
python -m sglang.launch_server \
  --model-path nvidia/DeepSeek-R1-0528-NVFP4-v2 \
  --moe-runner-backend flashinfer_trtllm \
  --speculative-moe-runner-backend triton
```

---

### TensorRT-LLM Configuration

**Python API:**
```python
# Load balancing configuration
from tensorrt_llm._torch.modules.fused_moe.moe_load_balancer import MoeLoadBalancer

load_balancer = MoeLoadBalancer(
    ep_rank=0,
    ep_size=8,
    num_layers_to_update_per_iter=1
)

# Add layers
for layer_id in range(num_moe_layers):
    load_balancer.add_layer(
        layer_id=layer_id,
        num_experts=num_experts,
        updates_enabled=True  # Dynamic routing
    )

load_balancer.finalize_model()
```

**Routing Method Selection:**
```python
from tensorrt_llm._torch.modules.fused_moe.routing import (
    DefaultMoeRoutingMethod,
    DeepSeekV3MoeRoutingMethod,
    LoadBalancedMoeRoutingMethod,
)

# Load balanced routing
routing_method = LoadBalancedMoeRoutingMethod(
    num_experts=num_experts,
    experts_per_token=top_k,
    moe_ep_size=ep_size,
    moe_ep_rank=ep_rank,
)

# DeepSeek V3 routing
routing_method = DeepSeekV3MoeRoutingMethod(
    num_experts=num_experts,
    num_shared_experts=num_shared_experts,
    experts_per_token=top_k,
    ...
)
```

---

## Code Sources & Implementation Details

### vLLM Key Files

| Component | File Path | Lines | Key Classes/Functions |
|-----------|-----------|-------|----------------------|
| **Main MoE Layer** | `fused_moe/layer.py` | - | FusedMoE, determine_expert_map() |
| **Router Base** | `fused_moe/router/base_router.py` | 99-250 | BaseRouter, eplb_map_to_physical_and_record() |
| **TopK Router** | `fused_moe/router/fused_topk_router.py` | 116-166 | FusedTopKRouter, fused_topk() |
| **Grouped Router** | `fused_moe/router/grouped_topk_router.py` | - | GroupedTopKRouter, grouped_topk() |
| **Bias Router** | `fused_moe/router/fused_topk_bias_router.py` | 173-200+ | FusedTopKBiasRouter |
| **Gate Linear** | `fused_moe/router/gate_linear.py` | 12-118 | GateLinear |
| **Parallel Config** | `config/parallel.py` | 97-883 | ParallelConfig, EPLBConfig |
| **MoE Config** | `fused_moe/config.py` | 1-1265 | FusedMoEConfig, FusedMoEParallelConfig |
| **EPLB State** | `distributed/eplb/eplb_state.py` | - | EplbState, EplbModelState |
| **EPLB Policy** | `distributed/eplb/policy/default.py` | - | DefaultEplbPolicy |
| **DeepSeek V2** | `model_executor/models/deepseek_v2.py` | - | DeepseekV2MoE |

### SGLang Key Files

| Component | File Path | Lines | Key Classes/Functions |
|-----------|-----------|-------|----------------------|
| **Router** | `srt/layers/moe/router.py` | 1-429 | FusedMoeRouter |
| **TopK** | `srt/layers/moe/topk.py` | 1-1100 | TopK, grouped_topk() |
| **Expert Distribution** | `srt/eplb/expert_distribution.py` | 1-1049 | ExpertDistributionRecorder |
| **Expert Location** | `srt/eplb/expert_location.py` | 1-574 | ExpertLocationMetadata |
| **Location Dispatch** | `srt/eplb/expert_location_dispatch.py` | 1-110 | ExpertLocationDispatchInfo |
| **EPLB Manager** | `srt/eplb/eplb_manager.py` | 1-119 | EPLBManager |
| **Server Args** | `srt/server_args.py` | 494-518 | Expert parallel arguments |

### TensorRT-LLM Key Files

| Component | File Path | Lines | Key Classes/Functions |
|-----------|-----------|-------|----------------------|
| **Routing Methods** | `_torch/modules/fused_moe/routing.py` | 1-805 | RoutingMethodType, BaseMoeRoutingMethod |
| **Load Balancer** | `_torch/modules/fused_moe/moe_load_balancer.py` | 1-1202 | MoeLoadBalancer, SingleLayerMoeLoadBalancer |
| **Host Sharer** | `_torch/modules/fused_moe/moe_load_balancer.py` | 127-372 | HostMoeTensorSharer |

---

## Feature Comparison Matrix

| Feature | vLLM | SGLang | TensorRT-LLM |
|---------|------|--------|--------------|
| **Expert Parallelism** | ✅ Full EP support | ✅ Full EP support | ❌ TP for MoE only |
| **Load Balancing** | ✅ EPLB | ✅ EPLB Manager | ✅ MoeLoadBalancer |
| **Redundant Experts** | ✅ Configurable | ✅ Configurable | ❌ No |
| **Expert Rearrangement** | ✅ Dynamic | ✅ Utilization-based | ✅ Weight migration |
| **Expert Placement** | ✅ Linear, Round-robin | ✅ Trivial, Custom, EPLB | ❌ N/A |
| **Routing Methods** | ✅ 8 types | ✅ Softmax, Sigmoid, Grouped | ✅ 7 types |
| **All2All Backends** | ✅ 8 backends | ✅ 6 backends | ❌ No |
| **DeepEP Support** | ✅ Yes | ✅ Yes | ❌ No |
| **FlashInfer Support** | ✅ Yes | ✅ Yes | ❌ No |
| **Mooncake Support** | ❌ No | ✅ Yes | ❌ No |
| **Overlap Scheduling** | ❌ No | ✅ TBO + SBO | ❌ No |
| **Sequence Parallel MoE** | ✅ Yes | ❌ No | ❌ No |
| **Locality-Aware Dispatch** | ❌ No | ✅ GPU > Node > Remote | ❌ No |
| **Expert Distribution Metrics** | ✅ Load tracking | ✅ Recording + metrics | ✅ Statistics tensor |
| **Shared Memory** | ❌ No | ❌ No | ✅ HostMoeTensorSharer |
| **MPI Integration** | ❌ No | ❌ No | ✅ Weight migration |
| **Static Routing** | ✅ Simulated | ✅ Static dispatch | ✅ Updates disabled |
| **Dynamic Routing** | ✅ EPLB | ✅ Dynamic dispatch | ✅ Updates enabled |
| **GPU-Aware Balancing** | ✅ EPLB policy | ✅ Utilization threshold | ✅ Round-robin |
| **Number of Models** | ✅ 20+ | ✅ 25+ | ✅ Routing method support |

---

## Best Practices & Recommendations

### When to Use Expert Parallelism

**Use EP when:**
- MoE model with many experts (>8)
- High expert count per layer
- Memory constraints prevent full TP
- Workload benefits from expert locality

**Use TP when:**
- Small number of experts (<8)
- Dense model with few MoE layers
- Simpler deployment preferred

### Load Balancing Configuration

**vLLM EPLB:**
```python
# Aggressive rebalancing
eplb_config = EPLBConfig(
    window_size=500,              # Smaller window for faster adaptation
    step_interval=1000,           # Frequent rearrangement
    num_redundant_experts=3,      # More redundancy
)

# Conservative rebalancing
eplb_config = EPLBConfig(
    window_size=2000,             # Larger window for stability
    step_interval=5000,           # Less frequent rearrangement
    num_redundant_experts=1,      # Minimal redundancy
)
```

**SGLang EPLB:**
```bash
# High-throughput deployment
python -m sglang.launch_server \
  --enable-eplb \
  --eplb-rebalance-num-iterations 500 \
  --eplb-min-rebalancing-utilization-threshold 0.9 \
  --enable-two-batch-overlap

# Latency-sensitive deployment
python -m sglang.launch_server \
  --enable-eplb \
  --deepep-mode low_latency \
  --eplb-rebalance-num-iterations 2000
```

### Backend Selection

**vLLM:**
- **High throughput:** `deepep_high_throughput` or `flashinfer_nvlink_one_sided`
- **Low latency:** `deepep_low_latency` or `flashinfer_nvlink_two_sided`
- **Compatibility:** `allgather_reducescatter` (default)

**SGLang:**
- **NVIDIA:** `deepep` or `flashinfer`
- **AMD ROCm:** `mori`
- **RDMA clusters:** `mooncake`
- **NPU:** `ascend_fuseep`

---

## Conclusion

All three systems provide MoE support with different approaches:

**vLLM** excels in:
- Comprehensive expert parallelism
- Dynamic load balancing (EPLB)
- Multiple routing methods
- Redundant expert support
- Sequence parallel MoE

**SGLang** excels in:
- Locality-aware expert dispatch
- Overlap scheduling (TBO/SBO)
- Mooncake elastic inference
- Expert distribution metrics
- Multiple communication backends

**TensorRT-LLM** excels in:
- Sophisticated routing methods
- Host memory sharing
- MPI-based weight migration
- Load balanced routing
- TensorRT optimization

The choice depends on:
1. **Model characteristics:** Number of experts, layers, size
2. **Infrastructure:** Single-node vs multi-node, interconnect type
3. **Workload:** Throughput vs latency, batch size
4. **Deployment:** Static vs dynamic load patterns
5. **Hardware:** NVIDIA vs AMD, NVLink vs InfiniBand
