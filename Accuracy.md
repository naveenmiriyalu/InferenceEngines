# Accuracy Testing and Validation: vLLM, SGLang, and TensorRT-LLM

## Table of Contents
1. [Executive Summary](#executive-summary)
2. [vLLM Accuracy Testing](#1-vllm-accuracy-testing)
3. [SGLang Accuracy Testing](#2-sglang-accuracy-testing)
4. [TensorRT-LLM Accuracy Testing](#3-tensorrt-llm-accuracy-testing)
5. [Comparative Analysis](#4-comparative-analysis)
6. [Best Practices](#5-best-practices)

---

## Executive Summary

This document compares accuracy testing and validation infrastructure across three major LLM serving frameworks. Each implements different but complementary approaches to ensure model correctness:

- **vLLM**: Exact token matching, logprob validation, batch invariance testing, tolerance-based benchmarks
- **SGLang**: Baseline threshold validation, per-commit accuracy suites, GitHub integration
- **TensorRT-LLM**: Statistical hypothesis testing, configuration-specific baselines, comprehensive reference data

**Key Findings:**
- **vLLM** provides the most comprehensive exact-match validation with HuggingFace baselines
- **TensorRT-LLM** offers the most statistically rigorous approach using hypothesis testing
- **SGLang** excels at continuous validation with per-commit accuracy tracking
- All frameworks support multiple precision modes (FP32/FP16/BF16/FP8)

---

## 1. vLLM Accuracy Testing

### 1.1 Testing Infrastructure

#### Core Test Directories
- `vllm/tests/entrypoints/llm/test_accuracy.py` - LMEval-based accuracy tests
- `vllm/tests/basic_correctness/` - Basic correctness validation
- `vllm/tests/models/utils.py` - Core test utilities
- `vllm/tests/v1/determinism/` - Determinism validation

---

### 1.2 Validation Methods

#### 1.2.1 Token-Level Exact Matching

**File:** `vllm/tests/models/utils.py` (Lines 25-52)

```python
def check_outputs_equal(
    *,
    outputs_0_lst: Sequence[TokensText],
    outputs_1_lst: Sequence[TokensText],
    name_0: str,
    name_1: str,
):
    """Compare two sequences - must be exactly equal"""
    for prompt_idx, (outputs_0, outputs_1) in enumerate(
        zip(outputs_0_lst, outputs_1_lst)
    ):
        output_ids_0, output_str_0 = outputs_0
        output_ids_1, output_str_1 = outputs_1

        # Both text and token outputs must match exactly
        assert output_str_0 == output_str_1, \
            f"Text mismatch at prompt {prompt_idx}"
        assert output_ids_0 == output_ids_1, \
            f"Token ID mismatch at prompt {prompt_idx}"
```

**Use Cases:**
- Greedy decoding (deterministic sampling)
- Output validation against HuggingFace baseline
- Model executor comparisons (V0 vs V1 engine)

---

#### 1.2.2 Logprob Closeness Checking

**File:** `vllm/tests/models/utils.py` (Lines 91-271)

```python
def check_logprobs_close(
    *,
    outputs_0_lst,
    outputs_1_lst,
    name_0: str,
    name_1: str,
    warn_on_mismatch: bool = False,
    num_outputs_0_skip_tokens: int = 0,
    always_check_logprobs: bool = False,
):
    """Validate top-K logprobs without requiring exact token matching"""
    # Allows token divergence with warnings
    # Optional: always_check_logprobs for strict validation
```

**Features:**
- Top-K logprob validation without exact token matching
- Supports prompt and sample logprobs
- Handles token mismatches gracefully with warnings
- Optional strict mode: `always_check_logprobs`

**Parameters:**
- `warn_on_mismatch` (bool): Issue warnings on token-wise mismatches
- `num_outputs_0_skip_tokens` (int): Skip initial tokens from first sequence
- `always_check_logprobs` (bool): Check logprobs even when tokens match

---

#### 1.2.3 Embedding Similarity

**File:** `vllm/tests/models/utils.py` (Lines 330-358)

```python
def check_embeddings_close(
    *,
    embeddings_0_lst: Sequence[list[float]],
    embeddings_1_lst: Sequence[list[float]],
    name_0: str,
    name_1: str,
    tol: float = 1e-3,
) -> None:
    """Compare embeddings using cosine similarity"""
    for embeddings_0, embeddings_1 in zip(embeddings_0_lst, embeddings_1_lst):
        sim = F.cosine_similarity(
            torch.tensor(embeddings_0),
            torch.tensor(embeddings_1),
            dim=0
        )
        assert sim >= 1 - tol, \
            f"Cosine similarity {sim:.4f} < threshold {1-tol:.4f}"
```

**Tolerance:** Default `tol=1e-3` (0.999 cosine similarity required)

---

### 1.3 Tolerance Thresholds

#### 1.3.1 LMEval Accuracy Tests

**File:** `vllm/tests/entrypoints/llm/test_accuracy.py` (Lines 27-31)

```python
RTOL = 0.03  # ±3% tolerance for benchmark accuracy

EXPECTED_VALUES = {
    "Qwen/Qwen3-1.7B": 0.68,
    "google/gemma-3-1b-it": 0.25,
}

# Validation: expected_value - RTOL < measured_value < expected_value + RTOL
def test_lm_eval_accuracy(model):
    measured_value = run_lm_eval(model)
    expected_value = EXPECTED_VALUES[model]
    assert (
        measured_value - RTOL < expected_value
        and measured_value + RTOL > expected_value
    )
```

---

#### 1.3.2 Numerical Tolerance in Fusion Tests

**File:** `vllm/tests/compile/passes/test_fusion.py` (Lines 284-288)

```python
if FP8_DTYPE == torch.float8_e4m3fn:
    ATOL, RTOL = (2e-3, 2e-3)  # FP8: 0.2% tolerance
else:
    ATOL, RTOL = (1e-2, 1e-2)  # BF16: 1% tolerance

torch.testing.assert_close(
    result_fused,
    result_unfused,
    atol=ATOL,
    rtol=RTOL
)
```

---

#### 1.3.3 Head Dtype Testing

**File:** `vllm/tests/models/language/pooling/test_head_dtype.py` (Line 47)

```python
assert torch.allclose(hf_output, vllm_output, atol=1e-2)  # 1% tolerance
```

---

#### 1.3.4 Mixed Precision Models

**File:** `vllm/tests/quantization/test_mixed_precision.py` (Lines 63-70)

```python
rtol = 0.05  # ±5% tolerance

for task, expect_accuracy in accuracy_numbers.items():
    measured_accuracy = results["results"][task]["acc,none"]
    assert (
        measured_accuracy - rtol < expect_accuracy
        and measured_accuracy + rtol > expect_accuracy
    ), f"Accuracy {measured_accuracy} outside tolerance for {task}"
```

---

### 1.4 Determinism Support

#### 1.4.1 Batch Invariance Testing

**File:** `vllm/tests/v1/determinism/test_batch_invariance.py` (Lines 24-130)

**Purpose:** Ensures identical outputs regardless of batch composition

**Strategy:**
1. Generate baseline output for "needle" prompt (bs=1)
2. Mix needle into random batches with filler prompts
3. Verify deterministic output across multiple trials
4. Environmental controls for reproducibility

```python
# Environmental controls
seed = int(os.getenv("VLLM_TEST_SEED", "12345"))
random.seed(seed)

# Sampling with fixed seed
sampling = SamplingParams(
    temperature=temperature,
    top_p=top_p,
    max_tokens=max_tokens,
    seed=20240919,
)

# Run multiple trials
for trial in range(num_trials):
    # Generate random batch with needle embedded
    batch = generate_random_batch_with_needle(...)
    outputs = llm.generate(batch, sampling)

    # Verify needle output matches baseline
    needle_output = extract_needle_output(outputs)
    assert needle_output == baseline_output
```

---

#### 1.4.2 Determinism Utilities

**File:** `vllm/tests/v1/determinism/utils.py` (Lines 45-88)

```python
def _random_prompt(min_words: int = 1024, max_words: int = 1024 * 2) -> str:
    """Generate realistic, varied prompts for determinism testing"""
    prompt_templates = [
        "Question: What is the capital of France?\nAnswer:",
        "Once upon a time in a distant galaxy, there lived",
        "The algorithm works by iterating through the array and",
        # ... more templates
    ]
    # Randomly combine templates with filler words

def _extract_step_logprobs(request_output):
    """Extract logprobs for determinism verification"""
    if getattr(request_output, "outputs", None):
        inner = request_output.outputs[0]
        if hasattr(inner, "logprobs") and inner.logprobs is not None:
            t = torch.tensor([
                inner.logprobs[i][tid].logprob
                for i, tid in enumerate(inner.token_ids)
            ], dtype=torch.float32)
            return t, inner.token_ids
```

---

#### 1.4.3 Supported Backends

```python
BACKENDS: list[str] = [
    "FLASH_ATTN",
    "TRITON_ATTN",
    "TRITON_MLA",
    "FLASH_ATTN_MLA",
]
```

---

#### 1.4.4 Environment Variables

```python
VLLM_TEST_SEED          # Random seed for prompt generation (default: 12345)
VLLM_NEEDLE_TRIALS      # Number of batch invariance trials (default: 5)
VLLM_NEEDLE_BATCH_SIZE  # Max batch size (default: 128)
VLLM_NEEDLE_TEMPERATURE # Sampling temperature (default: 0.0)
VLLM_NEEDLE_TOP_P       # Top-P sampling (default: 0.95)
VLLM_NEEDLE_MAX_TOKENS  # Max generation length (default: 128)
VLLM_MAX_MODEL_LEN      # Max model length (default: 5120)
```

---

### 1.5 Precision Support

#### Supported Precisions
- **FP32** (default, full precision)
- **FP16** (half precision)
- **BF16** (bfloat16)
- **FP8** (with KV cache variants)

#### Precision-Specific Tests

**File:** `vllm/tests/entrypoints/llm/test_accuracy.py`

```python
@pytest.mark.parametrize("model", FP8_KV_MODEL_NAMES)
def test_lm_eval_accuracy_v1_engine_fp8_kv_cache(model):
    """Test with FP8 KV cache quantization"""
    more_args = "kv_cache_dtype=fp8" if platform.is_tpu() else None
    run_test(model, more_args)
```

---

### 1.6 Test Runner Architecture

#### HfRunner (HuggingFace Baseline)

**File:** `vllm/tests/conftest.py` (Lines 294-760)

```python
class HfRunner:
    """Baseline runner using HuggingFace transformers"""

    def get_default_device(self):
        return "cpu" if current_platform.is_cpu() else current_platform.device_type

    def __init__(
        self,
        model_name: str,
        dtype: str = "auto",
        trust_remote_code: bool = True,
        # ... additional parameters
    ):
        self.model = AutoModelForCausalLM.from_pretrained(model_name, ...)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, ...)

    def generate_greedy(
        self,
        prompts: list[str],
        max_tokens: int,
    ) -> list[TokensText]:
        """Generate greedily (deterministic)"""
        # Use transformers.generate() with greedy decoding
```

---

#### VllmRunner (vLLM Test Instance)

**File:** `vllm/tests/conftest.py` (Lines 767-850)

```python
class VllmRunner:
    """Default parameters modified from LLM() for testing:
    - trust_remote_code: True (convenience)
    - seed: 0 (reproducibility)
    - max_model_len: 1024 (memory)
    - block_size: 16 (memory)
    - enable_chunked_prefill: False (reproducibility)
    - enforce_eager: False (test CUDA graphs)
    """

    def __init__(
        self,
        model_name: str,
        runner: RunnerOption = "auto",
        seed: int = 0,  # Reproducible
        max_model_len: int | None = 1024,
        dtype: str = "auto",
        tensor_parallel_size: int = 1,
    ):
        self.llm = LLM(model_name, seed=seed, ...)
```

---

#### Test Pattern

```python
@pytest.mark.parametrize("model", MODELS)
def test_models(hf_runner, vllm_runner, model: str):
    """Compare vLLM outputs against HuggingFace baseline"""

    # Generate baseline outputs with HuggingFace
    with hf_runner(model) as hf_model:
        hf_outputs = hf_model.generate_greedy(prompts, max_tokens)

    # Generate vLLM outputs
    with vllm_runner(model) as vllm_model:
        vllm_outputs = vllm_model.generate_greedy(prompts, max_tokens)

    # Validate exact match
    check_outputs_equal(
        outputs_0_lst=hf_outputs,
        outputs_1_lst=vllm_outputs,
        name_0="hf",
        name_1="vllm",
    )
```

---

### 1.7 Seed Management

**File:** `vllm/tests/test_seed_behavior.py` (Lines 11-25)

```python
def test_seed_behavior():
    """Verify seed produces reproducible results"""

    Platform.seed_everything(42)
    random_value_1 = random.randint(0, 100)
    np_random_value_1 = np.random.randint(0, 100)
    torch_random_value_1 = torch.randint(0, 100, (1,)).item()

    Platform.seed_everything(42)
    random_value_2 = random.randint(0, 100)
    np_random_value_2 = np.random.randint(0, 100)
    torch_random_value_2 = torch.randint(0, 100, (1,)).item()

    # Same seed produces identical values
    assert random_value_1 == random_value_2
    assert np_random_value_1 == np_random_value_2
    assert torch_random_value_1 == torch_random_value_2
```

---

## 2. SGLang Accuracy Testing

### 2.1 Test Infrastructure

#### Core Framework

**File:** `sglang/python/sglang/test/accuracy_test_runner.py` (Lines 1-288)

```python
@dataclass
class AccuracyTestParams:
    """Parameters for accuracy testing"""
    dataset: str                            # e.g., "mgsm_en", "gsm8k", "mmmu", "gpqa"
    baseline_accuracy: float                # Minimum accuracy threshold
    num_examples: Optional[int] = None
    num_threads: Optional[int] = None
    max_tokens: Optional[int] = None
    return_latency: bool = False
    thinking_mode: Optional[str] = None     # e.g., "deepseek-v3"
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    repeat: Optional[int] = None

@dataclass
class AccuracyTestResult:
    """Result of an accuracy test"""
    model: str
    dataset: str
    passed: bool
    score: Optional[float]
    baseline_accuracy: float
    error: Optional[str]
    latency: Optional[float] = None
    variant: Optional[str] = None
```

---

### 2.2 Evaluation Methods

#### Few-Shot GSM8K Evaluation

**File:** `sglang/python/sglang/test/few_shot_gsm8k.py` (Lines 22-120)

```python
def get_few_shot_examples(lines, k):
    """Build few-shot examples for GSM8K"""
    ret = ""
    for i in range(k):
        ret += get_one_example(lines, i, True) + "\n\n"
    return ret

def get_answer_value(answer_str):
    """Extract numeric answer from generated text"""
    answer_str = answer_str.replace(",", "")
    numbers = re.findall(r"\d+", answer_str)
    if len(numbers) < 1:
        return INVALID
    try:
        return ast.literal_eval(numbers[-1])
    except SyntaxError:
        return INVALID

def run_eval(args):
    """Run few-shot evaluation"""
    # Compute accuracy
    acc = np.mean(np.array(preds) == np.array(labels))
    invalid = np.mean(np.array(preds) == INVALID)

    # Compute throughput
    num_output_tokens = sum(
        s.get_meta_info("answer")["completion_tokens"] for s in states
    )
    output_throughput = num_output_tokens / latency

    return {
        "accuracy": acc,
        "invalid_rate": invalid,
        "throughput": output_throughput,
    }
```

---

#### Test Kit Pattern

**File:** `sglang/python/sglang/test/kits/gsm8k_accuracy_kit.py` (Lines 1-38)

```python
class GSM8KMixin:
    """Mixin for GSM8K accuracy testing"""
    gsm8k_accuracy_thres: float
    gsm8k_accept_length_thres: Optional[float] = None
    gsm8k_num_questions: int = 200
    gsm8k_parallel: int = 128

    def test_gsm8k(self):
        # Flush cache before test
        requests.get(self.base_url + "/flush_cache")

        args = SimpleNamespace(
            num_shots=5,
            data_path=None,
            num_questions=self.gsm8k_num_questions,
            max_new_tokens=512,
            parallel=self.gsm8k_parallel,
            host="http://127.0.0.1",
            port=int(self.base_url.split(":")[-1]),
        )

        metrics = run_eval_gsm8k(args)

        # Validate accuracy threshold
        self.assertGreaterEqual(metrics["accuracy"], self.gsm8k_accuracy_thres)

        # Optional: Validate speculative accept length
        if self.gsm8k_accept_length_thres is not None:
            server_info = requests.get(self.base_url + "/server_info")
            avg_spec_accept_length = server_info.json()["internal_states"][0][
                "avg_spec_accept_length"
            ]
            self.assertGreater(
                avg_spec_accept_length, self.gsm8k_accept_length_thres
            )
```

---

### 2.3 Evaluation Backends

**File:** `sglang/python/sglang/test/run_eval.py` (Lines 61-100)

**Supported Datasets:**
- `mmlu` - Multiple choice QA
- `math` - Mathematical reasoning
- `mgsm` / `mgsm_en` - Multilingual GSM8K
- `gpqa` - Graduate-level physics
- `gsm8k` - Grade school math

```python
def _run_simple_eval(
    model: ModelLaunchSettings,
    base_url: str,
    dataset: str,
    num_examples: Optional[int] = None,
    num_threads: Optional[int] = None,
    max_tokens: Optional[int] = None,
    return_latency: bool = False,
    thinking_mode: Optional[str] = None,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
) -> Tuple[bool, Optional[str], Optional[dict]]:
    """Run evaluation using simple_eval backend"""
    # Execute evaluation
    # Return success, error, metrics
```

---

### 2.4 Per-Commit Testing

**File:** `sglang/python/sglang/test/accuracy_test_runner.py` (Lines 198-288)

```python
def run_accuracy_test(
    model: ModelLaunchSettings,
    params: AccuracyTestParams,
    base_url: Optional[str] = None,
) -> AccuracyTestResult:
    """Run accuracy test for a single model"""

    # Use few_shot_eval for gsm8k by default
    has_extended_params = any(
        getattr(params, field) is not None
        for field in ("thinking_mode", "temperature", "top_p", "repeat")
    )

    if params.dataset == "gsm8k" and not has_extended_params:
        success, error, metrics = _run_few_shot_eval(...)
    else:
        success, error, metrics = _run_simple_eval(...)

    # Validate against baseline
    score = (
        metrics.get("score")
        or metrics.get("mean_score")
        or metrics.get("accuracy", 0.0)
    )
    passed = score >= params.baseline_accuracy

    return AccuracyTestResult(
        model=model.model_path,
        dataset=params.dataset,
        passed=passed,
        score=score,
        baseline_accuracy=params.baseline_accuracy,
        error=error if not passed else None,
        latency=metrics.get("latency"),
    )
```

---

### 2.5 GitHub Integration

**File:** `sglang/python/sglang/test/accuracy_test_runner.py` (Lines 47-72)

```python
def write_accuracy_github_summary(
    test_name: str,
    dataset: str,
    results: List[AccuracyTestResult],
) -> None:
    """Write accuracy test results to GitHub step summary"""
    summary = f"#### {test_name} - Accuracy ({dataset})\n"
    summary += "| config | status | score | baseline | error |\n"
    summary += "| ------ | ------ | ----- | -------- | ----- |\n"

    for result in results:
        config_name = f"{result.model} ({result.variant})" if result.variant else result.model
        status_emoji = "✅" if result.passed else "❌"
        score_str = f"{result.score:.4f}" if result.score is not None else "N/A"
        baseline_str = f"{result.baseline_accuracy:.4f}"
        error_str = result.error if result.error else "-"

        summary += (
            f"| {config_name} | {status_emoji} | "
            f"{score_str} | {baseline_str} | {error_str} |\n"
        )

    write_github_step_summary(summary)
```

---

## 3. TensorRT-LLM Accuracy Testing

### 3.1 Hypothesis Testing Framework

#### Core Parameters

**File:** `tensorrt_llm/tests/integration/defs/accuracy/accuracy_core.py` (Lines 42-119)

```python
def compute_theta(
    num_samples: int,
    sigma: float,
    alpha: float = 0.05,
    beta: float = 0.2
):
    """Compute theta (minimum detectable effect) for hypothesis testing.

    Args:
        num_samples: Number of samples
        sigma: Standard deviation (default 50%)
        alpha: Type I error - False Positive (default 5%)
        beta: Type II error - False Negative (default 20%)

    Returns:
        Theta: Minimum detectable effect size
    """
    scale = (2 * sigma**2 / num_samples)**0.5

    # Single-tail testing
    z_alpha = scipy.stats.norm.ppf(alpha)
    z_beta = scipy.stats.norm.ppf(beta)
    theta = -(z_alpha + z_beta) * scale
    return theta

def compute_threshold(
    num_samples: int,
    ref_accuracy: float,
    sigma: float,
    alpha: float = 0.05,
    higher_is_better: bool = True
):
    """Compute acceptance threshold for hypothesis testing.

    Args:
        num_samples: Number of samples to evaluate
        ref_accuracy: Reference/baseline accuracy
        sigma: Standard deviation
        alpha: Type I error (false positive rate)
        higher_is_better: Whether higher accuracy is better

    Returns:
        Threshold: Minimum acceptable accuracy
    """
    scale = (2 * sigma**2 / num_samples)**0.5

    z_alpha = scipy.stats.norm.ppf(alpha)
    if higher_is_better:
        return ref_accuracy + z_alpha * scale
    else:
        return ref_accuracy - z_alpha * scale
```

---

#### HypothesisTestingParams

```python
@dataclass(slots=True)
class HypothesisTestingParams:
    ref_accuracy: float
    num_samples: int
    alpha: float = 0.05  # Type I error
    beta: float = 0.2    # Type II error
    sigma: float = 50.0  # Standard deviation
    higher_is_better: bool = True
    theta: float = field(init=False)
    threshold: float = field(init=False)

    def __post_init__(self) -> None:
        self.theta = compute_theta(self.num_samples, sigma=self.sigma, ...)
        self.threshold = compute_threshold(self.num_samples, self.ref_accuracy, ...)

    def report(self, accuracy: Optional[float] = None) -> str:
        """Generate hypothesis testing report"""
        return f"""
===========================================================
= ACCURACY HYPOTHESIS TESTING
===========================================================
Alpha (Type I: False Positive): {self.alpha:.3f}
Beta (Type II: False Negative): {self.beta:.3f}
Sigma (Standard deviation): {self.sigma:.3f}
#Samples: {self.num_samples}
Higher is better: {self.higher_is_better}
Theta (Minimum detectable effect): {self.theta:.3f}
Reference accuracy: {self.ref_accuracy:.3f}
Threshold: {self.threshold:.3f}
===========================================================
"""

    def assert_passing(self, accuracy: float) -> None:
        """Assert that measured accuracy passes hypothesis test"""
        err_msg = f"Expected accuracy >= {self.threshold:.3f}, got {accuracy:.3f}"
        if self.higher_is_better:
            assert accuracy >= self.threshold, err_msg
        else:
            assert accuracy <= self.threshold, err_msg
```

---

### 3.2 AccuracyTask Framework

**File:** `tensorrt_llm/tests/integration/defs/accuracy/accuracy_core.py` (Lines 121-247)

```python
class AccuracyTask:
    """Base class for accuracy evaluation tasks"""

    REFERENCE_DIR = f"{os.path.dirname(__file__)}/references"

    # Dataset specification
    DATASET = None
    DATASET_DIR = None
    HIGHER_IS_BETTER = True

    # Hypothesis testing parameters
    ALPHA = None
    BETA = None
    SIGMA = None
    NUM_SAMPLES = None

    # Input/output constraints
    MAX_INPUT_LEN = None
    MAX_OUTPUT_LEN = None
    MAX_BATCH_SIZE = None

    # Evaluator setup
    EVALUATOR_CLS = None
    EVALUATOR_KWARGS = None

    def __init__(self, model_name: str):
        # Load reference data from YAML
        with open(f"{self.REFERENCE_DIR}/{self.DATASET}.yaml") as f:
            self.reference: List[dict] = yaml.safe_load(f).get(model_name, [])

    def get_hypothesis_testing_params(self, **acc_specs) -> HypothesisTestingParams:
        """Get hypothesis testing parameters from reference data.

        Matches accuracy specifications:
        - dtype (str): Model data type (default 'auto')
        - quant_algo (str): Quantization algorithm (default None)
        - kv_cache_quant_algo (str): KV cache quantization (default None)
        - spec_dec_algo (str): Speculative decoding algorithm (default None)
        - extra_acc_spec (str): Extra specifications (default None)
        """
        for entry in self.reference:
            matched = True
            for key, value in acc_specs.items():
                default = 'auto' if key == 'dtype' else None
                if entry.get(key, default) != value:
                    matched = False
                    break
            if matched:
                break
        else:
            if os.getenv("TRTLLM_ACCURACY_NO_REFERENCE") == "1":
                entry = {"accuracy": 0}
            else:
                raise ValueError(f"Not registered specs: {acc_specs}.")

        return HypothesisTestingParams(
            ref_accuracy=entry.get("accuracy"),
            alpha=entry.get("alpha", self.ALPHA),
            beta=entry.get("beta", self.BETA),
            sigma=entry.get("sigma", self.SIGMA),
            num_samples=entry.get("num_samples", self.NUM_SAMPLES),
            higher_is_better=entry.get("higher_is_better", self.HIGHER_IS_BETTER)
        )

    def evaluate(self, llm, extra_acc_spec=None, ...):
        """Evaluate model accuracy against reference"""
        hypothesis_testing_params = self.get_hypothesis_testing_params(
            dtype=llm.args.dtype,
            quant_algo=llm.args.quant_config.quant_algo,
            kv_cache_quant_algo=llm.args.quant_config.kv_cache_quant_algo,
            spec_dec_algo=spec_dec_algo,
            extra_acc_spec=extra_acc_spec
        )

        evaluator = self.EVALUATOR_CLS(
            num_samples=hypothesis_testing_params.num_samples,
            **evaluator_kwargs
        )

        accuracy = evaluator.evaluate(llm, sampling_params, streaming, **evaluate_kwargs)

        # Assert accuracy meets threshold
        hypothesis_testing_params.assert_passing(accuracy)
```

---

### 3.3 Reference Data

**File:** `tensorrt_llm/tests/integration/defs/accuracy/references/gsm8k.yaml`

```yaml
meta-llama/Llama-3.1-8B-Instruct:
  - accuracy: 74.20                              # FP32 baseline
  - spec_dec_algo: NGram
    accuracy: 74.20                              # With speculative decoding
  - quant_algo: FP8
    accuracy: 74.30                              # FP8 quantization
  - quant_algo: FP8
    kv_cache_quant_algo: FP8
    accuracy: 72.85                              # FP8 weights + KV cache
  - quant_algo: FP8
    kv_cache_quant_algo: NVFP4
    accuracy: 69.75                              # Mixed precision
```

**Dataset Coverage:**
- `gsm8k.yaml` - Grade school math (up to 200 samples)
- `cnn_dailymail.yaml` - Text summarization
- `gpqa_diamond.yaml` - Graduate physics
- `humaneval.yaml` - Code generation

---

### 3.4 Default Parameters

**File:** `tensorrt_llm/tests/integration/defs/perf/disagg/reporting/accuracy_validator.py` (Lines 92-110)

```python
DATASET_DEFAULTS = {
    "aime25": {
        "alpha": 0.05,
        "beta": 0.2,
        "sigma": 50,
        "num_samples": 30,  # AIME 2025 full sample size
        "higher_is_better": True,
    },
    "gsm8k": {
        "alpha": 0.05,
        "beta": 0.2,
        "sigma": 50,
        "num_samples": 200,  # Standard GSM8K size
        "higher_is_better": True,
    },
    # ... more datasets
}
```

---

### 3.5 Test Implementation

**File:** `tensorrt_llm/tests/integration/defs/accuracy/test_llm_api.py` (Lines 29-120)

```python
class TestLlama3_1_8B(LlmapiAccuracyTestHarness):
    MODEL_NAME = "meta-llama/Llama-3.1-8B"
    MODEL_PATH = f"{llm_models_root()}/llama-3.1-model/Meta-Llama-3.1-8B"

    @skip_pre_ada
    @skip_post_blackwell
    def test_fp8_rowwise(self):
        """Test FP8 rowwise quantization"""
        quant_config = QuantConfig(QuantAlgo.FP8_PER_CHANNEL_PER_TOKEN)

        with LLM(self.MODEL_PATH, quant_config=quant_config) as llm:
            task = CnnDailymail(self.MODEL_NAME)
            task.evaluate(llm)  # Auto-validates against hypothesis test

            task = MMLU(self.MODEL_NAME)
            task.evaluate(llm)

class TestLlama3_1_8BInstruct(LlmapiAccuracyTestHarness):
    MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
    MODEL_PATH = f"{llm_models_root()}/llama-3.1-model/Llama-3.1-8B-Instruct"

    @pytest.mark.skip_less_device(4)
    def test_tp2cp2(self):
        """Test with tensor parallelism + context parallelism"""
        with LLM(
            self.MODEL_PATH,
            tensor_parallel_size=2,
            context_parallel_size=2
        ) as llm:
            task = GSM8K(self.MODEL_NAME)
            task.evaluate(llm)

    @pytest.mark.parametrize("backend", ["xgrammar"])
    def test_guided_decoding(self, backend: str):
        """Test guided decoding (JSON mode)"""
        llm = LLM(self.MODEL_PATH, guided_decoding_backend=backend)
        with llm:
            task = JsonModeEval(self.MODEL_NAME)
            task.evaluate(llm)
```

---

## 4. Comparative Analysis

### 4.1 Testing Methodologies

| Feature | vLLM | SGLang | TensorRT-LLM |
|---------|------|--------|--------------|
| **Primary Method** | Exact token matching | Accuracy threshold | Hypothesis testing |
| **Tolerance Model** | Fixed tolerance (±3%) | Simple pass/fail | Statistical bounds |
| **Test Coverage** | Greedy + logprobs | Few-shot + simple eval | Multiple datasets |
| **Determinism Testing** | Batch invariance | Implicit | Reference-based |
| **Precision Testing** | Explicit (FP32/16/8) | Implicit | Configuration-based |
| **Statistical Rigor** | Low | Low | **High** |

---

### 4.2 Accuracy Guarantees

#### vLLM
- **Exact Match**: Greedy decoding produces identical tokens to HuggingFace
- **Logprob Tolerance**: Top-K logprob matching (allows token divergence)
- **Embedding Similarity**: Cosine similarity ≥ 0.999
- **Benchmark Tolerance**: ±3-5% for accuracy benchmarks (RTOL)
- **Numerical Precision**: 1-2% tolerance for fusion operators (ATOL)

#### SGLang
- **Baseline-Relative**: Model score ≥ baseline_accuracy
- **Few-Shot GSM8K**: Exact answer matching
- **Latency Tracking**: Optional latency measurement
- **Extended Eval**: Support for reasoning tasks (thinking modes)

#### TensorRT-LLM
- **Statistical Bounds**: Two-tailed hypothesis test
- **Power Analysis**: β = 0.2 (80% power), α = 0.05 (5% significance)
- **Minimum Effect Size**: Detectable deviation = theta (±2-3% for 50 samples)
- **Configuration-Specific**: Separate accuracy for each quant/parallelism combination

---

### 4.3 Flaky Test Prevention

#### vLLM Approaches
1. **Seeding**: Fixed seed=0 for VllmRunner
2. **Batch Invariance**: Confirms determinism across batch compositions
3. **Environmental Controls**: ROCm determinism flags
4. **Skip Markers**: Platform-specific skips (`@skip_pre_ada`)
5. **Timeout Management**: `@pytest.mark.timeout(1000)`

#### SGLang Approaches
1. **Baseline Definition**: Explicit baseline before running
2. **GitHub Integration**: Results published for visibility
3. **Server Lifecycle**: Fresh server per test
4. **Latency Bounds**: Optional accept_length_thres

#### TensorRT-LLM Approaches
1. **Reference YAML**: Fixed, versioned accuracy targets
2. **Hypothesis Testing**: Statistically robust bounds
3. **Integration Test Mode**: `INTEGRATION_TEST=1` to skip checks
4. **Per-Configuration**: Different accuracy for each setup
5. **No-Reference Mode**: `TRTLLM_ACCURACY_NO_REFERENCE=1`

---

## 5. Best Practices

### 5.1 Tolerance Setting

#### vLLM - Relative Tolerance
```python
RTOL = 0.03  # ±3%
assert (measured - RTOL < expected < measured + RTOL)
```

#### vLLM - Absolute Tolerance
```python
torch.testing.assert_close(a, b, atol=1e-2, rtol=1e-2)
```

#### SGLang - Absolute Baseline
```python
baseline_accuracy = 0.82
assert score >= baseline_accuracy
```

#### TensorRT-LLM - Statistical Threshold
```python
threshold = compute_threshold(num_samples=200, ref_accuracy=0.74, sigma=50)
assert measured >= threshold
```

---

### 5.2 Precision Support

| Framework | FP32 | FP16 | BF16 | FP8 | FP4 | Mixed |
|-----------|------|------|------|-----|-----|-------|
| **vLLM** | ✓ | ✓ | ✓ | ✓ | ✗ | ✓ |
| **SGLang** | ✓ | ✓ | ✓ | ✓ | ✗ | Implicit |
| **TensorRT-LLM** | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

---

### 5.3 Determinism Levels

| Framework | Level | Method |
|-----------|-------|--------|
| **vLLM** | Batch invariance | Needle-in-batch testing |
| **SGLang** | Implicit | Fixed seed via few-shot |
| **TensorRT-LLM** | Configuration-based | Reference accuracy matching |

---

### 5.4 Summary Table: Key Parameters

| Parameter | vLLM | SGLang | TensorRT-LLM |
|-----------|------|--------|--------------|
| **Default RTOL** | 0.03-0.05 | Model-dependent | N/A |
| **Default ATOL** | 1e-2 to 2e-3 | N/A | N/A |
| **Determinism Seed** | 0 | Server-managed | N/A |
| **Test Dataset** | Custom prompts | GSM8K/MMLU/others | GSM8K/CNN/GPQA/HumanEval |
| **Batch Size** | 1-128 | 1-128 | Variable |
| **Max Model Length** | 1024-8192 | Configurable | Configurable |
| **FP8 Support** | ✓ | ✓ | ✓ |
| **Hypothesis Testing** | ✗ | ✗ | ✓ |
| **Reference YAML** | ✗ | ✗ | ✓ |
| **GitHub Integration** | ✗ | ✓ | ✗ |

---

## Conclusion

### Strengths by Framework

**vLLM:**
- ✅ Comprehensive exact-match validation
- ✅ Batch invariance testing ensures robustness
- ✅ Multiple precision support with specific tolerances
- ✅ Flexible logprob-based validation

**SGLang:**
- ✅ Per-commit accuracy suites
- ✅ GitHub integration for visibility
- ✅ Support for extended evaluation modes (thinking)
- ✅ Latency tracking

**TensorRT-LLM:**
- ✅ Most statistically rigorous (hypothesis testing)
- ✅ Configuration-specific baselines via YAML
- ✅ Comprehensive dataset coverage
- ✅ Quantization-aware accuracy expectations

### Recommended Practices

**Use vLLM's approach for:**
- Token-level correctness in greedy sampling
- Batch invariance validation
- Cross-framework comparison (HF baseline)

**Use SGLang's approach for:**
- Per-commit continuous evaluation
- Reasoning task validation
- Latency-aware testing

**Use TensorRT-LLM's approach for:**
- Production accuracy requirements
- Configuration-specific validation
- Statistical confidence in thresholds

---

**Document Version:** 1.0
**Last Updated:** 2026-03-29
**Research Scope:** vLLM (latest), SGLang (latest), TensorRT-LLM (latest)
