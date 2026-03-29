# Tool Calling / Function Calling Implementation Comparison

Comprehensive comparison of tool calling (also known as function calling) implementations across vLLM, SGLang, and TensorRT-LLM.

**Last Updated:** 2026-03-28

---

## Table of Contents

1. [Overview & Architecture Comparison](#overview--architecture-comparison)
2. [Command-Line Options](#command-line-options)
3. [Tool Parser Implementations](#tool-parser-implementations)
4. [Structured Output Integration](#structured-output-integration)
5. [Streaming Support](#streaming-support)
6. [OpenAI API Compatibility](#openai-api-compatibility)
7. [Usage Examples](#usage-examples)
8. [Feature Comparison Matrix](#feature-comparison-matrix)
9. [Code Sources & Implementation Details](#code-sources--implementation-details)

---

## Overview & Architecture Comparison

### vLLM

**Architecture:** Abstract ToolParser framework with 37 model-specific parsers

**Key Components:**
- **Base Framework:** `/vllm/tool_parsers/abstract_tool_parser.py` (lines 34-120)
- **Parser Registry:** `/vllm/tool_parsers/__init__.py` (lines 24-157) - Lazy loading system
- **Structured Outputs:** `/vllm/config/structured_outputs.py` (lines 18-74) - Multi-backend support
- **Manager:** `/vllm/parser/parser_manager.py` (lines 190-220) - Unified parser selection

**Design Philosophy:**
- Extensible plugin system with lazy and eager loading
- Separation of tool parsing and structured output backends
- Integration with xgrammar, outlines, lm-format-enforcer, guidance
- Unified parser handling both reasoning and tool calling

**Parsers:** 37 tool parsers covering models from mistral, llama, qwen, deepseek, glm, granite, phi, internlm, jamba, and more.

---

### SGLang

**Architecture:** Format detector pattern with FunctionCallParser orchestrator

**Key Components:**
- **FunctionCallParser:** `/sglang/python/sglang/srt/function_call/function_call_parser.py` (lines 39-215)
- **Base Detector:** `/sglang/python/sglang/srt/function_call/base_format_detector.py` (lines 26-347)
- **Grammar Manager:** `/sglang/python/sglang/srt/constrained/grammar_manager.py` (lines 24-196)
- **MCP Support:** `/sglang/python/sglang/srt/entrypoints/openai/tool_server.py` (lines 1-176)

**Design Philosophy:**
- Format detection pattern with pluggable detectors
- Constrained generation via grammar_manager with XGrammar backend
- MCP (Model Context Protocol) tool server integration
- Environment-based strict validation levels

**Detectors:** 24 format detectors supporting hermes, mistral, pythonic, qwen25, deepseek v3/v3.1/v3.2, glm, kimi_k2, llama32, and more.

---

### TensorRT-LLM

**Architecture:** BaseToolParser framework with model-specific implementations

**Key Components:**
- **Base Framework:** `/TensorRT-LLM/tensorrt_llm/serve/tool_parser/base_tool_parser.py` (lines 1-325)
- **Parser Factory:** `/TensorRT-LLM/tensorrt_llm/serve/tool_parser/tool_parser_factory.py`
- **Guided Decoding:** `/TensorRT-LLM/tensorrt_llm/serve/openai_protocol.py` (lines 193-235)
- **Postprocessing:** `/TensorRT-LLM/tensorrt_llm/serve/postprocess_handlers.py` (lines 137-159)

**Design Philosophy:**
- Minimal set of highly-optimized parsers
- Structural tag support via XGrammar/LLGuidance
- Tight integration with TensorRT optimization pipeline
- Streamlined factory pattern

**Parsers:** 6 parsers (qwen3, qwen3_coder, kimi_k2, deepseek_v3, deepseek_v31, deepseek_v32).

---

## Command-Line Options

### vLLM

**File:** `/vllm/entrypoints/openai/cli_args.py` (lines 111-133)

```bash
# Enable automatic tool choice for supported models
--enable-auto-tool-choice

# Select tool call parser (required with --enable-auto-tool-choice)
--tool-call-parser {mistral|llama3_json|openai|hermes|pythonic|deepseekv3|...}

# Exclude tool definitions when tool_choice='none'
--exclude-tools-when-tool-choice-none

# Load custom tool parser plugin
--tool-parser-plugin <path_to_plugin>

# Enable demo tools or specify tool server URLs
--tool-server {demo|host:port[,host:port,...]}

# Structured output backend selection
--structured-outputs-backend {auto|xgrammar|guidance|outlines|lm-format-enforcer}

# Structured output options
--disable-any-whitespace
--disable-additional-properties
--reasoning-parser <parser_name>
--reasoning-parser-plugin <path>
--enable-in-reasoning
```

**Example Server Startup:**
```bash
vllm serve mistralai/Mistral-7B-Instruct-v0.3 \
    --chat-template examples/tool_chat_template_mistral.jinja \
    --enable-auto-tool-choice \
    --tool-call-parser mistral \
    --tool-server demo
```

**Available Parser Names:**
deepseekv3, deepseekv31, deepseekv32, ernie45, functiongemma, gigachat3, glm4_moe, glm47_moe, granite, granite4, granite_20b_fc, hermes, hunyuan_a13b, internlm2, jamba, kimi_k2, llama3_json, llama4_json, llama4_pythonic, longcat, minimax, minimax_m2, mistral, olmo3, openai, phi4mini_json, pythonic, qwen3_coder, qwen3_xml, seed_oss, step3, step3p5, xlam

---

### SGLang

**File:** `/sglang/python/sglang/srt/server_args.py` (lines 3678-3691)

```bash
# Specify parser for tool-call interactions
--tool-call-parser {deepseekv3|deepseekv31|deepseekv32|glm|glm45|glm47|gpt-oss|kimi_k2|lfm2|llama3|mimo|mistral|pythonic|qwen|qwen25|qwen3_coder|step3|step3p5|minimax-m2|trinity|interns1|hermes|gigachat3}

# MCP tool server integration (demo or comma-separated URLs)
--tool-server {demo|http://host1:port1,http://host2:port2,...}
```

**Environment Variables:**
```bash
# Forward unknown tool calls in error handling
SGLANG_FORWARD_UNKNOWN_TOOLS=false

# Tool strictness level: 0=OFF, 1=FUNCTION, 2=PARAMETER
SGLANG_TOOL_STRICT_LEVEL=0

# Grammar processing poll interval (seconds)
SGLANG_GRAMMAR_POLL_INTERVAL=0.005

# Max grammar polling iterations
SGLANG_GRAMMAR_MAX_POLL_ITERATIONS=10000
```

**Example Server Startup:**
```bash
python -m sglang.launch_server \
    --model-path mistralai/Mistral-7B-Instruct-v0.3 \
    --tool-call-parser mistral \
    --tool-server demo
```

**Available Parser Names:**
deepseekv3, deepseekv31, deepseekv32, glm, glm45, glm47, gpt-oss, kimi_k2, lfm2, llama3, mimo, mistral, pythonic, qwen, qwen25, qwen3_coder, step3, step3p5, minimax-m2, trinity, interns1, hermes, gigachat3

---

### TensorRT-LLM

**File:** `/TensorRT-LLM/tensorrt_llm/commands/serve.py` (lines 421-427)

```bash
# Specify parser for tool models
--tool_parser {qwen3|qwen3_coder|kimi_k2|deepseek_v3|deepseek_v31|deepseek_v32}
```

**Example Server Startup:**
```bash
trtllm-serve \
    Qwen/Qwen2.5-7B-Instruct \
    --tool_parser qwen3
```

**Available Parser Names:**
qwen3, qwen3_coder, kimi_k2, deepseek_v3, deepseek_v31, deepseek_v32

**Note:** TensorRT-LLM has a smaller set of parsers, focusing on high-performance inference for specific model families.

---

## Tool Parser Implementations

### vLLM - 37 Parsers

**Location:** `/vllm/tool_parsers/`

| Parser | File | Format Type | Model Support |
|--------|------|-------------|---------------|
| MistralToolParser | `mistral_tool_parser.py` | JSON array: `[TOOL_CALLS] [{...}, ...]` | Mistral 7B Instruct v0.3+ |
| Llama3JsonToolParser | `llama_tool_parser.py` | JSON with specific structure | Llama 3.x, 4.x |
| Llama4PythonicToolParser | `llama4_pythonic_tool_parser.py` | Python syntax | Llama 4.x |
| Hermes2ProToolParser | `hermes_tool_parser.py` | XML: `<tool_call>{...}</tool_call>` | Hermes 2 Pro |
| OpenAIToolParser | `openai_tool_parser.py` | OpenAI format | GPT-compatible models |
| DeepSeekV3ToolParser | `deepseekv3_tool_parser.py` | Unicode delimiters + JSON | DeepSeek V3 |
| DeepSeekV31ToolParser | `deepseekv31_tool_parser.py` | Compact unicode format | DeepSeek V3.1 |
| DeepSeekV32ToolParser | `deepseekv32_tool_parser.py` | DSML XML or JSON | DeepSeek V3.2 |
| Qwen3XMLToolParser | `qwen3xml_tool_parser.py` | XML-based | Qwen 3 |
| Qwen3CoderToolParser | `qwen3coder_tool_parser.py` | Custom format | Qwen 3 Coder |
| KimiK2ToolParser | `kimi_k2_tool_parser.py` | Custom delimiters | Kimi K2 |
| Glm4MoeModelToolParser | `glm4_moe_tool_parser.py` | GLM format | GLM-4 MoE |
| Glm47MoeModelToolParser | `glm47_moe_tool_parser.py` | GLM format | GLM-4.7 MoE |
| PythonicToolParser | `pythonic_tool_parser.py` | Python syntax: `[func(arg=val), ...]` | Various models |
| GraniteToolParser | `granite_tool_parser.py` | Granite format | IBM Granite |
| Granite4ToolParser | `granite4_tool_parser.py` | Granite 4 format | IBM Granite 4 |
| Granite20bFCToolParser | `granite_20b_fc_tool_parser.py` | Granite 20B FC format | IBM Granite 20B FC |
| InternLM2ToolParser | `internlm2_tool_parser.py` | InternLM format | InternLM 2 |
| JambaToolParser | `jamba_tool_parser.py` | Jamba format | AI21 Jamba |
| FunctionGemmaToolParser | `functiongemma_tool_parser.py` | Gemma function format | Google Gemma Function |
| Phi4MiniJsonToolParser | `phi4mini_tool_parser.py` | JSON format | Microsoft Phi-4 Mini |
| MinimaxToolParser | `minimax_tool_parser.py` | Minimax format | Minimax models |
| MinimaxM2ToolParser | `minimax_m2_tool_parser.py` | Minimax M2 format | Minimax M2 |
| xLAMToolParser | `xlam_tool_parser.py` | xLAM format | xLAM models |
| Step3ToolParser | `step3_tool_parser.py` | Step format | Step 3 |
| Step3p5ToolParser | `step3p5_tool_parser.py` | Step format | Step 3.5 |
| Olmo3PythonicToolParser | `olmo3_tool_parser.py` | Pythonic | OLMo 3 |
| SeedOssToolParser | `seed_oss_tool_parser.py` | Custom format | Seed OSS |
| HunyuanA13BToolParser | `hunyuan_a13b_tool_parser.py` | Hunyuan format | Hunyuan A13B |
| LongcatFlashToolParser | `longcat_tool_parser.py` | LongCat format | LongCat Flash |
| GigaChat3ToolParser | `gigachat3_tool_parser.py` | GigaChat format | GigaChat 3 |
| Ernie45ToolParser | `ernie45_tool_parser.py` | Ernie format | Baidu Ernie 4.5 |

**Registration System:**
- Lazy loading via `_TOOL_PARSERS_TO_REGISTER` dict (`__init__.py:24-157`)
- Plugin support via `ToolParserManager.import_tool_parser(plugin_path)`
- Automatic registration at import time (lines 160-166)

---

### SGLang - 24 Format Detectors

**Location:** `/sglang/python/sglang/srt/function_call/`

| Detector | File | Format Type | Model Support |
|----------|------|-------------|---------------|
| MistralDetector | `mistral_detector.py` | JSON array: `[TOOL_CALLS] [{...}, ...]` | Mistral models |
| HermesDetector | `hermes_detector.py` | XML: `<tool_call>{...}</tool_call>` | Hermes models |
| PythonicDetector | `pythonic_detector.py` | Python: `[func(arg1=val1), ...]` | Various models |
| Qwen25Detector | `qwen25_detector.py` | Wrapped JSON: `<tool_call>\n{...}\n</tool_call>` | Qwen 2.5 |
| DeepSeekV3Detector | `deepseekv3_detector.py` | Unicode + JSON | DeepSeek V3 |
| DeepSeekV31Detector | `deepseekv31_detector.py` | Compact unicode | DeepSeek V3.1 |
| DeepSeekV32Detector | `deepseekv32_detector.py` | DSML format | DeepSeek V3.2 |
| Glm4MoeDetector | `glm4_moe_detector.py` | GLM format | GLM-4 MoE |
| Glm47MoeDetector | `glm47_moe_detector.py` | GLM format | GLM-4.7 MoE |
| KimiK2Detector | `kimik2_detector.py` | Kimi format | Kimi K2 |
| Llama32Detector | `llama32_detector.py` | Llama format | Llama 3.2 |
| Qwen3CoderDetector | `qwen3_coder_detector.py` | Qwen coder format | Qwen 3 Coder |
| GptOssDetector | `gpt_oss_detector.py` | GPT OSS format | GPT OSS |
| Lfm2Detector | `lfm2_detector.py` | LFM format | LFM-2 |
| MiMoDetector | `mimo_detector.py` | MiMo format | MiMo |
| MinimaxM2Detector | `minimax_m2.py` | Minimax M2 | Minimax M2 |
| InternlmDetector | `internlm_detector.py` | InternLM format | InternLM |
| Step3Detector | `step3_detector.py` | Step format | Step 3 |
| GigaChat3Detector | `gigachat3_detector.py` | GigaChat format | GigaChat 3 |
| TrinityDetector | `trinity_detector.py` | Trinity format | Trinity |

**Detector Registry:**
- Central registry in `FunctionCallParser` (`function_call_parser.py:48-72`)
- Maps format string to detector class
- Example: `"mistral": MistralDetector`

---

### TensorRT-LLM - 6 Parsers

**Location:** `/TensorRT-LLM/tensorrt_llm/serve/tool_parser/`

| Parser | File | Format Type | Model Support |
|--------|------|-------------|---------------|
| Qwen3ToolParser | `qwen3_tool_parser.py` | XML: `<tool_call>\n{...}\n</tool_call>` | Qwen 2.5, Qwen 3 |
| Qwen3CoderToolParser | `qwen3_coder_parser.py` | XML nested: `<function=...><parameter=...>` | Qwen 3 Coder |
| DeepSeekV3Parser | `deepseekv3_parser.py` | Unicode delimiters + JSON | DeepSeek V3 |
| DeepSeekV31Parser | `deepseekv31_parser.py` | Compact unicode | DeepSeek V3.1 |
| DeepSeekV32Parser | `deepseekv32_parser.py` | DSML XML or JSON | DeepSeek V3.2 |
| KimiK2ToolParser | `kimi_k2_tool_parser.py` | Custom: `<\|tool_call_begin\|>` | Kimi K2 |

**Factory Pattern:**
- `ToolParserFactory.parsers` dict maps names to classes
- `ToolParserFactory.create_tool_parser(name)` creates instances
- Streamlined for production deployment

---

## Structured Output Integration

### vLLM - Multi-Backend Support

**Configuration:** `/vllm/config/structured_outputs.py` (lines 18-74)

```python
@config
class StructuredOutputsConfig:
    backend: StructuredOutputsBackend = "auto"
    # Options: "auto", "xgrammar", "guidance", "outlines", "lm-format-enforcer"

    disable_any_whitespace: bool = False
    disable_additional_properties: bool = False
    reasoning_parser: str = ""
    reasoning_parser_plugin: str = ""
    enable_in_reasoning: bool = False
```

**Backend Integration:** `/vllm/v1/structured_output/__init__.py` (lines 115-142)

**Available Backends:**
1. **XGrammar** - `/vllm/v1/structured_output/backend_xgrammar.py`
   - Fastest, recommended for production
   - Full JSON schema support
   - Structural tag support

2. **Guidance** - `/vllm/v1/structured_output/backend_guidance.py`
   - Microsoft Guidance library
   - Rich grammar capabilities

3. **Outlines** - `/vllm/v1/structured_output/backend_outlines.py`
   - Regex and JSON schema constraints
   - Flexible generation control

4. **LM Format Enforcer** - `/vllm/v1/structured_output/backend_lm_format_enforcer.py`
   - Alternative constraint system
   - Character-level format enforcement

**Tool Integration:**
- Tools are converted to JSON schemas via `adjust_request()` (`abstract_tool_parser.py:56-84`)
- `StructuredOutputsParams` set on `SamplingParams` (line 263)
- Backend automatically selected or explicitly configured

---

### SGLang - Grammar Manager with XGrammar

**Grammar Manager:** `/sglang/python/sglang/srt/constrained/grammar_manager.py` (lines 24-196)

**Key Features:**
- **Single Backend:** XGrammar (highly optimized)
- **Constraint Types:** json_schema, regex, ebnf, structural_tag
- **Caching System:** Cache keys like `("json", schema)`, `("regex", pattern)`
- **Distributed Sync:** Synchronizes ready/failed requests across all ranks

**Processing Flow:**
1. `process_req_with_grammar()` (lines 67-105):
   - Checks sampling params for grammar constraints
   - Creates cache keys
   - Returns grammar object or future

2. `get_ready_grammar_requests()` (lines 107-196):
   - Polls grammar queue with configurable interval
   - Default poll interval: 0.005s (5ms)
   - Max iterations: 10,000

**Tool Integration:**
- `FunctionCallParser.get_structure_constraint()` generates structural tags
- Grammar manager applies XGrammar to constrain output
- Guarantees format compliance

**Environment Controls:**
```bash
SGLANG_GRAMMAR_POLL_INTERVAL=0.005  # Polling interval (seconds)
SGLANG_GRAMMAR_MAX_POLL_ITERATIONS=10000  # Max poll iterations
```

---

### TensorRT-LLM - XGrammar/LLGuidance Integration

**Guided Decoding:** `/TensorRT-LLM/tensorrt_llm/serve/openai_protocol.py` (lines 193-235)

**Conversion Function:**
```python
def _response_format_to_guided_decoding_params(
    response_format: Optional[ResponseFormat]
) -> Optional[GuidedDecodingParams]:

    if response_format.type == "json":
        return GuidedDecodingParams(json=response_format.schema)
    elif response_format.type == "json_schema":
        return GuidedDecodingParams(json=response_format.json_schema)
    elif response_format.type == "regex":
        return GuidedDecodingParams(regex=response_format.regex)
    elif response_format.type == "ebnf":
        return GuidedDecodingParams(grammar=response_format.ebnf)
    elif response_format.type == "structural_tag":
        return GuidedDecodingParams(
            structural_tag=response_format.model_dump_json(
                by_alias=True, exclude_none=True))
```

**Structural Tag Support:**
- Parsers provide `structure_info()` returning a function
- Function takes tool name → returns `StructureInfo(begin, end, trigger)`
- Used for XGrammar structural tag-based generation

**Example - Qwen3:**
```python
def structure_info(self) -> _GetInfoFunc:
    return lambda name: StructureInfo(
        begin='<tool_call>\n{"name":"' + name + '", "arguments":',
        end="}\n</tool_call>",
        trigger="<tool_call>",
    )
```

**Integration Points:**
- `CompletionRequest` (line 333): Converts response_format
- `ChatCompletionRequest` (line 681): Applies to chat completions
- `ResponsesRequest` (lines 827-838): Responses API support

---

## Streaming Support

### vLLM - State Machine with Delta Updates

**Implementation:** `/vllm/tool_parsers/abstract_tool_parser.py` (lines 100-119)

**Streaming Method:**
```python
def extract_tool_calls_streaming(
    self,
    previous_text: str,
    current_text: str,
    delta_text: str,
    previous_token_ids: list[int],
    current_token_ids: list[int],
    delta_token_ids: list[int],
    request: ChatCompletionRequest,
) -> DeltaMessage | None:
    """Instance method for streaming tool parsing with state across tokens"""
```

**State Management:**
- `self.prev_tool_call_arr` - Previous tool call states
- `self.current_tool_id` - Currently streaming tool
- `self.current_tool_name_sent` - Tool name emission flag
- `self.streamed_args_for_tool` - Streamed JSON per tool

**Delta Calculation:**
- Utilities in `/vllm/tool_parsers/utils.py`:
  - `compute_tool_delta()` - Calculates argument differences
  - `find_common_prefix/suffix()` - Manages partial JSON
  - `partial_json_loads()` - Parses incomplete JSON

**Streaming Flow:**
1. Parse new token delta
2. Check for new tool call start
3. Emit tool name with empty parameters (first time)
4. Stream JSON arguments incrementally
5. Calculate diff from previous state
6. Return `DeltaMessage` with updates

---

### SGLang - Incremental Parsing with Buffer

**Implementation:** `/sglang/python/sglang/srt/function_call/base_format_detector.py`

**Streaming Method:**
```python
def parse_streaming_increment(
    self,
    new_text: str,
    tools: List[Tool]
) -> StreamingParseResult:
    """Parse streaming increment"""
```

**State Management (lines 29-46):**
```python
self._buffer = ""  # Accumulates incomplete patterns
self.prev_tool_call_arr = []  # Stores complete tool call info
self.current_tool_id = -1  # Currently streaming tool (-1 = none)
self.current_tool_name_sent = False  # Tool name sent flag
self.streamed_args_for_tool = []  # Raw JSON arguments per tool
```

**Streaming Architecture:**
1. Accumulate chunks in `_buffer`
2. Parse complete patterns when boundaries detected
3. Extract tool names first
4. Stream JSON arguments as fragments become available
5. Return `StreamingParseResult(normal_text, calls)`

**Format-Specific Handling:**
- Each detector implements custom streaming logic
- Handles partial tokens via `_ends_with_partial_token()`
- Supports multiple tool calls in single response

---

### TensorRT-LLM - Sophisticated State Tracking

**Implementation:** `/TensorRT-LLM/tensorrt_llm/serve/tool_parser/base_tool_parser.py` (lines 111-299)

**Core Method:**
```python
def parse_streaming_increment(
    self,
    text: str,
    tools: List[Tool]
) -> StreamingParseResult:
    """Handles streaming incremental parsing - most complex method"""
```

**State Management (lines 20-36):**
```python
self._buffer = ""  # Incomplete pattern buffer
self.prev_tool_call_arr = []  # Previous tool call states
self.current_tool_id = -1  # Currently streaming tool index
self.current_tool_name_sent = False  # Name emission flag
self.streamed_args_for_tool = []  # Streamed JSON per tool
```

**Streaming Process:**

**Phase 1 - Tool Name Streaming (lines 199-226):**
```python
if current_tool_id == -1:
    # Parse tool name
    # Validate against available tools
    # Emit tool with empty parameters
    # Set current_tool_name_sent = True
```

**Phase 2 - Argument Streaming (lines 228-285):**
```python
else:
    # Accumulate JSON fragments
    # Parse partial JSON
    # Calculate diff using find_common_prefix()
    # Emit argument updates
    # Detect tool completion
    # Move to next tool if separator found
```

**Key Features:**
- Validates tool names during streaming
- Separates name emission from argument streaming
- Tracks argument diffs for incremental updates
- Handles multiple sequential tool calls
- Buffers incomplete patterns across chunks

---

## OpenAI API Compatibility

All three systems provide OpenAI-compatible APIs for tool calling.

### Data Models (Common Across All Three)

**ToolCall:**
```python
class ToolCall:
    id: str  # Unique tool call ID
    type: str = "function"
    function: FunctionCall
```

**FunctionCall:**
```python
class FunctionCall:
    name: str  # Function name
    arguments: str  # JSON string
```

**DeltaToolCall (Streaming):**
```python
class DeltaToolCall:
    index: int  # Tool call index
    id: Optional[str] = None
    type: Optional[str] = None
    function: Optional[DeltaFunctionCall] = None
```

**ChatCompletionRequest:**
```python
class ChatCompletionRequest:
    model: str
    messages: List[ChatCompletionMessageParam]
    tools: Optional[List[ChatCompletionToolsParam]] = None
    tool_choice: Optional[Union[
        Literal["none"],
        Literal["auto"],
        Literal["required"],
        ChatCompletionNamedToolChoiceParam
    ]] = "none"
    parallel_tool_calls: Optional[bool] = True
```

---

### vLLM - OpenAI Serving

**Protocol:** `/vllm/entrypoints/openai/engine/protocol.py` (lines 218-256)

**Request Protocol:** `/vllm/entrypoints/openai/chat_completion/protocol.py` (lines 174-184)

**Serving Integration:** `/vllm/entrypoints/openai/chat_completion/serving.py`

**Tool Parser Initialization (lines 130-136):**
```python
self.enable_auto_tools: bool = enable_auto_tools
self.tool_parser = ParserManager.get_tool_parser(
    tool_parser_name=tool_parser,
    enable_auto_tools=enable_auto_tools,
    model_name=self.model_config.model,
)
self.exclude_tools_when_tool_choice_none = exclude_tools_when_tool_choice_none
```

**Tool Call Extraction (lines 1332-1375):**
```python
if self.tool_parser is not None:
    tool_parser = self.tool_parser(tokenizer)
    tool_call_info = tool_parser.extract_tool_calls(
        "",
        request=request,
        token_ids=token_ids,
    )
    content = tool_call_info.content
    message = ChatMessage(
        role=role,
        reasoning=reasoning,
        content=content,
        tool_calls=tool_call_info.tool_calls,
    )
```

**Auto Tool Choice:**
- Enabled via `--enable-auto-tool-choice`
- When enabled, `tool_choice` defaults to "auto" if tools provided
- Parser validates and extracts tool calls from model output

---

### SGLang - OpenAI Chat Serving

**Serving:** `/sglang/python/sglang/srt/entrypoints/openai/serving_chat.py`

**Request Validation (lines 192-238):**
```python
def _validate_request(self, request: ChatCompletionRequest) -> Optional[str]:
    # Lines 197-210: Tool validation
    if request.tool_choice == "required" and not request.tools:
        return "Tools cannot be empty if tool choice is set to required."

    # Lines 212-219: JSON schema validation for tool parameters
    for i, tool in enumerate(request.tools or []):
        try:
            Draft202012Validator.check_schema(tool.function.parameters)
```

**Tool Constraint Handling (lines 260-264):**
```python
sampling_params = request.to_sampling_params(
    stop=processed_messages.stop,
    model_generation_config=self.default_sampling_params,
    tool_call_constraint=processed_messages.tool_call_constraint,
)
```

**Tool Call Parser Initialization (lines 99-100):**
```python
self.tool_call_parser = self.tokenizer_manager.server_args.tool_call_parser
self.reasoning_parser = self.tokenizer_manager.server_args.reasoning_parser
```

**Integration Flow:**
```
Request → Validation → Constraint Generation → Grammar Manager →
Model Generation → Parser Extraction → Response
```

---

### TensorRT-LLM - Dual API Support

**Protocol:** `/TensorRT-LLM/tensorrt_llm/serve/openai_protocol.py`

**Chat Completions Support:**

**Request Model (lines 534+):**
```python
class ChatCompletionRequest(OpenAIBaseModel):
    model: str
    messages: List[ChatCompletionMessageParam]
    tools: Optional[List[ChatCompletionToolsParam]] = None
    tool_choice: Optional[Union[
        Literal["none", "auto"],
        ChatCompletionNamedToolChoiceParam
    ]] = "none"
```

**Tool Choice Validation (lines 716-722):**
```python
@classmethod
def check_tool_choice(cls, data):
    if "tool_choice" not in data and data.get("tools"):
        data["tool_choice"] = "auto"  # Auto-enable if tools provided
    if "tool_choice" in data and data["tool_choice"] != "none":
        if not data.get("tools"):
            raise ValueError(
                "When using `tool_choice`, `tools` must be set.")
```

**Responses API Support:**

TensorRT-LLM also supports Responses API (Gemini-style) for tool calling:

```python
class ResponsesRequest(OpenAIBaseModel):
    model: str
    input: Union[str, List[ResponsesInputItem]]
    tools: Optional[List[Tool]] = None
    tool_choice: Optional[Union[Literal["none", "auto"], ToolChoice]] = "auto"
```

**Post-processing:** `/TensorRT-LLM/tensorrt_llm/serve/postprocess_handlers.py`

**Tool Parser Application (lines 137-159):**
```python
def apply_tool_parser(
    args: ChatPostprocArgs,
    output_index: int,
    text: str,
    streaming: bool
) -> Tuple[str, List[ToolCallItem]]:
    """Parse tool calls from generated text"""
    if args.tool_parser and args.tools:
        if output_index not in args.tool_parser_dict:
            args.tool_parser_dict[output_index] = (
                ToolParserFactory.create_tool_parser(args.tool_parser))
        tool_parser = args.tool_parser_dict[output_index]

    if tool_parser and args.tools:
        if not streaming:
            result = tool_parser.detect_and_parse(text, args.tools)
        else:
            result = tool_parser.parse_streaming_increment(text, args.tools)
        normal_text, calls = result.normal_text, result.calls
    else:
        normal_text, calls = text, []

    return normal_text, calls
```

---

## Usage Examples

### vLLM

#### Offline Inference

**File:** `/vllm/examples/offline_inference/chat_with_tools.py` (lines 48-148)

```python
from vllm import LLM, SamplingParams
import json

model_name = "mistralai/Mistral-7B-Instruct-v0.3"
llm = LLM(
    model=model_name,
    tokenizer_mode="mistral",
    config_format="mistral",
    load_format="mistral",
)

# Define tools
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Get the current weather in a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "The city name"},
                    "state": {"type": "string", "description": "State abbreviation"},
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                },
                "required": ["city", "state", "unit"],
            },
        },
    }
]

messages = [
    {
        "role": "user",
        "content": "Can you tell me what the temperature will be in Dallas, in fahrenheit?"
    },
]

sampling_params = SamplingParams(max_tokens=8192, temperature=0.0)
outputs = llm.chat(messages, sampling_params=sampling_params, tools=tools)
output = outputs[0].outputs[0].text.strip()

# Parse tool calls
tool_calls = json.loads(output)
tool_answers = [
    available_tools[call["name"]](**call["arguments"])
    for call in tool_calls
]

# Continue conversation with tool results
messages.append({"role": "assistant", "content": output})
messages.append({
    "role": "tool",
    "content": "\n\n".join(tool_answers),
    "tool_call_id": generate_random_id(),
})

outputs = llm.chat(messages, sampling_params, tools=tools)
print(outputs[0].outputs[0].text)
```

#### Online Serving

**File:** `/vllm/examples/online_serving/openai_chat_completion_client_with_tools.py`

**Server:**
```bash
vllm serve mistralai/Mistral-7B-Instruct-v0.3 \
    --chat-template examples/tool_chat_template_mistral.jinja \
    --enable-auto-tool-choice \
    --tool-call-parser mistral
```

**Client:**
```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="EMPTY"
)

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Get weather in a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string"},
                    "state": {"type": "string"},
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                },
                "required": ["city", "state", "unit"],
            },
        },
    }
]

response = client.chat.completions.create(
    model="mistralai/Mistral-7B-Instruct-v0.3",
    messages=[
        {"role": "user", "content": "What's the weather in San Francisco?"}
    ],
    tools=tools,
    tool_choice="auto",
)

# Process tool calls
if response.choices[0].message.tool_calls:
    for tool_call in response.choices[0].message.tool_calls:
        func_name = tool_call.function.name
        func_args = json.loads(tool_call.function.arguments)
        result = available_tools[func_name](**func_args)
        print(f"Result: {result}")
```

---

### SGLang

**Server:**
```bash
python -m sglang.launch_server \
    --model-path mistralai/Mistral-7B-Instruct-v0.3 \
    --tool-call-parser mistral
```

**Client (identical to vLLM):**
```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:30000/v1",
    api_key="EMPTY"
)

# Same API as vLLM
response = client.chat.completions.create(
    model="mistralai/Mistral-7B-Instruct-v0.3",
    messages=[...],
    tools=tools,
    tool_choice="auto",
)
```

**Streaming:**
```python
stream = client.chat.completions.create(
    model="mistralai/Mistral-7B-Instruct-v0.3",
    messages=messages,
    tools=tools,
    stream=True,
)

for chunk in stream:
    if chunk.choices[0].delta.tool_calls:
        print(chunk.choices[0].delta.tool_calls[0])
```

**MCP Tools:**
```python
response = client.responses.create(
    model=model,
    input="Execute: print('Hello from Python!')",
    instructions="Use the Python tool to execute code.",
    tools=[
        {
            "type": "mcp",
            "server_label": "code_interpreter",
            "server_url": "http://localhost:8888",
            "allowed_tools": ["*"],
        }
    ],
)
```

---

### TensorRT-LLM

#### Chat Completions API

**File:** `/TensorRT-LLM/examples/serve/compatibility/chat_completions/example_06_tool_calling.py`

**Server:**
```bash
trtllm-serve Qwen/Qwen2.5-7B-Instruct --tool_parser qwen3
```

**Client:**
```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="tensorrt_llm"
)

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather in a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City and state, e.g. San Francisco, CA",
                    },
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                },
                "required": ["location"],
            },
        },
    }
]

response = client.chat.completions.create(
    model="Qwen/Qwen2.5-7B-Instruct",
    messages=[
        {"role": "user", "content": "What is the weather in San Francisco?"}
    ],
    tools=tools,
    tool_choice="auto",
    max_tokens=4096,
)

# Process tool calls
if response.choices[0].message.tool_calls:
    for tool_call in response.choices[0].message.tool_calls:
        func_name = tool_call.function.name
        func_args = json.loads(tool_call.function.arguments)

        # Execute function
        result = get_weather(**func_args)

        # Send result back
        final_response = client.chat.completions.create(
            model="Qwen/Qwen2.5-7B-Instruct",
            messages=[
                {"role": "user", "content": "What is the weather in San Francisco?"},
                response.choices[0].message,
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(result),
                },
            ],
            max_tokens=4096,
        )
        print(final_response.choices[0].message.content)
```

#### Responses API (Gemini-style)

**File:** `/TensorRT-LLM/examples/serve/compatibility/responses/example_05_tool_calling.py`

```python
tools = [
    {
        "name": "get_weather",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
            },
            "required": ["location"],
        },
        "type": "function",
        "description": "Get the current weather in a location",
    }
]

response = client.responses.create(
    model=model,
    input="What is the weather in San Francisco?",
    tools=tools,
    tool_choice="auto",
    max_output_tokens=4096,
)

# Process function call
if response.output[0].type == "function_call":
    func_name = response.output[0].name
    func_args = json.loads(response.output[0].arguments)
    tool_call_id = response.output[0].call_id

    # Execute and send result
    result = eval(f"{func_name}(**{func_args})")

    response = client.responses.create(
        model=model,
        input=[
            {
                "type": "function_call_output",
                "call_id": tool_call_id,
                "output": json.dumps(result),
            }
        ],
        previous_response_id=response.id,
        tools=tools,
    )
```

---

## Feature Comparison Matrix

| Feature | vLLM | SGLang | TensorRT-LLM |
|---------|------|--------|--------------|
| **Number of Parsers** | 37 | 24 | 6 |
| **Parser Architecture** | Abstract ToolParser | Format Detector | BaseToolParser |
| **Registry System** | Lazy + Eager Loading | Central Registry | Factory Pattern |
| **Streaming Support** | ✅ State machine with deltas | ✅ Incremental buffer | ✅ Sophisticated state tracking |
| **OpenAI API Compatibility** | ✅ Full | ✅ Full | ✅ Full + Responses API |
| **Structured Output Backends** | 4 (xgrammar, guidance, outlines, lm-format-enforcer) | 1 (xgrammar) | 2 (xgrammar, llguidance) |
| **Structural Tag Support** | ✅ Yes | ✅ Yes | ✅ Yes |
| **JSON Schema Validation** | ✅ Yes | ✅ Yes (Draft202012) | ✅ Yes |
| **Regex Constraints** | ✅ Yes | ✅ Yes | ✅ Yes |
| **EBNF Grammar** | ✅ Yes | ✅ Yes | ✅ Yes |
| **Tool Choice Options** | auto, none, required, named | auto, none, required, named | auto, none, named |
| **Parallel Tool Calls** | ✅ Yes | ✅ Yes | ✅ Yes |
| **Plugin System** | ✅ Custom parsers via plugins | ❌ No | ❌ No |
| **MCP Tool Server** | ✅ Demo + custom URLs | ✅ Demo + custom URLs | ❌ No |
| **Reasoning Parser** | ✅ Separate reasoning parser support | ✅ Yes | ❌ No |
| **Tool Strictness Levels** | ❌ No | ✅ 3 levels (OFF, FUNCTION, PARAMETER) | ❌ No |
| **Exclude Tools on None** | ✅ Optional flag | ❌ No | ❌ No |
| **Auto-enable Tool Choice** | ✅ When tools provided | ✅ When tools provided | ✅ When tools provided |
| **Grammar Caching** | ✅ Backend-specific | ✅ Central cache with futures | ✅ Yes |
| **Distributed Sync** | ❌ No | ✅ Multi-rank sync | ❌ No |
| **Environment Config** | ❌ No | ✅ 4 env variables | ❌ No |
| **Tool Validation** | ✅ Name validation | ✅ Schema validation (Draft202012) | ✅ Name validation |
| **Pythonic Format** | ✅ Yes (no structural tag) | ✅ Yes (no structural tag) | ❌ No |
| **XML Formats** | ✅ Multiple (hermes, qwen3) | ✅ Multiple (hermes, qwen25) | ✅ Multiple (qwen3, deepseek_v32) |
| **Unicode Delimiters** | ✅ DeepSeek formats | ✅ DeepSeek formats | ✅ DeepSeek formats |
| **Model-Specific Optimizations** | ✅ Per-parser | ✅ Per-detector | ✅ TensorRT optimized |

---

## Code Sources & Implementation Details

### vLLM Architecture

#### Core Framework Files

| Component | File Path | Lines | Key Classes/Functions |
|-----------|-----------|-------|----------------------|
| Abstract Base | `/vllm/tool_parsers/abstract_tool_parser.py` | 34-120 | `ToolParser` base class |
| Parser Registry | `/vllm/tool_parsers/__init__.py` | 24-157 | `_TOOL_PARSERS_TO_REGISTER` |
| Parser Manager | `/vllm/parser/parser_manager.py` | 190-308 | `get_tool_parser()`, `get_parser()` |
| CLI Args | `/vllm/entrypoints/openai/cli_args.py` | 111-133 | Tool-related arguments |
| Protocol | `/vllm/entrypoints/openai/engine/protocol.py` | 218-256 | `ToolCall`, `FunctionCall` |
| Request Protocol | `/vllm/entrypoints/openai/chat_completion/protocol.py` | 174-184, 637-708 | Request validation |
| Serving | `/vllm/entrypoints/openai/chat_completion/serving.py` | 130-136, 1332-1375 | Tool extraction |
| Structured Output Config | `/vllm/config/structured_outputs.py` | 18-74 | `StructuredOutputsConfig` |
| Backend Manager | `/vllm/v1/structured_output/__init__.py` | 35-175 | `StructuredOutputManager` |
| XGrammar Backend | `/vllm/v1/structured_output/backend_xgrammar.py` | - | XGrammar integration |
| Utilities | `/vllm/tool_parsers/utils.py` | - | JSON parsing, streaming helpers |

#### Example Parsers

| Parser | File Path | Format Pattern |
|--------|-----------|----------------|
| Mistral | `/vllm/tool_parsers/mistral_tool_parser.py` | `[TOOL_CALLS] [{...}, ...]` |
| Llama 3 JSON | `/vllm/tool_parsers/llama_tool_parser.py` | JSON structure |
| Hermes 2 Pro | `/vllm/tool_parsers/hermes_tool_parser.py` | `<tool_call>{...}</tool_call>` |
| DeepSeek V3 | `/vllm/tool_parsers/deepseekv3_tool_parser.py` | Unicode + JSON |
| Qwen 3 XML | `/vllm/tool_parsers/qwen3xml_tool_parser.py` | XML-based |
| Pythonic | `/vllm/tool_parsers/pythonic_tool_parser.py` | `[func(arg=val), ...]` |

---

### SGLang Architecture

#### Core Framework Files

| Component | File Path | Lines | Key Classes/Functions |
|-----------|-----------|-------|----------------------|
| FunctionCallParser | `/sglang/python/sglang/srt/function_call/function_call_parser.py` | 39-215 | `FunctionCallParser` orchestrator |
| BaseFormatDetector | `/sglang/python/sglang/srt/function_call/base_format_detector.py` | 26-347 | Abstract detector base |
| Core Types | `/sglang/python/sglang/srt/function_call/core_types.py` | 1-34 | `ToolCallItem`, `StreamingParseResult`, `StructureInfo` |
| Grammar Manager | `/sglang/python/sglang/srt/constrained/grammar_manager.py` | 24-196 | Constrained generation |
| Tool Server | `/sglang/python/sglang/srt/entrypoints/openai/tool_server.py` | 1-176 | MCP integration |
| OpenAI Serving | `/sglang/python/sglang/srt/entrypoints/openai/serving_chat.py` | 87-300+ | Chat completions handler |
| Server Args | `/sglang/python/sglang/srt/server_args.py` | 414, 3678-3691 | CLI arguments |
| Environment Config | `/sglang/python/sglang/srt/environ.py` | 145-156, 275, 453 | Env variables |
| Protocol | `/sglang/python/sglang/srt/entrypoints/openai/protocol.py` | 457-527 | Data models |

#### Example Detectors

| Detector | File Path | Format Pattern |
|----------|-----------|----------------|
| Mistral | `/sglang/python/sglang/srt/function_call/mistral_detector.py` | `[TOOL_CALLS] [{...}, ...]` |
| Hermes | `/sglang/python/sglang/srt/function_call/hermes_detector.py` | `<tool_call>{...}</tool_call>` |
| Pythonic | `/sglang/python/sglang/srt/function_call/pythonic_detector.py` | `[func(arg1=val1), ...]` |
| Qwen25 | `/sglang/python/sglang/srt/function_call/qwen25_detector.py` | `<tool_call>\n{...}\n</tool_call>` |
| DeepSeek V3 | `/sglang/python/sglang/srt/function_call/deepseekv3_detector.py` | Unicode + JSON |

---

### TensorRT-LLM Architecture

#### Core Framework Files

| Component | File Path | Lines | Key Classes/Functions |
|-----------|-----------|-------|----------------------|
| BaseToolParser | `/TensorRT-LLM/tensorrt_llm/serve/tool_parser/base_tool_parser.py` | 1-325 | Abstract base with streaming |
| ToolParserFactory | `/TensorRT-LLM/tensorrt_llm/serve/tool_parser/tool_parser_factory.py` | - | Factory pattern |
| Core Types | `/TensorRT-LLM/tensorrt_llm/serve/tool_parser/core_types.py` | 23-35 | `StructureInfo`, `_GetInfoFunc` |
| Utilities | `/TensorRT-LLM/tensorrt_llm/serve/tool_parser/utils.py` | - | JSON helpers |
| Serve Command | `/TensorRT-LLM/tensorrt_llm/commands/serve.py` | 421-427, 475-571 | CLI arguments |
| Protocol | `/TensorRT-LLM/tensorrt_llm/serve/openai_protocol.py` | 193-235, 372-394, 514-722 | Data models, guided decoding |
| Postprocessing | `/TensorRT-LLM/tensorrt_llm/serve/postprocess_handlers.py` | 44-80, 137-159 | Tool parsing application |
| OpenAI Server | `/TensorRT-LLM/tensorrt_llm/serve/openai_server.py` | - | Main server class |
| Responses Utils | `/TensorRT-LLM/tensorrt_llm/serve/responses_utils.py` | - | Responses API support |

#### All Parsers

| Parser | File Path | Format Pattern |
|--------|-----------|----------------|
| Qwen3 | `/TensorRT-LLM/tensorrt_llm/serve/tool_parser/qwen3_tool_parser.py` | `<tool_call>\n{...}\n</tool_call>` |
| Qwen3 Coder | `/TensorRT-LLM/tensorrt_llm/serve/tool_parser/qwen3_coder_parser.py` | `<function=...><parameter=...>` |
| DeepSeek V3 | `/TensorRT-LLM/tensorrt_llm/serve/tool_parser/deepseekv3_parser.py` | Unicode + JSON |
| DeepSeek V3.1 | `/TensorRT-LLM/tensorrt_llm/serve/tool_parser/deepseekv31_parser.py` | Unicode compact |
| DeepSeek V3.2 | `/TensorRT-LLM/tensorrt_llm/serve/tool_parser/deepseekv32_parser.py` | DSML XML or JSON |
| Kimi K2 | `/TensorRT-LLM/tensorrt_llm/serve/tool_parser/kimi_k2_tool_parser.py` | `<\|tool_call_begin\|>` |

---

## Key Differences Summary

### Parser Coverage
- **vLLM:** Widest coverage (37 parsers), best for diverse model support
- **SGLang:** Balanced coverage (24 detectors), strong cache-aware integration
- **TensorRT-LLM:** Focused coverage (6 parsers), optimized for performance

### Extensibility
- **vLLM:** Plugin system for custom parsers, flexible backend selection
- **SGLang:** Environment-based configuration, MCP tool server support
- **TensorRT-LLM:** Factory pattern, tight TensorRT integration

### Structured Output
- **vLLM:** 4 backends (xgrammar, guidance, outlines, lm-format-enforcer)
- **SGLang:** Single optimized backend (xgrammar), distributed sync
- **TensorRT-LLM:** 2 backends (xgrammar, llguidance), structural tags

### Unique Features
- **vLLM:** Reasoning parser, tool exclusion flag, unified parser management
- **SGLang:** Tool strictness levels, MCP integration, grammar caching with futures
- **TensorRT-LLM:** Responses API, TensorRT optimization, streamlined deployment

### Best Use Cases
- **vLLM:** General-purpose inference with maximum model compatibility
- **SGLang:** High-throughput serving with RadixAttention and cache-aware scheduling
- **TensorRT-LLM:** Production deployments needing maximum performance on NVIDIA hardware

---

## Conclusion

All three systems provide robust tool calling implementations with OpenAI API compatibility. The choice depends on:

1. **Model Support Needs:**
   - Wide variety → vLLM (37 parsers)
   - Balanced → SGLang (24 detectors)
   - Specific models → TensorRT-LLM (6 parsers)

2. **Performance Requirements:**
   - Maximum throughput → SGLang (cache-aware scheduling)
   - Maximum latency optimization → TensorRT-LLM (TensorRT optimization)
   - Balanced → vLLM (multiple backends)

3. **Integration Requirements:**
   - Custom parsers → vLLM (plugin system)
   - MCP tools → vLLM or SGLang
   - Production NVIDIA → TensorRT-LLM

4. **Flexibility Needs:**
   - Backend choice → vLLM (4 backends)
   - Environment config → SGLang (extensive env vars)
   - Streamlined deployment → TensorRT-LLM (factory pattern)

Each system excels in different areas, making the choice dependent on specific deployment requirements and use cases.
