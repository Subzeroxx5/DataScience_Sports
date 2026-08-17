"""Provider-agnostic LLM client abstraction (Milestone 8B; extended for
tool calling in Milestone 9A; extended with a local-inference provider in
the Pre-Milestone 14A checkpoint).

The RAG-only agent (Milestone 8B) and the tool-calling agent (Milestone
9A, and later hybrid) depend only on the protocols below — never on a
concrete SDK class — so the experiment can swap models or providers
without touching agent code. Two concrete implementations exist:
AnthropicLLMClient (Anthropic API) and OllamaLLMClient (a locally hosted
Ollama server). Both implement LLMClient (single-shot structured
extraction, used by the RAG-only agent) and ToolCallingLLMClient
(multi-turn tool orchestration, used by the tool-calling/hybrid agents)
side by side on one class, one constructor, and one model configuration
— there is deliberately no architecture-specific client class and no
architecture-specific secret/connection handling — see
milestones/current.md (Milestone 9A, Step 2) and the Pre-Milestone 14A
"Local Real-LLM Backend" checkpoint (Step 2: "Provider selection must be
configuration-driven... Do not use architecture-specific providers").

Model configuration is centralized here (DEFAULT_MODEL, DEFAULT_MAX_TOKENS,
DEFAULT_EFFORT for Anthropic; DEFAULT_OLLAMA_MODEL, DEFAULT_OLLAMA_BASE_URL,
DEFAULT_OLLAMA_TEMPERATURE for Ollama) rather than repeated at call sites.
Which provider an experiment actually uses is a config value
(src.experiments.config.ExperimentConfig.llm_provider /
src.experiments.agent_factory.build_llm_client), never hard-coded inside
an agent.

On determinism: the milestone's instruction is "temperature = 0, or the
lowest deterministic setting the provider supports." DEFAULT_MODEL
(claude-opus-4-8) does not accept temperature/top_p/top_k at all — passing
any of them is a 400 (removed on this model tier). The closest available
lever is `output_config.effort`; DEFAULT_EFFORT="low" is the model's
"short, scoped, non-creative task" setting, the best fit for a structured
extraction job that should read evidence rather than compose prose. No
model in this family guarantees byte-identical output across calls, with
or without a temperature parameter — see the extraction validation layer
(src/agents/extraction.py) for how the pipeline stays trustworthy despite
that: a plausible-but-wrong response is rejected at the provenance-check
step, not accepted just because the model was "confident." Ollama, by
contrast, does accept `temperature`, so OllamaLLMClient uses
DEFAULT_OLLAMA_TEMPERATURE = 0.0 — the actual lowest-deterministic-setting
lever for that provider.

Secrets: the Anthropic API key is never read or stored by this module
directly — anthropic.Anthropic() resolves ANTHROPIC_API_KEY (or an `ant
auth login` profile) from the environment on its own. See .env.example.
Ollama requires no API key at all (a local, unauthenticated server) — see
OllamaLLMClient's docstring.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Protocol, TypeVar

from pydantic import BaseModel, Field

DEFAULT_MODEL = "claude-opus-4-8"
DEFAULT_MAX_TOKENS = 4096
DEFAULT_EFFORT = "low"

# Tool-calling loops (Milestone 9A) tend to need more turns' worth of
# headroom than a single structured-extraction call; a dedicated default
# keeps the RAG-only agent's DEFAULT_MAX_TOKENS unchanged.
DEFAULT_TOOL_MAX_TOKENS = 8192

# --- Local (Ollama) provider defaults -------------------------------------
#
# Pre-Milestone 14A checkpoint: llama3.1:8b is the ONE local model used by
# all three architectures (RAG, TOOL, HYBRID) — verified locally to support
# both structured/schema-constrained JSON output (RAG's generate_structured)
# and native tool calling, including a full multi-turn tool-call ->
# tool-result -> final-answer round trip (TOOL/HYBRID's create_turn). Never
# duplicate this identifier elsewhere — every call site reads it from here
# (via ExperimentConfig.model_name, populated from this constant).
DEFAULT_OLLAMA_MODEL = "llama3.1:8b"
DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_OLLAMA_TEMPERATURE = 0.0

# Central, single-source-of-truth record of the local-provider choice for
# the final experiment (Pre-Milestone 14A checkpoint, Step 4: "Record
# centrally... Do not hard-code the model name in multiple agents").
# src.experiments.config.ExperimentConfig / agent_factory.py read these,
# rather than any agent module defining its own copy.
LLM_PROVIDER = "ollama"
LLM_MODEL = DEFAULT_OLLAMA_MODEL

T = TypeVar("T", bound=BaseModel)


class LLMProviderName(str, Enum):
    """Configuration-driven provider selection (Pre-Milestone 14A
    checkpoint, Step 2) — src.experiments.agent_factory.build_llm_client
    branches on this, never on architecture."""

    ANTHROPIC = "anthropic"
    OLLAMA = "ollama"


class LLMClient(Protocol):
    """Anything that can turn (system prompt, user prompt) into a
    validated instance of a given Pydantic response model."""

    def generate_structured(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        response_model: type[T],
    ) -> T:
        ...


class ToolSchema(BaseModel):
    """One LLM-callable tool definition, in the provider-agnostic shape
    every concrete ToolCallingLLMClient is responsible for translating
    into its own wire format (e.g. Anthropic's tools=[...] parameter)."""

    name: str = Field(..., min_length=1)
    description: str = Field(..., min_length=1)
    input_schema: dict[str, Any]


class ToolUseBlock(BaseModel):
    """One tool invocation the model requested during a turn."""

    id: str = Field(..., min_length=1)
    name: str = Field(..., min_length=1)
    input: dict[str, Any]


class ToolCallTurn(BaseModel):
    """The model's response for one turn of a tool-calling loop.

    Provider-agnostic: a fake LLMClient used in tests constructs this
    directly, with no dependency on any Anthropic SDK response type.
    """

    stop_reason: str
    text: str | None = None
    tool_uses: list[ToolUseBlock] = Field(default_factory=list)


class ToolCallingLLMClient(Protocol):
    """Anything that can run one turn of a multi-turn tool-calling
    conversation: given a system prompt, the conversation so far (in the
    minimal role/content-block shape below), and the tools available,
    return the model's next turn.

    `messages` uses the same minimal shape as the Anthropic Messages API
    (`[{"role": "user" | "assistant", "content": [...]}]`) since that is
    expressive enough for any tool-calling provider without inventing a
    second message format — but nothing in src/agents/tool_agent.py
    inspects provider-specific fields beyond this shape.
    """

    def create_turn(
        self,
        *,
        system_prompt: str,
        messages: list[dict[str, Any]],
        tools: list[ToolSchema],
    ) -> ToolCallTurn:
        ...


class AnthropicLLMClient:
    """LLM client backed by the Anthropic Claude API — implements both
    LLMClient (generate_structured, Milestone 8B) and ToolCallingLLMClient
    (create_turn, Milestone 9A) on one class, one constructor, and one
    model/max_tokens/effort configuration. See the module docstring for
    why this is intentionally not split into two client classes.

    generate_structured() uses client.messages.parse(...,
    output_format=response_model), the SDK's schema-validating
    convenience method. create_turn() uses the lower-level
    client.messages.create(..., tools=...) directly, since tool calling
    has no equivalent single-call convenience wrapper — src/agents/
    tool_agent.py drives the multi-turn loop around it.
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        effort: str = DEFAULT_EFFORT,
    ) -> None:
        import anthropic  # deferred: keep this module importable without the

        # `anthropic` package installed, so unit tests using a fake
        # LLMClient never require it (see tests/test_rag_agent.py).
        self.model = model
        self.max_tokens = max_tokens
        self.effort = effort
        self._client = anthropic.Anthropic()

    def generate_structured(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        response_model: type[T],
    ) -> T:
        response = self._client.messages.parse(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
            output_config={"effort": self.effort},
            output_format=response_model,
        )
        return response.parsed_output

    def create_turn(
        self,
        *,
        system_prompt: str,
        messages: list[dict[str, Any]],
        tools: list[ToolSchema],
    ) -> ToolCallTurn:
        response = self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system_prompt,
            messages=messages,
            tools=[tool.model_dump() for tool in tools],
            output_config={"effort": self.effort},
        )

        text_blocks = [block.text for block in response.content if block.type == "text"]
        tool_uses = [
            ToolUseBlock(id=block.id, name=block.name, input=block.input)
            for block in response.content
            if block.type == "tool_use"
        ]
        return ToolCallTurn(
            stop_reason=response.stop_reason,
            text=" ".join(text_blocks) if text_blocks else None,
            tool_uses=tool_uses,
        )


def _anthropic_messages_to_ollama(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Converts the Anthropic-Messages-API-shaped `messages` list every
    ToolCallingLLMClient caller already builds (src/agents/tool_agent.py,
    src/agents/hybrid_agent.py — content is either a plain string or a
    list of {"type": "text"|"tool_use"|"tool_result", ...} blocks) into
    Ollama's native chat message shape. This is the ONLY place that
    format conversion happens — src/agents/tool_agent.py itself is
    unmodified and stays provider-agnostic."""
    ollama_messages: list[dict[str, Any]] = []
    for message in messages:
        role = message["role"]
        content = message["content"]
        if isinstance(content, str):
            ollama_messages.append({"role": role, "content": content})
            continue

        text_parts = [block["text"] for block in content if block.get("type") == "text"]
        tool_use_blocks = [block for block in content if block.get("type") == "tool_use"]
        tool_result_blocks = [block for block in content if block.get("type") == "tool_result"]

        if tool_use_blocks:
            ollama_messages.append({
                "role": "assistant",
                "content": " ".join(text_parts),
                "tool_calls": [
                    {"function": {"name": block["name"], "arguments": block["input"]}}
                    for block in tool_use_blocks
                ],
            })
        elif tool_result_blocks:
            for block in tool_result_blocks:
                ollama_messages.append({"role": "tool", "content": block["content"]})
        elif text_parts:
            ollama_messages.append({"role": role, "content": " ".join(text_parts)})
    return ollama_messages


def _tool_schemas_to_ollama(tools: list[ToolSchema]) -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.input_schema,
            },
        }
        for tool in tools
    ]


class OllamaLLMClient:
    """LLM client backed by a locally hosted Ollama server (Pre-Milestone
    14A checkpoint) — implements both LLMClient (generate_structured) and
    ToolCallingLLMClient (create_turn) on one class, mirroring
    AnthropicLLMClient's shape so agents/agent_factory never need to know
    which concrete provider they are talking to.

    Local inference through this class is REAL execution, never the
    project's deterministic MOCK fakes (src/evaluation/*_evaluation.py's
    Deterministic*PolicyLLMClient classes) — see
    src.experiments.config.ExecutionMode / agent_factory.build_llm_client.

    Requires `ollama serve` (or `brew services start ollama`) running
    locally and the configured model already pulled (`ollama pull
    <model>`) — this class does not pull a model on your behalf, since
    silently downloading multi-gigabyte weights mid-experiment would be a
    surprising side effect.

    Uses `httpx` directly against Ollama's native `/api/chat` endpoint —
    no LangChain or other orchestration framework, per the checkpoint's
    Step 3 instruction to keep this the smallest clean adapter.
    """

    def __init__(
        self,
        model: str = DEFAULT_OLLAMA_MODEL,
        base_url: str = DEFAULT_OLLAMA_BASE_URL,
        temperature: float = DEFAULT_OLLAMA_TEMPERATURE,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        timeout: float = 180.0,
    ) -> None:
        import httpx  # deferred, mirrors AnthropicLLMClient's deferred `anthropic` import —
        # keeps this module importable without httpx installed for callers that never use Ollama.

        self.model = model
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._client = httpx.Client(base_url=self.base_url, timeout=timeout)

    def _post_chat(self, payload: dict[str, Any]) -> dict[str, Any]:
        payload = {
            **payload,
            "model": self.model,
            "stream": False,
            "options": {"temperature": self.temperature, "num_predict": self.max_tokens},
        }
        try:
            response = self._client.post("/api/chat", json=payload)
            response.raise_for_status()
        except Exception as exc:  # connection refused, timeout, 4xx/5xx, ... — bounded, typed failure
            raise OllamaRequestError(
                f"Ollama request to {self.base_url}/api/chat failed (model={self.model!r}): {exc!r}"
            ) from exc
        return response.json()

    def generate_structured(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        response_model: type[T],
    ) -> T:
        result = self._post_chat({
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "format": response_model.model_json_schema(),
        })
        content = result.get("message", {}).get("content", "")
        try:
            return response_model.model_validate_json(content)
        except Exception as exc:
            raise OllamaRequestError(
                f"Ollama response did not match {response_model.__name__}'s schema: {exc!r}; raw content={content!r}"
            ) from exc

    def create_turn(
        self,
        *,
        system_prompt: str,
        messages: list[dict[str, Any]],
        tools: list[ToolSchema],
    ) -> ToolCallTurn:
        ollama_messages = [{"role": "system", "content": system_prompt}] + _anthropic_messages_to_ollama(messages)
        result = self._post_chat({
            "messages": ollama_messages,
            "tools": _tool_schemas_to_ollama(tools),
        })
        message = result.get("message", {})
        raw_tool_calls = message.get("tool_calls") or []

        tool_uses = [
            ToolUseBlock(
                id=str(call.get("id") or f"call_{index}"),
                name=call["function"]["name"],
                input=call["function"]["arguments"],
            )
            for index, call in enumerate(raw_tool_calls)
        ]
        text = message.get("content") or None
        return ToolCallTurn(
            stop_reason="tool_use" if tool_uses else "end_turn",
            text=text,
            tool_uses=tool_uses,
        )


class OllamaRequestError(RuntimeError):
    """Raised for any bounded Ollama-request failure (connection refused,
    timeout, HTTP error status, or a response that doesn't match the
    requested structured-output schema) — never a bare/unbounded
    exception type, so callers (src/agents/*.py, which already catch
    Exception around each LLM call and record it in their trace) get a
    clear, provider-attributable error message."""
