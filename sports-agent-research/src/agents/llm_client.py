"""Provider-agnostic LLM client abstraction (Milestone 8B; extended for
tool calling in Milestone 9A).

The RAG-only agent (Milestone 8B) and the tool-calling agent (Milestone
9A, and later hybrid) depend only on the protocols below — never on a
concrete SDK class — so the experiment can swap models or providers
without touching agent code. AnthropicLLMClient is the only concrete
implementation today, and is the single class both architectures use: it
implements LLMClient (single-shot structured extraction, used by the
RAG-only agent) and ToolCallingLLMClient (multi-turn tool orchestration,
used by the tool-calling agent) side by side, sharing one constructor,
one model/max_tokens/effort configuration, and one Anthropic API key
resolution path. There is deliberately no second client class, no second
model-configuration system, and no architecture-specific secret handling
— see milestones/current.md (Milestone 9A, Step 2).

Model configuration is centralized here (DEFAULT_MODEL, DEFAULT_MAX_TOKENS,
DEFAULT_EFFORT) rather than repeated at call sites.

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
step, not accepted just because the model was "confident."

Secrets: the Anthropic API key is never read or stored by this module
directly — anthropic.Anthropic() resolves ANTHROPIC_API_KEY (or an `ant
auth login` profile) from the environment on its own. See .env.example.
"""

from __future__ import annotations

from typing import Any, Protocol, TypeVar

from pydantic import BaseModel, Field

DEFAULT_MODEL = "claude-opus-4-8"
DEFAULT_MAX_TOKENS = 4096
DEFAULT_EFFORT = "low"

# Tool-calling loops (Milestone 9A) tend to need more turns' worth of
# headroom than a single structured-extraction call; a dedicated default
# keeps the RAG-only agent's DEFAULT_MAX_TOKENS unchanged.
DEFAULT_TOOL_MAX_TOKENS = 8192

T = TypeVar("T", bound=BaseModel)


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
