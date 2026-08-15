"""Provider-agnostic LLM client abstraction (Milestone 8B).

The RAG-only agent (and, later, the tool-calling and hybrid agents)
depend only on the LLMClient protocol below — never on a concrete SDK
class — so the experiment can swap models or providers without touching
agent code. AnthropicLLMClient is the only concrete implementation today.

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

from typing import Protocol, TypeVar

from pydantic import BaseModel

DEFAULT_MODEL = "claude-opus-4-8"
DEFAULT_MAX_TOKENS = 4096
DEFAULT_EFFORT = "low"

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


class AnthropicLLMClient:
    """LLMClient implementation backed by the Anthropic Claude API.

    Uses client.messages.parse(..., output_format=response_model), the
    SDK's schema-validating convenience method — the response is already a
    validated instance of response_model, not a JSON string to re-parse.
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
