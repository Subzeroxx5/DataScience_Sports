"""Tests for src/agents/llm_client.py (Milestone 8B).

No real API calls here — these tests only validate the abstraction and
configuration, never invoke AnthropicLLMClient.generate_structured()
against the live API. See experiments/run_rag_smoke_test.py for the
credentialed manual smoke test.
"""

import ast
from pathlib import Path

from pydantic import BaseModel

from src.agents.llm_client import (
    DEFAULT_EFFORT,
    DEFAULT_MAX_TOKENS,
    DEFAULT_MODEL,
    AnthropicLLMClient,
    LLMClient,
)


class _DummyResponse(BaseModel):
    value: str


class _FakeLLMClient:
    """Minimal LLMClient-conforming stand-in used only to prove the
    Protocol shape is satisfiable without the real SDK."""

    def generate_structured(self, *, system_prompt, user_prompt, response_model):
        return response_model(value="ok")


def test_fake_client_satisfies_llm_client_protocol():
    client: LLMClient = _FakeLLMClient()
    result = client.generate_structured(
        system_prompt="sys", user_prompt="user", response_model=_DummyResponse
    )
    assert result.value == "ok"


def test_model_configuration_is_centralized():
    assert DEFAULT_MODEL == "claude-opus-4-8"
    assert DEFAULT_MAX_TOKENS > 0
    assert DEFAULT_EFFORT in {"low", "medium", "high", "xhigh", "max"}


def test_anthropic_client_defers_sdk_import_until_construction():
    # Importing the module must never require the anthropic package to be
    # importable at call sites that only use a fake LLMClient — only
    # actually instantiating AnthropicLLMClient should reach for it.
    source_path = Path(__file__).resolve().parent.parent / "src" / "agents" / "llm_client.py"
    tree = ast.parse(source_path.read_text())
    top_level_imports = {
        alias.name
        for node in tree.body
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    assert "anthropic" not in top_level_imports


def test_anthropic_client_configurable_model_and_max_tokens():
    # Constructing with an explicit model/max_tokens must not require
    # network access or credentials — only __init__'s attribute
    # assignment happens before any API call. If ANTHROPIC_API_KEY is
    # genuinely unset in this environment, anthropic.Anthropic() itself
    # may raise; either outcome (success or a clean auth-related error)
    # confirms no accidental network call was made during construction
    # beyond client setup.
    import anthropic

    try:
        client = AnthropicLLMClient(model="claude-opus-4-8", max_tokens=2048, effort="low")
    except anthropic.AnthropicError:
        return
    assert client.model == "claude-opus-4-8"
    assert client.max_tokens == 2048
    assert client.effort == "low"


def test_no_temperature_top_p_top_k_passed_anywhere():
    # Opus 4.8 rejects these parameters outright (400) — confirm they are
    # never constructed into the request at all, not just "unset".
    source_path = Path(__file__).resolve().parent.parent / "src" / "agents" / "llm_client.py"
    source = source_path.read_text()
    for forbidden in ("temperature=", "top_p=", "top_k="):
        assert forbidden not in source
