"""Tests for src/agents/llm_client.py's OllamaLLMClient (Pre-Milestone
14A checkpoint, Step 9): message-format conversion, request/response
handling, and bounded error handling — all against a mocked HTTP layer.
No running Ollama server is required or contacted.
"""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from src.agents.llm_client import (
    DEFAULT_OLLAMA_BASE_URL,
    DEFAULT_OLLAMA_MODEL,
    DEFAULT_OLLAMA_TEMPERATURE,
    LLM_MODEL,
    LLM_PROVIDER,
    OllamaLLMClient,
    OllamaRequestError,
    ToolSchema,
    ToolUseBlock,
    _anthropic_messages_to_ollama,
    _tool_schemas_to_ollama,
)


class _FakeResponse:
    def __init__(self, json_payload: dict, status_code: int = 200):
        self._json_payload = json_payload
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self) -> dict:
        return self._json_payload


def _client_with_fake_post(monkeypatch, response_payload: dict | None = None, raise_exc: Exception | None = None):
    client = OllamaLLMClient()

    def fake_post(path, json=None):
        if raise_exc is not None:
            raise raise_exc
        return _FakeResponse(response_payload)

    monkeypatch.setattr(client._client, "post", fake_post)
    return client


# ---------------------------------------------------------------------------
# Central constants (Step 4)
# ---------------------------------------------------------------------------


def test_central_provider_and_model_constants_are_recorded():
    assert LLM_PROVIDER == "ollama"
    assert LLM_MODEL == DEFAULT_OLLAMA_MODEL


def test_default_ollama_temperature_is_the_lowest_deterministic_setting():
    assert DEFAULT_OLLAMA_TEMPERATURE == 0.0


# ---------------------------------------------------------------------------
# Message-format conversion (Anthropic-shape -> Ollama-shape)
# ---------------------------------------------------------------------------


def test_plain_string_content_passes_through():
    messages = [{"role": "user", "content": "hello"}]
    assert _anthropic_messages_to_ollama(messages) == [{"role": "user", "content": "hello"}]


def test_tool_use_blocks_become_assistant_tool_calls():
    messages = [{
        "role": "assistant",
        "content": [{"type": "tool_use", "id": "call_1", "name": "get_odds", "input": {"game_id": "G-1"}}],
    }]
    converted = _anthropic_messages_to_ollama(messages)
    assert converted == [{
        "role": "assistant", "content": "",
        "tool_calls": [{"function": {"name": "get_odds", "arguments": {"game_id": "G-1"}}}],
    }]


def test_tool_result_blocks_become_tool_role_messages():
    messages = [{
        "role": "user",
        "content": [{"type": "tool_result", "tool_use_id": "call_1", "content": "[]", "is_error": False}],
    }]
    converted = _anthropic_messages_to_ollama(messages)
    assert converted == [{"role": "tool", "content": "[]"}]


def test_text_blocks_are_joined():
    messages = [{"role": "user", "content": [{"type": "text", "text": "part one"}]}]
    assert _anthropic_messages_to_ollama(messages) == [{"role": "user", "content": "part one"}]


def test_tool_schemas_convert_to_ollama_function_format():
    tools = [ToolSchema(name="get_odds", description="desc", input_schema={"type": "object", "properties": {}})]
    converted = _tool_schemas_to_ollama(tools)
    assert converted == [{
        "type": "function",
        "function": {"name": "get_odds", "description": "desc", "parameters": {"type": "object", "properties": {}}},
    }]


# ---------------------------------------------------------------------------
# generate_structured
# ---------------------------------------------------------------------------


class _Extracted(BaseModel):
    sportsbook: str
    american_odds: int


def test_generate_structured_parses_valid_json_response(monkeypatch):
    client = _client_with_fake_post(
        monkeypatch, {"message": {"content": '{"sportsbook": "FanDuel", "american_odds": 125}'}}
    )
    result = client.generate_structured(system_prompt="extract", user_prompt="text", response_model=_Extracted)
    assert result == _Extracted(sportsbook="FanDuel", american_odds=125)


def test_generate_structured_raises_ollama_error_on_schema_mismatch(monkeypatch):
    client = _client_with_fake_post(monkeypatch, {"message": {"content": "not json"}})
    with pytest.raises(OllamaRequestError):
        client.generate_structured(system_prompt="extract", user_prompt="text", response_model=_Extracted)


def test_generate_structured_raises_ollama_error_on_connection_failure(monkeypatch):
    client = _client_with_fake_post(monkeypatch, raise_exc=ConnectionError("refused"))
    with pytest.raises(OllamaRequestError):
        client.generate_structured(system_prompt="extract", user_prompt="text", response_model=_Extracted)


# ---------------------------------------------------------------------------
# create_turn
# ---------------------------------------------------------------------------


def test_create_turn_with_tool_calls_returns_tool_use_blocks(monkeypatch):
    client = _client_with_fake_post(monkeypatch, {
        "message": {
            "content": "",
            "tool_calls": [{"function": {"name": "get_odds", "arguments": {"game_id": "G-1"}}}],
        }
    })
    turn = client.create_turn(system_prompt="sys", messages=[{"role": "user", "content": "hi"}], tools=[])
    assert turn.stop_reason == "tool_use"
    assert turn.tool_uses == [ToolUseBlock(id=turn.tool_uses[0].id, name="get_odds", input={"game_id": "G-1"})]


def test_create_turn_without_tool_calls_returns_end_turn(monkeypatch):
    client = _client_with_fake_post(monkeypatch, {"message": {"content": "final answer"}})
    turn = client.create_turn(system_prompt="sys", messages=[{"role": "user", "content": "hi"}], tools=[])
    assert turn.stop_reason == "end_turn"
    assert turn.text == "final answer"
    assert turn.tool_uses == []


def test_create_turn_raises_ollama_error_on_connection_failure(monkeypatch):
    client = _client_with_fake_post(monkeypatch, raise_exc=TimeoutError("timed out"))
    with pytest.raises(OllamaRequestError):
        client.create_turn(system_prompt="sys", messages=[{"role": "user", "content": "hi"}], tools=[])


# ---------------------------------------------------------------------------
# Construction defaults
# ---------------------------------------------------------------------------


def test_client_defaults_match_the_recorded_central_constants():
    client = OllamaLLMClient()
    assert client.model == DEFAULT_OLLAMA_MODEL
    assert client.base_url == DEFAULT_OLLAMA_BASE_URL
    assert client.temperature == DEFAULT_OLLAMA_TEMPERATURE


def test_client_construction_does_not_contact_the_network(monkeypatch):
    # Constructing OllamaLLMClient must never itself make a request —
    # only generate_structured/create_turn do. If it did, this test
    # would hang/fail without a running server.
    OllamaLLMClient()
