"""Tests for configuration-driven LLM provider selection (Pre-Milestone
14A checkpoint, Step 2): src.experiments.config.ExperimentConfig.llm_provider
and src.experiments.agent_factory.build_llm_client. No running Ollama
server or Anthropic credentials required — REAL-mode client construction
never itself makes a network call (only generate_structured/create_turn
do), and MOCK mode never touches either provider.
"""

from src.agents.llm_client import AnthropicLLMClient, LLMProviderName, OllamaLLMClient
from src.evaluation.hybrid_agent_evaluation import DeterministicHybridPolicyLLMClient
from src.evaluation.rag_agent_evaluation import DeterministicRagPolicyLLMClient
from src.evaluation.tool_agent_evaluation import DeterministicToolPolicyLLMClient
from src.experiments.agent_factory import build_llm_client
from src.experiments.config import ExecutionMode, ExperimentConfig
from src.models import ArchitectureType


def _config(**overrides) -> ExperimentConfig:
    defaults = dict(
        experiment_id="provider-test", experiment_name="provider-test",
        repetitions=1, execution_mode=ExecutionMode.REAL,
    )
    defaults.update(overrides)
    return ExperimentConfig(**defaults)


# ---------------------------------------------------------------------------
# ExperimentConfig.llm_provider
# ---------------------------------------------------------------------------


def test_llm_provider_defaults_to_anthropic_preserving_prior_milestone_behavior():
    config = _config()
    assert config.llm_provider == LLMProviderName.ANTHROPIC


def test_llm_provider_accepts_ollama():
    config = _config(llm_provider=LLMProviderName.OLLAMA, model_name="llama3.1:8b")
    assert config.llm_provider == LLMProviderName.OLLAMA


def test_llm_provider_accepts_plain_string_value():
    config = _config(llm_provider="ollama")
    assert config.llm_provider == LLMProviderName.OLLAMA


def test_llm_provider_round_trips_through_json():
    config = _config(llm_provider=LLMProviderName.OLLAMA, model_name="llama3.1:8b")
    restored = ExperimentConfig.model_validate_json(config.model_dump_json())
    assert restored.llm_provider == LLMProviderName.OLLAMA


# ---------------------------------------------------------------------------
# build_llm_client — REAL mode provider branching
# ---------------------------------------------------------------------------


def test_default_provider_builds_anthropic_client_for_every_architecture():
    config = _config()
    for architecture in (ArchitectureType.RAG, ArchitectureType.TOOL, ArchitectureType.HYBRID):
        client = build_llm_client(architecture, config)
        assert isinstance(client, AnthropicLLMClient)


def test_ollama_provider_builds_ollama_client_for_every_architecture():
    config = _config(llm_provider=LLMProviderName.OLLAMA, model_name="llama3.1:8b")
    for architecture in (ArchitectureType.RAG, ArchitectureType.TOOL, ArchitectureType.HYBRID):
        client = build_llm_client(architecture, config)
        assert isinstance(client, OllamaLLMClient)


def test_all_three_architectures_get_the_identical_model_and_provider():
    config = _config(llm_provider=LLMProviderName.OLLAMA, model_name="llama3.1:8b", temperature=0.0)
    clients = [
        build_llm_client(architecture, config)
        for architecture in (ArchitectureType.RAG, ArchitectureType.TOOL, ArchitectureType.HYBRID)
    ]
    models = {client.model for client in clients}
    temperatures = {client.temperature for client in clients}
    assert models == {"llama3.1:8b"}
    assert temperatures == {0.0}


def test_ollama_client_uses_default_temperature_when_config_temperature_is_none():
    from src.agents.llm_client import DEFAULT_OLLAMA_TEMPERATURE

    config = _config(llm_provider=LLMProviderName.OLLAMA, model_name="llama3.1:8b", temperature=None)
    client = build_llm_client(ArchitectureType.TOOL, config)
    assert client.temperature == DEFAULT_OLLAMA_TEMPERATURE


def test_rag_gets_smaller_max_tokens_than_tool_and_hybrid_regardless_of_provider():
    from src.agents.llm_client import DEFAULT_MAX_TOKENS, DEFAULT_TOOL_MAX_TOKENS

    for provider in (LLMProviderName.ANTHROPIC, LLMProviderName.OLLAMA):
        config = _config(llm_provider=provider, model_name="llama3.1:8b" if provider == LLMProviderName.OLLAMA else "claude-opus-4-8")
        rag_client = build_llm_client(ArchitectureType.RAG, config)
        tool_client = build_llm_client(ArchitectureType.TOOL, config)
        assert rag_client.max_tokens == DEFAULT_MAX_TOKENS
        assert tool_client.max_tokens == DEFAULT_TOOL_MAX_TOKENS


# ---------------------------------------------------------------------------
# build_llm_client — MOCK mode is unaffected by llm_provider
# ---------------------------------------------------------------------------


def test_mock_mode_ignores_llm_provider_and_always_returns_deterministic_fakes():
    for provider in (LLMProviderName.ANTHROPIC, LLMProviderName.OLLAMA):
        config = _config(execution_mode=ExecutionMode.MOCK, llm_provider=provider)
        assert isinstance(build_llm_client(ArchitectureType.RAG, config), DeterministicRagPolicyLLMClient)
        assert isinstance(build_llm_client(ArchitectureType.TOOL, config), DeterministicToolPolicyLLMClient)
        assert isinstance(build_llm_client(ArchitectureType.HYBRID, config), DeterministicHybridPolicyLLMClient)


# ---------------------------------------------------------------------------
# No architecture-specific provider (Step 2/Acceptance Criteria) — the
# provider (client class) must be identical across all three
# architectures for a fixed config; only max_tokens is allowed to vary
# by architecture (RAG's single-shot call vs. TOOL/HYBRID's multi-turn
# loop), never the provider itself.
# ---------------------------------------------------------------------------


def test_probe_real_llm_connectivity_uses_the_configured_provider_not_anthropic_always(monkeypatch):
    """Regression test (Milestone 14A pre-execution check): the
    connectivity probe used to hard-code AnthropicLLMClient regardless
    of config.llm_provider, so an Ollama-configured experiment would
    always report "REAL EXPERIMENT: NOT RUN" even with Ollama up and
    reachable. probe_real_llm_connectivity must build its probe client
    via build_llm_client() so it actually exercises the configured
    provider."""
    from src.agents.llm_client import ToolCallTurn
    from src.experiments.runner import probe_real_llm_connectivity

    calls = []

    def fake_ollama_create_turn(self, **kwargs):
        calls.append("ollama")
        return ToolCallTurn(stop_reason="end_turn", text="ready")

    monkeypatch.setattr(OllamaLLMClient, "create_turn", fake_ollama_create_turn)

    def fail_if_called(self, **kwargs):
        calls.append("anthropic")
        raise AssertionError("AnthropicLLMClient.create_turn must not be called for an Ollama-configured probe")

    monkeypatch.setattr(AnthropicLLMClient, "create_turn", fail_if_called)

    config = _config(llm_provider=LLMProviderName.OLLAMA, model_name="llama3.1:8b")
    ok, err = probe_real_llm_connectivity(config)

    assert ok is True
    assert err is None
    assert calls == ["ollama"]


def test_probe_real_llm_connectivity_still_uses_anthropic_by_default(monkeypatch):
    calls = []

    def fake_create_turn(self, **kwargs):
        calls.append("anthropic")
        from src.agents.llm_client import ToolCallTurn

        return ToolCallTurn(stop_reason="end_turn", text="ready")

    monkeypatch.setattr(AnthropicLLMClient, "create_turn", fake_create_turn)

    from src.experiments.runner import probe_real_llm_connectivity

    config = _config()  # default provider = anthropic
    ok, err = probe_real_llm_connectivity(config)

    assert ok is True
    assert calls == ["anthropic"]


def test_provider_type_is_identical_across_architectures_for_both_providers():
    for provider, model in (
        (LLMProviderName.ANTHROPIC, "claude-opus-4-8"),
        (LLMProviderName.OLLAMA, "llama3.1:8b"),
    ):
        config = _config(llm_provider=provider, model_name=model)
        client_types = {
            type(build_llm_client(architecture, config))
            for architecture in (ArchitectureType.RAG, ArchitectureType.TOOL, ArchitectureType.HYBRID)
        }
        assert len(client_types) == 1
