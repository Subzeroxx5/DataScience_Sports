"""Centralized architecture instantiation for the experiment runner
(Milestone 12, Step 8): one `create_agent(architecture, config)` rather
than separate `run_rag_experiment.py` / `run_tool_experiment.py` /
`run_hybrid_experiment.py` scripts each wiring up their own agent.

LLM client selection (Step 11/12) reuses the EXACT deterministic fake
policies already built and verified in Milestones 9B-11
(`DeterministicRagPolicyLLMClient`, `DeterministicToolPolicyLLMClient`,
`DeterministicHybridPolicyLLMClient`) for MOCK mode — these drive the
real retrieval/tool-call pipelines with only the LLM decision layer
faked, never fabricating a final BettingAnalysis directly. REAL mode is
configuration-driven (Pre-Milestone 14A checkpoint, Step 2:
`ExperimentConfig.llm_provider`) between `AnthropicLLMClient` and
`OllamaLLMClient` — never a per-architecture choice; whichever provider
is configured, every architecture shares the identical model
configuration (Step 3).
"""

from __future__ import annotations

from src.agents.base import Agent
from src.agents.hybrid_agent import HybridAgent
from src.agents.llm_client import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_OLLAMA_BASE_URL,
    DEFAULT_OLLAMA_TEMPERATURE,
    DEFAULT_TOOL_MAX_TOKENS,
    AnthropicLLMClient,
    LLMProviderName,
    OllamaLLMClient,
)
from src.agents.rag_agent import RagOnlyAgent
from src.agents.tool_agent import ToolCallingAgent
from src.evaluation.hybrid_agent_evaluation import DeterministicHybridPolicyLLMClient
from src.evaluation.rag_agent_evaluation import DeterministicRagPolicyLLMClient
from src.evaluation.tool_agent_evaluation import DeterministicToolPolicyLLMClient
from src.experiments.config import ExecutionMode, ExperimentConfig
from src.models import ArchitectureType
from src.providers.controlled import ControlledOddsProvider
from src.rag.retriever import Retriever
from src.tools.sportsbook_tools import SportsbookTools


def build_llm_client(architecture: ArchitectureType, config: ExperimentConfig):
    """Step 11: MOCK mode reuses the existing fake-LLM infrastructure
    verbatim. Step 12: REAL mode builds ONE provider client — Anthropic
    or Ollama, selected by `config.llm_provider`, never by architecture
    — configured identically (model/effort or model/temperature) for
    RAG, TOOL, and HYBRID alike."""
    if config.execution_mode == ExecutionMode.MOCK:
        if architecture == ArchitectureType.RAG:
            return DeterministicRagPolicyLLMClient()
        if architecture == ArchitectureType.TOOL:
            return DeterministicToolPolicyLLMClient()
        if architecture == ArchitectureType.HYBRID:
            return DeterministicHybridPolicyLLMClient()
        raise ValueError(f"unknown architecture: {architecture!r}")

    # REAL: RAG makes single-shot structured-extraction calls
    # (DEFAULT_MAX_TOKENS); tool-calling/hybrid drive a multi-turn loop
    # that needs more headroom (DEFAULT_TOOL_MAX_TOKENS) — matching the
    # per-architecture defaults already established in Milestones 8B/9A.
    # This bound applies identically regardless of provider.
    max_tokens = DEFAULT_MAX_TOKENS if architecture == ArchitectureType.RAG else DEFAULT_TOOL_MAX_TOKENS

    if config.llm_provider == LLMProviderName.OLLAMA:
        temperature = config.temperature if config.temperature is not None else DEFAULT_OLLAMA_TEMPERATURE
        return OllamaLLMClient(
            model=config.model_name, base_url=DEFAULT_OLLAMA_BASE_URL,
            temperature=temperature, max_tokens=max_tokens,
        )

    return AnthropicLLMClient(model=config.model_name, max_tokens=max_tokens, effort=config.effort)


def create_agent(
    architecture: ArchitectureType, config: ExperimentConfig, llm_client=None
) -> tuple[Agent, object]:
    """Returns (agent, auxiliary_handle) — the auxiliary handle is
    whatever that architecture's own evaluator needs for its independent
    hallucination re-check (a Retriever for RAG, SportsbookTools for
    tool/hybrid) — see src/evaluation/*_evaluation.py::evaluate_scenario.
    """
    llm_client = llm_client if llm_client is not None else build_llm_client(architecture, config)

    if architecture == ArchitectureType.RAG:
        retriever = Retriever.from_directory()
        agent = RagOnlyAgent(retriever, llm_client=llm_client, top_k=config.rag_top_k)
        return agent, retriever

    if architecture == ArchitectureType.TOOL:
        tools = SportsbookTools(ControlledOddsProvider())
        agent = ToolCallingAgent(tools, llm_client=llm_client, max_iterations=config.max_tool_iterations)
        return agent, tools

    if architecture == ArchitectureType.HYBRID:
        retriever = Retriever.from_directory()
        tools = SportsbookTools(ControlledOddsProvider())
        agent = HybridAgent(
            retriever, tools, llm_client=llm_client,
            top_k=config.rag_top_k, max_tool_iterations=config.max_tool_iterations,
        )
        return agent, tools

    raise ValueError(f"unknown architecture: {architecture!r}")
