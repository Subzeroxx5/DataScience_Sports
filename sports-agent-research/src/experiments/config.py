"""Controlled experiment configuration (Milestone 12).

Centralizes every setting the experiment runner needs so nothing is
scattered through the codebase (milestones/current.md, Milestone 12
Step 2): which architectures/scenarios/repetitions to run, the model
configuration shared identically across all three architectures
(Step 3: "Architecture should remain the primary independent
variable"), RAG top_k, the tool-calling loop bound, execution mode
(MOCK/REAL), and where output is written.

This module contains no agent, no LLM call, and no ground-truth read —
just configuration, the deterministic scenario manifest (Step 7 — never
includes an expected answer), the deterministic architecture-rotation
policy (Step 9), and reproducibility metadata (Step 15 — checksums of
the controlled artifacts the experiment depends on).
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path

from pydantic import BaseModel, Field, field_validator

from src.agents.llm_client import DEFAULT_EFFORT, DEFAULT_MODEL, LLMProviderName
from src.agents.rag_evidence import DEFAULT_RAG_TOP_K
from src.agents.tool_agent import MAX_TOOL_ITERATIONS
from src.evaluation.dataset import DATA_DIR, load_scenario_definitions_by_id
from src.evaluation.tool_agent_evaluation import DEFAULT_SCENARIO_IDS
from src.models import ArchitectureType, MarketType

DEFAULT_OUTPUT_DIR = "results/experiments"

# The controlled artifacts an experiment's reproducibility depends on —
# checksummed into experiment metadata (Step 15), never modified by this
# milestone. Paths are relative to the project root.
_CHECKSUM_ARTIFACTS: dict[str, Path] = {
    "current_odds": DATA_DIR / "current_odds.json",
    "rag_corpus": DATA_DIR / "rag_documents" / "corpus.jsonl",
    "ground_truth": DATA_DIR / "ground_truth.json",
    "quant_ground_truth": DATA_DIR / "quant_ground_truth.json",
    "rag_index_config": DATA_DIR / "rag_index" / "config.json",
}

# Kept in sync with docs/QUANT_STRATEGY.md's implemented calculation set
# (Milestone 7A/7B) — a plain identifier, not a hash, since the quant
# module itself is source code (already version-controlled) rather than
# a data artifact.
QUANT_METHODOLOGY_VERSION = "market_consensus_leave_one_out_v1"


class ExecutionMode(str, Enum):
    """Step 2/11-12: MOCK drives the real architecture pipelines with a
    deterministic fake LLM (no API cost, reproducible); REAL drives them
    with the configured Anthropic model."""

    MOCK = "mock"
    REAL = "real"


class ExperimentScenario(BaseModel):
    """One scenario as presented to every architecture identically
    (Step 6/7). Deliberately excludes any expected answer — ground truth
    is evaluator-only (Step 20). `quant_evaluable` is benchmark
    *metadata* (whether the market structurally has two-sided coverage),
    not a ground-truth value — it is never used to compute or shortcut
    an agent's answer."""

    scenario_id: str = Field(..., min_length=1)
    game_id: str = Field(..., min_length=1)
    market_type: MarketType
    selected_outcome: str = Field(..., min_length=1)
    query: str = Field(..., min_length=1)
    quant_evaluable: bool | None = None


def _canonical_query(market_type: MarketType, selected_outcome: str) -> str:
    """Step 6: one natural-language query template, identical in form
    for every architecture and every scenario — no sportsbook names, no
    per-architecture tuning, matching the milestone's own example
    ("Compare the available moneyline prices for the Lakers and
    identify the best current value.")."""
    return (
        f"Compare the available {market_type.value} prices for "
        f"{selected_outcome} and identify the best current value."
    )


def build_scenario_manifest(scenario_ids: list[str]) -> list[ExperimentScenario]:
    """Deterministic scenario manifest (Step 7), built from the existing
    controlled benchmark (src/evaluation/dataset.py) — never from a new,
    separate scenario source. `quant_ground_truth`'s `quant_evaluable`
    flag is read here only as descriptive metadata; the manifest never
    carries `expected_best_sportsbook`/`expected_odds`/`expected_ev`/any
    ground-truth answer field."""
    from src.evaluation.quant_ground_truth import generate_all_quant_ground_truth

    definitions_by_id = load_scenario_definitions_by_id()
    quant_evaluable_by_id = {q.scenario_id: q.quant_evaluable for q in generate_all_quant_ground_truth()}

    manifest: list[ExperimentScenario] = []
    for scenario_id in scenario_ids:
        definition = definitions_by_id[scenario_id]
        market = definition["market"]
        market_type = MarketType(market["market_type"])
        selected_outcome = market["selected_outcome"]
        manifest.append(
            ExperimentScenario(
                scenario_id=scenario_id,
                game_id=definition["game"]["game_id"],
                market_type=market_type,
                selected_outcome=selected_outcome,
                query=_canonical_query(market_type, selected_outcome),
                quant_evaluable=quant_evaluable_by_id.get(scenario_id),
            )
        )
    return manifest


# Canonical architecture order used at repetition 0; later repetitions
# rotate from this base (Step 9).
DEFAULT_ARCHITECTURE_ORDER: list[ArchitectureType] = [
    ArchitectureType.RAG,
    ArchitectureType.TOOL,
    ArchitectureType.HYBRID,
]

EXECUTION_ORDER_POLICY = "rotating"


def execution_order_for_repetition(
    architectures: list[ArchitectureType], repetition_index: int
) -> list[ArchitectureType]:
    """Step 9: a deterministic left-rotation by `repetition_index mod
    len(architectures)` — no randomness, nothing to seed. For
    architectures=[RAG, TOOL, HYBRID]:

        repetition_index=0 -> [RAG, TOOL, HYBRID]
        repetition_index=1 -> [TOOL, HYBRID, RAG]
        repetition_index=2 -> [HYBRID, RAG, TOOL]
        repetition_index=3 -> [RAG, TOOL, HYBRID]   (cycle repeats)

    matching the milestone's own example exactly.
    """
    if not architectures:
        return []
    offset = repetition_index % len(architectures)
    return architectures[offset:] + architectures[:offset]


class ExperimentConfig(BaseModel):
    """Step 2: the one place every experiment setting is defined.
    Reused identically across all three architectures (Step 3) — the
    runner never lets one architecture see a different model, effort,
    top_k, or scenario set than another.
    """

    experiment_id: str = Field(..., min_length=1)
    experiment_name: str = Field(..., min_length=1)

    architectures: list[ArchitectureType] = Field(default_factory=lambda: list(DEFAULT_ARCHITECTURE_ORDER))
    scenario_ids: list[str] = Field(default_factory=lambda: list(DEFAULT_SCENARIO_IDS))
    repetitions: int = Field(..., ge=1)

    # Provider selection is configuration-driven (Pre-Milestone 14A
    # checkpoint, Step 2) — never architecture-specific. Defaults to the
    # Anthropic API (this field did not exist before that checkpoint;
    # every prior milestone's behavior is unchanged by its default).
    llm_provider: LLMProviderName = LLMProviderName.ANTHROPIC
    model_name: str = DEFAULT_MODEL
    # Anthropic's DEFAULT_MODEL rejects `temperature` outright (see
    # src/agents/llm_client.py's module docstring) — always None/N/A
    # there, recorded for documentation completeness rather than as a
    # usable lever; `effort` is the determinism control for that
    # provider. Ollama DOES use `temperature` as its determinism lever
    # (see DEFAULT_OLLAMA_TEMPERATURE) — when llm_provider=ollama, a
    # non-None value here is passed straight to OllamaLLMClient.
    temperature: float | None = None
    effort: str = DEFAULT_EFFORT

    rag_top_k: int = Field(default=DEFAULT_RAG_TOP_K, ge=1)
    max_tool_iterations: int = Field(default=MAX_TOOL_ITERATIONS, ge=1)

    output_dir: str = DEFAULT_OUTPUT_DIR
    execution_mode: ExecutionMode = ExecutionMode.MOCK
    execution_order_policy: str = EXECUTION_ORDER_POLICY

    provider_type: str = "ControlledOddsProvider"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    quant_methodology_version: str = QUANT_METHODOLOGY_VERSION

    @field_validator("architectures")
    @classmethod
    def architectures_not_empty_and_unique(cls, value: list[ArchitectureType]) -> list[ArchitectureType]:
        if not value:
            raise ValueError("architectures cannot be empty")
        if len(set(value)) != len(value):
            raise ValueError("architectures cannot contain duplicates")
        return value

    @field_validator("scenario_ids")
    @classmethod
    def scenario_ids_not_empty(cls, value: list[str]) -> list[str]:
        if not value:
            raise ValueError("scenario_ids cannot be empty")
        return value

    def expected_run_count(self) -> int:
        return len(self.architectures) * len(self.scenario_ids) * self.repetitions


class ExperimentMetadata(BaseModel):
    """Step 15/16: config plus everything needed to interpret or
    reproduce the experiment later — persisted as config.json alongside
    the raw results. Never stores API keys."""

    config: ExperimentConfig
    created_at: datetime
    artifact_checksums: dict[str, str]

    @classmethod
    def build(cls, config: ExperimentConfig) -> "ExperimentMetadata":
        return cls(
            config=config,
            created_at=datetime.now(timezone.utc),
            artifact_checksums=compute_artifact_checksums(),
        )


def compute_artifact_checksums() -> dict[str, str]:
    """Step 15: sha256 of each controlled artifact this experiment's
    reproducibility depends on. Missing files are recorded explicitly
    rather than silently omitted, so a checksum mismatch/absence is
    itself informative."""
    checksums: dict[str, str] = {}
    for name, path in _CHECKSUM_ARTIFACTS.items():
        if not path.is_file():
            checksums[name] = "missing"
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        checksums[name] = f"sha256:{digest}"
    return checksums
