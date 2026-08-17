"""Dashboard data-loading and orchestration helpers (Milestone 13).

Thin layer between the Streamlit UI and the existing project modules —
contains no sportsbook/quant business logic of its own (see
docs/ARCHITECTURE.md, "Dashboard / UI": "must not define or influence
experimental logic, quant calculations, or ground truth"). Every
displayed number originates from an existing model, agent, or evaluator
function; this module only constructs requests via existing factories,
loads persisted experiment output, and filters/groups it.

    Dashboard
       |
       v
    Existing Agent / Experiment Result   (this module's only job)
       |
       v
    Display                              (dashboard/*_view.py)
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import streamlit as st

from src.agents.base import Agent, AgentRequest
from src.agents.hybrid_agent import HybridAgentTrace, HybridAnalysisIncomplete
from src.agents.rag_agent import RagAgentTrace, RagAnalysisIncomplete
from src.agents.tool_agent import ToolAgentTrace, ToolAnalysisIncomplete
from src.evaluation import hybrid_agent_evaluation, rag_agent_evaluation, tool_agent_evaluation
from src.evaluation.dataset import load_scenario_definitions, load_scenario_definitions_by_id
from src.evaluation.ground_truth import generate_all_ground_truth
from src.evaluation.metrics import FailureCategory
from src.evaluation.quant_ground_truth import generate_all_quant_ground_truth
from src.experiments.agent_factory import create_agent
from src.experiments.config import (
    ExecutionMode,
    ExperimentConfig,
    ExperimentMetadata,
    ExperimentScenario,
    build_scenario_manifest,
)
from src.experiments.runner import ExperimentSummary, RawExperimentRun
from src.models import ArchitectureType, BettingAnalysis, GroundTruth, QuantGroundTruth

DEFAULT_EXPERIMENTS_DIR = Path("results/experiments")

AgentTrace = RagAgentTrace | ToolAgentTrace | HybridAgentTrace

_ARCH_RESULT_MODEL = {
    ArchitectureType.RAG: rag_agent_evaluation.RagAgentEvaluationResult,
    ArchitectureType.TOOL: tool_agent_evaluation.ToolAgentEvaluationResult,
    ArchitectureType.HYBRID: hybrid_agent_evaluation.HybridAgentEvaluationResult,
}

_INCOMPLETE_EXCEPTIONS = (RagAnalysisIncomplete, ToolAnalysisIncomplete, HybridAnalysisIncomplete)


# ---------------------------------------------------------------------------
# Scenario manifest (Section 4) — reuse the Milestone 12 manifest builder,
# never a hand-reconstructed scenario in the UI.
# ---------------------------------------------------------------------------


def all_scenario_ids() -> list[str]:
    return sorted(d["scenario_id"] for d in load_scenario_definitions())


@st.cache_data(show_spinner=False)
def full_scenario_manifest() -> list[ExperimentScenario]:
    """Every controlled scenario, in the exact shape the experiment
    runner uses (src.experiments.config.build_scenario_manifest) — no
    ground-truth fields, one canonical query per scenario."""
    return build_scenario_manifest(all_scenario_ids())


def build_demo_request(scenario: ExperimentScenario) -> AgentRequest:
    """The SAME AgentRequest shape the experiment runner builds for a
    scenario (src.experiments.runner.run_experiment) — identity + the
    canonical query only, never a ground-truth field."""
    definition = load_scenario_definitions_by_id()[scenario.scenario_id]
    return AgentRequest(
        scenario_id=scenario.scenario_id,
        game_id=definition["game"]["game_id"],
        market_type=scenario.market_type,
        selected_outcome=scenario.selected_outcome,
        query=scenario.query,
    )


def game_context(scenario_id: str) -> dict:
    """Non-ground-truth game/market context (team names, sport) for
    display only — the same benchmark definition the manifest itself
    reads from."""
    return load_scenario_definitions_by_id()[scenario_id]


# ---------------------------------------------------------------------------
# Demo-mode agent execution (Sections 5-10)
# ---------------------------------------------------------------------------


@st.cache_resource(show_spinner="Loading architecture (index/model)...")
def _cached_agent(
    architecture: ArchitectureType,
    execution_mode: ExecutionMode,
    model_name: str,
    effort: str,
    rag_top_k: int,
    max_tool_iterations: int,
) -> tuple[Agent, object]:
    """Construct one architecture's agent exactly via
    src.experiments.agent_factory.create_agent — the same factory the
    experiment runner uses (Section 5: "Do not create dashboard-specific
    agents"). Cached per (architecture, execution_mode, model settings)
    for the Streamlit session so the RAG vector index/embedding model
    and the sportsbook tool provider are not rebuilt on every click
    (Section 26). Only construction is cached — agent.analyze() itself
    is never cached, so every "Run Analysis" click is a genuine,
    independent execution (Section 26: caching must not contaminate
    repeated runs; demo caching and experiment execution stay separate —
    this function is never called from the experiment runner)."""
    config = ExperimentConfig(
        experiment_id="dashboard-demo",
        experiment_name="dashboard-demo",
        architectures=[architecture],
        scenario_ids=["S001"],  # unused by create_agent(); ExperimentConfig just requires non-empty
        repetitions=1,
        model_name=model_name,
        effort=effort,
        rag_top_k=rag_top_k,
        max_tool_iterations=max_tool_iterations,
        execution_mode=execution_mode,
    )
    return create_agent(architecture, config)


@dataclass
class DemoRunResult:
    """Everything the Demo view needs to render one agent run — wraps
    the agent's own output/trace, adds nothing new. `analysis` is None
    exactly when the architecture raised its own *AnalysisIncomplete
    exception, or a lower-level failure (e.g. no real LLM configured)
    occurred; `trace` is still populated whenever the agent produced one
    (every *AnalysisIncomplete carries the full trace — see
    src/agents/*.py)."""

    architecture: ArchitectureType
    scenario: ExperimentScenario
    request: AgentRequest
    analysis: BettingAnalysis | None
    trace: AgentTrace | None
    incomplete: bool
    error_message: str | None
    # The same Retriever (RAG/HYBRID) or SportsbookTools (TOOL/HYBRID)
    # handle create_agent() returned — exposed so the Demo view can
    # render a live sportsbook comparison table for the TOOL
    # architecture, whose trace only stores a result-summary string, not
    # structured per-book prices (see
    # src/agents/tool_agent.py::ToolCallRecord). Read-only: never used to
    # bypass or duplicate the agent's own analysis.
    aux_handle: object = None


def run_demo_analysis(
    architecture: ArchitectureType,
    scenario: ExperimentScenario,
    execution_mode: ExecutionMode,
    model_name: str,
    effort: str,
    rag_top_k: int,
    max_tool_iterations: int,
) -> DemoRunResult:
    """Run one architecture on one scenario through the existing agent
    interface only (Section 5): create_agent(...) -> AgentRequest ->
    BettingAnalysis. Never touches ground truth. Handles a real LLM not
    being configured, or the architecture legitimately finding
    insufficient evidence, without crashing (Section 25) — both surface
    as a normal DemoRunResult with `incomplete=True` and the trace/error
    preserved for display, never a fabricated result."""
    agent, aux_handle = _cached_agent(
        architecture, execution_mode, model_name, effort, rag_top_k, max_tool_iterations
    )
    request = build_demo_request(scenario)

    try:
        analysis = agent.analyze(request)
    except _INCOMPLETE_EXCEPTIONS as exc:
        return DemoRunResult(
            architecture=architecture, scenario=scenario, request=request,
            analysis=None, trace=exc.trace, incomplete=True, error_message=str(exc),
            aux_handle=aux_handle,
        )
    except Exception as exc:  # e.g. real LLM not configured / network failure
        return DemoRunResult(
            architecture=architecture, scenario=scenario, request=request,
            analysis=None, trace=getattr(agent, "last_trace", None), incomplete=True,
            error_message=f"{type(exc).__name__}: {exc}", aux_handle=aux_handle,
        )

    return DemoRunResult(
        architecture=architecture, scenario=scenario, request=request,
        analysis=analysis, trace=agent.last_trace, incomplete=False, error_message=None,
        aux_handle=aux_handle,
    )


# ---------------------------------------------------------------------------
# Experiment (research) loading (Section 11) — read-only, never
# recalculates an agent output.
# ---------------------------------------------------------------------------


class ExperimentLoadError(Exception):
    """Raised for a missing/incomplete experiment directory (Section 25)
    — caught by the UI layer to show a clear message, never a crash."""


@dataclass
class LoadedExperiment:
    experiment_dir: Path
    metadata: ExperimentMetadata
    manifest: list[ExperimentScenario]
    raw_runs: list[RawExperimentRun]
    summary: ExperimentSummary


def list_experiment_ids(base_dir: Path | str = DEFAULT_EXPERIMENTS_DIR) -> list[str]:
    """Every experiment_id with a persisted config.json under base_dir,
    newest first by directory mtime. Returns [] (never raises) if
    base_dir does not exist yet — Section 25: a missing experiment
    directory must not crash the dashboard."""
    base = Path(base_dir)
    if not base.is_dir():
        return []
    candidates = [p for p in base.iterdir() if p.is_dir() and (p / "config.json").is_file()]
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return [p.name for p in candidates]


def load_experiment(experiment_id: str, base_dir: Path | str = DEFAULT_EXPERIMENTS_DIR) -> LoadedExperiment:
    experiment_dir = Path(base_dir) / experiment_id
    if not experiment_dir.is_dir():
        raise ExperimentLoadError(f"No experiment directory found at {experiment_dir}")

    config_path = experiment_dir / "config.json"
    manifest_path = experiment_dir / "manifest.json"
    raw_path = experiment_dir / "raw_results.jsonl"
    summary_path = experiment_dir / "summary.json"
    missing = [p.name for p in (config_path, manifest_path, raw_path, summary_path) if not p.is_file()]
    if missing:
        raise ExperimentLoadError(f"Experiment {experiment_id!r} is missing file(s): {missing}")

    metadata = ExperimentMetadata.model_validate_json(config_path.read_text())
    manifest = [ExperimentScenario.model_validate(item) for item in json.loads(manifest_path.read_text())]

    raw_runs: list[RawExperimentRun] = []
    for line in raw_path.read_text().splitlines():
        line = line.strip()
        if line:
            raw_runs.append(RawExperimentRun.model_validate_json(line))

    summary = ExperimentSummary.model_validate_json(summary_path.read_text())

    return LoadedExperiment(
        experiment_dir=experiment_dir, metadata=metadata, manifest=manifest,
        raw_runs=raw_runs, summary=summary,
    )


def is_mock_experiment(loaded: LoadedExperiment) -> bool:
    return loaded.metadata.config.execution_mode == ExecutionMode.MOCK


# ---------------------------------------------------------------------------
# Filtering / grouping raw runs (Sections 17-21) — pure list operations,
# no metric recomputation.
# ---------------------------------------------------------------------------


def filter_raw_runs(
    runs: list[RawExperimentRun],
    architecture: ArchitectureType | None = None,
    scenario_id: str | None = None,
    repetition: int | None = None,
    status: FailureCategory | None = None,
) -> list[RawExperimentRun]:
    result = runs
    if architecture is not None:
        result = [r for r in result if r.architecture == architecture]
    if scenario_id is not None:
        result = [r for r in result if r.scenario_id == scenario_id]
    if repetition is not None:
        result = [r for r in result if r.repetition == repetition]
    if status is not None:
        result = [r for r in result if r.common_result.execution_status == status]
    return result


def group_by_scenario(runs: list[RawExperimentRun]) -> dict[str, list[RawExperimentRun]]:
    groups: dict[str, list[RawExperimentRun]] = {}
    for run in runs:
        groups.setdefault(run.scenario_id, []).append(run)
    return groups


def group_by_architecture(runs: list[RawExperimentRun]) -> dict[ArchitectureType, list[RawExperimentRun]]:
    groups: dict[ArchitectureType, list[RawExperimentRun]] = {}
    for run in runs:
        groups.setdefault(run.architecture, []).append(run)
    return groups


def failure_counts_by_architecture(runs: list[RawExperimentRun]) -> dict[ArchitectureType, dict[str, int]]:
    """Tally of execution_status values per architecture — a plain count
    of an already-computed categorical field (mirrors
    src.evaluation.metrics.summarize()'s own failure_counts pattern),
    never a new failure classification."""
    counts: dict[ArchitectureType, dict[str, int]] = {}
    for run in runs:
        by_status = counts.setdefault(run.architecture, {})
        status = run.common_result.execution_status.value
        by_status[status] = by_status.get(status, 0) + 1
    return counts


def reconstruct_architecture_specific_result(run: RawExperimentRun):
    """The full per-architecture evaluator result (RagAgentEvaluationResult
    / ToolAgentEvaluationResult / HybridAgentEvaluationResult) the run was
    persisted from — reconstructed via that model's own validation, never
    a dashboard-defined shape."""
    model = _ARCH_RESULT_MODEL[run.architecture]
    return model.model_validate(run.architecture_specific_result)


def hybrid_conflict_summary(runs: list[RawExperimentRun]) -> dict | None:
    """Section 20: hybrid-specific aggregates, computed only via the
    existing src.evaluation.hybrid_agent_evaluation.summarize_results
    (Milestone 10B/11) — never a dashboard-local formula. None if no
    hybrid runs are present."""
    hybrid_runs = [r for r in runs if r.architecture == ArchitectureType.HYBRID]
    if not hybrid_runs:
        return None
    results = [reconstruct_architecture_specific_result(r) for r in hybrid_runs]
    return hybrid_agent_evaluation.summarize_results(results)


# ---------------------------------------------------------------------------
# Ground truth — research/evaluation context only (Section 17), never fed
# back into an agent.
# ---------------------------------------------------------------------------


@st.cache_data(show_spinner=False)
def ground_truth_by_scenario() -> dict[str, GroundTruth]:
    return {gt.scenario_id: gt for gt in generate_all_ground_truth()}


@st.cache_data(show_spinner=False)
def quant_ground_truth_by_scenario() -> dict[str, QuantGroundTruth]:
    return {qgt.scenario_id: qgt for qgt in generate_all_quant_ground_truth()}
