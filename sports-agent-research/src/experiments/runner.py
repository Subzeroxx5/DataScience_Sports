"""Controlled experiment runner (Milestone 12).

    ExperimentConfig
          |
          v
       Scenario
          |
          v
      Repetition
          |
          v
     Architecture            (rotated per Step 9)
          |
          v
     AgentRequest             (identical across architectures for a
          |                    given scenario — Step 6)
          v
        Agent                 (src/experiments/agent_factory.py)
          |
          v
    BettingAnalysis
          |
          v
   Unified Evaluator          (src/evaluation/*_agent_evaluation.py,
          |                    Milestone 9B-11 — reused verbatim, never
          |                    a runner-local accuracy formula)
          v
   ArchitectureEvaluationResult (metrics.EvaluationResult, Milestone 11)
          |
          v
    Persistent Raw Result    (results/experiments/<id>/raw_results.jsonl)

Ground-truth isolation (Step 20): GroundTruth/QuantGroundTruth are
loaded once here and passed ONLY into each evaluator's `evaluate_scenario`
call — never into the AgentRequest built for the agent itself. See
tests/test_experiment_runner.py's dedicated isolation test.

Failure isolation (Step 13): each evaluator's own `evaluate_scenario`
already catches every agent-level failure and returns a typed result
(never raises) — see Milestones 9B/10B/11. This module adds one more
defensive layer around agent *construction* itself, so a single broken
run is recorded with an explicit failure category and the batch
continues.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel

from src.agents.base import AgentRequest
from src.evaluation import hybrid_agent_evaluation as hybrid_eval
from src.evaluation import metrics
from src.evaluation import rag_agent_evaluation as rag_eval
from src.evaluation import tool_agent_evaluation as tool_eval
from src.evaluation.dataset import load_scenario_definitions_by_id
from src.evaluation.ground_truth import generate_all_ground_truth
from src.evaluation.quant_ground_truth import generate_all_quant_ground_truth
from src.evaluation.tool_agent_evaluation import DEFAULT_SCENARIO_IDS, _stale_odds_by_scenario_key
from src.experiments.agent_factory import build_llm_client, create_agent
from src.experiments.config import (
    DEFAULT_ARCHITECTURE_ORDER,
    ExecutionMode,
    ExperimentConfig,
    ExperimentMetadata,
    ExperimentScenario,
    build_scenario_manifest,
    execution_order_for_repetition,
)
from src.models import ArchitectureType, GroundTruth, QuantGroundTruth

INFRASTRUCTURE_VALIDATION_LABEL = "INFRASTRUCTURE VALIDATION ONLY"
NOT_FINAL_RESULTS_LABEL = "NOT FINAL RESEARCH RESULTS"

_EVALUATE_SCENARIO = {
    ArchitectureType.RAG: rag_eval.evaluate_scenario,
    ArchitectureType.TOOL: tool_eval.evaluate_scenario,
    ArchitectureType.HYBRID: hybrid_eval.evaluate_scenario,
}
_TO_COMMON_RESULT = {
    ArchitectureType.RAG: rag_eval.to_common_result,
    ArchitectureType.TOOL: tool_eval.to_common_result,
    ArchitectureType.HYBRID: hybrid_eval.to_common_result,
}


class ExperimentRunKey(BaseModel):
    """Step 4: uniquely identifies one raw run — never aggregated before
    being saved."""

    architecture: ArchitectureType
    scenario_id: str
    repetition: int

    def as_tuple(self) -> tuple[str, str, int]:
        return (self.architecture.value, self.scenario_id, self.repetition)


class RawExperimentRun(BaseModel):
    """One persisted raw run (Step 14/16/17/18). `common_result` is the
    Milestone 11 unified shape (for cross-architecture aggregation);
    `architecture_specific_result` is the FULL original per-architecture
    evaluator result (RagAgentEvaluationResult / ToolAgentEvaluationResult
    / HybridAgentEvaluationResult, as a JSON-safe dict) — architecture-
    specific traces (hybrid conflict resolutions, tool redundant-call
    counts, RAG retrieval depth, ...) are never discarded just because
    the common evaluator doesn't aggregate all of them (Step 18)."""

    experiment_id: str
    architecture: ArchitectureType
    scenario_id: str
    repetition: int
    execution_order_position: int
    timestamp: datetime

    common_result: metrics.EvaluationResult
    architecture_specific_result: dict

    def key(self) -> ExperimentRunKey:
        return ExperimentRunKey(architecture=self.architecture, scenario_id=self.scenario_id, repetition=self.repetition)


class ExperimentSummary(BaseModel):
    """Persisted as summary.json. Deliberately holds only data — no
    declared "winner" anywhere (Step 21/23)."""

    experiment_id: str
    expected_runs: int
    recorded_runs: int
    successful_runs: int
    failed_runs: int
    comparison: metrics.ArchitectureComparison
    labels: list[str] = [INFRASTRUCTURE_VALIDATION_LABEL]


def _experiment_dir(config: ExperimentConfig) -> Path:
    return Path(config.output_dir) / config.experiment_id


def _load_existing_run_keys(raw_results_path: Path) -> set[tuple[str, str, int]]:
    if not raw_results_path.is_file():
        return set()
    keys: set[tuple[str, str, int]] = set()
    with raw_results_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = RawExperimentRun.model_validate_json(line)
            keys.add(record.key().as_tuple())
    return keys


def _build_failure_run(
    experiment_id: str,
    architecture: ArchitectureType,
    scenario: ExperimentScenario,
    repetition: int,
    execution_order_position: int,
    ground_truth: GroundTruth,
    exc: Exception,
) -> RawExperimentRun:
    """Step 13: agent-construction (or other unexpected runner-level)
    failure — record it explicitly rather than crashing the batch."""
    common_result = metrics.EvaluationResult(
        scenario_id=scenario.scenario_id,
        architecture=architecture,
        execution_status=metrics.FailureCategory.UNKNOWN_FAILURE,
        quant_evaluable=bool(scenario.quant_evaluable),
        predicted_best_sportsbooks=[],
        expected_best_sportsbooks=ground_truth.expected_best_sportsbooks,
        best_line_correct=None,
        predicted_best_odds=None,
        expected_best_odds=ground_truth.expected_best_odds,
        best_odds_correct=None,
        predicted_positive_ev=None,
        expected_positive_ev=None,
        ev_classification_correct=None,
        predicted_ev=None,
        expected_ev=None,
        ev_absolute_error=None,
        predicted_market_reference_probability=None,
        expected_market_reference_probability=None,
        market_reference_absolute_error=None,
        freshness_correct=None,
        completeness=None,
        unsupported_claim_count=0,
        total_verifiable_claims=0,
        hallucination_detected=False,
        retrieval_metrics=None,
        tool_metrics=None,
        latency_metrics=metrics.LatencyMetrics(total_latency_seconds=0.0),
        errors=[repr(exc)],
    )
    return RawExperimentRun(
        experiment_id=experiment_id,
        architecture=architecture,
        scenario_id=scenario.scenario_id,
        repetition=repetition,
        execution_order_position=execution_order_position,
        timestamp=datetime.now(timezone.utc),
        common_result=common_result,
        architecture_specific_result={"error": repr(exc), "note": "agent construction or runner-level failure"},
    )


def _run_one(
    experiment_id: str,
    architecture: ArchitectureType,
    scenario: ExperimentScenario,
    request: AgentRequest,
    repetition: int,
    execution_order_position: int,
    config: ExperimentConfig,
    ground_truth: GroundTruth,
    quant_ground_truth: QuantGroundTruth,
    stale_odds_map: dict,
) -> RawExperimentRun:
    try:
        agent, aux_handle = create_agent(architecture, config)
        architecture_result = _EVALUATE_SCENARIO[architecture](
            agent, aux_handle, request, ground_truth, quant_ground_truth, stale_odds_map
        )
        common_result = _TO_COMMON_RESULT[architecture](architecture_result)
    except Exception as exc:  # defensive: agent construction or an unexpected runner bug
        return _build_failure_run(
            experiment_id, architecture, scenario, repetition, execution_order_position, ground_truth, exc
        )

    return RawExperimentRun(
        experiment_id=experiment_id,
        architecture=architecture,
        scenario_id=scenario.scenario_id,
        repetition=repetition,
        execution_order_position=execution_order_position,
        timestamp=datetime.now(timezone.utc),
        common_result=common_result,
        architecture_specific_result=architecture_result.model_dump(mode="json"),
    )


def run_experiment(config: ExperimentConfig, resume: bool = False) -> ExperimentSummary:
    experiment_dir = _experiment_dir(config)
    raw_results_path = experiment_dir / "raw_results.jsonl"

    if raw_results_path.is_file() and not resume:
        raise FileExistsError(
            f"{raw_results_path} already has recorded runs — pass resume=True to continue "
            "this experiment, or use a different experiment_id/output_dir."
        )

    experiment_dir.mkdir(parents=True, exist_ok=True)

    manifest = build_scenario_manifest(config.scenario_ids)
    (experiment_dir / "manifest.json").write_text(
        json.dumps([s.model_dump(mode="json") for s in manifest], indent=2, sort_keys=True)
    )
    (experiment_dir / "config.json").write_text(ExperimentMetadata.build(config).model_dump_json(indent=2))

    ground_truth_by_id = {gt.scenario_id: gt for gt in generate_all_ground_truth()}
    quant_ground_truth_by_id = {qgt.scenario_id: qgt for qgt in generate_all_quant_ground_truth()}
    scenario_definitions_by_id = load_scenario_definitions_by_id()
    stale_odds_map = _stale_odds_by_scenario_key()

    seen_keys = _load_existing_run_keys(raw_results_path)
    all_runs: list[RawExperimentRun] = []
    if seen_keys:
        with raw_results_path.open() as f:
            for line in f:
                line = line.strip()
                if line:
                    all_runs.append(RawExperimentRun.model_validate_json(line))

    with raw_results_path.open("a") as raw_file:
        for scenario in manifest:
            definition = scenario_definitions_by_id[scenario.scenario_id]
            # Ground truth (Step 20) is never read here — only game/market
            # identity + the canonical query already computed in the
            # manifest — so the SAME AgentRequest reaches every
            # architecture for this scenario (Step 6).
            request = AgentRequest(
                scenario_id=scenario.scenario_id,
                game_id=definition["game"]["game_id"],
                market_type=scenario.market_type,
                selected_outcome=scenario.selected_outcome,
                query=scenario.query,
            )
            ground_truth = ground_truth_by_id[scenario.scenario_id]
            quant_ground_truth = quant_ground_truth_by_id[scenario.scenario_id]

            for repetition in range(1, config.repetitions + 1):
                order = execution_order_for_repetition(config.architectures, repetition - 1)
                for position, architecture in enumerate(order):
                    key = ExperimentRunKey(
                        architecture=architecture, scenario_id=scenario.scenario_id, repetition=repetition
                    )
                    if key.as_tuple() in seen_keys:
                        continue  # Step 25: skip already-recorded runs when resuming

                    run = _run_one(
                        config.experiment_id, architecture, scenario, request, repetition, position,
                        config, ground_truth, quant_ground_truth, stale_odds_map,
                    )
                    seen_keys.add(key.as_tuple())
                    all_runs.append(run)
                    raw_file.write(run.model_dump_json())
                    raw_file.write("\n")
                    raw_file.flush()

    summary = summarize_experiment(config, all_runs)
    (experiment_dir / "summary.json").write_text(summary.model_dump_json(indent=2))
    return summary


def summarize_experiment(config: ExperimentConfig, runs: list[RawExperimentRun]) -> ExperimentSummary:
    """Step 21-24: aggregate via the Milestone 11 unified evaluator only
    — no runner-local accuracy formula. `metrics.summarize()` groups by
    scenario_id internally, so passing every repetition's common_result
    for one architecture yields real Step 22 consistency values."""
    by_architecture: dict[ArchitectureType, list[metrics.EvaluationResult]] = {}
    for run in runs:
        by_architecture.setdefault(run.architecture, []).append(run.common_result)

    comparison = metrics.compare_architectures(
        rag_results=by_architecture.get(ArchitectureType.RAG),
        tool_results=by_architecture.get(ArchitectureType.TOOL),
        hybrid_results=by_architecture.get(ArchitectureType.HYBRID),
    )

    successful = sum(
        1 for run in runs if run.common_result.execution_status in metrics.SUCCESSFUL_CATEGORIES
    )

    return ExperimentSummary(
        experiment_id=config.experiment_id,
        expected_runs=config.expected_run_count(),
        recorded_runs=len(runs),
        successful_runs=successful,
        failed_runs=len(runs) - successful,
        comparison=comparison,
    )


# ---------------------------------------------------------------------------
# CLI (Steps 29-30)
# ---------------------------------------------------------------------------


def _load_config_from_file(path: str) -> ExperimentConfig:
    """Milestone 14A, Section 2/14: load a frozen ExperimentConfig from a
    machine-readable JSON file (e.g. experiments/final_experiment.json)
    instead of assembling one from ad hoc CLI flags — the authoritative
    configuration for a final experiment is the file, not whatever flags
    happened to be typed on the command line. Accepts either a bare
    ExperimentConfig JSON document or a wrapper object with a top-level
    "config" key (the documented final_experiment.json shape, which also
    carries non-config notes/rationale alongside it)."""
    payload = json.loads(Path(path).read_text())
    config_payload = payload["config"] if "config" in payload else payload
    return ExperimentConfig.model_validate(config_payload)


def _build_config_from_args(args: argparse.Namespace) -> ExperimentConfig:
    scenario_ids = args.scenario_ids.split(",") if args.scenario_ids else list(DEFAULT_SCENARIO_IDS)
    architectures = (
        [ArchitectureType(a) for a in args.architectures.split(",")]
        if args.architectures
        else list(DEFAULT_ARCHITECTURE_ORDER)
    )
    experiment_id = args.experiment_id or f"{args.experiment_name}-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"
    return ExperimentConfig(
        experiment_id=experiment_id,
        experiment_name=args.experiment_name,
        architectures=architectures,
        scenario_ids=scenario_ids,
        repetitions=args.repetitions,
        execution_mode=ExecutionMode(args.mode),
        output_dir=args.output_dir,
    )


def _print_dry_run(config: ExperimentConfig) -> None:
    manifest = build_scenario_manifest(config.scenario_ids)
    print(f"Architectures: {[a.value for a in config.architectures]}")
    print(f"Scenarios: {len(manifest)} ({config.scenario_ids})")
    print(f"Repetitions: {config.repetitions}")
    print(f"Expected runs: {config.expected_run_count()}")
    print(f"Provider: {config.llm_provider.value}")
    print(f"Model: {config.model_name}")
    if config.llm_provider.value == "ollama":
        print(f"Temperature: {config.temperature}")
    else:
        print(f"Temperature: {config.temperature} (N/A for this model family — see `effort`)")
    print(f"Effort: {config.effort}")
    print(f"RAG top_k: {config.rag_top_k}")
    print(f"Max tool iterations: {config.max_tool_iterations}")
    print(f"Execution-order policy: {config.execution_order_policy}")
    print(f"Output path: {_experiment_dir(config)}")


def _print_summary(summary: ExperimentSummary) -> None:
    print(f"Expected raw runs: {summary.expected_runs}")
    print(f"Recorded raw runs: {summary.recorded_runs}")
    print(f"Successful: {summary.successful_runs}")
    print(f"Failed: {summary.failed_runs}")
    print()
    for label, arch_summary in (
        ("RAG", summary.comparison.rag_summary),
        ("TOOL", summary.comparison.tool_summary),
        ("HYBRID", summary.comparison.hybrid_summary),
    ):
        print(f"{label} summary:")
        if arch_summary is None:
            print("  (not run)")
            continue
        print(f"  runs={arch_summary.runs} successes={arch_summary.successes} failures={arch_summary.failures}")
        print(f"  success_rate={arch_summary.success_rate}")
        print(f"  best_line_accuracy={arch_summary.best_line_accuracy}")
        print(f"  best_odds_accuracy={arch_summary.best_odds_accuracy}")
        print(f"  ev_classification_accuracy={arch_summary.ev_classification_accuracy}")
        print(f"  mean_ev_absolute_error={arch_summary.mean_ev_absolute_error}")
        print(f"  market_reference_mae={arch_summary.market_reference_mae}")
        print(f"  freshness_accuracy={arch_summary.freshness_accuracy}")
        print(f"  mean_completeness={arch_summary.mean_completeness}")
        print(f"  unsupported_claim_rate={arch_summary.unsupported_claim_rate}")
        print(f"  consistency={arch_summary.consistency}")
        print(f"  mean_total_latency_seconds={arch_summary.mean_total_latency_seconds}")
        print()
    for label in summary.labels:
        print(label)
    print(NOT_FINAL_RESULTS_LABEL)


def probe_real_llm_connectivity(config: ExperimentConfig) -> tuple[bool, str | None]:
    """Cheap, single-turn connectivity check before spending a whole
    experiment's worth of real inference calls (used by main() below and,
    Milestone 14A, by experiments/run_final_experiment.py's preflight).

    Builds the probe client via build_llm_client() — the same
    configuration-driven provider selection every real run uses (Pre-
    Milestone 14A checkpoint) — so the probe actually exercises whichever
    provider config.llm_provider names (Anthropic or Ollama), never
    hard-coding one provider regardless of configuration. Returns (ok,
    error_repr) — never raises."""
    try:
        probe_client = build_llm_client(ArchitectureType.TOOL, config)
        probe_client.create_turn(
            system_prompt="Reply with the single word: ready.",
            messages=[{"role": "user", "content": "ready?"}],
            tools=[],
        )
        return True, None
    except Exception as exc:
        return False, repr(exc)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Controlled experiment runner (Milestone 12)")
    parser.add_argument("--mode", choices=["mock", "real"], default="mock")
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--output-dir", default="results/experiments")
    parser.add_argument("--scenario-ids", default=None, help="comma-separated scenario IDs")
    parser.add_argument("--architectures", default=None, help="comma-separated: rag,tool,hybrid")
    parser.add_argument("--experiment-name", default="experiment")
    parser.add_argument("--experiment-id", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--config", default=None,
        help="path to a frozen ExperimentConfig JSON file (Milestone 14A); "
             "overrides --scenario-ids/--architectures/--repetitions/--mode/etc.",
    )
    args = parser.parse_args(argv)

    config = _load_config_from_file(args.config) if args.config else _build_config_from_args(args)

    if args.dry_run:
        _print_dry_run(config)
        return

    if config.execution_mode == ExecutionMode.REAL:
        ok, error = probe_real_llm_connectivity(config)
        if not ok:
            print("REAL EXPERIMENT: NOT RUN")
            print(f"Could not reach the real Anthropic API — is ANTHROPIC_API_KEY set? ({error})")
            return

    summary = run_experiment(config, resume=args.resume)
    _print_summary(summary)


if __name__ == "__main__":
    main(sys.argv[1:])
