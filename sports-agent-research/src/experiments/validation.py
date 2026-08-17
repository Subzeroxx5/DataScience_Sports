"""Final-dataset completeness/integrity validation (Milestone 14A,
Sections 19-24, 27).

Pure, read-only analysis of an already-persisted raw_results.jsonl plus
the frozen config it was run from — never recalculates an agent output
or a metric, and never mutates the dataset it inspects. Schema
validation reuses src.experiments.runner.RawExperimentRun's own pydantic
validation (Milestone 12) rather than a parallel schema check.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from pathlib import Path

from src.evaluation import metrics
from src.experiments.config import ExperimentConfig
from src.experiments.runner import RawExperimentRun
from src.models import ArchitectureType

RunKey = tuple[str, str, int]  # (architecture, scenario_id, repetition)


@dataclass
class DatasetValidationReport:
    experiment_id: str
    mode: str
    model: str
    temperature: float | None
    scenario_count: int
    repetitions: int

    expected_runs: int
    recorded_runs: int
    architecture_run_counts: dict[str, int]

    successful_runs: int
    failed_runs: int

    duplicate_run_keys: list[RunKey]
    missing_run_keys: list[RunKey]

    scenario_coverage: dict[str, dict[str, int]]  # scenario_id -> architecture -> observed count
    scenario_coverage_pass: bool
    architecture_coverage_pass: bool

    schema_validation_pass: bool
    schema_validation_errors: list[str] = field(default_factory=list)

    ground_truth_isolation_pass: bool = True
    architecture_isolation_pass: bool = True

    pre_post_hash_matches: dict[str, bool] = field(default_factory=dict)

    @property
    def hashes_all_match(self) -> bool:
        return all(self.pre_post_hash_matches.values()) if self.pre_post_hash_matches else False

    @property
    def dataset_valid(self) -> bool:
        return (
            self.recorded_runs == self.expected_runs
            and not self.duplicate_run_keys
            and not self.missing_run_keys
            and self.scenario_coverage_pass
            and self.architecture_coverage_pass
            and self.schema_validation_pass
            and self.ground_truth_isolation_pass
            and self.architecture_isolation_pass
            and self.hashes_all_match
        )


def load_and_validate_raw_results(raw_results_path: Path) -> tuple[list[RawExperimentRun], list[str]]:
    """Parses every line via RawExperimentRun's own pydantic schema
    (Milestone 12) — a malformed line is a schema error, not a crash."""
    runs: list[RawExperimentRun] = []
    errors: list[str] = []
    if not raw_results_path.is_file():
        return runs, [f"{raw_results_path} does not exist"]
    for line_number, line in enumerate(raw_results_path.read_text().splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            runs.append(RawExperimentRun.model_validate_json(line))
        except Exception as exc:
            errors.append(f"line {line_number}: {exc!r}")
    return runs, errors


def find_duplicate_and_missing_keys(
    runs: list[RawExperimentRun], config: ExperimentConfig,
) -> tuple[list[RunKey], list[RunKey]]:
    seen: dict[RunKey, int] = {}
    for run in runs:
        key = run.key().as_tuple()
        seen[key] = seen.get(key, 0) + 1
    duplicates = sorted(key for key, count in seen.items() if count > 1)

    expected_keys = {
        (architecture.value, scenario_id, repetition)
        for architecture in config.architectures
        for scenario_id in config.scenario_ids
        for repetition in range(1, config.repetitions + 1)
    }
    missing = sorted(expected_keys - set(seen.keys()))
    return duplicates, missing


def build_scenario_coverage(
    runs: list[RawExperimentRun], config: ExperimentConfig,
) -> tuple[dict[str, dict[str, int]], bool, bool]:
    coverage: dict[str, dict[str, int]] = {
        scenario_id: {architecture.value: 0 for architecture in config.architectures}
        for scenario_id in config.scenario_ids
    }
    for run in runs:
        if run.scenario_id in coverage and run.architecture.value in coverage[run.scenario_id]:
            coverage[run.scenario_id][run.architecture.value] += 1

    scenario_coverage_pass = all(
        count == config.repetitions
        for by_architecture in coverage.values()
        for count in by_architecture.values()
    )

    architecture_totals = {architecture.value: 0 for architecture in config.architectures}
    for by_architecture in coverage.values():
        for architecture_value, count in by_architecture.items():
            architecture_totals[architecture_value] += count
    expected_per_architecture = len(config.scenario_ids) * config.repetitions
    architecture_coverage_pass = all(
        total == expected_per_architecture for total in architecture_totals.values()
    )

    return coverage, scenario_coverage_pass, architecture_coverage_pass


def ground_truth_isolation_ok() -> bool:
    """Structural check (Section 23): the AgentRequest(...) construction
    inside src/experiments/runner.py never references ground truth — the
    same check Milestone 12's own test suite already enforces
    (tests/test_experiment_runner.py), re-run here so the postflight
    audit does not merely assume it."""
    source = (Path(__file__).resolve().parent / "runner.py").read_text()
    tree = ast.parse(source)
    try:
        request_construction = next(
            node for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "AgentRequest"
        )
    except StopIteration:
        return False
    call_source = ast.get_source_segment(source, request_construction) or ""
    return "ground_truth" not in call_source.lower()


def architecture_isolation_ok() -> bool:
    """Structural check: RAG-only never imports src.tools/src.providers,
    tool-only never imports src.rag — the same boundary already enforced
    by tests/test_rag_agent.py / tests/test_tool_agent.py's AST-based
    tests, re-checked here rather than assumed."""
    project_root = Path(__file__).resolve().parent.parent.parent

    rag_source = (project_root / "src" / "agents" / "rag_agent.py").read_text()
    rag_tree = ast.parse(rag_source)
    rag_imports = {
        node.module for node in ast.walk(rag_tree) if isinstance(node, ast.ImportFrom) and node.module
    } | {alias.name for node in ast.walk(rag_tree) if isinstance(node, ast.Import) for alias in node.names}
    rag_ok = not any(m.startswith("src.tools") or m.startswith("src.providers") for m in rag_imports)

    tool_source = (project_root / "src" / "agents" / "tool_agent.py").read_text()
    tool_tree = ast.parse(tool_source)
    tool_imports = {
        node.module for node in ast.walk(tool_tree) if isinstance(node, ast.ImportFrom) and node.module
    } | {alias.name for node in ast.walk(tool_tree) if isinstance(node, ast.Import) for alias in node.names}
    tool_ok = not any(m.startswith("src.rag") for m in tool_imports)

    return rag_ok and tool_ok


def validate_final_dataset(
    config: ExperimentConfig,
    raw_results_path: Path,
    pre_fingerprints: dict[str, str] | None = None,
    post_fingerprints: dict[str, str] | None = None,
) -> DatasetValidationReport:
    runs, schema_errors = load_and_validate_raw_results(raw_results_path)
    duplicates, missing = find_duplicate_and_missing_keys(runs, config)
    coverage, scenario_pass, architecture_pass = build_scenario_coverage(runs, config)

    architecture_run_counts = {architecture.value: 0 for architecture in config.architectures}
    for run in runs:
        architecture_run_counts[run.architecture.value] = architecture_run_counts.get(run.architecture.value, 0) + 1

    successful = sum(1 for run in runs if run.common_result.execution_status in metrics.SUCCESSFUL_CATEGORIES)

    hash_matches: dict[str, bool] = {}
    if pre_fingerprints is not None and post_fingerprints is not None:
        hash_matches = {name: post_fingerprints.get(name) == value for name, value in pre_fingerprints.items()}

    return DatasetValidationReport(
        experiment_id=config.experiment_id,
        mode=config.execution_mode.value,
        model=config.model_name,
        temperature=config.temperature,
        scenario_count=len(config.scenario_ids),
        repetitions=config.repetitions,
        expected_runs=config.expected_run_count(),
        recorded_runs=len(runs),
        architecture_run_counts=architecture_run_counts,
        successful_runs=successful,
        failed_runs=len(runs) - successful,
        duplicate_run_keys=duplicates,
        missing_run_keys=missing,
        scenario_coverage=coverage,
        scenario_coverage_pass=scenario_pass,
        architecture_coverage_pass=architecture_pass,
        schema_validation_pass=not schema_errors,
        schema_validation_errors=schema_errors,
        ground_truth_isolation_pass=ground_truth_isolation_ok(),
        architecture_isolation_pass=architecture_isolation_ok(),
        pre_post_hash_matches=hash_matches,
    )
