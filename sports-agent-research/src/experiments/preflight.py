"""Final-experiment preflight validation (Milestone 14A, Section 6).

Every check here either (a) shells out to the project's own pytest
suite — the authoritative correctness check, never reimplemented ad hoc
— or (b) exercises the actual, already-existing pipeline functions once
each in a read-only way (regenerate ground truth and diff against the
persisted file; load the real vector index and run one retrieval query;
load the real ControlledOddsProvider and list games; call the real
hybrid reconciliation function on a synthetic conflict). Nothing here
defines a new formula, a new failure taxonomy, or a parallel
implementation of anything src/ already does.
"""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

from src.evaluation.dataset import DATA_DIR

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent


@dataclass
class PreflightReport:
    checks: dict[str, bool] = field(default_factory=dict)
    details: dict[str, str] = field(default_factory=dict)

    @property
    def all_passed(self) -> bool:
        return all(self.checks.values())

    def record(self, name: str, passed: bool, detail: str = "") -> None:
        self.checks[name] = passed
        if detail:
            self.details[name] = detail


def _check_pytest_zero_failures(report: PreflightReport) -> None:
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q"], capture_output=True, text=True, cwd=_REPO_ROOT,
    )
    passed = result.returncode == 0
    report.record(
        "pytest_zero_failures", passed,
        result.stdout.strip().splitlines()[-1] if result.stdout.strip() else result.stderr[-500:],
    )


def _check_ground_truth_reproducible(report: PreflightReport) -> None:
    """Deterministic generator still reproduces the persisted file
    (Section 6, "Ground Truth"): regenerate in memory, diff against
    data/ground_truth.json — never overwrite it."""
    from src.evaluation.ground_truth import generate_all_ground_truth

    try:
        regenerated = [gt.model_dump(mode="json") for gt in generate_all_ground_truth()]
        persisted = json.loads((DATA_DIR / "ground_truth.json").read_text())
        passed = regenerated == persisted
        report.record("ground_truth_reproducible", passed, "" if passed else "regenerated output != data/ground_truth.json")
    except Exception as exc:
        report.record("ground_truth_reproducible", False, repr(exc))


def _check_quant_ground_truth_reproducible(report: PreflightReport) -> None:
    from src.evaluation.quant_ground_truth import generate_all_quant_ground_truth

    try:
        regenerated = [q.model_dump(mode="json") for q in generate_all_quant_ground_truth()]
        persisted = json.loads((DATA_DIR / "quant_ground_truth.json").read_text())
        passed = regenerated == persisted
        report.record(
            "quant_ground_truth_reproducible", passed,
            "" if passed else "regenerated output != data/quant_ground_truth.json",
        )
    except Exception as exc:
        report.record("quant_ground_truth_reproducible", False, repr(exc))


def _check_rag_corpus_and_index(report: PreflightReport, rag_top_k: int, embedding_model: str) -> None:
    corpus_path = DATA_DIR / "rag_documents" / "corpus.jsonl"
    report.record("rag_corpus_exists", corpus_path.is_file() and corpus_path.stat().st_size > 0)

    index_config_path = DATA_DIR / "rag_index" / "config.json"
    try:
        index_config = json.loads(index_config_path.read_text())
        embedding_match = index_config.get("embedding_model") == embedding_model
        report.record(
            "embedding_config_matches_experiment_config", embedding_match,
            f"index config embedding_model={index_config.get('embedding_model')!r} vs experiment config={embedding_model!r}",
        )
    except Exception as exc:
        report.record("embedding_config_matches_experiment_config", False, repr(exc))

    try:
        from src.rag.retriever import Retriever

        retriever = Retriever.from_directory()
        results = retriever.retrieve("moneyline price comparison", k=min(rag_top_k, 3))
        report.record("rag_index_loads", True)
        report.record("retrieval_smoke_query_works", len(results) > 0)
    except Exception as exc:
        report.record("rag_index_loads", False, repr(exc))
        report.record("retrieval_smoke_query_works", False, repr(exc))


def _check_tools(report: PreflightReport) -> None:
    try:
        from src.providers.controlled import ControlledOddsProvider
        from src.tools.sportsbook_tools import SportsbookTools

        tools = SportsbookTools(ControlledOddsProvider())
        games = tools.get_games()
        report.record("controlled_odds_provider_loads", True)
        report.record("sportsbook_tools_work", len(games) > 0)
    except Exception as exc:
        report.record("controlled_odds_provider_loads", False, repr(exc))
        report.record("sportsbook_tools_work", False, repr(exc))


def _check_hybrid_reconciliation_policy(report: PreflightReport) -> None:
    """Confirm the deterministic reconciliation function itself — not a
    reimplementation of it — still applies "current structured tool data
    wins" (Section 6, "Hybrid")."""
    try:
        from datetime import datetime, timezone

        from src.agents.hybrid_reconciliation import (
            ConflictResolutionReason,
            RagPriceObservation,
            reconcile_outcome,
        )
        from src.models import SourceType

        records = reconcile_outcome(
            selected_outcome="Los Angeles Lakers",
            tool_prices={"FanDuel": 130},
            rag_prices={
                "FanDuel": RagPriceObservation(
                    american_odds=120, is_current=True,
                    timestamp=datetime.now(timezone.utc), document_id="smoke-check",
                )
            },
        )
        record = records[0]
        passed = (
            record.authoritative_odds == 130
            and record.authoritative_source == SourceType.TOOL
            and record.conflict_resolution_reason == ConflictResolutionReason.CURRENT_TOOL_DATA_PRECEDENCE
        )
        report.record("hybrid_reconciliation_policy_current_tool_wins", passed)
    except Exception as exc:
        report.record("hybrid_reconciliation_policy_current_tool_wins", False, repr(exc))


def _check_architecture_boundaries_and_ground_truth_isolation(report: PreflightReport) -> None:
    """The AST-based architecture-isolation tests (tests/test_rag_agent.py,
    tests/test_tool_agent.py, tests/test_hybrid_agent.py) and the
    ground-truth-isolation tests (tests/test_experiment_runner.py) are
    already part of the pytest suite checked above — re-asserting their
    logic here would duplicate it. This records that dependency
    explicitly rather than silently assuming it."""
    report.record(
        "architecture_boundaries_and_ground_truth_isolation_covered_by_pytest",
        report.checks.get("pytest_zero_failures", False),
        "relies on pytest_zero_failures — see tests/test_rag_agent.py, "
        "tests/test_tool_agent.py, tests/test_hybrid_agent.py, tests/test_experiment_runner.py",
    )


def run_preflight_checks(config) -> PreflightReport:
    """Runs every Section 6 preflight check and returns a report whose
    `all_passed` must be True before any real LLM call is made."""
    report = PreflightReport()
    _check_pytest_zero_failures(report)
    _check_ground_truth_reproducible(report)
    _check_quant_ground_truth_reproducible(report)
    _check_rag_corpus_and_index(report, config.rag_top_k, config.embedding_model)
    _check_tools(report)
    _check_hybrid_reconciliation_policy(report)
    _check_architecture_boundaries_and_ground_truth_isolation(report)
    return report
