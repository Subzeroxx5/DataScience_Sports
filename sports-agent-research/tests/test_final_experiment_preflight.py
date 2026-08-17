"""Tests for src/experiments/preflight.py (Milestone 14A, Section 6).

Exercises each individual check function directly against the real,
already-present controlled artifacts (no real LLM calls — these are
read-only smoke checks against local data/index/provider state). The
slow, pytest-subprocess-invoking check (_check_pytest_zero_failures) is
deliberately NOT exercised here to avoid a recursive full-suite run
inside the test suite itself; it is covered by every other passing test
in this suite already being green.
"""

from src.experiments import preflight


def _fresh_report() -> preflight.PreflightReport:
    return preflight.PreflightReport()


def test_preflight_report_all_passed_true_when_empty():
    assert _fresh_report().all_passed is True


def test_preflight_report_all_passed_false_on_any_failure():
    report = _fresh_report()
    report.record("a", True)
    report.record("b", False, "why it failed")
    assert report.all_passed is False
    assert report.details["b"] == "why it failed"


def test_ground_truth_reproducible_check_passes_for_the_current_dataset():
    report = _fresh_report()
    preflight._check_ground_truth_reproducible(report)
    assert report.checks["ground_truth_reproducible"] is True


def test_quant_ground_truth_reproducible_check_passes_for_the_current_dataset():
    report = _fresh_report()
    preflight._check_quant_ground_truth_reproducible(report)
    assert report.checks["quant_ground_truth_reproducible"] is True


def test_rag_corpus_and_index_check_passes_with_matching_embedding_model():
    report = _fresh_report()
    preflight._check_rag_corpus_and_index(report, rag_top_k=5, embedding_model="sentence-transformers/all-MiniLM-L6-v2")
    assert report.checks["rag_corpus_exists"] is True
    assert report.checks["embedding_config_matches_experiment_config"] is True
    assert report.checks["rag_index_loads"] is True
    assert report.checks["retrieval_smoke_query_works"] is True


def test_rag_index_check_fails_on_embedding_model_mismatch():
    report = _fresh_report()
    preflight._check_rag_corpus_and_index(report, rag_top_k=5, embedding_model="not-the-real-model")
    assert report.checks["embedding_config_matches_experiment_config"] is False


def test_tools_check_passes_for_the_controlled_provider():
    report = _fresh_report()
    preflight._check_tools(report)
    assert report.checks["controlled_odds_provider_loads"] is True
    assert report.checks["sportsbook_tools_work"] is True


def test_hybrid_reconciliation_policy_check_confirms_current_tool_data_wins():
    report = _fresh_report()
    preflight._check_hybrid_reconciliation_policy(report)
    assert report.checks["hybrid_reconciliation_policy_current_tool_wins"] is True


def test_architecture_boundary_check_depends_on_pytest_result():
    report = _fresh_report()
    report.record("pytest_zero_failures", True)
    preflight._check_architecture_boundaries_and_ground_truth_isolation(report)
    assert report.checks["architecture_boundaries_and_ground_truth_isolation_covered_by_pytest"] is True

    report2 = _fresh_report()
    report2.record("pytest_zero_failures", False)
    preflight._check_architecture_boundaries_and_ground_truth_isolation(report2)
    assert report2.checks["architecture_boundaries_and_ground_truth_isolation_covered_by_pytest"] is False
