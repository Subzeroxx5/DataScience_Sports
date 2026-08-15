"""Tests for the unified evaluation framework's core metric formulas
(Milestone 11, src/evaluation/metrics.py): architecture enum reuse,
metric applicability, best-line/best-odds/EV/market-reference/
freshness/completeness/hallucination definitions. See
tests/test_evaluation_aggregation.py for the generic aggregate
functions, tests/test_evaluation_consistency.py for the consistency
metric, and tests/test_evaluation_parity.py for cross-architecture
shared-function-usage tests.
"""

import pytest

from src.evaluation import metrics
from src.models import ArchitectureType


# ---------------------------------------------------------------------------
# Architecture enum (Step 3)
# ---------------------------------------------------------------------------


def test_architecture_enum_is_reused_from_models_not_reinvented():
    # The unified framework must not define its own parallel identifier.
    assert metrics.EvaluationResult.model_fields["architecture"].annotation is ArchitectureType


def test_architecture_enum_values_are_canonical():
    assert ArchitectureType.RAG.value == "rag"
    assert ArchitectureType.TOOL.value == "tool"
    assert ArchitectureType.HYBRID.value == "hybrid"


# ---------------------------------------------------------------------------
# Metric applicability (Step 20)
# ---------------------------------------------------------------------------


def test_metric_applicability_matches_the_documented_table():
    rag = metrics.METRIC_APPLICABILITY[ArchitectureType.RAG]
    tool = metrics.METRIC_APPLICABILITY[ArchitectureType.TOOL]
    hybrid = metrics.METRIC_APPLICABILITY[ArchitectureType.HYBRID]

    for applicability in (rag, tool, hybrid):
        assert applicability.best_line is True
        assert applicability.best_odds is True
        assert applicability.ev_classification is True
        assert applicability.freshness is True
        assert applicability.completeness is True
        assert applicability.hallucination is True
        assert applicability.latency is True

    assert rag.retrieval_recall is True
    assert rag.tool_calls is False
    assert rag.conflict_resolution is False

    assert tool.retrieval_recall is False
    assert tool.tool_calls is True
    assert tool.conflict_resolution is False

    assert hybrid.retrieval_recall is True
    assert hybrid.tool_calls is True
    assert hybrid.conflict_resolution is True


# ---------------------------------------------------------------------------
# Best-line accuracy (Step 5) — tie semantics
# ---------------------------------------------------------------------------


def test_best_line_correct_exact_match():
    assert metrics.best_line_correct(["DraftKings", "FanDuel"], ["DraftKings", "FanDuel"]) is True


def test_best_line_correct_ignores_order():
    assert metrics.best_line_correct(["FanDuel", "DraftKings"], ["DraftKings", "FanDuel"]) is True


def test_best_line_partial_tie_subset_is_not_correct():
    # A single-sportsbook prediction does not get full credit for a
    # multi-sportsbook tie (Step 5's explicit example).
    assert metrics.best_line_correct(["DraftKings"], ["DraftKings", "FanDuel"]) is False


def test_best_line_extra_sportsbook_is_not_correct():
    assert metrics.best_line_correct(["DraftKings", "FanDuel", "BetMGM"], ["DraftKings", "FanDuel"]) is False


def test_best_line_correct_not_applicable_when_no_prediction():
    assert metrics.best_line_correct(None, ["DraftKings"]) is None


# ---------------------------------------------------------------------------
# Best-odds accuracy (Step 6) — exact integer equality
# ---------------------------------------------------------------------------


def test_best_odds_correct_exact_match():
    assert metrics.best_odds_correct(125, 125) is True


def test_best_odds_incorrect_no_tolerance():
    # Off by one is still wrong — no float/int tolerance for American odds.
    assert metrics.best_odds_correct(124, 125) is False


def test_best_odds_not_applicable_when_no_prediction():
    assert metrics.best_odds_correct(None, 125) is None


# ---------------------------------------------------------------------------
# EV classification (Step 7)
# ---------------------------------------------------------------------------


def test_ev_classification_correct():
    assert metrics.ev_classification_correct(True, True) is True
    assert metrics.ev_classification_correct(True, False) is False


def test_ev_classification_not_applicable_when_non_quant_evaluable():
    assert metrics.ev_classification_correct(True, None) is None
    assert metrics.ev_classification_correct(None, True) is None
    assert metrics.ev_classification_correct(None, None) is None


# ---------------------------------------------------------------------------
# EV / market-reference absolute error (Steps 8-9)
# ---------------------------------------------------------------------------


def test_ev_absolute_error_unrounded():
    assert metrics.ev_absolute_error(0.123456789, 0.1) == pytest.approx(0.023456789)


def test_ev_absolute_error_not_applicable_when_missing():
    assert metrics.ev_absolute_error(None, 0.1) is None
    assert metrics.ev_absolute_error(0.1, None) is None


def test_market_reference_absolute_error_same_shape_as_ev_error():
    assert metrics.market_reference_absolute_error(0.44, 0.43) == pytest.approx(0.01)
    assert metrics.market_reference_absolute_error(None, 0.43) is None


# ---------------------------------------------------------------------------
# Freshness (Step 10, 15, 28)
# ---------------------------------------------------------------------------


def test_freshness_current_prediction_correct():
    diagnostics = metrics.evaluate_freshness(predicted_value=135, expected_current_value=135, known_stale_value=120)
    assert diagnostics.applicable is True
    assert diagnostics.freshness_correct is True
    assert diagnostics.matched_known_stale_value is False
    assert diagnostics.stale_value_matched is None


def test_freshness_known_stale_prediction_incorrect_with_diagnostic():
    diagnostics = metrics.evaluate_freshness(predicted_value=120, expected_current_value=135, known_stale_value=120)
    assert diagnostics.freshness_correct is False
    assert diagnostics.matched_known_stale_value is True
    assert diagnostics.stale_value_matched == 120


def test_freshness_unknown_wrong_prediction_not_labeled_stale():
    # Wrong, but not a match for the known stale value — must not be
    # mislabeled as a stale-data failure without support (Step 15).
    diagnostics = metrics.evaluate_freshness(predicted_value=125, expected_current_value=135, known_stale_value=120)
    assert diagnostics.freshness_correct is False
    assert diagnostics.matched_known_stale_value is False
    assert diagnostics.stale_value_matched is None


def test_freshness_not_applicable_when_no_expected_current_value():
    diagnostics = metrics.evaluate_freshness(predicted_value=125, expected_current_value=None, known_stale_value=None)
    assert diagnostics.applicable is False
    assert diagnostics.freshness_correct is None


def test_freshness_no_prediction_is_a_failure_not_not_applicable():
    diagnostics = metrics.evaluate_freshness(predicted_value=None, expected_current_value=135, known_stale_value=120)
    assert diagnostics.applicable is True
    assert diagnostics.freshness_correct is False


# ---------------------------------------------------------------------------
# Completeness (Step 11)
# ---------------------------------------------------------------------------


def test_completeness_full_coverage():
    assert metrics.completeness({"DraftKings", "FanDuel"}, {"DraftKings", "FanDuel"}) == pytest.approx(1.0)


def test_completeness_partial_coverage():
    assert metrics.completeness({"DraftKings"}, {"DraftKings", "FanDuel"}) == pytest.approx(0.5)


def test_completeness_fabricated_entries_must_be_excluded_by_caller():
    # completeness() trusts its `acquired` input is already
    # provenance-checked; a caller passing a bogus/fabricated sportsbook
    # not in `available` gets no credit for it.
    assert metrics.completeness({"DraftKings", "FakeBook"}, {"DraftKings", "FanDuel"}) == pytest.approx(0.5)


def test_completeness_not_applicable_when_nothing_required():
    assert metrics.completeness(set(), set()) is None


# ---------------------------------------------------------------------------
# Unsupported-claim rate (Steps 12-13, 26)
# ---------------------------------------------------------------------------


def test_unsupported_claim_rate_hand_example():
    assert metrics.unsupported_claim_rate(1, 5) == pytest.approx(0.20)


def test_unsupported_claim_rate_zero_unsupported():
    assert metrics.unsupported_claim_rate(0, 5) == pytest.approx(0.0)


def test_unsupported_claim_rate_denominator_zero_is_not_zero_percent():
    assert metrics.unsupported_claim_rate(0, 0) is None


def test_quant_errors_are_not_hallucinations():
    # Step 12: a quantitative calculation error (wrong EV number) is a
    # distinct concern from a fabricated claim — never conflated. The
    # metric layer keeps these as entirely separate fields/functions:
    # ev_absolute_error() never feeds unsupported_claim_rate() and vice
    # versa.
    ev_error = metrics.ev_absolute_error(0.05, 0.01)
    assert ev_error == pytest.approx(0.04)
    # A large EV error alone says nothing about hallucination count.
    assert metrics.unsupported_claim_rate(0, 1) == pytest.approx(0.0)
