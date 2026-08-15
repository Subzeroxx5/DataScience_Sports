"""Tests for src/calculations/market.py (Milestone 7A)."""

import ast
import math
from pathlib import Path

import pytest

from src.calculations.market import (
    MIN_CONSENSUS_BOOKS,
    MarketDispersion,
    calculate_absolute_distance_from_consensus,
    calculate_leave_one_out_consensus,
    calculate_market_consensus,
    calculate_market_dispersion,
    calculate_no_vig_probabilities,
    calculate_overround,
    calculate_probability_edge,
    calculate_signed_distance_from_consensus,
    remove_vig_from_probabilities,
)
from src.calculations.odds_math import expected_value, implied_probability


# ---------------------------------------------------------------------------
# Overround
# ---------------------------------------------------------------------------


def test_overround_positive_negative_two_sided_market():
    overround = calculate_overround([120, -140])
    assert overround == pytest.approx(1.0378787878787878)


def test_overround_even_money_market():
    # Two even-money sides (-110/-110 is the classic "standard vig" case).
    overround = calculate_overround([-110, -110])
    assert overround == pytest.approx(implied_probability(-110) * 2)
    assert overround > 1.0


def test_overround_fair_market_sums_to_one():
    # +100/-100 are both exactly 50% implied, a theoretically fair market.
    overround = calculate_overround([100, -100])
    assert overround == pytest.approx(1.0)


def test_overround_rejects_invalid_odds():
    with pytest.raises(ValueError):
        calculate_overround([0, -140])


def test_overround_rejects_empty_input():
    with pytest.raises(ValueError):
        calculate_overround([])


# ---------------------------------------------------------------------------
# No-vig / remove vig
# ---------------------------------------------------------------------------


def test_no_vig_probabilities_known_example():
    fair = calculate_no_vig_probabilities([120, -140])
    assert fair[0] == pytest.approx(0.4380, abs=1e-4)
    assert fair[1] == pytest.approx(0.5620, abs=1e-4)


def test_no_vig_probabilities_sum_to_one():
    fair = calculate_no_vig_probabilities([120, -140])
    assert sum(fair) == pytest.approx(1.0)


def test_no_vig_probabilities_three_outcome_market():
    fair = calculate_no_vig_probabilities([150, 200, -400])
    assert sum(fair) == pytest.approx(1.0)
    assert len(fair) == 3


def test_no_vig_probabilities_rejects_empty_odds():
    with pytest.raises(ValueError):
        calculate_no_vig_probabilities([])


def test_remove_vig_from_probabilities_matches_odds_based_wrapper():
    raw = [implied_probability(120), implied_probability(-140)]
    direct = remove_vig_from_probabilities(raw)
    via_odds = calculate_no_vig_probabilities([120, -140])
    assert direct == pytest.approx(via_odds)


def test_remove_vig_rejects_malformed_probabilities():
    with pytest.raises(ValueError):
        remove_vig_from_probabilities([-0.1, 0.5])
    with pytest.raises(ValueError):
        remove_vig_from_probabilities([1.5, 0.5])
    with pytest.raises(ValueError):
        remove_vig_from_probabilities([float("nan"), 0.5])
    with pytest.raises(ValueError):
        remove_vig_from_probabilities([float("inf"), 0.5])


def test_remove_vig_rejects_empty_input():
    with pytest.raises(ValueError):
        remove_vig_from_probabilities([])


# ---------------------------------------------------------------------------
# Market consensus
# ---------------------------------------------------------------------------


def test_market_consensus_exact_arithmetic_mean():
    consensus = calculate_market_consensus([0.438, 0.442, 0.435, 0.440])
    assert consensus == pytest.approx(0.43875)


def test_market_consensus_rejects_invalid_probability():
    with pytest.raises(ValueError):
        calculate_market_consensus([0.438, 1.5, 0.435])


def test_market_consensus_rejects_empty_input():
    with pytest.raises(ValueError):
        calculate_market_consensus([])


def test_market_consensus_rejects_nan():
    with pytest.raises(ValueError):
        calculate_market_consensus([0.4, float("nan")])


# ---------------------------------------------------------------------------
# Leave-one-sportsbook-out consensus
# ---------------------------------------------------------------------------


SPORTSBOOK_PROBS = [
    ("DraftKings", 0.438),
    ("FanDuel", 0.442),
    ("BetMGM", 0.435),
    ("Caesars", 0.440),
]


def test_leave_one_out_target_excluded_from_calculation():
    consensus = calculate_leave_one_out_consensus(SPORTSBOOK_PROBS, "FanDuel")
    # FanDuel's own 0.442 must not appear in the averaged set.
    expected = (0.438 + 0.435 + 0.440) / 3
    assert consensus == pytest.approx(expected)


def test_leave_one_out_correct_result_known_example():
    consensus = calculate_leave_one_out_consensus(SPORTSBOOK_PROBS, "FanDuel")
    assert consensus == pytest.approx(0.4376666667, abs=1e-6)


def test_leave_one_out_unknown_target_rejected():
    with pytest.raises(ValueError):
        calculate_leave_one_out_consensus(SPORTSBOOK_PROBS, "PointsBet")


def test_leave_one_out_insufficient_remaining_books_rejected():
    two_books = [("DraftKings", 0.438), ("FanDuel", 0.442)]
    with pytest.raises(ValueError):
        calculate_leave_one_out_consensus(two_books, "FanDuel")


def test_leave_one_out_exactly_minimum_remaining_books_succeeds():
    three_books = [("DraftKings", 0.438), ("FanDuel", 0.442), ("BetMGM", 0.435)]
    consensus = calculate_leave_one_out_consensus(three_books, "FanDuel")
    assert consensus == pytest.approx((0.438 + 0.435) / 2)
    assert MIN_CONSENSUS_BOOKS == 2


def test_leave_one_out_rejects_duplicate_sportsbook_identifiers():
    duplicated = [("DraftKings", 0.438), ("DraftKings", 0.440), ("BetMGM", 0.435), ("Caesars", 0.440)]
    with pytest.raises(ValueError):
        calculate_leave_one_out_consensus(duplicated, "DraftKings")


def test_leave_one_out_rejects_blank_sportsbook_identifier():
    malformed = [("", 0.438), ("FanDuel", 0.442), ("BetMGM", 0.435), ("Caesars", 0.440)]
    with pytest.raises(ValueError):
        calculate_leave_one_out_consensus(malformed, "FanDuel")


def test_leave_one_out_rejects_empty_input():
    with pytest.raises(ValueError):
        calculate_leave_one_out_consensus([], "FanDuel")


def test_leave_one_out_rejects_invalid_probability_among_remaining():
    invalid = [("DraftKings", 0.438), ("FanDuel", 0.442), ("BetMGM", 1.5), ("Caesars", 0.440)]
    with pytest.raises(ValueError):
        calculate_leave_one_out_consensus(invalid, "FanDuel")


# ---------------------------------------------------------------------------
# Probability edge
# ---------------------------------------------------------------------------


def test_probability_edge_positive():
    edge = calculate_probability_edge(0.46, 0.4348)
    assert edge == pytest.approx(0.0252)


def test_probability_edge_negative():
    edge = calculate_probability_edge(0.40, 0.45)
    assert edge == pytest.approx(-0.05)


def test_probability_edge_zero():
    edge = calculate_probability_edge(0.5, 0.5)
    assert edge == pytest.approx(0.0)


def test_probability_edge_rejects_out_of_range_probability():
    with pytest.raises(ValueError):
        calculate_probability_edge(1.2, 0.4)
    with pytest.raises(ValueError):
        calculate_probability_edge(0.4, -0.1)


def test_probability_edge_returns_raw_decimal_not_percentage():
    edge = calculate_probability_edge(0.46, 0.4348)
    assert edge < 1.0  # not pre-multiplied by 100


# ---------------------------------------------------------------------------
# EV integration — reuses the existing EV function, no new arithmetic
# ---------------------------------------------------------------------------


def test_ev_integration_market_reference_probability_130_at_46_percent():
    ev = expected_value(130, 0.46)
    assert ev == pytest.approx(0.058, abs=1e-9)


def test_ev_integration_uses_leave_one_out_consensus_as_reference():
    # The intended later flow: leave-one-out consensus -> expected_value().
    consensus = calculate_leave_one_out_consensus(SPORTSBOOK_PROBS, "FanDuel")
    ev = expected_value(130, consensus)
    assert ev == pytest.approx(expected_value(130, 0.4376666667), abs=1e-6)


def test_market_module_does_not_reimplement_ev():
    # market.py must not define its own expected_value function.
    import src.calculations.market as market_module

    assert not hasattr(market_module, "calculate_expected_value")
    assert not hasattr(market_module, "expected_value")


# ---------------------------------------------------------------------------
# Market dispersion
# ---------------------------------------------------------------------------


def test_dispersion_mean_median_stddev_range():
    dispersion = calculate_market_dispersion([0.438, 0.442, 0.435, 0.440])
    assert isinstance(dispersion, MarketDispersion)
    assert dispersion.mean_probability == pytest.approx(0.43875)
    assert dispersion.median_probability == pytest.approx(0.439)
    assert dispersion.std_dev == pytest.approx(0.0025860201081971523)
    assert dispersion.probability_range == pytest.approx(0.007, abs=1e-9)
    assert dispersion.book_count == 4


def test_dispersion_single_value_is_explicitly_defined():
    dispersion = calculate_market_dispersion([0.5])
    assert dispersion.std_dev == 0.0
    assert dispersion.probability_range == 0.0
    assert dispersion.mean_probability == 0.5
    assert dispersion.median_probability == 0.5
    assert dispersion.book_count == 1


def test_dispersion_rejects_empty_input():
    with pytest.raises(ValueError):
        calculate_market_dispersion([])


def test_dispersion_rejects_invalid_probability():
    with pytest.raises(ValueError):
        calculate_market_dispersion([0.4, 2.0])


# ---------------------------------------------------------------------------
# Distance from consensus
# ---------------------------------------------------------------------------


def test_signed_distance_from_consensus_positive():
    distance = calculate_signed_distance_from_consensus(0.45, 0.40)
    assert distance == pytest.approx(0.05)


def test_signed_distance_from_consensus_negative():
    distance = calculate_signed_distance_from_consensus(0.35, 0.40)
    assert distance == pytest.approx(-0.05)


def test_absolute_distance_from_consensus_is_always_nonnegative():
    assert calculate_absolute_distance_from_consensus(0.35, 0.40) == pytest.approx(0.05)
    assert calculate_absolute_distance_from_consensus(0.45, 0.40) == pytest.approx(0.05)


def test_distance_functions_are_unambiguously_named_differently():
    signed = calculate_signed_distance_from_consensus(0.35, 0.40)
    absolute = calculate_absolute_distance_from_consensus(0.35, 0.40)
    assert signed == -absolute


# ---------------------------------------------------------------------------
# Input validation — NaN / infinity / out-of-range
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad_value", [-0.1, 1.1, float("nan"), float("inf"), float("-inf")])
def test_probability_edge_rejects_all_bad_values(bad_value):
    with pytest.raises(ValueError):
        calculate_probability_edge(bad_value, 0.4)
    with pytest.raises(ValueError):
        calculate_probability_edge(0.4, bad_value)


# ---------------------------------------------------------------------------
# Architecture boundary
# ---------------------------------------------------------------------------


def test_market_module_has_no_forbidden_imports():
    source_path = Path(__file__).resolve().parent.parent / "src" / "calculations" / "market.py"
    tree = ast.parse(source_path.read_text())

    forbidden_prefixes = (
        "src.rag",
        "src.agents",
        "src.providers",
        "src.tools",
        "src.evaluation",
        "anthropic",
        "openai",
        "requests",
        "httpx",
        "sentence_transformers",
        "faiss",
    )
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert not alias.name.startswith(forbidden_prefixes)
        elif isinstance(node, ast.ImportFrom) and node.module:
            assert not node.module.startswith(forbidden_prefixes)


def test_market_module_only_imports_odds_math_and_stdlib():
    source_path = Path(__file__).resolve().parent.parent / "src" / "calculations" / "market.py"
    tree = ast.parse(source_path.read_text())

    allowed = {"__future__", "dataclasses", "statistics", "typing", "src.calculations.odds_math"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert alias.name in allowed, f"unexpected import: {alias.name}"
        elif isinstance(node, ast.ImportFrom) and node.module:
            assert node.module in allowed, f"unexpected import: {node.module}"
