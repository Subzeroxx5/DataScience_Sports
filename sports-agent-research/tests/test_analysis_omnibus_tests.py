"""Tests for src/analysis/omnibus_tests.py (Milestone 14B, Section 32):
Friedman test on small synthetic matched triples.
"""

import pytest

from src.analysis.omnibus_tests import friedman_test


def test_friedman_returns_none_for_empty_input():
    assert friedman_test([]) is None


def test_friedman_reports_insufficient_blocks_below_three():
    result = friedman_test([(1, 2, 3), (2, 3, 1)])
    assert result.n == 2
    assert result.statistic is None
    assert "fewer than 3" in result.note


def test_friedman_three_way_tie_in_every_block_reports_no_variation():
    triples = [(5.0, 5.0, 5.0)] * 4
    result = friedman_test(triples)
    assert result.statistic == 0.0
    assert result.p_value == 1.0
    assert "tie" in result.note


def test_friedman_identical_triple_repeated_is_not_treated_as_degenerate():
    # Every block holding the SAME (1, 2, 3) triple is a perfectly
    # consistent ranking across blocks (C always highest, A always
    # lowest) — this must be reported as a real, highly significant
    # result, not misclassified as "no variation" merely because the
    # triples are identical to each other across blocks.
    triples = [(1.0, 2.0, 3.0)] * 10
    result = friedman_test(triples)
    assert result.statistic is not None
    assert result.statistic > 0
    assert result.p_value < 0.05


def test_friedman_detects_a_consistent_ranking_difference():
    # Column C (index 2) is always ranked highest, A always lowest.
    triples = [(1.0, 2.0, 3.0)] * 10
    result = friedman_test(triples)
    assert result.n == 10
    assert result.statistic is not None
    assert result.p_value is not None
    assert result.p_value < 0.05


def test_friedman_n_matches_number_of_triples():
    triples = [(1.0, 2.0, 1.5), (2.0, 1.0, 1.8), (1.2, 1.9, 2.1), (3.0, 1.0, 2.0)]
    result = friedman_test(triples)
    assert result.n == 4


def test_friedman_random_permutations_give_high_p_value():
    # Balanced, non-systematic pattern across many blocks -> should NOT
    # reject the null (no reliable architecture ranking difference).
    triples = [
        (1, 2, 3), (2, 3, 1), (3, 1, 2), (1, 3, 2),
        (2, 1, 3), (3, 2, 1), (1, 2, 3), (2, 3, 1),
        (3, 1, 2), (1, 3, 2), (2, 1, 3), (3, 2, 1),
    ]
    result = friedman_test(triples)
    assert result.p_value > 0.05
