"""Tests for src/analysis/confidence_intervals.py (Milestone 14B,
Section 32): Wilson score interval, against known reference values —
never hard-coded to the actual final experiment result.
"""

import pytest

from src.analysis.confidence_intervals import wilson_score_interval


def test_wilson_interval_matches_known_reference_p_half_n_100():
    # Textbook reference: p=0.5, n=100, 95% Wilson CI ~= [0.404, 0.596]
    result = wilson_score_interval(50, 100)
    assert result.proportion == 0.5
    assert result.lower == pytest.approx(0.4038, abs=1e-3)
    assert result.upper == pytest.approx(0.5962, abs=1e-3)


def test_wilson_interval_matches_known_reference_p_zero():
    # p=0 must still produce a non-degenerate upper bound (unlike the
    # naive normal approximation, which collapses to [0, 0]).
    result = wilson_score_interval(0, 20)
    assert result.lower == 0.0
    assert result.upper > 0.0


def test_wilson_interval_stays_within_0_and_1():
    for successes, n in [(0, 1), (1, 1), (10, 10), (0, 500), (500, 500)]:
        result = wilson_score_interval(successes, n)
        assert 0.0 <= result.lower <= result.upper <= 1.0


def test_wilson_interval_none_for_zero_observations():
    assert wilson_score_interval(0, 0) is None


def test_wilson_interval_rejects_successes_greater_than_n():
    with pytest.raises(ValueError):
        wilson_score_interval(11, 10)


def test_wilson_interval_rejects_negative_successes():
    with pytest.raises(ValueError):
        wilson_score_interval(-1, 10)


def test_wilson_interval_narrows_as_n_increases():
    small = wilson_score_interval(5, 10)
    large = wilson_score_interval(500, 1000)
    assert (large.upper - large.lower) < (small.upper - small.lower)


def test_wilson_interval_records_n_and_confidence():
    result = wilson_score_interval(7, 10, confidence=0.90)
    assert result.n == 10
    assert result.confidence == 0.90
