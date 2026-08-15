"""Unit tests for deterministic American odds and EV calculations."""

import math

import pytest

from src.calculations.odds_math import (
    best_odds,
    compare_american_odds,
    decimal_odds,
    expected_value,
    implied_probability,
    is_positive_ev,
    profit_if_win,
)


# ---------------------------------------------------------------------------
# implied_probability
# ---------------------------------------------------------------------------


def test_implied_probability_positive_odds():
    assert implied_probability(150) == pytest.approx(0.4)


def test_implied_probability_negative_odds():
    assert implied_probability(-200) == pytest.approx(0.6667, abs=1e-4)


def test_implied_probability_plus_100_and_minus_100_are_equal():
    assert implied_probability(100) == pytest.approx(implied_probability(-100))


def test_implied_probability_rejects_zero_odds():
    with pytest.raises(ValueError):
        implied_probability(0)


# ---------------------------------------------------------------------------
# decimal_odds
# ---------------------------------------------------------------------------


def test_decimal_odds_positive():
    assert decimal_odds(150) == pytest.approx(2.5)


def test_decimal_odds_negative():
    assert decimal_odds(-200) == pytest.approx(1.5)


def test_decimal_odds_rejects_zero():
    with pytest.raises(ValueError):
        decimal_odds(0)


# ---------------------------------------------------------------------------
# profit_if_win
# ---------------------------------------------------------------------------


def test_profit_if_win_positive_odds():
    assert profit_if_win(150) == pytest.approx(1.5)


def test_profit_if_win_negative_odds():
    assert profit_if_win(-200) == pytest.approx(0.5)


def test_profit_if_win_scales_with_stake():
    assert profit_if_win(150, stake=10) == pytest.approx(15.0)


def test_profit_if_win_rejects_nonpositive_stake():
    with pytest.raises(ValueError):
        profit_if_win(150, stake=0)
    with pytest.raises(ValueError):
        profit_if_win(150, stake=-5)


# ---------------------------------------------------------------------------
# expected_value
# ---------------------------------------------------------------------------


def test_expected_value_positive_ev_example():
    ev = expected_value(150, 0.45)
    assert ev == pytest.approx(0.125)


def test_expected_value_negative_ev_example():
    ev = expected_value(150, 0.35)
    assert ev == pytest.approx(-0.125)


def test_expected_value_break_even():
    # At the odds' own implied probability, EV must be exactly 0.
    true_probability = implied_probability(150)
    ev = expected_value(150, true_probability)
    assert ev == pytest.approx(0.0, abs=1e-9)


def test_expected_value_rejects_probability_out_of_range():
    with pytest.raises(ValueError):
        expected_value(150, -0.1)
    with pytest.raises(ValueError):
        expected_value(150, 1.1)


def test_expected_value_rejects_zero_odds():
    with pytest.raises(ValueError):
        expected_value(0, 0.5)


def test_expected_value_rejects_nonpositive_stake():
    with pytest.raises(ValueError):
        expected_value(150, 0.45, stake=0)


# ---------------------------------------------------------------------------
# is_positive_ev
# ---------------------------------------------------------------------------


def test_is_positive_ev_true_case():
    assert is_positive_ev(150, 0.45) is True


def test_is_positive_ev_false_case():
    assert is_positive_ev(150, 0.35) is False


def test_is_positive_ev_break_even_is_not_positive():
    true_probability = implied_probability(150)
    assert is_positive_ev(150, true_probability) is False


# ---------------------------------------------------------------------------
# compare_american_odds / best_odds
# ---------------------------------------------------------------------------


def test_compare_odds_plus120_beats_plus110():
    assert compare_american_odds(120, 110) == 1
    assert compare_american_odds(110, 120) == -1


def test_compare_odds_plus110_beats_minus110():
    assert compare_american_odds(110, -110) == 1
    assert compare_american_odds(-110, 110) == -1


def test_compare_odds_minus110_beats_minus120():
    assert compare_american_odds(-110, -120) == 1
    assert compare_american_odds(-120, -110) == -1


def test_compare_odds_minus120_beats_minus150():
    assert compare_american_odds(-120, -150) == 1
    assert compare_american_odds(-150, -120) == -1


def test_compare_odds_tie_at_equivalent_probability():
    assert compare_american_odds(100, -100) == 0


def test_compare_odds_rejects_zero():
    with pytest.raises(ValueError):
        compare_american_odds(0, 110)


def test_best_odds_selects_most_favorable():
    assert best_odds([-110, 150, -200, 130]) == 150


def test_best_odds_all_negative():
    assert best_odds([-150, -110, -120]) == -110


def test_best_odds_tie_breaks_to_first_occurrence():
    # 100 and -100 are equivalent; first occurrence in the list wins.
    assert best_odds([100, -100]) == 100
    assert best_odds([-100, 100]) == -100


def test_best_odds_rejects_empty_list():
    with pytest.raises(ValueError):
        best_odds([])


def test_odds_math_never_returns_nan_or_inf():
    assert math.isfinite(implied_probability(150))
    assert math.isfinite(expected_value(-110, 0.5))
