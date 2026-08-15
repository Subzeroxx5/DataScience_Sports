"""Deterministic American odds and expected value calculations.

These functions are the auditable, reproducible baseline used to generate
and validate experimental ground truth (see docs/research_design.md,
"Mathematics" and "Ground Truth Methodology"). They must never depend on an
LLM. Formulas are fixed and applied over fixed inputs only.
"""

from __future__ import annotations

MIN_AMERICAN_ODDS = -100_000
MAX_AMERICAN_ODDS = 100_000


def _validate_odds(american_odds: int) -> None:
    if american_odds == 0:
        raise ValueError("american_odds cannot be 0")
    if not (MIN_AMERICAN_ODDS <= american_odds <= MAX_AMERICAN_ODDS):
        raise ValueError(
            f"american_odds must be between {MIN_AMERICAN_ODDS} and "
            f"{MAX_AMERICAN_ODDS}, got {american_odds}"
        )


def _validate_probability(probability: float, name: str = "probability") -> None:
    if not (0.0 <= probability <= 1.0):
        raise ValueError(f"{name} must be between 0 and 1, got {probability}")


def _validate_stake(stake: float) -> None:
    if stake <= 0:
        raise ValueError(f"stake must be positive, got {stake}")


def implied_probability(american_odds: int) -> float:
    """Convert American odds to implied probability.

    Positive odds: 100 / (odds + 100)
    Negative odds: abs(odds) / (abs(odds) + 100)
    """
    _validate_odds(american_odds)
    if american_odds > 0:
        return 100 / (american_odds + 100)
    return abs(american_odds) / (abs(american_odds) + 100)


def decimal_odds(american_odds: int) -> float:
    """Convert American odds to decimal odds.

    Positive odds: (odds / 100) + 1
    Negative odds: (100 / abs(odds)) + 1
    """
    _validate_odds(american_odds)
    if american_odds > 0:
        return (american_odds / 100) + 1
    return (100 / abs(american_odds)) + 1


def profit_if_win(american_odds: int, stake: float = 1.0) -> float:
    """Profit on a winning bet of the given stake (not including stake return).

    Positive odds: stake * (odds / 100)
    Negative odds: stake * (100 / abs(odds))
    """
    _validate_odds(american_odds)
    _validate_stake(stake)
    if american_odds > 0:
        return stake * (american_odds / 100)
    return stake * (100 / abs(american_odds))


def expected_value(
    american_odds: int, true_probability: float, stake: float = 1.0
) -> float:
    """Expected value of a bet.

    EV = (true_probability * profit_if_win) - ((1 - true_probability) * stake)
    """
    _validate_odds(american_odds)
    _validate_probability(true_probability, "true_probability")
    _validate_stake(stake)
    win_profit = profit_if_win(american_odds, stake)
    return (true_probability * win_profit) - ((1 - true_probability) * stake)


_EV_EPSILON = 1e-9


def is_positive_ev(american_odds: int, true_probability: float, stake: float = 1.0) -> bool:
    """Whether the expected value is strictly greater than zero.

    EV == 0 (break-even) is NOT positive EV. A small epsilon absorbs
    floating-point rounding dust at the break-even boundary so that
    mathematically break-even bets are never misclassified as positive.
    """
    ev = expected_value(american_odds, true_probability, stake)
    return ev > _EV_EPSILON


def compare_american_odds(odds_a: int, odds_b: int) -> int:
    """Compare two American odds by favorability to the bettor.

    Returns 1 if odds_a is better, -1 if odds_b is better, 0 if equivalent.

    Favorability is ranked by implied probability: lower implied probability
    is better for the bettor (higher payout per unit risked), regardless of
    the positive/negative sign convention. This correctly orders mixed
    comparisons such as +110 vs -110.
    """
    _validate_odds(odds_a)
    _validate_odds(odds_b)
    prob_a = implied_probability(odds_a)
    prob_b = implied_probability(odds_b)
    if prob_a < prob_b:
        return 1
    if prob_a > prob_b:
        return -1
    return 0


def best_odds(odds_list: list[int]) -> int:
    """Return the most favorable American odds from a non-empty list.

    Ties are broken by first occurrence in the input list (stable). Callers
    that need to know about a tie should compare candidates themselves;
    this function intentionally returns a single value.
    """
    if not odds_list:
        raise ValueError("odds_list cannot be empty")
    best = odds_list[0]
    for candidate in odds_list[1:]:
        if compare_american_odds(candidate, best) > 0:
            best = candidate
    return best
