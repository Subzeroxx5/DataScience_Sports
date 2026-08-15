"""Deterministic market-level quantitative mathematics (Milestone 7A).

Builds on top of src/calculations/odds_math.py's implied_probability and
expected_value — this module never reimplements those formulas, only
combines their outputs across a market (multiple mutually exclusive
outcomes, multiple sportsbooks quoting the same outcome).

This module is a low-level deterministic dependency only. It must not
import agents, RAG, retrievers, tools, providers, or any LLM SDK — see
docs/ARCHITECTURE.md ("Shared Quant Engine") and
.claude/rules/architecture-boundaries.md.

Terminology: values produced here (no-vig probability, market consensus,
leave-one-out consensus) are market-derived reference probabilities, not
claims about the true probability of a sporting outcome — see
docs/QUANT_STRATEGY.md, "Critical Framing". They are deliberately never
named `true_probability`.

No internal rounding: every function returns full floating-point
precision. Display rounding belongs to a future presentation/report
layer, not here.
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import median, mean as _mean, pstdev
from typing import Sequence

from src.calculations.odds_math import implied_probability

MIN_CONSENSUS_BOOKS = 2
"""Minimum number of OTHER sportsbooks that must remain after excluding
the target sportsbook in a leave-one-out consensus (see
calculate_leave_one_out_consensus). A target + 2 comparison books (3
total) is the smallest valid configuration; a target + 1 comparison book
is rejected as insufficient for this research design."""


def _validate_probability(value: float, name: str = "probability") -> None:
    if not (0.0 <= value <= 1.0):
        raise ValueError(f"{name} must be between 0 and 1, got {value}")


def _validate_probabilities(values: Sequence[float], name: str = "probabilities") -> None:
    if not values:
        raise ValueError(f"{name} cannot be empty")
    for value in values:
        _validate_probability(value, name)


# ---------------------------------------------------------------------------
# Overround / no-vig
# ---------------------------------------------------------------------------


def calculate_overround(odds: Sequence[int]) -> float:
    """Sum of raw implied probabilities across a mutually exclusive market.

    For a fairly-priced (zero-vig) market this would equal 1.0; real
    sportsbook prices sum to slightly more than 1.0 (the "vig"/"juice").
    """
    if not odds:
        raise ValueError("odds cannot be empty")
    return sum(implied_probability(o) for o in odds)


def remove_vig_from_probabilities(raw_probabilities: Sequence[float]) -> list[float]:
    """Normalize raw implied probabilities so they sum to ~1.0 (no-vig / fair).

    fair_i = raw_i / sum(raw_probabilities)

    Operates on already-computed probabilities (general n-outcome
    market); calculate_no_vig_probabilities is the odds-based convenience
    wrapper around this for the common two-outcome case.
    """
    _validate_probabilities(raw_probabilities, "raw_probabilities")
    total = sum(raw_probabilities)
    if total <= 0:
        raise ValueError(f"sum of raw_probabilities must be positive, got {total}")
    return [p / total for p in raw_probabilities]


def calculate_no_vig_probabilities(odds: Sequence[int]) -> list[float]:
    """Convenience wrapper: implied_probability() for each price, then
    remove_vig_from_probabilities(). Values are "fair"/"no-vig"
    probabilities — market-derived estimates, never `true_probability`.
    """
    if not odds:
        raise ValueError("odds cannot be empty")
    raw_probabilities = [implied_probability(o) for o in odds]
    return remove_vig_from_probabilities(raw_probabilities)


# ---------------------------------------------------------------------------
# Market consensus
# ---------------------------------------------------------------------------


def calculate_market_consensus(probabilities: Sequence[float]) -> float:
    """Unweighted arithmetic mean of fair probabilities across sportsbooks.

    Initial research policy (see docs/QUANT_STRATEGY.md): no sportsbook
    weighting, liquidity weighting, "sharp book" weighting, or predictive
    modeling — those introduce assumptions beyond the current research
    question and are explicitly deferred.
    """
    _validate_probabilities(probabilities)
    return _mean(probabilities)


def calculate_leave_one_out_consensus(
    sportsbook_probabilities: Sequence[tuple[str, float]],
    excluded_sportsbook: str,
) -> float:
    """Market consensus excluding one sportsbook's own probability from
    the reference used to evaluate that same sportsbook (reduces
    circularity — see docs/QUANT_STRATEGY.md).

    Args:
        sportsbook_probabilities: (sportsbook_name, fair_probability)
            pairs. Sportsbook names must be unique, non-blank strings.
        excluded_sportsbook: the sportsbook to exclude (must be present).

    Raises:
        ValueError: sportsbook_probabilities is empty; any sportsbook
            identifier is blank/non-string; sportsbook identifiers are
            not unique; excluded_sportsbook is not present; any
            probability is out of [0, 1]; or fewer than
            MIN_CONSENSUS_BOOKS sportsbooks remain after exclusion.
    """
    if not sportsbook_probabilities:
        raise ValueError("sportsbook_probabilities cannot be empty")

    names = [name for name, _ in sportsbook_probabilities]
    for name in names:
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"malformed sportsbook identifier: {name!r}")
    if len(set(names)) != len(names):
        raise ValueError("sportsbook_probabilities contains duplicate sportsbook identifiers")
    if excluded_sportsbook not in names:
        raise ValueError(f"unknown sportsbook: {excluded_sportsbook!r}")

    remaining = [p for name, p in sportsbook_probabilities if name != excluded_sportsbook]
    _validate_probabilities(remaining, "remaining probabilities")
    if len(remaining) < MIN_CONSENSUS_BOOKS:
        raise ValueError(
            f"at least {MIN_CONSENSUS_BOOKS} comparison sportsbooks required after "
            f"excluding {excluded_sportsbook!r}, got {len(remaining)}"
        )
    return calculate_market_consensus(remaining)


# ---------------------------------------------------------------------------
# Probability edge
# ---------------------------------------------------------------------------


def calculate_probability_edge(
    market_reference_probability: float, book_implied_probability: float
) -> float:
    """probability_edge = market_reference_probability - book_implied_probability.

    Returned as a raw decimal (e.g. 0.0252), never pre-formatted as a
    percentage — display formatting belongs to a future report layer.
    """
    _validate_probability(market_reference_probability, "market_reference_probability")
    _validate_probability(book_implied_probability, "book_implied_probability")
    return market_reference_probability - book_implied_probability


# ---------------------------------------------------------------------------
# Market dispersion
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MarketDispersion:
    """Descriptive statistics only — high dispersion is not, by itself, an
    exploitable opportunity; it is reported for later analysis."""

    mean_probability: float
    median_probability: float
    std_dev: float
    probability_range: float
    book_count: int


def calculate_market_dispersion(probabilities: Sequence[float]) -> MarketDispersion:
    """Descriptive dispersion statistics across sportsbooks' fair
    probabilities for the same outcome: mean, median, population standard
    deviation, and range (max - min).

    A single probability is valid input (book_count=1): std_dev and
    probability_range are both 0.0 in that case, not an error.
    """
    _validate_probabilities(probabilities)
    values = list(probabilities)
    return MarketDispersion(
        mean_probability=_mean(values),
        median_probability=median(values),
        std_dev=pstdev(values),
        probability_range=max(values) - min(values),
        book_count=len(values),
    )


def calculate_signed_distance_from_consensus(
    book_probability: float, consensus_probability: float
) -> float:
    """book_probability - consensus_probability (positive = book prices
    this outcome as more likely than the consensus)."""
    _validate_probability(book_probability, "book_probability")
    _validate_probability(consensus_probability, "consensus_probability")
    return book_probability - consensus_probability


def calculate_absolute_distance_from_consensus(
    book_probability: float, consensus_probability: float
) -> float:
    """abs(book_probability - consensus_probability)."""
    return abs(calculate_signed_distance_from_consensus(book_probability, consensus_probability))
