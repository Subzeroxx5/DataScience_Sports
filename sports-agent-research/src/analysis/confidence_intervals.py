"""Confidence intervals for proportions (Milestone 14B, Section 7).

Wilson score interval — preferred over the naive p +/- 1.96*SE normal
approximation because it stays within [0, 1] and remains well-behaved
for proportions near 0 or 1 or with small n, both of which occur in this
dataset (e.g. freshness accuracy at 100%).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# z-score for a 95% two-sided confidence interval.
_Z_95 = 1.959963984540054


@dataclass
class WilsonInterval:
    proportion: float
    lower: float
    upper: float
    n: int
    confidence: float


def wilson_score_interval(successes: int, n: int, confidence: float = 0.95) -> WilsonInterval | None:
    """Closed-form Wilson score interval for a binomial proportion.

    Returns None when n == 0 (no observations — an interval around an
    undefined proportion would be fabricated, not computed).

    Formula (Wilson, 1927):
        center = (p + z^2/(2n)) / (1 + z^2/n)
        half_width = z * sqrt(p(1-p)/n + z^2/(4n^2)) / (1 + z^2/n)
    """
    if n <= 0:
        return None
    if successes < 0 or successes > n:
        raise ValueError(f"successes={successes} must be between 0 and n={n}")

    z = _Z_95 if confidence == 0.95 else _z_for_confidence(confidence)
    p = successes / n
    z2 = z * z

    denominator = 1 + z2 / n
    center = (p + z2 / (2 * n)) / denominator
    half_width = (z * math.sqrt((p * (1 - p) / n) + (z2 / (4 * n * n)))) / denominator

    lower = max(0.0, center - half_width)
    upper = min(1.0, center + half_width)
    return WilsonInterval(proportion=p, lower=lower, upper=upper, n=n, confidence=confidence)


def _z_for_confidence(confidence: float) -> float:
    """Two-sided z critical value for an arbitrary confidence level,
    via the inverse standard normal CDF (Acklam's rational
    approximation would be overkill here — scipy is already a project
    dependency, so defer to it for anything other than the hard-coded
    95% case above, which needs no import)."""
    from scipy.stats import norm

    return float(norm.ppf(1 - (1 - confidence) / 2))
