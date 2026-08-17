"""Three-architecture omnibus comparison (Milestone 14B, Sections 11-13).

Friedman test: a repeated-measures, nonparametric omnibus test for
matched observations under 3+ related conditions (here: the same
scenario_id[+repetition] evaluated under RAG, TOOL, and HYBRID) — the
appropriate choice per Section 11/12. Delegates to
scipy.stats.friedmanchisquare rather than a hand-rolled reimplementation.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class FriedmanResult:
    n: int  # number of matched triples (blocks)
    statistic: float | None
    p_value: float | None
    note: str = ""


def friedman_test(triples: list[tuple[float, float, float]]) -> FriedmanResult | None:
    """`triples` is a list of (rag_value, tool_value, hybrid_value) for
    the SAME matched (scenario_id[, repetition]) unit — see
    src.analysis.pairing.align_three_way. Requires at least 3 matched
    blocks (scipy's own minimum for a meaningful chi-square
    approximation); returns a result with statistic=None and an
    explanatory note when there are too few blocks, rather than raising
    or fabricating a p-value.

    The only true degenerate case is when EVERY block is a three-way tie
    (e.g. every block is (5, 5, 5)) — there is no within-block ranking
    information anywhere, so the chi-square statistic's denominator is
    zero. This is NOT the same as every block holding the identical
    (but non-tied) triple, e.g. (1, 2, 3) repeated: that is a perfectly
    consistent ranking across blocks and is exactly the case Friedman
    should report as maximally significant, not degenerate — so the
    degeneracy check below inspects ties WITHIN each block, never
    equality ACROSS blocks. scipy itself does not raise for the
    zero-variance case; it silently returns NaN (with a RuntimeWarning),
    which is detected and reported explicitly here instead of being
    surfaced as a fabricated statistic.
    """
    if not triples:
        return None
    n = len(triples)
    if n < 3:
        return FriedmanResult(n=n, statistic=None, p_value=None, note="fewer than 3 matched blocks — Friedman test not meaningful")

    rag_values = [t[0] for t in triples]
    tool_values = [t[1] for t in triples]
    hybrid_values = [t[2] for t in triples]

    if all(len(set(block)) == 1 for block in triples):
        return FriedmanResult(
            n=n, statistic=0.0, p_value=1.0,
            note="every block is a three-way tie — no within-block ranking information to test",
        )

    import math

    from scipy.stats import friedmanchisquare

    try:
        result = friedmanchisquare(rag_values, tool_values, hybrid_values)
        statistic = float(result.statistic)
        p_value = float(result.pvalue)
    except ValueError as exc:
        return FriedmanResult(n=n, statistic=None, p_value=None, note=f"test not computable: {exc}")

    if math.isnan(statistic) or math.isnan(p_value):
        return FriedmanResult(
            n=n, statistic=None, p_value=None,
            note="test not computable: insufficient rank variation across blocks (scipy returned NaN)",
        )
    return FriedmanResult(n=n, statistic=statistic, p_value=p_value)
