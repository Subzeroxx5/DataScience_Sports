"""Paired architecture-comparison tests (Milestone 14B, Sections 8-11).

McNemar's test (paired binary outcomes) always uses the exact binomial
form here — Section 8: "Use exact McNemar/binomial handling when
discordant-pair counts are small," and this dataset's discordant counts
are always small (paired N tops out at 110), so the exact form is used
unconditionally rather than switching methods after inspecting counts.

Wilcoxon signed-rank (paired continuous/ordinal outcomes) delegates to
scipy.stats.wilcoxon — never a hand-rolled reimplementation of a
standard, easy-to-get-subtly-wrong rank test.

Holm-Bonferroni correction is a small, well-defined step-down procedure
implemented directly (no library adds it as an isolated one-line
utility).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class McNemarResult:
    paired_n: int
    a_correct_b_incorrect: int  # discordant: A correct, B incorrect
    a_incorrect_b_correct: int  # discordant: B correct, A incorrect
    concordant_both_correct: int
    concordant_both_incorrect: int
    accuracy_a: float
    accuracy_b: float
    difference_pp: float  # accuracy_a - accuracy_b, in percentage points
    p_value: float
    method: str
    note: str = ""


def mcnemar_test(pairs: list[tuple[bool, bool]]) -> McNemarResult | None:
    """Exact (binomial) McNemar's test on paired binary outcomes.

    `pairs` is a list of (correct_a, correct_b) booleans for the SAME
    underlying (scenario_id, repetition) observation under architectures
    A and B (see src.analysis.pairing.align_pair) — already filtered to
    exclude N/A values. Returns None when there are zero paired
    observations (an undefined comparison, not a fabricated p=1.0).

    p-value = 2 * min(P(X <= b), P(X >= b)) under X ~ Binomial(b+c, 0.5),
    where b, c are the two discordant-pair counts — the standard exact
    two-sided McNemar test, computed via scipy.stats.binomtest.
    """
    if not pairs:
        return None

    a_correct_b_incorrect = sum(1 for a, b in pairs if a and not b)
    a_incorrect_b_correct = sum(1 for a, b in pairs if not a and b)
    both_correct = sum(1 for a, b in pairs if a and b)
    both_incorrect = sum(1 for a, b in pairs if not a and not b)

    n = len(pairs)
    accuracy_a = sum(1 for a, _ in pairs if a) / n
    accuracy_b = sum(1 for _, b in pairs if b) / n

    discordant = a_correct_b_incorrect + a_incorrect_b_correct
    if discordant == 0:
        p_value = 1.0
        note = "zero discordant pairs — architectures agreed on every paired observation"
    else:
        from scipy.stats import binomtest

        k = min(a_correct_b_incorrect, a_incorrect_b_correct)
        p_value = binomtest(k, discordant, 0.5, alternative="two-sided").pvalue
        note = ""

    return McNemarResult(
        paired_n=n,
        a_correct_b_incorrect=a_correct_b_incorrect,
        a_incorrect_b_correct=a_incorrect_b_correct,
        concordant_both_correct=both_correct,
        concordant_both_incorrect=both_incorrect,
        accuracy_a=accuracy_a,
        accuracy_b=accuracy_b,
        difference_pp=(accuracy_a - accuracy_b) * 100,
        p_value=float(p_value),
        method="exact_binomial",
        note=note,
    )


@dataclass
class WilcoxonResult:
    paired_n: int
    median_a: float
    median_b: float
    statistic: float | None
    p_value: float | None
    method: str
    note: str = ""


def wilcoxon_signed_rank(pairs: list[tuple[float, float]]) -> WilcoxonResult | None:
    """Wilcoxon signed-rank test on paired continuous/ordinal outcomes,
    via scipy.stats.wilcoxon (zero_method="wilcox": zero differences are
    dropped before ranking — scipy's documented default and the most
    common convention; documented here per Section 28's requirement to
    state assumptions/handling explicitly).

    Returns None for zero pairs. When every pair has an identical
    difference of zero, scipy cannot compute a statistic (there is
    nothing to rank) — that degenerate case is reported explicitly
    (medians identical, no test performed) rather than raising.
    """
    if not pairs:
        return None

    from src.evaluation import metrics

    values_a = [a for a, _ in pairs]
    values_b = [b for _, b in pairs]
    median_a = metrics.median(values_a)
    median_b = metrics.median(values_b)

    differences = [a - b for a, b in pairs]
    if all(d == 0 for d in differences):
        return WilcoxonResult(
            paired_n=len(pairs), median_a=median_a, median_b=median_b,
            statistic=None, p_value=None, method="wilcoxon_signed_rank",
            note="all paired differences are exactly zero — no test performed, medians identical",
        )

    from scipy.stats import wilcoxon

    try:
        result = wilcoxon(values_a, values_b, zero_method="wilcox")
        return WilcoxonResult(
            paired_n=len(pairs), median_a=median_a, median_b=median_b,
            statistic=float(result.statistic), p_value=float(result.pvalue),
            method="wilcoxon_signed_rank",
        )
    except ValueError as exc:
        # e.g. too few non-zero differences for scipy to compute a
        # statistic — report the descriptive medians without a test
        # rather than crashing the whole analysis run.
        return WilcoxonResult(
            paired_n=len(pairs), median_a=median_a, median_b=median_b,
            statistic=None, p_value=None, method="wilcoxon_signed_rank",
            note=f"test not computable: {exc}",
        )


def holm_correction(p_values: list[float | None]) -> list[float | None]:
    """Holm-Bonferroni step-down correction for family-wise multiple
    testing (Section 9-10) over the 3 pairwise architecture comparisons
    within one metric family. None entries (comparisons that had no
    p-value, e.g. zero paired observations) pass through unchanged and
    are excluded from the correction family — they were never tested.

    Standard step-down procedure: sort ascending, multiply the i-th
    smallest (1-indexed) by (m - i + 1), enforce monotonicity, cap at 1.0.
    """
    indexed = [(i, p) for i, p in enumerate(p_values) if p is not None]
    if not indexed:
        return list(p_values)

    indexed.sort(key=lambda item: item[1])
    m = len(indexed)

    adjusted: dict[int, float] = {}
    running_max = 0.0
    for rank, (original_index, p) in enumerate(indexed):
        multiplier = m - rank
        candidate = min(1.0, p * multiplier)
        running_max = max(running_max, candidate)
        adjusted[original_index] = running_max

    return [adjusted.get(i) if p is not None else None for i, p in enumerate(p_values)]
