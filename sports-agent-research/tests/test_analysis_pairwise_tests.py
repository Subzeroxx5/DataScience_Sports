"""Tests for src/analysis/pairwise_tests.py (Milestone 14B, Section 32):
McNemar exact/binomial handling, Holm correction, and Wilcoxon signed-
rank — all against small, known synthetic datasets, never the actual
final experiment result.
"""

import pytest

from src.analysis.pairwise_tests import holm_correction, mcnemar_test, wilcoxon_signed_rank

# ---------------------------------------------------------------------------
# McNemar's test
# ---------------------------------------------------------------------------


def test_mcnemar_discordant_pair_counts():
    pairs = [(True, True)] * 5 + [(True, False)] * 3 + [(False, True)] * 1 + [(False, False)] * 2
    result = mcnemar_test(pairs)
    assert result.paired_n == 11
    assert result.a_correct_b_incorrect == 3
    assert result.a_incorrect_b_correct == 1
    assert result.concordant_both_correct == 5
    assert result.concordant_both_incorrect == 2


def test_mcnemar_accuracy_and_difference():
    pairs = [(True, False)] * 8 + [(False, False)] * 2  # A: 8/10, B: 0/10
    result = mcnemar_test(pairs)
    assert result.accuracy_a == pytest.approx(0.8)
    assert result.accuracy_b == pytest.approx(0.0)
    assert result.difference_pp == pytest.approx(80.0)


def test_mcnemar_exact_binomial_method_always_used():
    pairs = [(True, False)] * 3 + [(False, True)] * 1 + [(True, True)] * 6
    result = mcnemar_test(pairs)
    assert result.method == "exact_binomial"


def test_mcnemar_zero_discordant_pairs_gives_p_one():
    pairs = [(True, True)] * 5 + [(False, False)] * 5
    result = mcnemar_test(pairs)
    assert result.p_value == 1.0
    assert "zero discordant" in result.note


def test_mcnemar_symmetric_discordant_counts_give_p_one():
    pairs = [(True, False)] * 5 + [(False, True)] * 5
    result = mcnemar_test(pairs)
    assert result.p_value == pytest.approx(1.0)


def test_mcnemar_returns_none_for_empty_pairs():
    assert mcnemar_test([]) is None


def test_mcnemar_p_value_matches_known_exact_computation():
    # b=15, c=5 discordant -> exact two-sided binomial p on min(15,5)=5
    # out of 20 at p=0.5. Known value ~=0.0414.
    pairs = [(True, True)] * 80 + [(True, False)] * 15 + [(False, True)] * 5 + [(False, False)] * 10
    result = mcnemar_test(pairs)
    assert result.p_value == pytest.approx(0.04139, abs=1e-4)


# ---------------------------------------------------------------------------
# Holm-Bonferroni correction
# ---------------------------------------------------------------------------


def test_holm_correction_matches_textbook_example():
    # Classic example: raw p = [.01, .02, .03, .04, .05] -> Holm-adjusted
    # = [.05, .08, .09, .09, .09]
    raw = [0.01, 0.02, 0.03, 0.04, 0.05]
    adjusted = holm_correction(raw)
    assert adjusted == pytest.approx([0.05, 0.08, 0.09, 0.09, 0.09])


def test_holm_correction_is_monotonic_non_decreasing_in_sorted_order():
    raw = [0.2, 0.001, 0.15, 0.001, 0.3]
    adjusted = holm_correction(raw)
    order = sorted(range(len(raw)), key=lambda i: raw[i])
    sorted_adjusted = [adjusted[i] for i in order]
    assert sorted_adjusted == sorted(sorted_adjusted)


def test_holm_correction_caps_at_one():
    raw = [0.9, 0.95, 0.99]
    adjusted = holm_correction(raw)
    assert all(p <= 1.0 for p in adjusted)


def test_holm_correction_preserves_none_entries():
    raw = [0.01, None, 0.03]
    adjusted = holm_correction(raw)
    assert adjusted[1] is None
    assert adjusted[0] is not None
    assert adjusted[2] is not None


def test_holm_correction_none_entries_excluded_from_family_size():
    # With one None dropped, the remaining 2 p-values should be
    # corrected as a family of 2, not 3.
    raw = [0.01, None, 0.04]
    adjusted = holm_correction(raw)
    # smallest of the 2 remaining * 2
    assert adjusted[0] == pytest.approx(0.02)
    assert adjusted[2] == pytest.approx(0.04)


def test_holm_correction_empty_list():
    assert holm_correction([]) == []


def test_holm_correction_all_none():
    assert holm_correction([None, None]) == [None, None]


# ---------------------------------------------------------------------------
# Wilcoxon signed-rank
# ---------------------------------------------------------------------------


def test_wilcoxon_returns_none_for_empty_pairs():
    assert wilcoxon_signed_rank([]) is None


def test_wilcoxon_all_zero_differences_reports_no_test():
    pairs = [(1.0, 1.0), (2.0, 2.0), (3.0, 3.0)]
    result = wilcoxon_signed_rank(pairs)
    assert result.statistic is None
    assert result.p_value is None
    assert "zero" in result.note
    assert result.median_a == result.median_b == 2.0


def test_wilcoxon_computes_medians_correctly():
    pairs = [(1.0, 5.0), (2.0, 6.0), (3.0, 7.0)]
    result = wilcoxon_signed_rank(pairs)
    assert result.median_a == 2.0
    assert result.median_b == 6.0


def test_wilcoxon_detects_a_consistent_shift():
    # B is always exactly 1.0 higher than A -> should be a small p-value
    # once there are enough pairs for the test to be informative.
    pairs = [(float(i), float(i) + 1.0) for i in range(10)]
    result = wilcoxon_signed_rank(pairs)
    assert result.p_value is not None
    assert result.p_value < 0.05


def test_wilcoxon_paired_n_matches_input_length():
    pairs = [(1.0, 2.0), (3.0, 1.0), (2.0, 2.5)]
    result = wilcoxon_signed_rank(pairs)
    assert result.paired_n == 3
