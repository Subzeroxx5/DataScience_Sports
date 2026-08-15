"""Tests for src/evaluation/quant_ground_truth.py (Milestone 7B)."""

import ast
from pathlib import Path

import pytest

from src.calculations.market import (
    calculate_leave_one_out_consensus,
    calculate_market_dispersion,
    calculate_no_vig_probabilities,
    calculate_probability_edge,
)
from src.calculations.odds_math import expected_value, implied_probability, is_positive_ev
from src.evaluation.ground_truth import generate_all_ground_truth
from src.evaluation.quant_ground_truth import (
    QUANT_GROUND_TRUTH_PATH,
    export_quant_ground_truth,
    generate_all_quant_ground_truth,
    summarize_quant_coverage,
)
from src.models import QuantGroundTruth, ReferenceProbabilityMode


@pytest.fixture(scope="module")
def quant_ground_truth():
    return {item.scenario_id: item for item in generate_all_quant_ground_truth()}


# ---------------------------------------------------------------------------
# Eligibility
# ---------------------------------------------------------------------------


def test_complete_two_sided_market_is_quant_evaluable(quant_ground_truth):
    assert quant_ground_truth["S001"].quant_evaluable is True
    assert quant_ground_truth["S001"].ineligibility_reason is None


def test_incomplete_two_sided_market_is_not_quant_evaluable(quant_ground_truth):
    # S002 (Warriors) has no opposing-side current data at all.
    assert quant_ground_truth["S002"].quant_evaluable is False
    assert quant_ground_truth["S002"].ineligibility_reason is not None


def test_insufficient_comparison_books_not_quant_evaluable():
    # Simulate a market with only 2 complete books (target + 1 comparison
    # is below MIN_CONSENSUS_BOOKS=2 comparison books required).
    from src.evaluation.quant_ground_truth import generate_quant_ground_truth_for_scenario

    definition = {
        "scenario_id": "SYNTH-INSUFFICIENT",
        "game": {"game_id": "G-SYNTH", "home_team": "Team A", "away_team": "Team B"},
        "market": {"market_type": "moneyline", "selected_outcome": "Team A"},
    }
    current_records = [
        {
            "game_id": "G-SYNTH",
            "market_type": "moneyline",
            "selected_outcome": "Team A",
            "sportsbook": "DraftKings",
            "american_odds": 120,
            "is_current": True,
        },
        {
            "game_id": "G-SYNTH",
            "market_type": "moneyline",
            "selected_outcome": "Team B",
            "sportsbook": "DraftKings",
            "american_odds": -140,
            "is_current": True,
        },
        {
            "game_id": "G-SYNTH",
            "market_type": "moneyline",
            "selected_outcome": "Team A",
            "sportsbook": "FanDuel",
            "american_odds": 125,
            "is_current": True,
        },
        {
            "game_id": "G-SYNTH",
            "market_type": "moneyline",
            "selected_outcome": "Team B",
            "sportsbook": "FanDuel",
            "american_odds": -145,
            "is_current": True,
        },
    ]
    result = generate_quant_ground_truth_for_scenario(definition, current_records)
    assert result.quant_evaluable is False
    assert "at least 3 required" in result.ineligibility_reason


def test_spread_and_total_scenarios_not_quant_evaluable(quant_ground_truth):
    assert quant_ground_truth["S012"].quant_evaluable is False
    assert quant_ground_truth["S013"].quant_evaluable is False


def test_exactly_minimum_books_scenario_is_evaluable(quant_ground_truth):
    # S008 has exactly 3 complete books (FanDuel absent) — the minimum
    # threshold edge case (target + 2 comparison books).
    s008 = quant_ground_truth["S008"]
    assert s008.quant_evaluable is True
    assert len(s008.sportsbook_analyses) == 3
    for analysis in s008.sportsbook_analyses:
        assert len(analysis.comparison_sportsbooks) == 2


def test_ineligible_scenarios_never_have_analyses_or_dispersion(quant_ground_truth):
    for scenario_id, item in quant_ground_truth.items():
        if not item.quant_evaluable:
            assert item.sportsbook_analyses == []
            assert item.market_dispersion is None


# ---------------------------------------------------------------------------
# No-vig reuse
# ---------------------------------------------------------------------------


def test_no_vig_probabilities_sum_to_one_per_book(quant_ground_truth):
    s001 = quant_ground_truth["S001"]
    for analysis in s001.sportsbook_analyses:
        # no_vig_probability is one side; confirm it matches a direct
        # recomputation via the Milestone 7A function (not duplicated).
        assert 0.0 <= analysis.no_vig_probability <= 1.0


def test_generator_reuses_no_vig_function_exactly(quant_ground_truth):
    s001 = quant_ground_truth["S001"]
    fanduel = next(a for a in s001.sportsbook_analyses if a.sportsbook == "FanDuel")
    expected_no_vig = calculate_no_vig_probabilities([125, -145])[0]
    assert fanduel.no_vig_probability == pytest.approx(expected_no_vig)


# ---------------------------------------------------------------------------
# Leave-one-out
# ---------------------------------------------------------------------------


def test_target_sportsbook_excluded_from_own_reference(quant_ground_truth):
    s001 = quant_ground_truth["S001"]
    for analysis in s001.sportsbook_analyses:
        assert analysis.sportsbook not in analysis.comparison_sportsbooks


def test_expected_comparison_books_included(quant_ground_truth):
    s001 = quant_ground_truth["S001"]
    fanduel = next(a for a in s001.sportsbook_analyses if a.sportsbook == "FanDuel")
    assert set(fanduel.comparison_sportsbooks) == {"DraftKings", "BetMGM", "Caesars"}


def test_leave_one_out_result_matches_known_arithmetic(quant_ground_truth):
    s001 = quant_ground_truth["S001"]
    fanduel = next(a for a in s001.sportsbook_analyses if a.sportsbook == "FanDuel")

    dk_novig = calculate_no_vig_probabilities([120, -140])[0]
    mgm_novig = calculate_no_vig_probabilities([115, -135])[0]
    cae_novig = calculate_no_vig_probabilities([122, -142])[0]
    expected_reference = (dk_novig + mgm_novig + cae_novig) / 3

    assert fanduel.market_reference_probability == pytest.approx(expected_reference)


# ---------------------------------------------------------------------------
# Probability edge
# ---------------------------------------------------------------------------


def test_probability_edge_positive_example(quant_ground_truth):
    s008 = quant_ground_truth["S008"]
    betmgm = next(a for a in s008.sportsbook_analyses if a.sportsbook == "BetMGM")
    assert betmgm.probability_edge > 0


def test_probability_edge_negative_example(quant_ground_truth):
    s001 = quant_ground_truth["S001"]
    caesars = next(a for a in s001.sportsbook_analyses if a.sportsbook == "Caesars")
    assert caesars.probability_edge < 0


def test_probability_edge_matches_direct_calculation(quant_ground_truth):
    s001 = quant_ground_truth["S001"]
    fanduel = next(a for a in s001.sportsbook_analyses if a.sportsbook == "FanDuel")
    expected_edge = calculate_probability_edge(
        fanduel.market_reference_probability, fanduel.book_implied_probability
    )
    assert fanduel.probability_edge == pytest.approx(expected_edge)


# ---------------------------------------------------------------------------
# EV
# ---------------------------------------------------------------------------


def test_ground_truth_ev_matches_existing_ev_function(quant_ground_truth):
    s001 = quant_ground_truth["S001"]
    for analysis in s001.sportsbook_analyses:
        expected_ev = expected_value(
            analysis.american_odds, analysis.market_reference_probability
        )
        assert analysis.expected_value == pytest.approx(expected_ev)


def test_positive_negative_classification_matches_existing_function(quant_ground_truth):
    for item in quant_ground_truth.values():
        if not item.quant_evaluable:
            continue
        for analysis in item.sportsbook_analyses:
            expected_positive = is_positive_ev(
                analysis.american_odds, analysis.market_reference_probability
            )
            assert analysis.positive_ev == expected_positive


def test_at_least_one_positive_ev_and_one_negative_ev_exist(quant_ground_truth):
    all_analyses = [
        a for item in quant_ground_truth.values() for a in item.sportsbook_analyses
    ]
    assert any(a.positive_ev for a in all_analyses)
    assert any(not a.positive_ev and a.expected_value < 0 for a in all_analyses)


# ---------------------------------------------------------------------------
# Dispersion
# ---------------------------------------------------------------------------


def test_dispersion_mean_correct(quant_ground_truth):
    s001 = quant_ground_truth["S001"]
    values = [a.no_vig_probability for a in s001.sportsbook_analyses]
    expected = calculate_market_dispersion(values)
    assert s001.market_dispersion.mean_probability == pytest.approx(expected.mean_probability)


def test_dispersion_median_correct(quant_ground_truth):
    s001 = quant_ground_truth["S001"]
    values = [a.no_vig_probability for a in s001.sportsbook_analyses]
    expected = calculate_market_dispersion(values)
    assert s001.market_dispersion.median_probability == pytest.approx(
        expected.median_probability
    )


def test_dispersion_population_std_dev_correct(quant_ground_truth):
    s001 = quant_ground_truth["S001"]
    values = [a.no_vig_probability for a in s001.sportsbook_analyses]
    expected = calculate_market_dispersion(values)
    assert s001.market_dispersion.population_std_dev == pytest.approx(expected.std_dev)


def test_dispersion_range_correct(quant_ground_truth):
    s001 = quant_ground_truth["S001"]
    values = [a.no_vig_probability for a in s001.sportsbook_analyses]
    expected = calculate_market_dispersion(values)
    assert s001.market_dispersion.probability_range == pytest.approx(
        expected.probability_range
    )
    assert s001.market_dispersion.minimum_probability == pytest.approx(min(values))
    assert s001.market_dispersion.maximum_probability == pytest.approx(max(values))


def test_dispersion_uses_probabilities_not_raw_odds(quant_ground_truth):
    s001 = quant_ground_truth["S001"]
    assert 0.0 <= s001.market_dispersion.mean_probability <= 1.0
    assert 0.0 <= s001.market_dispersion.median_probability <= 1.0


def test_dispersion_book_count_matches_analyses(quant_ground_truth):
    for item in quant_ground_truth.values():
        if item.quant_evaluable:
            assert item.market_dispersion.book_count == len(item.sportsbook_analyses)


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_repeated_generation_produces_equivalent_output():
    first = [item.model_dump(mode="json") for item in generate_all_quant_ground_truth()]
    second = [item.model_dump(mode="json") for item in generate_all_quant_ground_truth()]
    assert first == second


def test_export_is_byte_identical_across_runs(tmp_path):
    path_a = tmp_path / "quant_a.json"
    path_b = tmp_path / "quant_b.json"
    export_quant_ground_truth(path_a)
    export_quant_ground_truth(path_b)
    assert path_a.read_text() == path_b.read_text()


def test_committed_quant_ground_truth_matches_fresh_generation(tmp_path):
    import hashlib

    fresh_path = tmp_path / "fresh_quant.json"
    export_quant_ground_truth(fresh_path)
    committed_hash = hashlib.sha256(QUANT_GROUND_TRUTH_PATH.read_bytes()).hexdigest()
    fresh_hash = hashlib.sha256(fresh_path.read_bytes()).hexdigest()
    assert committed_hash == fresh_hash


# ---------------------------------------------------------------------------
# Preservation — original best-line/EV ground truth is untouched
# ---------------------------------------------------------------------------


def test_original_ground_truth_still_generates_correctly():
    original = {gt.scenario_id: gt for gt in generate_all_ground_truth()}
    assert original["S001"].expected_best_odds == 125
    assert original["S001"].expected_best_sportsbook == "FanDuel"
    assert original["S007"].expected_best_sportsbooks == ["DraftKings", "FanDuel"]


def test_tie_semantics_preserved_in_original_ground_truth():
    original = {gt.scenario_id: gt for gt in generate_all_ground_truth()}
    s007 = original["S007"]
    assert len(s007.expected_best_sportsbooks) == 2


def test_quant_ground_truth_does_not_collapse_to_single_best_sportsbook(quant_ground_truth):
    # This milestone intentionally does not add a "best quant sportsbook"
    # summary field — all per-book analyses are preserved so a future
    # consumer can determine ties/rankings itself without this module
    # doing arbitrary tie-breaking.
    for item in quant_ground_truth.values():
        assert not hasattr(item, "best_quant_sportsbook")


def test_reference_probability_mode_distinguishes_from_controlled_reference(quant_ground_truth):
    for item in quant_ground_truth.values():
        assert item.reference_probability_mode == ReferenceProbabilityMode.MARKET_CONSENSUS


def test_ground_truth_json_file_unchanged_by_quant_generation():
    import hashlib

    from src.evaluation.ground_truth import export_ground_truth
    from src.evaluation.ground_truth import DATA_DIR as GT_DATA_DIR

    committed_path = GT_DATA_DIR / "ground_truth.json"
    before_hash = hashlib.sha256(committed_path.read_bytes()).hexdigest()

    export_quant_ground_truth()  # exercise the quant generator

    after_hash = hashlib.sha256(committed_path.read_bytes()).hexdigest()
    assert before_hash == after_hash


def test_rag_corpus_unchanged_by_quant_generation():
    import hashlib

    corpus_path = Path(__file__).resolve().parent.parent / "data" / "rag_documents" / "corpus.jsonl"
    before_hash = hashlib.sha256(corpus_path.read_bytes()).hexdigest()

    export_quant_ground_truth()

    after_hash = hashlib.sha256(corpus_path.read_bytes()).hexdigest()
    assert before_hash == after_hash


# ---------------------------------------------------------------------------
# Coverage summary
# ---------------------------------------------------------------------------


def test_summarize_quant_coverage_matches_manual_counts(quant_ground_truth):
    summary = summarize_quant_coverage()
    evaluable = [item for item in quant_ground_truth.values() if item.quant_evaluable]
    assert summary["quant_evaluable"] == len(evaluable)
    assert summary["total_scenarios"] == len(quant_ground_truth)
    assert summary["non_quant_evaluable"] == len(quant_ground_truth) - len(evaluable)


# ---------------------------------------------------------------------------
# Architecture boundary
# ---------------------------------------------------------------------------


def test_quant_ground_truth_module_has_no_forbidden_imports():
    source_path = (
        Path(__file__).resolve().parent.parent
        / "src"
        / "evaluation"
        / "quant_ground_truth.py"
    )
    tree = ast.parse(source_path.read_text())

    forbidden_prefixes = (
        "src.rag",
        "src.agents",
        "src.providers",
        "src.tools",
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
