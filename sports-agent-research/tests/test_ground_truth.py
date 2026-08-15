"""Ground-truth verification tests.

Every ground-truth value is independently recomputed here using the
deterministic calculation library and compared against the generated
ground truth — nothing is trusted just because it was written into JSON.
"""

import pytest

from src.calculations.odds_math import (
    best_odds,
    expected_value,
    implied_probability,
    is_positive_ev,
)
from src.evaluation.dataset import load_test_scenarios
from src.evaluation.ground_truth import (
    export_ground_truth,
    generate_all_ground_truth,
    generate_ground_truth_for_scenario,
)


def _scenarios_by_id():
    return {s.scenario_id: s for s in load_test_scenarios()}


def _ground_truth_by_id():
    return {gt.scenario_id: gt for gt in generate_all_ground_truth()}


def test_ground_truth_generated_for_every_scenario():
    scenarios = _scenarios_by_id()
    ground_truth = _ground_truth_by_id()
    assert set(scenarios.keys()) == set(ground_truth.keys())


@pytest.mark.parametrize("scenario_id", [f"S{i:03d}" for i in range(1, 15)])
def test_ground_truth_matches_independent_recalculation(scenario_id):
    scenarios = _scenarios_by_id()
    ground_truth = _ground_truth_by_id()

    scenario = scenarios[scenario_id]
    gt = ground_truth[scenario_id]

    odds_values = [o.american_odds for o in scenario.sportsbook_odds]
    expected_best = best_odds(odds_values)
    expected_implied = implied_probability(expected_best)
    expected_ev = expected_value(expected_best, scenario.estimated_true_probability)
    expected_positive = is_positive_ev(expected_best, scenario.estimated_true_probability)

    assert gt.expected_best_odds == expected_best
    assert gt.expected_implied_probability == pytest.approx(expected_implied)
    assert gt.expected_ev == pytest.approx(expected_ev)
    assert gt.expected_positive_ev == expected_positive

    # Best sportsbook(s) must actually offer the best odds.
    for sportsbook_name in gt.expected_best_sportsbooks:
        matching = [o for o in scenario.sportsbook_odds if o.sportsbook == sportsbook_name]
        assert len(matching) == 1
        assert matching[0].american_odds == expected_best
    assert gt.expected_best_sportsbook in gt.expected_best_sportsbooks

    # expected_sportsbooks must equal the set of sportsbooks actually present.
    assert set(gt.expected_sportsbooks) == {o.sportsbook for o in scenario.sportsbook_odds}


# ---------------------------------------------------------------------------
# Step 15 hand-verified scenarios
# ---------------------------------------------------------------------------


def test_hand_verified_positive_ev_scenario_S014():
    gt = _ground_truth_by_id()["S014"]
    assert gt.expected_best_sportsbook == "FanDuel"
    assert gt.expected_best_odds == 150
    assert gt.expected_implied_probability == pytest.approx(0.4)
    assert gt.expected_ev == pytest.approx(0.125)
    assert gt.expected_positive_ev is True


def test_hand_verified_negative_odds_best_line_scenario_S002():
    gt = _ground_truth_by_id()["S002"]
    assert gt.expected_best_sportsbook == "FanDuel"
    assert gt.expected_best_odds == -110

    scenario = _scenarios_by_id()["S002"]
    available_odds = {o.sportsbook: o.american_odds for o in scenario.sportsbook_odds}
    assert available_odds == {
        "DraftKings": -120,
        "FanDuel": -110,
        "BetMGM": -130,
        "Caesars": -115,
    }
    # -110 is closer to even money than -115, -120, -130: it requires the
    # smallest stake to win $100, i.e. the lowest implied probability/risk.
    for odds in (-115, -120, -130):
        assert implied_probability(-110) < implied_probability(odds)


# ---------------------------------------------------------------------------
# Tie scenario (S007) and break-even scenario (S006)
# ---------------------------------------------------------------------------


def test_tie_scenario_preserves_all_best_sportsbooks():
    gt = _ground_truth_by_id()["S007"]
    assert gt.expected_best_odds == 125
    assert set(gt.expected_best_sportsbooks) == {"DraftKings", "FanDuel"}
    assert gt.expected_best_sportsbook in gt.expected_best_sportsbooks


def test_break_even_scenario_is_not_classified_positive():
    gt = _ground_truth_by_id()["S006"]
    assert gt.expected_ev == pytest.approx(0.0, abs=1e-9)
    assert gt.expected_positive_ev is False


def test_missing_data_scenario_excludes_absent_sportsbook():
    gt = _ground_truth_by_id()["S008"]
    assert "FanDuel" not in gt.expected_sportsbooks
    assert set(gt.expected_sportsbooks) == {"DraftKings", "BetMGM", "Caesars"}


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------


def test_ground_truth_generation_is_deterministic():
    first = [gt.model_dump(mode="json") for gt in generate_all_ground_truth()]
    second = [gt.model_dump(mode="json") for gt in generate_all_ground_truth()]
    assert first == second


def test_export_ground_truth_is_byte_identical_across_runs(tmp_path):
    path_a = tmp_path / "ground_truth_a.json"
    path_b = tmp_path / "ground_truth_b.json"
    export_ground_truth(path_a)
    export_ground_truth(path_b)
    assert path_a.read_text() == path_b.read_text()
