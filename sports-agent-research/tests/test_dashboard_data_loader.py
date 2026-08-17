"""Tests for dashboard/data_loader.py (Milestone 13): scenario manifest
reuse, ground-truth-free AgentRequest construction, demo-mode agent
execution (MOCK only — no paid API calls), experiment loading, and
raw-run filtering/grouping/aggregation. Mirrors the fixture style of
tests/test_experiment_persistence.py.
"""

import pytest

from dashboard import data_loader
from src.experiments import runner as experiment_runner
from src.experiments.config import ExecutionMode, ExperimentConfig
from src.models import ArchitectureType


# ---------------------------------------------------------------------------
# Scenario manifest / request construction (Section 4-5)
# ---------------------------------------------------------------------------


def test_full_scenario_manifest_covers_every_controlled_scenario():
    manifest = data_loader.full_scenario_manifest()
    assert {s.scenario_id for s in manifest} == set(data_loader.all_scenario_ids())
    assert len(manifest) >= 11  # at least the Milestone 9B/10B/11 default subset


def test_manifest_scenarios_have_no_ground_truth_fields():
    from src.experiments.config import ExperimentScenario

    forbidden = {"expected_best_sportsbook", "expected_best_odds", "expected_ev"}
    assert not (set(ExperimentScenario.model_fields.keys()) & forbidden)


def test_build_demo_request_matches_scenario_identity():
    manifest = data_loader.full_scenario_manifest()
    scenario = next(s for s in manifest if s.scenario_id == "S001")
    request = data_loader.build_demo_request(scenario)
    assert request.scenario_id == "S001"
    assert request.market_type == scenario.market_type
    assert request.selected_outcome == scenario.selected_outcome
    assert request.query == scenario.query


def test_build_demo_request_has_no_ground_truth_fields():
    from src.agents.base import AgentRequest

    forbidden = {"expected_best_sportsbook", "expected_best_odds", "expected_ev"}
    assert not (set(AgentRequest.model_fields.keys()) & forbidden)


def test_game_context_returns_team_names_not_ground_truth():
    context = data_loader.game_context("S001")
    assert "game" in context
    assert "home_team" in context["game"]
    assert "expected_best_sportsbook" not in context


# ---------------------------------------------------------------------------
# Demo-mode agent execution (Sections 5, 25) — MOCK only, no API calls.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("architecture", [ArchitectureType.RAG, ArchitectureType.TOOL, ArchitectureType.HYBRID])
def test_run_demo_analysis_mock_mode_produces_a_result_for_every_architecture(architecture):
    manifest = data_loader.full_scenario_manifest()
    scenario = next(s for s in manifest if s.scenario_id == "S001")
    result = data_loader.run_demo_analysis(
        architecture, scenario, ExecutionMode.MOCK,
        model_name="claude-opus-4-8", effort="low", rag_top_k=5, max_tool_iterations=6,
    )
    assert result.architecture == architecture
    assert result.trace is not None
    assert result.aux_handle is not None


def test_run_demo_analysis_never_receives_ground_truth_in_its_request():
    manifest = data_loader.full_scenario_manifest()
    scenario = next(s for s in manifest if s.scenario_id == "S001")
    result = data_loader.run_demo_analysis(
        ArchitectureType.TOOL, scenario, ExecutionMode.MOCK,
        model_name="claude-opus-4-8", effort="low", rag_top_k=5, max_tool_iterations=6,
    )
    request_fields = set(result.request.model_dump().keys())
    assert not (request_fields & {"expected_best_sportsbook", "expected_best_odds", "expected_ev"})


# ---------------------------------------------------------------------------
# Experiment loading (Sections 11, 25)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def small_experiment(tmp_path_factory):
    output_dir = tmp_path_factory.mktemp("dashboard_experiment")
    config = ExperimentConfig(
        experiment_id="dashboard-test",
        experiment_name="dashboard-test",
        architectures=[ArchitectureType.RAG, ArchitectureType.TOOL, ArchitectureType.HYBRID],
        scenario_ids=["S001", "S007"],
        repetitions=2,
        output_dir=str(output_dir),
    )
    experiment_runner.run_experiment(config)
    return output_dir


def test_list_experiment_ids_finds_the_persisted_experiment(small_experiment):
    ids = data_loader.list_experiment_ids(small_experiment)
    assert "dashboard-test" in ids


def test_list_experiment_ids_empty_for_nonexistent_directory(tmp_path):
    assert data_loader.list_experiment_ids(tmp_path / "does_not_exist") == []


def test_load_experiment_reads_all_four_files(small_experiment):
    loaded = data_loader.load_experiment("dashboard-test", small_experiment)
    assert loaded.metadata.config.experiment_id == "dashboard-test"
    assert len(loaded.manifest) == 2
    assert len(loaded.raw_runs) == 12  # 3 architectures x 2 scenarios x 2 repetitions
    assert loaded.summary.recorded_runs == 12


def test_load_experiment_missing_directory_raises_clear_error(tmp_path):
    with pytest.raises(data_loader.ExperimentLoadError):
        data_loader.load_experiment("does-not-exist", tmp_path)


def test_load_experiment_incomplete_directory_raises_clear_error(tmp_path):
    experiment_dir = tmp_path / "partial"
    experiment_dir.mkdir()
    (experiment_dir / "config.json").write_text("{}")
    # manifest.json / raw_results.jsonl / summary.json deliberately missing
    with pytest.raises(data_loader.ExperimentLoadError):
        data_loader.load_experiment("partial", tmp_path)


def test_is_mock_experiment_true_for_mock_config(small_experiment):
    loaded = data_loader.load_experiment("dashboard-test", small_experiment)
    assert data_loader.is_mock_experiment(loaded) is True


# ---------------------------------------------------------------------------
# Filtering / grouping / aggregation (Sections 17-21)
# ---------------------------------------------------------------------------


def test_filter_raw_runs_by_architecture(small_experiment):
    loaded = data_loader.load_experiment("dashboard-test", small_experiment)
    filtered = data_loader.filter_raw_runs(loaded.raw_runs, architecture=ArchitectureType.HYBRID)
    assert filtered
    assert all(r.architecture == ArchitectureType.HYBRID for r in filtered)


def test_filter_raw_runs_by_scenario_and_repetition(small_experiment):
    loaded = data_loader.load_experiment("dashboard-test", small_experiment)
    filtered = data_loader.filter_raw_runs(loaded.raw_runs, scenario_id="S001", repetition=1)
    assert filtered
    assert all(r.scenario_id == "S001" and r.repetition == 1 for r in filtered)
    assert len(filtered) == 3  # one per architecture


def test_group_by_scenario_covers_every_scenario(small_experiment):
    loaded = data_loader.load_experiment("dashboard-test", small_experiment)
    groups = data_loader.group_by_scenario(loaded.raw_runs)
    assert set(groups.keys()) == {"S001", "S007"}


def test_group_by_architecture_covers_every_architecture(small_experiment):
    loaded = data_loader.load_experiment("dashboard-test", small_experiment)
    groups = data_loader.group_by_architecture(loaded.raw_runs)
    assert set(groups.keys()) == {ArchitectureType.RAG, ArchitectureType.TOOL, ArchitectureType.HYBRID}
    assert all(len(runs) == 4 for runs in groups.values())  # 2 scenarios x 2 repetitions


def test_failure_counts_by_architecture_sums_to_total_runs(small_experiment):
    loaded = data_loader.load_experiment("dashboard-test", small_experiment)
    counts = data_loader.failure_counts_by_architecture(loaded.raw_runs)
    total = sum(n for by_status in counts.values() for n in by_status.values())
    assert total == len(loaded.raw_runs)


def test_reconstruct_architecture_specific_result_round_trips(small_experiment):
    from src.evaluation.hybrid_agent_evaluation import HybridAgentEvaluationResult

    loaded = data_loader.load_experiment("dashboard-test", small_experiment)
    hybrid_run = next(r for r in loaded.raw_runs if r.architecture == ArchitectureType.HYBRID)
    result = data_loader.reconstruct_architecture_specific_result(hybrid_run)
    assert isinstance(result, HybridAgentEvaluationResult)
    assert result.scenario_id == hybrid_run.scenario_id


def test_hybrid_conflict_summary_uses_the_existing_milestone11_aggregator(small_experiment):
    from src.evaluation import hybrid_agent_evaluation

    loaded = data_loader.load_experiment("dashboard-test", small_experiment)
    hybrid_runs = [r for r in loaded.raw_runs if r.architecture == ArchitectureType.HYBRID]
    expected = hybrid_agent_evaluation.summarize_results(
        [data_loader.reconstruct_architecture_specific_result(r) for r in hybrid_runs]
    )
    actual = data_loader.hybrid_conflict_summary(loaded.raw_runs)
    assert actual == expected


def test_hybrid_conflict_summary_none_when_no_hybrid_runs():
    assert data_loader.hybrid_conflict_summary([]) is None


def test_ground_truth_lookup_available_for_research_context_only():
    ground_truth = data_loader.ground_truth_by_scenario()
    assert "S001" in ground_truth
    assert ground_truth["S001"].expected_best_sportsbook
