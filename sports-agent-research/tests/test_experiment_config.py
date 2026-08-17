"""Tests for src/experiments/config.py (Milestone 12): ExperimentConfig
validation, the deterministic scenario manifest, the deterministic
architecture-rotation policy, and reproducibility-metadata checksums.
No agent execution in this file — pure configuration/data logic only.
"""

import pytest
from pydantic import ValidationError

from src.experiments.config import (
    DEFAULT_ARCHITECTURE_ORDER,
    ExecutionMode,
    ExperimentConfig,
    ExperimentMetadata,
    _canonical_query,
    build_scenario_manifest,
    compute_artifact_checksums,
    execution_order_for_repetition,
)
from src.models import ArchitectureType, MarketType


def _config(**overrides) -> ExperimentConfig:
    defaults = dict(experiment_id="exp-1", experiment_name="test", repetitions=2)
    defaults.update(overrides)
    return ExperimentConfig(**defaults)


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------


def test_valid_architectures_accepted():
    config = _config(architectures=[ArchitectureType.RAG, ArchitectureType.TOOL])
    assert config.architectures == [ArchitectureType.RAG, ArchitectureType.TOOL]


def test_default_architectures_are_all_three_in_canonical_order():
    config = _config()
    assert config.architectures == [ArchitectureType.RAG, ArchitectureType.TOOL, ArchitectureType.HYBRID]


def test_empty_architectures_rejected():
    with pytest.raises(ValidationError):
        _config(architectures=[])


def test_duplicate_architectures_rejected():
    with pytest.raises(ValidationError):
        _config(architectures=[ArchitectureType.RAG, ArchitectureType.RAG])


def test_valid_repetitions_accepted():
    assert _config(repetitions=5).repetitions == 5


def test_repetitions_below_one_rejected():
    with pytest.raises(ValidationError):
        _config(repetitions=0)


def test_empty_scenario_ids_rejected():
    with pytest.raises(ValidationError):
        _config(scenario_ids=[])


def test_centralized_model_settings_present():
    config = _config()
    assert config.model_name == "claude-opus-4-8"
    assert config.effort == "low"
    # This model family rejects temperature outright (see
    # src/agents/llm_client.py) — recorded as None/N/A, not a usable lever.
    assert config.temperature is None
    assert config.rag_top_k >= 1
    assert config.max_tool_iterations >= 1


def test_execution_mode_defaults_to_mock():
    assert _config().execution_mode == ExecutionMode.MOCK


def test_expected_run_count_hand_example():
    # 3 architectures x 4 scenarios x 2 repetitions = 24
    config = _config(
        architectures=[ArchitectureType.RAG, ArchitectureType.TOOL, ArchitectureType.HYBRID],
        scenario_ids=["S001", "S002", "S003", "S004"],
        repetitions=2,
    )
    assert config.expected_run_count() == 24


# ---------------------------------------------------------------------------
# Scenario manifest (Step 6/7) — no ground truth, one canonical query
# ---------------------------------------------------------------------------


def test_manifest_contains_no_ground_truth_fields():
    manifest = build_scenario_manifest(["S001"])
    forbidden = {
        "expected_best_sportsbook", "expected_best_odds", "expected_ev",
        "expected_positive_ev", "estimated_true_probability", "market_reference_probability",
    }
    from src.experiments.config import ExperimentScenario

    assert not (set(ExperimentScenario.model_fields.keys()) & forbidden)


def test_manifest_quant_evaluable_is_metadata_not_an_answer():
    manifest = build_scenario_manifest(["S001", "S002"])
    by_id = {s.scenario_id: s for s in manifest}
    # S001 is one of the 4 two-sided quant-evaluable scenarios; S002 is not.
    assert by_id["S001"].quant_evaluable is True
    assert by_id["S002"].quant_evaluable is False


def test_canonical_query_has_no_sportsbook_names_or_per_architecture_tuning():
    query = _canonical_query(MarketType.MONEYLINE, "Los Angeles Lakers")
    assert query == "Compare the available moneyline prices for Los Angeles Lakers and identify the best current value."
    for sportsbook in ("DraftKings", "FanDuel", "BetMGM", "Caesars"):
        assert sportsbook not in query


def test_canonical_query_function_takes_no_architecture_parameter():
    # Structural guarantee that the query cannot be tuned per architecture
    # (Step 6: "Do not write easier queries for one architecture").
    import inspect

    params = list(inspect.signature(_canonical_query).parameters)
    assert "architecture" not in params


def test_manifest_is_deterministic():
    manifest_a = build_scenario_manifest(["S001", "S007", "S012", "S013"])
    manifest_b = build_scenario_manifest(["S001", "S007", "S012", "S013"])
    assert manifest_a == manifest_b


def test_manifest_covers_moneyline_spread_and_total():
    manifest = build_scenario_manifest(["S001", "S012", "S013"])
    market_types = {s.market_type for s in manifest}
    assert market_types == {MarketType.MONEYLINE, MarketType.SPREAD, MarketType.TOTAL}


# ---------------------------------------------------------------------------
# Architecture rotation (Step 9) — deterministic, no randomness
# ---------------------------------------------------------------------------


def test_rotation_matches_milestone_example():
    architectures = [ArchitectureType.RAG, ArchitectureType.TOOL, ArchitectureType.HYBRID]
    assert execution_order_for_repetition(architectures, 0) == [
        ArchitectureType.RAG, ArchitectureType.TOOL, ArchitectureType.HYBRID,
    ]
    assert execution_order_for_repetition(architectures, 1) == [
        ArchitectureType.TOOL, ArchitectureType.HYBRID, ArchitectureType.RAG,
    ]
    assert execution_order_for_repetition(architectures, 2) == [
        ArchitectureType.HYBRID, ArchitectureType.RAG, ArchitectureType.TOOL,
    ]


def test_rotation_cycles_back_after_full_period():
    architectures = list(DEFAULT_ARCHITECTURE_ORDER)
    assert execution_order_for_repetition(architectures, 3) == execution_order_for_repetition(architectures, 0)


def test_rotation_is_deterministic_not_random():
    architectures = list(DEFAULT_ARCHITECTURE_ORDER)
    results = {tuple(a.value for a in execution_order_for_repetition(architectures, 1)) for _ in range(10)}
    assert len(results) == 1  # always the same order, never randomized


def test_rotation_handles_two_architectures():
    architectures = [ArchitectureType.RAG, ArchitectureType.TOOL]
    assert execution_order_for_repetition(architectures, 0) == [ArchitectureType.RAG, ArchitectureType.TOOL]
    assert execution_order_for_repetition(architectures, 1) == [ArchitectureType.TOOL, ArchitectureType.RAG]


# ---------------------------------------------------------------------------
# Reproducibility metadata (Step 15)
# ---------------------------------------------------------------------------


def test_checksums_cover_expected_artifacts():
    checksums = compute_artifact_checksums()
    assert set(checksums.keys()) == {
        "current_odds", "rag_corpus", "ground_truth", "quant_ground_truth", "rag_index_config",
    }
    for value in checksums.values():
        assert value.startswith("sha256:") or value == "missing"


def test_checksums_are_stable_across_calls():
    assert compute_artifact_checksums() == compute_artifact_checksums()


def test_experiment_metadata_never_stores_api_keys():
    metadata = ExperimentMetadata.build(_config())
    payload = metadata.model_dump_json()
    assert "ANTHROPIC_API_KEY" not in payload
    assert "sk-ant-" not in payload


def test_experiment_metadata_round_trips_through_json():
    metadata = ExperimentMetadata.build(_config())
    restored = ExperimentMetadata.model_validate_json(metadata.model_dump_json())
    assert restored == metadata
