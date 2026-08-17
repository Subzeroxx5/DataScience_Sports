"""Tests for src/experiments/validation.py (Milestone 14A, Sections
19-24, 27): duplicate/missing run-key detection, scenario/architecture
coverage, raw-result schema validation, ground-truth/architecture
isolation audits, and that failed runs remain in the dataset rather than
being dropped. Uses small MOCK experiments purely as synthetic fixtures
for exercising the validation logic — never presented as final data.
"""

import json

import pytest

from src.experiments import runner as experiment_runner
from src.experiments import validation
from src.experiments.config import ExperimentConfig
from src.experiments.runner import RawExperimentRun
from src.models import ArchitectureType


def _config(output_dir, **overrides) -> ExperimentConfig:
    defaults = dict(
        experiment_id="validation-fixture",
        experiment_name="validation-fixture",
        architectures=[ArchitectureType.RAG, ArchitectureType.TOOL, ArchitectureType.HYBRID],
        scenario_ids=["S001", "S007"],
        repetitions=2,
        output_dir=str(output_dir),
    )
    defaults.update(overrides)
    return ExperimentConfig(**defaults)


@pytest.fixture(scope="module")
def small_experiment(tmp_path_factory):
    output_dir = tmp_path_factory.mktemp("validation_fixture")
    config = _config(output_dir)
    experiment_runner.run_experiment(config)
    return config, output_dir / config.experiment_id / "raw_results.jsonl"


# ---------------------------------------------------------------------------
# Schema validation (Section 22)
# ---------------------------------------------------------------------------


def test_load_and_validate_raw_results_parses_every_line(small_experiment):
    _, raw_path = small_experiment
    runs, errors = validation.load_and_validate_raw_results(raw_path)
    assert len(runs) == 12  # 3 architectures x 2 scenarios x 2 repetitions
    assert errors == []


def test_load_and_validate_raw_results_missing_file_reports_error(tmp_path):
    runs, errors = validation.load_and_validate_raw_results(tmp_path / "does_not_exist.jsonl")
    assert runs == []
    assert errors


def test_load_and_validate_raw_results_flags_malformed_line(tmp_path):
    path = tmp_path / "raw_results.jsonl"
    path.write_text("{not valid json\n")
    runs, errors = validation.load_and_validate_raw_results(path)
    assert runs == []
    assert len(errors) == 1
    assert "line 1" in errors[0]


# ---------------------------------------------------------------------------
# Duplicate / missing run keys (Section 19)
# ---------------------------------------------------------------------------


def test_find_duplicate_and_missing_keys_none_for_a_complete_dataset(small_experiment):
    config, raw_path = small_experiment
    runs, _ = validation.load_and_validate_raw_results(raw_path)
    duplicates, missing = validation.find_duplicate_and_missing_keys(runs, config)
    assert duplicates == []
    assert missing == []


def test_find_duplicate_and_missing_keys_detects_duplicates(small_experiment):
    config, raw_path = small_experiment
    runs, _ = validation.load_and_validate_raw_results(raw_path)
    duplicated_runs = runs + [runs[0]]
    duplicates, _ = validation.find_duplicate_and_missing_keys(duplicated_runs, config)
    assert duplicates == [runs[0].key().as_tuple()]


def test_find_duplicate_and_missing_keys_detects_missing(small_experiment):
    config, raw_path = small_experiment
    runs, _ = validation.load_and_validate_raw_results(raw_path)
    incomplete_runs = runs[:-1]
    _, missing = validation.find_duplicate_and_missing_keys(incomplete_runs, config)
    assert len(missing) == 1
    assert missing[0] == runs[-1].key().as_tuple()


# ---------------------------------------------------------------------------
# Scenario / architecture coverage (Sections 20-21)
# ---------------------------------------------------------------------------


def test_scenario_and_architecture_coverage_pass_for_a_complete_dataset(small_experiment):
    config, raw_path = small_experiment
    runs, _ = validation.load_and_validate_raw_results(raw_path)
    coverage, scenario_pass, architecture_pass = validation.build_scenario_coverage(runs, config)
    assert scenario_pass is True
    assert architecture_pass is True
    for scenario_id in config.scenario_ids:
        for architecture in config.architectures:
            assert coverage[scenario_id][architecture.value] == config.repetitions


def test_scenario_coverage_fails_when_a_repetition_is_missing(small_experiment):
    config, raw_path = small_experiment
    runs, _ = validation.load_and_validate_raw_results(raw_path)
    incomplete_runs = [r for r in runs if not (r.scenario_id == "S001" and r.architecture == ArchitectureType.HYBRID and r.repetition == 2)]
    _, scenario_pass, _ = validation.build_scenario_coverage(incomplete_runs, config)
    assert scenario_pass is False


# ---------------------------------------------------------------------------
# Isolation audits (Section 23)
# ---------------------------------------------------------------------------


def test_ground_truth_isolation_ok_is_true_for_the_current_runner():
    assert validation.ground_truth_isolation_ok() is True


def test_architecture_isolation_ok_is_true_for_the_current_agents():
    assert validation.architecture_isolation_ok() is True


# ---------------------------------------------------------------------------
# Failed runs remain preserved (Section 15/16)
# ---------------------------------------------------------------------------


def test_failed_runs_remain_in_the_dataset_and_are_counted(tmp_path, monkeypatch):
    config = _config(tmp_path, architectures=[ArchitectureType.RAG, ArchitectureType.TOOL, ArchitectureType.HYBRID], scenario_ids=["S001"], repetitions=1)

    real_create_agent = experiment_runner.create_agent

    def flaky_create_agent(architecture, cfg, llm_client=None):
        if architecture == ArchitectureType.HYBRID:
            raise RuntimeError("simulated failure — must still be recorded")
        return real_create_agent(architecture, cfg, llm_client)

    monkeypatch.setattr(experiment_runner, "create_agent", flaky_create_agent)
    experiment_runner.run_experiment(config)

    raw_path = tmp_path / config.experiment_id / "raw_results.jsonl"
    runs, errors = validation.load_and_validate_raw_results(raw_path)
    assert errors == []
    assert len(runs) == 3  # all 3 runs recorded, including the failed one
    report = validation.validate_final_dataset(config, raw_path)
    assert report.recorded_runs == 3
    assert report.failed_runs == 1
    assert report.successful_runs == 2
    # coverage/duplicate/missing checks must still pass — failure isolation
    # does not mean the run is dropped from the dataset.
    assert report.scenario_coverage_pass is True
    assert report.duplicate_run_keys == []
    assert report.missing_run_keys == []


# ---------------------------------------------------------------------------
# Full report / dataset_valid (Section 27)
# ---------------------------------------------------------------------------


def test_validate_final_dataset_is_valid_with_matching_hashes(small_experiment):
    config, raw_path = small_experiment
    fingerprints = {"a": "sha256:x"}
    report = validation.validate_final_dataset(
        config, raw_path, pre_fingerprints=fingerprints, post_fingerprints=fingerprints,
    )
    assert report.dataset_valid is True


def test_validate_final_dataset_is_invalid_when_a_hash_changed(small_experiment):
    config, raw_path = small_experiment
    report = validation.validate_final_dataset(
        config, raw_path, pre_fingerprints={"a": "sha256:x"}, post_fingerprints={"a": "sha256:CHANGED"},
    )
    assert report.dataset_valid is False
    assert report.hashes_all_match is False


def test_validate_final_dataset_is_invalid_without_any_hash_comparison(small_experiment):
    # No pre/post fingerprints supplied at all — a dataset must never be
    # declared valid without an artifact-integrity check having run.
    config, raw_path = small_experiment
    report = validation.validate_final_dataset(config, raw_path)
    assert report.dataset_valid is False
