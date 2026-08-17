"""Tests for src/experiments/runner.py's persistence layer (Milestone
12): raw-result/config/manifest/summary files, duplicate-run protection,
resume behavior, and mock-mode reproducibility across separate output
directories. Uses small (1-2 scenario) mock experiments throughout — no
real API calls.
"""

import json

import pytest

from src.experiments import runner
from src.experiments.config import ExperimentConfig
from src.models import ArchitectureType


def _config(output_dir, experiment_id="persist-test", **overrides) -> ExperimentConfig:
    defaults = dict(
        experiment_id=experiment_id,
        experiment_name="persist-test",
        architectures=[ArchitectureType.RAG, ArchitectureType.TOOL, ArchitectureType.HYBRID],
        scenario_ids=["S001", "S007"],
        repetitions=1,
        output_dir=str(output_dir),
    )
    defaults.update(overrides)
    return ExperimentConfig(**defaults)


@pytest.fixture(scope="module")
def small_experiment(tmp_path_factory):
    output_dir = tmp_path_factory.mktemp("persist_a")
    config = _config(output_dir)
    summary = runner.run_experiment(config)
    return config, summary, output_dir / config.experiment_id


# ---------------------------------------------------------------------------
# File structure (Step 14)
# ---------------------------------------------------------------------------


def test_raw_results_file_has_one_record_per_run(small_experiment):
    _, summary, experiment_dir = small_experiment
    raw_path = experiment_dir / "raw_results.jsonl"
    assert raw_path.is_file()
    lines = raw_path.read_text().strip().splitlines()
    assert len(lines) == summary.recorded_runs == 6  # 3 architectures x 2 scenarios x 1 repetition


def test_raw_results_are_one_json_object_per_line(small_experiment):
    _, _, experiment_dir = small_experiment
    for line in (experiment_dir / "raw_results.jsonl").read_text().strip().splitlines():
        record = json.loads(line)
        assert "architecture" in record
        assert "scenario_id" in record
        assert "repetition" in record
        assert "common_result" in record
        assert "architecture_specific_result" in record


def test_config_file_persisted_with_metadata(small_experiment):
    config, _, experiment_dir = small_experiment
    payload = json.loads((experiment_dir / "config.json").read_text())
    assert payload["config"]["experiment_id"] == config.experiment_id
    assert "artifact_checksums" in payload
    assert "created_at" in payload


def test_manifest_file_persisted(small_experiment):
    _, _, experiment_dir = small_experiment
    payload = json.loads((experiment_dir / "manifest.json").read_text())
    assert {s["scenario_id"] for s in payload} == {"S001", "S007"}


def test_summary_file_persisted(small_experiment):
    _, summary, experiment_dir = small_experiment
    payload = json.loads((experiment_dir / "summary.json").read_text())
    assert payload["experiment_id"] == summary.experiment_id
    assert payload["recorded_runs"] == 6


def test_raw_results_remain_separate_from_summary(small_experiment):
    _, _, experiment_dir = small_experiment
    raw_payload = (experiment_dir / "raw_results.jsonl").read_text()
    summary_payload = json.loads((experiment_dir / "summary.json").read_text())
    # The summary file must not itself contain the raw per-line records —
    # only the ArchitectureSummary/Comparison aggregate structure (which,
    # per Milestone 11, embeds its OWN raw_results list separately from
    # this file's line-oriented log — they are two different serializations
    # of the same underlying data, not one collapsing into the other).
    assert "raw_results.jsonl" not in summary_payload  # sanity: no accidental self-reference
    assert len(raw_payload.strip().splitlines()) == 6


def test_expected_runs_matches_configuration(small_experiment):
    config, summary, _ = small_experiment
    assert summary.expected_runs == config.expected_run_count() == 6


# ---------------------------------------------------------------------------
# Duplicate protection / resume (Step 25)
# ---------------------------------------------------------------------------


def test_rerunning_same_experiment_without_resume_is_refused(small_experiment, tmp_path_factory):
    config, _, experiment_dir = small_experiment
    with pytest.raises(FileExistsError):
        runner.run_experiment(config)


def test_resume_skips_existing_keys_and_adds_only_new_ones(small_experiment):
    config, original_summary, experiment_dir = small_experiment
    extended_config = config.model_copy(update={"scenario_ids": ["S001", "S007", "S008"]})

    new_summary = runner.run_experiment(extended_config, resume=True)

    raw_lines = (experiment_dir / "raw_results.jsonl").read_text().strip().splitlines()
    assert len(raw_lines) == 9  # original 6 + 3 new (S008 x 3 architectures x 1 repetition)
    assert new_summary.recorded_runs == 9

    # No duplicate (architecture, scenario_id, repetition) keys.
    keys = [
        (json.loads(line)["architecture"], json.loads(line)["scenario_id"], json.loads(line)["repetition"])
        for line in raw_lines
    ]
    assert len(keys) == len(set(keys))


# ---------------------------------------------------------------------------
# Mock reproducibility across separate directories (Step 26)
# ---------------------------------------------------------------------------


def _strip_operational_fields(record: dict) -> dict:
    stripped = dict(record)
    stripped.pop("timestamp", None)
    stripped.pop("experiment_id", None)
    common = dict(stripped["common_result"])
    latency = {k: None for k in common["latency_metrics"]}
    common["latency_metrics"] = latency
    stripped["common_result"] = common
    # Architecture-specific results carry their own latency sub-fields too,
    # under architecture-specific names (e.g. tool's
    # llm_decision_latency_seconds/tool_execution_latency_seconds, hybrid's
    # rag_retrieval_latency_seconds/rag_llm_latency_seconds/etc. — see
    # src/evaluation/{rag,tool,hybrid}_agent_evaluation.py's Result models).
    # Match by suffix rather than a fixed name list so every architecture's
    # real timing fields are covered, not just the ones matching the common
    # metrics.py names by substring coincidence.
    arch_specific = dict(stripped["architecture_specific_result"])
    for key in list(arch_specific):
        if key.endswith("_latency_seconds") or key == "errors":
            arch_specific[key] = None
    stripped["architecture_specific_result"] = arch_specific
    return stripped


def test_two_independent_mock_runs_match_on_research_relevant_fields(tmp_path_factory):
    dir_a = tmp_path_factory.mktemp("repro_a")
    dir_b = tmp_path_factory.mktemp("repro_b")
    config_a = _config(dir_a, experiment_id="repro-a", scenario_ids=["S001", "S008"])
    config_b = _config(dir_b, experiment_id="repro-b", scenario_ids=["S001", "S008"])

    runner.run_experiment(config_a)
    runner.run_experiment(config_b)

    lines_a = (dir_a / "repro-a" / "raw_results.jsonl").read_text().strip().splitlines()
    lines_b = (dir_b / "repro-b" / "raw_results.jsonl").read_text().strip().splitlines()

    records_a = sorted(
        (_strip_operational_fields(json.loads(line)) for line in lines_a),
        key=lambda r: (r["architecture"], r["scenario_id"], r["repetition"]),
    )
    records_b = sorted(
        (_strip_operational_fields(json.loads(line)) for line in lines_b),
        key=lambda r: (r["architecture"], r["scenario_id"], r["repetition"]),
    )
    assert records_a == records_b


def test_reproducibility_covers_predictions_evaluation_and_failure_states(tmp_path_factory):
    dir_a = tmp_path_factory.mktemp("repro_c")
    dir_b = tmp_path_factory.mktemp("repro_d")
    config_a = _config(dir_a, experiment_id="repro-c", scenario_ids=["S001"], architectures=[ArchitectureType.TOOL])
    config_b = _config(dir_b, experiment_id="repro-d", scenario_ids=["S001"], architectures=[ArchitectureType.TOOL])

    summary_a = runner.run_experiment(config_a)
    summary_b = runner.run_experiment(config_b)

    result_a = summary_a.comparison.tool_summary.raw_results[0]
    result_b = summary_b.comparison.tool_summary.raw_results[0]

    assert result_a.predicted_best_sportsbooks == result_b.predicted_best_sportsbooks
    assert result_a.predicted_best_odds == result_b.predicted_best_odds
    assert result_a.execution_status == result_b.execution_status
    assert result_a.predicted_ev == result_b.predicted_ev
