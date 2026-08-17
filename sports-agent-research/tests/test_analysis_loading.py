"""Tests for src/analysis/loading.py (Milestone 14B, Sections 1-2, 32):
dataset revalidation must STOP (raise) on an invalid/incomplete/tampered
dataset rather than silently proceeding, and loading (and by extension
running the full analysis) must never modify the raw dataset files.
Uses a small synthetic MOCK experiment as a fixture — never the actual
final experiment result.
"""

import json

import pytest

from src.analysis.loading import FinalDatasetInvalidError, load_final_dataset
from src.experiments import runner as experiment_runner
from src.experiments.config import ExperimentConfig
from src.models import ArchitectureType


def _config(output_dir, **overrides) -> ExperimentConfig:
    defaults = dict(
        experiment_id="analysis-fixture",
        experiment_name="analysis-fixture",
        architectures=[ArchitectureType.RAG, ArchitectureType.TOOL, ArchitectureType.HYBRID],
        scenario_ids=["S001", "S007"],
        repetitions=2,
        output_dir=str(output_dir),
    )
    defaults.update(overrides)
    return ExperimentConfig(**defaults)


@pytest.fixture(scope="module")
def small_experiment_dir(tmp_path_factory):
    output_dir = tmp_path_factory.mktemp("analysis_loading_fixture")
    config = _config(output_dir)
    experiment_runner.run_experiment(config)
    experiment_dir = output_dir / config.experiment_id

    # This fixture is a MOCK experiment used purely to exercise the
    # loading/revalidation machinery — it has no artifact_hashes.json
    # pre/post snapshot the way a real Milestone 14A run does, so build
    # a minimal one so validate_final_dataset() has something to compare.
    from src.experiments import fingerprint

    fp = fingerprint.compute_final_experiment_fingerprints()
    (experiment_dir / "artifact_hashes.json").write_text(
        json.dumps({"pre_run": fp, "post_run": fp})
    )
    return experiment_dir


# ---------------------------------------------------------------------------
# Successful load
# ---------------------------------------------------------------------------


def test_load_final_dataset_succeeds_for_a_valid_experiment(small_experiment_dir):
    dataset = load_final_dataset(small_experiment_dir)
    assert len(dataset.raw_runs) == 12  # 3 architectures x 2 scenarios x 2 repetitions
    assert dataset.validation_report.dataset_valid is True


# ---------------------------------------------------------------------------
# STOP behavior — Section 1: "If dataset validation fails: STOP."
# ---------------------------------------------------------------------------


def test_load_final_dataset_raises_for_missing_directory(tmp_path):
    with pytest.raises(FinalDatasetInvalidError):
        load_final_dataset(tmp_path / "does_not_exist")


def test_load_final_dataset_raises_for_missing_required_file(tmp_path):
    experiment_dir = tmp_path / "partial"
    experiment_dir.mkdir()
    (experiment_dir / "config.json").write_text("{}")
    # manifest.json / raw_results.jsonl / artifact_hashes.json missing
    with pytest.raises(FinalDatasetInvalidError):
        load_final_dataset(experiment_dir)


def test_load_final_dataset_raises_when_artifact_hash_mismatches(tmp_path, small_experiment_dir):
    import shutil

    tampered_dir = tmp_path / "tampered"
    shutil.copytree(small_experiment_dir, tampered_dir)

    hashes = json.loads((tampered_dir / "artifact_hashes.json").read_text())
    hashes["post_run"] = {key: "sha256:tampered" for key in hashes["pre_run"]}
    (tampered_dir / "artifact_hashes.json").write_text(json.dumps(hashes))

    with pytest.raises(FinalDatasetInvalidError):
        load_final_dataset(tampered_dir)


def test_load_final_dataset_raises_when_a_raw_result_line_is_corrupted(tmp_path, small_experiment_dir):
    import shutil

    corrupted_dir = tmp_path / "corrupted"
    shutil.copytree(small_experiment_dir, corrupted_dir)

    raw_path = corrupted_dir / "raw_results.jsonl"
    lines = raw_path.read_text().strip().splitlines()
    lines[0] = "{not valid json"
    raw_path.write_text("\n".join(lines) + "\n")

    with pytest.raises(FinalDatasetInvalidError):
        load_final_dataset(corrupted_dir)


# ---------------------------------------------------------------------------
# Raw dataset must remain unchanged (Section 2, 32)
# ---------------------------------------------------------------------------


def test_loading_never_modifies_the_raw_dataset_files(small_experiment_dir):
    files = ["raw_results.jsonl", "config.json", "manifest.json", "artifact_hashes.json"]
    before = {name: (small_experiment_dir / name).read_bytes() for name in files}
    before_mtimes = {name: (small_experiment_dir / name).stat().st_mtime_ns for name in files}

    load_final_dataset(small_experiment_dir)
    load_final_dataset(small_experiment_dir)  # loading twice must still be read-only

    after = {name: (small_experiment_dir / name).read_bytes() for name in files}
    after_mtimes = {name: (small_experiment_dir / name).stat().st_mtime_ns for name in files}

    assert before == after
    assert before_mtimes == after_mtimes


def test_full_analysis_pipeline_never_modifies_the_raw_dataset_files(small_experiment_dir):
    from src.analysis.final_analysis import run_final_analysis

    files = ["raw_results.jsonl", "config.json", "manifest.json", "artifact_hashes.json"]
    before = {name: (small_experiment_dir / name).read_bytes() for name in files}

    run_final_analysis(small_experiment_dir)

    after = {name: (small_experiment_dir / name).read_bytes() for name in files}
    assert before == after
