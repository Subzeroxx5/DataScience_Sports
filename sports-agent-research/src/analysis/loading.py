"""Loads a frozen Milestone 14A experiment directory for analysis
(Milestone 14B, Section 1-2). Read-only: never modifies raw_results.jsonl,
config.json, manifest.json, metadata.json, artifact_hashes.json, or
summary.json. Revalidates the dataset via the existing
src.experiments.validation module — never a second validation
implementation.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from src.experiments.config import ExperimentConfig, ExperimentMetadata, ExperimentScenario
from src.experiments.runner import RawExperimentRun
from src.experiments.validation import (
    DatasetValidationReport,
    load_and_validate_raw_results,
    validate_final_dataset,
)


class FinalDatasetInvalidError(Exception):
    """Raised when the frozen dataset fails revalidation — analysis must
    not proceed on an invalid dataset (Section 1: "If dataset validation
    fails: STOP. Do not perform statistical analysis.")."""


@dataclass
class LoadedFinalDataset:
    experiment_dir: Path
    config: ExperimentConfig
    manifest: list[ExperimentScenario]
    raw_runs: list[RawExperimentRun]
    validation_report: DatasetValidationReport


def load_final_dataset(experiment_dir: Path | str) -> LoadedFinalDataset:
    experiment_dir = Path(experiment_dir)

    metadata_path = experiment_dir / "config.json"
    manifest_path = experiment_dir / "manifest.json"
    raw_path = experiment_dir / "raw_results.jsonl"
    hashes_path = experiment_dir / "artifact_hashes.json"

    for path in (metadata_path, manifest_path, raw_path, hashes_path):
        if not path.is_file():
            raise FinalDatasetInvalidError(f"missing required file: {path}")

    metadata = ExperimentMetadata.model_validate_json(metadata_path.read_text())
    manifest = [ExperimentScenario.model_validate(item) for item in json.loads(manifest_path.read_text())]

    # Reuses the exact same parsing/error-collection src.experiments.
    # validation already uses for its own schema check — never a second,
    # weaker raw-line parser that could crash with an uncaught pydantic
    # exception instead of a clear FinalDatasetInvalidError.
    raw_runs, parse_errors = load_and_validate_raw_results(raw_path)
    if parse_errors:
        raise FinalDatasetInvalidError(
            f"frozen dataset at {experiment_dir} has malformed raw_results.jsonl record(s): {parse_errors}"
        )

    hashes = json.loads(hashes_path.read_text())
    pre_fingerprints = hashes.get("pre_run")
    post_fingerprints = hashes.get("post_run")

    report = validate_final_dataset(
        metadata.config, raw_path,
        pre_fingerprints=pre_fingerprints, post_fingerprints=post_fingerprints,
    )
    if not report.dataset_valid:
        raise FinalDatasetInvalidError(
            f"frozen dataset at {experiment_dir} failed revalidation: {report}"
        )

    return LoadedFinalDataset(
        experiment_dir=experiment_dir, config=metadata.config,
        manifest=manifest, raw_runs=raw_runs, validation_report=report,
    )
