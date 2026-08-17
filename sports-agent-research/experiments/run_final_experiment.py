"""Final controlled experiment orchestrator (Milestone 14A).

Manual, credential-gated real-API script — mirrors the existing
convention in this directory (run_rag_smoke_test.py,
run_hybrid_agent_real_llm_evaluation.py, ...): not part of the automated
pytest suite, never required for a milestone's test baseline.

Wraps src.experiments.runner.run_experiment() (Milestone 12, unmodified
execution engine — this script never reimplements the run loop) with
everything Milestone 14A adds around it:

    load frozen config (experiments/final_experiment.json)
        |
        v
    preflight checks (src.experiments.preflight)  -- abort if any fail
        |
        v
    pre-run fingerprints + environment metadata (src.experiments.fingerprint)
        |
        v
    connectivity probe (src.experiments.runner.probe_real_llm_connectivity)
        |
        v
    run_experiment()                              -- Milestone 12, unmodified
        |
        v
    post-run fingerprints -> compare to pre-run
        |
        v
    dataset validation report (src.experiments.validation)

Usage:

    python experiments/run_final_experiment.py \
        --config experiments/final_experiment.json

Requires ANTHROPIC_API_KEY to be set (execution_mode=real in the frozen
config). If it is not set, this prints "REAL EXPERIMENT: NOT RUN" and
exits cleanly — it never silently substitutes a mock run for a final
research dataset.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from src.experiments import fingerprint, preflight, validation  # noqa: E402
from src.experiments.config import ExecutionMode  # noqa: E402
from src.experiments.runner import (  # noqa: E402
    _experiment_dir,
    probe_real_llm_connectivity,
    run_experiment,
)
from src.experiments.runner import _load_config_from_file as load_config_from_file  # noqa: E402


def _print_frozen_configuration(config, expected_runs: int) -> None:
    print("FINAL FROZEN CONFIGURATION")
    print(f"Experiment ID: {config.experiment_id}")
    print(f"Mode: {config.execution_mode.value.upper()}")
    print()
    print("Architectures:")
    for architecture in config.architectures:
        print(f"- {architecture.value.upper()}")
    print()
    print(f"Model: {config.model_name}")
    print(f"Temperature: {config.temperature}")
    print(f"RAG top_k: {config.rag_top_k}")
    print(f"Tool iteration limit: {config.max_tool_iterations}")
    print()
    print(f"Scenarios: {len(config.scenario_ids)} ({config.scenario_ids})")
    print(f"Repetitions: {config.repetitions}")
    print()
    print(f"Expected total runs: {expected_runs}")
    print(f"Execution-order policy: {config.execution_order_policy}")
    print()


def _print_dataset_validation(report: validation.DatasetValidationReport) -> None:
    print()
    print("FINAL DATASET VALIDATION")
    print(f"Expected runs: {report.expected_runs}")
    print(f"Recorded runs: {report.recorded_runs}")
    print()
    for architecture, count in sorted(report.architecture_run_counts.items()):
        print(f"{architecture.upper()} runs: {count}")
    print()
    print(f"Successful: {report.successful_runs}")
    print(f"Failed: {report.failed_runs}")
    print()
    print(f"Duplicate run keys: {report.duplicate_run_keys or 'none'}")
    print(f"Missing run keys: {report.missing_run_keys or 'none'}")
    print()
    print(f"Scenario coverage: {'PASS' if report.scenario_coverage_pass else 'FAIL'}")
    print(f"Architecture coverage: {'PASS' if report.architecture_coverage_pass else 'FAIL'}")
    print(f"Ground-truth isolation: {'PASS' if report.ground_truth_isolation_pass else 'FAIL'}")
    print(f"Architecture isolation: {'PASS' if report.architecture_isolation_pass else 'FAIL'}")
    print()
    for name, matched in sorted(report.pre_post_hash_matches.items()):
        print(f"Pre/post {name} hash: {'MATCH' if matched else 'MISMATCH'}")
    print()
    print(f"Raw dataset validation: {'PASS' if report.schema_validation_pass else 'FAIL'}")
    if report.schema_validation_errors:
        for error in report.schema_validation_errors:
            print(f"  - {error}")
    print()
    print(f"OVERALL DATASET VALID: {'YES' if report.dataset_valid else 'NO'}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Final controlled experiment orchestrator (Milestone 14A)")
    parser.add_argument("--config", default="experiments/final_experiment.json")
    parser.add_argument("--skip-preflight", action="store_true", help="for testing only — never use for a real final run")
    args = parser.parse_args(argv)

    config = load_config_from_file(args.config)
    expected_runs = config.expected_run_count()
    _print_frozen_configuration(config, expected_runs)

    if config.execution_mode != ExecutionMode.REAL:
        print("REFUSING TO PROCEED: final research dataset requires execution_mode=real "
              f"(config has {config.execution_mode.value!r}). Mock runs are infrastructure "
              "validation only and must never be presented as the final dataset (Section 10).")
        return 1

    if not args.skip_preflight:
        print("Running preflight checks (Section 6)...")
        preflight_report = preflight.run_preflight_checks(config)
        for name, passed in preflight_report.checks.items():
            print(f"  [{'PASS' if passed else 'FAIL'}] {name}")
            if not passed and name in preflight_report.details:
                print(f"        {preflight_report.details[name]}")
        if not preflight_report.all_passed:
            print()
            print("PREFLIGHT FAILED — final experiment will NOT be started.")
            return 1
        print("All preflight checks passed.")
        print()

    ok, error = probe_real_llm_connectivity(config)
    if not ok:
        print("REAL EXPERIMENT: NOT RUN")
        print(f"Could not reach the real Anthropic API — is ANTHROPIC_API_KEY set? ({error})")
        return 1

    experiment_dir = _experiment_dir(config)
    experiment_dir.mkdir(parents=True, exist_ok=True)

    pre_fingerprints = fingerprint.compute_final_experiment_fingerprints(args.config)
    environment_metadata = fingerprint.capture_environment_metadata()
    (experiment_dir / "artifact_hashes.json").write_text(
        json.dumps({"pre_run": pre_fingerprints}, indent=2, sort_keys=True)
    )
    (experiment_dir / "metadata.json").write_text(
        json.dumps({"environment": environment_metadata}, indent=2, sort_keys=True)
    )

    print(f"Starting final experiment run: {expected_runs} expected raw runs -> {experiment_dir}")
    summary = run_experiment(config)
    print(f"Run complete: {summary.recorded_runs} recorded, {summary.successful_runs} successful, "
          f"{summary.failed_runs} failed.")

    post_fingerprints = fingerprint.compute_final_experiment_fingerprints(args.config)
    hashes_payload = json.loads((experiment_dir / "artifact_hashes.json").read_text())
    hashes_payload["post_run"] = post_fingerprints
    (experiment_dir / "artifact_hashes.json").write_text(json.dumps(hashes_payload, indent=2, sort_keys=True))

    report = validation.validate_final_dataset(
        config, experiment_dir / "raw_results.jsonl",
        pre_fingerprints=pre_fingerprints, post_fingerprints=post_fingerprints,
    )
    (experiment_dir / "descriptive_summary.json").write_text(summary.model_dump_json(indent=2))
    (experiment_dir / "dataset_validation_report.json").write_text(
        json.dumps(
            {
                "experiment_id": report.experiment_id, "mode": report.mode, "model": report.model,
                "temperature": report.temperature, "scenario_count": report.scenario_count,
                "repetitions": report.repetitions, "expected_runs": report.expected_runs,
                "recorded_runs": report.recorded_runs, "architecture_run_counts": report.architecture_run_counts,
                "successful_runs": report.successful_runs, "failed_runs": report.failed_runs,
                "duplicate_run_keys": report.duplicate_run_keys, "missing_run_keys": report.missing_run_keys,
                "scenario_coverage": report.scenario_coverage,
                "scenario_coverage_pass": report.scenario_coverage_pass,
                "architecture_coverage_pass": report.architecture_coverage_pass,
                "schema_validation_pass": report.schema_validation_pass,
                "schema_validation_errors": report.schema_validation_errors,
                "ground_truth_isolation_pass": report.ground_truth_isolation_pass,
                "architecture_isolation_pass": report.architecture_isolation_pass,
                "pre_post_hash_matches": report.pre_post_hash_matches,
                "dataset_valid": report.dataset_valid,
            },
            indent=2, sort_keys=True,
        )
    )

    _print_dataset_validation(report)
    return 0 if report.dataset_valid else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
