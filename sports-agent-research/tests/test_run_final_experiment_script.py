"""Tests for experiments/run_final_experiment.py (Milestone 14A, Section
28: "mock/final directory separation"). Runs the script as a subprocess
with --skip-preflight (the slow, pytest-subprocess-invoking preflight
check is already covered by tests/test_final_experiment_preflight.py)
so these stay fast. No real LLM calls are made or required — both
scenarios below are refused before any network call happens.
"""

import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "experiments" / "run_final_experiment.py"


def _run_script(config_path: Path, extra_args: list[str] | None = None) -> subprocess.CompletedProcess:
    args = [sys.executable, str(SCRIPT), "--config", str(config_path), "--skip-preflight"]
    if extra_args:
        args.extend(extra_args)
    return subprocess.run(args, capture_output=True, text=True, cwd=REPO_ROOT)


def _write_config(tmp_path: Path, output_dir: Path, execution_mode: str, experiment_id: str) -> Path:
    payload = {
        "config": {
            "experiment_id": experiment_id,
            "experiment_name": "script-test",
            "architectures": ["tool"],
            "scenario_ids": ["S001"],
            "repetitions": 1,
            "execution_mode": execution_mode,
            "output_dir": str(output_dir),
        }
    }
    config_path = tmp_path / "test_config.json"
    config_path.write_text(json.dumps(payload))
    return config_path


def test_script_refuses_mock_mode_for_final_dataset(tmp_path):
    output_dir = tmp_path / "results"
    config_path = _write_config(tmp_path, output_dir, "mock", "script-test-mock")

    result = _run_script(config_path)

    assert result.returncode != 0
    assert "REFUSING TO PROCEED" in result.stdout
    assert not output_dir.exists()  # nothing written — mock must never become the final dataset


def test_script_prints_frozen_configuration_before_refusing(tmp_path):
    output_dir = tmp_path / "results"
    config_path = _write_config(tmp_path, output_dir, "mock", "script-test-mock-2")

    result = _run_script(config_path)

    assert "FINAL FROZEN CONFIGURATION" in result.stdout
    assert "Mode: MOCK" in result.stdout


def test_script_real_mode_without_credentials_reports_not_run_and_writes_nothing(tmp_path, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    output_dir = tmp_path / "results"
    config_path = _write_config(tmp_path, output_dir, "real", "script-test-real")

    result = _run_script(config_path)

    assert result.returncode != 0
    assert "REAL EXPERIMENT: NOT RUN" in result.stdout
    assert not output_dir.exists()  # no partial experiment directory left behind


def test_script_never_silently_falls_back_to_mock_when_real_is_requested(tmp_path, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    output_dir = tmp_path / "results"
    config_path = _write_config(tmp_path, output_dir, "real", "script-test-real-2")

    result = _run_script(config_path)

    assert "MOCK" not in result.stdout.split("REAL EXPERIMENT: NOT RUN")[-1]
