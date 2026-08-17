"""Tests for the frozen final-experiment configuration (Milestone 14A,
Sections 2-5): experiments/final_experiment.json loads into a valid
ExperimentConfig, the expected run count matches the milestone's own
formula, every control is frozen at the documented default, and the
runner CLI's --config flag loads it correctly.
"""

import json
from pathlib import Path

from src.experiments.config import ExecutionMode, ExperimentConfig
from src.experiments.runner import _load_config_from_file
from src.models import ArchitectureType

FINAL_CONFIG_PATH = Path(__file__).resolve().parent.parent / "experiments" / "final_experiment.json"


def test_final_experiment_config_file_exists():
    assert FINAL_CONFIG_PATH.is_file()


def test_final_experiment_config_loads_as_valid_experiment_config():
    config = _load_config_from_file(str(FINAL_CONFIG_PATH))
    assert isinstance(config, ExperimentConfig)


def test_final_experiment_config_is_real_mode_not_mock():
    config = _load_config_from_file(str(FINAL_CONFIG_PATH))
    assert config.execution_mode == ExecutionMode.REAL


def test_final_experiment_expected_run_count_matches_architectures_x_scenarios_x_repetitions():
    config = _load_config_from_file(str(FINAL_CONFIG_PATH))
    assert config.expected_run_count() == len(config.architectures) * len(config.scenario_ids) * config.repetitions


def test_final_experiment_uses_all_three_architectures():
    config = _load_config_from_file(str(FINAL_CONFIG_PATH))
    assert set(config.architectures) == {ArchitectureType.RAG, ArchitectureType.TOOL, ArchitectureType.HYBRID}


def test_final_experiment_uses_the_local_ollama_provider():
    # Pre-Milestone 14A checkpoint, Step 11: the frozen config now
    # records the local provider so the final experiment can run
    # without per-call API cost — verified locally to support both
    # structured output and tool calling (see
    # experiments/run_ollama_three_architecture_smoke_test.py).
    from src.agents.llm_client import DEFAULT_OLLAMA_MODEL, LLMProviderName

    config = _load_config_from_file(str(FINAL_CONFIG_PATH))
    assert config.llm_provider == LLMProviderName.OLLAMA
    assert config.model_name == DEFAULT_OLLAMA_MODEL


def test_final_experiment_repetitions_is_ten_by_documented_default():
    config = _load_config_from_file(str(FINAL_CONFIG_PATH))
    assert config.repetitions == 10


def test_final_experiment_scenario_ids_are_unique_and_nonempty():
    config = _load_config_from_file(str(FINAL_CONFIG_PATH))
    assert config.scenario_ids
    assert len(set(config.scenario_ids)) == len(config.scenario_ids)


def test_final_experiment_temperature_is_frozen_and_explicit():
    # Pre-Milestone 14A checkpoint: the frozen config uses the local
    # Ollama provider, which DOES use `temperature` as its determinism
    # lever (DEFAULT_OLLAMA_TEMPERATURE=0.0) — unlike the Anthropic model
    # family this project used before, which rejects temperature
    # outright and relies on `effort` instead. Either way, the value
    # must be frozen and explicit, never left ambiguous.
    from src.agents.llm_client import DEFAULT_OLLAMA_TEMPERATURE, LLMProviderName

    config = _load_config_from_file(str(FINAL_CONFIG_PATH))
    assert config.llm_provider == LLMProviderName.OLLAMA
    assert config.temperature == DEFAULT_OLLAMA_TEMPERATURE
    assert config.effort  # inert for Ollama, but still present/frozen for documentation continuity


def test_final_experiment_execution_order_policy_is_the_milestone12_rotation():
    config = _load_config_from_file(str(FINAL_CONFIG_PATH))
    assert config.execution_order_policy == "rotating"


def test_final_experiment_output_directory_uses_final_prefixed_experiment_id():
    config = _load_config_from_file(str(FINAL_CONFIG_PATH))
    assert config.experiment_id.startswith("final")


def test_final_experiment_config_json_has_notes_with_rationale():
    payload = json.loads(FINAL_CONFIG_PATH.read_text())
    assert "notes" in payload
    assert "repetition_rationale" in payload["notes"]
    assert "scenario_set_rationale" in payload["notes"]


def test_final_experiment_config_json_declares_expected_run_count_consistently():
    payload = json.loads(FINAL_CONFIG_PATH.read_text())
    config = ExperimentConfig.model_validate(payload["config"])
    assert payload["expected_run_count"] == config.expected_run_count()


def test_runner_cli_loads_final_config_via_config_flag(capsys):
    from src.experiments.runner import main

    main(["--config", str(FINAL_CONFIG_PATH), "--dry-run"])
    output = capsys.readouterr().out
    assert "Expected runs: " + str(_load_config_from_file(str(FINAL_CONFIG_PATH)).expected_run_count()) in output


def test_config_flag_overrides_other_cli_args(capsys):
    """--config is authoritative — it must not be silently combined with
    --repetitions/--scenario-ids/etc. from the same invocation."""
    from src.experiments.runner import main

    main(["--config", str(FINAL_CONFIG_PATH), "--repetitions", "1", "--dry-run"])
    output = capsys.readouterr().out
    assert "Repetitions: 10" in output  # the frozen file's value, not the CLI flag's
