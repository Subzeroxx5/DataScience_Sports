"""Tests for src/experiments/fingerprint.py (Milestone 14A, Sections
7-9): artifact/prompt fingerprints, pre/post comparison, and environment
metadata capture. No real LLM calls."""

from pathlib import Path

from src.experiments import fingerprint


def test_fingerprints_cover_every_expected_controlled_artifact():
    fingerprints = fingerprint.compute_final_experiment_fingerprints()
    expected_keys = {
        "test_scenarios", "ground_truth", "quant_ground_truth", "rag_corpus", "rag_index_config",
        "rag_extraction_system_prompt", "tool_agent_system_prompt",
    }
    assert expected_keys <= set(fingerprints.keys())


def test_fingerprints_are_sha256_prefixed():
    fingerprints = fingerprint.compute_final_experiment_fingerprints()
    for value in fingerprints.values():
        assert value == "missing" or value.startswith("sha256:")


def test_fingerprints_are_stable_across_calls():
    assert fingerprint.compute_final_experiment_fingerprints() == fingerprint.compute_final_experiment_fingerprints()


def test_fingerprints_include_config_hash_when_path_given(tmp_path):
    config_path = tmp_path / "cfg.json"
    config_path.write_text('{"a": 1}')
    fingerprints = fingerprint.compute_final_experiment_fingerprints(config_path)
    assert "final_experiment_config" in fingerprints
    assert fingerprints["final_experiment_config"].startswith("sha256:")


def test_fingerprints_omit_config_hash_when_no_path_given():
    fingerprints = fingerprint.compute_final_experiment_fingerprints()
    assert "final_experiment_config" not in fingerprints


def test_missing_artifact_is_recorded_as_missing_not_an_exception(tmp_path, monkeypatch):
    monkeypatch.setitem(fingerprint.FINAL_EXPERIMENT_ARTIFACTS, "test_scenarios", tmp_path / "does_not_exist.json")
    fingerprints = fingerprint.compute_final_experiment_fingerprints()
    assert fingerprints["test_scenarios"] == "missing"


def test_compare_fingerprints_detects_match():
    before = {"a": "sha256:aaa", "b": "sha256:bbb"}
    after = {"a": "sha256:aaa", "b": "sha256:bbb"}
    assert fingerprint.compare_fingerprints(before, after) == {"a": True, "b": True}


def test_compare_fingerprints_detects_mismatch():
    before = {"a": "sha256:aaa"}
    after = {"a": "sha256:changed"}
    assert fingerprint.compare_fingerprints(before, after) == {"a": False}


def test_compare_fingerprints_missing_key_in_after_counts_as_changed():
    before = {"a": "sha256:aaa"}
    after = {}
    assert fingerprint.compare_fingerprints(before, after) == {"a": False}


def test_environment_metadata_has_required_fields():
    metadata = fingerprint.capture_environment_metadata()
    for key in ("python_version", "platform", "machine", "package_versions", "git_commit_hash", "git_status", "git_diff_stat", "git_working_tree_clean"):
        assert key in metadata


def test_environment_metadata_never_invents_a_clean_git_state():
    """git_working_tree_clean must reflect the ACTUAL git status output,
    never be hardcoded True — this repo has uncommitted changes as of
    Milestone 14A, so this must currently be False. If the tree is ever
    genuinely clean, git_status will be empty and this becomes True —
    the assertion checks the *derivation*, not a fixed expectation."""
    metadata = fingerprint.capture_environment_metadata()
    assert metadata["git_working_tree_clean"] == (metadata["git_status"] == "")


def test_environment_metadata_package_versions_are_nonempty_strings():
    metadata = fingerprint.capture_environment_metadata()
    for name, version in metadata["package_versions"].items():
        assert isinstance(version, str) and version
