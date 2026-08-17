"""Artifact/environment fingerprinting for the final controlled
experiment (Milestone 14A).

Independent of, and never modifies, Milestone 12's
`compute_artifact_checksums()` (src/experiments/config.py) — that
function's exact artifact set and dict-key contract is covered by an
existing Milestone 12 test (`test_checksums_cover_expected_artifacts`)
and must not change. This module adds the additional inputs Milestone
14A's pre/post-run integrity check needs (the benchmark scenario
definitions, the frozen final-experiment config file itself, and the
canonical system-prompt constants — Section 2/7/24 of the milestone),
as new, separate functionality.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import platform
import subprocess
import sys
from pathlib import Path

from src.agents.extraction import RAG_EXTRACTION_SYSTEM_PROMPT
from src.agents.tool_agent import TOOL_AGENT_SYSTEM_PROMPT
from src.evaluation.dataset import DATA_DIR

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent

# The controlled inputs a final experiment's validity depends on beyond
# what Milestone 12 already checksums for its own metadata (current
# odds, ground truth, quant ground truth, RAG corpus, RAG index config —
# see src/experiments/config.py's _CHECKSUM_ARTIFACTS). test_scenarios.json
# is the one benchmark artifact Milestone 12 did not need to fingerprint
# (its own manifest is built from it at run time) but Milestone 14A's
# Section 7/24 explicitly requires.
FINAL_EXPERIMENT_ARTIFACTS: dict[str, Path] = {
    "test_scenarios": DATA_DIR / "test_scenarios.json",
    "ground_truth": DATA_DIR / "ground_truth.json",
    "quant_ground_truth": DATA_DIR / "quant_ground_truth.json",
    "rag_corpus": DATA_DIR / "rag_documents" / "corpus.jsonl",
    "rag_index_config": DATA_DIR / "rag_index" / "config.json",
    # The manifest.json every run writes is generated deterministically
    # from test_scenarios.json (already hashed above) via
    # _canonical_query()/build_scenario_manifest() in this file — hashing
    # the module itself catches a change to the canonical query wording
    # that a data-only hash could never detect (Section 9: "canonical
    # scenario manifest/query file").
    "experiment_config_module": _REPO_ROOT / "src" / "experiments" / "config.py",
}


def _sha256_file(path: Path) -> str:
    if not path.is_file():
        return "missing"
    return f"sha256:{hashlib.sha256(path.read_bytes()).hexdigest()}"


def _sha256_text(text: str) -> str:
    return f"sha256:{hashlib.sha256(text.encode('utf-8')).hexdigest()}"


def compute_final_experiment_fingerprints(config_path: Path | str | None = None) -> dict[str, str]:
    """Sha256 fingerprints of every controlled input the final
    experiment's validity depends on (Section 7): the benchmark scenario
    definitions, both ground-truth files, the RAG corpus and its index
    configuration, the canonical system prompts (Section 2: "canonical
    prompt/template versions or hashes"), and — when a path is given —
    the frozen experiment config file itself. Comparing this dict before
    and after the run (Section 24) is how a mid-experiment change to any
    of these is detected; a changed or newly-"missing" value for a key
    present in the "before" snapshot means the final dataset is invalid.
    """
    fingerprints = {name: _sha256_file(path) for name, path in FINAL_EXPERIMENT_ARTIFACTS.items()}
    fingerprints["rag_extraction_system_prompt"] = _sha256_text(RAG_EXTRACTION_SYSTEM_PROMPT)
    fingerprints["tool_agent_system_prompt"] = _sha256_text(TOOL_AGENT_SYSTEM_PROMPT)
    if config_path is not None:
        fingerprints["final_experiment_config"] = _sha256_file(Path(config_path))
    return fingerprints


def compare_fingerprints(before: dict[str, str], after: dict[str, str]) -> dict[str, bool]:
    """name -> True if unchanged between the two snapshots. Every key
    present in `before` is checked against `after`; a key missing from
    `after` counts as changed (False), never silently skipped."""
    return {name: after.get(name) == value for name, value in before.items()}


def _pip_package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not installed"


def _git(*args: str) -> str:
    try:
        result = subprocess.run(
            ["git", *args], capture_output=True, text=True, check=True, cwd=_REPO_ROOT,
        )
        return result.stdout.strip()
    except Exception as exc:  # git unavailable, not a repo, etc. — record honestly
        return f"unavailable ({exc!r})"


def capture_environment_metadata() -> dict:
    """Section 8/9: reproducibility metadata. Never invents a clean git
    state — records whatever `git status --short` / `git diff --stat`
    actually show. Package versions come from the environment actually
    running the experiment, not from requirements.txt (which only pins
    minimum intent, not what is actually installed)."""
    git_status = _git("status", "--short")
    return {
        "python_version": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "package_versions": {
            "pydantic": _pip_package_version("pydantic"),
            "anthropic": _pip_package_version("anthropic"),
            "sentence-transformers": _pip_package_version("sentence-transformers"),
            "faiss-cpu": _pip_package_version("faiss-cpu"),
            "streamlit": _pip_package_version("streamlit"),
            "numpy": _pip_package_version("numpy"),
            "pandas": _pip_package_version("pandas"),
            "pytest": _pip_package_version("pytest"),
        },
        "git_commit_hash": _git("rev-parse", "HEAD"),
        "git_status": git_status,
        "git_diff_stat": _git("diff", "--stat"),
        "git_working_tree_clean": git_status == "",
    }
