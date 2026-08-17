"""Paired-observation alignment (Milestone 14B, Section 5).

The same scenario/repetition was run under all three architectures, so
architecture comparisons must be paired, not treated as independent
samples. Run-level pairing aligns by (scenario_id, repetition); the
consistency metric is itself computed across repetitions, so its
architecture-level comparison aligns by scenario_id alone.

Pure data-shaping only — no metric formulas live here.
"""

from __future__ import annotations

from typing import Callable, TypeVar

from src.experiments.runner import RawExperimentRun
from src.models import ArchitectureType

T = TypeVar("T")

RunKey = tuple[str, int]  # (scenario_id, repetition)


def index_by_architecture_and_key(
    runs: list[RawExperimentRun],
) -> dict[ArchitectureType, dict[RunKey, RawExperimentRun]]:
    index: dict[ArchitectureType, dict[RunKey, RawExperimentRun]] = {}
    for run in runs:
        index.setdefault(run.architecture, {})[(run.scenario_id, run.repetition)] = run
    return index


def align_pair(
    runs: list[RawExperimentRun],
    architecture_a: ArchitectureType,
    architecture_b: ArchitectureType,
    value_fn: Callable[[RawExperimentRun], T | None],
) -> tuple[list[tuple[T, T]], int]:
    """Returns (aligned_pairs, dropped_count). A (scenario_id,
    repetition) key contributes a pair only when BOTH architectures have
    a run for that key AND value_fn returns a non-None value for both —
    N/A values are dropped, never coerced to a fabricated default
    (Section 22 of Milestone 14A, carried forward here)."""
    index = index_by_architecture_and_key(runs)
    keys_a = set(index.get(architecture_a, {}))
    keys_b = set(index.get(architecture_b, {}))
    common_keys = sorted(keys_a & keys_b)

    pairs: list[tuple[T, T]] = []
    dropped = 0
    for key in common_keys:
        value_a = value_fn(index[architecture_a][key])
        value_b = value_fn(index[architecture_b][key])
        if value_a is None or value_b is None:
            dropped += 1
            continue
        pairs.append((value_a, value_b))
    return pairs, dropped


def align_three_way(
    runs: list[RawExperimentRun],
    value_fn: Callable[[RawExperimentRun], T | None],
    architectures: tuple[ArchitectureType, ArchitectureType, ArchitectureType] = (
        ArchitectureType.RAG, ArchitectureType.TOOL, ArchitectureType.HYBRID,
    ),
) -> tuple[list[tuple[T, T, T]], int]:
    """Same alignment rule as align_pair, but requires all three
    architectures to have a non-None value for the same
    (scenario_id, repetition) key — needed for the Friedman omnibus
    test (Section 11/12)."""
    index = index_by_architecture_and_key(runs)
    common_keys = sorted(
        set(index.get(architectures[0], {}))
        & set(index.get(architectures[1], {}))
        & set(index.get(architectures[2], {}))
    )

    triples: list[tuple[T, T, T]] = []
    dropped = 0
    for key in common_keys:
        values = tuple(value_fn(index[architecture][key]) for architecture in architectures)
        if any(v is None for v in values):
            dropped += 1
            continue
        triples.append(values)  # type: ignore[arg-type]
    return triples, dropped


def group_by_scenario(runs: list[RawExperimentRun]) -> dict[str, list[RawExperimentRun]]:
    groups: dict[str, list[RawExperimentRun]] = {}
    for run in runs:
        groups.setdefault(run.scenario_id, []).append(run)
    return groups


def group_by_architecture(runs: list[RawExperimentRun]) -> dict[ArchitectureType, list[RawExperimentRun]]:
    groups: dict[ArchitectureType, list[RawExperimentRun]] = {}
    for run in runs:
        groups.setdefault(run.architecture, []).append(run)
    return groups


def group_by_architecture_and_scenario(
    runs: list[RawExperimentRun],
) -> dict[ArchitectureType, dict[str, list[RawExperimentRun]]]:
    groups: dict[ArchitectureType, dict[str, list[RawExperimentRun]]] = {}
    for run in runs:
        groups.setdefault(run.architecture, {}).setdefault(run.scenario_id, []).append(run)
    return groups
