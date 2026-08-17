"""Hallucination / unsupported-claim analysis (Milestone 14B, Section 14).

Preserves the Milestone 11 definition (unsupported_claim_count /
total_verifiable_claims, hallucination_detected) exactly —
metrics.unsupported_claim_rate() is reused, never redefined. Per-type
fabrication breakdown (fabricated sportsbook / odds / provenance /
other) is reported only to the extent the persisted schema actually
distinguishes it; deterministic calculation errors are never persisted
under hallucination_detected in the first place (Milestone 8B-11's own
provenance-validation design keeps those separate — validated here by
construction, not reclassified).
"""

from __future__ import annotations

from dataclasses import dataclass

from src.evaluation import metrics
from src.experiments.runner import RawExperimentRun


@dataclass
class HallucinationArchitectureStats:
    architecture: str
    total_verifiable_claims: int
    total_unsupported_claims: int
    unsupported_claim_rate: float | None
    runs_with_hallucination_detected: int
    runs_with_at_least_one_unsupported_claim: int


def hallucination_stats_by_architecture(
    architecture: str, runs: list[RawExperimentRun],
) -> HallucinationArchitectureStats:
    total_claims = sum(run.common_result.total_verifiable_claims for run in runs)
    total_unsupported = sum(run.common_result.unsupported_claim_count for run in runs)
    rate = metrics.unsupported_claim_rate(total_unsupported, total_claims)

    return HallucinationArchitectureStats(
        architecture=architecture,
        total_verifiable_claims=total_claims,
        total_unsupported_claims=total_unsupported,
        unsupported_claim_rate=rate,
        runs_with_hallucination_detected=sum(1 for run in runs if run.common_result.hallucination_detected),
        runs_with_at_least_one_unsupported_claim=sum(
            1 for run in runs if run.common_result.unsupported_claim_count >= 1
        ),
    )
