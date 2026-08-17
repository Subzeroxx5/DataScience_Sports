"""Architecture-specific latency breakdown (Milestone 14B, Section 17).

Total latency is already covered by src.analysis.descriptive (reusing
the common metrics.LatencyMetrics field). This module additionally
preserves each architecture's own phase-level timing — retrieval/LLM/
quant for RAG, LLM/tool/quant for TOOL, retrieval/LLM/tool/quant for
HYBRID — read directly from the already-persisted
architecture_specific_result (the full per-architecture evaluator
Result, Milestone 9B-11), never recomputed.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.evaluation import metrics
from src.experiments.runner import RawExperimentRun
from src.models import ArchitectureType

_PHASE_FIELDS: dict[ArchitectureType, dict[str, str]] = {
    ArchitectureType.RAG: {
        "retrieval": "retrieval_latency_seconds",
        "llm": "llm_latency_seconds",
        "quant": "quant_latency_seconds",
    },
    ArchitectureType.TOOL: {
        "llm": "llm_decision_latency_seconds",
        "tool": "tool_execution_latency_seconds",
        "quant": "quant_latency_seconds",
    },
    ArchitectureType.HYBRID: {
        "rag_retrieval": "rag_retrieval_latency_seconds",
        "rag_llm": "rag_llm_latency_seconds",
        "tool_llm": "tool_llm_latency_seconds",
        "tool_execution": "tool_execution_latency_seconds",
        "reconciliation": "reconciliation_latency_seconds",
        "quant": "quant_latency_seconds",
    },
}


@dataclass
class PhaseLatencyStats:
    phase: str
    mean: float | None
    median: float | None


def architecture_phase_latency(
    architecture: ArchitectureType, runs: list[RawExperimentRun],
) -> list[PhaseLatencyStats]:
    phase_fields = _PHASE_FIELDS.get(architecture, {})
    stats = []
    for phase, field_name in phase_fields.items():
        values = [
            run.architecture_specific_result.get(field_name)
            for run in runs
            if run.architecture_specific_result.get(field_name) is not None
        ]
        stats.append(PhaseLatencyStats(phase=phase, mean=metrics.mean(values), median=metrics.median(values)))
    return stats
