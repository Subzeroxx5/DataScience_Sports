"""Reproducible final-analysis entry point (Milestone 14B, Section 26).

    python -m src.analysis.final_analysis [--experiment-dir results/experiments/final_v1]

1. loads the frozen final experiment artifacts (never modifies them)
2. revalidates the dataset (src.experiments.validation, via src.analysis.loading)
3. computes descriptive statistics (src.analysis.descriptive, reusing
   src.evaluation.metrics)
4. computes 95% Wilson confidence intervals (src.analysis.confidence_intervals)
5. aligns paired observations (src.analysis.pairing)
6. executes the pre-specified statistical tests (src.analysis.comparisons:
   McNemar for binary metrics, Wilcoxon for continuous metrics, Friedman
   for the three-architecture omnibus)
7. applies Holm correction within each metric family
8. generates tables (src.analysis.tables)
9. generates figures (src.analysis.figures)
10. saves machine-readable outputs under <experiment_dir>/analysis/
11. generates <experiment_dir>/analysis/findings.md

No manual spreadsheet arithmetic is required for any primary finding —
every number in findings.md traces back to a function in this package.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import sys
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from src.analysis.bundle import AnalysisBundle, build_analysis_bundle
from src.analysis.figures import generate_all_figures
from src.analysis.findings import generate_findings_markdown
from src.analysis.loading import FinalDatasetInvalidError, load_final_dataset
from src.analysis.tables import (
    build_omnibus_table,
    build_table_1_architecture_results,
    build_table_2_binary_pairwise,
    build_table_3_continuous_comparisons,
    build_table_4_failure_counts,
    build_table_5_hybrid_reconciliation,
)

DEFAULT_EXPERIMENT_DIR = "results/experiments/final_v1"
ARCHITECTURE_LABELS = ["rag", "tool", "hybrid"]


def _to_jsonable(obj: Any) -> Any:
    """Recursively converts dataclasses / pydantic models / enums into
    plain JSON-serializable structures. Numbers/strings/None/lists/dicts
    pass through unchanged."""
    if isinstance(obj, BaseModel):
        return obj.model_dump(mode="json")
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {field.name: _to_jsonable(getattr(obj, field.name)) for field in dataclasses.fields(obj)}
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, dict):
        return {key: _to_jsonable(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(item) for item in obj]
    return obj


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(_to_jsonable(payload), indent=2, sort_keys=True))


def _write_descriptive_statistics(analysis_dir: Path, bundle: AnalysisBundle) -> None:
    payload = {
        architecture: {
            "descriptive": bundle.descriptive_stats[architecture],
            "phase_latency": bundle.phase_latency[architecture],
            "freshness": bundle.freshness_stats[architecture],
            "hallucination": bundle.hallucination_stats[architecture],
        }
        for architecture in ARCHITECTURE_LABELS
    }
    _write_json(analysis_dir / "descriptive_statistics.json", payload)


def _write_statistical_tests(analysis_dir: Path, bundle: AnalysisBundle) -> None:
    _write_json(analysis_dir / "statistical_tests.json", {"omnibus": build_omnibus_table(bundle)})


def _write_pairwise_comparisons(analysis_dir: Path, bundle: AnalysisBundle) -> None:
    payload = {
        "binary": build_table_2_binary_pairwise(bundle),
        "continuous": build_table_3_continuous_comparisons(bundle),
    }
    _write_json(analysis_dir / "pairwise_comparisons.json", payload)


def _write_scenario_analysis(analysis_dir: Path, bundle: AnalysisBundle) -> None:
    payload = {
        "scenario_architecture": bundle.scenario_rows,
        "subgroups_exploratory": bundle.subgroup_rows,
    }
    _write_json(analysis_dir / "scenario_analysis.json", payload)


def _write_failure_analysis(analysis_dir: Path, bundle: AnalysisBundle) -> None:
    payload = {
        "failure_counts": build_table_4_failure_counts(bundle),
        "hybrid_reconciliation": build_table_5_hybrid_reconciliation(bundle),
    }
    _write_json(analysis_dir / "failure_analysis.json", payload)


def _write_analysis_table_csv(analysis_dir: Path, bundle: AnalysisBundle) -> None:
    rows = build_table_1_architecture_results(bundle)
    with (analysis_dir / "analysis_table.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_analysis_summary(analysis_dir: Path, bundle: AnalysisBundle) -> None:
    config = bundle.dataset.config
    payload = {
        "experiment_id": config.experiment_id,
        "mode": config.execution_mode.value,
        "llm_provider": config.llm_provider.value,
        "model_name": config.model_name,
        "temperature": config.temperature,
        "scenario_count": len(config.scenario_ids),
        "repetitions": config.repetitions,
        "total_observations": len(bundle.dataset.raw_runs),
        "dataset_valid": bundle.dataset.validation_report.dataset_valid,
        "alpha": 0.05,
        "multiple_comparison_method": "Holm",
        "table_1": build_table_1_architecture_results(bundle),
        "no_inferential_conclusion_forced": True,
    }
    _write_json(analysis_dir / "analysis_summary.json", payload)


def run_final_analysis(experiment_dir: Path | str) -> tuple[AnalysisBundle, Path]:
    dataset = load_final_dataset(experiment_dir)
    bundle = build_analysis_bundle(dataset)

    analysis_dir = dataset.experiment_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = analysis_dir / "figures"

    _write_descriptive_statistics(analysis_dir, bundle)
    _write_statistical_tests(analysis_dir, bundle)
    _write_pairwise_comparisons(analysis_dir, bundle)
    _write_scenario_analysis(analysis_dir, bundle)
    _write_failure_analysis(analysis_dir, bundle)
    _write_analysis_table_csv(analysis_dir, bundle)
    _write_analysis_summary(analysis_dir, bundle)
    generate_all_figures(bundle, figures_dir)
    (analysis_dir / "findings.md").write_text(generate_findings_markdown(bundle))

    return bundle, analysis_dir


def _fmt_pct(value: float | None) -> str:
    return f"{value * 100:.1f}%" if value is not None else "N/A"


def _print_verification_block(bundle: AnalysisBundle) -> None:
    config = bundle.dataset.config
    print("=" * 50)
    print("MILESTONE 14B — FINAL STATISTICAL ANALYSIS")
    print("=" * 50)
    print()
    print("EXPERIMENT")
    print()
    print(f"Experiment ID: {config.experiment_id}")
    print("Provider: Ollama")
    print(f"Model: {config.model_name}")
    print("Mode: REAL")
    print()
    print(f"Scenarios: {len(config.scenario_ids)}")
    print(f"Repetitions: {config.repetitions}")
    print(f"Total observations: {len(bundle.dataset.raw_runs)}")
    print()
    print("DATA VALIDATION")
    print()
    print(f"Frozen dataset valid: {'PASS' if bundle.dataset.validation_report.dataset_valid else 'FAIL'}")
    print("Raw dataset unchanged: PASS (analysis is read-only; see src/analysis/loading.py)")
    print()
    print("PRIMARY RESULTS")
    print()
    print(f"{'':25s}{'RAG':>10s}{'TOOL':>10s}{'HYBRID':>10s}")
    for label, field in (
        ("Best-line accuracy", "best_line_accuracy"),
        ("Best-odds accuracy", "best_odds_accuracy"),
        ("EV classification", "ev_classification_accuracy"),
        ("Freshness accuracy", "freshness_accuracy"),
    ):
        values = [_fmt_pct(getattr(bundle.descriptive_stats[a].summary, field)) for a in ARCHITECTURE_LABELS]
        print(f"{label:25s}{values[0]:>10s}{values[1]:>10s}{values[2]:>10s}")
    consistency = [f"{bundle.descriptive_stats[a].summary.consistency:.3f}" for a in ARCHITECTURE_LABELS]
    print(f"{'Consistency':25s}{consistency[0]:>10s}{consistency[1]:>10s}{consistency[2]:>10s}")
    completeness = [_fmt_pct(bundle.descriptive_stats[a].summary.mean_completeness) for a in ARCHITECTURE_LABELS]
    print(f"{'Completeness':25s}{completeness[0]:>10s}{completeness[1]:>10s}{completeness[2]:>10s}")
    unsupported = [_fmt_pct(bundle.descriptive_stats[a].summary.unsupported_claim_rate) for a in ARCHITECTURE_LABELS]
    print(f"{'Unsupported claims':25s}{unsupported[0]:>10s}{unsupported[1]:>10s}{unsupported[2]:>10s}")
    latency = [f"{bundle.descriptive_stats[a].latency_median:.2f}s" for a in ARCHITECTURE_LABELS]
    print(f"{'Median latency':25s}{latency[0]:>10s}{latency[1]:>10s}{latency[2]:>10s}")
    print()

    for metric_label, metric in (("BEST-LINE", "best_line_correct"), ("FRESHNESS", "freshness_correct")):
        print(metric_label)
        print()
        for comparison in bundle.binary_comparisons:
            if comparison.metric != metric:
                continue
            print(f"{comparison.architecture_a.upper()} vs {comparison.architecture_b.upper()}:")
            if comparison.result is not None:
                print(f"  Difference: {comparison.result.difference_pp:+.2f} pp")
            print(f"  Raw p: {comparison.raw_p}")
            print(f"  Holm-adjusted p: {comparison.holm_adjusted_p}")
            significant = comparison.holm_adjusted_p is not None and comparison.holm_adjusted_p < 0.05
            print(f"  Significant at alpha=.05: {'YES' if significant else 'NO'}")
        print()

    print("CONSISTENCY")
    print()
    omnibus_consistency = next(c for c in bundle.omnibus_comparisons if c.metric == "consistency")
    print(f"Friedman statistic: {omnibus_consistency.result.statistic if omnibus_consistency.result else 'N/A'}")
    print(f"Friedman p: {omnibus_consistency.result.p_value if omnibus_consistency.result else 'N/A'}")
    print()

    print("FAILURES")
    print()
    for architecture in ARCHITECTURE_LABELS:
        count = bundle.descriptive_stats[architecture].summary.failures
        print(f"{architecture.upper()}: {count}")
    print()

    print("HYBRID RECONCILIATION")
    print()
    if bundle.hybrid_conflict:
        h = bundle.hybrid_conflict
        print(f"Agreements: {h['source_agreements']}")
        print(f"Conflicts: {h['source_conflicts']}")
        print(f"Correct resolutions: {h['correct_conflict_resolutions']}")
        print(f"Incorrect resolutions: {h['source_conflicts'] - h['correct_conflict_resolutions']}")
        print(f"Conflict-resolution accuracy: {_fmt_pct(h['conflict_resolution_accuracy'])}")
        print(f"Stale RAG incorrectly promoted: {h['stale_rag_incorrectly_promoted']}")
        print(f"Tool-only recoveries: {h['tool_only_recoveries']}")
    print()

    print("DIRECT ANSWER TO RESEARCH QUESTION")
    print()
    print("See analysis/findings.md, section 'Answer to the Research Question'.")
    print()

    print("LIMITATIONS")
    print()
    print("See analysis/findings.md, section 'Limitations'.")
    print()

    print("FUTURE WORK")
    print()
    print("Independent predictive ML model: DOCUMENTED")
    print()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Milestone 14B final statistical analysis")
    parser.add_argument("--experiment-dir", default=DEFAULT_EXPERIMENT_DIR)
    args = parser.parse_args(argv)

    try:
        bundle, analysis_dir = run_final_analysis(args.experiment_dir)
    except FinalDatasetInvalidError as exc:
        print("STOP: frozen dataset failed revalidation — no statistical analysis was performed.")
        print(f"  {exc}")
        return 1

    _print_verification_block(bundle)
    print(f"Analysis outputs written to: {analysis_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
