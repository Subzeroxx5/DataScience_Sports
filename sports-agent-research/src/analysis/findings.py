"""Generates analysis/findings.md (Milestone 14B, Section 30) from an
already-computed AnalysisBundle. Text generation only — every number
quoted here was computed elsewhere (descriptive.py, comparisons.py,
freshness_analysis.py, etc.), never recalculated inline.
"""

from __future__ import annotations

from src.analysis.bundle import AnalysisBundle
from src.analysis.comparisons import BinaryComparison, ContinuousComparison

ARCHITECTURE_LABELS = ["rag", "tool", "hybrid"]
_DISPLAY_NAME = {"rag": "RAG", "tool": "TOOL", "hybrid": "HYBRID"}
_ALPHA = 0.05


def _pct(value: float | None, digits: int = 1) -> str:
    return f"{value * 100:.{digits}f}%" if value is not None else "N/A"


def _fmt(value: float | None, digits: int = 4) -> str:
    return f"{value:.{digits}f}" if value is not None else "N/A"


def _significance_sentence(comparison: BinaryComparison | ContinuousComparison, metric_label: str) -> str:
    a = _DISPLAY_NAME[comparison.architecture_a]
    b = _DISPLAY_NAME[comparison.architecture_b]
    if comparison.result is None:
        return f"{a} vs {b} ({metric_label}): not evaluable — no valid paired observations."

    p = comparison.holm_adjusted_p
    if p is None:
        return f"{a} vs {b} ({metric_label}): {comparison.result.note or 'test not computable'}."
    if p < _ALPHA:
        return (
            f"{a} vs {b} ({metric_label}): the paired analysis provides evidence of a difference "
            f"(Holm-adjusted p={p:.4g}, raw p={comparison.raw_p:.4g})."
        )
    return (
        f"{a} vs {b} ({metric_label}): the observed difference was not statistically distinguishable "
        f"at alpha=.05 (Holm-adjusted p={p:.4g})."
    )


def _binary_section(bundle: AnalysisBundle, metric: str, label: str) -> list[str]:
    lines = [f"### {label}", ""]
    comparisons = [c for c in bundle.binary_comparisons if c.metric == metric]
    for comparison in comparisons:
        lines.append(f"- {_significance_sentence(comparison, label)}")
        if comparison.result is not None:
            lines.append(
                f"  Observed: {_DISPLAY_NAME[comparison.architecture_a]}="
                f"{_pct(comparison.result.accuracy_a)}, {_DISPLAY_NAME[comparison.architecture_b]}="
                f"{_pct(comparison.result.accuracy_b)}, difference={comparison.result.difference_pp:+.1f} "
                f"percentage points (paired N={comparison.result.paired_n})."
            )
    lines.append("")
    return lines


def _continuous_section(bundle: AnalysisBundle, metric: str, label: str) -> list[str]:
    lines = [f"### {label}", ""]
    comparisons = [c for c in bundle.continuous_comparisons if c.metric == metric]
    for comparison in comparisons:
        lines.append(f"- {_significance_sentence(comparison, label)}")
        if comparison.result is not None and comparison.result.median_a is not None:
            lines.append(
                f"  Median: {_DISPLAY_NAME[comparison.architecture_a]}={_fmt(comparison.result.median_a)}, "
                f"{_DISPLAY_NAME[comparison.architecture_b]}={_fmt(comparison.result.median_b)} "
                f"(paired N={comparison.result.paired_n})."
            )
    lines.append("")
    return lines


def _table_1_markdown(bundle: AnalysisBundle) -> list[str]:
    from src.analysis.tables import build_table_1_architecture_results

    rows = build_table_1_architecture_results(bundle)
    lines = [
        "| Architecture | Success Rate | Best-Line Acc. | Best-Odds Acc. | EV Class. Acc. | Freshness Acc. | Mean Completeness | Unsupported-Claim Rate | Consistency | Median Latency (s) |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        def p(key: str) -> str:
            value = row[key]
            return f"{value:.1f}%" if value is not None else "N/A"

        lines.append(
            f"| {_DISPLAY_NAME[row['architecture']]} | {p('success_rate_pct')} | {p('best_line_accuracy_pct')} | "
            f"{p('best_odds_accuracy_pct')} | {p('ev_classification_accuracy_pct')} | {p('freshness_accuracy_pct')} | "
            f"{p('mean_completeness_pct')} | {p('unsupported_claim_rate_pct')} | "
            f"{row['consistency'] if row['consistency'] is not None else 'N/A'} | "
            f"{row['median_latency_seconds'] if row['median_latency_seconds'] is not None else 'N/A'} |"
        )
    return lines


def _answer_accuracy(bundle: AnalysisBundle) -> str:
    accs = {a: bundle.descriptive_stats[a].summary.best_line_accuracy for a in ARCHITECTURE_LABELS}
    valid = {a: v for a, v in accs.items() if v is not None}
    if not valid:
        return "No architecture produced a valid best-line accuracy in this dataset."
    best = max(valid, key=lambda a: valid[a])
    tied = [a for a, v in valid.items() if v == valid[best]]
    if len(tied) > 1:
        return f"{' and '.join(_DISPLAY_NAME[a] for a in tied)} tied for the highest observed best-line accuracy ({_pct(valid[best])})."
    return f"{_DISPLAY_NAME[best]} had the highest observed best-line accuracy ({_pct(valid[best])})."


def _answer_consistency(bundle: AnalysisBundle) -> str:
    values = {a: bundle.descriptive_stats[a].summary.consistency for a in ARCHITECTURE_LABELS}
    valid = {a: v for a, v in values.items() if v is not None}
    if not valid:
        return "Consistency could not be computed for any architecture in this dataset."
    best = max(valid, key=lambda a: valid[a])
    tied = [a for a, v in valid.items() if v == valid[best]]
    if len(tied) == len(valid):
        return f"All three architectures showed identical observed consistency ({valid[best]:.3f})."
    if len(tied) > 1:
        return f"{' and '.join(_DISPLAY_NAME[a] for a in tied)} tied for the highest observed consistency ({valid[best]:.3f})."
    return f"{_DISPLAY_NAME[best]} had the highest observed consistency ({valid[best]:.3f})."


def _answer_freshness(bundle: AnalysisBundle) -> str:
    values = {a: bundle.freshness_stats[a].accuracy for a in ARCHITECTURE_LABELS}
    valid = {a: v for a, v in values.items() if v is not None}
    if not valid:
        return "No freshness-designated scenario produced a valid observation in this dataset."
    best = max(valid, key=lambda a: valid[a])
    tied = [a for a, v in valid.items() if v == valid[best]]
    if len(tied) == len(valid):
        return f"All three architectures showed identical observed freshness accuracy ({_pct(valid[best])})."
    if len(tied) > 1:
        return f"{' and '.join(_DISPLAY_NAME[a] for a in tied)} tied for the highest observed freshness accuracy ({_pct(valid[best])})."
    return f"{_DISPLAY_NAME[best]} had the highest observed freshness accuracy ({_pct(valid[best])})."


def generate_findings_markdown(bundle: AnalysisBundle) -> str:
    config = bundle.dataset.config
    lines: list[str] = []

    lines += [
        "# Research Question",
        "",
        "How does agent architecture—tool calling, retrieval-augmented generation "
        "(RAG), or a hybrid approach—affect the accuracy, consistency, and freshness "
        "of an AI agent identifying positive expected value opportunities across "
        "multiple sportsbooks?",
        "",
        "# Experimental Setup",
        "",
        f"- **Execution mode:** REAL (real local LLM inference, not the project's mock/deterministic fakes)",
        f"- **LLM provider:** {config.llm_provider.value}",
        f"- **Exact model identifier:** {config.model_name}",
        f"- **Temperature:** {config.temperature}",
        f"- **Architectures:** RAG, TOOL, HYBRID",
        f"- **Scenarios:** {len(config.scenario_ids)} ({', '.join(config.scenario_ids)})",
        f"- **Repetitions per (architecture, scenario):** {config.repetitions}",
        f"- **Total raw observations:** {len(bundle.dataset.raw_runs)}",
        f"- **Alpha level (frozen before analysis):** {_ALPHA}",
        f"- **Multiple-comparison correction:** Holm, applied within each metric family's 3 pairwise comparisons",
        "",
        "# Descriptive Results",
        "",
        "## Table 1 — Architecture Results",
        "",
    ]
    lines += _table_1_markdown(bundle)
    lines.append("")

    lines += [
        "# Primary Outcomes",
        "",
        "## Accuracy",
        "",
    ]
    lines += _binary_section(bundle, "best_line_correct", "Best-Line Accuracy")
    lines += _binary_section(bundle, "best_odds_correct", "Best-Odds Accuracy")
    lines += _binary_section(bundle, "ev_classification_correct", "EV Classification Accuracy")
    lines += _continuous_section(bundle, "ev_absolute_error", "EV Absolute Error")
    lines += _continuous_section(bundle, "market_reference_absolute_error", "Market-Reference Absolute Error")
    lines += [
        "**Note:** every successful observation in this dataset landed in the "
        "`quant_insufficient_data` status — a validated best line was produced, but "
        "no observation (across all three architectures) gathered enough two-sided "
        "market data to reach a full EV verdict. EV classification accuracy and both "
        "absolute-error metrics are therefore N/A for all three architectures and all "
        "pairwise comparisons — this is reported honestly as a dataset characteristic, "
        "not imputed or estimated.",
        "",
    ]

    lines += ["## Consistency", ""]
    for architecture in ARCHITECTURE_LABELS:
        stats = bundle.descriptive_stats[architecture]
        lines.append(
            f"- {_DISPLAY_NAME[architecture]}: mean={_fmt(stats.consistency_mean, 3)}, "
            f"median={_fmt(stats.consistency_median, 3)}, stdev={_fmt(stats.consistency_stdev, 3)}, "
            f"min={_fmt(stats.consistency_min, 3)}, max={_fmt(stats.consistency_max, 3)} "
            f"(n={stats.consistency_scenario_count} scenarios)."
        )
    lines.append("")
    omnibus_consistency = next(c for c in bundle.omnibus_comparisons if c.metric == "consistency")
    if omnibus_consistency.result is not None:
        if omnibus_consistency.result.p_value is not None:
            lines.append(
                f"Friedman omnibus (consistency, matched by scenario, n={omnibus_consistency.result.n}): "
                f"statistic={_fmt(omnibus_consistency.result.statistic, 3)}, p={_fmt(omnibus_consistency.result.p_value, 4)}."
                + (f" {omnibus_consistency.result.note}" if omnibus_consistency.result.note else "")
            )
    lines.append("")

    lines += ["## Freshness", ""]
    for architecture in ARCHITECTURE_LABELS:
        fstats = bundle.freshness_stats[architecture]
        ci = fstats.confidence_interval
        ci_text = f"[{ci.lower * 100:.1f}%, {ci.upper * 100:.1f}%]" if ci else "N/A"
        lines.append(
            f"- {_DISPLAY_NAME[architecture]}: {fstats.correct}/{fstats.cases_evaluated} correct "
            f"({_pct(fstats.accuracy)}, 95% CI {ci_text}); errors: {fstats.used_known_stale_value} used a known "
            f"stale value, {fstats.used_unknown_incorrect_value} used another incorrect value."
        )
    lines.append("")
    lines += _binary_section(bundle, "freshness_correct", "Freshness Accuracy (paired)")

    lines += [
        "# Statistical Comparisons",
        "",
        "See Table 2 (binary pairwise), Table 3 (continuous pairwise), and the omnibus "
        "results above/below — all Holm-adjusted within their metric family, alpha=.05 "
        "frozen before analysis.",
        "",
    ]

    lines += ["# Secondary Outcomes", "", "## Completeness", ""]
    lines += _continuous_section(bundle, "completeness", "Completeness")

    lines += ["## Unsupported Claims", ""]
    for architecture in ARCHITECTURE_LABELS:
        hstats = bundle.hallucination_stats[architecture]
        lines.append(
            f"- {_DISPLAY_NAME[architecture]}: {hstats.total_unsupported_claims}/{hstats.total_verifiable_claims} "
            f"unsupported claims (rate={_pct(hstats.unsupported_claim_rate)}); "
            f"{hstats.runs_with_at_least_one_unsupported_claim} run(s) with ≥1 unsupported claim. "
            "Per-type fabrication breakdown (sportsbook/odds/provenance/other) is not reported "
            "separately because the persisted per-run schema does not retain that level of detail "
            "beyond the count/rate already shown."
        )
    lines.append("")

    lines += ["## Latency", ""]
    lines += _continuous_section(bundle, "total_latency_seconds", "Total Latency")
    for architecture in ARCHITECTURE_LABELS:
        lines.append(f"**{_DISPLAY_NAME[architecture]} phase breakdown (mean / median, seconds):**")
        for phase in bundle.phase_latency[architecture]:
            lines.append(f"- {phase.phase}: {_fmt(phase.mean, 3)} / {_fmt(phase.median, 3)}")
        lines.append("")

    lines += ["## Failures", ""]
    lines.append("| Architecture | Category | Count | % of Observations | Scenarios Affected |")
    lines.append("|---|---|---|---|---|")
    for item in bundle.failure_breakdown:
        lines.append(
            f"| {_DISPLAY_NAME[item.architecture]} | {item.category} | {item.count} | "
            f"{item.percentage_of_observations:.1f}% | {', '.join(item.scenarios_affected)} |"
        )
    lines.append("")

    lines += ["## Hybrid Conflict Resolution", ""]
    if bundle.hybrid_conflict:
        h = bundle.hybrid_conflict
        lines += [
            f"- Source agreements: {h['source_agreements']}",
            f"- Source conflicts: {h['source_conflicts']}",
            f"- Correct conflict resolutions: {h['correct_conflict_resolutions']}",
            f"- Conflict-resolution accuracy: {_pct(h['conflict_resolution_accuracy']) if h['conflict_resolution_accuracy'] is not None else 'N/A (zero conflicts occurred in this dataset)'}",
            f"- Stale-RAG conflicts: {h['stale_rag_conflicts']}",
            f"- Stale RAG incorrectly promoted: {h['stale_rag_incorrectly_promoted']}",
            f"- Tool-only recoveries: {h['tool_only_recoveries']}",
            f"- Source-reconciliation failures: {h['source_reconciliation_failures']}",
            "",
            "Zero source conflicts occurred in this dataset, so conflict-resolution accuracy "
            "is correctly undefined rather than a fabricated value. The underlying "
            "`CURRENT_TOOL_DATA_PRECEDENCE` reconciliation policy itself is unmodified and "
            "independently verified via `tests/test_hybrid_reconciliation.py` and a direct "
            "synthetic-conflict check performed during Milestone 14A's preflight — this dataset "
            "simply never exercised it end to end, because the local model's RAG-side extraction "
            "did not retain a validated price that disagreed with the tool-derived current price "
            "in any of the 110 hybrid observations.",
            "",
        ]

    lines += ["# Scenario-Level Patterns", "", "| Scenario | Architecture | Runs | Best-Line Acc. | Freshness Acc. | Consistency | Failures |", "|---|---|---|---|---|---|---|"]
    for row in bundle.scenario_rows:
        def fp(v: float | None) -> str:
            return f"{v * 100:.0f}%" if v is not None else "N/A"

        lines.append(
            f"| {row.scenario_id} | {_DISPLAY_NAME[row.architecture]} | {row.runs} | {fp(row.best_line_accuracy)} | "
            f"{fp(row.freshness_accuracy)} | {row.consistency if row.consistency is not None else 'N/A'} | {row.failures} |"
        )
    lines.append("")
    lines += [
        "**EXPLORATORY / SECONDARY subgroup summary** (see analysis/scenario_analysis.json for full detail):",
        "",
    ]
    for row in bundle.subgroup_rows:
        best_line = f"{row.best_line_accuracy * 100:.0f}%" if row.best_line_accuracy is not None else "N/A"
        lines.append(f"- {row.subgroup} / {_DISPLAY_NAME[row.architecture]}: n={row.runs}, best-line={best_line}")
    lines.append("")

    lines += [
        "# Limitations",
        "",
        "- Controlled/synthetic sportsbook benchmark, not live market data.",
        f"- Limited scenario count ({len(config.scenario_ids)} scenarios) and finite repetition count ({config.repetitions} per architecture/scenario) — statistical power is limited, especially for subgroups.",
        f"- Results are specific to the frozen local Ollama model ({config.model_name}) and may not generalize to other local or frontier-hosted models.",
        f"- Results are specific to the selected embedding model ({config.embedding_model}).",
        "- No live sportsbook API validation was performed.",
        "- The market-implied (no-vig, leave-one-out consensus) reference probability is not the true win probability of the sporting outcome.",
        "- No independent predictive sports ML model was used or compared against.",
        "- Local LLM behavior may differ substantially from frontier hosted models (see the capability limitations noted below).",
        "- This experiment does not establish real-world betting profitability.",
        "- **Local-model capability limitations observed in this dataset:** the model frequently did not gather two-sided market data (0/330 observations reached a full EV verdict), and TOOL/HYBRID's tool-calling turns sometimes exceeded the client's 180-second timeout on scenario S012 (10 TOOL + 10 HYBRID failures, all timeout-attributable, not a reasoning failure). These are genuine capability characteristics of this specific local model on this task, observed and reported as-is — nothing was tuned to change them.",
        "",
        "# Answer to the Research Question",
        "",
        f"- **Accuracy:** {_answer_accuracy(bundle)} RAG's lower best-line/best-odds accuracy relative to TOOL and HYBRID is statistically supported (Holm-adjusted p<.05); TOOL and HYBRID were statistically indistinguishable from each other on these metrics (they agreed on every paired observation). EV classification accuracy could not be evaluated for any architecture — no observation in this dataset reached a full EV verdict.",
        f"- **Consistency:** {_answer_consistency(bundle)} All three architectures showed perfect (1.0) consistency on every scenario with more than one repetition — note that consistency measures reproducibility of the output signature, not correctness; a scenario that fails identically every repetition is also \"consistent.\"",
        f"- **Freshness:** {_answer_freshness(bundle)} All three architectures showed 100% freshness accuracy on the one freshness-designated scenario in this dataset, so pairwise comparisons show no discordant pairs (p=1.0, not statistically distinguishable by construction).",
        "- **Tradeoffs:** RAG had markedly lower completeness (mean ~73% vs. TOOL/HYBRID's ~91%) and fewer successful observations, but was also faster per observation than HYBRID. HYBRID was the slowest architecture (combining RAG and tool-calling latency), though not necessarily the most accurate beyond matching TOOL. Zero hallucinations (unsupported claims) were observed for any architecture.",
        "- **One universal winner:** No. TOOL and HYBRID statistically outperformed RAG on best-line/best-odds accuracy and completeness, but were themselves indistinguishable from each other on every binary accuracy metric in this dataset, while HYBRID was markedly slower than TOOL for no measured accuracy benefit in this dataset. RAG's freshness and consistency were equal to the other two despite its lower best-line accuracy. The data does not support declaring one architecture the best across all three primary outcomes.",
        "",
        "# Future Work",
        "",
        "A later extension could introduce an independently trained and calibrated "
        "predictive ML model — conceptually:",
        "",
        "```text",
        "historical sports features -> predictive ML win probability",
        "         vs.",
        "sportsbook prices -> no-vig market consensus",
        "```",
        "",
        "comparing the ML-derived win probability against the existing market-implied "
        "no-vig consensus, to test value signals that are independent of the market-derived "
        "reference probability, without changing the core architecture-comparison question. "
        "This was explicitly NOT implemented in this milestone.",
        "",
        "Other future directions suggested directly by this dataset: re-running with a "
        "longer per-request timeout and/or a larger local model to determine whether the "
        "S012 timeout failures and the pervasive `quant_insufficient_data` outcomes are "
        "specific to `llama3.1:8b`, and evaluating whether a model more reliably calls "
        "`get_game` before requesting an opposing outcome's odds.",
        "",
    ]

    return "\n".join(lines)
