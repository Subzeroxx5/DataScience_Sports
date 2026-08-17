# Manuscript Outline

Detailed paper structure for a future manuscript, built from the actual
completed project and its Milestone 14B results. This is an outline only —
see `docs/MANUSCRIPT_DRAFT.md` for the first-draft prose.

# Title

Working title: *"Retrieval, Tools, or Both? A Controlled Comparison of Agent
Architectures for Sportsbook Price Identification"*

# Abstract

Outline (no fabricated numbers — see draft for actual cited values):

- **Problem:** identifying the best available sportsbook price for a given
  market requires an AI agent to gather current, multi-source information
  reliably.
- **Research question:** how does agent architecture (RAG-only,
  tool-calling-only, hybrid) affect accuracy, consistency, and freshness?
- **Methodology:** controlled, synthetic sportsbook benchmark; deterministic
  ground truth; one locally hosted LLM (`llama3.1:8b`) held constant across
  all three architectures; 11 scenarios x 10 repetitions x 3 architectures
  = 330 real observations; a shared deterministic quantitative engine;
  paired nonparametric inferential statistics (McNemar, Wilcoxon,
  Friedman) with Holm correction.
- **Architectures:** RAG-only, tool-calling-only, hybrid (tool data takes
  precedence over conflicting RAG data).
- **Primary findings:** TOOL and HYBRID significantly outperformed RAG on
  best-line/best-odds accuracy; TOOL and HYBRID were statistically
  indistinguishable from each other; consistency and freshness showed no
  distinguishable difference across architectures; HYBRID was significantly
  slower than TOOL with no measured accuracy benefit.
- **Conclusion:** no single architecture dominates across all primary
  outcomes; architecture selection involves an accuracy/completeness vs.
  latency tradeoff in this setting.

# 1. Introduction

- The sportsbook line-comparison problem: the same event/market is priced
  differently across sportsbooks, and identifying the best price (and
  whether it represents positive expected value relative to a market
  reference) requires gathering current, multi-sportsbook data.
- AI agents increasingly rely on external information sources — retrieved
  documents, callable tools, or both — to answer such queries.
- Freshness and reliability of the underlying information matter: a
  retrieval-based agent may surface stale evidence; a tool-calling agent
  depends on reliable multi-step orchestration.
- Why architecture matters: the choice between RAG, tool calling, and a
  hybrid combination is a design decision with measurable consequences for
  accuracy, consistency, and freshness — not merely an implementation detail.
- Research question (stated verbatim, matching project documentation).
- Contribution: a controlled, reproducible experimental framework
  (deterministic benchmark, shared quant engine, unified evaluation
  framework, frozen final experiment) comparing all three architectures
  under identical controls with real local LLM inference, plus the
  resulting statistical findings and full reproducibility artifacts.

# 2. Background / Related Concepts

*(Suitable for later citation work — literature claims marked [CITATION
NEEDED]; see `docs/CITATION_NEEDS.md` for the full audit.)*

- LLM agents and external information access. [CITATION NEEDED]
- Retrieval-augmented generation (RAG). [CITATION NEEDED]
- Tool-calling / function-calling agents. [CITATION NEEDED]
- Hybrid RAG + tool-calling architectures. [CITATION NEEDED]
- Sportsbook odds and implied probability (American odds convention,
  overround/vig). [CITATION NEEDED]
- No-vig (fair) probability normalization and market consensus. [CITATION
  NEEDED]
- Expected value in a betting-markets context. [CITATION NEEDED]
- LLM output reliability / hallucination in agentic contexts. [CITATION
  NEEDED]

# 3. Methods

## 3.1 Research Design

- Independent variable: agent architecture (RAG-only, tool-calling-only,
  hybrid).
- Primary dependent variables: accuracy (best-line, best-odds, EV
  classification), consistency, freshness.
- Secondary dependent variables: completeness, unsupported-claim
  (hallucination) rate, latency, failure rate/category, hybrid
  conflict-resolution behavior.

## 3.2 Controlled Benchmark

- Synthetic, controlled sportsbook dataset (games, markets, per-sportsbook
  American odds) with deterministically computed ground truth — never live
  market data.
- Ground truth generated once from the benchmark, independent of any
  architecture, and never exposed to an agent as input.
- A representative 11-scenario subset spanning positive/negative/mixed-sign
  odds, a best-line tie, a missing-sportsbook case, a current/stale
  freshness case, and moneyline/spread/total market types.

## 3.3 Architectures

- **RAG-only:** retrieves evidence from a vector-indexed controlled
  document corpus (sentence-transformer embeddings, exact FAISS inner-product
  index); no access to structured sportsbook tools.
- **Tool-calling-only:** a bounded multi-turn tool-calling loop against
  structured sportsbook tools backed by `OddsProvider`; no access to the RAG
  corpus.
- **Hybrid:** access to both channels; deterministic, LLM-free source
  reconciliation where current structured tool data always takes precedence
  over a conflicting RAG-derived price for the same sportsbook/outcome.

## 3.4 Shared Quantitative Engine

- American-odds -> implied-probability conversion.
- No-vig (fair) probability normalization.
- Leave-one-sportsbook-out market consensus.
- Probability edge and expected value, computed from the market-consensus
  reference probability.
- Market dispersion (descriptive spread statistics).
- Identical implementation used by all three architectures — never
  reimplemented per architecture.
- Explicit framing: the market-consensus reference probability is not a
  claim about the true probability of the sporting outcome.

## 3.5 Experimental Controls

Held constant across all three architectures: LLM provider and exact model,
temperature, RAG top_k, tool-call iteration bound, embedding model, scenario
set, one canonical architecture-neutral query per scenario, repetition
count, the shared quantitative engine, and the unified evaluation framework.
Architecture execution order was rotated deterministically across
repetitions to avoid confounding order effects with architecture identity.

## 3.6 Experimental Procedure

- Configuration frozen before execution (`experiments/final_experiment.json`).
- Preflight validation (test suite, ground-truth reproducibility, RAG index
  and retrieval smoke test, tool/provider smoke test, hybrid reconciliation
  policy smoke test, real-inference connectivity probe) before any
  observation was recorded.
- Execution via a single unified experiment runner — never a per-architecture
  script — recording every observation (including failures) with its
  architecture, scenario, repetition, and execution-order position.
- Post-execution validation: expected-vs-recorded run count, duplicate/
  missing-key check, full scenario-coverage matrix, raw-result schema
  validation, ground-truth and architecture isolation re-audits, and a 9-way
  pre/post artifact-fingerprint comparison (all matched).

## 3.7 Statistical Analysis

Only methods actually used in the final analysis:

- **Wilson score confidence intervals** for proportion metrics (best-line,
  best-odds, freshness, success rate), preferred over the naive normal
  approximation for stability near 0%/100%.
- **McNemar's exact test** (binomial form) for paired binary outcomes
  (best-line correctness, best-odds correctness, EV-classification
  correctness, freshness correctness).
- **Wilcoxon signed-rank test** for paired continuous outcomes (EV absolute
  error, market-reference absolute error, completeness, total latency).
- **Friedman test** for the three-architecture omnibus comparison on
  matched blocks (completeness, total latency per observation; consistency
  per scenario).
- **Holm-Bonferroni correction**, applied within each metric family's three
  pairwise architecture comparisons (never pooled across unrelated metric
  families).
- Alpha = 0.05, frozen before any test was run.

# 4. Results

Reference: `results/experiments/final_v1/analysis/` (tables, figures,
`findings.md`). No results are recalculated in the manuscript.

## Accuracy

- Table 1 (architecture descriptive results); Table 2 (binary pairwise
  comparisons); Figure 1 (best-line accuracy by architecture); Figure 2 (EV
  classification accuracy by architecture — all N/A).

## Consistency

- Table 1; Figure 4 (consistency by architecture); Friedman omnibus result.

## Freshness

- Table 1; Figure 3 (freshness accuracy by architecture); Table 2
  (freshness pairwise comparisons).

## Secondary Metrics

- Completeness and latency: Table 3 (continuous pairwise comparisons);
  Figure 6 (median latency by architecture).
- Unsupported claims: Figure 5 (unsupported-claim rate by architecture).

## Failure Patterns

- Table 4 (failure counts by architecture and category).

## Hybrid Conflict Resolution

- Table 5 (hybrid reconciliation statistics).

# 5. Discussion

- Architecture tradeoffs: accuracy/completeness (RAG lower) vs. latency
  (HYBRID highest, TOOL lowest); no architecture leads on every metric.
- Likely mechanism for RAG's lower accuracy/completeness: the
  architecture-neutral canonical query does not name sportsbooks explicitly,
  and semantic retrieval does not guarantee both sides of a two-sided
  market are retrieved for a given top_k.
- Likely mechanism for the pervasive `quant_insufficient_data` outcome
  across all three architectures: the local model did not reliably gather
  both sides of a market (e.g., not consistently calling a game-lookup tool
  before requesting the opposing outcome's odds) — a model-capability
  characteristic, not an architecture-design flaw, since the same tool
  schema and prompts are used regardless of provider.
- Structured-tool benefits: TOOL and HYBRID's authoritative, tool-derived
  data structurally prevents a hallucinated final claim, consistent with
  the zero observed unsupported claims across all three architectures.
- Hybrid reconciliation behavior: the deterministic current-tool-data
  precedence mechanism is verified intact independently of this dataset,
  but the dataset itself never exercised a live conflict — discuss why
  (RAG-side extraction did not retain disagreeing validated prices) and
  what would be needed to observe the mechanism in action (a scenario
  design that reliably surfaces a stale RAG price alongside a differing
  current tool price, verified retrievable by the local model).
- Latency/complexity tradeoff: HYBRID pays the combined cost of both
  channels; whether that cost is justified depends on whether a given
  deployment needs RAG's contextual/historical coverage in addition to
  TOOL's current-data reliability.
- Explicitly avoid causal overstatement: this is a controlled, single-model,
  single-embedding-model study: findings describe *this* model's behavior
  under *this* architecture design, not a universal claim about RAG, tool
  calling, or hybrid architectures in general.

# 6. Limitations

(Use the validated limitations list verbatim — see
`docs/FINAL_RESEARCH_SUMMARY.md`, "Limitations.")

# 7. Future Work

- Live sportsbook providers behind the existing `OddsProvider` abstraction.
- Broader/expanded scenario coverage.
- Alternative LLMs (other local models and frontier-hosted models) via the
  existing configuration-driven provider abstraction.
- Independent predictive ML model extension: compare an independently
  trained/calibrated predictive win-probability model against the
  market-implied no-vig consensus, without changing the core
  architecture-comparison question. (Not implemented.)

# 8. Conclusion

Directly answer the research question using the evidence-calibrated
conclusion from `findings.md` / `docs/FINAL_RESEARCH_SUMMARY.md` — no
universal winner is declared; architecture-specific tradeoffs are stated
explicitly (accuracy/completeness advantage for TOOL/HYBRID over RAG;
latency advantage for TOOL over HYBRID; no distinguishable difference in
consistency or freshness in this dataset).
