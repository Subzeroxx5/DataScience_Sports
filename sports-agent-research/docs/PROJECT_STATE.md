# Project State

Authoritative, short-form handoff for future sessions. See
`docs/ROADMAP.md` for the full milestone list and `milestones/current.md`
for the active milestone's scope.

## Research Question

How does agent architecture—tool calling, retrieval-augmented generation
(RAG), or a hybrid approach—affect the accuracy, consistency, and
freshness of an AI agent identifying positive expected value
opportunities across multiple sportsbooks?

## Completed Functional Milestones

```text
Milestone 1   — Project foundation and research design       — COMPLETE
Milestone 2   — Core Pydantic domain models                   — COMPLETE
Milestone 3   — Deterministic American-odds and EV math       — COMPLETE
Milestone 4   — Controlled benchmark + deterministic ground truth — COMPLETE
Milestone 5   — Provider + sportsbook tool subsystem           — COMPLETE
Milestone 6A  — Two-sided market readiness                     — COMPLETE
Milestone 6B  — Controlled RAG corpus                          — COMPLETE
Milestone 6C  — Embeddings, vector index, retrieval evaluation — COMPLETE
Milestone 7A  — Core quantitative market mathematics            — COMPLETE
Milestone 7B  — Quantitative ground-truth integration            — COMPLETE
Milestone 8A  — RAG evidence pipeline and agent contract          — COMPLETE
Milestone 8B  — RAG-only LLM agent and structured quant analysis  — COMPLETE
Milestone 9A  — Tool-calling agent core                           — COMPLETE
Milestone 9B  — Tool-calling agent end-to-end verification        — COMPLETE
Milestone 10A — Hybrid RAG + tool-calling agent core               — COMPLETE
Milestone 10B — Hybrid agent end-to-end verification                — COMPLETE
Milestone 11   — Unified evaluation framework                       — COMPLETE
Milestone 12   — Controlled experiment runner                        — COMPLETE
Milestone 13   — Dashboard and research visualization                — COMPLETE
Milestone 14A  — Freeze and run the final controlled experiment       — COMPLETE
Milestone 14B  — Statistical analysis and research findings           — COMPLETE
Milestone 15   — Research conclusions, manuscript, and presentation support — COMPLETE
```

## Current Documentation Checkpoint

```text
6B.5 — Project Documentation and Agent Instructions (COMPLETE)
Pre-14A — Local Real-LLM Backend (COMPLETE)
```

## Next Functional Milestone

```text
None — project roadmap complete (see docs/ROADMAP.md).
```

All 15 planned milestones are COMPLETE. Remaining items (live sportsbook
providers, broader scenario coverage, alternative LLMs, an independently
trained predictive ML model) are documented future work in
`docs/FINAL_RESEARCH_SUMMARY.md` and `docs/MANUSCRIPT_DRAFT.md`, not an
active or implied next milestone — see `milestones/current.md`.

## Current Test Baseline

```text
Current full test suite:
1097 passed / 0 failed
```

(Recorded from `pytest -q` at the end of Milestone 14B.)

## Current System State

The project currently has:

- validated domain models (`src/models.py`)
- deterministic betting mathematics (`src/calculations/odds_math.py`)
- a controlled sportsbook benchmark (`data/current_odds.json`,
  `data/test_scenarios.json`)
- deterministic ground truth (`data/ground_truth.json`,
  `src/evaluation/ground_truth.py`)
- current/stale sportsbook data (`data/historical_odds.json`)
- two-sided markets (both mutually exclusive outcomes for 4 core
  moneyline games)
- an `OddsProvider` abstraction (`src/providers/base.py`)
- a controlled JSON provider (`src/providers/controlled.py`)
- a sportsbook tool layer (`src/tools/sportsbook_tools.py`)
- best-line logic (`SportsbookTools.find_best_line`)
- a controlled RAG corpus (`data/rag_documents/corpus.jsonl`,
  `src/rag/documents.py`, `src/rag/build_corpus.py`)
- local embeddings (`src/rag/embeddings.py`,
  `sentence-transformers/all-MiniLM-L6-v2`, 384-dim, normalized)
- an exact FAISS vector index (`src/rag/vector_index.py`,
  `data/rag_index/`)
- a semantic retriever (`src/rag/retriever.py`) that preserves stale
  documents in baseline retrieval rather than filtering them
- a deterministic retrieval evaluation set (`data/retrieval_queries.json`,
  12 queries) and Recall@K/Hit@K evaluation
  (`src/rag/evaluate_retrieval.py`)
- core quantitative market mathematics (`src/calculations/market.py`):
  overround, no-vig/fair probabilities, market consensus,
  leave-one-sportsbook-out consensus, probability edge, market
  dispersion
- market-consensus quantitative ground truth
  (`data/quant_ground_truth.json`, `src/evaluation/quant_ground_truth.py`,
  `QuantGroundTruth`/`SportsbookValueGroundTruth`/
  `MarketDispersionGroundTruth` in `src/models.py`), kept entirely
  separate from the Milestone 4 controlled-reference
  `data/ground_truth.json` (unmodified) — 4 of 14 scenarios are
  quant-evaluable (the 4 two-sided markets from Milestone 6A), the rest
  are explicitly marked `quant_evaluable=False` with a reason, never
  fabricated or silently skipped
- a common agent contract (`src/agents/base.py`: `Agent` ABC,
  `AgentRequest` — no ground truth fields) and a deterministic RAG
  evidence pipeline (`src/agents/rag_evidence.py`: `RagEvidenceItem`,
  `RagEvidenceBundle`, `build_rag_evidence_bundle`, `render_rag_context`,
  `compute_evidence_diagnostics`) that reuses the Milestone 6C
  `Retriever` verbatim — preserves rank/score/freshness exactly as
  retrieved (stale documents are never filtered, reordered, or
  "corrected"), makes zero LLM calls, and cannot reach `src.tools`,
  `src.providers`, `data/current_odds.json`, or any ground-truth file
  (enforced by AST-based boundary tests)
- `BettingAnalysis` extended additively with optional
  `market_reference_probability`, `probability_edge`, `best_sportsbooks`
  (ties), `estimated_true_probability`, `expected_value`, `positive_ev`
  (now Optional — never fabricated when no reference probability can be
  derived), and a new `status: AnalysisStatus` field (`ok` |
  `insufficient_quant_evidence`), enforced consistent by a model
  validator
- the first concrete architecture, `RagOnlyAgent`
  (`src/agents/rag_agent.py`): RAG evidence (Milestone 8A, reused
  verbatim) -> LLM structured extraction (`src/agents/llm_client.py`
  `LLMClient` Protocol + `AnthropicLLMClient`, `claude-opus-4-8`,
  `src/agents/extraction.py` extraction schema/prompt) -> provenance
  validation (`validate_extraction_provenance`, rejects hallucinated
  sportsbooks/odds/source-document-ids, strips-not-rejects unverifiable
  opposing-side claims) -> shared deterministic quant engine
  (`src/calculations/`, never reimplemented) -> `BettingAnalysis`
  (`architecture=rag`). The LLM never performs betting arithmetic. Stale
  RAG evidence is preserved honestly, never "corrected." Zero sportsbook
  prices surviving validation raises a typed `RagAnalysisIncomplete`
  (carries the full `RagAgentTrace`) rather than fabricating a result.
  Every run records a `RagAgentTrace` (retrieved doc IDs/scores,
  extraction result, rejection reasons, validation/quant status,
  retrieval/LLM/quant/total latencies, errors). Architecture isolation
  from `src.tools`/`src.providers`/ground-truth files enforced by
  AST-based tests. Unit tests use only a fake `LLMClient` (no paid API
  calls); `experiments/run_rag_smoke_test.py` is the manual,
  credential-gated real-API smoke test (not part of the automated suite;
  a missing `ANTHROPIC_API_KEY` does not fail the milestone).

- the second concrete architecture, `ToolCallingAgent`
  (`src/agents/tool_agent.py`): a bounded multi-turn tool-calling loop
  (`MAX_TOOL_ITERATIONS = 6`) drives an LLM against five LLM-callable
  tool schemas (`src/agents/tool_schemas.py`: `get_games`, `get_game`,
  `get_odds`, `get_sportsbook_odds`, `find_best_line`) that validate
  inputs and thinly wrap the existing `SportsbookTools` -> `OddsProvider`
  layer — never reimplemented. `src/agents/llm_client.py` gained a
  `ToolCallingLLMClient` protocol (`create_turn`) implemented by the same
  `AnthropicLLMClient` class used by the RAG-only agent — one client
  class, one model/max_tokens/effort configuration, one secret path,
  shared across both architectures. The agent's final `BettingAnalysis`
  is built exclusively from the actual Pydantic objects each tool call
  returned (folded into an internal `market_state`), never from the
  LLM's prose — so a hallucinated final claim about a sportsbook or
  price cannot reach the output. Redundant tool calls (identical
  name+arguments) are flagged, not hidden; a tool failure (unknown
  sportsbook/game/market/outcome) is recorded explicitly and never
  substituted; hitting the loop bound without the model naturally
  finishing raises `ToolAnalysisIncomplete` (mirrors `RagAnalysisIncomplete`)
  rather than fabricating a result. Every run records a `ToolAgentTrace`
  (tool calls with arguments/success/redundancy/latency, call order,
  redundant-call count, validation/quant status, LLM-decision/tool-
  execution/quant/total latencies, errors). Architecture isolation from
  `src.rag`/ground-truth files enforced by AST-based tests. Unit tests
  use only a fake `ToolCallingLLMClient` (no paid API calls);
  `experiments/run_tool_agent_smoke_test.py` is the manual,
  credential-gated real-API smoke test (real Claude API calls only — the
  sportsbook data source remains the controlled provider throughout).

- an end-to-end evaluation harness for the tool-calling agent
  (`src/evaluation/tool_agent_evaluation.py`, Milestone 9B):
  `evaluate_scenario()`/`evaluate_scenarios()` drive `ToolCallingAgent`
  against a representative 11-scenario controlled-benchmark subset
  (positive/negative/mixed-sign odds, a best-line tie, a missing-
  sportsbook case, a current/stale freshness case, moneyline/spread/
  total markets, quant-evaluable positive- and negative-EV cases) and
  compare its `BettingAnalysis` output against `GroundTruth` (best
  line/odds) and `QuantGroundTruth` (market-consensus EV/reference
  probability) — entirely outside the agent; ground truth never enters
  an `AgentRequest`, tool prompt, tool result, LLM message, or quant
  input (enforced by an AST-based test forbidding `src.evaluation`
  imports in `src/agents/tool_agent.py`/`tool_schemas.py`). Comparisons
  use set-based tie semantics for best-line accuracy, exact-integer
  comparison for odds, and `abs()` error for EV/reference-probability
  (no rounding). `ToolAgentEvaluationResult` records execution status
  (`ExecutionStatus`: `success`, `quant_insufficient_data`,
  `tool_argument_error`, `tool_data_missing`, `tool_loop_limit`,
  `llm_output_invalid`, `quant_validation_error`, `final_output_invalid`
  — never collapsed into a generic "error"), best-line/odds/EV/
  market-reference correctness, freshness correctness (verified against
  `data/historical_odds.json`), completeness (`sportsbooks_considered`
  vs. `GroundTruth.expected_sportsbooks`), an independent hallucination
  re-check (re-queries `SportsbookTools` directly for the claimed
  best-line price — never trusts the agent's own trace), tool-call
  efficiency (count/unique/redundant/failed), and the full latency
  breakdown. `DeterministicToolPolicyLLMClient` is the default fake LLM:
  a fixed, request-driven policy (discover the game, gather the
  requested outcome's current odds, then the opposing outcome's if
  moneyline) that never reads ground truth — authoritative numbers still
  come only from tool output + the shared quant engine. On the
  11-scenario default set: 100% best-line/best-odds/EV-classification/
  freshness accuracy, 0.0 mean EV and market-reference absolute error,
  100% completeness, 0% hallucination rate, 0 redundant tool calls,
  reproducible byte-for-byte (excluding latency) across repeated runs.
  `python -m src.evaluation.tool_agent_evaluation` is the evaluation
  command (`--json` for machine-readable output);
  `experiments/run_tool_agent_real_llm_evaluation.py` is the manual,
  credential-gated real-API evaluation over a 5-scenario subset (not
  part of the automated suite; its results never affect pytest
  pass/fail).

- the third and final concrete architecture, `HybridAgent`
  (`src/agents/hybrid_agent.py`, Milestone 10A): the only agent module
  permitted to access both the RAG evidence pipeline and the sportsbook
  tool layer. Reuses every existing component verbatim — `build_rag_
  evidence_bundle`/`render_rag_context` (Milestone 8A),
  `RAG_EXTRACTION_SYSTEM_PROMPT`/`ExtractedMarketEvidence`/
  `validate_extraction_provenance` (Milestone 8B), `TOOL_SCHEMAS`/
  `execute_tool`/`TOOL_AGENT_SYSTEM_PROMPT`/`MAX_TOOL_ITERATIONS`
  (Milestone 9A) — nothing is reimplemented. New deterministic-only
  module `src/agents/hybrid_reconciliation.py`
  (`HybridMarketRecord`/`reconcile_outcome`) decides, per
  (sportsbook, outcome), which of a RAG-derived and a tool-derived price
  is authoritative, with **zero LLM involvement**: current structured
  tool data always wins; current-RAG-only data is used only when no tool
  coverage exists at all; stale (or freshness-unknown) RAG-only data is
  never promoted to a current price. Every conflict is recorded with an
  explicit `ConflictResolutionReason`, never averaged or silently
  dropped. The shared quant engine (`src/calculations/`) consumes only
  `HybridMarketRecord.authoritative_odds` — never a raw `tool_odds`/
  `rag_odds` field directly — so a hallucinated LLM claim structurally
  cannot reach the final numbers (same guarantee as the tool-calling
  agent, plus RAG-side provenance validation on top). `BettingAnalysis.
  sources` is populated with `SourceReference` entries for exactly what
  fed the final numbers (`architecture=hybrid`). Every run records a
  `HybridAgentTrace` (RAG doc IDs/scores/latency, full tool-call trace,
  every reconciled record with its conflict-resolution reason, source
  agreement/conflict/RAG-only/tool-only counts, an explicit
  `HybridFailureCategory` — `success`/`rag_retrieval_failure`/
  `tool_failure`/`source_reconciliation_failure`/
  `insufficient_current_data`/`llm_output_invalid`/
  `quant_insufficient_data`/`final_output_invalid`). Graceful
  degradation verified: RAG-channel failure alone does not fail the
  agent if tools are sufficient; tool-channel failure alone does not
  promote stale RAG data; both channels failing raises typed
  `HybridAnalysisIncomplete` (mirrors `RagAnalysisIncomplete`/
  `ToolAnalysisIncomplete`) rather than fabricating a result. The
  RAG-only and tool-calling agents are byte-for-byte unmodified by this
  milestone and retain their original single-channel import boundaries
  (verified by AST-based architecture-integrity tests). Unit tests use
  only a fake combined LLM client (no paid API calls);
  `experiments/run_hybrid_agent_smoke_test.py` is the manual,
  credential-gated real-API smoke test (not part of the automated suite;
  a missing `ANTHROPIC_API_KEY` does not fail the milestone).

- an end-to-end evaluation harness for the hybrid agent
  (`src/evaluation/hybrid_agent_evaluation.py`, Milestone 10B):
  `evaluate_scenario()`/`evaluate_scenarios()` drive `HybridAgent`
  against the same representative 11-scenario controlled-benchmark
  subset used for the tool-calling evaluator, and compare against
  `GroundTruth`/`QuantGroundTruth` entirely outside the agent. Reuses
  (never redefines) the tool-agent evaluator's metric primitives —
  `DEFAULT_SCENARIO_IDS`, `_stale_odds_by_scenario_key`,
  `_detect_hallucination`, `_rate`/`_mean` — so best-line/best-odds/EV/
  freshness/completeness/hallucination definitions stay identical
  across architectures. `execution_status` reuses `HybridAgentTrace`'s
  own `HybridFailureCategory` enum directly rather than inventing a
  parallel one. Adds hybrid-specific metrics: RAG/tool agreement and
  conflict counts, correct-conflict-resolution count (current-tool-data
  precedence applied), conflict-resolution accuracy (undefined, not
  zero, when no conflicts exist), stale-RAG-conflict count, a
  stale-RAG-incorrectly-promoted regression guard (always 0 by
  construction), tool-only-recovery and RAG-only-observed counts, and a
  source-reconciliation-failure flag. `DeterministicHybridPolicyLLMClient`
  is the default fake LLM: its tool-calling side delegates verbatim to
  `DeterministicToolPolicyLLMClient` (Milestone 9B); its RAG-extraction
  side deterministically parses whatever the RAG evidence pipeline
  actually retrieved into a structured, honest extraction, filtered to
  the requesting game's own `game_id` (a real bug — cross-game retrieval
  noise being treated as a same-game "opposing outcome" — was caught and
  fixed during this milestone's own evaluation-harness development, not
  in any Milestone 10A code). On the 11-scenario default set: 100%
  best-line/best-odds/EV-classification/freshness accuracy, 0.0 mean EV
  and market-reference absolute error, 100% conflict-resolution
  accuracy, 0 stale-RAG-incorrectly-promoted, 0% hallucination rate,
  reproducible byte-for-byte (excluding latency) across repeated runs.
  `python -m src.evaluation.hybrid_agent_evaluation` (`--json` optional)
  is the evaluation command; `experiments/
  run_hybrid_agent_real_llm_evaluation.py` is the manual, credential-
  gated 5-scenario real-API evaluation (not part of the automated suite;
  its results never affect pytest pass/fail). RAG-only/tool-calling/
  hybrid `AgentRequest`→`BettingAnalysis` contract parity and all three
  architectures' access-boundary isolation re-verified intact.

- a unified evaluation framework (`src/evaluation/metrics.py`, Milestone
  11) — one shared metric layer used identically by all three
  per-architecture evaluators, removing the duplication that would
  otherwise accumulate as `calculate_rag_best_line_accuracy()` /
  `calculate_tool_best_line_accuracy()` / `calculate_hybrid_best_line_
  accuracy()`-style copies. Reuses `src.models.ArchitectureType`
  directly as the one canonical architecture identifier (never a
  parallel string). Defines: a unified failure taxonomy
  (`FailureCategory`, 14 values, e.g. `retrieval_failure`,
  `tool_loop_limit`, `provenance_validation_failure`,
  `source_reconciliation_failure`); pure metric formulas —
  `best_line_correct` (set-based tie semantics), `best_odds_correct`
  (exact integer equality), `ev_classification_correct`/
  `ev_absolute_error`/`market_reference_absolute_error` (unrounded,
  `None` — never 0 — when not applicable), `evaluate_freshness`
  (`FreshnessDiagnostics`: judged from the final authoritative value,
  with an explicit known-stale-value match indicator so a merely-wrong
  prediction is never mislabeled "stale" without support),
  `completeness`, `unsupported_claim_rate`; a consistency metric
  (`ConsistencySignature`/`compute_consistency` — modal-signature-count
  / total-runs over best-sportsbooks/best-odds/positive-ev/market-
  reference-probability/EV, deliberately excluding latency and
  reasoning-summary text; defined and unit-tested here, exercised with
  real repeated runs starting Milestone 12); N/A-aware generic
  aggregation (`rate`/`mean`/`median`/`population_stdev`/`minimum`/
  `maximum`, all ignoring `None` rather than treating it as 0); the
  common per-run result (`EvaluationResult`) and `MetricApplicability`
  table (Recall@K/Hit@K deliberately excluded — those remain
  `src/rag/evaluate_retrieval.py`'s own, Milestone 6C, domain,
  referenced but never duplicated); `ArchitectureSummary` (preserves
  full raw per-run results, not just aggregates) and
  `ArchitectureComparison` (holds only data — no automatic "winner").
  `tool_agent_evaluation.py`/`hybrid_agent_evaluation.py` were refactored
  to call these shared functions internally (their own result models,
  field names, and all pre-existing tests are unchanged) and each
  gained a `to_common_result()` converter. A real bug was found and
  fixed during this refactor: the RAG-extraction fake-LLM parser
  (`extract_honest_rag_evidence`, shared by the hybrid and new RAG
  evaluators) previously let a later-parsed document silently overwrite
  an earlier one for the same (sportsbook, outcome) — e.g. a stale
  snapshot overwriting a higher-ranked current one — now the
  highest-ranked (first-seen) document wins, a deterministic relevance
  tie-break that doesn't thumb the scale toward "current." New
  `src/evaluation/rag_agent_evaluation.py` fills the gap noted in
  milestones/current.md (the RAG-only agent had no dedicated evaluator
  before this milestone) — same shape as the tool/hybrid evaluators,
  `execution_status` typed directly as the shared `FailureCategory`
  (no intermediate architecture-specific enum), evaluated at
  `RAG_EVALUATION_TOP_K=10` (an evaluation-configuration choice matching
  precedent already established elsewhere in the project for reliable
  two-sided corpus coverage — `RagOnlyAgent`'s own default
  `DEFAULT_RAG_TOP_K=5` is unmodified). On the shared 11-scenario
  default set, all three architectures now reach 100%
  best-line/best-odds/EV-classification/freshness accuracy and 0%
  hallucination rate with byte-for-byte (excluding latency)
  reproducibility; a full RAG/TOOL/HYBRID `ArchitectureComparison` round
  trips through JSON with per-run results preserved.

- a controlled experiment runner (`src/experiments/`, Milestone 12):
  `config.py` (`ExperimentConfig` — one frozen set of controls: model,
  effort, temperature [`None`, this model family rejects it], RAG top_k,
  max tool iterations, repetitions, execution mode, applied identically
  to all three architectures; `ExperimentScenario`/
  `build_scenario_manifest()` — one plain, architecture-neutral
  canonical query per scenario derived only from market type and
  selected outcome, e.g. "Compare the available moneyline prices for Los
  Angeles Lakers and identify the best current value.", `_canonical_
  query()` structurally takes no architecture parameter; deterministic
  left-rotation execution-order policy across repetitions [Rep1
  RAG→TOOL→HYBRID, Rep2 TOOL→HYBRID→RAG, Rep3 HYBRID→RAG→TOOL]; SHA-256
  checksums of every controlled artifact for reproducibility metadata,
  never an API key); `agent_factory.py` (`create_agent(architecture,
  config)` — one centralized factory for all three architectures; MOCK
  mode reuses the exact Milestone 9B/10B/11 deterministic fake-LLM
  policies verbatim, still driving the real retrieval/tool pipelines;
  REAL mode uses one `AnthropicLLMClient` configuration shared across
  architectures); `runner.py` (`run_experiment()` drives every
  (architecture, scenario, repetition) combination through the same
  Milestone 11 evaluator functions — `_EVALUATE_SCENARIO`/
  `_TO_COMMON_RESULT` hold direct references to
  `rag_agent_evaluation.py`/`tool_agent_evaluation.py`/
  `hybrid_agent_evaluation.py`'s real functions, verified by identity in
  tests, never a runner-local accuracy formula; ground truth reaches
  only the evaluator call, never `AgentRequest` [verified by AST scan];
  one architecture's agent-construction failure is recorded as an
  `UNKNOWN_FAILURE` run rather than aborting the batch; raw,
  never-pre-aggregated results are persisted one JSON object per line to
  `results/experiments/<experiment_id>/raw_results.jsonl` alongside
  `config.json`/`manifest.json`/`summary.json`; re-running an experiment
  without `resume=True` raises `FileExistsError`, `resume=True` appends
  only unrecorded `(architecture, scenario_id, repetition)` keys; post-
  run `ArchitectureSummary`/`ArchitectureComparison` aggregation and
  cross-repetition `consistency` reused directly from Milestone 11 — the
  first milestone to exercise that calculation with real repeated-run
  data). CLI: `python -m src.experiments.runner --mode {mock,real}
  [--repetitions] [--output-dir] [--scenario-ids] [--architectures]
  [--experiment-id] [--dry-run] [--resume]`; `--dry-run` previews the
  resolved config without executing; REAL mode probes connectivity first
  and prints "REAL EXPERIMENT: NOT RUN" and exits cleanly (rather than
  silently falling back to mock) when no credentials are available. An
  18-run [3 architectures x 3 scenarios x 2 repetitions]
  INFRASTRUCTURE-VALIDATION-ONLY mock demonstration ran 18/18 successful
  with full per-architecture summaries, explicitly labeled "NOT FINAL
  RESEARCH RESULTS"; the fairer, architecture-neutral canonical query
  (deliberately less retrieval-optimized than the sportsbook-name-heavy
  queries the Milestone 9B-11 evaluators use for their own metric
  testing) combined with `RagOnlyAgent`'s true default `top_k=5` causes
  RAG's mock-mode `ev_classification_accuracy` to come back `None` in
  these demo runs — an honest, expected research-infrastructure finding
  recorded as-is, not tuned away. Mock-mode reproducibility verified
  record-for-record across two independent output directories
  (excluding timestamps, experiment/run IDs, and all latency
  measurements). No inferential statistics, no dashboard, and no changes
  to any of the three agents' or the unified evaluator's behavior.

- a Streamlit research dashboard (`dashboard/`, Milestone 13): a thin UI
  layer over the existing agents/experiments/evaluation modules — never
  defines a betting/quant formula, ground-truth value, or agent-behavior
  change of its own (`dashboard/data_loader.py` constructs agents only
  via `src.experiments.agent_factory.create_agent`, the same factory the
  experiment runner uses; `dashboard/charts.py`/`research_view.py`
  aggregate only through existing `src.evaluation.metrics`/
  `hybrid_agent_evaluation.summarize_results` functions). Two modes:
  **Demo** (`dashboard/demo_view.py`) — pick a scenario from the full
  controlled scenario manifest (`src.experiments.config.
  build_scenario_manifest`) and an architecture (RAG/TOOL/HYBRID), click
  "Run Analysis" to execute it live (MOCK reuses the Milestone 9B/10B/11
  deterministic fake-LLM policies at no API cost; REAL calls the
  configured Anthropic model and degrades to a clear, non-crashing
  message when no credentials are configured), and inspect the resulting
  `BettingAnalysis`, a per-sportsbook comparison table, and the full
  architecture trace (RAG: retrieved document IDs/rank/score/freshness;
  Tool: tool calls with arguments/success/latency; Hybrid: both plus
  every reconciled record's authoritative source and conflict-resolution
  reason, with an explicit RAG-snapshot-vs-current-tool freshness-
  conflict display). Nothing is cached across "Run Analysis" clicks
  except the expensive, read-only Retriever/FAISS-index and
  SportsbookTools construction (`st.cache_resource`) — every click is a
  genuine, independent agent execution. **Research Comparison**
  (`dashboard/research_view.py`) — loads a persisted Milestone 12
  experiment directory (`results/experiments/<experiment_id>/`) read-
  only and renders experiment metadata (with an unmissable "MOCK —
  INFRASTRUCTURE VALIDATION ONLY" banner whenever `execution_mode=mock`,
  never presented as a research finding), per-architecture summaries,
  7 bar-chart comparisons (best-line/EV-classification/freshness
  accuracy, completeness, unsupported-claim rate, consistency, mean
  latency — no automatic "winner"), per-scenario drill-down (with
  ground truth labeled "EXPECTED / GROUND TRUTH" and shown only in this
  research context, never fed back into an agent), repetition/
  consistency inspection, failure analysis (grouped by the Milestone 11
  `FailureCategory` taxonomy), hybrid conflict analysis, and filterable
  raw-result access plus optional CSV/JSON export (never overwriting the
  original experiment files). Percentage/N/A/small-nonzero-error
  formatting (`dashboard/formatting.py`) never displays a missing value
  as a misleading 0. An AST-based structural test
  (`tests/test_dashboard_structure.py`) confirms `dashboard/` defines no
  quant/odds formula of its own and never constructs a concrete `Agent`
  subclass directly. Verified via `streamlit.testing.v1.AppTest` end to
  end for all three architectures (including a live freshness-conflict
  case, S009) and the Research view (loaded a real 24-run mock
  experiment: 3 architectures x 4 scenarios x 2 repetitions), plus a real
  `streamlit run dashboard/app.py` server launch (HTTP 200). No live
  sportsbook API introduced (`ControlledOddsProvider` throughout); no
  final experiment run; no inferential statistics; no research
  conclusion generated.

- a local, real (non-Anthropic) LLM provider (`OllamaLLMClient`,
  Pre-Milestone 14A checkpoint, `src/agents/llm_client.py`): implements
  the same `LLMClient`/`ToolCallingLLMClient` protocols as
  `AnthropicLLMClient` on one class (`generate_structured` via
  schema-constrained JSON output, `create_turn` via native tool
  calling), talking to a locally hosted Ollama server over HTTP
  (`httpx`, no new orchestration framework). Provider selection is
  configuration-driven (`ExperimentConfig.llm_provider`,
  `LLMProviderName.ANTHROPIC` | `.OLLAMA`) and read by
  `src.experiments.agent_factory.build_llm_client` — never
  architecture-specific; RAG/TOOL/HYBRID always receive the identical
  provider/model/temperature for a given config. Default local model:
  `llama3.1:8b` at `temperature=0.0` (`DEFAULT_OLLAMA_MODEL`/
  `DEFAULT_OLLAMA_TEMPERATURE`), centrally recorded as `LLM_PROVIDER`/
  `LLM_MODEL` — never duplicated in any agent module. MOCK mode is
  unaffected by this field (always the Milestone 9B-11 deterministic
  fakes, regardless of `llm_provider`); local inference through
  `OllamaLLMClient` is classified as REAL execution, never MOCK. No
  agent (`rag_agent.py`/`tool_agent.py`/`hybrid_agent.py`), the quant
  engine, ground truth, the RAG corpus, or evaluation metrics were
  modified — this is a provider substitution only, verified via a live
  end-to-end smoke run of all three architectures against the real
  local model (`experiments/run_ollama_three_architecture_smoke_test.py`):
  RAG (real retrieval + real local extraction, `BettingAnalysis`
  validates), TOOL (a real tool call — `get_odds` — actually occurred,
  not merely succeeded without one, `BettingAnalysis` validates), and
  HYBRID (real RAG+tool workflow, `BettingAnalysis` validates; the
  deterministic `CURRENT_TOOL_DATA_PRECEDENCE` reconciliation policy
  itself is unmodified and independently re-verified both by a direct
  synthetic-conflict check and by `tests/test_hybrid_reconciliation.py`
  — the smoke run's own scenarios (S001, S009-S011) happened not to
  surface a live source conflict this session, an honestly-reported
  local-model-extraction observation, not a defect). 26 new tests
  (`tests/test_ollama_llm_client.py` — message-format conversion,
  request/response handling, bounded `OllamaRequestError` handling, all
  against a mocked HTTP layer; `tests/test_llm_provider_selection.py` —
  config defaulting/round-trip, per-architecture client-type identity)
  pass with Ollama stopped, confirming standard `pytest -q` never
  requires a running local server. A follow-up readiness-verification
  pass (before Milestone 14A began) confirmed this integration needed
  no rework and found one real defect: `probe_real_llm_connectivity`
  (`src/experiments/runner.py`) was hard-coded to always build an
  `AnthropicLLMClient` regardless of the configured provider, which
  would have made any Ollama-configured real run always report "REAL
  EXPERIMENT: NOT RUN" — fixed to build its probe client via
  `build_llm_client()` (configuration-driven, like every other real-mode
  client construction), covered by a new regression test. One artifact-
  fingerprint gap was also closed: `src/experiments/fingerprint.py` now
  additionally hashes `src/experiments/config.py` itself, so a change to
  the canonical query template's wording — not just its input data —
  is detectable in the pre/post integrity check.

- Milestone 14A — freeze and run the final controlled experiment
  (`experiments/final_experiment.json` frozen to `llm_provider=ollama`,
  `model_name=llama3.1:8b`, `temperature=0.0`, `rag_top_k=5`,
  `max_tool_iterations=6`, the 11-scenario `DEFAULT_SCENARIO_IDS` set,
  `repetitions=10`, rotating execution order): the real, non-mock final
  dataset — 3 architectures x 11 scenarios x 10 repetitions = 330
  observations — executed via the unmodified Milestone 12
  `run_experiment()` through a new orchestrator
  (`experiments/run_final_experiment.py`): preflight (a full nested
  `pytest -q`, ground-truth/quant-ground-truth regeneration-vs-file
  diffs, a live RAG-index/retrieval smoke query, provider/tools smoke
  checks, a direct synthetic-conflict call confirming
  `CURRENT_TOOL_DATA_PRECEDENCE` is intact) → a provider-aware real-
  inference connectivity probe → the run itself → pre/post artifact
  fingerprinting (9 hashes: benchmark scenarios, both ground-truth
  files, RAG corpus, RAG index config, both system prompts, the
  canonical-query-template module, the frozen config file — all 9
  MATCHED) → a full dataset validation report
  (`src/experiments/validation.py`). Result, persisted at
  `results/experiments/final_v1/` (`config.json`, `manifest.json`,
  `metadata.json`, `artifact_hashes.json`, `raw_results.jsonl`,
  `summary.json`/`descriptive_summary.json`,
  `dataset_validation_report.json`): 330/330 expected runs recorded
  (110 RAG / 110 TOOL / 110 HYBRID, exactly even), zero duplicate or
  missing run keys, a complete 10/10/10 scenario-coverage matrix across
  all 11 scenarios, zero raw-result schema-validation errors,
  ground-truth isolation and architecture isolation both re-audited
  PASS post-execution, and the deterministic architecture rotation
  confirmed actually applied from the persisted
  `execution_order_position` field (not merely assumed). 290 successful
  observations (`quant_insufficient_data` — a validated best line
  without enough two-sided data for an EV verdict; RAG 90, TOOL 100,
  HYBRID 100) and 40 failed observations preserved as-recorded, never
  dropped or retried: RAG's 20 (`insufficient_retrieved_evidence`, all
  on the spread/total scenarios S012/S013) are a genuine
  architecture-level evidence-pipeline gap; TOOL's 10 and HYBRID's 10
  (all on S012) share one root cause — a client-side 180s
  `OllamaRequestError` read-timeout on that scenario's tool-calling
  turns — an infrastructure/runtime characteristic of this local
  model+scenario combination, distinguished from architecture-level
  failures via the raw `errors` field rather than reclassified after
  the fact. Median observation latency ~16.2s (mean ~24.8s, range
  3.7s-183.9s) confirms genuine local inference throughout, never the
  deterministic mock policies. Zero hybrid source conflicts were
  recorded across all 110 hybrid observations in this dataset — the
  reconciliation policy itself remains independently verified intact
  (synthetic direct call + the unmodified, fully-passing
  `tests/test_hybrid_reconciliation.py`); this is a real, honestly-
  reported characteristic of how this local model's RAG-side extraction
  interacts with the existing, unmodified provenance validator, not a
  defect. No agent, prompt, quant formula, ground truth, RAG corpus, or
  evaluation metric was modified during or after execution (confirmed
  via `git diff` and the 9-way pre/post hash match). No arbitrary
  retries; no inferential statistics performed; no "winning"
  architecture declared. Dataset explicitly validated eligible for
  Milestone 14B's statistical analysis.

- Milestone 14B — statistical analysis and research findings
  (`src/analysis/`, read-only against `results/experiments/final_v1/` —
  never modifies raw_results.jsonl/config.json/manifest.json, verified
  by file-hash/mtime checks and a dedicated test): paired alignment by
  (scenario_id, repetition) across architectures and by scenario_id
  alone for consistency (`pairing.py`); Wilson score confidence
  intervals (`confidence_intervals.py`); exact-binomial McNemar,
  Wilcoxon signed-rank, and Holm-Bonferroni correction
  (`pairwise_tests.py`, via `scipy.stats`); Friedman omnibus
  (`omnibus_tests.py`); a frozen metric-family framework — 4 binary
  metrics (best-line/best-odds/EV-classification/freshness correctness)
  via McNemar, 4 continuous metrics (EV/market-reference absolute
  error, completeness, total latency) via Wilcoxon, 3 omnibus metrics
  (completeness, latency, consistency) via Friedman — each of the 3
  architecture pairs Holm-corrected within its own metric family, never
  pooled globally (`comparisons.py`). Every metric formula is reused
  directly from `src.evaluation.metrics`/
  `hybrid_agent_evaluation.summarize_results` (Milestone 11), never
  redefined. Two real bugs were found by new synthetic tests (never
  hard-coded to the actual final result) and fixed before touching the
  real dataset's output: `loading.py` crashed with a raw pydantic
  exception instead of `FinalDatasetInvalidError` on a corrupted raw
  record (fixed to reuse `validation.load_and_validate_raw_results`
  rather than a second raw-line parser); `omnibus_tests.py`'s Friedman
  degeneracy check conflated "every block holds an identical triple"
  (a maximally significant, perfectly-consistent-ranking case) with
  "every block is a three-way tie" (the true zero-variance case scipy
  can't compute), misreporting the former as "no variation, p=1.0"
  (fixed to check within-block ties only, and to detect scipy's silent
  NaN return for genuine degeneracy). `python -m src.analysis.
  final_analysis` is the one reproducible entry point, writing 8
  machine-readable files, 6 figures (Wilson-interval error bars, N/A
  rendered as an honest label rather than a misleading zero-height bar,
  no 3D), and `findings.md` to `results/experiments/final_v1/analysis/`.
  Result: every EV-classification/EV-error/market-reference-error
  comparison correctly reports N/A (zero observations in the frozen
  dataset ever reached a full quant verdict) rather than a fabricated
  p-value; RAG's best-line/best-odds accuracy (88.9%) is significantly
  lower than TOOL/HYBRID's (100%, Holm-adjusted p=0.0059) while TOOL and
  HYBRID are statistically indistinguishable from each other (p=1.0,
  zero discordant pairs); freshness (100% for all three) and
  consistency (1.0 for all three, Friedman correctly degenerate) show no
  distinguishable difference; RAG's completeness (72.7%) is
  significantly lower than TOOL/HYBRID's (90.9%); latency differs
  significantly across all three pairs (TOOL fastest, HYBRID slowest);
  zero hallucinations; hybrid conflict-resolution accuracy is correctly
  N/A (zero live conflicts in this dataset). `findings.md` answers the
  research question directly without forcing a universal winner,
  distinguishes statistical from practical significance throughout,
  documents limitations including local-model-specific capability
  findings, and includes the independent predictive-ML-model extension
  as documented future work — not implemented. No agent, prompt, quant
  formula, ground truth, or evaluation metric was modified; no selective
  rerunning; 70 new tests, all synthetic-data-based.

- Milestone 15 — research conclusions, manuscript, and presentation
  support (`docs/`, documentation-only — no agent, prompt, model
  configuration, scenario, raw result, ground truth, statistical test,
  or metric definition was modified; verified via `git diff` and file
  mtime checks against `results/experiments/final_v1/`): every numeric
  claim in the new documents is traced directly to a fresh read of
  `results/experiments/final_v1/analysis/`, never reconstructed from
  memory. `docs/FINAL_RESEARCH_SUMMARY.md` (research question,
  experimental design, the 6-step quant pipeline, primary/secondary
  results reproduced verbatim from `findings.md`, a direct
  evidence-calibrated answer to the research question, limitations,
  future work including the ML-predictive-model extension diagram —
  explicitly not implemented); `docs/MANUSCRIPT_OUTLINE.md` (structural
  outline, Methods 3.1-3.7 limited to methods actually used);
  `docs/MANUSCRIPT_DRAFT.md` (~450-line first-draft manuscript — Abstract
  through Conclusion, 8 `[CITATION NEEDED]` markers for literature
  claims, zero fabricated citations or statistics, every number
  cross-verified against `pairwise_comparisons.json`/
  `descriptive_statistics.json`); `docs/PRESENTATION_STORYBOARD.md`
  (12-slide content plan, not slides); `docs/RESULTS_ASSET_INDEX.md`
  (full inventory of Figures 1-6/Tables 1-5/omnibus table with path,
  contents, manuscript section, presentation slide, source artifact);
  `docs/REPRODUCIBILITY.md` (10-step workflow, MOCK vs. REAL execution
  mode explicitly distinguished, no secrets embedded);
  `docs/CITATION_NEEDS.md` (10-row audit of every `[CITATION NEEDED]`
  marker — no sources searched, browsed, or invented). `docs/
  ARCHITECTURE.md` rewritten (previously stale since Milestone 6C —
  described the RAG pipeline, agents, quant engine, and dashboard as
  "future" despite being fully implemented since Milestones 6C-13; now
  reflects the completed system, including the Milestone 12
  `src/experiments/` and Milestone 14B `src/analysis/` layers it
  previously omitted) — no working implementation code changed.
  Terminology audit (Section 11): market-implied/no-vig consensus
  probability is never conflated with true win probability across all
  new documents, zero instances of forbidden absolute language
  ("proved," "guarantee(s/d)," "always better"), and every "positive
  EV"/profitability reference is correctly scoped to this project's own
  methodology, never claiming real-world profitability. Conclusion audit
  (Section 12): no architecture is declared a universal winner; the
  stated conclusion is evidence-calibrated (TOOL/HYBRID beat RAG on
  accuracy/completeness, TOOL beats HYBRID on latency,
  consistency/freshness indistinguishable across all three) with
  tradeoffs stated explicitly. Full pytest suite re-run at the end of
  this milestone: 1097 passed / 0 failed, confirming the
  documentation-only work did not alter application behavior.

The project does **not** yet have:

- live sportsbook API integration (an optional future extension, behind
  the existing `OddsProvider` abstraction)
- an independently trained/calibrated predictive ML model compared
  against the market-implied no-vig consensus (documented future work,
  not implemented — see `docs/FINAL_RESEARCH_SUMMARY.md`)
- filled-in citations for the `[CITATION NEEDED]` markers in
  `docs/MANUSCRIPT_DRAFT.md` (tracked in `docs/CITATION_NEEDS.md`;
  explicitly deferred, no sources searched or invented)

## Important Decisions

- Controlled data is the primary experimental benchmark.
- Live sportsbook APIs are an optional future extension.
- Future APIs must sit behind `OddsProvider`.
- Structured tool data represents current sportsbook information.
- RAG may intentionally contain stale sportsbook snapshots.
- Two-sided markets support later no-vig calculations.
- Shared deterministic math must be used across architectures.
- The quant extension (implied probability, no-vig probability,
  leave-one-sportsbook-out market consensus, probability edge, expected
  value, market dispersion) is implemented in
  `src/calculations/market.py` (Milestone 7A) and wired into
  `data/quant_ground_truth.json` (Milestone 7B). See
  `docs/QUANT_STRATEGY.md`.
- Controlled-reference ground truth (`data/ground_truth.json`,
  `estimated_true_probability`-based) and market-consensus ground truth
  (`data/quant_ground_truth.json`, leave-one-out-based) are two distinct,
  separately-generated files — neither replaces the other.
- Market consensus is a market-derived reference probability, not
  objective true probability.
- Ground truth must never be exposed to RAG or agent prompts.

Full rationale for each decision: `docs/DECISIONS.md`.

## Retrieval Quality (Milestone 6C)

Baseline retrieval quality against the 12-query evaluation set
(`data/retrieval_queries.json`): Recall@1 0.60, Recall@3 0.92, Recall@5
0.98; Hit@1 0.75, Hit@3 1.00, Hit@5 1.00; 0 failed queries at k=5. Note:
in the freshness test case, the intentionally stale document
outranked its current counterpart (semantic similarity does not account
for freshness) — this is a real, unforced observation about baseline
semantic retrieval, not a defect, and is exactly the kind of behavior
Milestone 8+ (RAG-only agent) will need to be measured against.
