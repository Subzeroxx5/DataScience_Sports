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
```

## Current Documentation Checkpoint

```text
6B.5 — Project Documentation and Agent Instructions (COMPLETE)
```

## Next Functional Milestone

```text
Milestone 12 — Controlled Experiment Runner
```

## Current Test Baseline

```text
Current full test suite:
849 passed / 0 failed
```

(Recorded from `pytest -q` at the end of Milestone 11.)

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

The project does **not** yet have:

- an experiment runner
- a dashboard

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
