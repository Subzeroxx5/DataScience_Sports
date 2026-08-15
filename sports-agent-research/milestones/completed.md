# Completed Milestones

- Milestone 1 — Project foundation and research design — COMPLETE
- Milestone 2 — Core Pydantic models — COMPLETE
- Milestone 3 — Deterministic betting mathematics — COMPLETE
- Milestone 4 — Controlled sportsbook benchmark and ground truth — COMPLETE
- Milestone 5 — Provider and sportsbook tool subsystem — COMPLETE
- Milestone 6A — Two-sided market readiness — COMPLETE
- Milestone 6B — Controlled RAG corpus — COMPLETE
- Checkpoint 6B.5 — Project documentation and agent instructions — COMPLETE
- Milestone 6C — Embeddings, vector index, and retrieval evaluation — COMPLETE
  (414 passed / 0 failed; see final report in session transcript for
  Recall@K/Hit@K figures and the honest freshness-ranking finding)
- Milestone 7A — Core quantitative market mathematics — COMPLETE
  (462 passed / 0 failed; overround, no-vig probabilities, market
  consensus, leave-one-sportsbook-out consensus, probability edge, and
  market dispersion implemented in `src/calculations/market.py`; ground
  truth and RAG corpus verified byte-unchanged)
- Milestone 7B — Quantitative ground-truth integration — COMPLETE
  (496 passed / 0 failed; new `data/quant_ground_truth.json` +
  `src/evaluation/quant_ground_truth.py` + `QuantGroundTruth`/
  `SportsbookValueGroundTruth`/`MarketDispersionGroundTruth` models;
  4/14 scenarios quant-evaluable, 15 sportsbook analyses, 2 positive EV /
  13 negative EV; `data/ground_truth.json` and RAG corpus verified
  byte-unchanged; reproducible byte-for-byte across repeated generation)
- Milestone 8A — RAG evidence pipeline and agent contract — COMPLETE
  (551 passed / 0 failed; `src/agents/base.py` [`Agent` ABC, `AgentRequest`]
  + `src/agents/rag_evidence.py` [`RagEvidenceItem`, `RagEvidenceBundle`,
  `build_rag_evidence_bundle`, `render_rag_context`,
  `compute_evidence_diagnostics`]; reuses the Milestone 6C `Retriever`
  verbatim, preserves rank/score/freshness exactly as retrieved (stale
  documents never filtered/reordered); `BettingAnalysis` extended
  additively with optional `market_reference_probability`,
  `probability_edge`, `best_sportsbooks`; zero LLM calls; architecture
  isolation from `src.tools`/`src.providers`/ground-truth files verified
  by AST-based tests)
- Milestone 8B — RAG-only LLM agent and structured quantitative analysis — COMPLETE
  (592 passed / 0 failed; first concrete architecture, `RagOnlyAgent`
  [`src/agents/rag_agent.py`]: reuses Milestone 8A's `RagEvidenceBundle`
  verbatim, extracts sportsbook prices via an LLM constrained to
  extraction-only [`src/agents/llm_client.py` `LLMClient` Protocol +
  `AnthropicLLMClient` (`claude-opus-4-8`), `src/agents/extraction.py`
  prompt/schema], validates every claim's provenance against actually-
  retrieved documents [`validate_extraction_provenance`; rejects
  hallucinated sportsbook/odds/source-document-id, strips-not-rejects
  unverifiable opposing-side claims], then applies the unmodified shared
  quant engine [`src/calculations/`]; `BettingAnalysis` extended with
  `status: AnalysisStatus` and Optional EV fields so
  `insufficient_quant_evidence` cases never fabricate an expected value;
  zero validated prices raises typed `RagAnalysisIncomplete` rather than
  a fabricated result; stale RAG evidence preserved honestly, never
  "corrected"; full `RagAgentTrace` recorded per run [retrieved doc
  IDs/scores, extraction result, rejection reasons, validation/quant
  status, retrieval/LLM/quant/total latencies, errors]; architecture
  isolation from `src.tools`/`src.providers`/ground-truth files verified
  by AST-based tests; unit tests use only a fake `LLMClient` — no paid
  API calls; `experiments/run_rag_smoke_test.py` is the manual,
  credential-gated real-API smoke test, not run in CI, not required to
  pass — real-API run this session had no `ANTHROPIC_API_KEY` set and
  failed gracefully as designed [caught, recorded in trace `errors`,
  script did not crash])
- Milestone 9A — Tool-calling agent core — COMPLETE
  (631 passed / 0 failed; second concrete architecture, `ToolCallingAgent`
  [`src/agents/tool_agent.py`]: a bounded multi-turn tool-calling loop
  [`MAX_TOOL_ITERATIONS = 6`] drives an LLM against five validated,
  LLM-callable tool schemas [`src/agents/tool_schemas.py`: `get_games`,
  `get_game`, `get_odds`, `get_sportsbook_odds`, `find_best_line`] that
  thinly wrap the existing `SportsbookTools` -> `OddsProvider` layer,
  never reimplemented; `src/agents/llm_client.py` gained a
  `ToolCallingLLMClient` protocol [`create_turn`] implemented by the
  same `AnthropicLLMClient` class the RAG-only agent uses — one client
  class, one model/max_tokens/effort configuration, one secret path,
  shared across both architectures; the final `BettingAnalysis` is built
  exclusively from the actual Pydantic objects each tool call returned
  [an internal `market_state`], never from the LLM's prose, so a
  hallucinated final claim cannot reach the output [verified: tool
  returns +120, LLM's final text claims +180, output remains +120];
  redundant tool calls flagged not hidden, tool failures [unknown
  sportsbook/game/market/outcome] recorded explicitly with no
  substitution, loop-bound violations raise typed `ToolAnalysisIncomplete`
  [mirrors `RagAnalysisIncomplete`] rather than fabricating a result;
  full `ToolAgentTrace` recorded per run [tool calls with
  arguments/success/redundancy/latency, call order, redundant-call
  count, validation/quant status, LLM-decision/tool-execution/quant/
  total latencies, errors]; architecture isolation from
  `src.rag`/ground-truth files verified by AST-based tests; unit tests
  use only a fake `ToolCallingLLMClient` — no paid API calls;
  `experiments/run_tool_agent_smoke_test.py` is the manual,
  credential-gated real-API smoke test [sportsbook data source remains
  the controlled provider throughout — only the LLM call is real], not
  run in CI, not required to pass — real-API run this session had no
  `ANTHROPIC_API_KEY` set and failed gracefully as designed [connectivity
  probe caught the auth error, printed "REAL LLM SMOKE TEST: NOT RUN",
  script did not crash])
- Milestone 9B — Tool-calling agent end-to-end verification — COMPLETE
  (668 passed / 0 failed; end-to-end evaluation harness
  [`src/evaluation/tool_agent_evaluation.py`]: `evaluate_scenario()`/
  `evaluate_scenarios()` drive the unmodified `ToolCallingAgent` against
  an 11-scenario representative controlled-benchmark subset [positive/
  negative/mixed-sign odds, a best-line tie (S007), a missing-sportsbook
  case (S008), a current/stale freshness case (S009), moneyline/spread/
  total markets, quant-evaluable positive- and negative-EV cases] and
  compare against `GroundTruth`/`QuantGroundTruth` entirely outside the
  agent — ground truth never enters an `AgentRequest`, tool prompt, tool
  result, LLM message, or quant input [AST-enforced: `src/agents/
  tool_agent.py`/`tool_schemas.py` forbidden from importing
  `src.evaluation`]; set-based tie semantics for best-line accuracy,
  exact-integer odds comparison, unrounded `abs()` EV/reference-
  probability error; `ExecutionStatus` gives 8 explicit failure/outcome
  categories, never a generic "error"; independent hallucination
  re-check re-queries `SportsbookTools` directly rather than trusting
  the agent's own trace; `DeterministicToolPolicyLLMClient` is a
  request-driven fake-LLM policy that never reads ground truth
  [discover game -> gather requested outcome -> gather opposing outcome
  if moneyline -> stop]; on the 11-scenario default set: 100%
  best-line/best-odds/EV-classification/freshness accuracy, 0.0 mean EV
  and market-reference absolute error, 100% completeness, 0%
  hallucination rate, reproducible byte-for-byte [excluding latency]
  across repeated runs, confirmed via two independent full-suite runs;
  `python -m src.evaluation.tool_agent_evaluation` [`--json` optional]
  is the evaluation command; `experiments/
  run_tool_agent_real_llm_evaluation.py` is the manual, credential-gated
  5-scenario real-API evaluation [not part of the automated suite, never
  affects pytest pass/fail — no `ANTHROPIC_API_KEY` set this session,
  printed "REAL LLM EVALUATION: NOT RUN" and exited cleanly];
  RAG-vs-tool `AgentRequest`/`BettingAnalysis` contract parity confirmed;
  no hybrid agent implemented)
- Milestone 10A — Hybrid RAG + tool-calling agent core — COMPLETE
  (710 passed / 0 failed; third and final concrete architecture,
  `HybridAgent` [`src/agents/hybrid_agent.py`] — the only agent module
  permitted to access both RAG evidence and sportsbook tools; reuses
  every existing component verbatim [`build_rag_evidence_bundle`/
  `render_rag_context`, `RAG_EXTRACTION_SYSTEM_PROMPT`/
  `ExtractedMarketEvidence`/`validate_extraction_provenance`,
  `TOOL_SCHEMAS`/`execute_tool`/`TOOL_AGENT_SYSTEM_PROMPT`/
  `MAX_TOOL_ITERATIONS`] — nothing duplicated; new pure-Python
  `src/agents/hybrid_reconciliation.py` [`HybridMarketRecord`/
  `reconcile_outcome`] deterministically decides per (sportsbook,
  outcome) which of a RAG-derived and tool-derived price is
  authoritative [current tool data always wins; current-RAG-only used
  only with zero tool coverage; stale/unknown-freshness RAG-only never
  promoted; every conflict recorded with an explicit
  `ConflictResolutionReason`, never averaged]; shared quant engine
  consumes only `authoritative_odds`, never a raw source field directly,
  so a hallucinated LLM claim structurally cannot reach the final
  numbers [verified: tool returns +120, LLM's final text claims +180,
  output remains +120]; `BettingAnalysis.sources` populated with exactly
  what fed the final numbers; full `HybridAgentTrace` per run
  [RAG doc IDs/scores/latency, complete tool-call trace, every
  reconciled record, source agreement/conflict/RAG-only/tool-only
  counts, explicit `HybridFailureCategory` (8 values, never a generic
  "error")]; graceful degradation verified [RAG failure alone doesn't
  fail the agent if tools suffice; tool failure alone never promotes
  stale RAG data; both failing raises typed `HybridAnalysisIncomplete`];
  RAG-only and tool-calling agents verified byte-for-byte unmodified,
  their original single-channel import boundaries intact
  [AST-enforced]; unit tests use only a fake combined LLM client — no
  paid API calls; `experiments/run_hybrid_agent_smoke_test.py` is the
  manual, credential-gated real-API smoke test [not run in CI, not
  required to pass — no `ANTHROPIC_API_KEY` set this session, printed
  "REAL LLM HYBRID SMOKE TEST: NOT RUN" and exited cleanly])
- Milestone 10B — Hybrid agent end-to-end verification — COMPLETE
  (743 passed / 0 failed; end-to-end evaluation harness
  [`src/evaluation/hybrid_agent_evaluation.py`]: `evaluate_scenario()`/
  `evaluate_scenarios()` drive the unmodified `HybridAgent` against the
  same 11-scenario representative subset used for the tool-calling
  evaluator [Milestone 9B], reusing rather than redefining its metric
  primitives [`DEFAULT_SCENARIO_IDS`, `_stale_odds_by_scenario_key`,
  `_detect_hallucination`, `_rate`/`_mean`] so cross-architecture
  comparison never has to reconcile incompatible definitions;
  `execution_status` reuses `HybridAgentTrace`'s own
  `HybridFailureCategory` enum directly; hybrid-specific metrics added
  [RAG/tool agreement/conflict counts, correct-conflict-resolution
  count, conflict-resolution accuracy (undefined not zero with no
  conflicts), stale-RAG-conflict count, a stale-RAG-incorrectly-
  promoted regression guard, tool-only-recovery/RAG-only-observed
  counts, source-reconciliation-failure flag]; freshness judged at the
  final authoritative-market-state level, not merely "was current data
  seen somewhere in the trace"; `DeterministicHybridPolicyLLMClient`
  composes `DeterministicToolPolicyLLMClient` [Milestone 9B, reused
  verbatim] for tool orchestration and adds a deterministic RAG-context
  parser that never reads ground truth; a real bug in that parser
  [cross-game retrieval noise mistaken for a same-game "opposing
  outcome," caught via a live 11-scenario run, not a defect in any
  Milestone 10A agent code] was found and fixed by scoping extraction to
  the requesting game's own `game_id`; on the 11-scenario default set:
  100% best-line/best-odds/EV-classification/freshness/conflict-
  resolution accuracy, 0.0 mean EV and market-reference absolute error,
  0 stale-RAG-incorrectly-promoted, 0% hallucination rate, reproducible
  byte-for-byte [excluding latency] across repeated runs, confirmed via
  two independent full-suite runs; `python -m
  src.evaluation.hybrid_agent_evaluation` [`--json` optional] is the
  evaluation command; `experiments/
  run_hybrid_agent_real_llm_evaluation.py` is the manual, credential-
  gated 5-scenario real-API evaluation [not part of the automated suite,
  never affects pytest pass/fail — no `ANTHROPIC_API_KEY` set this
  session, printed "REAL LLM HYBRID EVALUATION: NOT RUN" and exited
  cleanly]; RAG-only/tool-calling/hybrid `AgentRequest`→`BettingAnalysis`
  contract parity and all three architectures' access-boundary isolation
  re-verified intact; full architecture experiment runner intentionally
  NOT implemented yet)
- Milestone 11 — Unified evaluation framework — COMPLETE
  (849 passed / 0 failed; new `src/evaluation/metrics.py` — one shared
  metric layer used identically by all three per-architecture
  evaluators, eliminating architecture-specific reimplementations of
  the same formula; reuses `src.models.ArchitectureType` directly as
  the one canonical identifier; unified `FailureCategory` taxonomy [14
  values]; pure metric formulas [`best_line_correct` set-based tie
  semantics, `best_odds_correct` exact-integer, `ev_classification_
  correct`/`ev_absolute_error`/`market_reference_absolute_error`
  unrounded and `None`-not-zero when inapplicable, `evaluate_freshness`
  judged from the final authoritative value with an explicit known-
  stale-value-match diagnostic, `completeness`, `unsupported_claim_
  rate`]; consistency metric [`ConsistencySignature`/
  `compute_consistency`, modal-signature-count/total-runs, excludes
  latency/reasoning text — defined and unit-tested here, Milestone 12
  will exercise it with real repeated runs]; N/A-aware generic
  aggregation [`rate`/`mean`/`median`/`population_stdev`/`minimum`/
  `maximum`]; common per-run result [`EvaluationResult`] +
  `MetricApplicability` table [retrieval-quality Recall@K/Hit@K
  deliberately kept separate, still owned by `src/rag/
  evaluate_retrieval.py`]; `ArchitectureSummary` [preserves full raw
  per-run results] + `ArchitectureComparison` [data only, no automatic
  "winner"]; `tool_agent_evaluation.py`/`hybrid_agent_evaluation.py`
  refactored to call the shared functions internally [own result
  models/field names/all pre-existing tests unchanged], each gained
  `to_common_result()`; a real bug was found and fixed in the shared
  RAG-extraction fake-LLM parser [`extract_honest_rag_evidence`]:
  multiple retrieved documents for the same (sportsbook, outcome) — e.g.
  a stale snapshot and a higher-ranked current one — let the
  later-parsed one silently win; now the highest-ranked (first-seen)
  document wins, a deterministic relevance tie-break, not a thumb on
  the scale toward "current"; new `src/evaluation/rag_agent_evaluation.py`
  fills the previously-noted gap [RAG-only had no dedicated evaluator],
  same shape as tool/hybrid, `execution_status` typed directly as the
  shared `FailureCategory`, evaluated at `RAG_EVALUATION_TOP_K=10`
  [evaluation-configuration choice, `RagOnlyAgent`'s own
  `DEFAULT_RAG_TOP_K=5` default unmodified]; on the shared 11-scenario
  set, all three architectures now reach 100% best-line/best-odds/
  EV-classification/freshness accuracy and 0% hallucination rate,
  byte-for-byte [excluding latency] reproducible; a full RAG/TOOL/HYBRID
  `ArchitectureComparison` round-trips through JSON with per-run results
  preserved and no winner declared; no experiment runner, statistical
  significance claim, or dashboard added)

Detailed verification reports for each milestone were produced at the
time of completion (final report blocks in the session transcript for
each milestone); this file is the index, not a re-derivation of those
details. See `docs/PROJECT_STATE.md` for current system state and
`docs/ROADMAP.md` for the full milestone list.
