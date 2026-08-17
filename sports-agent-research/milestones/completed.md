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
- Milestone 12 — Controlled experiment runner — COMPLETE
  (896 passed / 0 failed; new `src/experiments/` package —
  `config.py` [`ExperimentConfig` centralizing model/effort/temperature/
  RAG top_k/max tool iterations/repetitions/execution mode as one frozen
  set of controls applied identically to all three architectures;
  `ExperimentScenario`/`build_scenario_manifest()` — one plain
  architecture-neutral canonical query per scenario, e.g. "Compare the
  available moneyline prices for Los Angeles Lakers and identify the
  best current value.", structurally unable to take a per-architecture
  parameter, no ground-truth fields; deterministic left-rotation
  execution-order policy across repetitions [Rep1 RAG→TOOL→HYBRID, Rep2
  TOOL→HYBRID→RAG, Rep3 HYBRID→RAG→TOOL]; SHA-256 checksums of every
  controlled artifact — current odds, RAG corpus, ground truth, quant
  ground truth, RAG index config — never an API key]; `agent_factory.py`
  — one `create_agent(architecture, config)` reused for all three
  architectures, MOCK mode reusing the exact Milestone 9B/10B/11
  deterministic fake-LLM policies verbatim, REAL mode using one
  `AnthropicLLMClient` configuration shared across architectures;
  `runner.py` — `run_experiment()` drives every (architecture, scenario,
  repetition) combination through the *same* Milestone 11 evaluator
  functions [`_EVALUATE_SCENARIO`/`_TO_COMMON_RESULT` dicts hold direct
  references to the real `rag_agent_evaluation.py`/
  `tool_agent_evaluation.py`/`hybrid_agent_evaluation.py` functions,
  verified by identity in tests, never a runner-local accuracy formula],
  isolates ground truth to the evaluator call only [`AgentRequest`
  construction verified via AST scan to never reference ground truth],
  isolates per-run failures [one architecture's agent-construction
  failure is recorded as an `UNKNOWN_FAILURE` run and does not abort the
  batch], and persists raw, never-pre-aggregated results one JSON object
  per line to `results/experiments/<experiment_id>/raw_results.jsonl`
  alongside `config.json`/`manifest.json`/`summary.json`; explicit
  refuse-by-default duplicate protection [`FileExistsError` unless
  `resume=True`, which then only appends unrecorded
  `(architecture, scenario_id, repetition)` keys]; post-run
  `ArchitectureSummary`/`ArchitectureComparison` aggregation and
  cross-repetition `consistency` reused directly from Milestone 11 —
  the first milestone to actually exercise that calculation with real
  repeated-run data; a CLI [`python -m src.experiments.runner --mode
  {mock,real} [--repetitions] [--output-dir] [--scenario-ids]
  [--architectures] [--experiment-id] [--dry-run] [--resume]`] including
  a config-preview `--dry-run` mode and a REAL-mode connectivity probe
  that prints "REAL EXPERIMENT: NOT RUN" and exits cleanly rather than
  silently falling back to mock when no credentials are available; new
  test files `tests/test_experiment_config.py` [24 tests],
  `tests/test_experiment_runner.py` [12 tests],
  `tests/test_experiment_persistence.py` [11 tests] — 47 new tests total
  covering config validation, manifest/rotation determinism, checksum
  stability, unified-evaluator-reuse-by-identity, ground-truth isolation
  by AST inspection, common-query-reaches-all-architectures, failure
  isolation, consistency aggregation, expected-vs-recorded run-count
  validation, file persistence, duplicate/resume protection, and
  mock-mode reproducibility across two independent output directories
  [verified both by the automated test and manually — record-for-record
  equality on every research-relevant field, excluding timestamps,
  experiment/run IDs, and all latency measurements]; an
  INFRASTRUCTURE-VALIDATION-ONLY 18-run mock demonstration [3
  architectures x 3 scenarios x 2 repetitions] ran 18/18 successful with
  full per-architecture summaries printed and explicitly labeled "NOT
  FINAL RESEARCH RESULTS"; the fairer, architecture-neutral canonical
  query [deliberately less retrieval-optimized than the sportsbook-
  name-heavy queries the Milestone 9B-11 evaluators use for their own
  metric testing] combined with `RagOnlyAgent`'s true default
  `top_k=5` causes RAG's mock-mode `ev_classification_accuracy` to come
  back `None` in these demo runs — an honest, expected research-
  infrastructure finding recorded as-is, not tuned away; a real bug was
  found and fixed in this milestone's own new reproducibility test
  [`tests/test_experiment_persistence.py`'s `_LATENCY_FIELDS` substring
  list didn't cover every architecture-specific latency field name —
  e.g. the tool evaluator's `llm_decision_latency_seconds`/
  `tool_execution_latency_seconds` — causing the test to compare
  un-stripped timing values and fail spuriously; fixed by matching any
  key ending in `_latency_seconds` instead of a fixed name list; not a
  runner or evaluator defect]; no inferential statistics, no dashboard,
  no ground truth exposed to any agent, no changes to any of the three
  agents' or the unified evaluator's behavior; REAL-mode smoke test not
  run — no `ANTHROPIC_API_KEY` set this session, CLI printed "REAL
  EXPERIMENT: NOT RUN" and exited cleanly, non-blocking)
- Milestone 13 — Dashboard and research visualization — COMPLETE
  (945 passed / 0 failed; new `dashboard/` Streamlit package — thin UI
  layer, no sportsbook/quant business logic of its own [verified by
  `tests/test_dashboard_structure.py`'s AST scan: no quant/odds formula
  definitions, agents constructed only via `src.experiments.agent_
  factory.create_agent`, never a concrete `Agent` subclass directly];
  `data_loader.py` [scenario manifest reuse via `src.experiments.config.
  build_scenario_manifest`, ground-truth-free `AgentRequest`
  construction, MOCK/REAL demo-mode agent execution with `st.cache_
  resource`-cached Retriever/SportsbookTools construction only —
  `agent.analyze()` itself is never cached, so every "Run Analysis"
  click is independent —, persisted-experiment loading/filtering/
  grouping, and hybrid-conflict aggregation via the existing
  `hybrid_agent_evaluation.summarize_results` [Milestone 10B/11],
  never a dashboard-local formula]; `formatting.py` [percentage/N/A/
  odds/probability formatting, small-nonzero-error display that never
  collapses to a misleading "0.000000", never mutates the underlying
  value]; `charts.py` [pure `build_metric_dataframe()` plus a thin
  `st.bar_chart` wrapper for the 7 Section-14 comparisons — no 3D, no
  decorative chart types]; `demo_view.py` [Mode A: scenario +
  architecture + execution-mode selection, explicit "Run Analysis"
  action — never auto-executes on widget change —, `BettingAnalysis`
  result summary, per-sportsbook comparison table sourced from each
  architecture's own trace/tool re-query, and a full architecture trace
  expander including a RAG-snapshot-vs-current-tool freshness-conflict
  display for hybrid, matching the milestone's own example format];
  `research_view.py` [Mode B: loads a persisted Milestone 12 experiment
  directory read-only, an unmissable "MOCK — INFRASTRUCTURE VALIDATION
  ONLY" banner whenever `execution_mode=mock` [never presented as a
  research finding], experiment metadata panel, per-architecture summary
  table, all 7 comparison charts [no automatic "winner"], scenario
  drill-down [ground truth labeled "EXPECTED / GROUND TRUTH" and shown
  only in this research context, never fed back into an agent],
  repetition/consistency inspection, failure analysis grouped by the
  Milestone 11 `FailureCategory` taxonomy, hybrid conflict analysis, and
  filterable raw-result access plus optional CSV/JSON export that never
  overwrites the original experiment files]; `app.py` [thin sidebar
  Demo/Research-Comparison mode switch]; new test files
  `tests/test_dashboard_formatting.py` [14 tests],
  `tests/test_dashboard_charts.py` [5 tests],
  `tests/test_dashboard_data_loader.py` [24 tests, including a 3-
  architecture x 2-scenario x 2-repetition mock experiment fixture and a
  MOCK-only, no-paid-API-call demo-execution check for every
  architecture], `tests/test_dashboard_structure.py` [6 tests] — 49 new
  tests total, zero regressions against the 896-passed Milestone 12
  baseline; manually verified end to end via `streamlit.testing.v1.
  AppTest` [no browser automation needed] for all three architectures in
  Demo mode [including a live freshness-conflict case, scenario S009,
  reproducing the milestone's own RAG-snapshot/current-tool/
  authoritative/resolution example], the missing-experiment-directory
  and REAL-mode-without-credentials graceful-degradation paths [neither
  crashes], and the Research view against a real 24-run mock experiment
  [3 architectures x 4 scenarios x 2 repetitions]; also confirmed with a
  real `streamlit run dashboard/app.py` server launch [HTTP 200]; added
  `streamlit`/`pandas` to `requirements.txt` and a `results/experiments/`
  `.gitignore` continuation; README gained a short Dashboard section; no
  live sportsbook API introduced [`ControlledOddsProvider` throughout],
  no final experiment run, no inferential statistics, no research
  conclusion generated, no changes to any agent, the unified evaluator,
  or the experiment runner)

- Checkpoint Pre-14A — Local real-LLM backend (Ollama) — COMPLETE
  (1027 passed / 0 failed; new `OllamaLLMClient` [`src/agents/llm_client.py`]
  implementing the same `LLMClient`/`ToolCallingLLMClient` protocols as
  `AnthropicLLMClient` on one class [schema-constrained structured JSON
  output + native multi-turn tool calling via `httpx` against a locally
  hosted Ollama server — no new orchestration framework]; configuration-
  driven provider selection [`ExperimentConfig.llm_provider`,
  `LLMProviderName.ANTHROPIC` default | `.OLLAMA`, read by
  `agent_factory.build_llm_client` — never architecture-specific];
  central `LLM_PROVIDER`/`LLM_MODEL`/`DEFAULT_OLLAMA_MODEL`
  [`llama3.1:8b`]/`DEFAULT_OLLAMA_TEMPERATURE` [0.0] constants; verified
  live via `experiments/run_ollama_three_architecture_smoke_test.py` —
  RAG, TOOL [a real `get_odds` tool call confirmed to actually occur],
  and HYBRID all produced valid `BettingAnalysis` against real local
  inference; no agent, quant engine, ground truth, RAG corpus, or
  evaluation metric touched; 26 new tests, all passing with the Ollama
  service stopped, confirming standard `pytest -q` never requires a
  running local server; a follow-up readiness-verification pass
  confirmed nothing needed to be redone and fixed one real defect
  [`probe_real_llm_connectivity` in `src/experiments/runner.py` was
  hard-coded to `AnthropicLLMClient` regardless of configured provider,
  which would have made an Ollama-configured real run always report
  "REAL EXPERIMENT: NOT RUN" — fixed to build its probe client via
  `build_llm_client()`, plus one artifact-fingerprint gap closed
  [`src/experiments/fingerprint.py` now also hashes
  `src/experiments/config.py` itself, so a change to the canonical query
  template's wording is detectable, not just its input data]; both
  fixes covered by new regression tests; `experiments/final_experiment.json`
  updated to record `llm_provider=ollama`/`model_name=llama3.1:8b`/
  `temperature=0.0` before any run under `experiment_id=final_v1` had
  started; Milestone 14A itself not begun)
- Milestone 14A — Freeze and run the final controlled experiment — COMPLETE
  (1027 passed / 0 failed; the final, real [non-mock] controlled
  experiment: 3 architectures [RAG/TOOL/HYBRID] x 11 scenarios
  [`DEFAULT_SCENARIO_IDS`] x 10 repetitions = 330 frozen, real-Ollama
  [`llama3.1:8b`, temperature=0.0] observations, executed via the
  unmodified Milestone 12 `run_experiment()` through a new orchestrator
  [`experiments/run_final_experiment.py`: preflight checks including a
  full nested `pytest -q`, ground-truth/quant-ground-truth
  regeneration-vs-persisted-file diffs, a live RAG-index/retrieval smoke
  query, `ControlledOddsProvider`/`SportsbookTools` smoke checks, and a
  direct synthetic-conflict call confirming the hybrid
  `CURRENT_TOOL_DATA_PRECEDENCE` policy is intact — then a provider-aware
  real-inference connectivity probe, pre/post artifact fingerprinting
  [9 hashes: benchmark scenarios, both ground-truth files, RAG corpus,
  RAG index config, both system prompts, the canonical-query-template
  module, and the frozen config file itself — all 9 MATCHED after
  execution], environment-metadata capture, and a full dataset
  validation report [`src/experiments/validation.py`]); balanced
  deterministic architecture rotation confirmed actually applied
  [repetition 1 RAG→TOOL→HYBRID, repetition 2 TOOL→HYBRID→RAG,
  repetition 3 HYBRID→RAG→TOOL, repetition 4 cycling back — verified
  from the persisted `execution_order_position` field, not merely
  assumed]; result: 330/330 expected runs recorded [110 RAG / 110 TOOL /
  110 HYBRID, exactly even], zero duplicate keys, zero missing keys,
  full 10/10/10 scenario-coverage matrix across all 11 scenarios, zero
  raw-result schema-validation errors, ground-truth isolation and
  architecture isolation both re-audited PASS post-execution; 290
  successful observations [`quant_insufficient_data`, i.e. a validated
  best line with insufficient two-sided data for an EV verdict — RAG 90,
  TOOL 100, HYBRID 100] and 40 failed observations preserved
  as-recorded, never dropped or retried [RAG 20x
  `insufficient_retrieved_evidence`, a genuine architecture-level
  evidence-pipeline gap on the spread/total scenarios S012/S013; TOOL
  10x and HYBRID 10x, both traced to the identical root cause — a
  client-side 180s `OllamaRequestError` read-timeout on S012's
  tool-calling turns, an infrastructure/runtime characteristic of this
  local model+scenario combination rather than a reasoning failure,
  distinguished from architecture-level failures per the raw `errors`
  field rather than reclassified after the fact]; median observation
  latency ~16.2s [mean ~24.8s, range 3.7s-183.9s] confirms genuine local
  inference throughout, never the deterministic mock policies; zero
  hybrid source conflicts were recorded across all 110 hybrid
  observations in this dataset [the reconciliation POLICY itself was
  independently verified intact via a synthetic direct call and the
  unmodified, fully-passing `tests/test_hybrid_reconciliation.py` — this
  is a real, honestly-reported characteristic of how this local model's
  RAG-side extraction interacts with the existing, unmodified provenance
  validator, not a defect]; no agent, prompt, quant formula, ground
  truth, RAG corpus, or evaluation metric modified during or after
  execution [confirmed via `git diff` and the 9-way pre/post hash match];
  no arbitrary retries; no inferential statistics performed; no
  "winning" architecture declared; dataset explicitly validated eligible
  for Milestone 14B's statistical analysis)
- Milestone 14B — Statistical analysis and research findings — COMPLETE
  (1097 passed / 0 failed; new `src/analysis/` package — read-only
  statistical analysis of the frozen `results/experiments/final_v1/`
  dataset, never modifying raw_results.jsonl/config.json/manifest.json
  [verified by file-hash/mtime checks and a dedicated test]: `loading.py`
  [revalidates via the unmodified Milestone 14A `validate_final_dataset`,
  reuses `validation.load_and_validate_raw_results` rather than a second
  raw-line parser — a real gap there, a corrupted record crashing with a
  raw pydantic exception instead of `FinalDatasetInvalidError`, was found
  by a new synthetic test and fixed]; `pairing.py` [paired alignment by
  (scenario_id, repetition) across architectures, and by scenario_id
  alone for consistency, per Section 5 — N/A values dropped, never
  coerced to zero]; `confidence_intervals.py` [Wilson score interval,
  verified against textbook reference values]; `pairwise_tests.py`
  [exact-binomial McNemar via `scipy.stats.binomtest`, Wilcoxon signed-
  rank via `scipy.stats.wilcoxon`, Holm-Bonferroni correction verified
  against a textbook example]; `omnibus_tests.py` [Friedman via
  `scipy.stats.friedmanchisquare` — a real bug was found and fixed here
  by a new synthetic test: the original degeneracy check conflated
  "every block holds an identical triple" [a maximally significant,
  perfectly-consistent-ranking case] with "every block is a three-way
  tie" [the true zero-variance case scipy can't compute], misreporting
  the former as "no variation, p=1.0"; fixed to check within-block ties
  only, and to detect scipy's silent NaN return for genuine degeneracy];
  `comparisons.py` [the frozen metric-family framework — 4 binary
  metrics via McNemar, 4 continuous metrics via Wilcoxon, 3 omnibus
  metrics via Friedman, each of the 3 architecture pairs corrected via
  Holm within its own metric family, never pooled globally]; plus
  `descriptive.py`/`freshness_analysis.py`/`hallucination_analysis.py`/
  `failure_analysis.py`/`hybrid_conflict_analysis.py`/
  `latency_analysis.py`/`scenario_analysis.py`/`subgroups.py`
  [exploratory, clearly labeled] — every metric FORMULA reused directly
  from `src.evaluation.metrics`/`hybrid_agent_evaluation.summarize_results`,
  never redefined; `tables.py` [5 required tables], `figures.py` [6 bar
  charts with Wilson-interval error bars, no 3D, N/A rendered as an
  honest "N/A" label rather than a misleading zero-height bar], and
  `findings.md` generation, all driven by one `final_analysis.py`
  orchestrator (`python -m src.analysis.final_analysis`) writing to
  `results/experiments/final_v1/analysis/` [8 machine-readable files +
  `figures/` + `findings.md`]. Result: every EV-classification/EV-error/
  market-reference-error comparison correctly reports N/A [zero
  observations in the frozen dataset ever reached a full quant verdict]
  rather than a fabricated p-value; RAG's best-line/best-odds accuracy
  [88.9%] is significantly lower than TOOL/HYBRID's [100%, Holm-adjusted
  p=0.0059] while TOOL and HYBRID are statistically indistinguishable
  from each other [p=1.0, zero discordant pairs]; freshness [100% for
  all three] and consistency [1.0 for all three, Friedman correctly
  degenerate] show no distinguishable difference; RAG's completeness
  [72.7%] is significantly lower than TOOL/HYBRID's [90.9%]; latency
  differs significantly across all three pairs [TOOL fastest, HYBRID
  slowest]; zero hallucinations; hybrid conflict-resolution accuracy is
  correctly N/A [zero live conflicts in this dataset, mechanism verified
  intact separately]; findings.md answers the research question directly
  without forcing a universal winner, distinguishes statistical from
  practical significance throughout, documents limitations including
  local-model-specific capability findings, and includes the independent
  predictive-ML-model extension as documented future work, not
  implemented; no agent, prompt, quant formula, ground truth, or
  evaluation metric modified; no selective rerunning; 70 new tests using
  synthetic data only, never hard-coded to the actual final result)
- Milestone 15 — Research conclusions, manuscript, and presentation
  support — COMPLETE (1097 passed / 0 failed, unchanged from Milestone
  14B — a documentation-only milestone; no agent, prompt, model
  configuration, scenario, raw result, ground truth, statistical test, or
  metric definition was modified [confirmed via `git diff` and file
  mtime checks: `raw_results.jsonl`/`config.json`/`manifest.json` and the
  Milestone 14B `analysis/` outputs retain their original timestamps]).
  Seven new `docs/` deliverables, every numeric claim traced directly to
  a fresh read of `results/experiments/final_v1/analysis/` [not
  reconstructed from memory]: `FINAL_RESEARCH_SUMMARY.md` [research
  question, experimental design, the 6-step quant pipeline, primary/
  secondary results reproduced verbatim from `findings.md`, a direct,
  evidence-calibrated answer to the research question, limitations, and
  future work including the ML-predictive-model extension diagram —
  explicitly not implemented]; `MANUSCRIPT_OUTLINE.md` [structural
  outline, 8 sections, Methods 3.1-3.7 limited to methods actually used];
  `MANUSCRIPT_DRAFT.md` [~450-line first-draft manuscript prose —
  Abstract/Introduction/Background/Methods/Results/Discussion/
  Limitations/Future Work/Conclusion — 8 `[CITATION NEEDED]` markers for
  literature claims, zero fabricated citations, every statistic
  cross-verified against `pairwise_comparisons.json`/
  `descriptive_statistics.json`]; `PRESENTATION_STORYBOARD.md` [12-slide
  content plan, not slides]; `RESULTS_ASSET_INDEX.md` [full inventory of
  Figures 1-6/Tables 1-5/omnibus table with path, contents, manuscript
  section, presentation slide, source artifact]; `REPRODUCIBILITY.md`
  [10-step workflow, MOCK vs. REAL execution mode explicitly
  distinguished, no secrets embedded]; `CITATION_NEEDS.md` [10-row audit
  of every `[CITATION NEEDED]` marker — no sources searched, browsed, or
  invented]. `docs/ARCHITECTURE.md` rewritten [previously stale since
  Milestone 6C, described the RAG pipeline/agents/quant engine/dashboard
  as "future" despite being fully implemented; now reflects the completed
  system, including the Milestone 12 `src/experiments/` and Milestone 14B
  `src/analysis/` layers it previously omitted entirely] — no working
  implementation code changed. Terminology audit [Section 11]: grepped
  all new documents, confirmed market-implied/no-vig consensus
  probability is never conflated with true win probability, zero
  instances of forbidden absolute language ["proved," "guarantee(s/d),"
  "always better"], and every "positive EV"/profitability reference
  correctly scoped to this project's own methodology, never claiming
  real-world profitability. Conclusion audit [Section 12]: grepped for
  "winner"/"dominates"/"superior" — confirmed no architecture is declared
  a universal winner; the stated conclusion is evidence-calibrated
  [TOOL/HYBRID beat RAG on accuracy/completeness, TOOL beats HYBRID on
  latency, consistency/freshness indistinguishable across all three] with
  tradeoffs stated explicitly rather than collapsed into one verdict. All
  numeric claims in the three primary new documents were individually
  cross-checked against fresh reads of the Milestone 14B JSON outputs;
  none required removal or a "needs verification" mark. Full pytest suite
  re-run at the end of this milestone: 1097 passed / 0 failed, confirming
  the documentation-only work did not alter application behavior.

Detailed verification reports for each milestone were produced at the
time of completion (final report blocks in the session transcript for
each milestone); this file is the index, not a re-derivation of those
details. See `docs/PROJECT_STATE.md` for current system state and
`docs/ROADMAP.md` for the full milestone list.
