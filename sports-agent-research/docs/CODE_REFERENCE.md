# Code Reference

A function-level walkthrough of the implemented system, organized by the
same layers as `docs/ARCHITECTURE.md`. This is a reading aid, not a new
source of truth — every description below is derived directly from the
actual module/function docstrings in the code (read fresh from disk), not
from memory. If this document and the code ever disagree, the code wins.

---

## 1. Models — `src/models.py`

The shared data contracts every other layer passes data through. No
layer is allowed to invent its own parallel shape for these concepts.

- **`Game`** — a sporting event (teams, sport); validates the two teams
  are not identical.
- **`Market`** — what's being evaluated: `MarketType`
  (moneyline/spread/total) + outcome + optional line; a spread/total
  market must carry a line, enforced by a model validator.
- **`SportsbookOdds`** — one sportsbook's American-odds quote for one
  outcome; validates the odds value is in a plausible American-odds
  range.
- **`TestScenario`** — one controlled benchmark scenario: a game, a
  market, and the sportsbooks/odds available for it; enforces at least
  one sportsbook and no duplicates.
- **`GroundTruth`** — the deterministically computed correct answer
  (best line/odds/sportsbooks) for a scenario — Milestone 4's
  controlled-reference ground truth, never derived from a market
  consensus.
- **`BestLineResult`** — the mathematically best currently-available
  odds for one outcome, with tie-aware `best_sportsbooks`.
- **`SportsbookValueGroundTruth`** / **`MarketDispersionGroundTruth`** /
  **`QuantGroundTruth`** — Milestone 7B's market-consensus ground truth
  family: per-sportsbook edge vs. a leave-one-out consensus, market
  dispersion statistics, and the per-scenario container tying them
  together, each scenario explicitly flagged `quant_evaluable` (never
  silently skipped when ineligible).
- **`BettingAnalysis`** — the one common structured output every agent
  architecture must produce (`architecture: ArchitectureType` tags
  which); `AnalysisStatus` (`ok` / `insufficient_quant_evidence`) is kept
  consistent with whether an EV verdict was actually reached, via a
  model validator — never fabricated.
- **`SourceReference`** — provenance metadata (what backed a claim, from
  which architecture channel).

## 2. Calculations — `src/calculations/`

Deterministic betting mathematics — the single, never-reimplemented
source of truth for every number an architecture reports. No LLM
involvement anywhere in this layer.

**`odds_math.py`** (single-book math):
- `implied_probability(american_odds)` — American odds → implied
  probability.
- `decimal_odds(american_odds)` — American odds → decimal odds.
- `profit_if_win(american_odds, stake)` — profit on a winning bet
  (excludes stake return).
- `expected_value(american_odds, true_probability, stake)` — EV of a bet
  given an assumed true probability.
- `is_positive_ev(...)` — strict `EV > 0` classification.
- `compare_american_odds(odds_a, odds_b)` / `best_odds(odds_list)` —
  favorability comparison and selection, correct across the
  positive/negative American-odds sign boundary.

**`market.py`** (cross-book market math, Milestone 7A):
- `calculate_overround(odds)` — sum of raw implied probabilities across
  a market (the sportsbook's built-in margin/"vig").
- `remove_vig_from_probabilities(raw_probabilities)` — normalizes raw
  probabilities to sum to 1.0 (the "no-vig"/fair probability).
- `calculate_no_vig_probabilities(odds)` — convenience wrapper chaining
  `implied_probability` → vig removal for a whole market.
- `calculate_market_consensus(probabilities)` — unweighted mean of fair
  probabilities across sportsbooks.
- `calculate_leave_one_out_consensus(sportsbook_probabilities, excluded_sportsbook)`
  — market consensus excluding one sportsbook's own quote from its own
  reference point (so a book is never compared against itself).
- `calculate_probability_edge(market_reference_probability, book_implied_probability)`
  — the edge a specific book's price offers vs. the reference.
- `calculate_market_dispersion(probabilities)` — descriptive-only spread
  statistics across books (explicitly not, by itself, evidence of value).
- `calculate_signed_distance_from_consensus` /
  `calculate_absolute_distance_from_consensus` — a book's deviation from
  consensus, signed and unsigned.

## 3. Providers — `src/providers/`

- **`OddsProvider`** (`base.py`, abstract) — the interface every current
  or future sportsbook data source must implement:
  `get_games()`, `get_game(game_id)`,
  `get_odds(game_id, market_type, selected_outcome)`,
  `get_sportsbook_odds(game_id, sportsbook, market_type, selected_outcome)`.
- **`ControlledOddsProvider`** (`controlled.py`) — the only current
  implementation, backed by the controlled JSON dataset in `data/`.
  Raises typed `GameNotFoundError`/`OddsNotFoundError` rather than
  returning empty/null on a bad lookup. A future live-API provider would
  implement the same interface with zero change to any layer above it.

## 4. Tools — `src/tools/sportsbook_tools.py`

**`SportsbookTools`** — the *only* boundary through which a tool-calling
or hybrid agent may reach current structured sportsbook data. Thin
wrapper over `OddsProvider`: `get_games`, `get_game`, `get_odds`,
`get_sportsbook_odds`, and `find_best_line` (the mathematically best
current price for a game/market/outcome). Five typed errors
(`GameNotFoundError`, `MarketNotFoundError`, `OutcomeNotFoundError`,
`SportsbookNotFoundError`, `NoOddsAvailableError`) so a tool failure is
always classifiable, never a silent empty result.

## 5. RAG — `src/rag/`

- **`documents.py`** — `RagDocument` schema (one retrievable corpus
  entry) and `RagSourceType` (what kind of information it carries);
  validates required fields are non-blank and odds are in range.
- **`build_corpus.py`** — deterministically generates the controlled RAG
  corpus (`generate_documents`/`export_corpus`) from the same benchmark
  data as the ground truth — never includes ground-truth answers
  (enforced separately by `tests/test_rag_corpus.py`).
- **`embeddings.py`** — `EmbeddingModel`, a thin wrapper around
  `sentence-transformers`: `encode_document(s)` / `encode_query` produce
  normalized vectors for `sentence-transformers/all-MiniLM-L6-v2`.
- **`vector_index.py`** — `VectorIndex`, an exact (non-approximate)
  FAISS inner-product index with a `document_id` ↔ row mapping;
  `build_index`/`search`/`save_index`/`load_index`.
- **`build_index.py`** — `build_index`/`export_index`: deterministically
  builds and persists the index from the corpus.
- **`retriever.py`** — `Retriever.retrieve(query, k)`: the one semantic
  retrieval path, returning the top-k documents by embedding similarity;
  preserves rank/score/freshness exactly as retrieved (never filters or
  "corrects" a stale document).
- **`evaluate_retrieval.py`** — deterministic retrieval quality
  evaluation: `recall_at_k`, `hit_at_k`, `evaluate_query`/`evaluate_all`,
  `aggregate_metrics`, `failed_queries` — independent of any agent.

## 6. Agents — `src/agents/`

- **`base.py`** — `AgentRequest` (the common input; carries no
  ground-truth field by construction) and `Agent` (ABC every
  architecture implements via `analyze(request) -> BettingAnalysis`).
- **`llm_client.py`** — provider-agnostic client protocols:
  `LLMClient.generate_structured` (schema-constrained single-turn
  extraction) and `ToolCallingLLMClient.create_turn` (one turn of a
  multi-turn tool loop). Two concrete implementations share both
  protocols on one class each: `AnthropicLLMClient` (Claude API) and
  `OllamaLLMClient` (local Ollama server over `httpx`, with
  `_anthropic_messages_to_ollama`/`_tool_schemas_to_ollama` format
  converters and a bounded `OllamaRequestError`). Provider choice is
  configuration-driven, never architecture-specific.
- **`extraction.py`** — the RAG agent's structured-output contract:
  `ExtractedSportsbookPrice`/`ExtractedMarketEvidence` (what the LLM is
  allowed to claim) and `validate_extraction_provenance(extraction, evidence_bundle)`
  — splits claims into accepted vs. rejected, rejecting anything not
  actually traceable to a retrieved document (a hallucinated sportsbook,
  odds value, or source-document id).
- **`rag_evidence.py`** — `build_rag_evidence_bundle(request, retriever, k)`
  retrieves and assembles evidence; `render_rag_context(bundle)`
  deterministically renders it as the LLM's prompt context;
  `compute_evidence_diagnostics(bundle)` derives evaluation diagnostics
  from evidence alone (never consults ground truth).
- **`rag_agent.py`** — **`RagOnlyAgent`**: RAG evidence → LLM structured
  extraction → provenance validation → shared quant engine →
  `BettingAnalysis`. Raises `RagAnalysisIncomplete` (carries the full
  `RagAgentTrace`) rather than fabricating a result when zero prices
  survive validation. No access to `src.tools`/`src.providers`.
- **`tool_schemas.py`** — the five LLM-callable tool schemas
  (`GetGamesInput`, `GetGameInput`, `GetOddsInput`,
  `GetSportsbookOddsInput`, `FindBestLineInput`) and
  `execute_tool(tools, name, arguments)`, which validates arguments and
  dispatches to the real `SportsbookTools` — never a parallel
  implementation.
- **`tool_agent.py`** — **`ToolCallingAgent`**: a bounded (≤6-iteration)
  multi-turn tool-calling loop against `SportsbookTools`. The final
  `BettingAnalysis` is built exclusively from actual tool-call return
  values folded into an internal `market_state`
  (`_fold_into_market_state`) — never from the LLM's own prose, so a
  hallucinated claim structurally cannot reach the output. Raises
  `ToolAnalysisIncomplete` on hitting the loop bound without resolution.
- **`hybrid_reconciliation.py`** — the hybrid agent's one deterministic,
  LLM-free decision point: `reconcile_outcome(selected_outcome, tool_prices, rag_prices)`
  decides per (sportsbook, outcome) which source is authoritative, with
  every decision tagged a `ConflictResolutionReason`; current tool data
  always wins over a conflicting RAG-derived price, and stale RAG data
  is never promoted to a current price.
- **`hybrid_agent.py`** — **`HybridAgent`**: the only architecture
  permitted to touch both the RAG and tool channels. Runs both, folds
  each channel's results (`_fold_rag_prices`/`_fold_tool_result`),
  reconciles via `reconcile_outcome`, and runs the shared quant pipeline
  only on `HybridMarketRecord.authoritative_odds` — never a raw
  tool/RAG field directly. Raises `HybridAnalysisIncomplete` only if
  *both* channels fail; a single-channel failure degrades gracefully.

## 7. Evaluation — `src/evaluation/`

- **`ground_truth.py`** / **`quant_ground_truth.py`** — pure,
  deterministic generators for the two ground-truth families
  (controlled-reference and market-consensus), each independent of any
  architecture and never exposed to an agent.
- **`dataset.py`** — loads/joins the controlled benchmark
  (`load_current_odds_records`, `load_test_scenarios`, etc.).
- **`metrics.py`** — the unified metric layer (Milestone 11) all three
  evaluators share: `best_line_correct` (set-based tie semantics),
  `best_odds_correct` (exact integer equality),
  `ev_classification_correct`/`ev_absolute_error`/
  `market_reference_absolute_error` (unrounded, `None` — never 0 — when
  not applicable), `evaluate_freshness`, `completeness`,
  `unsupported_claim_rate`, `compute_consistency` (modal-signature-count
  ÷ total-runs across repeated runs), and N/A-aware aggregation
  (`rate`/`mean`/`median`/`population_stdev`). `FailureCategory` is the
  one shared 14-value failure taxonomy. `ArchitectureSummary`/
  `ArchitectureComparison` hold data only — never compute a "winner."
- **`rag_agent_evaluation.py`** / **`tool_agent_evaluation.py`** /
  **`hybrid_agent_evaluation.py`** — one evaluator per architecture, each
  exposing `evaluate_scenario`/`evaluate_scenarios` (drive the real agent
  against a scenario and compare its output to ground truth, entirely
  outside the agent) and `to_common_result` (convert to the shared
  `EvaluationResult` shape). Each also defines its own deterministic fake
  LLM policy (`DeterministicRagPolicyLLMClient`,
  `DeterministicToolPolicyLLMClient`,
  `DeterministicHybridPolicyLLMClient`) used only in MOCK-mode testing —
  never given ground truth, only real retrieved/tool evidence to work
  from.

## 8. Experiments — `src/experiments/`

- **`config.py`** — `ExperimentConfig`, the one place every experiment
  setting is defined (model, temperature, RAG top_k, max tool
  iterations, repetitions, execution mode — applied identically to all
  three architectures); `build_scenario_manifest` builds one
  architecture-neutral canonical query per scenario
  (`_canonical_query`); `execution_order_for_repetition` deterministically
  rotates architecture execution order per repetition;
  `compute_artifact_checksums` hashes every controlled input for
  reproducibility metadata.
- **`agent_factory.py`** — `build_llm_client`/`create_agent`: the one
  centralized factory building any of the three architectures for a
  given config — no architecture ever constructs its own client.
- **`runner.py`** — `run_experiment(config, resume)`: drives every
  (architecture, scenario, repetition) combination through the real
  per-architecture evaluator functions (never a runner-local accuracy
  formula), persists one JSON object per run to `raw_results.jsonl`, and
  refuses to silently overwrite an existing experiment
  (`FileExistsError` unless `resume=True`).
  `probe_real_llm_connectivity` checks the configured provider is
  reachable before spending a real run.
- **`fingerprint.py`** — `compute_final_experiment_fingerprints`/
  `compare_fingerprints`: sha256 fingerprints of every controlled input
  (scenarios, ground truth, RAG corpus/index config, prompts, the config
  file itself), diffed before vs. after a run so any drift is caught,
  not assumed absent.
- **`preflight.py`** — `run_preflight_checks(config)`: the final-run
  gate — a full nested `pytest -q`, ground-truth regeneration-vs-file
  diffs, RAG/tool smoke checks, and a hybrid-reconciliation-policy smoke
  check, all before any real inference call is made.
- **`validation.py`** — `validate_final_dataset(...)`: post-run
  completeness/integrity check — duplicate/missing run keys, scenario
  coverage, ground-truth and architecture-isolation structural checks,
  and the fingerprint match — producing the
  `dataset_validation_report.json` Milestone 14B analysis depends on.

## 9. Analysis — `src/analysis/`

Read-only statistical analysis of a frozen experiment directory — never
modifies the raw dataset it analyzes.

- **`loading.py`** — `load_final_dataset(experiment_dir)`: revalidates
  and loads the frozen dataset; raises `FinalDatasetInvalidError` (never
  a raw parser exception) on a corrupted record.
- **`pairing.py`** — `align_pair`/`align_three_way`: pairs observations
  by `(scenario_id, repetition)` across architectures for paired testing,
  dropping (and counting) anything that can't be aligned rather than
  guessing.
- **`confidence_intervals.py`** — `wilson_score_interval(successes, n, confidence)`:
  closed-form Wilson interval for a binomial proportion.
- **`pairwise_tests.py`** — `mcnemar_test` (exact binomial McNemar for
  paired binary outcomes), `wilcoxon_signed_rank` (paired
  continuous/ordinal outcomes), `holm_correction` (family-wise
  multiple-comparison correction) — all via `scipy.stats`.
- **`omnibus_tests.py`** — `friedman_test(triples)`: three-way omnibus
  comparison across matched RAG/TOOL/HYBRID triples, with an explicit
  degeneracy check distinguishing "every block is a true three-way tie"
  from "every block agrees on a non-trivial ranking" (a real bug in this
  distinction was found and fixed during Milestone 14B, per
  `milestones/completed.md`).
- **`comparisons.py`** — `compute_binary_comparisons`/
  `compute_continuous_comparisons`/`compute_omnibus_comparisons`: the
  frozen metric-family framework (4 binary metrics via McNemar, 4
  continuous via Wilcoxon, 3 omnibus via Friedman), each pair
  Holm-corrected within its own family.
- **`descriptive.py`** — `architecture_descriptive_stats`: per-architecture
  mean/median/stdev/min/max for every metric.
- **`freshness_analysis.py`** / **`hallucination_analysis.py`** /
  **`failure_analysis.py`** / **`hybrid_conflict_analysis.py`** /
  **`latency_analysis.py`** / **`scenario_analysis.py`** /
  **`subgroups.py`** — one focused, exploratory-labeled analysis module
  each, all consuming metric formulas from `src.evaluation.metrics`
  rather than redefining them.
- **`tables.py`** — `build_table_1..5_*`/`build_omnibus_table`: the
  required output tables, each a pure function of the analysis bundle.
- **`figures.py`** — `generate_all_figures(bundle, output_dir)`: the 6
  research figures (bar charts with Wilson-interval error bars; N/A
  rendered as an honest label, never a misleading zero-height bar).
- **`findings.py`** — `generate_findings_markdown(bundle)`: composes
  `findings.md`'s narrative sections and direct research-question answer
  from the computed bundle only.
- **`bundle.py`** — `build_analysis_bundle(dataset)`: assembles every
  computed artifact above into one `AnalysisBundle`.
- **`final_analysis.py`** — `run_final_analysis(experiment_dir)`: the one
  reproducible entry point (`python -m src.analysis.final_analysis`)
  that loads the dataset, builds the bundle, and writes every
  machine-readable file, figure, and `findings.md`.

## 10. Dashboard — `dashboard/`

A thin Streamlit UI layer with no betting/quant logic of its own
(verified by an AST-based structural test).

- **`data_loader.py`** — `run_demo_analysis(...)` constructs and runs an
  agent live via the same `create_agent` factory the experiment runner
  uses; `load_experiment`/`list_experiment_ids` load a persisted
  experiment directory read-only; `reconstruct_architecture_specific_result`,
  `hybrid_conflict_summary`, `failure_counts_by_architecture` support the
  Research view's drill-downs.
- **`demo_view.py`** — `render_demo_mode()`: the single-scenario,
  single-architecture live-execution UI, plus per-architecture trace
  rendering (`_render_rag_trace`/`_render_tool_trace`/
  `_render_hybrid_trace`) and an explicit freshness/conflict display
  (`_render_freshness_conflicts`).
- **`research_view.py`** — `render_research_mode()`: loads a persisted
  experiment read-only and renders metadata, per-architecture summaries,
  comparison charts, scenario drill-down, failure analysis, hybrid
  conflict analysis, and raw-data export.
- **`charts.py`** — `render_comparison_charts(comparison)`: one bar chart
  per Section-14 metric, built only from
  `src.evaluation.metrics`/`hybrid_agent_evaluation.summarize_results`
  output.
- **`formatting.py`** — display-only helpers (`format_percentage`,
  `format_error`, `format_freshness`, etc.) that never render a missing
  value (`None`) as a misleading `0`.
- **`app.py`** — `main()`: the Streamlit entry point, switching between
  Demo and Research Comparison modes.

---

See `docs/ARCHITECTURE.md` for how these layers connect,
`docs/QUANT_STRATEGY.md` for the calculations layer's full mathematical
detail, and `docs/EXPERIMENT_RULES.md` for the access-boundary rules
enforced between layers 6 (Agents) and 3–5 (Providers/Tools/RAG).
