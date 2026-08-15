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
```

## Current Documentation Checkpoint

```text
6B.5 — Project Documentation and Agent Instructions (COMPLETE)
```

## Next Functional Milestone

```text
Milestone 9 — Tool-calling agent
```

## Current Test Baseline

```text
Current full test suite:
592 passed / 0 failed
```

(Recorded from `pytest -q` at the end of Milestone 8B.)

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

The project does **not** yet have:

- a tool-calling LLM agent
- a hybrid agent
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
