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

Detailed verification reports for each milestone were produced at the
time of completion (final report blocks in the session transcript for
each milestone); this file is the index, not a re-derivation of those
details. See `docs/PROJECT_STATE.md` for current system state and
`docs/ROADMAP.md` for the full milestone list.
