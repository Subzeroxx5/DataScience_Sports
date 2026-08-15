# Research Validity Rules

- Agent architecture is the main independent variable — everything else
  should be held constant where practical (see
  `docs/EXPERIMENT_RULES.md`).
- Shared math must remain identical across architectures — never let one
  architecture use a different formula or implementation than another.
- Ground truth cannot be provided to agents.
- RAG cannot contain benchmark answers (see the leakage restrictions in
  `docs/EXPERIMENT_RULES.md`, enforced by `tests/test_rag_corpus.py`).
- Controlled/synthetic data must be described accurately — never present
  it as real sportsbook data.
- Market consensus is not objective true probability — document it as a
  market-derived reference value only.
- Freshness differences must be intentional and documented, not
  accidental data drift.
- Do not manually modify experimental results.
- Do not claim statistical significance without appropriate testing.
- Preserve architecture access boundaries (RAG-only / tool-only / hybrid)
  as defined in `docs/EXPERIMENT_RULES.md`.
