# Architecture Decisions

## ADR-001 — Controlled Benchmark

**Decision:** Use controlled sportsbook data for the primary experiment.

**Reason:** Reproducibility and deterministic ground truth.

**Future:** Live API data may be added as a secondary demonstration or
validation.

---

## ADR-002 — OddsProvider Abstraction

**Decision:** Sportsbook tools depend on `OddsProvider`.

**Reason:** Allows structured data sources to change without changing
agents.

---

## ADR-003 — Deterministic Math

**Decision:** Betting and quant calculations are implemented
deterministically in Python.

**Reason:** LLM arithmetic variability must not contaminate architecture
comparison.

---

## ADR-004 — Shared Quant Engine

**Decision:** All architectures use identical quantitative logic.

**Reason:** Agent architecture remains the primary independent variable.

---

## ADR-005 — Two-Sided Markets

**Decision:** Quant-evaluable markets must represent mutually exclusive
outcomes.

**Reason:** Required for no-vig calculations.

---

## ADR-006 — Leave-One-Book-Out Consensus

**Decision:** The target sportsbook is excluded from the market
consensus used to evaluate its own line.

**Reason:** Reduces circularity.

---

## ADR-007 — Controlled Freshness Differences

**Decision:** RAG may contain stale sportsbook snapshots while
structured provider data represents current values.

**Reason:** Allows freshness performance to be tested intentionally.

---

## ADR-008 — RAG Retrieval Evaluated Before RAG Agent

**Decision:** Embedding/vector retrieval quality will be tested
independently before LLM reasoning is introduced.

**Reason:** Allows retrieval failures to be separated from agent
reasoning failures.
