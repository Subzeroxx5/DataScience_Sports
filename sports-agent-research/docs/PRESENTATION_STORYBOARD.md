# Presentation Storyboard

Content plan for a ~10-12 slide oral presentation. This is a storyboard
only — no slide deck is created in this milestone. Every number listed is
traced to `results/experiments/final_v1/analysis/`; do not add numbers not
present there.

## Slide 1 — Title

- Project title: "Retrieval, Tools, or Both? A Controlled Comparison of
  Agent Architectures for Sportsbook Price Identification"
- Research question (short form): does agent architecture affect accuracy,
  consistency, and freshness when identifying +EV sportsbook opportunities?
- Context: controlled research prototype, not a production betting tool.

## Slide 2 — Why This Matters

- Problem: the same market is priced differently across sportsbooks.
- Information goes stale: a price that was current a moment ago may not be
  now.
- How an agent *gets* its information (retrieval vs. tools vs. both) is a
  design choice with measurable consequences — not just an implementation
  detail.

## Slide 3 — Research Question

Display prominently, verbatim:

> "How does agent architecture—tool calling, retrieval-augmented generation
> (RAG), or a hybrid approach—affect the accuracy, consistency, and
> freshness of an AI agent identifying positive expected value
> opportunities across multiple sportsbooks?
> Can an LLM-assisted betting tracker reliably identify best available sportsbook lines?
> Can a local Ollama model, specifically llama3.1:8b, handle the orchestration reliably? 

## Slide 4 — Experimental Design

- Independent variable: RAG-only vs. TOOL-only vs. HYBRID.
- Controlled (identical across all three): same LLM (`llama3.1:8b`, local,
  temperature 0.0), same 11-scenario controlled benchmark, same
  architecture-neutral query per scenario, same quantitative engine, same
  evaluation framework, same repetition count (10), deterministic balanced
  execution-order rotation.
- Ground truth: deterministic, never exposed to any agent.

## Slide 5 — Architecture Diagram

```text
RAG
Corpus -> Retriever -> LLM (extraction only)

TOOL
LLM -> Sportsbook Tools -> OddsProvider

HYBRID
RAG + Tools -> Deterministic Source Reconciliation
              (current tool data wins on conflict)

All three:
   -> Shared Quant Engine (implied prob / no-vig / consensus / EV)
   -> BettingAnalysis (common structured output)
   -> Unified Evaluation
```

## Slide 6 — Quantitative Method

```text
American odds
   -> implied probability
   -> no-vig (fair) probability
   -> leave-one-sportsbook-out market consensus
   -> expected value (vs. that consensus)
```

- One sentence, stated plainly: the market-consensus reference probability
  is **not** the true win probability of the game — it's a market-derived
  reference point.
- Avoid formulas on the slide itself; keep to the pipeline above.

## Slide 7 — The Experiment

- Model: `llama3.1:8b` (local, via Ollama), temperature 0.0 — identical
  across all three architectures.
- 11 scenarios x 10 repetitions x 3 architectures = **330 real
  observations** (REAL local inference, not the project's mock/simulation
  mode).
- Dataset fully validated: 330/330 recorded, zero duplicate/missing keys,
  full scenario coverage, all controlled-artifact fingerprints matched
  before vs. after execution.

## Slide 8 — Accuracy Results

- Use: `results/experiments/final_v1/analysis/figures/1_best_line_accuracy.png`
- Headline numbers: RAG 88.9%, TOOL 100.0%, HYBRID 100.0% (best-line
  accuracy).
- RAG vs. TOOL and RAG vs. HYBRID: statistically significant difference
  (Holm-adjusted p=0.0059). TOOL vs. HYBRID: not distinguishable (p=1.0 —
  they agreed on every paired case).
- Callout: EV classification accuracy is N/A for all three — no observation
  in the whole dataset reached a full EV verdict (discuss on Slide 10).

## Slide 9 — Consistency & Freshness

- Use: `figures/4_consistency.png` and `figures/3_freshness_accuracy.png`.
- Consistency: 1.000 for all three architectures — no distinguishable
  difference (nothing varied to compare).
- Freshness: 100% for all three on the one freshness-designated scenario —
  no architecture made a freshness error in this dataset.
- Caveat to say out loud: freshness/consistency showing "no difference"
  here means the dataset didn't surface a difference — not proof none
  exists in general (small scenario count).

## Slide 10 — Why the Architectures Behaved Differently

- RAG: 20 failures (`insufficient_retrieved_evidence`) concentrated on the
  spread/total scenarios (S012, S013) — retrieval at a fixed top_k didn't
  always surface both sides of the market.
- TOOL / HYBRID: 10 + 10 failures, both on S012, both traced to a
  client-side 180-second inference timeout — an infrastructure
  characteristic of this local model on this scenario, not a reasoning
  failure.
- Hybrid conflict resolution: 310 source agreements, **0** conflicts across
  110 hybrid observations — the current-tool-data-precedence rule is
  verified correct independently, but this dataset never produced a live
  disagreement to show it acting.
- Zero hallucinations (unsupported claims) for any architecture, any
  scenario.

## Slide 11 — Limitations

- Controlled/synthetic benchmark — not live market data.
- 11 scenarios, 10 repetitions — limited statistical power.
- Specific to `llama3.1:8b` and this embedding model; may not generalize.
- Market consensus != true win probability.
- No predictive ML model was built or compared.
- Does not establish real-world betting profitability.

## Slide 12 — Conclusion & Future Work

- No single architecture wins across accuracy, consistency, and freshness.
- TOOL and HYBRID beat RAG on accuracy/completeness; TOOL beat HYBRID on
  latency; consistency and freshness were indistinguishable across all
  three.
- Future work:
  - live sportsbook providers behind the existing abstraction
  - broader scenario coverage
  - alternative LLMs (same experiment, different model)
  - an independently trained/calibrated predictive ML model, compared
    against the market-implied no-vig consensus (not implemented here)
