# Current Milestone

## Milestone 9 — Tool-Calling Agent

## Objective

Implement the second concrete architecture: a tool-calling `Agent`
subclass (`src/agents/base.py`) that takes an `AgentRequest` and uses an
LLM with function/tool calling against the structured sportsbook tool
layer (`src/tools/sportsbook_tools.py`, `OddsProvider`) — never the RAG
corpus — to gather current sportsbook prices, then applies the shared
quant engine (`src/calculations/market.py` + `odds_math.py`) to whatever
it successfully gathered, and returns a `BettingAnalysis`
(`architecture=tool_calling` or equivalent).

See `docs/EXPERIMENT_RULES.md` ("Tool-Only Boundary") for the exact
access boundary this agent must respect (structured tools only — never
`src.rag`), `docs/QUANT_STRATEGY.md` for the calculations to reuse (never
reimplement), and `docs/ARCHITECTURE.md` ("Agents") for how this fits the
overall system.

This file is a placeholder recording the next milestone's identity only;
detailed scope (tool schema design, LLM provider/model choice, call-loop
strategy and failure handling, how `market_reference_probability`/
`probability_edge` get populated on `BettingAnalysis`, determinism
controls, test requirements) should be defined when Milestone 9 begins.

## Explicitly Out of Scope (until stated otherwise)

- hybrid agent
- experiment runner
- dashboard
- changes to the RAG-only agent (`src/agents/rag_agent.py`,
  `src/agents/extraction.py`, `src/agents/llm_client.py`, Milestone 8B —
  complete)
- changes to the provider/tool subsystem's existing behavior, the quant
  engine's formulas, or the RAG corpus/vector index/retriever
- exposing ground truth (`data/ground_truth.json`,
  `data/quant_ground_truth.json`) to the agent in any form
