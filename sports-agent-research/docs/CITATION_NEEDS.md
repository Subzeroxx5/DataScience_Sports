# Citation Needs

Audit of every literature/background claim in `docs/MANUSCRIPT_DRAFT.md` and
`docs/MANUSCRIPT_OUTLINE.md` marked `[CITATION NEEDED]`. No sources are
proposed, browsed, or invented in this milestone — this file only lists
what needs one and what kind.

| # | Section | Claim | Type of source needed |
|---|---|---|---|
| 1 | Abstract / Introduction | LLM agents increasingly rely on external information (retrieval, tools) to answer questions requiring current or verifiable data. | Survey/overview paper on LLM agents and tool/retrieval augmentation. |
| 2 | Introduction / Background | Retrieval-augmented generation (RAG): supplying a model with passages retrieved via semantic similarity search from a corpus. | Foundational RAG paper(s) (e.g. the original RAG architecture proposal and a modern survey). |
| 3 | Introduction / Background | Tool calling / function calling: letting a model invoke predefined structured operations and use their typed results. | Foundational or widely cited tool-use/function-calling paper, or official API documentation describing the pattern generally. |
| 4 | Background | Hybrid RAG + tool-calling agent designs are comparatively less studied. | Survey or comparative study of hybrid retrieval+tool-use agent architectures. |
| 5 | Background | American odds and their conversion to implied probability. | Sports-betting mathematics reference (textbook, industry glossary, or methodology paper). |
| 6 | Background | Sportsbook margin ("vig"/"overround") causes raw implied probabilities to sum to more than 1.0. | Same as #5 — sports-betting/market-microstructure reference. |
| 7 | Background | No-vig (fair) probability normalization methodology. | Same as #5. |
| 8 | Background | Market-consensus reference probability (averaging no-vig probabilities across books, often leave-one-out) as a value-detection method. | Sports-betting analytics reference or academic market-efficiency paper. |
| 9 | Background | Expected value (EV) as a standard framing for identifying favorably priced bets, and the distinction between market-consensus probability and true outcome probability. | Sports-betting analytics reference; possibly a sports-market-efficiency academic paper. |
| 10 | Background | LLM output reliability / hallucination — models stating claims not supported by their actual inputs. | Survey paper on LLM hallucination / faithfulness in agentic or RAG contexts. |

## Notes

- These are the only literature-dependent claims in the current manuscript
  draft/outline; all quantitative results elsewhere in the manuscript are
  this project's own findings and are traceable to
  `results/experiments/final_v1/analysis/` (Milestone 14B), not literature
  claims.
- No academic references have been added, searched for, or fabricated as
  part of this milestone. Filling in `[CITATION NEEDED]` markers with real
  sources is explicitly deferred to future work on the manuscript.
