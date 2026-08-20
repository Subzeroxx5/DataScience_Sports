# Citation Needs — Resolved

Audit of every literature/background claim in `docs/MANUSCRIPT_DRAFT.md`
and `docs/MANUSCRIPT_OUTLINE.md` that was originally marked
`[CITATION NEEDED]`. All 10 have since been filled in with real, verified
sources — none fabricated, none cited from memory alone. Full
bibliographic entries are in `docs/MANUSCRIPT_DRAFT.md`'s References
section.

## Verification method

Each source below was located via web search and then independently
confirmed by fetching its original abstract/publication page directly
(arXiv abstract page, NeurIPS proceedings page, or journal page) to check
the exact title, author list, venue, and year before it was written into
the manuscript. No citation in this project was generated purely from
training-data recall without this fetch-based check.

| # | Section | Claim | Source used | Verified via |
|---|---|---|---|---|
| 1 | Abstract / Introduction | LLM agents increasingly rely on external information (retrieval, tools) to answer questions requiring current or verifiable data. | Wang et al. (2023), *A Survey on Large Language Model based Autonomous Agents*, arXiv:2308.11432 | Fetched arXiv abstract page directly |
| 2 | Introduction / Background | Retrieval-augmented generation (RAG): supplying a model with passages retrieved via semantic similarity search from a corpus. | Lewis et al. (2020), *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*, NeurIPS 2020, arXiv:2005.11401 | Web search confirmed against NeurIPS proceedings page and arXiv listing |
| 3 | Introduction / Background | Tool calling / function calling: letting a model invoke predefined structured operations and use their typed results. | Schick et al. (2023), *Toolformer: Language Models Can Teach Themselves to Use Tools*, arXiv:2302.04761 | Fetched arXiv abstract page directly |
| 4 | Background | Hybrid RAG + tool-calling agent designs are comparatively less studied. | Singh et al. (2025), *Agentic Retrieval-Augmented Generation: A Survey on Agentic RAG*, arXiv:2501.09136 | Fetched arXiv abstract page directly |
| 5 | Background | American odds and their conversion to implied probability. | Štrumbelj (2014), *On Determining Probability Forecasts from Betting Odds*, International Journal of Forecasting, 30(4), 934–943 | Web search confirmed against ScienceDirect listing |
| 6 | Background | Sportsbook margin ("vig"/"overround") causes raw implied probabilities to sum to more than 1.0. | Shin (1993), *Measuring the Incidence of Insider Trading in a Market for State-Contingent Claims*, The Economic Journal, 103(420), 1141–1153 | Web search confirmed against multiple academic citation records |
| 7 | Background | No-vig (fair) probability normalization methodology. | Štrumbelj (2014) (same as #5 — directly compares basic normalization vs. Shin's method) | Same as #5 |
| 8 | Background | Market-consensus reference probability (averaging no-vig probabilities across books, often leave-one-out) as a value-detection method. | Wolfers & Zitzewitz (2004), *Prediction Markets*, Journal of Economic Perspectives, 18(2), 107–126 | Web search confirmed against AEA/JEP listing |
| 9 | Background | Expected value (EV) as a standard framing for identifying favorably priced bets, and the distinction between market-consensus probability and true outcome probability. | Levitt (2004), *Why Are Gambling Markets Organised So Differently from Financial Markets?*, The Economic Journal, 114(495), 223–246 | Web search confirmed against Economic Journal listing |
| 10 | Background | LLM output reliability / hallucination — models stating claims not supported by their actual inputs. | Huang et al. (2023), *A Survey on Hallucination in Large Language Models: Principles, Taxonomy, Challenges, and Open Questions*, arXiv:2311.05232 | Fetched arXiv abstract page directly |

## Notes

- These were the only literature-dependent claims in the manuscript
  draft/outline; all quantitative results elsewhere in the manuscript
  remain this project's own findings, traceable to
  `results/experiments/final_v1/analysis/` (Milestone 14B), not to any
  literature source.
- Items 5–9 draw on a small set of sports-betting-market and prediction-
  market economics papers (Shin 1993, Levitt 2004, Wolfers & Zitzewitz
  2004, Štrumbelj 2014) rather than one single reference per bullet, since
  no single source was found that alone covers implied-probability
  conversion, vig, no-vig normalization, market-consensus aggregation,
  and the market-probability-vs-true-probability distinction.
- This project's own quantitative methodology (no-vig normalization,
  leave-one-out consensus, probability edge) was implemented and
  validated independently in `src/calculations/market.py`
  (Milestone 7A) before any of this literature was located — the
  citations here support the background narrative, not the
  implementation, which was verified by its own unit tests.
- Full APA-style entries are in `docs/MANUSCRIPT_DRAFT.md`'s References
  section, not duplicated here.
