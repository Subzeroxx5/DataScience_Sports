"""Common agent contract shared by the three future architectures
(RAG-only, tool-calling-only, hybrid) — Milestone 8A.

This module defines the interface only. No concrete agent is implemented
here or anywhere else yet (see milestones/current.md — Milestone 8A is
evidence-pipeline-only; Milestone 8B implements the first concrete
RAG-only agent).

AgentRequest is the common experimental input every architecture
receives: enough to identify the scenario/game/market/outcome under
evaluation and the natural-language query to answer, and nothing else.
It deliberately excludes any ground-truth field (expected_best_sportsbook,
expected_ev, estimated_true_probability, market_reference_probability,
etc. — see src/models.py GroundTruth / QuantGroundTruth) so that
constructing a valid request can never leak the answer to whichever
architecture receives it.

BettingAnalysis (src/models.py) remains the common final output contract
every architecture must eventually produce — Agent.analyze() returns one.
This module does not populate it; see milestones/current.md.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from pydantic import BaseModel, Field, field_validator

from src.models import ArchitectureType, BettingAnalysis, MarketType


class AgentRequest(BaseModel):
    """Common experimental input for every architecture.

    Contains no ground truth: no expected sportsbook, no expected odds,
    no reference probability, no EV. Only identity (which scenario/game/
    market/outcome to analyze) and the natural-language query to answer.
    """

    scenario_id: str = Field(..., min_length=1)
    game_id: str = Field(..., min_length=1)
    market_type: MarketType
    selected_outcome: str = Field(..., min_length=1)
    query: str = Field(..., min_length=1)

    @field_validator("scenario_id", "game_id", "selected_outcome", "query")
    @classmethod
    def not_blank(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("field cannot be empty or whitespace")
        return stripped


class Agent(ABC):
    """Common interface every architecture-specific agent implements.

    Concrete subclasses (RAG-only, tool-calling-only, hybrid — none exist
    yet) each fix `architecture` to their own ArchitectureType and
    implement analyze() using only the layers permitted for that
    architecture (see docs/EXPERIMENT_RULES.md). This class enforces
    nothing about *how* analyze() gets its answer, only the shared
    input/output contract every architecture must honor so the experiment
    can compare them on equal terms.
    """

    architecture: ArchitectureType

    @abstractmethod
    def analyze(self, request: AgentRequest) -> BettingAnalysis:
        """Analyze one scenario/query and return a structured BettingAnalysis.

        Must not accept or consult ground truth (data/ground_truth.json,
        data/quant_ground_truth.json, or their generator modules) —
        ground truth is for evaluation only, never an information source
        (see docs/EXPERIMENT_RULES.md, "Ground Truth").
        """
        raise NotImplementedError
