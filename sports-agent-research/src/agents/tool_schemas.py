"""LLM-callable tool schemas and validated dispatcher for the
tool-calling agent (Milestone 9A).

Every tool exposed here is a thin, validated wrapper around the existing
SportsbookTools layer (src/tools/sportsbook_tools.py) — no sportsbook
lookups, JSON parsing, or betting math are implemented here, and no
implementation is duplicated. Tool descriptions intentionally describe
behavior ("current structured sportsbook odds"), never storage details
(JSON file paths, provider class names) — the LLM should reason about
what the tool returns, not how it is stored.

execute_tool() validates the model-supplied arguments against a Pydantic
input model before calling SportsbookTools, and returns both:

  - a JSON-safe structured result, to send back to the LLM as a
    tool_result, and
  - the original Pydantic object(s) SportsbookTools returned, for
    src/agents/tool_agent.py to fold directly into its internal market
    state.

This is the boundary that keeps quantitative analysis fed by structured
Python data rather than by re-parsing the LLM's prose (see
milestones/current.md, Milestone 9A Step 15): tool_agent.py never reads
odds values back out of anything the model said — only out of what this
function actually returned.

Unknown tool names, invalid arguments, and every SportsbookTools/
OddsProvider exception (unknown game, sportsbook, market, or outcome)
propagate unchanged from execute_tool() — never caught or translated
here, so the caller can record an explicit tool failure instead of
fabricating a result (see docs/EXPERIMENT_RULES.md and
src/tools/sportsbook_tools.py's own "Error propagation" note).
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from src.agents.llm_client import ToolSchema
from src.models import MarketType
from src.tools.sportsbook_tools import SportsbookTools


class GetGamesInput(BaseModel):
    """get_games takes no arguments."""


class GetGameInput(BaseModel):
    game_id: str = Field(..., min_length=1)


class GetOddsInput(BaseModel):
    game_id: str = Field(..., min_length=1)
    market_type: MarketType
    selected_outcome: str = Field(..., min_length=1)


class GetSportsbookOddsInput(BaseModel):
    game_id: str = Field(..., min_length=1)
    sportsbook: str = Field(..., min_length=1)
    market_type: MarketType
    selected_outcome: str = Field(..., min_length=1)


class FindBestLineInput(BaseModel):
    game_id: str = Field(..., min_length=1)
    market_type: MarketType
    selected_outcome: str = Field(..., min_length=1)


_MARKET_TYPE_PROPERTY = {
    "type": "string",
    "enum": [market_type.value for market_type in MarketType],
    "description": "The betting market: moneyline, spread, or total.",
}
_GAME_ID_PROPERTY = {"type": "string", "description": "The game's identifier."}
_OUTCOME_PROPERTY = {
    "type": "string",
    "description": "The specific outcome to price, e.g. a team name.",
}

TOOL_SCHEMAS: list[ToolSchema] = [
    ToolSchema(
        name="get_games",
        description=(
            "List every game currently available for analysis. Call this "
            "to discover game_id values or browse what games exist."
        ),
        input_schema={"type": "object", "properties": {}, "required": []},
    ),
    ToolSchema(
        name="get_game",
        description=(
            "Get structured details (teams, sport, start time) for one "
            "specific game by its game_id."
        ),
        input_schema={
            "type": "object",
            "properties": {"game_id": _GAME_ID_PROPERTY},
            "required": ["game_id"],
        },
    ),
    ToolSchema(
        name="get_odds",
        description=(
            "Get current structured sportsbook odds from every "
            "sportsbook currently quoting a price for one exact "
            "game/market/outcome. Use this to see all current prices for "
            "one side of a market."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "game_id": _GAME_ID_PROPERTY,
                "market_type": _MARKET_TYPE_PROPERTY,
                "selected_outcome": _OUTCOME_PROPERTY,
            },
            "required": ["game_id", "market_type", "selected_outcome"],
        },
    ),
    ToolSchema(
        name="get_sportsbook_odds",
        description=(
            "Get the current structured odds from exactly one named "
            "sportsbook for one exact game/market/outcome."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "game_id": _GAME_ID_PROPERTY,
                "sportsbook": {
                    "type": "string",
                    "description": "The sportsbook name, e.g. DraftKings.",
                },
                "market_type": _MARKET_TYPE_PROPERTY,
                "selected_outcome": _OUTCOME_PROPERTY,
            },
            "required": ["game_id", "sportsbook", "market_type", "selected_outcome"],
        },
    ),
    ToolSchema(
        name="find_best_line",
        description=(
            "Find the mathematically best currently available price for "
            "one exact game/market/outcome, and every sportsbook tied at "
            "that price."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "game_id": _GAME_ID_PROPERTY,
                "market_type": _MARKET_TYPE_PROPERTY,
                "selected_outcome": _OUTCOME_PROPERTY,
            },
            "required": ["game_id", "market_type", "selected_outcome"],
        },
    ),
]

_INPUT_MODELS: dict[str, type[BaseModel]] = {
    "get_games": GetGamesInput,
    "get_game": GetGameInput,
    "get_odds": GetOddsInput,
    "get_sportsbook_odds": GetSportsbookOddsInput,
    "find_best_line": FindBestLineInput,
}

KNOWN_TOOL_NAMES = frozenset(_INPUT_MODELS)


def execute_tool(
    tools: SportsbookTools, name: str, arguments: dict[str, Any]
) -> tuple[Any, Any]:
    """Validate `arguments`, call the corresponding SportsbookTools
    method, and return (json_safe_structured_result, raw_pydantic_result).

    Raises ValueError for an unknown tool name, pydantic.ValidationError
    for invalid arguments, or whatever SportsbookTools/OddsProvider raises
    for an unknown game/sportsbook/market/outcome — all left for the
    caller (src/agents/tool_agent.py) to catch and record as an explicit
    tool failure.
    """
    if name not in KNOWN_TOOL_NAMES:
        raise ValueError(f"unknown tool: {name!r}")

    validated = _INPUT_MODELS[name](**arguments)

    if name == "get_games":
        games = tools.get_games()
        return [game.model_dump(mode="json") for game in games], games

    if name == "get_game":
        game = tools.get_game(validated.game_id)
        return game.model_dump(mode="json"), game

    if name == "get_odds":
        odds = tools.get_odds(validated.game_id, validated.market_type, validated.selected_outcome)
        return [o.model_dump(mode="json") for o in odds], odds

    if name == "get_sportsbook_odds":
        odds = tools.get_sportsbook_odds(
            validated.game_id,
            validated.sportsbook,
            validated.market_type,
            validated.selected_outcome,
        )
        return odds.model_dump(mode="json"), odds

    # name == "find_best_line": the only remaining member of KNOWN_TOOL_NAMES.
    best_line = tools.find_best_line(
        validated.game_id, validated.market_type, validated.selected_outcome
    )
    return best_line.model_dump(mode="json"), best_line
