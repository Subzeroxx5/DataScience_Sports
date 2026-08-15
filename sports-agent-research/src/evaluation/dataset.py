"""Loaders for the controlled sportsbook dataset.

This module only assembles Pydantic models from the static JSON dataset
files in data/. It contains no betting math, no tools, no RAG, and no
agent logic — see src/calculations/odds_math.py for the deterministic math
and src/evaluation/ground_truth.py for ground-truth generation.

Dataset layout (data/):
- current_odds.json    flat ledger of current, structured sportsbook prices
- historical_odds.json flat ledger of stale/superseded prices (freshness
                        scenarios only; not used to build TestScenario
                        objects, reserved for future RAG document ingestion)
- test_scenarios.json  scenario definitions (game, market, reference
                        probability) that reference current_odds.json by
                        game_id rather than duplicating odds data
"""

from __future__ import annotations

import json
from pathlib import Path

from src.models import Game, Market, SportsbookOdds, TestScenario

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"


def _load_json(filename: str) -> list[dict]:
    path = DATA_DIR / filename
    with path.open() as f:
        return json.load(f)


def load_current_odds_records() -> list[dict]:
    return _load_json("current_odds.json")


def load_historical_odds_records() -> list[dict]:
    return _load_json("historical_odds.json")


def load_scenario_definitions() -> list[dict]:
    return _load_json("test_scenarios.json")


def _odds_records_for_game(game_id: str, records: list[dict]) -> list[dict]:
    return [r for r in records if r["game_id"] == game_id]


def build_test_scenario(definition: dict, current_odds_records: list[dict]) -> TestScenario:
    """Join a scenario definition with its current odds records by game_id."""
    game = Game(**definition["game"])
    market = Market(**definition["market"])
    game_records = _odds_records_for_game(game.game_id, current_odds_records)
    if not game_records:
        raise ValueError(f"no current odds records found for game_id={game.game_id!r}")
    sportsbook_odds = [
        SportsbookOdds(
            sportsbook=record["sportsbook"],
            american_odds=record["american_odds"],
            is_current=record["is_current"],
            timestamp=record.get("timestamp"),
        )
        for record in game_records
    ]
    return TestScenario(
        scenario_id=definition["scenario_id"],
        game=game,
        market=market,
        sportsbook_odds=sportsbook_odds,
        estimated_true_probability=definition["estimated_true_probability"],
    )


def load_test_scenarios() -> list[TestScenario]:
    definitions = load_scenario_definitions()
    current_odds_records = load_current_odds_records()
    return [build_test_scenario(d, current_odds_records) for d in definitions]


def load_scenario_definitions_by_id() -> dict[str, dict]:
    return {d["scenario_id"]: d for d in load_scenario_definitions()}
