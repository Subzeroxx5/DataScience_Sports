"""Deterministic RAG corpus generator (Milestone 6B).

Data flow:

    Controlled Benchmark (data/current_odds.json, historical_odds.json,
    test_scenarios.json -- game/market fields only)
          |
          v
    Deterministic Transformation (this module: fixed text templates,
    no LLM, no randomness, no wall-clock time)
          |
          v
    Validated RagDocument objects (src/rag/documents.py)
          |
          v
    data/rag_documents/corpus.jsonl

Hard constraints enforced by construction, not just by convention:

- This module never imports src.evaluation.ground_truth and never reads
  data/ground_truth.json. Ground truth must never leak into the corpus
  (see tests/test_rag_corpus.py for an automated leakage scan of the
  generated content as a second line of defense).
- This module never reads test_scenarios.json's `estimated_true_probability`
  or `category` fields — only `game` and `market.market_type` /
  `market.selected_outcome`, which are objective facts about the
  benchmark, not ground-truth-adjacent values.
- No uuid4/datetime.now()/random: every document_id, market_id, and piece
  of content is a fixed function of the source data, so two runs against
  unchanged source data produce a byte-identical corpus.jsonl.

Scope: only moneyline markets get sportsbook_snapshot / market_context
documents, matching the project's established convention (see
data/README.md "Market Type Scope") that spread/total scenarios (S012,
S013) are schema-validation-only, not part of the core EV-analysis
benchmark. game_context documents are generated for every game regardless
of market type, since they carry no odds or market-specific content.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from src.evaluation.dataset import (
    DATA_DIR,
    load_current_odds_records,
    load_historical_odds_records,
    load_scenario_definitions,
)
from src.models import MarketType
from src.rag.documents import RagDocument, RagSourceType

RAG_DOCUMENTS_DIR = DATA_DIR / "rag_documents"
CORPUS_PATH = RAG_DOCUMENTS_DIR / "corpus.jsonl"


def _slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.strip().lower()).strip("-")


def _format_odds(american_odds: int) -> str:
    return f"+{american_odds}" if american_odds > 0 else str(american_odds)


def _load_games() -> dict[str, dict]:
    """Map game_id -> {home_team, away_team, sport}, from test_scenarios.json's
    `game` field only. Never reads estimated_true_probability or category."""
    games: dict[str, dict] = {}
    for definition in load_scenario_definitions():
        game = definition["game"]
        games[game["game_id"]] = {
            "home_team": game["home_team"],
            "away_team": game["away_team"],
            "sport": game["sport"],
        }
    return games


def _load_moneyline_game_ids() -> set[str]:
    """game_ids whose scenario market is moneyline, from test_scenarios.json's
    `market.market_type` field only."""
    return {
        definition["game"]["game_id"]
        for definition in load_scenario_definitions()
        if definition["market"]["market_type"] == MarketType.MONEYLINE.value
    }


def _build_sportsbook_snapshot(record: dict, games: dict[str, dict]) -> RagDocument:
    game = games[record["game_id"]]
    market_id = f"{record['game_id']}-{record['market_type']}"
    formatted_odds = _format_odds(record["american_odds"])
    document_id = "-".join(
        [
            _slugify(record["game_id"]),
            record["market_type"],
            _slugify(record["selected_outcome"]),
            _slugify(record["sportsbook"]),
            record["version"],
        ]
    )

    if record["is_current"]:
        content = (
            f"{record['sportsbook']} lists {record['selected_outcome']} at "
            f"{formatted_odds} on the moneyline for the {game['home_team']} "
            f"versus {game['away_team']} game."
        )
    else:
        content = (
            f"As of an earlier update, {record['sportsbook']} had listed "
            f"{record['selected_outcome']} at {formatted_odds} on the moneyline "
            f"for the {game['home_team']} versus {game['away_team']} game."
        )

    return RagDocument(
        document_id=document_id,
        source_type=RagSourceType.SPORTSBOOK_SNAPSHOT,
        content=content,
        game_id=record["game_id"],
        market_id=market_id,
        market_type=MarketType(record["market_type"]),
        selected_outcome=record["selected_outcome"],
        sportsbook=record["sportsbook"],
        american_odds=record["american_odds"],
        is_current=record["is_current"],
        timestamp=record.get("timestamp"),
        version=record["version"],
    )


def _build_game_context(game_id: str, game: dict) -> RagDocument:
    content = (
        f"{game['home_team']} host the {game['away_team']} in this controlled "
        f"research scenario ({game['sport']}). This matchup is synthetic and "
        f"used only to evaluate agent architectures; it does not represent a "
        f"real-world event."
    )
    return RagDocument(
        document_id=f"{_slugify(game_id)}-game-context",
        source_type=RagSourceType.GAME_CONTEXT,
        content=content,
        game_id=game_id,
    )


def _build_market_context(game_id: str, game: dict) -> RagDocument:
    market_id = f"{game_id}-{MarketType.MONEYLINE.value}"
    content = (
        f"This controlled moneyline market for the {game['home_team']} versus "
        f"{game['away_team']} game is tracked by multiple sportsbooks. Each "
        f"sportsbook may offer a different price for either team, and prices "
        f"may change over time in this research dataset."
    )
    return RagDocument(
        document_id=f"{_slugify(game_id)}-moneyline-market-context",
        source_type=RagSourceType.MARKET_CONTEXT,
        content=content,
        game_id=game_id,
        market_id=market_id,
        market_type=MarketType.MONEYLINE,
    )


def generate_documents() -> list[RagDocument]:
    games = _load_games()
    moneyline_game_ids = _load_moneyline_game_ids()
    current_odds_records = load_current_odds_records()
    historical_odds_records = load_historical_odds_records()

    documents: list[RagDocument] = []

    for record in current_odds_records:
        if record["market_type"] != MarketType.MONEYLINE.value:
            continue
        documents.append(_build_sportsbook_snapshot(record, games))

    for record in historical_odds_records:
        if record["market_type"] != MarketType.MONEYLINE.value:
            continue
        documents.append(_build_sportsbook_snapshot(record, games))

    for game_id, game in games.items():
        documents.append(_build_game_context(game_id, game))

    for game_id in sorted(moneyline_game_ids):
        documents.append(_build_market_context(game_id, games[game_id]))

    documents.sort(key=_sort_key)
    return documents


def _sort_key(document: RagDocument) -> tuple:
    return (
        document.game_id,
        document.market_id or "",
        document.selected_outcome or "",
        document.sportsbook or "",
        document.version or "",
        document.document_id,
    )


def export_corpus(output_path: Path | None = None) -> Path:
    output_path = output_path or CORPUS_PATH
    output_path.parent.mkdir(parents=True, exist_ok=True)
    documents = generate_documents()
    with output_path.open("w") as f:
        for document in documents:
            payload = document.model_dump(mode="json")
            f.write(json.dumps(payload, sort_keys=True))
            f.write("\n")
    return output_path


if __name__ == "__main__":
    written_path = export_corpus()
    documents = generate_documents()
    print(f"Wrote {len(documents)} RAG documents to {written_path}")
