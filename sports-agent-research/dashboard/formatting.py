"""Display-only formatting helpers (Milestone 13, Sections 15-16).

Pure functions: value in, display string out. Never rounds, clamps, or
otherwise mutates an underlying stored value — only how it is shown.
Missing values are rendered as "N/A", never as a misleading 0/0.0%.
"""

from __future__ import annotations

NOT_AVAILABLE = "N/A"
UNAVAILABLE = "Unavailable"
INSUFFICIENT_EVIDENCE = "Insufficient evidence"


def format_percentage(value: float | None, ndigits: int = 1) -> str:
    """0.833333 -> "83.3%"; None -> "N/A" (Section 15). Never called on
    an already-rounded stored value — formatting only."""
    if value is None:
        return NOT_AVAILABLE
    return f"{value * 100:.{ndigits}f}%"


def format_error(value: float | None, ndigits: int = 6) -> str:
    """Enough decimal precision that a small non-zero error is never
    misrepresented as 0.00 (Section 16). Falls back to scientific
    notation for values so small that fixed-point rounding at `ndigits`
    would display as all zeros, so a genuinely tiny but non-zero error
    is never shown as "0.000000"."""
    if value is None:
        return NOT_AVAILABLE
    if value != 0 and abs(value) < 10 ** (-ndigits):
        return f"{value:.2e}"
    return f"{value:.{ndigits}f}"


def format_odds(value: int | None) -> str:
    if value is None:
        return NOT_AVAILABLE
    return f"{value:+d}"


def format_probability(value: float | None, ndigits: int = 4) -> str:
    if value is None:
        return NOT_AVAILABLE
    return f"{value:.{ndigits}f}"


def format_bool(value: bool | None, true_text: str = "Yes", false_text: str = "No") -> str:
    if value is None:
        return NOT_AVAILABLE
    return true_text if value else false_text


def format_list(values: list[str] | None) -> str:
    if not values:
        return NOT_AVAILABLE
    return ", ".join(values)


def format_enum(value) -> str:
    if value is None:
        return NOT_AVAILABLE
    return getattr(value, "value", str(value)).replace("_", " ").upper()


def format_seconds(value: float | None, ndigits: int = 4) -> str:
    if value is None:
        return NOT_AVAILABLE
    return f"{value:.{ndigits}f}s"


def format_freshness(value: bool | None) -> str:
    """freshness_correct is None when a scenario is not a designated
    freshness scenario at all — that is genuinely "not applicable", a
    different meaning than True/False, so it gets its own label rather
    than collapsing into the generic NOT_AVAILABLE."""
    if value is None:
        return "N/A (not a freshness scenario)"
    return "Current" if value else "Stale/incorrect"

def format_current_stale(is_current: bool | None) -> str:
    if is_current is None:
        return NOT_AVAILABLE
    return "CURRENT" if is_current else "STALE"
