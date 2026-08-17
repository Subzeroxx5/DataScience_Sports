"""Tests for dashboard/formatting.py (Milestone 13, Sections 15-16):
percentage/N/A formatting, error precision, and that formatting never
mutates the underlying value it's given."""

from dashboard import formatting as fmt


def test_format_percentage_converts_fraction_to_percent_string():
    assert fmt.format_percentage(0.833333) == "83.3%"


def test_format_percentage_none_is_not_available():
    assert fmt.format_percentage(None) == "N/A"


def test_format_percentage_zero_is_shown_as_zero_percent_not_na():
    # 0.0 is a real, meaningful value (e.g. 0% accuracy) — must not be
    # conflated with "missing".
    assert fmt.format_percentage(0.0) == "0.0%"


def test_format_error_default_precision_preserves_small_nonzero_value():
    # A naive 2-4 digit rounding would show this as "0.0000" — Section 16
    # explicitly forbids that for a genuinely non-zero error.
    assert fmt.format_error(0.0000012345) != "0.000000"
    assert float(fmt.format_error(0.0000012345)) != 0.0


def test_format_error_tiny_value_falls_back_to_scientific_notation():
    tiny = 1e-9
    result = fmt.format_error(tiny, ndigits=6)
    assert "e" in result.lower()
    assert result != "0.000000"


def test_format_error_none_is_not_available():
    assert fmt.format_error(None) == "N/A"


def test_format_error_zero_is_exact_zero_not_na():
    assert fmt.format_error(0.0) == "0.000000"


def test_format_odds_shows_explicit_sign():
    assert fmt.format_odds(130) == "+130"
    assert fmt.format_odds(-110) == "-110"


def test_format_odds_none_is_not_available():
    assert fmt.format_odds(None) == "N/A"


def test_format_bool_true_false_and_none():
    assert fmt.format_bool(True) == "Yes"
    assert fmt.format_bool(False) == "No"
    assert fmt.format_bool(None) == "N/A"


def test_format_list_empty_and_none():
    assert fmt.format_list([]) == "N/A"
    assert fmt.format_list(None) == "N/A"
    assert fmt.format_list(["FanDuel", "DraftKings"]) == "FanDuel, DraftKings"


def test_format_freshness_none_is_labeled_not_applicable_distinct_from_false():
    # None (not a freshness scenario) must read differently from False
    # (a freshness scenario that was answered incorrectly).
    assert fmt.format_freshness(None) != fmt.format_freshness(False)
    assert "not a freshness scenario" in fmt.format_freshness(None)


def test_format_current_stale():
    assert fmt.format_current_stale(True) == "CURRENT"
    assert fmt.format_current_stale(False) == "STALE"
    assert fmt.format_current_stale(None) == "N/A"


def test_formatting_never_mutates_input_value():
    value = 0.123456789
    fmt.format_error(value)
    fmt.format_percentage(value)
    assert value == 0.123456789  # unchanged — formatting is display-only
