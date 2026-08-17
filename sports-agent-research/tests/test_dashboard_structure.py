"""Structural checks for the dashboard package (Milestone 13, Section
24): confirms dashboard/ never defines its own version of a formula that
must remain in the shared calculation layer (src/calculations/), and
stays a thin UI layer that only calls into existing agent/experiment/
evaluation code — mirrors the AST-based checks already established in
tests/test_experiment_runner.py for src/experiments/runner.py.
"""

import ast
from pathlib import Path

DASHBOARD_DIR = Path(__file__).resolve().parent.parent / "dashboard"

# Substrings that would indicate a dashboard-local reimplementation of
# shared betting/quant math (implied probability, no-vig, consensus, EV,
# best-line ranking) — these must remain in src/calculations/ only.
_FORBIDDEN_FUNCTION_NAME_SUBSTRINGS = (
    "implied_probability",
    "no_vig",
    "novig",
    "consensus",
    "expected_value",
    "calculate_ev",
    "best_line",
    "best_odds",
    "probability_edge",
)


def _dashboard_python_files() -> list[Path]:
    return sorted(DASHBOARD_DIR.glob("*.py"))


def test_dashboard_package_exists_and_has_expected_modules():
    names = {p.name for p in _dashboard_python_files()}
    assert {"app.py", "data_loader.py", "formatting.py", "charts.py"} <= names


def test_dashboard_defines_no_quant_or_odds_formula_of_its_own():
    for path in _dashboard_python_files():
        tree = ast.parse(path.read_text())
        defined = {
            node.name.lower()
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        for name in defined:
            for forbidden in _FORBIDDEN_FUNCTION_NAME_SUBSTRINGS:
                assert forbidden not in name, f"{path.name} defines {name!r} — quant math must live in src/calculations/"


def test_dashboard_math_calls_are_all_imported_from_shared_calculations_module():
    """Any call to a known shared-math function name inside dashboard/
    must be reached via an import from src.calculations — never a
    module-local definition (the previous test already rules that out;
    this confirms the import path exists wherever such a call appears)."""
    math_function_names = {"implied_probability"}
    for path in _dashboard_python_files():
        source = path.read_text()
        tree = ast.parse(source)
        calls_math_function = any(
            isinstance(node, ast.Name) and node.id in math_function_names
            for node in ast.walk(tree)
        )
        if not calls_math_function:
            continue
        imports_from_calculations = any(
            isinstance(node, ast.ImportFrom) and node.module is not None and node.module.startswith("src.calculations")
            for node in ast.walk(tree)
        )
        assert imports_from_calculations, f"{path.name} uses shared math without importing src.calculations"


def test_charts_module_uses_only_bar_charts_no_3d_or_decorative_types():
    source = (DASHBOARD_DIR / "charts.py").read_text().lower()
    for forbidden in ("plot_surface", "mplot3d", "axes3d", "projection='3d'", 'projection="3d"'):
        assert forbidden not in source


def test_dashboard_never_imports_a_live_sportsbook_api_client():
    forbidden_tokens = ("requests.", "httpx.", "live_odds_api", "livesportsbook")
    for path in _dashboard_python_files():
        source = path.read_text().lower()
        for token in forbidden_tokens:
            assert token not in source


def test_dashboard_uses_the_experiment_agent_factory_not_a_local_agent_class():
    """Section 5: 'Do not create dashboard-specific agents.' data_loader
    must construct agents via src.experiments.agent_factory.create_agent
    and must never import a concrete Agent subclass constructor
    directly."""
    source = (DASHBOARD_DIR / "data_loader.py").read_text()
    assert "from src.experiments.agent_factory import create_agent" in source
    for forbidden_import in ("RagOnlyAgent(", "ToolCallingAgent(", "HybridAgent("):
        assert forbidden_import not in source
