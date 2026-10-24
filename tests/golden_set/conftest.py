"""
Złote zbiory modułów (`tests/golden_set/<moduł>/`) uruchamia się wyłącznie ręcznie.

Każdy moduł testów zaczyna się od `pytestmark = pytest.mark.golden`, a zwykłe `python -m pytest`
i CI pomijają ten marker (`addopts` w pyproject.toml). Uruchomienie:
`python -m pytest tests/golden_set -m golden -v -s`.
"""

from __future__ import annotations


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    """Print one line per golden run, with its status and report, after the session."""
    from tests.golden_set.golden_support import PROJECT_ROOT, SESSION_RUNS

    if not SESSION_RUNS:
        return
    terminalreporter.section("Złote zbiory — przebiegi")
    for run in SESSION_RUNS:
        terminalreporter.write_line(run.summary())
        terminalreporter.write_line(f"  {run.result_path.relative_to(PROJECT_ROOT)}")
