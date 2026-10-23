"""
Testy E2E używają prawdziwych modeli, bazy RAG i klucza OpenRouter i trwają godzinami,
dlatego uruchamia się je wyłącznie ręcznie, tak jak złote zbiory (`tests/golden_set/`).

Każdy test z tego katalogu dostaje marker `e2e`, który zwykłe `python -m pytest` i CI
pomijają (`addopts` w pyproject.toml). Uruchomienie wymaga `-m e2e`.
"""

from pathlib import Path

import pytest

E2E_DIR = Path(__file__).resolve().parent


@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(config, items):
    """Mark every E2E test before pytest filters the tests by marker."""
    for item in items:
        if E2E_DIR in Path(item.path).resolve().parents:
            item.add_marker(pytest.mark.e2e)


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    """Print the summary table of the E2E runs at the end of the pytest output."""
    from e2e_support import E2ESession, text_summary

    # Sesja istnieje tylko wtedy, gdy w tym przebiegu ruszył choć jeden test E2E.
    session = config.stash.get(E2ESession.KEY, None)
    if session is None or not session.results:
        return
    terminalreporter.section("Podsumowanie E2E")
    for line in text_summary(session.results):
        terminalreporter.write_line(line)
    terminalreporter.write_line(f"\nRaport: {session.report_path}")
