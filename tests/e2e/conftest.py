"""
Testy E2E używają prawdziwych modeli, bazy RAG i klucza OpenRouter i trwają godzinami,
dlatego uruchamia się je wyłącznie ręcznie, tak jak złote zbiory (`tests/golden_set/`).

Każdy moduł testów zaczyna się od `pytestmark = pytest.mark.e2e`, a zwykłe `python -m pytest`
i CI pomijają ten marker (`addopts` w pyproject.toml). Uruchomienie wymaga `-m e2e`.
"""


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    """Print the summary table of the E2E runs at the end of the pytest output."""
    from tests.e2e.e2e_support import E2ESession, text_summary

    # Sesja istnieje tylko wtedy, gdy w tym przebiegu ruszył choć jeden test E2E.
    session = config.stash.get(E2ESession.KEY, None)
    if session is None or not session.results:
        return
    terminalreporter.section("Podsumowanie E2E")
    for line in text_summary(session.results):
        terminalreporter.write_line(line)
    terminalreporter.write_line(f"\nRaport: {session.report_path}")
