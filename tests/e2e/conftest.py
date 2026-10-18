"""
Testy E2E używają prawdziwych modeli, bazy RAG i klucza OpenRouter i trwają godzinami,
dlatego uruchamia się je wyłącznie ręcznie. Bez PISMAK_E2E=1 pytest nie zbiera z tego
katalogu żadnego testu, więc zwykłe `python -m pytest` i CI przy pushu ich nie uruchamiają.
"""

import os

collect_ignore_glob = [] if os.getenv("PISMAK_E2E") == "1" else ["test_*.py"]


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    """Print the summary table of the E2E runs at the end of the pytest output."""
    if os.getenv("PISMAK_E2E") != "1":
        return
    from e2e_support import E2ESession, text_summary

    session = config.stash.get(E2ESession.KEY, None)
    if session is None or not session.results:
        return
    terminalreporter.section("Podsumowanie E2E")
    for line in text_summary(session.results):
        terminalreporter.write_line(line)
    terminalreporter.write_line(f"\nRaport: {session.report_path}")
