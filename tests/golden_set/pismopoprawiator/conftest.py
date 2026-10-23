"""Tabela podsumowania złotego zbioru Pismopoprawiatora na końcu wyjścia pytest."""


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    """Print the per-model and per-mode summary after a run on real models."""
    from pismopoprawiator_support import GoldenSession

    session = config.stash.get(GoldenSession.KEY, None)
    if session is None or not session.runs:
        return
    terminalreporter.section("Podsumowanie złotego zbioru Pismopoprawiatora")
    for line in session.text_summary():
        terminalreporter.write_line(line)
    terminalreporter.write_line(f"\nRaport: {session.report_path}")
