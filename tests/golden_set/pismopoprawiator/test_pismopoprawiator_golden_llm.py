"""
Złoty zbiór Pismopoprawiatora na prawdziwych modelach. Tylko ręcznie: `-m golden_set` albo
`-m llm`.

Każdy model z MODELE (albo PISMAK_GOLDEN_MODELE) poprawia każde pismo z CASES w dwóch trybach:
całe pismo w jednym wywołaniu i każdy akapit osobno. Test nie przechodzi, gdy korekta zmieniła
treść (cel: 0 naruszeń). Liczba naprawionych błędów i czasy trafiają do raportu
outputs/golden_set/pismopoprawiator/<czas>_<etykieta>/report.md, przepisywanego po każdym
uruchomieniu. Opcjonalnie PISMAK_GOLDEN_LABEL nazywa przebieg.
"""

from __future__ import annotations

import os

import pytest
from dotenv import load_dotenv
from pismopoprawiator_cases import CASES, TRYBY, GoldenCase, selected_models
from pismopoprawiator_support import (
    PROJECT_ROOT,
    GoldenSession,
    available_ollama_models,
    model_problem,
    resolve_model,
)

pytestmark = pytest.mark.llm

RUNS = [
    pytest.param(model, mode, case, id=f"{model}-{mode}-{case.id}")
    for model in selected_models()
    for mode in TRYBY
    for case in CASES
]


@pytest.fixture(scope="session")
def ollama_models() -> list[str] | None:
    load_dotenv(PROJECT_ROOT / ".env")
    return available_ollama_models()


@pytest.fixture(scope="session")
def golden_session(request: pytest.FixtureRequest):
    session = GoldenSession(label=os.getenv("PISMAK_GOLDEN_LABEL", "run"))
    # conftest.py drukuje z niej tabelę podsumowania na końcu wyjścia pytest.
    request.config.stash[GoldenSession.KEY] = session
    yield session
    session.close()


@pytest.mark.parametrize(("model_name", "mode", "case"), RUNS)
def test_pismopoprawiator_golden(
    model_name: str,
    mode: str,
    case: GoldenCase,
    ollama_models: list[str] | None,
    golden_session: GoldenSession,
) -> None:
    """Verify a real model corrects the case's language without changing its content.

    The edge scenario is the case title (seeded errors next to protected numbers, names,
    quotes and markers); the run fails on any content violation, fixes only go to the report.
    """
    model = resolve_model(model_name)
    reason = model_problem(model, ollama_models)
    if reason:
        golden_session.add_not_run(model, mode, case, reason)
        pytest.skip(f"NIEURUCHOMIONY: {reason}")

    run = golden_session.run(model, mode, case)

    print(
        f"\n{run.id}: {run.status}, korekta {run.correction_status}, naruszenia "
        f"{len(run.violations)}, naprawione {len(run.fixed)}/{len(run.fixed) + len(run.missed)}, "
        f"nadmiarowe {len(run.extra)}, czas {run.elapsed_seconds:.0f} s"
    )
    assert run.violations == [], (
        f"{run.id}: " + " | ".join(run.violations) + f" — raport: {golden_session.report_path}"
    )
