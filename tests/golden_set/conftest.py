"""
Złote zbiory modułów (`tests/golden_set/<moduł>/`) uruchamia się wyłącznie ręcznie.

Każdy test z tego katalogu dostaje marker `golden_set`, który zwykłe `python -m pytest` i CI
pomijają (`addopts` w pyproject.toml). Uruchomienie: `python -m pytest tests/golden_set -m
golden_set`; same testy na modelach: `-m llm`.
"""

from pathlib import Path

import pytest

GOLDEN_SET_DIR = Path(__file__).resolve().parent


@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(config, items):
    """Mark every golden-set test before pytest filters the tests by marker."""
    for item in items:
        if GOLDEN_SET_DIR in Path(item.path).resolve().parents:
            item.add_marker(pytest.mark.golden_set)
