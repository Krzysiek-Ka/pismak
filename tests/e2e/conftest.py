"""
Testy E2E używają prawdziwych modeli, bazy RAG i klucza OpenRouter i trwają godzinami,
dlatego uruchamia się je wyłącznie ręcznie. Bez PISMAK_E2E=1 pytest nie zbiera z tego
katalogu żadnego testu, więc zwykłe `python -m pytest` i CI przy pushu ich nie uruchamiają.
"""

import os

collect_ignore_glob = [] if os.getenv("PISMAK_E2E") == "1" else ["test_*.py"]
