from __future__ import annotations

import logging
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from threading import Lock
from time import perf_counter

_LLM_LOCK = Lock()
_LOGGER = logging.getLogger("pismak")
# Najdłuższe pojedyncze żądanie do lokalnego modelu trwa do 900 s
# (OLLAMA_REQUEST_TIMEOUT_SECONDS w serwer.py). Kto czeka dłużej, stoi za kilkoma
# sprawami albo za zawieszonym żądaniem, więc dostaje komunikat zamiast czekać bez końca.
LLM_LOCK_WAIT_TIMEOUT_SECONDS = 900.0
# Ollama na jednym komputerze obsługuje jedno żądanie naraz. OpenRouter nie obciąża
# lokalnej maszyny, więc jego żądania nie czekają w tej kolejce.
UNQUEUED_PROVIDERS = frozenset({"openrouter"})


class LLMBusyError(RuntimeError):
    """Raised when a local model request waited too long for its turn."""


@contextmanager
def llm_generation_lock(
    on_wait: Callable[[], None] | None = None,
    *,
    stage: str = "llm_generation",
    session_id: str | None = None,
    provider: str = "ollama",
) -> Iterator[None]:
    """Serialize local LLM requests, notify callers that must wait and bound the wait."""
    started = perf_counter()
    queued = provider not in UNQUEUED_PROVIDERS
    if queued and not _LLM_LOCK.acquire(blocking=False):
        if on_wait is not None:
            on_wait()
        if not _LLM_LOCK.acquire(timeout=LLM_LOCK_WAIT_TIMEOUT_SECONDS):
            _LOGGER.warning(
                "llm_wait_timeout stage=%s session_id=%s provider=%s wait_seconds=%.3f",
                stage,
                session_id or "-",
                provider,
                perf_counter() - started,
                extra={"session_id": session_id},
            )
            raise LLMBusyError(
                f"Lokalny model był zajęty dłużej niż {LLM_LOCK_WAIT_TIMEOUT_SECONDS:.0f} s."
            )
    acquired_at = perf_counter()
    _LOGGER.info(
        "llm_started stage=%s session_id=%s provider=%s wait_seconds=%.3f",
        stage,
        session_id or "-",
        provider,
        acquired_at - started,
        extra={"session_id": session_id},
    )

    try:
        yield
    except Exception as exc:
        _LOGGER.error(
            "llm_failed stage=%s session_id=%s error_type=%s",
            stage,
            session_id or "-",
            type(exc).__name__,
            extra={"session_id": session_id},
        )
        raise
    finally:
        if queued:
            _LLM_LOCK.release()
        _LOGGER.info(
            "llm_completed stage=%s session_id=%s elapsed_seconds=%.3f",
            stage,
            session_id or "-",
            perf_counter() - acquired_at,
            extra={"session_id": session_id},
        )
