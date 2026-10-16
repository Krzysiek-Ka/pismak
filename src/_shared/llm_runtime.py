from __future__ import annotations

import logging
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from threading import Lock
from time import perf_counter

_LLM_LOCK = Lock()
_LOGGER = logging.getLogger("pismak")


@contextmanager
def llm_generation_lock(
    on_wait: Callable[[], None] | None = None,
    *,
    stage: str = "llm_generation",
    session_id: str | None = None,
) -> Iterator[None]:
    """Serialize local LLM requests and notify callers that must wait."""
    started = perf_counter()
    acquired = _LLM_LOCK.acquire(blocking=False)
    if not acquired:
        if on_wait is not None:
            on_wait()
        _LLM_LOCK.acquire()
    acquired_at = perf_counter()
    _LOGGER.info(
        "llm_started stage=%s session_id=%s wait_seconds=%.3f",
        stage,
        session_id or "-",
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
        _LLM_LOCK.release()
        _LOGGER.info(
            "llm_completed stage=%s session_id=%s elapsed_seconds=%.3f",
            stage,
            session_id or "-",
            perf_counter() - acquired_at,
            extra={"session_id": session_id},
        )
