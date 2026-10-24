"""Tests for the shared lock that serializes LLM generations."""

from __future__ import annotations

import logging
from threading import Event, Thread

import pytest

from src._shared import llm_runtime
from src._shared.llm_runtime import llm_generation_lock


class _RecordHandler(logging.Handler):
    def __init__(self) -> None:
        super().__init__(logging.DEBUG)
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


@pytest.fixture
def records():
    """Collect records of the project logger regardless of its propagation settings."""
    handler = _RecordHandler()
    logger = logging.getLogger("pismak")
    previous_level = logger.level
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    yield handler.records
    logger.removeHandler(handler)
    logger.setLevel(previous_level)


def _messages(records: list[logging.LogRecord]) -> list[str]:
    return [record.getMessage() for record in records]


class TestLlmGenerationLock:
    def test_llm_generation_lock__serializes_calls_and_notifies_waiters(self) -> None:
        """Verify concurrent local generations queue and notify a waiting session."""
        first_entered = Event()
        release_first = Event()
        second_entered = Event()
        notified = Event()

        def first_call() -> None:
            with llm_generation_lock():
                first_entered.set()
                release_first.wait(timeout=2)

        def second_call() -> None:
            with llm_generation_lock(on_wait=notified.set):
                second_entered.set()

        first = Thread(target=first_call)
        second = Thread(target=second_call)
        first.start()
        assert first_entered.wait(timeout=1)
        second.start()

        assert notified.wait(timeout=1)
        assert not second_entered.is_set()

        release_first.set()
        first.join(timeout=1)
        second.join(timeout=1)

        assert not first.is_alive()
        assert not second.is_alive()
        assert second_entered.is_set()

    def test_llm_generation_lock__does_not_notify_when_the_lock_is_free(self) -> None:
        """Verify a session that does not wait is not shown a misleading queue message."""
        notifications: list[str] = []

        with llm_generation_lock(on_wait=lambda: notifications.append("wait")):
            pass

        assert notifications == []

    def test_llm_generation_lock__releases_the_lock_after_a_failed_generation(self) -> None:
        """Verify one failing request cannot block every later generation in the process."""
        with pytest.raises(RuntimeError, match="model down"):
            with llm_generation_lock():
                raise RuntimeError("model down")

        assert not llm_runtime._LLM_LOCK.locked()

    def test_llm_generation_lock__waiter_whose_notification_fails_does_not_take_the_lock(
        self,
    ) -> None:
        """Verify an interrupted waiter (e.g. a Streamlit rerun) leaves the lock to its holder."""
        holder_entered = Event()
        release_holder = Event()

        def holder() -> None:
            with llm_generation_lock():
                holder_entered.set()
                release_holder.wait(timeout=2)

        thread = Thread(target=holder)
        thread.start()
        assert holder_entered.wait(timeout=1)

        class Rerun(Exception):
            pass

        def interrupted() -> None:
            raise Rerun()

        try:
            with pytest.raises(Rerun):
                with llm_generation_lock(on_wait=interrupted):
                    raise AssertionError("A waiter must not enter while the lock is held")
        finally:
            release_holder.set()
            thread.join(timeout=1)

        assert not llm_runtime._LLM_LOCK.locked()

    def test_llm_generation_lock__logs_other_failures_with_type_but_without_message(
        self, records
    ) -> None:
        """Verify failure logs name the error type and never copy its possibly private text."""
        with pytest.raises(ValueError):
            with llm_generation_lock(stage="brief"):
                raise ValueError("treść sprawy użytkownika")

        failed = [r for r in records if r.getMessage().startswith("llm_failed")]
        assert len(failed) == 1
        assert failed[0].levelno == logging.ERROR
        assert "error_type=ValueError" in failed[0].getMessage()
        assert "treść sprawy użytkownika" not in failed[0].getMessage()

    def test_llm_generation_lock__logs_start_and_completion_with_the_case_id(self, records) -> None:
        """Verify both lifecycle records carry the session id used to route them to a case log."""
        with llm_generation_lock(stage="planner", session_id="case-7"):
            pass

        lifecycle = [
            r for r in records if r.getMessage().startswith(("llm_started", "llm_completed"))
        ]
        assert [r.getMessage().split()[0] for r in lifecycle] == ["llm_started", "llm_completed"]
        assert all(getattr(r, "session_id", None) == "case-7" for r in lifecycle)

    def test_llm_generation_lock__logs_completion_even_when_generation_fails(self, records) -> None:
        """Verify elapsed time is recorded for failed calls too, so slow failures stay visible."""
        with pytest.raises(RuntimeError):
            with llm_generation_lock(stage="reranker"):
                raise RuntimeError("timeout")

        assert any(m.startswith("llm_completed stage=reranker") for m in _messages(records))

    def test_llm_generation_lock__uses_a_dash_when_no_case_is_known(self, records) -> None:
        """Verify log lines stay parseable when a generation runs outside a UI session."""
        with llm_generation_lock():
            pass

        assert any("session_id=- " in m for m in _messages(records))

    def test_llm_generation_lock__waiter_gets_busy_error_after_the_wait_limit(
        self, monkeypatch, records
    ) -> None:
        """Verify a second case stops waiting with a clear error instead of hanging forever."""
        monkeypatch.setattr(llm_runtime, "LLM_LOCK_WAIT_TIMEOUT_SECONDS", 0.05)
        notified = Event()
        entered = Event()

        with llm_generation_lock():
            with pytest.raises(llm_runtime.LLMBusyError):
                with llm_generation_lock(on_wait=notified.set, stage="draft", session_id="case-2"):
                    entered.set()

        assert notified.is_set()
        assert not entered.is_set()
        assert not llm_runtime._LLM_LOCK.locked()
        timeouts = [r for r in records if r.getMessage().startswith("llm_wait_timeout")]
        assert len(timeouts) == 1
        assert "stage=draft session_id=case-2" in timeouts[0].getMessage()
        assert getattr(timeouts[0], "session_id", None) == "case-2"

    def test_llm_generation_lock__lock_is_free_for_the_next_case_after_a_timeout(
        self, monkeypatch
    ) -> None:
        """Verify a waiter that gave up does not leave the lock taken or released twice."""
        monkeypatch.setattr(llm_runtime, "LLM_LOCK_WAIT_TIMEOUT_SECONDS", 0.05)
        with llm_generation_lock():
            with pytest.raises(llm_runtime.LLMBusyError):
                with llm_generation_lock():
                    pass

        with llm_generation_lock():
            assert llm_runtime._LLM_LOCK.locked()
        assert not llm_runtime._LLM_LOCK.locked()

    def test_llm_generation_lock__openrouter_does_not_wait_for_the_local_model(self) -> None:
        """Verify an external provider runs while Ollama is busy and leaves the local lock alone."""
        notifications: list[str] = []

        with llm_generation_lock():
            with llm_generation_lock(
                on_wait=lambda: notifications.append("wait"), provider="openrouter"
            ):
                assert llm_runtime._LLM_LOCK.locked()
            assert llm_runtime._LLM_LOCK.locked()

        assert notifications == []
        assert not llm_runtime._LLM_LOCK.locked()

    def test_llm_generation_lock__openrouter_failure_does_not_release_the_local_lock(self) -> None:
        """Verify a failing external call never frees the lock another case holds."""
        with llm_generation_lock():
            with pytest.raises(RuntimeError):
                with llm_generation_lock(provider="openrouter"):
                    raise RuntimeError("429")
            assert llm_runtime._LLM_LOCK.locked()
        assert not llm_runtime._LLM_LOCK.locked()
