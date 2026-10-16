"""Tests for the shared lock that serializes LLM generations."""

from __future__ import annotations

import logging
from threading import Event, Thread
from types import SimpleNamespace

import pytest
from openai import LengthFinishReasonError

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

    def test_llm_generation_lock__logs_truncated_output_as_a_warning_and_reraises(
        self, records
    ) -> None:
        """Verify an output-limit stop is logged as truncation, not as a generic failure."""
        error = LengthFinishReasonError(completion=SimpleNamespace(usage=None))

        with pytest.raises(LengthFinishReasonError):
            with llm_generation_lock(stage="draft", session_id="case-1"):
                raise error

        messages = _messages(records)
        assert any(
            message.startswith("llm_output_truncated stage=draft session_id=case-1")
            for message in messages
        )
        assert not any(message.startswith("llm_failed") for message in messages)
        warning = next(r for r in records if r.getMessage().startswith("llm_output_truncated"))
        assert warning.levelno == logging.WARNING

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
