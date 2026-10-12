from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest

from src._shared.context_budget import LLMInputLimitError
from src._shared.serwer import LLMResponseError
from src.Zrodlowyszukiwator.brief_to_evidence.planner import (
    ZRODLOWYSZUKIWACZ_PLANNER_MAX_OUTPUT_TOKENS,
    LLMQueryPlanner,
)

VALID_PLAN = (
    '{"case_summary":"ok","intents":[{"semantic_query":"problem",'
    '"lexical_query":"problem","purpose":"research","priority":2}]}'
)
BRIEF = {
    "problem_description": "Problem",
    "desired_outcome": "Rozwiązanie",
    "requested_actions": [],
}


class _CountingClient:
    def __init__(self) -> None:
        class Completions:
            calls = 0

            def create(self, **kwargs):
                self.calls += 1
                raise RuntimeError("stop after counting the call")

        self.chat = type("Chat", (), {"completions": Completions()})()


class _ScriptedClient:
    """Return the given response contents one per call, each finished normally."""

    def __init__(self, *contents: str | None) -> None:
        class Completions:
            def __init__(self) -> None:
                self.calls: list[dict] = []

            def create(self, **kwargs):
                content = contents[len(self.calls)]
                self.calls.append(kwargs)
                message = SimpleNamespace(content=content)
                choice = SimpleNamespace(finish_reason="stop", message=message)
                return SimpleNamespace(model="vendor/model", choices=[choice])

        self.chat = SimpleNamespace(completions=Completions())


@pytest.fixture
def pismak_log(caplog):
    """Capture the application log even after another test disabled its propagation."""
    logger = logging.getLogger("pismak")
    logger.addHandler(caplog.handler)
    with caplog.at_level(logging.WARNING, logger="pismak"):
        yield caplog
    logger.removeHandler(caplog.handler)


class TestLlmQueryPlanner:
    def test_plan__retries_invalid_output_with_the_validation_errors(self) -> None:
        """Verify a response that fails the schema is repaired by one more request."""
        client = _ScriptedClient("```json\n" + VALID_PLAN + "\n```", VALID_PLAN)
        planner = LLMQueryPlanner(client=client, model="openrouter/free", allow_fallback=False)

        plan = planner.plan(BRIEF)

        calls = client.chat.completions.calls
        assert plan.case_summary == "ok"
        assert len(calls) == 2
        assert "Błędy walidacji" not in calls[0]["messages"][1]["content"]
        assert "Błędy walidacji" in calls[1]["messages"][1]["content"]
        assert "json_invalid" in calls[1]["messages"][1]["content"]

    def test_plan__retries_an_empty_response(self) -> None:
        """Verify a model that returns no content gets a second request."""
        client = _ScriptedClient(None, VALID_PLAN)
        planner = LLMQueryPlanner(client=client, model="openrouter/free", allow_fallback=False)

        assert planner.plan(BRIEF).case_summary == "ok"
        assert len(client.chat.completions.calls) == 2

    def test_plan__reports_the_schema_error_after_two_invalid_responses(self, pismak_log) -> None:
        """Verify the raised error and the log both say what the model got wrong."""
        out_of_range = VALID_PLAN.replace('"priority":2', '"priority":9')
        client = _ScriptedClient("not json", out_of_range)
        planner = LLMQueryPlanner(client=client, model="openrouter/free", allow_fallback=False)

        with pytest.raises(LLMResponseError, match="ResearchPlan") as error:
            planner.plan(BRIEF)

        assert len(client.chat.completions.calls) == 2
        assert "intents.0.priority" in str(error.value)
        failures = [r.getMessage() for r in pismak_log.records if "llm_request_failed" in r.msg]
        assert "attempt=1 retry_scheduled=true" in failures[0]
        assert "error_type=ValidationError" in failures[0]
        assert "json_invalid" in failures[0]
        assert "attempt=2 retry_scheduled=false" in failures[-1]
        assert "intents.0.priority" in failures[-1]
        assert 'response_json="not json"' in pismak_log.text
        assert '\\"priority\\":9' in pismak_log.text

    def test_plan__logs_the_provider_error_text(self, pismak_log) -> None:
        """Verify a failed request leaves the provider's own explanation in the log."""

        class Completions:
            def create(self, **kwargs):
                raise RuntimeError("Rate limit exceeded:\n free-models-per-day")

        client = SimpleNamespace(chat=SimpleNamespace(completions=Completions()))
        planner = LLMQueryPlanner(client=client, model="openrouter/free", allow_fallback=False)

        with pytest.raises(RuntimeError, match="Rate limit exceeded"):
            planner.plan(BRIEF)

        assert "error_detail=Rate limit exceeded: free-models-per-day" in pismak_log.text

    def test_plan__retries_truncated_output_with_a_compact_prompt(self) -> None:
        """Verify a token-truncated planner response is retried once in concise form."""

        class Completions:
            def __init__(self) -> None:
                self.calls = []

            def create(self, **kwargs):
                self.calls.append(kwargs)
                content = (
                    '{"partial":'
                    if len(self.calls) == 1
                    else (
                        '{"case_summary":"ok","intents":[{"semantic_query":"problem",'
                        '"lexical_query":"problem","purpose":"research","priority":2}]}'
                    )
                )
                finish_reason = "length" if len(self.calls) == 1 else "stop"
                return type(
                    "Completion",
                    (),
                    {
                        "model": "vendor/model",
                        "choices": [
                            type(
                                "Choice",
                                (),
                                {
                                    "finish_reason": finish_reason,
                                    "message": type("Message", (), {"content": content})(),
                                },
                            )()
                        ],
                    },
                )()

        class Client:
            def __init__(self) -> None:
                self.chat = type("Chat", (), {"completions": Completions()})()

        client = Client()
        planner = LLMQueryPlanner(client=client, model="openrouter/free", allow_fallback=False)

        plan = planner.plan(
            {
                "problem_description": "Problem",
                "desired_outcome": "Rozwiązanie",
                "requested_actions": [],
            }
        )

        assert plan.case_summary == "ok"
        assert len(client.chat.completions.calls) == 2
        assert "krótszą" in client.chat.completions.calls[1]["messages"][1]["content"]

    def test_plan__does_not_replace_openrouter_errors_with_local_plan(self) -> None:
        """Verify remote failures are raised instead of silently using a deterministic plan."""

        class Completions:
            calls = 0
            kwargs: dict[str, object] = {}

            def create(self, **kwargs):
                self.calls += 1
                self.kwargs = kwargs
                raise RuntimeError("remote request failed")

        class Client:
            def __init__(self) -> None:
                self.chat = type("Chat", (), {"completions": Completions()})()

        client = Client()
        planner = LLMQueryPlanner(client=client, model="openrouter/free", allow_fallback=False)

        with pytest.raises(RuntimeError, match="remote request failed"):
            planner.plan(
                {
                    "problem_description": "Problem",
                    "desired_outcome": "Rozwiązanie",
                    "requested_actions": [],
                }
            )

        assert client.chat.completions.calls == 1
        assert client.chat.completions.kwargs["max_tokens"] == (
            ZRODLOWYSZUKIWACZ_PLANNER_MAX_OUTPUT_TOKENS
        )

    def test_plan__rejects_oversized_brief_without_calling_the_remote_model(
        self, monkeypatch
    ) -> None:
        """Verify a brief beyond the window is reported instead of truncated by the provider."""
        monkeypatch.setenv("PISMAK_OPENROUTER_CONTEXT_TOKENS", "4096")
        client = _CountingClient()
        planner = LLMQueryPlanner(
            client=client,
            model="openrouter/free",
            allow_fallback=False,
            provider="openrouter",
        )

        with pytest.raises(LLMInputLimitError):
            planner.plan({"problem_description": "p" * 20_000, "requested_actions": []})

        assert client.chat.completions.calls == 0

    def test_plan__uses_deterministic_plan_for_oversized_brief_on_local_model(
        self, monkeypatch
    ) -> None:
        """Verify a local brief too long for the model still yields search intents."""
        monkeypatch.delenv("PISMAK_CONTEXT_TOKENS", raising=False)
        client = _CountingClient()
        planner = LLMQueryPlanner(client=client, model="local")

        plan = planner.plan(
            {
                "problem_description": "p" * 40_000,
                "desired_outcome": "Rozwiązanie",
                "requested_actions": [],
            }
        )

        assert client.chat.completions.calls == 0
        assert [intent.semantic_query for intent in plan.intents] == ["p" * 40_000, "Rozwiązanie"]

    def test_plan__asks_the_model_about_a_long_brief_when_its_window_is_large(
        self, monkeypatch
    ) -> None:
        """Verify the same long brief reaches a model whose own window can hold it."""
        monkeypatch.delenv("PISMAK_CONTEXT_TOKENS", raising=False)
        client = _CountingClient()
        planner = LLMQueryPlanner(
            client=client,
            model="pismak-pllum",
            allow_fallback=False,
            context_tokens=131072,
        )

        with pytest.raises(RuntimeError, match="stop after counting the call"):
            planner.plan({"problem_description": "p" * 40_000, "requested_actions": []})

        assert client.chat.completions.calls == 1
