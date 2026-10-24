"""Tests for the LLM query planner."""

from __future__ import annotations

import logging
from threading import Event, Thread
from types import SimpleNamespace

import pytest

from src._shared import llm_runtime
from src._shared.context_budget import LLMInputLimitError
from src._shared.llm_runtime import LLMBusyError, llm_generation_lock
from src._shared.serwer import LLMOutputLimitError, LLMResponseError
from src.Dopytywator.structures import CaseBrief
from src.Zrodlowyszukiwator.brief_to_evidence.planner import (
    ZRODLOWYSZUKIWACZ_PLANNER_MAX_OUTPUT_TOKENS,
    LLMQueryPlanner,
)


class EmptyPlanClient:
    class Chat:
        class Completions:
            def create(self, **kwargs):
                class Message:
                    content = '{"case_summary": "", "intents": []}'

                class Choice:
                    message = Message()

                class Response:
                    choices = [Choice()]

                return Response()

        completions = Completions()

    chat = Chat()


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


class _FinishingClient:
    """Return (content, finish_reason) pairs one per call."""

    def __init__(self, *replies: tuple[str | None, str]) -> None:
        class Completions:
            def __init__(self) -> None:
                self.calls: list[dict] = []

            def create(self, **kwargs):
                content, finish_reason = replies[len(self.calls)]
                self.calls.append(kwargs)
                message = SimpleNamespace(content=content)
                choice = SimpleNamespace(finish_reason=finish_reason, message=message)
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


class TestLLMQueryPlannerPlan:
    def test_plan__local_model_busy_waits_for_the_shared_lock(self) -> None:
        """Verify planner requests queue behind another local generation."""
        notified = Event()
        called = Event()

        class Client:
            class Chat:
                class Completions:
                    def create(self, **kwargs):
                        called.set()

                        class Message:
                            content = '{"case_summary": "Sprawa", "intents": []}'

                        class Choice:
                            message = Message()

                        class Response:
                            choices = [Choice()]

                        return Response()

                completions = Completions()

            chat = Chat()

        planner = LLMQueryPlanner(client=Client(), model="test", on_wait=notified.set)
        thread = Thread(target=lambda: planner.plan({"problem_description": "Problem"}))

        with llm_generation_lock():
            thread.start()
            assert notified.wait(timeout=1)
            assert not called.is_set()

        thread.join(timeout=1)

        assert not thread.is_alive()
        assert called.is_set()

    def test_plan__busy_local_model_is_reported_instead_of_using_the_fallback(
        self, monkeypatch
    ) -> None:
        """Verify a case that waited too long is told to retry, not given a fallback plan."""
        monkeypatch.setattr(llm_runtime, "LLM_LOCK_WAIT_TIMEOUT_SECONDS", 0.05)
        planner = LLMQueryPlanner(client=_CountingClient(), model="test")

        with llm_generation_lock():
            with pytest.raises(LLMBusyError):
                planner.plan({"problem_description": "Brak przejścia"})

    def test_plan__openrouter_does_not_wait_for_the_local_model(self) -> None:
        """Verify an OpenRouter planner runs at once while Ollama serves another case."""
        notified = Event()
        planner = LLMQueryPlanner(
            client=EmptyPlanClient(),
            model="test",
            on_wait=notified.set,
            provider="openrouter",
            allow_fallback=True,
        )

        with llm_generation_lock():
            result = planner.plan({"problem_description": "Brak przejścia"})

        assert not notified.is_set()
        assert result.intents

    def test_plan__plan_without_intents_returns_the_fallback_plan(self) -> None:
        """Verify an empty structured response does not erase useful fallback queries."""
        planner = LLMQueryPlanner(client=EmptyPlanClient(), model="test")

        result = planner.plan({"problem_description": "Problem", "requested_actions": []})

        assert len(result.intents) == 1
        assert result.intents[0].semantic_query == "Problem"

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

    def test_plan__returns_a_valid_prefix_when_the_compact_retry_is_truncated_too(
        self, pismak_log
    ) -> None:
        """Verify a complete plan cut off only after its last intent is still used."""
        truncated = VALID_PLAN[:-1] + ',"extra":'
        client = _FinishingClient((truncated, "length"), (truncated, "length"))
        planner = LLMQueryPlanner(client=client, model="m", allow_fallback=False)

        plan = planner.plan(BRIEF)

        assert plan.case_summary == "ok"
        assert len(client.chat.completions.calls) == 2
        assert "llm_partial_response_recovered stage=query_planner" in pismak_log.text

    def test_plan__raises_an_output_limit_error_when_a_truncated_plan_cannot_be_used(
        self, pismak_log
    ) -> None:
        """Verify an unusable cut-off plan on OpenRouter fails clearly instead of guessing."""
        client = _FinishingClient(('{"case_summary": "o', "length"), ('{"intents": [', "length"))
        planner = LLMQueryPlanner(client=client, model="m", allow_fallback=False)

        with pytest.raises(LLMOutputLimitError, match="ucięty"):
            planner.plan(BRIEF)

        assert "llm_output_limit stage=query_planner" in pismak_log.text

    def test_plan__local_model_falls_back_after_an_unusable_truncated_plan(self) -> None:
        """Verify the local model still gets a deterministic search plan after truncation."""
        client = _FinishingClient(('{"case_summary": "o', "length"), ('{"intents": [', "length"))
        planner = LLMQueryPlanner(client=client, model="m", allow_fallback=True)

        plan = planner.plan(BRIEF)

        assert [intent.semantic_query for intent in plan.intents] == ["Problem", "Rozwiązanie"]

    def test_plan__rejects_a_plan_without_intents_when_fallback_is_disabled(self) -> None:
        """Verify an empty OpenRouter plan is an error, not a search with no queries."""
        client = _ScriptedClient('{"case_summary": "ok", "intents": []}')
        planner = LLMQueryPlanner(client=client, model="m", allow_fallback=False)

        with pytest.raises(RuntimeError, match="Planner zwrócił plan bez zapytań."):
            planner.plan(BRIEF)

    def test_plan__accepts_a_case_brief_model(self) -> None:
        """Verify the UI can pass its CaseBrief object directly to the planner."""
        client = _ScriptedClient(VALID_PLAN)
        planner = LLMQueryPlanner(client=client, model="m", allow_fallback=False)

        planner.plan(
            CaseBrief(
                title="Sprawa",
                problem_description="Problem",
                desired_outcome="Rozwiązanie",
                requested_actions=["Działanie"],
            )
        )

        sent = client.chat.completions.calls[0]["messages"][1]["content"]
        assert '"title": "Sprawa"' in sent
        assert '"requested_actions": [\n    "Działanie"\n  ]' in sent

    def test_plan__rejects_a_brief_of_an_unsupported_type_even_with_fallback(self) -> None:
        """Verify a wrong argument is a programming error and not hidden by the fallback."""
        planner = LLMQueryPlanner(client=_CountingClient(), model="m", allow_fallback=True)

        with pytest.raises(TypeError, match="case_brief musi być dict"):
            planner.plan("Brak przejścia")

    def test_plan__remembers_the_model_that_answered(self) -> None:
        """Verify the run records the model OpenRouter actually routed the request to."""
        planner = LLMQueryPlanner(client=_ScriptedClient(VALID_PLAN), model="openrouter/free")

        planner.plan(BRIEF)

        assert planner.last_model == "vendor/model"

    def test_plan__request_schema_satisfies_strict_mode(self) -> None:
        """Verify every field is required and extra fields are forbidden, as strict mode needs."""
        client = _ScriptedClient(VALID_PLAN)
        planner = LLMQueryPlanner(client=client, model="m", allow_fallback=False)

        planner.plan(BRIEF)

        json_schema = client.chat.completions.calls[0]["response_format"]["json_schema"]
        assert json_schema["name"] == "ResearchPlan"
        assert json_schema["strict"] is True
        assert json_schema["schema"]["additionalProperties"] is False
        assert set(json_schema["schema"]["required"]) == {"case_summary", "intents"}

    def test_plan__openrouter_request_requires_providers_that_honour_the_schema(self) -> None:
        """Verify OpenRouter routes only to providers supporting response_format; Ollama gets none."""
        openrouter = _ScriptedClient(VALID_PLAN)
        ollama = _ScriptedClient(VALID_PLAN)

        LLMQueryPlanner(client=openrouter, model="m", provider="openrouter").plan(BRIEF)
        LLMQueryPlanner(client=ollama, model="m", provider="ollama").plan(BRIEF)

        assert openrouter.chat.completions.calls[0]["extra_body"] == {
            "provider": {"require_parameters": True}
        }
        assert "extra_body" not in ollama.chat.completions.calls[0]

    def test_plan__truncation_log_shows_the_fitted_output_limit(
        self, pismak_log, monkeypatch
    ) -> None:
        """Verify a truncation log shows the fitted limit, not the 4096 upper bound."""
        monkeypatch.delenv("PISMAK_CONTEXT_TOKENS", raising=False)
        client = _FinishingClient(('{"case_summary": "uci', "length"), ('{"case_summ', "length"))
        planner = LLMQueryPlanner(
            client=client, model="m", allow_fallback=False, context_tokens=2048
        )

        with pytest.raises(LLMOutputLimitError):
            planner.plan(BRIEF)

        first, second = (call["max_tokens"] for call in client.chat.completions.calls)
        assert first < ZRODLOWYSZUKIWACZ_PLANNER_MAX_OUTPUT_TOKENS
        messages = [record.getMessage() for record in pismak_log.records]
        attempt_1 = next(m for m in messages if "llm_output_truncated" in m and "attempt=1" in m)
        final = next(m for m in messages if m.startswith("llm_output_limit"))
        assert f"max_output_tokens={first} " in attempt_1
        assert f"max_output_tokens={second} " in final

    def test_plan__every_attempt_uses_the_module_output_token_limit(self) -> None:
        """Verify the repair and the compact retry are capped like the first request."""
        invalid = _ScriptedClient("nie JSON", VALID_PLAN)
        truncated = _FinishingClient(('{"case_summary": "o', "length"), (VALID_PLAN, "stop"))

        for client in (invalid, truncated):
            LLMQueryPlanner(
                client=client, model="m", provider="openrouter", allow_fallback=False
            ).plan(BRIEF)

        calls = invalid.chat.completions.calls + truncated.chat.completions.calls
        assert [call["max_tokens"] for call in calls] == [
            ZRODLOWYSZUKIWACZ_PLANNER_MAX_OUTPUT_TOKENS
        ] * 4

    def test_plan__openrouter_timeout_or_provider_error_is_raised_without_a_retry(
        self, provider_error: Exception
    ) -> None:
        """Verify a remote failure is reported after one request, without a fallback plan."""

        class Completions:
            calls = 0

            def create(self, **kwargs):
                self.calls += 1
                raise provider_error

        client = SimpleNamespace(chat=SimpleNamespace(completions=Completions()))
        planner = LLMQueryPlanner(
            client=client, model="openrouter/free", provider="openrouter", allow_fallback=False
        )

        with pytest.raises(type(provider_error)):
            planner.plan(BRIEF)

        assert client.chat.completions.calls == 1

    def test_plan__local_timeout_or_provider_error_returns_the_fallback_plan(
        self, provider_error: Exception
    ) -> None:
        """Verify a slow or failing local model still leaves the case with search intents."""

        class Completions:
            def create(self, **kwargs):
                raise provider_error

        client = SimpleNamespace(chat=SimpleNamespace(completions=Completions()))

        plan = LLMQueryPlanner(client=client, model="local").plan(BRIEF)

        assert [intent.semantic_query for intent in plan.intents] == ["Problem", "Rozwiązanie"]

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "BŁĄD: odpowiedź HTTP 200 z obiektem error (OpenRouter) ma choices=None; "
            "_llm_plan robi choices[0] i rzuca TypeError zamiast LLMResponseError"
        ),
    )
    def test_plan__openrouter_answer_without_choices_raises_a_readable_error(self) -> None:
        """Verify a provider error sent with status 200 ends in the module's readable error."""

        class Completions:
            def create(self, **kwargs):
                return SimpleNamespace(model="m", choices=None)

        client = SimpleNamespace(chat=SimpleNamespace(completions=Completions()))
        planner = LLMQueryPlanner(
            client=client, model="openrouter/free", provider="openrouter", allow_fallback=False
        )

        with pytest.raises(LLMResponseError):
            planner.plan(BRIEF)


class TestLLMQueryPlannerFallbackPlan:
    def test_fallback_plan__searches_problem_goal_and_up_to_three_actions(self) -> None:
        """Verify the deterministic plan covers the whole brief within five intents."""
        brief = {
            "problem_description": "Brak przejścia",
            "desired_outcome": "Nowe przejście",
            "requested_actions": ["A1", " ", "A2", "A3", "A4"],
        }

        plan = LLMQueryPlanner._fallback_plan(brief)

        assert [i.semantic_query for i in plan.intents] == [
            "Brak przejścia",
            "Nowe przejście",
            "A1",
            "A2",
            "A3",
        ]
        assert [i.priority for i in plan.intents] == [3, 2, 3, 3, 3]
        assert plan.case_summary == "Nowe przejście"

    def test_fallback_plan__shortens_only_the_full_text_query(self) -> None:
        """Verify a long description keeps its full meaning for vectors but not for tsquery."""
        problem = "słowo " * 100

        [intent] = LLMQueryPlanner._fallback_plan({"problem_description": problem}).intents

        assert intent.semantic_query == problem.strip()
        assert len(intent.lexical_query) == 220

    def test_fallback_plan__summarizes_by_the_problem_when_there_is_no_goal(self) -> None:
        """Verify the pack summary is never empty when the brief has only a problem."""
        plan = LLMQueryPlanner._fallback_plan({"problem_description": "Brak przejścia"})

        assert plan.case_summary == "Brak przejścia"

    def test_fallback_plan__returns_no_intents_for_an_empty_brief(self) -> None:
        """Verify an empty brief leads to 'nothing found' instead of a search for blank text."""
        plan = LLMQueryPlanner._fallback_plan(
            {"problem_description": "  ", "desired_outcome": None, "requested_actions": None}
        )

        assert plan.intents == []
        assert plan.case_summary == "Sprawa z CaseBrief"
