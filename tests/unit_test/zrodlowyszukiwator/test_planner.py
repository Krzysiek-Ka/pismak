from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest

from src._shared.context_budget import LLMInputLimitError
from src._shared.serwer import LLMOutputLimitError, LLMResponseError
from src.Dopytywator.structures import CaseBrief
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


class TestFallbackPlan:
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
