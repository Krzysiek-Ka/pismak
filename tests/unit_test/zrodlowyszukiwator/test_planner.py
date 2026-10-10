from __future__ import annotations

import pytest

from src.Zrodlowyszukiwator.brief_to_evidence.planner import (
    ZRODLOWYSZUKIWACZ_PLANNER_MAX_OUTPUT_TOKENS,
    LLMQueryPlanner,
)


class TestLlmQueryPlanner:
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
