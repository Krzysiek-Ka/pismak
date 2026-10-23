"""Tests for the Pismopoprawiator orchestration: model calls, modes, failures and logs."""

from __future__ import annotations

import json
import logging
from types import SimpleNamespace

import httpx
import pytest
from openai import APITimeoutError

from src._shared.llm_runtime import LLMBusyError
from src._shared.serwer import LLMResponseError
from src.Pismogenerator.pismogenerator_helper_functions import render_letter
from src.Pismogenerator.structures import DraftLetter
from src.Pismopoprawiator import pismopoprawiator
from src.Pismopoprawiator import pismopoprawiator_helper_functions as helpers
from src.Pismopoprawiator.constants import (
    MAX_CALLS,
    POPRAWIATOR_MAX_ATTEMPTS,
    POPRAWIATOR_MAX_OUTPUT_TOKENS,
    POPRAWIATOR_TEMPERATURE,
    POPRAWIATOR_TIMEOUT_SECONDS,
)
from src.Zrodlowyszukiwator.brief_to_evidence.models import EvidencePack, ResearchPlan

ARGUMENTATION = (
    "Mieszkańcy zgłaszają korki, ktore spowalniają autobusy linii 145.\n\n"
    "Według obserwacji wnioskodawcy przejazd trwa  ponad 20 minut [wymaga weryfikacji] [1]."
)


def make_letter(argumentation: str = ARGUMENTATION) -> str:
    draft = DraftLetter(
        recipient="Zarząd Dróg i Utrzymania Miasta",
        request_summary="wprowadzenie buspasa",
        argumentation_markdown=argumentation,
        sources=["U1"],
        claims_requiring_verification=["Czas przejazdu."],
    )
    pack = EvidencePack(
        query_summary="Buspas",
        research_plan=ResearchPlan(case_summary="Buspas", intents=[]),
    )
    return render_letter(draft, pack, ["Obserwacje wnioskodawcy"])


def fix(akapit: str, oryginal: str, poprawka: str, kategoria: str = "ortografia") -> dict:
    return {
        "akapit": akapit,
        "oryginal": oryginal,
        "poprawka": poprawka,
        "kategoria": kategoria,
        "uzasadnienie": "błąd",
    }


class FakeLLM:
    """Return scripted structured answers the way LLMClient.generate_structured does."""

    def __init__(self, *replies: object, provider: str = "ollama") -> None:
        self.replies = list(replies)
        self.calls: list[dict] = []
        self.provider = provider
        self.model_name = "model-testowy"
        self.context_tokens = 8192
        self.last_model: str | None = None

    def generate_structured(self, **kwargs):
        self.calls.append(kwargs)
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        self.last_model = "fake/odpowiedz"
        return kwargs["response_model"].model_validate(reply)


@pytest.fixture
def pismak_log(caplog):
    """Capture the application log even after another test disabled its propagation."""
    logger = logging.getLogger("pismak")
    logger.addHandler(caplog.handler)
    with caplog.at_level(logging.DEBUG, logger="pismak"):
        yield caplog
    logger.removeHandler(caplog.handler)


class TestCorrectLetter:
    def test_correct_letter__applies_valid_fixes_and_keeps_the_sources_and_control_section(
        self,
    ) -> None:
        """Verify accepted fixes reach the letter while the rest of it stays byte for byte."""
        letter = make_letter()
        llm = FakeLLM(
            {
                "poprawki": [
                    fix("A5", "korki, ktore", "korki, które"),
                    fix("A6", "trwa  ponad", "trwa ponad", "spacje"),
                    fix("A6", "20 minut", "25 minut"),
                    fix("A5", "korki", "zatory", "styl"),
                ]
            }
        )

        result = pismopoprawiator.correct_letter(letter, llm)

        assert result.status == "corrected"
        assert result.mode == "whole"
        assert result.calls == 1
        assert result.original_text == letter
        assert result.corrected_text == letter.replace("ktore", "które").replace(
            "trwa  ponad", "trwa ponad"
        )
        assert [c.oryginal for c in result.applied] == ["korki, ktore", "trwa  ponad"]
        assert [(r.correction.oryginal, r.reason) for r in result.rejected] == [
            ("20 minut", "zmienia liczby i daty: usuwa „20”; dodaje „25”")
        ]
        assert [s.poprawka for s in result.suggestions] == ["zatory"]
        assert helpers.split_letter(result.corrected_text)[1] == helpers.split_letter(letter)[1]
        assert result.model == "fake/odpowiedz"
        assert result.provider == "ollama"

    def test_correct_letter__sends_numbered_paragraphs_without_sources_or_control_section(
        self,
    ) -> None:
        """Verify the model sees the letter body only, with editable and read-only blocks marked."""
        llm = FakeLLM({"poprawki": []})

        result = pismopoprawiator.correct_letter(make_letter(), llm)

        [call] = llm.calls
        prompt = call["user_prompt"]
        assert "[A1] (tylko kontekst — nie poprawiaj)\n**Adresat:**" in prompt
        assert "[A5] (edytowalny)\nMieszkańcy zgłaszają korki" in prompt
        assert "Obserwacje wnioskodawcy" not in prompt
        assert "Sekcja kontrolna" not in prompt
        assert "Czas przejazdu." not in prompt
        assert call["max_output_tokens"] == POPRAWIATOR_MAX_OUTPUT_TOKENS
        assert call["temperature"] == POPRAWIATOR_TEMPERATURE
        assert result.status == "unchanged"
        assert result.corrected_text == result.original_text

    @pytest.mark.parametrize(
        "error",
        [
            LLMResponseError("Model dwukrotnie zwrócił niepoprawny structured output."),
            LLMBusyError("Lokalny model był zajęty."),
            APITimeoutError(request=httpx.Request("POST", "http://127.0.0.1:11434/v1/")),
            RuntimeError("nieoczekiwany błąd"),
        ],
        ids=lambda error: type(error).__name__,
    )
    def test_correct_letter__returns_the_original_letter_when_the_model_fails(
        self, error: Exception
    ) -> None:
        """Verify a model failure never stops the pipeline and leaves the letter uncorrected."""
        letter = make_letter()

        result = pismopoprawiator.correct_letter(letter, FakeLLM(error))

        assert result.status == "failed"
        assert result.corrected_text == letter
        assert result.applied == []
        assert type(error).__name__ in result.errors[0]

    def test_correct_letter__returns_the_original_letter_after_an_unexpected_bug(
        self, monkeypatch
    ) -> None:
        """Verify even an error in the stage's own code ends with the uncorrected letter."""
        monkeypatch.setattr(helpers, "split_paragraphs", lambda body: 1 / 0)
        letter = make_letter()

        result = pismopoprawiator.correct_letter(letter, FakeLLM())

        assert result.status == "failed"
        assert result.corrected_text == letter
        assert result.errors == ["ZeroDivisionError"]

    def test_correct_letter__reverts_every_fix_when_the_whole_letter_guard_fails(
        self, monkeypatch
    ) -> None:
        """Verify the whole-letter guard returns the original and explains the rejected fixes."""
        monkeypatch.setattr(helpers, "content_violations", lambda *args: ["liczba akapitów"])
        letter = make_letter()
        llm = FakeLLM({"poprawki": [fix("A5", "korki, ktore", "korki, które")]})

        result = pismopoprawiator.correct_letter(letter, llm)

        assert result.status == "reverted"
        assert result.corrected_text == letter
        assert result.applied == []
        assert [r.reason for r in result.rejected] == ["wycofano: liczba akapitów"]
        assert "strażnik całości: liczba akapitów" in result.errors

    def test_correct_letter__does_not_call_the_model_without_editable_paragraphs(self) -> None:
        """Verify a letter made only of renderer blocks is returned unchanged without a request."""
        llm = FakeLLM()

        result = pismopoprawiator.correct_letter("**Adresat:** Urząd\n\n**Nadawca:**", llm)

        assert llm.calls == []
        assert result.status == "unchanged"

    def test_correct_letter__goes_paragraph_by_paragraph_when_the_letter_is_long(self) -> None:
        """Verify a long letter is split at paragraph boundaries with a bounded number of calls."""
        paragraphs = [f"Akapit {index} " + "tekst pisma, " * 160 for index in range(1, 9)]
        letter = make_letter("\n\n".join(paragraphs))
        llm = FakeLLM(*[{"poprawki": []}] * MAX_CALLS)

        result = pismopoprawiator.correct_letter(letter, llm)

        assert result.mode == "paragraphs"
        assert result.calls == len(llm.calls) == MAX_CALLS
        assert result.status == "unchanged"
        assert "pominięto akapity A11, A12: limit 6 wywołań modelu" in result.errors
        second = llm.calls[1]
        schema = second["response_model"].model_json_schema()
        assert schema["$defs"]["Correction"]["properties"]["akapit"]["enum"] == ["A5", "A6"]
        assert "[A5] (tylko kontekst — nie poprawiaj)" in second["user_prompt"]
        assert "[A6] (edytowalny)" in second["user_prompt"]

    def test_correct_letter__rejects_a_fix_in_a_paragraph_that_was_only_context(self) -> None:
        """Verify a paragraph shown as context in one call cannot be changed by that call."""
        long_text = "tekst pisma, " * 200
        letter = make_letter(f"Pierwszy ktore {long_text}\n\nDrugi ktore {long_text}")
        # Wywołania: A3 (formuła), A5 z kontekstem A4, A6 z kontekstem A5.
        llm = FakeLLM(
            {"poprawki": []},
            {"poprawki": []},
            {"poprawki": [fix("A5", "Pierwszy ktore", "Pierwszy, które", "interpunkcja")]},
        )

        result = pismopoprawiator.correct_letter(letter, llm, mode="paragraphs")

        assert result.status == "unchanged"
        assert [r.reason for r in result.rejected] == [
            "akapit A5 nie był edytowalny w tym wywołaniu"
        ]

    def test_correct_letter__keeps_fixes_from_calls_that_succeeded(self) -> None:
        """Verify one failed call in the paragraph mode does not drop the other paragraphs' fixes."""
        long_text = "tekst pisma, " * 200
        letter = make_letter(f"Pierwszy ktore {long_text}\n\nDrugi ktore {long_text}")
        llm = FakeLLM(
            {"poprawki": []},
            RuntimeError("przerwane połączenie"),
            {"poprawki": [fix("A6", "Drugi ktore", "Drugi, które", "interpunkcja")]},
        )

        result = pismopoprawiator.correct_letter(letter, llm, mode="paragraphs")

        assert result.status == "corrected"
        assert "Drugi, które" in result.corrected_text
        assert result.errors == ["A5: RuntimeError"]

    def test_correct_letter__logs_every_fix_and_hides_the_letter_below_debug(
        self, pismak_log
    ) -> None:
        """Verify the case log explains each decision and keeps the full text at DEBUG only."""
        letter = make_letter()
        llm = FakeLLM(
            {
                "poprawki": [
                    fix("A5", "korki, ktore", "korki, które"),
                    fix("A6", "20 minut", "25 minut"),
                ]
            }
        )

        pismopoprawiator.correct_letter(letter, llm, session_id="sprawa-1")

        info = [r.getMessage() for r in pismak_log.records if r.levelno >= logging.INFO]
        debug = [r.getMessage() for r in pismak_log.records if r.levelno == logging.DEBUG]
        applied = next(line for line in info if line.startswith("correction_applied"))
        rejected = next(line for line in info if line.startswith("correction_rejected"))
        finished = next(line for line in info if line.startswith("correction_finished"))
        assert 'akapit=A5 kategoria=ortografia oryginal="korki, ktore"' in applied
        assert 'poprawka="korki, które" uzasadnienie="błąd"' in applied
        assert 'powod="zmienia liczby i daty: usuwa „20”; dodaje „25”"' in rejected
        assert "status=corrected mode=whole calls=1 applied=1 rejected=1" in finished
        assert not any(json.dumps(letter, ensure_ascii=False) in line for line in info)
        assert any(json.dumps(letter, ensure_ascii=False) in line for line in debug)


class TestCorrectionModel:
    def test_correction_model__offers_only_the_call_paragraphs_and_known_categories(
        self,
    ) -> None:
        """Verify the schema guides the model while one odd value does not reject the answer."""
        model = pismopoprawiator.correction_model(["A1", "A2"])

        schema = model.model_json_schema()["$defs"]["Correction"]["properties"]
        parsed = model.model_validate(
            {"poprawki": [fix("[a2]", "x", "y", "Literowka"), fix("7", "x", "y", "szyk")]}
        )

        assert schema["akapit"]["enum"] == ["A1", "A2"]
        assert "literówka" in schema["kategoria"]["enum"]
        assert "styl" in schema["kategoria"]["enum"]
        assert [(c.akapit, c.kategoria) for c in parsed.poprawki] == [
            ("A2", "literówka"),
            ("A7", "szyk"),
        ]


class ScriptedCompletions:
    """Answer every request with the next scripted message content."""

    def __init__(self, *contents: str) -> None:
        self.contents = list(contents)
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        content = self.contents.pop(0)
        return SimpleNamespace(
            model="fake/poprawiator",
            usage=None,
            choices=[
                SimpleNamespace(finish_reason="stop", message=SimpleNamespace(content=content))
            ],
        )


class FakeOpenAI:
    def __init__(self, completions: ScriptedCompletions) -> None:
        self.chat = SimpleNamespace(completions=completions)
        self.options: dict = {}

    def with_options(self, **kwargs):
        self.options = kwargs
        return self


class TestMakeLlm:
    def test_make_llm__sets_the_stage_timeout_without_sdk_retries(self) -> None:
        """Verify every request of this stage is limited to its own timeout."""
        client = FakeOpenAI(ScriptedCompletions())

        llm = pismopoprawiator.make_llm(
            client, "pismak-bielik", "ollama", session_id="s", context_tokens=8192
        )

        assert client.options == {"timeout": POPRAWIATOR_TIMEOUT_SECONDS, "max_retries": 0}
        assert llm.stage == "pismopoprawiator"
        assert llm.context_tokens == 8192
        assert llm.model_name == "pismak-bielik"

    def test_make_llm__tries_at_most_twice_and_then_returns_the_original_letter(
        self, monkeypatch
    ) -> None:
        """Verify an invalid answer gets one repair request and no third attempt."""
        monkeypatch.delenv("PISMAK_CONTEXT_TOKENS", raising=False)
        completions = ScriptedCompletions("to nie JSON", '{"poprawki": "nie lista"}')
        llm = pismopoprawiator.make_llm(FakeOpenAI(completions), "model", "ollama")
        letter = make_letter()

        result = pismopoprawiator.correct_letter(letter, llm)

        assert len(completions.calls) == POPRAWIATOR_MAX_ATTEMPTS
        assert result.status == "failed"
        assert result.corrected_text == letter

    def test_make_llm__applies_a_valid_answer_from_the_real_client(self, monkeypatch) -> None:
        """Verify the structured answer parsed by LLMClient reaches the letter."""
        monkeypatch.delenv("PISMAK_CONTEXT_TOKENS", raising=False)
        answer = json.dumps({"poprawki": [fix("A5", "korki, ktore", "korki, które")]})
        completions = ScriptedCompletions(answer)
        llm = pismopoprawiator.make_llm(FakeOpenAI(completions), "model", "ollama")

        result = pismopoprawiator.correct_letter(make_letter(), llm)

        assert result.status == "corrected"
        assert result.model == "fake/poprawiator"
        assert completions.calls[0]["max_tokens"] <= POPRAWIATOR_MAX_OUTPUT_TOKENS


class TestPlanCalls:
    def test_plan_calls__sends_each_paragraph_alone_with_its_predecessor_as_context(
        self,
    ) -> None:
        """Verify the paragraph-by-paragraph mode used to compare against the whole letter."""
        body, _ = helpers.split_letter(make_letter())
        _, paragraphs = helpers.split_paragraphs(body)

        calls = pismopoprawiator.plan_calls(paragraphs, "paragraphs", chunk_chars=1)

        assert [([p.id for p in in_call], editable) for in_call, editable in calls] == [
            (["A2", "A3"], ["A3"]),
            (["A4", "A5"], ["A5"]),
            (["A5", "A6"], ["A6"]),
        ]

    def test_plan_calls__groups_short_paragraphs_up_to_the_chunk_size(self) -> None:
        """Verify the fallback mode keeps the number of calls low for short paragraphs."""
        body, _ = helpers.split_letter(make_letter())
        _, paragraphs = helpers.split_paragraphs(body)

        [(in_call, editable)] = pismopoprawiator.plan_calls(paragraphs, "paragraphs")

        assert editable == ["A3", "A5", "A6"]
        assert [p.id for p in in_call] == ["A2", "A3", "A4", "A5", "A6"]
