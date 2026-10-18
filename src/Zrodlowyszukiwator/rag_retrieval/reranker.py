from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from time import perf_counter
from typing import Any, TypeVar

from openai.lib._parsing import type_to_response_format_param
from pydantic import BaseModel, Field, ValidationError

from src._shared.context_budget import fit_output_tokens, input_char_budget, usage_summary
from src._shared.llm_runtime import LLMBusyError, llm_generation_lock
from src._shared.serwer import (
    COMPACT_OUTPUT_INSTRUCTION,
    REPAIR_PROMPT,
    LLMOutputLimitError,
    LLMResponseError,
    error_log_detail,
    log_model_response,
    provider_request_options,
    recover_partial_structured_response,
    structured_output_error_detail,
)

from .models import RetrievedChunk

ZRODLOWYSZUKIWACZ_RERANKER_MAX_OUTPUT_TOKENS = 4096
# Miejsce zostawiane na odpowiedź przy dopasowywaniu promptu do okna kontekstu.
RERANKER_RESERVED_OUTPUT_TOKENS = 1024
RERANKER_MAX_TEXT_CHARS = 1800
RERANKER_MIN_TEXT_CHARS = 500
# Dolna granica przedziału „fragment pomaga pośrednio” ze skali opisanej w SYSTEM_PROMPT.
# Niżej są fragmenty tylko tematycznie podobne, które niczego nie popierają.
RERANKER_MIN_SUPPORT_SCORE = 40
FALLBACK_REASON = "Wybrano przez fallback hybrid ranking."
LOGGER = logging.getLogger("pismak")


class RerankedItem(BaseModel):
    chunk_id: str
    score: int = Field(ge=0, le=100)
    reason: str


class RerankResult(BaseModel):
    selected: list[RerankedItem] = Field(default_factory=list)


@dataclass
class RerankOutcome:
    """Fragmenty wybrane przez reranker i informacja, czy model faktycznie ocenił ich trafność."""

    selected: list[tuple[RetrievedChunk, str, int]]
    # False: wybór pochodzi z fallbacku po rankingu hybrydowym, bo model nie odpowiedział
    # poprawnie. Pusta lista przy assessed=True to ocena modelu, że nic nie wspiera sprawy.
    assessed: bool


T = TypeVar("T", bound=BaseModel)


class LLMReranker:
    """
    Opcjonalny reranker. Model NIE generuje cytatów i NIE przepisuje źródeł.
    Wybiera wyłącznie chunk_id; dokładny tekst jest potem pobierany z kandydatów.
    """

    SYSTEM_PROMPT = """
Jesteś rerankerem źródeł dla Pismaka.

Otrzymujesz opis sprawy oraz kandydackie fragmenty oficjalnych dokumentów.
Wybierz fragmenty, które najbardziej bezpośrednio pomagają poprzeć:
- opis problemu,
- oczekiwany rezultat,
- konkretne requested_actions,
- argumenty interesu publicznego.

Zasady:
1. Wybieraj WYŁĄCZNIE podane chunk_id.
2. Nie twórz nowych faktów, cytatów ani źródeł.
3. Preferuj fragmenty bezpośrednio związane ze sprawą.
4. Strategia jest użyteczna dla kierunku ogólnego, ale nie premiuj jej tylko
   dlatego, że ma wyższą hierarchię. Standard/program może być bardziej konkretny.
5. Unikaj wielu niemal identycznych fragmentów, jeśli nie dodają nowej wartości.
6. reason ma być krótki i opisywać, co fragment wnosi do argumentacji.
7. score (0-100) oceniaj według skali:
   70-100 - fragment bezpośrednio odpowiada na problem, cel albo postulat,
   40-69 - fragment pomaga pośrednio, np. wyznacza ogólny kierunek polityki miasta,
   0-39 - fragment jest tylko tematycznie podobny i niczego w sprawie nie popiera.
8. Nie wybieraj fragmentów, które są tylko tematycznie podobne. Jeśli żaden fragment
   nie wspiera sprawy, zwróć pustą listę selected - to poprawna odpowiedź.
""".strip()

    def __init__(
        self,
        *,
        client: Any,
        model: str,
        max_selected: int = 6,
        on_wait: Any = None,
        session_id: str | None = None,
        allow_fallback: bool = True,
        provider: str = "ollama",
        context_tokens: int | None = None,
        min_score: int = RERANKER_MIN_SUPPORT_SCORE,
    ) -> None:
        """Initialize the optional LLM reranker, its selection limit and support threshold."""
        self.client = client
        self.model = model
        self.max_selected = max_selected
        self.min_score = min_score
        self.on_wait = on_wait
        self.session_id = session_id
        self.allow_fallback = allow_fallback
        self.provider = provider
        self.context_tokens = context_tokens
        self.last_model: str | None = None

    def _log_request_failed(
        self,
        error: Exception,
        *,
        response_model: type[BaseModel],
        attempt: int,
        retry_scheduled: bool,
        finish_reason: str | None,
        content: str | None,
        started: float,
        max_tokens: int,
    ) -> None:
        """Log why a reranking request failed, with the rejected response when there is one."""
        LOGGER.error(
            "llm_request_failed stage=reranking session_id=%s provider=%s model=%s "
            "response_model=%s max_output_tokens=%d attempt=%d retry_scheduled=%s "
            "finish_reason=%s error_type=%s error_detail=%s elapsed_seconds=%.3f",
            self.session_id or "-",
            self.provider,
            self.last_model or self.model,
            response_model.__name__,
            max_tokens,
            attempt,
            str(retry_scheduled).lower(),
            finish_reason or "-",
            type(error).__name__,
            error_log_detail(error),
            perf_counter() - started,
            extra={"session_id": self.session_id},
        )
        log_model_response(
            outcome="invalid",
            stage="reranking",
            session_id=self.session_id,
            response=content,
            provider=self.provider,
            model=self.last_model or self.model,
            attempt=attempt,
        )

    def _structured(self, user_prompt: str, response_model: type[T]) -> T:
        """Request and validate one structured reranking response from the model."""
        started = perf_counter()
        prompt = user_prompt
        content: str | None = None
        finish_reason: str | None = None
        attempt = 1
        max_tokens = ZRODLOWYSZUKIWACZ_RERANKER_MAX_OUTPUT_TOKENS
        try:
            for attempt in (1, 2):
                # Log błędu tej próby nie może pokazać odpowiedzi z poprzedniej.
                content = None
                finish_reason = None
                max_tokens = fit_output_tokens(
                    provider=self.provider,
                    prompt_chars=len(self.SYSTEM_PROMPT) + len(prompt),
                    max_output_tokens=ZRODLOWYSZUKIWACZ_RERANKER_MAX_OUTPUT_TOKENS,
                    stage="reranking",
                    session_id=self.session_id,
                    context_tokens=self.context_tokens,
                )
                with llm_generation_lock(
                    self.on_wait,
                    stage="reranking",
                    session_id=self.session_id,
                    provider=self.provider,
                ):
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=[
                            {"role": "system", "content": self.SYSTEM_PROMPT},
                            {"role": "user", "content": prompt},
                        ],
                        temperature=0.0,
                        max_tokens=max_tokens,
                        # Schemat zgodny z trybem strict. Z model_json_schema() pole selected
                        # nie było wymagane, więc pusty obiekt {} przechodził jako „nic nie
                        # wspiera sprawy”.
                        response_format=type_to_response_format_param(response_model),
                        **provider_request_options(self.provider),
                    )
                self.last_model = getattr(response, "model", None)
                choice = response.choices[0]
                content = choice.message.content
                finish_reason = getattr(choice, "finish_reason", None)
                if finish_reason == "length":
                    LOGGER.warning(
                        "llm_output_truncated stage=reranking session_id=%s provider=%s "
                        "model=%s max_output_tokens=%d attempt=%d retry_scheduled=%s",
                        self.session_id or "-",
                        self.provider,
                        self.last_model or self.model,
                        max_tokens,
                        attempt,
                        str(attempt == 1).lower(),
                        extra={"session_id": self.session_id},
                    )
                    log_model_response(
                        outcome="truncated",
                        stage="reranking",
                        session_id=self.session_id,
                        response=content,
                        provider=self.provider,
                        model=self.last_model or self.model,
                        attempt=attempt,
                    )
                    if attempt == 1:
                        prompt += "\n\n" + COMPACT_OUTPUT_INSTRUCTION
                        continue
                    partial = recover_partial_structured_response(content, response_model)
                    if partial is not None:
                        LOGGER.warning(
                            "llm_partial_response_recovered stage=reranking session_id=%s",
                            self.session_id or "-",
                            extra={"session_id": self.session_id},
                        )
                        return partial
                    raise LLMOutputLimitError(
                        "Wynik rerankera został ucięty i nie dało się go zwalidować.",
                        partial_response=content,
                    )
                try:
                    if not content:
                        raise LLMResponseError("Reranker zwrócił pustą odpowiedź")
                    result = response_model.model_validate_json(content)
                except (ValidationError, LLMResponseError) as exc:
                    detail = structured_output_error_detail(exc)
                    if attempt == 2:
                        raise LLMResponseError(
                            "Model nie zwrócił poprawnego structured output dla "
                            f"{response_model.__name__} mimo ponowienia. Ostatni błąd: {detail}"
                        ) from exc
                    self._log_request_failed(
                        exc,
                        response_model=response_model,
                        attempt=attempt,
                        retry_scheduled=True,
                        finish_reason=finish_reason,
                        content=content,
                        started=started,
                        max_tokens=max_tokens,
                    )
                    prompt += "\n\n" + REPAIR_PROMPT.format(validation_errors=detail)
                    continue
                break
            else:
                raise RuntimeError("Reranker nie zwrócił odpowiedzi.")
        except LLMOutputLimitError:
            LOGGER.warning(
                "llm_output_limit stage=reranking session_id=%s provider=%s model=%s "
                "max_output_tokens=%d elapsed_seconds=%.3f",
                self.session_id or "-",
                self.provider,
                self.last_model or self.model,
                max_tokens,
                perf_counter() - started,
                extra={"session_id": self.session_id},
            )
            log_model_response(
                outcome="invalid",
                stage="reranking",
                session_id=self.session_id,
                response=content,
                provider=self.provider,
                model=self.last_model or self.model,
                attempt=attempt,
            )
            raise
        except Exception as exc:
            self._log_request_failed(
                exc,
                response_model=response_model,
                attempt=attempt,
                retry_scheduled=False,
                finish_reason=finish_reason,
                content=content,
                started=started,
                max_tokens=max_tokens,
            )
            raise
        self.last_model = getattr(response, "model", None)
        LOGGER.info(
            "llm_request_completed stage=reranking session_id=%s provider=%s model=%s "
            "max_output_tokens=%d attempt=%d elapsed_seconds=%.3f %s",
            self.session_id or "-",
            self.provider,
            self.last_model or self.model,
            max_tokens,
            attempt,
            perf_counter() - started,
            usage_summary(response),
            extra={"session_id": self.session_id},
        )
        return result

    def _user_prompt(
        self,
        case_brief: dict,
        candidates: list[RetrievedChunk],
        text_chars: int,
    ) -> str:
        """Serialize the brief and the candidates with their text cut to the given length."""
        candidate_payload = []
        for c in candidates:
            candidate_payload.append(
                {
                    "chunk_id": c.chunk_id,
                    "title": c.title,
                    "document_kind": c.document_kind,
                    "pages": f"{c.page_start}-{c.page_end}",
                    "section_path": c.section_path,
                    # Do rerankera nie trzeba wysyłać całego 3500-znakowego chunka.
                    "text": c.text[:text_chars],
                    "hybrid_score": c.hybrid_score,
                }
            )

        return json.dumps(
            {
                "case_brief": case_brief,
                "max_selected": self.max_selected,
                "candidates": candidate_payload,
            },
            ensure_ascii=False,
            indent=2,
        )

    def _fitted_user_prompt(self, case_brief: dict, candidates: list[RetrievedChunk]) -> str:
        """Shorten candidate texts, then drop the lowest-ranked ones, until the prompt fits."""
        budget = input_char_budget(
            self.provider,
            reserved_output_tokens=RERANKER_RESERVED_OUTPUT_TOKENS,
            context_tokens=self.context_tokens,
        ) - len(self.SYSTEM_PROMPT)
        text_chars = RERANKER_MAX_TEXT_CHARS
        count = len(candidates)
        user_prompt = self._user_prompt(case_brief, candidates, text_chars)

        # Kandydaci są posortowani malejąco, więc odpadają najsłabsze trafienia.
        while len(user_prompt) > budget and (text_chars > RERANKER_MIN_TEXT_CHARS or count > 1):
            if text_chars > RERANKER_MIN_TEXT_CHARS:
                text_chars = max(RERANKER_MIN_TEXT_CHARS, int(text_chars * 0.8))
            else:
                count -= 1
            user_prompt = self._user_prompt(case_brief, candidates[:count], text_chars)

        if count < len(candidates) or text_chars < RERANKER_MAX_TEXT_CHARS:
            LOGGER.info(
                "reranker_prompt_fitted session_id=%s candidates=%d sent_candidates=%d "
                "text_chars=%d prompt_chars=%d",
                self.session_id or "-",
                len(candidates),
                count,
                text_chars,
                len(user_prompt),
                extra={"session_id": self.session_id},
            )
        return user_prompt

    def _fallback(self, candidates: list[RetrievedChunk]) -> RerankOutcome:
        """Keep the hybrid order when the model gave no usable judgement."""
        return RerankOutcome(
            selected=[(c, FALLBACK_REASON, 0) for c in candidates[: self.max_selected]],
            assessed=False,
        )

    def rerank(
        self,
        *,
        case_brief: dict,
        candidates: list[RetrievedChunk],
    ) -> RerankOutcome:
        """Select candidates the model scores as support, rejecting unknown or duplicate IDs."""
        if not candidates:
            return RerankOutcome(selected=[], assessed=True)

        user_prompt = self._fitted_user_prompt(case_brief, candidates)

        try:
            result = self._structured(user_prompt, RerankResult)
        except LLMBusyError:
            # Zajęty model to brak kolejki, nie słaba odpowiedź: bez cichego zastępstwa.
            raise
        except Exception:
            if not self.allow_fallback:
                raise
            # Retrieval ma działać nawet, jeśli mały lokalny LLM nie zwróci
            # poprawnego structured output.
            return self._fallback(candidates)

        candidate_map = {c.chunk_id: c for c in candidates}
        known = [item for item in result.selected if item.chunk_id in candidate_map]
        if result.selected and not known:
            # Same nieistniejące chunk_id to błąd odpowiedzi modelu, a nie ocena,
            # że żaden fragment nie wspiera sprawy.
            if not self.allow_fallback:
                raise RuntimeError("Reranker wskazał wyłącznie nieznane chunk_id.")
            return self._fallback(candidates)

        seen: set[str] = set()
        selected: list[tuple[RetrievedChunk, str, int]] = []

        for item in sorted(known, key=lambda x: x.score, reverse=True):
            if item.chunk_id in seen or item.score < self.min_score:
                continue
            selected.append((candidate_map[item.chunk_id], item.reason, item.score))
            seen.add(item.chunk_id)
            if len(selected) >= self.max_selected:
                break

        # Pusta lista po poprawnej odpowiedzi to ocena modelu: żaden kandydat nie wspiera
        # sprawy. Wcześniej trafiał wtedy do pisma ranking hybrydowy jako rzekome dowody.
        return RerankOutcome(selected=selected, assessed=True)
