from __future__ import annotations

import json
import logging
from time import perf_counter
from typing import Any, TypeVar

from pydantic import BaseModel, Field

from src._shared.context_budget import fit_output_tokens, input_char_budget, usage_summary
from src._shared.llm_runtime import llm_generation_lock
from src._shared.serwer import (
    COMPACT_OUTPUT_INSTRUCTION,
    LLMOutputLimitError,
    log_model_response,
    recover_partial_structured_response,
)

from .models import RetrievedChunk

ZRODLOWYSZUKIWACZ_RERANKER_MAX_OUTPUT_TOKENS = 4096
# Miejsce zostawiane na odpowiedź przy dopasowywaniu promptu do okna kontekstu.
RERANKER_RESERVED_OUTPUT_TOKENS = 1024
RERANKER_MAX_TEXT_CHARS = 1800
RERANKER_MIN_TEXT_CHARS = 500
LOGGER = logging.getLogger("pismak")


class RerankedItem(BaseModel):
    chunk_id: str
    score: int = Field(ge=0, le=100)
    reason: str


class RerankResult(BaseModel):
    selected: list[RerankedItem] = Field(default_factory=list)


T = TypeVar("T", bound=BaseModel)


class LLMReranker:
    """
    Opcjonalny reranker. Model NIE generuje cytatów i NIE przepisuje źródeł.
    Wybiera wyłącznie chunk_id; dokładny tekst jest potem pobierany z kandydatów.
    """

    SYSTEM_PROMPT = """
Jesteś rerankerem źródeł dla Pismaka.

Otrzymujesz opis sprawy oraz kandydackie fragmenty oficjalnych dokumentów.
Wybierz fragmenty, które najbezpośredniej pomagają poprzeć:
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
    ) -> None:
        """Initialize the optional LLM reranker and its selection limit."""
        self.client = client
        self.model = model
        self.max_selected = max_selected
        self.on_wait = on_wait
        self.session_id = session_id
        self.allow_fallback = allow_fallback
        self.provider = provider
        self.last_model: str | None = None

    def _structured(self, user_prompt: str, response_model: type[T]) -> T:
        """Request and validate one structured reranking response from the model."""
        schema = response_model.model_json_schema()
        started = perf_counter()
        compact_prompt = user_prompt
        content: str | None = None
        attempt = 1
        try:
            for attempt in (1, 2):
                max_tokens = fit_output_tokens(
                    provider=self.provider,
                    prompt_chars=len(self.SYSTEM_PROMPT) + len(compact_prompt),
                    max_output_tokens=ZRODLOWYSZUKIWACZ_RERANKER_MAX_OUTPUT_TOKENS,
                    stage="reranking",
                    session_id=self.session_id,
                )
                with llm_generation_lock(
                    self.on_wait,
                    stage="reranking",
                    session_id=self.session_id,
                ):
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=[
                            {"role": "system", "content": self.SYSTEM_PROMPT},
                            {"role": "user", "content": compact_prompt},
                        ],
                        temperature=0.0,
                        max_tokens=max_tokens,
                        response_format={
                            "type": "json_schema",
                            "json_schema": {
                                "name": response_model.__name__,
                                "strict": True,
                                "schema": schema,
                            },
                        },
                    )
                self.last_model = getattr(response, "model", None)
                choice = response.choices[0]
                content = choice.message.content
                if getattr(choice, "finish_reason", None) == "length":
                    LOGGER.warning(
                        "llm_output_truncated stage=reranking session_id=%s provider=%s "
                        "model=%s max_output_tokens=%d attempt=%d retry_scheduled=%s",
                        self.session_id or "-",
                        self.provider,
                        self.last_model or self.model,
                        ZRODLOWYSZUKIWACZ_RERANKER_MAX_OUTPUT_TOKENS,
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
                        compact_prompt += "\n\n" + COMPACT_OUTPUT_INSTRUCTION
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
                if not content:
                    raise RuntimeError("Reranker zwrócił pustą odpowiedź")
                result = response_model.model_validate_json(content)
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
                ZRODLOWYSZUKIWACZ_RERANKER_MAX_OUTPUT_TOKENS,
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
            LOGGER.error(
                "llm_request_failed stage=reranking session_id=%s provider=%s model=%s "
                "max_output_tokens=%d attempt=1 retry_scheduled=false error_type=%s "
                "elapsed_seconds=%.3f",
                self.session_id or "-",
                self.provider,
                self.last_model or self.model,
                ZRODLOWYSZUKIWACZ_RERANKER_MAX_OUTPUT_TOKENS,
                type(exc).__name__,
                perf_counter() - started,
                extra={"session_id": self.session_id},
            )
            raise
        self.last_model = getattr(response, "model", None)
        LOGGER.info(
            "llm_request_completed stage=reranking session_id=%s provider=%s model=%s "
            "max_output_tokens=%d attempt=1 elapsed_seconds=%.3f %s",
            self.session_id or "-",
            self.provider,
            self.last_model or self.model,
            max_tokens,
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

    def rerank(
        self,
        *,
        case_brief: dict,
        candidates: list[RetrievedChunk],
    ) -> list[tuple[RetrievedChunk, str, int]]:
        """Select and score candidate chunks while rejecting unknown or duplicate IDs."""
        if not candidates:
            return []

        user_prompt = self._fitted_user_prompt(case_brief, candidates)

        try:
            result = self._structured(user_prompt, RerankResult)
        except Exception:
            if not self.allow_fallback:
                raise
            # Retrieval ma działać nawet, jeśli mały lokalny LLM nie zwróci
            # poprawnego structured output.
            return [
                (c, "Wybrano przez fallback hybrid ranking.", 0)
                for c in candidates[: self.max_selected]
            ]

        candidate_map = {c.chunk_id: c for c in candidates}
        seen: set[str] = set()
        selected: list[tuple[RetrievedChunk, str, int]] = []

        for item in sorted(result.selected, key=lambda x: x.score, reverse=True):
            if item.chunk_id in seen or item.chunk_id not in candidate_map:
                continue
            selected.append((candidate_map[item.chunk_id], item.reason, item.score))
            seen.add(item.chunk_id)
            if len(selected) >= self.max_selected:
                break

        if not selected:
            if not self.allow_fallback:
                raise RuntimeError("Reranker nie wybrał żadnych źródeł.")
            return [
                (c, "Wybrano przez fallback hybrid ranking.", 0)
                for c in candidates[: self.max_selected]
            ]

        return selected
