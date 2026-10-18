from __future__ import annotations

import json
import logging
from time import perf_counter
from typing import Any

from openai.lib._parsing import type_to_response_format_param
from pydantic import BaseModel, ValidationError

from src._shared.context_budget import fit_output_tokens, usage_summary
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

from ..rag_retrieval.models import SearchIntent
from .models import ResearchPlan

ZRODLOWYSZUKIWACZ_PLANNER_MAX_OUTPUT_TOKENS = 4096
LOGGER = logging.getLogger("pismak")


class LLMQueryPlanner:
    """CaseBrief -> 3-5 celowanych zapytań do miejskiego RAG."""

    SYSTEM_PROMPT = """
Jesteś plannerem wyszukiwania źródeł dla Pismaka.

Na podstawie CaseBrief przygotuj 3-5 zapytań do bazy oficjalnych dokumentów
miejskich (strategie, polityki, programy, plany, standardy, zarządzenia).

Dla każdego intentu przygotuj:
- semantic_query: pełne pytanie/zdanie dobre dla wyszukiwania wektorowego,
- lexical_query: krótka kombinacja najważniejszych terminów/nazw/numerów,
- purpose: po co to zapytanie jest potrzebne,
- priority: 1-3.

Zasady:
1. Nie odpowiadaj na sprawę i nie twórz argumentacji.
2. Nie dodawaj faktów nieobecnych w CaseBrief.
3. Szukaj osobno źródeł dla problemu, celu i konkretnych requested_actions.
4. Jeśli CaseBrief zawiera dokładną nazwę ulicy, dokumentu, numer lub termin
   techniczny, zachowaj go w lexical_query.
5. Nie filtruj się wyłącznie do strategii wysokiego poziomu; szukaj też
   konkretnych standardów, programów i zasad wykonawczych.
""".strip()

    def __init__(
        self,
        *,
        client: Any,
        model: str,
        on_wait: Any = None,
        session_id: str | None = None,
        allow_fallback: bool = True,
        provider: str = "ollama",
        context_tokens: int | None = None,
    ) -> None:
        """Initialize the query planner with an LLM client and model identifier."""
        self.client = client
        self.model = model
        self.on_wait = on_wait
        self.session_id = session_id
        self.allow_fallback = allow_fallback
        self.provider = provider
        self.context_tokens = context_tokens
        self.last_model: str | None = None

    @staticmethod
    def _brief_dict(case_brief: Any) -> dict[str, Any]:
        """Convert a CaseBrief-like value into a dictionary for planning."""
        if isinstance(case_brief, dict):
            return case_brief
        if isinstance(case_brief, BaseModel):
            return case_brief.model_dump(mode="json")
        raise TypeError("case_brief musi być dict albo Pydantic BaseModel")

    def _log_request_failed(
        self,
        error: Exception,
        *,
        attempt: int,
        retry_scheduled: bool,
        finish_reason: str | None,
        content: str | None,
        started: float,
    ) -> None:
        """Log why a planner request failed, with the rejected response when there is one."""
        LOGGER.error(
            "llm_request_failed stage=query_planner session_id=%s provider=%s model=%s "
            "response_model=ResearchPlan max_output_tokens=%d attempt=%d retry_scheduled=%s "
            "finish_reason=%s error_type=%s error_detail=%s elapsed_seconds=%.3f",
            self.session_id or "-",
            self.provider,
            self.last_model or self.model,
            ZRODLOWYSZUKIWACZ_PLANNER_MAX_OUTPUT_TOKENS,
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
            stage="query_planner",
            session_id=self.session_id,
            response=content,
            provider=self.provider,
            model=self.last_model or self.model,
            attempt=attempt,
        )

    def _llm_plan(self, brief: dict[str, Any]) -> ResearchPlan:
        """Request a structured research plan from the configured language model."""
        started = perf_counter()
        user_prompt = json.dumps(brief, ensure_ascii=False, indent=2)
        content: str | None = None
        finish_reason: str | None = None
        attempt = 1
        try:
            for attempt in (1, 2):
                # Log błędu tej próby nie może pokazać odpowiedzi z poprzedniej.
                content = None
                finish_reason = None
                max_tokens = fit_output_tokens(
                    provider=self.provider,
                    prompt_chars=len(self.SYSTEM_PROMPT) + len(user_prompt),
                    max_output_tokens=ZRODLOWYSZUKIWACZ_PLANNER_MAX_OUTPUT_TOKENS,
                    stage="query_planner",
                    session_id=self.session_id,
                    context_tokens=self.context_tokens,
                )
                with llm_generation_lock(
                    self.on_wait,
                    stage="query_planner",
                    session_id=self.session_id,
                    provider=self.provider,
                ):
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=[
                            {"role": "system", "content": self.SYSTEM_PROMPT},
                            {"role": "user", "content": user_prompt},
                        ],
                        temperature=0.1,
                        max_tokens=max_tokens,
                        # Schemat zgodny z trybem strict: wszystkie pola wymagane i bez
                        # dodatkowych pól (model_json_schema() tego nie zapewnia).
                        response_format=type_to_response_format_param(ResearchPlan),
                        **provider_request_options(self.provider),
                    )
                self.last_model = getattr(response, "model", None)
                choice = response.choices[0]
                content = choice.message.content
                finish_reason = getattr(choice, "finish_reason", None)
                if finish_reason == "length":
                    LOGGER.warning(
                        "llm_output_truncated stage=query_planner session_id=%s provider=%s "
                        "model=%s max_output_tokens=%d attempt=%d retry_scheduled=%s",
                        self.session_id or "-",
                        self.provider,
                        self.last_model or self.model,
                        ZRODLOWYSZUKIWACZ_PLANNER_MAX_OUTPUT_TOKENS,
                        attempt,
                        str(attempt == 1).lower(),
                        extra={"session_id": self.session_id},
                    )
                    log_model_response(
                        outcome="truncated",
                        stage="query_planner",
                        session_id=self.session_id,
                        response=content,
                        provider=self.provider,
                        model=self.last_model or self.model,
                        attempt=attempt,
                    )
                    if attempt == 1:
                        user_prompt += "\n\n" + COMPACT_OUTPUT_INSTRUCTION
                        continue
                    partial = recover_partial_structured_response(content, ResearchPlan)
                    if partial is not None:
                        LOGGER.warning(
                            "llm_partial_response_recovered stage=query_planner session_id=%s",
                            self.session_id or "-",
                            extra={"session_id": self.session_id},
                        )
                        return partial
                    raise LLMOutputLimitError(
                        "Plan wyszukiwania został ucięty i nie dało się go zwalidować.",
                        partial_response=content,
                    )
                try:
                    if not content:
                        raise LLMResponseError("Planner zwrócił pustą odpowiedź")
                    plan = ResearchPlan.model_validate_json(content)
                except (ValidationError, LLMResponseError) as exc:
                    detail = structured_output_error_detail(exc)
                    if attempt == 2:
                        raise LLMResponseError(
                            "Model nie zwrócił poprawnego structured output dla ResearchPlan "
                            f"mimo ponowienia. Ostatni błąd: {detail}"
                        ) from exc
                    self._log_request_failed(
                        exc,
                        attempt=attempt,
                        retry_scheduled=True,
                        finish_reason=finish_reason,
                        content=content,
                        started=started,
                    )
                    user_prompt += "\n\n" + REPAIR_PROMPT.format(validation_errors=detail)
                    continue
                break
            else:
                raise RuntimeError("Planner nie zwrócił odpowiedzi.")
        except LLMOutputLimitError:
            LOGGER.warning(
                "llm_output_limit stage=query_planner session_id=%s provider=%s model=%s "
                "max_output_tokens=%d elapsed_seconds=%.3f",
                self.session_id or "-",
                self.provider,
                self.last_model or self.model,
                ZRODLOWYSZUKIWACZ_PLANNER_MAX_OUTPUT_TOKENS,
                perf_counter() - started,
                extra={"session_id": self.session_id},
            )
            log_model_response(
                outcome="invalid",
                stage="query_planner",
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
                attempt=attempt,
                retry_scheduled=False,
                finish_reason=finish_reason,
                content=content,
                started=started,
            )
            raise
        self.last_model = getattr(response, "model", None)
        LOGGER.info(
            "llm_request_completed stage=query_planner session_id=%s provider=%s model=%s "
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
        return plan

    @staticmethod
    def _fallback_plan(brief: dict[str, Any]) -> ResearchPlan:
        """Build a deterministic research plan from the brief when model planning is unavailable."""
        problem = str(brief.get("problem_description") or "").strip()
        goal = str(brief.get("desired_outcome") or "").strip()
        actions = [str(x).strip() for x in (brief.get("requested_actions") or []) if str(x).strip()]

        intents: list[SearchIntent] = []
        if problem:
            intents.append(
                SearchIntent(
                    semantic_query=problem,
                    lexical_query=problem[:220],
                    purpose="Źródła opisujące problem i właściwy kierunek polityki miasta.",
                    priority=3,
                )
            )
        if goal:
            intents.append(
                SearchIntent(
                    semantic_query=goal,
                    lexical_query=goal[:220],
                    purpose="Źródła wspierające oczekiwany rezultat.",
                    priority=2,
                )
            )
        for action in actions[:3]:
            intents.append(
                SearchIntent(
                    semantic_query=action,
                    lexical_query=action[:220],
                    purpose="Źródła odnoszące się do konkretnego postulatu.",
                    priority=3,
                )
            )

        return ResearchPlan(
            case_summary=goal or problem or "Sprawa z CaseBrief",
            intents=intents[:5],
        )

    def plan(self, case_brief: Any) -> ResearchPlan:
        """Return an LLM research plan or a deterministic fallback plan."""
        brief = self._brief_dict(case_brief)
        try:
            plan = self._llm_plan(brief)
            if plan.intents:
                return plan
            if not self.allow_fallback:
                raise RuntimeError("Planner zwrócił plan bez zapytań.")
        except LLMBusyError:
            # Zajęty model to brak kolejki, nie słaba odpowiedź: użytkownik ma spróbować
            # ponownie, a nie dostać po długim czekaniu pismo z planu zapasowego.
            raise
        except Exception:
            if not self.allow_fallback:
                raise
        return self._fallback_plan(brief)
