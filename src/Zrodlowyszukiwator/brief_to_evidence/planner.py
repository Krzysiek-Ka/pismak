from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel

from src._shared.llm_runtime import LLMBusyError
from src._shared.serwer import LLMClient

from ..rag_retrieval.models import SearchIntent
from .models import ResearchPlan

ZRODLOWYSZUKIWACZ_PLANNER_MAX_OUTPUT_TOKENS = 4096
# Stały seed: przy temperaturze 0.1 ten sam brief dawał różne zapytania, a złoty zbiór raz
# przechodził, a raz nie (accessibility_combining_tram_stop, 2026-10-24).
ZRODLOWYSZUKIWACZ_PLANNER_SEED = 47


class LLMQueryPlanner:
    """CaseBrief -> 3-5 celowanych zapytań do miejskiego RAG."""

    # Przykłady przedmiotu postulatu celowo nie pochodzą z żadnego złotego zbioru ani E2E:
    # przykład z przypadku testowego podpowiadałby modelowi odpowiedź, którą test sprawdza.
    SYSTEM_PROMPT = """
Jesteś plannerem wyszukiwania źródeł dla Pismaka.

Na podstawie CaseBrief przygotuj 3-5 zapytań do bazy oficjalnych dokumentów
miejskich (strategie, polityki, programy, plany, standardy, zarządzenia).

Dla każdego intentu przygotuj:
- semantic_query: pełne pytanie/zdanie dobre dla wyszukiwania wektorowego,
- lexical_query: krótka kombinacja najważniejszych terminów/nazw/numerów,
- purpose: po co to zapytanie jest potrzebne,
- priority: 1-3.

W key_concepts podaj przedmiot każdego postulatu. Postulaty to kolejne requested_actions
numerowane od 1; gdy lista jest pusta, postulatem 1 jest desired_outcome. Dla każdego:
- postulate: numer postulatu,
- concepts: 1-4 krótkie frazy (1-3 słowa), bez których dokument nie popiera tego
  postulatu; podaj też sam główny rzeczownik i nazwę, jakiej użyłby dokument urzędowy.
  Przykłady: „Publiczna toaleta w parku” → ["toaleta",
  "szalet miejski"]; „Zakaz wjazdu ciężarówek na osiedle” → ["ciężarówki", "ruch ciężki",
  "pojazdy ciężarowe"]; „Usuwanie graffiti z elewacji” → ["graffiti", "pseudograffiti"].
  Nie podawaj słów tylko tematycznych („rower”, „transport”, „miasto”) ani nazw miejsc
  (ulica, osiedle).

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
        # Wywołanie, ponowienia i logi zapytania do modelu obsługuje wyłącznie LLMClient.
        self.llm = LLMClient(
            client=client,
            model_name=model,
            on_wait=on_wait,
            stage="query_planner",
            session_id=session_id,
            provider=provider,
            context_tokens=context_tokens,
        )

    @property
    def last_model(self) -> str | None:
        """Return the model that answered the last planner request."""
        return self.llm.last_model

    @staticmethod
    def _brief_dict(case_brief: Any) -> dict[str, Any]:
        """Convert a CaseBrief-like value into a dictionary for planning."""
        if isinstance(case_brief, dict):
            return case_brief
        if isinstance(case_brief, BaseModel):
            return case_brief.model_dump(mode="json")
        raise TypeError("case_brief musi być dict albo Pydantic BaseModel")

    def _llm_plan(self, brief: dict[str, Any]) -> ResearchPlan:
        """Request a structured research plan from the configured language model."""
        return self.llm.generate_structured(
            system_prompt=self.SYSTEM_PROMPT,
            user_prompt=json.dumps(brief, ensure_ascii=False, indent=2),
            response_model=ResearchPlan,
            temperature=0.1,
            max_output_tokens=ZRODLOWYSZUKIWACZ_PLANNER_MAX_OUTPUT_TOKENS,
            seed=ZRODLOWYSZUKIWACZ_PLANNER_SEED,
        )

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
