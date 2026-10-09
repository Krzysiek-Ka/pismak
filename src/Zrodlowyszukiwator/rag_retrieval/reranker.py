from __future__ import annotations

import json
from typing import Any, TypeVar

from pydantic import BaseModel, Field

from src._shared.llm_runtime import llm_generation_lock

from .models import RetrievedChunk


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
    ) -> None:
        """Initialize the optional LLM reranker and its selection limit."""
        self.client = client
        self.model = model
        self.max_selected = max_selected
        self.on_wait = on_wait
        self.session_id = session_id

    def _structured(self, user_prompt: str, response_model: type[T]) -> T:
        """Request and validate one structured reranking response from the model."""
        schema = response_model.model_json_schema()
        with llm_generation_lock(
            self.on_wait,
            stage="reranking",
            session_id=self.session_id,
        ):
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.0,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": response_model.__name__,
                        "strict": True,
                        "schema": schema,
                    },
                },
            )
        content = response.choices[0].message.content
        if not content:
            raise RuntimeError("Reranker zwrócił pustą odpowiedź")
        return response_model.model_validate_json(content)

    def rerank(
        self,
        *,
        case_brief: dict,
        candidates: list[RetrievedChunk],
    ) -> list[tuple[RetrievedChunk, str, int]]:
        """Select and score candidate chunks while rejecting unknown or duplicate IDs."""
        if not candidates:
            return []

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
                    "text": c.text[:1800],
                    "hybrid_score": c.hybrid_score,
                }
            )

        user_prompt = json.dumps(
            {
                "case_brief": case_brief,
                "max_selected": self.max_selected,
                "candidates": candidate_payload,
            },
            ensure_ascii=False,
            indent=2,
        )

        try:
            result = self._structured(user_prompt, RerankResult)
        except Exception:
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
            return [
                (c, "Wybrano przez fallback hybrid ranking.", 0)
                for c in candidates[: self.max_selected]
            ]

        return selected
