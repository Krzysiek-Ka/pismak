from __future__ import annotations

import difflib
import json
import logging
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, Field

from src._shared.context_budget import input_char_budget
from src._shared.llm_runtime import LLMBusyError
from src._shared.serwer import LLMClient

from .models import RetrievedChunk
from .repository import fold_text, mentions_concept

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
    # Numer postulatu (z listy postulates), który fragment popiera.
    postulate: int = 1
    # Dosłowny cytat z fragmentu o przedmiocie postulatu; kod sprawdza go w tekście fragmentu.
    quote: str = ""
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


# Przeniesienie wyrazu z PDF („rowe- rzystów”), także z innym znakiem łącznika.
_DASHES = re.compile(r"[‐‑‒–—―]")
_LINE_BREAK_HYPHEN = re.compile(r"(?<=[^\W\d_])- (?=[a-ząćęłńóśźż])")
# Cytat liczy się jako wzięty z fragmentu, gdy tyle jego słów jest we fragmencie i gdy ma
# tyle kolejnych słów dokładnie jak w tekście (pominięty środek „…” jest wtedy dozwolony).
QUOTE_MIN_WORD_SHARE = 0.8
QUOTE_MIN_RUN = 5


def postulates(case_brief: dict) -> list[str]:
    """Return the brief's postulates: its requested actions, or the desired outcome alone."""
    actions = [str(a).strip() for a in case_brief.get("requested_actions") or [] if str(a).strip()]
    outcome = str(case_brief.get("desired_outcome") or "").strip()
    return actions or ([outcome] if outcome else [])


def _words(text: str) -> list[str]:
    """Return the words of a text without diacritics, with line-break hyphenation joined."""
    text = _DASHES.sub("-", unicodedata.normalize("NFKC", text).lower())
    text = _LINE_BREAK_HYPHEN.sub("", " ".join(text.split()))
    return re.findall(r"[^\W_]+", fold_text(text))


def quote_in_text(quote: str, text: str) -> bool:
    """
    Return whether the quote comes from the text, allowing a few miscopied words.

    Bielik przepisuje cytaty prawie dosłownie, ale psuje w nich polskie znaki
    („najskuteczniejszìyh”, „nadéczne” — złoty zbiór, 2026-10-25), więc dokładny podciąg
    odrzucał prawdziwe cytaty. Wymagane: co najmniej QUOTE_MIN_WORD_SHARE słów cytatu we
    fragmencie i ciąg QUOTE_MIN_RUN kolejnych słów (albo cały krótszy cytat) dokładnie z tekstu.
    """
    quote_words = _words(quote)
    text_words = _words(text)
    if not quote_words or not text_words:
        return False
    vocabulary = set(text_words)
    share = sum(word in vocabulary for word in quote_words) / len(quote_words)
    matcher = difflib.SequenceMatcher(None, quote_words, text_words, autojunk=False)
    run = matcher.find_longest_match(0, len(quote_words), 0, len(text_words)).size
    return share >= QUOTE_MIN_WORD_SHARE and run >= min(QUOTE_MIN_RUN, len(quote_words))


def unsupported_quote(
    item: RerankedItem, text: str, key_concepts: dict[int, list[str]]
) -> str | None:
    """Return why the item's quote is not support for its postulate, or None when it is."""
    if not quote_in_text(item.quote, text):
        return "quote_not_in_fragment"
    # Zły numer postulatu to częsty błąd małego modelu; wtedy liczy się przedmiot dowolnego.
    phrases = key_concepts.get(item.postulate) or [p for ps in key_concepts.values() for p in ps]
    if phrases and not any(mentions_concept(item.quote, phrase) for phrase in phrases):
        return "quote_without_concept"
    return None


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
9. Dla każdego wybranego fragmentu podaj postulate - numer postulatu z listy postulates,
   który fragment popiera - oraz quote: jedno zdanie z pola text tego fragmentu (najwyżej
   200 znaków), które wprost mówi o przedmiocie postulatu (key_concepts). Przepisz je
   słowo w słowo, nie streszczaj. Fragment, w którym nie ma zdania o przedmiocie
   postulatu, nie popiera go - nie wybieraj go. reason: najwyżej 12 słów.
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
        # Wywołanie, ponowienia i logi zapytania do modelu obsługuje wyłącznie LLMClient.
        self.llm = LLMClient(
            client=client,
            model_name=model,
            on_wait=on_wait,
            stage="reranking",
            session_id=session_id,
            provider=provider,
            context_tokens=context_tokens,
        )

    @property
    def last_model(self) -> str | None:
        """Return the model that answered the last reranking request."""
        return self.llm.last_model

    def _user_prompt(
        self,
        case_brief: dict,
        candidates: list[RetrievedChunk],
        text_chars: int,
        key_concepts: dict[int, list[str]],
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
                "postulates": [
                    {"number": number, "text": text, "key_concepts": key_concepts.get(number, [])}
                    for number, text in enumerate(postulates(case_brief), start=1)
                ],
                "max_selected": self.max_selected,
                "candidates": candidate_payload,
            },
            ensure_ascii=False,
            indent=2,
        )

    def _fitted_user_prompt(
        self,
        case_brief: dict,
        candidates: list[RetrievedChunk],
        key_concepts: dict[int, list[str]],
    ) -> str:
        """Shorten candidate texts, then drop the lowest-ranked ones, until the prompt fits."""
        budget = input_char_budget(
            self.provider,
            reserved_output_tokens=RERANKER_RESERVED_OUTPUT_TOKENS,
            context_tokens=self.context_tokens,
        ) - len(self.SYSTEM_PROMPT)
        text_chars = RERANKER_MAX_TEXT_CHARS
        count = len(candidates)
        user_prompt = self._user_prompt(case_brief, candidates, text_chars, key_concepts)

        # Kandydaci są posortowani malejąco, więc odpadają najsłabsze trafienia.
        while len(user_prompt) > budget and (text_chars > RERANKER_MIN_TEXT_CHARS or count > 1):
            if text_chars > RERANKER_MIN_TEXT_CHARS:
                text_chars = max(RERANKER_MIN_TEXT_CHARS, int(text_chars * 0.8))
            else:
                count -= 1
            user_prompt = self._user_prompt(
                case_brief, candidates[:count], text_chars, key_concepts
            )

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
        key_concepts: dict[int, list[str]] | None = None,
    ) -> RerankOutcome:
        """
        Select candidates the model scores as support, rejecting unknown or duplicate IDs.

        key_concepts: numer postulatu → frazy jego przedmiotu (z planu). Wsparciem jest tylko
        fragment z dosłownym cytatem, który występuje w jego tekście i wspomina przedmiot
        postulatu; bez key_concepts sprawdzany jest sam cytat.
        """
        if not candidates:
            return RerankOutcome(selected=[], assessed=True)
        key_concepts = key_concepts or {}

        user_prompt = self._fitted_user_prompt(case_brief, candidates, key_concepts)

        try:
            result = self.llm.generate_structured(
                system_prompt=self.SYSTEM_PROMPT,
                user_prompt=user_prompt,
                response_model=RerankResult,
                temperature=0.0,
                max_output_tokens=ZRODLOWYSZUKIWACZ_RERANKER_MAX_OUTPUT_TOKENS,
            )
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
        rejected: Counter[str] = Counter()

        for item in sorted(known, key=lambda x: x.score, reverse=True):
            if item.chunk_id in seen or item.score < self.min_score:
                continue
            chunk = candidate_map[item.chunk_id]
            # Ocena 0-100 nie odróżniała spraw spoza korpusu (negatywy dostawały 40-100 pkt):
            # wsparciem jest tylko sprawdzony cytat o przedmiocie postulatu (plan, punkt 4.1).
            problem = unsupported_quote(item, chunk.text, key_concepts)
            if problem:
                rejected[problem] += 1
                continue
            selected.append((chunk, f"{item.reason} Cytat: „{item.quote.strip()}”", item.score))
            seen.add(item.chunk_id)
            if len(selected) >= self.max_selected:
                break

        if rejected:
            LOGGER.info(
                "reranker_support_rejected session_id=%s kept=%d quote_not_in_fragment=%d "
                "quote_without_concept=%d",
                self.session_id or "-",
                len(selected),
                rejected["quote_not_in_fragment"],
                rejected["quote_without_concept"],
                extra={"session_id": self.session_id},
            )

        # Pusta lista po poprawnej odpowiedzi to ocena modelu: żaden kandydat nie wspiera
        # sprawy. Wcześniej trafiał wtedy do pisma ranking hybrydowy jako rzekome dowody.
        return RerankOutcome(selected=selected, assessed=True)
