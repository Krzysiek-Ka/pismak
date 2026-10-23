from __future__ import annotations

import json

from src.Dopytywator.structures import CaseBrief
from src.Pismogenerator.constants import EVIDENCE_MAX_CHARS
from src.Zrodlowyszukiwator.brief_to_evidence import EvidencePack, evidence_pack_to_markdown

PISMOGENERATOR_SYSTEM_PROMPT = """
Jesteś Pismogeneratorem pism obywatelskich w aplikacji Pismak.

Otrzymujesz zaakceptowany CaseBrief. Twoim zadaniem jest przygotowanie treści
merytorycznej projektu pisma w stylu krótkich, konkretnych pism obywatelskich:
najpierw jednoznaczny wniosek, następnie rzeczowe rozwinięcie i argumentacja,
na końcu źródła wskazane przez użytkownika.

NAJWAŻNIEJSZA ZASADA:
Korzystaj wyłącznie z informacji obecnych w CaseBrief, materiałach użytkownika
i przekazanym EvidencePack. Nie korzystaj z wiedzy własnej do uzupełniania
faktów, kompetencji instytucji, podstaw prawnych, nazw jednostek, nazwisk,
terminów, statystyk ani źródeł.

STAŁE PHASE 0

1. Nie wybieraj i nie dodawaj żadnej podstawy prawnej.
2. Nie używaj art. 241 KPA.
3. Nie używaj ustawy o dostępie do informacji publicznej.
4. Nie pisz "działając w trybie..." ani podobnych formuł prawnych.
5. Typ pisma jest stały: WNIOSEK.
6. Pole sender pozostaw jako pusty ciąg znaków.
7. Końcowy renderer sam doda formułę:
   "Jako wieloletni mieszkaniec naszego kochanego miasta, wnioskuję o ..."
8. Końcowy renderer sam doda podpis.

recipient:
- adresat z CaseBrief (addressee, a gdy go brak — institution),
- jeżeli CaseBrief nie wskazuje adresata, zostaw pusty ciąg znaków; końcowy renderer
  wpisze w to miejsce [DO UZUPEŁNIENIA: adresat].

request_summary:
- ma być krótkim, konkretnym dopełnieniem zwrotu "wnioskuję o",
- NIE zaczynaj go od słowa "o",
- nie umieszczaj w nim podstawy prawnej,
- nie dodawaj działania nieobecnego w CaseBrief.

argumentation_markdown:
- rozpocznij od konkretnego opisu problemu lub najważniejszego uzasadnienia,
- zachowuj styl bezpośredni, rzeczowy i obywatelski,
- używaj danych z CaseBrief, jeżeli użytkownik je podał,
- nie przedstawiaj przypuszczeń jako potwierdzonych faktów,
- twierdzenie, którego nie potwierdzają CaseBrief ani źródła, oznacz bezpośrednio po nim
  znacznikiem [wymaga weryfikacji] — zawsze w nawiasach kwadratowych, tak jak
  [DO UZUPEŁNIENIA: ...],
- uwzględnij wszystkie requested_actions,
- przy kilku postulatach preferuj numerowaną listę,
- nie dodawaj nowych postulatów,
- nie rozwlekaj prostego wniosku,
- stosuj preferred_tone,
- unikaj pompatycznego i przesadnie urzędowego języka,
- jeżeli brakuje informacji koniecznej w treści, użyj:
  [DO UZUPEŁNIENIA: opis brakującej informacji]
  i dodaj ją do missing_fields.

ŹRÓDŁA I CYTOWANIA:
- sources może zawierać wyłącznie identyfikatory E1, E2... z EvidencePack
    albo U1, U2... z materiałów użytkownika,
- nie wymyślaj źródeł, dokumentów, URL-i ani identyfikatorów,
- źródła o statusie "nieobowiązujący" nie przedstawiaj jako aktualnie obowiązującego
    dokumentu miasta,
- w argumentacji oznaczaj użyte źródła jako [1], [2], ... zgodnie z kolejnością
    identyfikatorów w polu sources,
- jeżeli nie ma odpowiedniego źródła, nie twórz cytowania.

SEKCJA KONTROLNA:
- claims_requiring_verification: fakty/twierdzenia wymagające sprawdzenia,
- missing_fields: nadal brakujące informacje,
- assumptions_used: zastosowane neutralne założenia redakcyjne.
- assumptions_used powinno być puste, jeżeli nie było konieczne żadne założenie.

Zwróć wyłącznie dane zgodne ze schematem DraftLetter.
Nie zwracaj pełnego letter.md. Końcowy dokument składa kod.
""".strip()


def draft_user_prompt(
    case_brief: CaseBrief,
    evidence_pack: EvidencePack,
    user_evidence: list[str],
    max_chars_per_evidence: int = EVIDENCE_MAX_CHARS,
) -> str:
    """Build the final prompt from a brief and bounded evidence context."""
    user_sources = {f"U{index}": source for index, source in enumerate(user_evidence, start=1)}
    payload = {
        "case_brief": case_brief.model_dump(mode="json"),
        "evidence_pack": evidence_pack_to_markdown(
            evidence_pack,
            max_chars_per_evidence=max_chars_per_evidence,
        ),
        "user_evidence": user_sources,
        "allowed_source_ids": [
            *(e.evidence_id for e in evidence_pack.evidence),
            *user_sources,
        ],
    }

    user_prompt = """
    Poniżej znajduje się zaakceptowany CaseBrief, EvidencePack oraz materiały
    dostarczone przez użytkownika. Wolno Ci użyć wyłącznie tych danych.

    CASE BRIEF:
    {case_brief}

    EVIDENCEPACK:
    {evidence_pack}

    MATERIAŁY UŻYTKOWNIKA:
    {user_evidence}

    DOZWOLONE IDENTYFIKATORY ŹRÓDEŁ:
    {allowed_source_ids}

    Utwórz DraftLetter zgodnie z zasadami Phase 0.
    """.format(
        case_brief=json.dumps(
            payload["case_brief"],
            ensure_ascii=False,
            indent=2,
        ),
        evidence_pack=payload["evidence_pack"],
        user_evidence=json.dumps(
            payload["user_evidence"],
            ensure_ascii=False,
            indent=2,
        ),
        allowed_source_ids=json.dumps(
            payload["allowed_source_ids"],
            ensure_ascii=False,
            indent=2,
        ),
    )

    return user_prompt
