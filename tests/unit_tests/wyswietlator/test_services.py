from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import pytest

from src.Dopytywator.structures import CaseBrief
from src.Pismogenerator.structures import DraftLetter
from src.Zrodlowyszukiwator.brief_to_evidence.models import EvidencePack, ResearchPlan

PROJECT_ROOT = Path(__file__).resolve().parents[3]
WYSWIETLATOR_DIR = PROJECT_ROOT / "src" / "Wyswietlator"
if str(WYSWIETLATOR_DIR) not in sys.path:
    sys.path.insert(0, str(WYSWIETLATOR_DIR))

services = importlib.import_module("services")


def make_brief(**overrides: object) -> CaseBrief:
    values: dict[str, object] = {
        "title": "Sprawa",
        "problem_description": "Problem z dostępem.",
        "desired_outcome": "Przywrócenie dostępu.",
        "requested_actions": ["Sprawdzenie sprawy"],
    }
    values.update(overrides)
    return CaseBrief(**values)


def make_empty_pack() -> EvidencePack:
    return EvidencePack(
        query_summary="Brak wyników",
        research_plan=ResearchPlan(case_summary="Brak wyników", intents=[]),
    )


class TestListGenerationModels:
    def test_list_generation_models__filters_embedding_models_case_insensitively(self) -> None:
        """Verify embedding-only models cannot be selected for generation."""

        class Models:
            data = [
                type("Model", (), {"id": "Bielik"})(),
                type("Model", (), {"id": "BGE-M3"})(),
                type("Model", (), {"id": "nomic-embed-text"})(),
            ]

        class Client:
            class ModelsApi:
                def list(self):
                    return Models()

            models = ModelsApi()

        assert services.list_generation_models(Client()) == ["Bielik"]


class TestBuildZrodlowyszukiwator:
    def test_build_zrodlowyszukiwator__fails_before_connecting_without_database_url(
        self, monkeypatch
    ) -> None:
        """Verify missing database configuration produces a clear setup error."""
        monkeypatch.delenv("DATABASE_URL", raising=False)

        try:
            services.build_zrodlowyszukiwator(
                client=object(),
                model="test",
            )
        except RuntimeError as error:
            assert str(error) == "Brak DATABASE_URL w .env."
        else:
            raise AssertionError("Expected missing DATABASE_URL error")


class TestRenderLetter:
    def test_render_letter__normalizes_blank_user_evidence_before_resolving_ids(self) -> None:
        """Verify blank user materials do not shift U-source identifiers in the rendered letter."""
        draft = DraftLetter(
            recipient="Urząd",
            request_summary="sprawdzenie sprawy",
            argumentation_markdown="A" * 120,
            sources=["U1"],
        )

        result = services.render_letter(
            draft,
            make_empty_pack(),
            ["  ", "Zdjęcie"],
        )

        assert "U1: Zdjęcie" in result


class TestSaveRun:
    def test_save_run__writes_only_provided_optional_artifacts(self, tmp_path: Path) -> None:
        """Verify a partial UI run saves its conversation without creating absent artifacts."""
        run_dir = services.save_run(
            project_root=tmp_path,
            messages=[{"role": "user", "content": "Opis"}],
            run_dir=tmp_path / "run",
        )

        assert run_dir == tmp_path / "run"
        assert json.loads((run_dir / "conversation.json").read_text(encoding="utf-8")) == [
            {"role": "user", "content": "Opis"}
        ]
        assert not (run_dir / "brief.json").exists()
        assert not (run_dir / "letter.md").exists()

    def test_save_run__creates_unique_directories_for_runs_in_same_second(
        self, tmp_path: Path
    ) -> None:
        """Verify simultaneous sessions cannot write artifacts into one output directory."""
        first = services.save_run(project_root=tmp_path, messages=[])
        second = services.save_run(project_root=tmp_path, messages=[])

        assert first != second
        assert first.parent == second.parent == tmp_path / "outputs"


class TestValidateUserMessage:
    def test_validate_user_message__accepts_message_at_limit(self) -> None:
        """Verify the configured maximum remains usable for legitimate long descriptions."""
        message = "x" * services.MAX_USER_MESSAGE_CHARS

        assert services.validate_user_message(message) is None

    def test_validate_user_message__rejects_message_over_limit(self) -> None:
        """Verify oversized user text is rejected before it enters session history."""
        message = "x" * (services.MAX_USER_MESSAGE_CHARS + 1)

        with pytest.raises(ValueError, match="Wiadomość jest zbyt długa"):
            services.validate_user_message(message)
