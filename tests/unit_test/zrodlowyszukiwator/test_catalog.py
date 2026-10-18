"""Tests for loading the JSON catalog of documents prepared for ingestion."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from src.Zrodlowyszukiwator.rag_ingestion.catalog import load_catalog
from src.Zrodlowyszukiwator.rag_ingestion.models import DocumentKind, LifecycleStatus


def entry(**overrides: object) -> dict[str, object]:
    values: dict[str, object] = {
        "filename": "polityka.pdf",
        "document_id": "polityka",
        "title": "Wrocławska polityka mobilności",
        "document_kind": "policy",
    }
    values.update(overrides)
    return values


def write_catalog(tmp_path: Path, *documents: dict[str, object]) -> Path:
    path = tmp_path / "document_catalog.json"
    path.write_text(
        json.dumps({"documents": list(documents)}, ensure_ascii=False), encoding="utf-8"
    )
    return path


class TestLoadCatalog:
    def test_load_catalog__keys_entries_by_file_name_without_keeping_it_in_the_metadata(
        self, tmp_path: Path
    ) -> None:
        """Verify ingestion finds a PDF by its name and the name is not a metadata field."""
        path = write_catalog(
            tmp_path, entry(), entry(filename="strategia.pdf", document_id="strategia")
        )

        catalog = load_catalog(path)

        assert list(catalog) == ["polityka.pdf", "strategia.pdf"]
        assert catalog["polityka.pdf"].document_id == "polityka"
        assert catalog["polityka.pdf"].document_kind is DocumentKind.POLICY
        assert "filename" not in catalog["polityka.pdf"].model_dump()

    def test_load_catalog__accepts_a_path_given_as_text(self, tmp_path: Path) -> None:
        """Verify the CLI can pass the --catalog argument straight through."""
        path = write_catalog(tmp_path, entry())

        assert list(load_catalog(str(path))) == ["polityka.pdf"]

    def test_load_catalog__reads_polish_characters_on_any_system_encoding(
        self, tmp_path: Path
    ) -> None:
        """Verify titles survive on Windows, whose default encoding is not UTF-8."""
        path = write_catalog(tmp_path, entry(title="Źródła żółtej łąki"))

        assert load_catalog(path)["polityka.pdf"].title == "Źródła żółtej łąki"

    def test_load_catalog__returns_nothing_for_an_empty_document_list(self, tmp_path: Path) -> None:
        """Verify an empty catalog is valid and makes every file report missing metadata."""
        assert load_catalog(write_catalog(tmp_path)) == {}

    def test_load_catalog__parses_dates_and_lifecycle_fields(self, tmp_path: Path) -> None:
        """Verify text values from JSON become the typed values used by retrieval filters."""
        path = write_catalog(
            tmp_path,
            entry(
                adoption_date="2010-11-18",
                is_active=False,
                lifecycle_status="superseded",
                retrieval_enabled=False,
            ),
        )

        metadata = load_catalog(path)["polityka.pdf"]

        assert metadata.adoption_date.isoformat() == "2010-11-18"
        assert metadata.lifecycle_status is LifecycleStatus.SUPERSEDED
        assert metadata.retrieval_enabled is False

    def test_load_catalog__rejects_a_misspelled_field(self, tmp_path: Path) -> None:
        """Verify a typo such as 'is_activ' fails loudly instead of being ignored."""
        path = write_catalog(tmp_path, entry(is_activ=False))

        with pytest.raises(ValidationError, match="is_activ"):
            load_catalog(path)

    def test_load_catalog__rejects_an_unknown_document_kind(self, tmp_path: Path) -> None:
        """Verify every document gets one of the ranked kinds used to order sources."""
        with pytest.raises(ValidationError, match="document_kind"):
            load_catalog(write_catalog(tmp_path, entry(document_kind="ustawa")))

    def test_load_catalog__rejects_a_superseded_document_marked_as_in_force(
        self, tmp_path: Path
    ) -> None:
        """Verify a contradictory entry cannot let a replaced document back into letters."""
        path = write_catalog(tmp_path, entry(lifecycle_status="superseded", is_active=True))

        with pytest.raises(ValidationError, match="superseded wymaga is_active=false"):
            load_catalog(path)

    def test_load_catalog__requires_a_file_name_for_every_entry(self, tmp_path: Path) -> None:
        """Verify an entry without a file name is reported instead of being skipped."""
        document = entry()
        del document["filename"]

        with pytest.raises(KeyError, match="filename"):
            load_catalog(write_catalog(tmp_path, document))

    def test_load_catalog__requires_the_documents_list(self, tmp_path: Path) -> None:
        """Verify a file with a different top-level layout is not read as an empty catalog."""
        path = tmp_path / "catalog.json"
        path.write_text(json.dumps([entry()]), encoding="utf-8")

        with pytest.raises(TypeError):
            load_catalog(path)

    def test_load_catalog__reports_malformed_json(self, tmp_path: Path) -> None:
        """Verify a broken catalog stops ingestion before any document is processed."""
        path = tmp_path / "catalog.json"
        path.write_text('{"documents": [', encoding="utf-8")

        with pytest.raises(json.JSONDecodeError):
            load_catalog(path)

    def test_load_catalog__later_entry_for_the_same_file_wins(self, tmp_path: Path) -> None:
        """Verify the owner's chosen behaviour: a repeated file name keeps its last entry."""
        path = write_catalog(tmp_path, entry(), entry(document_id="inny", title="Inny dokument"))

        catalog = load_catalog(path)

        assert list(catalog) == ["polityka.pdf"]
        assert catalog["polityka.pdf"].document_id == "inny"
