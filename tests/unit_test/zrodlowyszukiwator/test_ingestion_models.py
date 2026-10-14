from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.Zrodlowyszukiwator.rag_ingestion.catalog import load_catalog
from src.Zrodlowyszukiwator.rag_ingestion.cli import DEFAULT_CATALOG_PATH
from src.Zrodlowyszukiwator.rag_ingestion.models import DocumentMetadata, LifecycleStatus


def make_metadata(**overrides: object) -> DocumentMetadata:
    values: dict[str, object] = {
        "document_id": "doc-1",
        "title": "Dokument",
        "document_kind": "policy",
    }
    values.update(overrides)
    return DocumentMetadata.model_validate(values)


class TestDocumentMetadata:
    def test_document_metadata__accepts_every_entry_of_the_repository_catalog(self) -> None:
        """Verify the catalog shipped with the repository still fits the model that loads it."""
        catalog = load_catalog(DEFAULT_CATALOG_PATH)

        assert len(catalog) == 6
        superseded = [
            metadata.document_id
            for metadata in catalog.values()
            if metadata.lifecycle_status is LifecycleStatus.SUPERSEDED
        ]
        assert superseded == ["wroclaw-strategy-2030"]
        assert all(metadata.source_url for metadata in catalog.values())

    def test_document_metadata__keeps_documents_searchable_unless_the_catalog_disables_them(
        self,
    ) -> None:
        """Verify entries written before the switch existed are not silently dropped from search."""
        assert make_metadata().retrieval_enabled is True
        assert make_metadata(retrieval_enabled=False).retrieval_enabled is False

    @pytest.mark.parametrize("is_active", [True, None])
    def test_document_metadata__rejects_superseded_document_not_marked_as_out_of_force(
        self, is_active: bool | None
    ) -> None:
        """Verify a contradictory entry cannot let a replaced document through the search filter."""
        with pytest.raises(ValidationError, match="superseded wymaga is_active=false"):
            make_metadata(lifecycle_status="superseded", is_active=is_active)

    def test_document_metadata__accepts_superseded_document_marked_as_out_of_force(self) -> None:
        """Verify the consistent form of a replaced document is accepted."""
        metadata = make_metadata(lifecycle_status="superseded", is_active=False)

        assert metadata.lifecycle_status is LifecycleStatus.SUPERSEDED

    def test_document_metadata__rejects_unknown_lifecycle_status(self) -> None:
        """Verify a typo in the status is reported instead of being stored as a new status."""
        with pytest.raises(ValidationError, match="lifecycle_status"):
            make_metadata(lifecycle_status="superceded")

    def test_document_metadata__still_rejects_fields_it_does_not_know(self) -> None:
        """Verify a misspelled field name fails loudly rather than being ignored."""
        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            make_metadata(retrieval_enabeld=False)
