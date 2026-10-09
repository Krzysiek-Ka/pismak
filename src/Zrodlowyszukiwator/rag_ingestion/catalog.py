from __future__ import annotations

import json
from pathlib import Path

from .models import DocumentMetadata


def load_catalog(path: str | Path) -> dict[str, DocumentMetadata]:
    """Load document metadata from a JSON catalog keyed by filename."""
    path = Path(path)
    raw = json.loads(path.read_text(encoding="utf-8"))

    result: dict[str, DocumentMetadata] = {}
    for item in raw["documents"]:
        filename = item.pop("filename")
        result[filename] = DocumentMetadata.model_validate(item)
    return result
