from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .graph import build_graph
from .models import Document


def source_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_document_artifacts(
    document: Document,
    source: Path,
    output: str | Path,
    *,
    settings: dict[str, Any],
) -> Path:
    document, graph = build_graph(document)
    destination = Path(output).expanduser().resolve() / document.id
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "document.json").write_text(
        json.dumps(document.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (destination / "graph.json").write_text(
        json.dumps(graph.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    manifest = {
        "schema_version": 2,
        "document_id": document.id,
        "source_filename": source.name,
        "source_path": str(source),
        "source_sha256": document.sha256,
        "source_format": document.format,
        "source_copied": False,
        # Kept for readers of the first PDF-only artifact schema.
        "source_pdf_copied": False,
        "settings": settings,
        "unit_sources": [page.source for page in document.pages],
        "page_sources": [page.source for page in document.pages],
    }
    (destination / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return destination
