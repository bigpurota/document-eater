from __future__ import annotations

import os
from collections import Counter
from pathlib import Path
from typing import Iterable

from .office import ingest_docx, ingest_xlsx
from .pdf import OCRMode, ingest_pdf
from .text_formats import ingest_csv, ingest_markdown, ingest_txt, ingest_xml

SUPPORTED_EXTENSIONS = frozenset({".pdf", ".docx", ".xlsx", ".xml", ".csv", ".txt", ".md"})
WORKSPACE_MARKER = ".document-eater-workspace"
_LEGACY_WORKSPACE_MARKER = "run-manifest.json"


def _is_within(path: Path, directory: Path) -> bool:
    return path == directory or directory in path.parents


def discover_documents(path: str | Path, *, exclude_paths: Iterable[str | Path] = ()) -> list[Path]:
    source = Path(path).expanduser().resolve()
    excluded = tuple(Path(item).expanduser().resolve() for item in exclude_paths)
    if (
        source.is_file()
        and source.suffix.casefold() in SUPPORTED_EXTENSIONS
        and not any(_is_within(source, directory) for directory in excluded)
    ):
        return [source]
    if source.is_dir():
        documents: list[Path] = []
        for root_name, directory_names, filenames in os.walk(source, followlinks=False):
            root = Path(root_name).resolve()
            if any(_is_within(root, directory) for directory in excluded):
                directory_names.clear()
                continue
            if (root / WORKSPACE_MARKER).is_file() or (root / _LEGACY_WORKSPACE_MARKER).is_file():
                directory_names.clear()
                continue
            directory_names[:] = [
                name
                for name in directory_names
                if not any(_is_within((root / name).resolve(), directory) for directory in excluded)
                and not (root / name / WORKSPACE_MARKER).is_file()
                and not (root / name / _LEGACY_WORKSPACE_MARKER).is_file()
            ]
            for filename in filenames:
                candidate = (root / filename).resolve()
                if candidate.suffix.casefold() in SUPPORTED_EXTENSIONS:
                    documents.append(candidate)
        return sorted(documents)
    supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
    raise ValueError(f"Expected a supported document ({supported}) or directory, got: {source}")


def document_counts(paths: list[Path]) -> dict[str, int]:
    counts = Counter(path.suffix.casefold().removeprefix(".") for path in paths)
    return {
        kind: counts.get(kind, 0) for kind in ("pdf", "docx", "xlsx", "xml", "csv", "txt", "md")
    }


def ingest_document(
    path: str | Path,
    output: str | Path,
    *,
    ocr: OCRMode = "auto",
    languages: str = "rus+eng",
    dpi: int = 300,
    min_native_chars: int = 40,
) -> Path:
    source = Path(path).expanduser().resolve()
    suffix = source.suffix.casefold()
    if suffix == ".pdf":
        return ingest_pdf(
            source,
            output,
            ocr=ocr,
            languages=languages,
            dpi=dpi,
            min_native_chars=min_native_chars,
        )
    if suffix == ".docx":
        return ingest_docx(source, output)
    if suffix == ".xlsx":
        return ingest_xlsx(source, output)
    if suffix == ".xml":
        return ingest_xml(source, output)
    if suffix == ".csv":
        return ingest_csv(source, output)
    if suffix == ".txt":
        return ingest_txt(source, output)
    if suffix == ".md":
        return ingest_markdown(source, output)
    supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
    raise ValueError(f"Unsupported document type {suffix or '[none]'}; expected {supported}")
