from __future__ import annotations

from collections import Counter
from pathlib import Path

from .office import ingest_docx, ingest_xlsx
from .pdf import OCRMode, ingest_pdf
from .text_formats import ingest_csv, ingest_markdown, ingest_txt, ingest_xml

SUPPORTED_EXTENSIONS = frozenset({".pdf", ".docx", ".xlsx", ".xml", ".csv", ".txt", ".md"})


def discover_documents(path: str | Path) -> list[Path]:
    source = Path(path).expanduser().resolve()
    if source.is_file() and source.suffix.casefold() in SUPPORTED_EXTENSIONS:
        return [source]
    if source.is_dir():
        return sorted(
            candidate.resolve()
            for candidate in source.rglob("*")
            if candidate.is_file() and candidate.suffix.casefold() in SUPPORTED_EXTENSIONS
        )
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
