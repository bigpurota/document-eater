from __future__ import annotations

import csv
import io
import re
import shutil
import subprocess
from pathlib import Path
from typing import Literal

import pymupdf

from .artifacts import source_hash, write_document_artifacts
from .models import BBox, Block, Document, Page

OCRMode = Literal["auto", "never", "always"]
_SPACE = re.compile(r"[ \t]+")


def _clean_text(text: str) -> str:
    lines = [_SPACE.sub(" ", line).strip() for line in text.splitlines()]
    return "\n".join(line for line in lines if line).strip()


def _usable_native_text(text: str, min_chars: int) -> bool:
    compact = "".join(text.split())
    if len(compact) < min_chars:
        return False
    bad = sum(ch == "\ufffd" or (ord(ch) < 32 and ch not in "\n\t") for ch in text)
    return bad / max(1, len(text)) <= 0.03


def _native_blocks(page: pymupdf.Page, doc_id: str, page_number: int) -> list[Block]:
    result: list[Block] = []
    raw = page.get_text("dict", sort=True)
    for raw_block in raw.get("blocks", []):
        if raw_block.get("type") != 0:
            continue
        lines = []
        sizes = []
        for line in raw_block.get("lines", []):
            spans = line.get("spans", [])
            lines.append("".join(span.get("text", "") for span in spans))
            sizes.extend(float(span["size"]) for span in spans if span.get("size"))
        text = _clean_text("\n".join(lines))
        if not text:
            continue
        bbox = raw_block.get("bbox", (0.0, 0.0, 0.0, 0.0))
        order = len(result) + 1
        result.append(
            Block(
                id=f"{doc_id}:p{page_number}:b{order}",
                page=page_number,
                order=order,
                text=text,
                bbox=BBox(*(round(float(v), 2) for v in bbox)),
                source="native",
                font_size=round(sum(sizes) / len(sizes), 2) if sizes else None,
            )
        )
    return result


def _ocr_blocks(
    page: pymupdf.Page,
    doc_id: str,
    page_number: int,
    languages: str,
    dpi: int,
) -> list[Block]:
    if not shutil.which("tesseract"):
        raise RuntimeError("Tesseract is required for OCR pages but was not found")
    pix = page.get_pixmap(dpi=dpi, alpha=False, colorspace=pymupdf.csRGB)
    proc = subprocess.run(
        ["tesseract", "stdin", "stdout", "-l", languages, "tsv"],
        input=pix.tobytes("png"),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if proc.returncode:
        message = proc.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"Tesseract failed on page {page_number}: {message}")

    groups: dict[tuple[str, str, str], list[dict[str, str]]] = {}
    reader = csv.DictReader(io.StringIO(proc.stdout.decode("utf-8")), delimiter="\t")
    for row in reader:
        text = (row.get("text") or "").strip()
        confidence = float(row.get("conf") or -1)
        if text and confidence >= 0:
            key = (row["block_num"], row["par_num"], row["line_num"])
            groups.setdefault(key, []).append(row)

    scale = 72.0 / dpi
    blocks: list[Block] = []
    for words in groups.values():
        text = _clean_text(" ".join(word["text"] for word in words))
        if not text:
            continue
        left = min(int(word["left"]) for word in words)
        top = min(int(word["top"]) for word in words)
        right = max(int(word["left"]) + int(word["width"]) for word in words)
        bottom = max(int(word["top"]) + int(word["height"]) for word in words)
        order = len(blocks) + 1
        blocks.append(
            Block(
                id=f"{doc_id}:p{page_number}:b{order}",
                page=page_number,
                order=order,
                text=text,
                bbox=BBox(*(round(value * scale, 2) for value in (left, top, right, bottom))),
                source="ocr",
            )
        )
    blocks.sort(key=lambda b: (b.bbox.y0, b.bbox.x0))
    return [
        Block(**{**block.__dict__, "order": i, "id": f"{doc_id}:p{page_number}:b{i}"})
        for i, block in enumerate(blocks, 1)
    ]


def inspect_pdf(path: str | Path, min_native_chars: int = 40) -> dict:
    source = Path(path).expanduser().resolve()
    with pymupdf.open(source) as pdf:
        pages = []
        for index, page in enumerate(pdf, 1):
            text = _clean_text(page.get_text("text", sort=True))
            pages.append(
                {
                    "page": index,
                    "native_char_count": len("".join(text.split())),
                    "recommended_source": (
                        "native" if _usable_native_text(text, min_native_chars) else "ocr"
                    ),
                }
            )
    return {"path": str(source), "pages": pages}


def ingest_pdf(
    path: str | Path,
    output: str | Path,
    *,
    ocr: OCRMode = "auto",
    languages: str = "rus+eng",
    dpi: int = 300,
    min_native_chars: int = 40,
) -> Path:
    source = Path(path).expanduser().resolve()
    if source.suffix.lower() != ".pdf" or not source.is_file():
        raise ValueError(f"Expected an existing PDF file, got: {source}")
    if ocr not in {"auto", "never", "always"}:
        raise ValueError(f"Unsupported OCR mode: {ocr}")

    sha256 = source_hash(source)
    doc_id = sha256[:16]
    pages: list[Page] = []
    with pymupdf.open(source) as pdf:
        for number, raw_page in enumerate(pdf, 1):
            native = _native_blocks(raw_page, doc_id, number)
            native_text = "\n".join(block.text for block in native)
            native_chars = len("".join(native_text.split()))
            use_ocr = ocr == "always" or (
                ocr == "auto" and not _usable_native_text(native_text, min_native_chars)
            )
            blocks = _ocr_blocks(raw_page, doc_id, number, languages, dpi) if use_ocr else native
            source_kind = "ocr" if use_ocr else "native"
            pages.append(
                Page(
                    number=number,
                    width=round(raw_page.rect.width, 2),
                    height=round(raw_page.rect.height, 2),
                    source=source_kind,
                    native_char_count=native_chars,
                    blocks=blocks,
                )
            )

    document = Document(doc_id, source.name, sha256, pages, format="pdf")
    return write_document_artifacts(
        document,
        source,
        output,
        settings={
            "ocr": ocr,
            "languages": languages,
            "dpi": dpi,
            "min_native_chars": min_native_chars,
        },
    )
