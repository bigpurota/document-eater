from __future__ import annotations

import csv
import io
import re
from pathlib import Path
from typing import Any, Literal

from defusedxml import ElementTree

from .artifacts import source_hash, write_document_artifacts
from .models import Block, Document, Page

TextFormat = Literal["xml", "csv", "txt", "md"]
_HEADING = re.compile(r"^(?:\d+(?:\.\d+)*[.)]?\s+|[A-ZА-ЯЁ][A-ZА-ЯЁ\s\d._-]{3,}$)")


def _decode_text(source: Path) -> tuple[str, str]:
    payload = source.read_bytes()
    if payload.startswith((b"\xff\xfe", b"\xfe\xff")):
        return payload.decode("utf-16"), "utf-16"
    for encoding in ("utf-8-sig", "cp1251"):
        try:
            return payload.decode(encoding), encoding
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Could not decode text file as UTF-8 or Windows-1251: {source}")


def _write_text_document(
    source: Path,
    output: str | Path,
    blocks: list[Block],
    *,
    format: TextFormat,
    settings: dict[str, Any],
) -> Path:
    sha256 = source_hash(source)
    doc_id = sha256[:16]
    normalized = [
        Block(**{**block.__dict__, "id": f"{doc_id}:p1:b{number}", "order": number})
        for number, block in enumerate(blocks, 1)
    ]
    page = Page(
        number=1,
        width=0.0,
        height=0.0,
        source="native",
        native_char_count=sum(len("".join(block.text.split())) for block in normalized),
        blocks=normalized,
        kind="document",
        label=f"{format.upper()} document",
    )
    document = Document(doc_id, source.name, sha256, [page], format=format)
    return write_document_artifacts(document, source, output, settings=settings)


def _plain_text_blocks(text: str, *, markdown: bool = False) -> list[Block]:
    blocks: list[Block] = []
    pending: list[str] = []
    start_line = 1

    def flush(end_line: int) -> None:
        nonlocal pending
        value = "\n".join(pending).strip()
        if value:
            blocks.append(
                Block(
                    id="",
                    page=1,
                    order=0,
                    text=value,
                    bbox=None,
                    source="native",
                    role=(
                        "heading"
                        if (markdown and value.lstrip().startswith("#"))
                        or (len(value) <= 160 and _HEADING.match(value))
                        else "body"
                    ),
                    attrs={
                        "location": f"lines {start_line}-{end_line}",
                        "line_start": start_line,
                        "line_end": end_line,
                    },
                )
            )
        pending = []

    lines = text.splitlines()
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            flush(line_number - 1)
            start_line = line_number + 1
        else:
            if not pending:
                start_line = line_number
            pending.append(line.rstrip())
    flush(len(lines))
    return blocks


def ingest_txt(path: str | Path, output: str | Path) -> Path:
    source = Path(path).expanduser().resolve()
    if source.suffix.casefold() != ".txt" or not source.is_file():
        raise ValueError(f"Expected an existing TXT file, got: {source}")
    text, encoding = _decode_text(source)
    return _write_text_document(
        source,
        output,
        _plain_text_blocks(text),
        format="txt",
        settings={"encoding": encoding, "paragraph_policy": "blank_lines"},
    )


def ingest_markdown(path: str | Path, output: str | Path) -> Path:
    source = Path(path).expanduser().resolve()
    if source.suffix.casefold() != ".md" or not source.is_file():
        raise ValueError(f"Expected an existing Markdown file, got: {source}")
    text, encoding = _decode_text(source)
    return _write_text_document(
        source,
        output,
        _plain_text_blocks(text, markdown=True),
        format="md",
        settings={"encoding": encoding, "parser": "markdown_blocks"},
    )


def ingest_csv(path: str | Path, output: str | Path) -> Path:
    source = Path(path).expanduser().resolve()
    if source.suffix.casefold() != ".csv" or not source.is_file():
        raise ValueError(f"Expected an existing CSV file, got: {source}")
    text, encoding = _decode_text(source)
    sample = text[:8192]
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
    except csv.Error:
        dialect = csv.excel
    try:
        has_header = csv.Sniffer().has_header(sample)
    except csv.Error:
        has_header = False
    rows = list(csv.reader(io.StringIO(text), dialect))
    headers = rows[0] if rows and has_header else []
    blocks: list[Block] = []
    for row_number, row in enumerate(rows, 1):
        populated = [
            (column, value.strip()) for column, value in enumerate(row, 1) if value.strip()
        ]
        if not populated:
            continue
        rendered = []
        cells = []
        for column, value in populated:
            label = headers[column - 1].strip() if column <= len(headers) else f"column {column}"
            rendered.append(f"{label}: {value}" if not (has_header and row_number == 1) else value)
            cells.append({"column": column, "label": label, "value": value})
        blocks.append(
            Block(
                id="",
                page=1,
                order=0,
                text=" | ".join(rendered),
                bbox=None,
                source="native",
                role="heading" if has_header and row_number == 1 else "body",
                attrs={"location": f"row {row_number}", "row": row_number, "cells": cells},
            )
        )
    return _write_text_document(
        source,
        output,
        blocks,
        format="csv",
        settings={
            "encoding": encoding,
            "delimiter": dialect.delimiter,
            "has_header": has_header,
        },
    )


def _xml_name(value: str) -> str:
    return value.rsplit("}", 1)[-1]


def ingest_xml(path: str | Path, output: str | Path) -> Path:
    source = Path(path).expanduser().resolve()
    if source.suffix.casefold() != ".xml" or not source.is_file():
        raise ValueError(f"Expected an existing XML file, got: {source}")
    root = ElementTree.parse(source).getroot()
    blocks: list[Block] = []

    def walk(element: Any, location: str) -> None:
        attributes = {_xml_name(key): value for key, value in element.attrib.items()}
        direct_text = " ".join((element.text or "").split())
        parts = [f"<{_xml_name(element.tag)}>"]
        parts.extend(f"@{key}: {value}" for key, value in attributes.items())
        if direct_text:
            parts.append(direct_text)
        if len(parts) > 1:
            blocks.append(
                Block(
                    id="",
                    page=1,
                    order=0,
                    text=" | ".join(parts),
                    bbox=None,
                    source="native",
                    attrs={
                        "location": location,
                        "xml_path": location,
                        "tag": _xml_name(element.tag),
                        "attributes": attributes,
                    },
                )
            )
        counts: dict[str, int] = {}
        for child in element:
            name = _xml_name(child.tag)
            counts[name] = counts.get(name, 0) + 1
            walk(child, f"{location}/{name}[{counts[name]}]")
            tail = " ".join((child.tail or "").split())
            if tail:
                blocks.append(
                    Block(
                        id="",
                        page=1,
                        order=0,
                        text=tail,
                        bbox=None,
                        source="native",
                        attrs={"location": location, "xml_path": location},
                    )
                )

    root_name = _xml_name(root.tag)
    walk(root, f"/{root_name}[1]")
    return _write_text_document(
        source,
        output,
        blocks,
        format="xml",
        settings={"parser": "defusedxml", "external_entities": "disabled"},
    )
