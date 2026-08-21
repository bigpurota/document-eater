from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

TextSource = Literal["native", "ocr"]
DocumentFormat = Literal["pdf", "docx", "xlsx", "xml", "csv", "txt", "md"]
UnitKind = Literal["page", "document", "sheet"]


@dataclass(frozen=True)
class BBox:
    x0: float
    y0: float
    x1: float
    y1: float


@dataclass(frozen=True)
class Block:
    id: str
    page: int
    order: int
    text: str
    bbox: BBox | None
    source: TextSource
    font_size: float | None = None
    role: Literal["heading", "body"] = "body"
    attrs: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Page:
    number: int
    width: float
    height: float
    source: TextSource
    native_char_count: int
    blocks: list[Block] = field(default_factory=list)
    kind: UnitKind = "page"
    label: str | None = None
    attrs: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Document:
    id: str
    filename: str
    sha256: str
    pages: list[Page]
    format: DocumentFormat = "pdf"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class GraphNode:
    id: str
    kind: Literal["document", "page", "block"]
    attrs: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class GraphEdge:
    source: str
    target: str
    kind: Literal["contains", "next", "in_section"]


@dataclass(frozen=True)
class DocumentGraph:
    nodes: list[GraphNode]
    edges: list[GraphEdge]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
