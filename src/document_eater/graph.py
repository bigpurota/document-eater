from __future__ import annotations

import re
import statistics

from .models import Block, Document, DocumentGraph, GraphEdge, GraphNode

_NUMBERED_HEADING = re.compile(r"^(?:\d+(?:\.\d+)*[.)]?|[IVXLC]+[.)])\s+\S", re.I)


def _with_roles(document: Document) -> Document:
    sizes = [b.font_size for p in document.pages for b in p.blocks if b.font_size]
    median = statistics.median(sizes) if sizes else None
    pages = []
    for page in document.pages:
        blocks = []
        for block in page.blocks:
            short = len(block.text) <= 160 and block.text.count("\n") <= 2
            larger = bool(median and block.font_size and block.font_size >= median * 1.18)
            numbered = bool(_NUMBERED_HEADING.match(block.text.strip()))
            role = (
                "heading" if block.role == "heading" or short and (larger or numbered) else "body"
            )
            blocks.append(Block(**{**block.__dict__, "role": role}))
        pages.append(type(page)(**{**page.__dict__, "blocks": blocks}))
    return type(document)(**{**document.__dict__, "pages": pages})


def build_graph(document: Document) -> tuple[Document, DocumentGraph]:
    document = _with_roles(document)
    nodes = [
        GraphNode(
            document.id,
            "document",
            {"filename": document.filename, "format": document.format},
        )
    ]
    edges: list[GraphEdge] = []
    previous_block: str | None = None
    current_section: str | None = None

    for page in document.pages:
        page_id = f"{document.id}:p{page.number}"
        nodes.append(
            GraphNode(
                page_id,
                "page",
                {
                    "number": page.number,
                    "source": page.source,
                    "kind": page.kind,
                    "label": page.label,
                    **page.attrs,
                },
            )
        )
        edges.append(GraphEdge(document.id, page_id, "contains"))
        for block in page.blocks:
            nodes.append(
                GraphNode(
                    block.id,
                    "block",
                    {
                        "page": block.page,
                        "order": block.order,
                        "role": block.role,
                        "source": block.source,
                        "text": block.text,
                        "bbox": block.bbox.__dict__ if block.bbox else None,
                        **block.attrs,
                    },
                )
            )
            edges.append(GraphEdge(page_id, block.id, "contains"))
            if previous_block:
                edges.append(GraphEdge(previous_block, block.id, "next"))
            if block.role == "heading":
                current_section = block.id
            elif current_section:
                edges.append(GraphEdge(block.id, current_section, "in_section"))
            previous_block = block.id
    return document, DocumentGraph(nodes, edges)
