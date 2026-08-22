from __future__ import annotations

import gc
import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any

from mcp.server import MCPServer

from .audit import audit_corpus
from .index import search
from .llm import (
    ABLITERATED_GENERATION,
    ABLITERATED_MODEL,
    BASE_GENERATION,
    BASE_MODEL,
    QwenClient,
)
from .rag import (
    DEFAULT_EMBEDDING_MODEL,
    BgeM3Encoder,
    BgeM3Reranker,
    FastEmbedEncoder,
    HybridRetriever,
)

mcp = MCPServer(
    "document-eater",
    title="Document Eater",
    description="Private local document ingestion, retrieval, and requirement audit tools.",
    instructions=(
        "Use these tools for private documents. Do not send returned document text to a "
        "non-local model unless the data owner explicitly permits it. UNKNOWN means that "
        "the available evidence is insufficient, not that a requirement failed."
    ),
)
_retrievers: dict[tuple[str, str, str], HybridRetriever] = {}
DEFAULT_WORKSPACE = ".document-eater-workspace"
DEFAULT_EMBEDDING_CACHE = os.environ.get("DOCUMENT_EATER_MODEL_CACHE", "models/retrieval")
DEFAULT_SEARCH_RESULTS = 4
MAX_SEARCH_RESULTS = 6
DEFAULT_SEARCH_TEXT_CHARS = 1_000
MAX_SEARCH_TEXT_CHARS = 1_500
DEFAULT_SEARCH_TOTAL_CHARS = 6_000
MAX_SEARCH_TOTAL_CHARS = 8_000
DEFAULT_AUDIT_PAGE_SIZE = 3
MAX_AUDIT_PAGE_SIZE = 5
DEFAULT_AUDIT_TOTAL_CHARS = 6_000
MAX_AUDIT_TOTAL_CHARS = 8_000
DEFAULT_PAGE_TEXT_CHARS = 4_000
MAX_PAGE_TEXT_CHARS = 5_000
DEFAULT_GRAPH_EDGE_LIMIT = 12
MAX_GRAPH_EDGE_LIMIT = 20
DEFAULT_GRAPH_TOTAL_CHARS = 6_000
MAX_GRAPH_TOTAL_CHARS = 8_000


def _release_retrieval_memory() -> None:
    """Drop cached retrievers and return unused Metal allocations after batch work."""
    _retrievers.clear()
    gc.collect()
    try:
        import torch

        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except (ImportError, RuntimeError):
        pass


def _bounded(value: int, *, minimum: int, maximum: int) -> int:
    return max(minimum, min(int(value), maximum))


def _truncate(value: Any, limit: int) -> tuple[str, bool]:
    text = str(value or "")
    if len(text) <= limit:
        return text, False
    return text[:limit], True


def _compact_audit_item(item: dict[str, Any], *, include_evidence: bool) -> dict[str, Any]:
    raw_requirement = dict(item.get("requirement") or {})
    requirement = {
        "id": str(raw_requirement.get("id") or "")[:160],
        "text": _truncate(raw_requirement.get("text"), 1_000)[0],
        "document_id": str(raw_requirement.get("document_id") or "")[:200],
        "filename": str(raw_requirement.get("filename") or "")[:300],
        "source_path": str(raw_requirement.get("source_path") or "")[:500] or None,
        "page": raw_requirement.get("page"),
        "location": str(raw_requirement.get("location") or "")[:300],
        "block_id": str(raw_requirement.get("block_id") or "")[:200],
    }
    requirement_truncated = len(str(raw_requirement.get("text") or "")) > 1_000
    rationale, rationale_truncated = _truncate(item.get("rationale"), 600)
    compact = {
        "requirement": requirement,
        "status": item.get("status"),
        "rationale": rationale,
        "used_citations": [str(value)[:240] for value in item.get("used_citations", [])[:5]],
        "model": item.get("model"),
        "text_truncated": requirement_truncated or rationale_truncated,
    }
    if include_evidence:
        evidence = []
        for raw in item.get("retrieved_evidence", [])[:2]:
            value = dict(raw)
            preview, preview_truncated = _truncate(value.get("preview"), 400)
            evidence.append(
                {
                    "label": str(value.get("label") or "")[:300],
                    "preview": preview,
                    "preview_truncated": preview_truncated,
                    "chunk_id": str(value.get("chunk_id") or "")[:200],
                    "document_id": str(value.get("document_id") or "")[:200],
                    "pages": list(value.get("pages") or [])[:2],
                    "locations": [str(entry)[:300] for entry in (value.get("locations") or [])[:2]],
                    "retrieval_score": value.get("retrieval_score"),
                }
            )
        compact["retrieved_evidence"] = evidence
    return compact


def _compact_search_hit(hit: Any, text_limit: int) -> dict[str, Any]:
    raw = asdict(hit)
    text, text_truncated = _truncate(raw.get("text"), text_limit)
    return {
        "chunk_id": str(raw.get("chunk_id") or "")[:200],
        "document_id": str(raw.get("document_id") or "")[:200],
        "page_start": raw.get("page_start"),
        "page_end": raw.get("page_end"),
        "heading": _truncate(raw.get("heading"), 300)[0],
        "text": text,
        "text_truncated": text_truncated,
        "block_ids": [str(value)[:200] for value in (raw.get("block_ids") or [])[:8]],
        "score": raw.get("score"),
        "location_start": str(raw.get("location_start") or "")[:300],
        "location_end": str(raw.get("location_end") or "")[:300],
        "retrieval_scores": raw.get("retrieval_scores") or {},
    }


def _resolve_run_file(value: str, filename: str) -> Path:
    """Resolve either an exact run file or the newest file under a run root."""
    requested = Path(value).expanduser().resolve()
    if requested.is_file():
        return requested
    root = requested if requested.is_dir() else requested.parent
    matches = [path for path in root.glob(f"*/{filename}") if path.is_file()]
    if not matches:
        raise ValueError(f"No {filename} found at or below: {requested}")
    return max(matches, key=lambda path: (path.stat().st_mtime_ns, str(path)))


def _resolve_artifacts_root(value: str) -> Path:
    """Resolve an exact artifacts folder or the newest one under a run root."""
    requested = Path(value).expanduser().resolve()
    if requested.is_dir():
        return requested
    parent = requested.parent
    matches = [path for path in parent.glob("*/artifacts") if path.is_dir()]
    if not matches:
        raise ValueError(f"No artifacts directory found at or below: {requested}")
    return max(matches, key=lambda path: (path.stat().st_mtime_ns, str(path)))


def _hybrid_retriever(
    database_path: str, embedding_model: str, embedding_cache: str, mode: str = "quality"
) -> HybridRetriever:
    database = str(Path(database_path).expanduser().resolve())
    cache = str(Path(embedding_cache).expanduser().resolve())
    key = (database, f"{mode}:{embedding_model}", cache)
    if key not in _retrievers:
        if mode == "quality":
            _retrievers[key] = HybridRetriever(database, BgeM3Encoder(cache), BgeM3Reranker(cache))
        elif mode == "hybrid":
            _retrievers[key] = HybridRetriever(database, FastEmbedEncoder(embedding_model, cache))
        else:
            raise ValueError("mode must be 'quality' or 'hybrid'")
    return _retrievers[key]


def _client(profile: str, base_url: str | None) -> QwenClient:
    if profile not in {"base", "abliterated"}:
        raise ValueError("profile must be 'base' or 'abliterated'")
    abliterated = profile == "abliterated"
    return QwenClient(
        base_url or os.environ.get("DOCUMENT_EATER_QWEN_URL", "http://127.0.0.1:8080/v1"),
        ABLITERATED_MODEL if abliterated else BASE_MODEL,
        generation_options=ABLITERATED_GENERATION if abliterated else BASE_GENERATION,
        use_system_prompt=not abliterated,
    )


@mcp.tool()
def audit_documents(
    input_path: str,
    output_path: str = DEFAULT_WORKSPACE,
    use_qwen: bool = False,
    profile: str = "base",
    qwen_base_url: str | None = None,
    retrieval_mode: str = "quality",
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    embedding_cache: str = DEFAULT_EMBEDDING_CACHE,
    force_rebuild: bool = False,
) -> dict[str, Any]:
    """Audit once or reuse an unchanged corpus; set force_rebuild only when explicitly requested."""
    _release_retrieval_memory()
    try:
        report = audit_corpus(
            input_path,
            output_path,
            client=_client(profile, qwen_base_url) if use_qwen else None,
            retrieval_mode=retrieval_mode,  # type: ignore[arg-type]
            embedding_model=embedding_model,
            embedding_cache=embedding_cache,
            force_rebuild=force_rebuild,
        )
    finally:
        _release_retrieval_memory()
    return {
        "verification_mode": report.verification_mode,
        "retrieval_mode": report.retrieval_mode,
        "summary": report.summary,
        "requirements": len(report.items),
        "reused": report.reused,
        "report_path": str(Path(report.run_directory) / "report.html"),
        "audit_path": str(Path(report.run_directory) / "audit.json"),
    }


@mcp.tool()
def prepare_corpus(
    input_path: str,
    workspace: str = DEFAULT_WORKSPACE,
    retrieval_mode: str = "quality",
    embedding_cache: str = DEFAULT_EMBEDDING_CACHE,
    force_rebuild: bool = False,
) -> dict[str, Any]:
    """Extract and index a private document corpus locally for subsequent search."""
    _release_retrieval_memory()
    try:
        report = audit_corpus(
            input_path,
            workspace,
            client=None,
            retrieval_mode=retrieval_mode,  # type: ignore[arg-type]
            embedding_cache=embedding_cache,
            force_rebuild=force_rebuild,
        )
    finally:
        _release_retrieval_memory()
    return {
        "document_workspace": str(Path(report.run_directory).resolve()),
        "pdf_workspace": str(Path(report.run_directory).resolve()),
        "database_path": str(Path(report.run_directory).resolve() / "index.sqlite3"),
        "retrieval_mode": report.retrieval_mode,
        "requirement_candidates": len(report.items),
        "reused": report.reused,
        "candidate_report": str(Path(report.run_directory).resolve() / "report.html"),
    }


@mcp.tool()
def search_corpus(
    query: str,
    database_path: str = f"{DEFAULT_WORKSPACE}/index.sqlite3",
    limit: int = DEFAULT_SEARCH_RESULTS,
    mode: str = "quality",
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    embedding_cache: str = DEFAULT_EMBEDDING_CACHE,
    max_chars_per_hit: int = DEFAULT_SEARCH_TEXT_CHARS,
    max_total_chars: int = DEFAULT_SEARCH_TOTAL_CHARS,
) -> dict[str, Any]:
    """Search locally with bounded snippets so one call cannot exhaust agent context."""
    database_path = str(_resolve_run_file(database_path, "index.sqlite3"))
    requested_limit = _bounded(limit, minimum=1, maximum=MAX_SEARCH_RESULTS)
    text_limit = _bounded(max_chars_per_hit, minimum=200, maximum=MAX_SEARCH_TEXT_CHARS)
    total_limit = _bounded(max_total_chars, minimum=2_000, maximum=MAX_SEARCH_TOTAL_CHARS)
    if mode in {"quality", "hybrid"}:
        hits = _hybrid_retriever(database_path, embedding_model, embedding_cache, mode).search(
            query, limit=requested_limit
        )
    elif mode == "lexical":
        hits = search(database_path, query, limit=requested_limit)
    else:
        raise ValueError("mode must be 'quality', 'hybrid', or 'lexical'")
    items: list[dict[str, Any]] = []
    for hit in hits[:requested_limit]:
        item = _compact_search_hit(hit, text_limit)
        candidate = [*items, item]
        if items and len(json.dumps(candidate, ensure_ascii=False)) > total_limit:
            break
        items = candidate
    return {
        "query": query,
        "returned": len(items),
        "limit": requested_limit,
        "max_chars_per_hit": text_limit,
        "max_total_chars": total_limit,
        "items": items,
    }


@mcp.tool()
def list_audit_items(
    audit_path: str = f"{DEFAULT_WORKSPACE}/audit.json",
    status: str | None = None,
    offset: int = 0,
    limit: int = DEFAULT_AUDIT_PAGE_SIZE,
    include_evidence: bool = False,
    max_total_chars: int = DEFAULT_AUDIT_TOTAL_CHARS,
) -> dict[str, Any]:
    """Read a compact page of audit items; evidence previews are opt-in and bounded."""
    if offset < 0:
        raise ValueError("offset must not be negative")
    path = _resolve_run_file(audit_path, "audit.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    items = payload.get("items", [])
    if status:
        requested = status.upper()
        items = [item for item in items if item.get("status") == requested]
    page_size = _bounded(limit, minimum=1, maximum=MAX_AUDIT_PAGE_SIZE)
    selected = items[offset : offset + page_size]
    total_limit = _bounded(max_total_chars, minimum=2_000, maximum=MAX_AUDIT_TOTAL_CHARS)
    compact_items: list[dict[str, Any]] = []
    for item in selected:
        compact = _compact_audit_item(item, include_evidence=include_evidence)
        candidate = [*compact_items, compact]
        if compact_items and len(json.dumps(candidate, ensure_ascii=False)) > total_limit:
            break
        compact_items = candidate
    return {
        "status": status.upper() if status else None,
        "total": len(items),
        "offset": offset,
        "returned": len(compact_items),
        "has_more": offset + len(compact_items) < len(items),
        "next_offset": offset + len(compact_items),
        "max_total_chars": total_limit,
        "items": compact_items,
    }


@mcp.tool()
def get_audit_summary(
    audit_path: str = f"{DEFAULT_WORKSPACE}/audit.json",
) -> dict[str, Any]:
    """Return the compact summary and report location for a completed audit."""
    path = _resolve_run_file(audit_path, "audit.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        "verification_mode": payload.get("verification_mode"),
        "retrieval_mode": payload.get("retrieval_mode"),
        "summary": payload.get("summary", {}),
        "requirements": len(payload.get("items", [])),
        "html_report": str(path.with_name("report.html")),
    }


@mcp.tool()
def read_document_page(
    document_id: str,
    page: int,
    artifacts_root: str = f"{DEFAULT_WORKSPACE}/artifacts",
    offset: int = 0,
    max_chars: int = DEFAULT_PAGE_TEXT_CHARS,
) -> dict[str, Any]:
    """Read a bounded character window from an extracted page, unit, or sheet."""
    if page < 1:
        raise ValueError("page must be positive")
    if offset < 0:
        raise ValueError("offset must not be negative")
    root = _resolve_artifacts_root(artifacts_root)
    matches = []
    for path in root.rglob("document.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("id") == document_id:
            matches.append((path, payload))
    if not matches:
        raise ValueError(f"Document not found in artifacts: {document_id}")
    path, document = sorted(matches, key=lambda item: str(item[0]))[-1]
    selected = next((item for item in document["pages"] if item["number"] == page), None)
    if selected is None:
        raise ValueError(f"Page {page} not found in document {document_id}")
    text_limit = _bounded(max_chars, minimum=200, maximum=MAX_PAGE_TEXT_CHARS)
    total_chars = sum(len(str(block.get("text") or "")) for block in selected["blocks"])
    if offset > total_chars:
        raise ValueError(f"offset {offset} exceeds page text length {total_chars}")
    remaining = text_limit
    cursor = 0
    blocks = []
    for raw_block in selected["blocks"]:
        text = str(raw_block.get("text") or "")
        block_start = cursor
        block_end = block_start + len(text)
        cursor = block_end
        if block_end <= offset or remaining <= 0:
            continue
        local_start = max(0, offset - block_start)
        snippet = text[local_start : local_start + remaining]
        if not snippet:
            continue
        attrs = raw_block.get("attrs") or {}
        blocks.append(
            {
                "id": raw_block.get("id"),
                "role": raw_block.get("role"),
                "source": raw_block.get("source"),
                "location": attrs.get("location"),
                "text": snippet,
                "text_start": block_start + local_start,
                "text_end": block_start + local_start + len(snippet),
                "text_truncated": local_start > 0 or local_start + len(snippet) < len(text),
            }
        )
        remaining -= len(snippet)
    returned_chars = text_limit - remaining
    next_offset = offset + returned_chars
    manifest_path = path.with_name("manifest.json")
    manifest = (
        json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}
    )
    return {
        "document_id": document_id,
        "filename": str(document["filename"])[:300],
        "source_path": str(manifest.get("source_path") or "")[:500] or None,
        "offset": offset,
        "returned_chars": returned_chars,
        "total_chars": total_chars,
        "truncated": next_offset < total_chars,
        "next_offset": next_offset,
        "page": {
            "number": selected.get("number"),
            "label": selected.get("label"),
            "kind": selected.get("kind"),
            "source": selected.get("source"),
            "blocks": blocks,
        },
    }


@mcp.tool()
def graph_neighbors(
    node_id: str,
    artifacts_root: str = f"{DEFAULT_WORKSPACE}/artifacts",
    offset: int = 0,
    limit: int = DEFAULT_GRAPH_EDGE_LIMIT,
    max_total_chars: int = DEFAULT_GRAPH_TOTAL_CHARS,
) -> dict[str, Any]:
    """Return a compact page of structural graph edges touching a node."""
    if offset < 0:
        raise ValueError("offset must not be negative")
    root = _resolve_artifacts_root(artifacts_root)
    edges = []
    nodes = []
    for path in root.rglob("graph.json"):
        graph = json.loads(path.read_text(encoding="utf-8"))
        matching_edges = [
            edge
            for edge in graph.get("edges", [])
            if edge.get("source") == node_id or edge.get("target") == node_id
        ]
        if matching_edges:
            edges.extend(matching_edges)
            related_ids = {node_id}
            for edge in matching_edges:
                related_ids.update((edge["source"], edge["target"]))
            nodes.extend(node for node in graph.get("nodes", []) if node.get("id") in related_ids)
    edge_limit = _bounded(limit, minimum=1, maximum=MAX_GRAPH_EDGE_LIMIT)
    edges = sorted(edges, key=lambda edge: (edge.get("source", ""), edge.get("target", "")))
    total = len(edges)
    selected_edges = edges[offset : offset + edge_limit]
    compact_nodes_by_id = {}
    seen_nodes = set()
    for node in nodes:
        node_key = node.get("id")
        if node_key in seen_nodes:
            continue
        seen_nodes.add(node_key)
        raw_attrs = dict(node.get("attrs") or {})
        attrs = {
            key: (_truncate(value, 300)[0] if isinstance(value, str) else value)
            for key, value in raw_attrs.items()
            if key
            in {
                "filename",
                "format",
                "number",
                "source",
                "kind",
                "label",
                "page",
                "order",
                "role",
                "location",
                "sheet",
                "row",
                "column",
            }
        }
        if "text" in raw_attrs:
            attrs["text"], attrs["text_truncated"] = _truncate(raw_attrs["text"], 300)
        compact_nodes_by_id[node_key] = {
            "id": str(node.get("id") or "")[:200],
            "kind": node.get("kind"),
            "attrs": attrs,
        }
    total_limit = _bounded(max_total_chars, minimum=2_000, maximum=MAX_GRAPH_TOTAL_CHARS)
    packed_edges: list[dict[str, Any]] = []
    packed_nodes: list[dict[str, Any]] = []
    for edge in selected_edges:
        candidate_edges = [*packed_edges, edge]
        related_ids = {node_id}
        for candidate in candidate_edges:
            related_ids.update((candidate.get("source"), candidate.get("target")))
        candidate_nodes = [
            value for key, value in compact_nodes_by_id.items() if key in related_ids
        ]
        payload = {"nodes": candidate_nodes, "edges": candidate_edges}
        if packed_edges and len(json.dumps(payload, ensure_ascii=False)) > total_limit:
            break
        packed_edges = candidate_edges
        packed_nodes = candidate_nodes
    return {
        "node_id": node_id,
        "total": total,
        "offset": offset,
        "returned": len(packed_edges),
        "has_more": offset + len(packed_edges) < total,
        "next_offset": offset + len(packed_edges),
        "max_total_chars": total_limit,
        "nodes": packed_nodes,
        "edges": packed_edges,
    }


def main() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
