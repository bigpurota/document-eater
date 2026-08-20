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
    description="Private local PDF ingestion, retrieval, and requirement audit tools.",
    instructions=(
        "Use these tools for private documents. Do not send returned document text to a "
        "non-local model unless the data owner explicitly permits it. UNKNOWN means that "
        "the available evidence is insufficient, not that a requirement failed."
    ),
)
_retrievers: dict[tuple[str, str, str], HybridRetriever] = {}


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
    output_path: str = "audit-run",
    use_qwen: bool = False,
    profile: str = "base",
    qwen_base_url: str | None = None,
    retrieval_mode: str = "quality",
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    embedding_cache: str = "models/retrieval",
) -> dict[str, Any]:
    """Audit one PDF or a directory: ingest, find requirements, retrieve evidence, write reports."""
    _release_retrieval_memory()
    try:
        report = audit_corpus(
            input_path,
            output_path,
            client=_client(profile, qwen_base_url) if use_qwen else None,
            retrieval_mode=retrieval_mode,  # type: ignore[arg-type]
            embedding_model=embedding_model,
            embedding_cache=embedding_cache,
        )
    finally:
        _release_retrieval_memory()
    return {
        "verification_mode": report.verification_mode,
        "retrieval_mode": report.retrieval_mode,
        "summary": report.summary,
        "requirements": len(report.items),
        "report_path": str(Path(report.run_directory) / "report.html"),
        "audit_path": str(Path(report.run_directory) / "audit.json"),
    }


@mcp.tool()
def prepare_corpus(
    input_path: str,
    workspace: str = "audit-run",
    retrieval_mode: str = "quality",
    embedding_cache: str = "models/retrieval",
) -> dict[str, Any]:
    """OCR and index a private PDF corpus locally so the agent can search it afterward."""
    _release_retrieval_memory()
    try:
        report = audit_corpus(
            input_path,
            workspace,
            client=None,
            retrieval_mode=retrieval_mode,  # type: ignore[arg-type]
            embedding_cache=embedding_cache,
        )
    finally:
        _release_retrieval_memory()
    return {
        "pdf_workspace": str(Path(report.run_directory).resolve()),
        "database_path": str(Path(report.run_directory).resolve() / "index.sqlite3"),
        "retrieval_mode": report.retrieval_mode,
        "requirement_candidates": len(report.items),
        "candidate_report": str(Path(report.run_directory).resolve() / "report.html"),
    }


@mcp.tool()
def search_corpus(
    query: str,
    database_path: str = "audit-run/index.sqlite3",
    limit: int = 10,
    mode: str = "quality",
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    embedding_cache: str = "models/retrieval",
) -> list[dict[str, Any]]:
    """Search the local document index and return page/block provenance with each result."""
    database_path = str(_resolve_run_file(database_path, "index.sqlite3"))
    if mode in {"quality", "hybrid"}:
        hits = _hybrid_retriever(database_path, embedding_model, embedding_cache, mode).search(
            query, limit=limit
        )
    elif mode == "lexical":
        hits = search(database_path, query, limit=limit)
    else:
        raise ValueError("mode must be 'quality', 'hybrid', or 'lexical'")
    return [asdict(hit) for hit in hits]


@mcp.tool()
def list_audit_items(
    audit_path: str = "audit-run/audit.json", status: str | None = None
) -> list[dict[str, Any]]:
    """Read requirement results from an existing audit, optionally filtered by status."""
    path = _resolve_run_file(audit_path, "audit.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    items = payload.get("items", [])
    if status:
        requested = status.upper()
        items = [item for item in items if item.get("status") == requested]
    return items


@mcp.tool()
def get_audit_summary(audit_path: str = "audit-run/audit.json") -> dict[str, Any]:
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
    artifacts_root: str = "audit-run/artifacts",
) -> dict[str, Any]:
    """Open one extracted page by document ID with exact block IDs and bounding boxes."""
    if page < 1:
        raise ValueError("page must be positive")
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
    manifest_path = path.with_name("manifest.json")
    manifest = (
        json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}
    )
    return {
        "document_id": document_id,
        "filename": document["filename"],
        "source_path": manifest.get("source_path"),
        "page": selected,
    }


@mcp.tool()
def graph_neighbors(
    node_id: str,
    artifacts_root: str = "audit-run/artifacts",
) -> dict[str, Any]:
    """Return structural graph edges touching a document/page/block node."""
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
    return {"node_id": node_id, "nodes": nodes, "edges": edges}


def main() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
