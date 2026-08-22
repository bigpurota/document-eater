from __future__ import annotations

import json

from document_eater import mcp_server
from document_eater.index import SearchHit


def test_search_corpus_bounds_count_and_text(monkeypatch, tmp_path):
    database = tmp_path / "index.sqlite3"
    database.write_bytes(b"placeholder")
    hits = [
        SearchHit(
            chunk_id=f"doc:c{index}",
            document_id="doc",
            page_start=index,
            page_end=index,
            heading="Heading",
            text="x" * 3_000,
            block_ids=[f"doc:p{index}:b1"],
            score=1.0,
        )
        for index in range(1, 11)
    ]
    monkeypatch.setattr(mcp_server, "search", lambda *_args, **_kwargs: hits)

    result = mcp_server.search_corpus(
        "requirement",
        database_path=str(database),
        limit=99,
        mode="lexical",
        max_chars_per_hit=900,
    )

    assert 1 <= result["returned"] <= mcp_server.MAX_SEARCH_RESULTS
    assert len(result["items"]) == result["returned"]
    assert len(json.dumps(result["items"])) <= result["max_total_chars"]
    assert all(len(item["text"]) <= 900 for item in result["items"])
    assert all(item["text_truncated"] is True for item in result["items"])


def test_list_audit_items_is_paginated_and_compact_by_default(tmp_path):
    audit = tmp_path / "audit.json"
    items = [
        {
            "status": "UNKNOWN",
            "requirement": {"id": f"REQ-{index}", "text": "A requirement"},
            "rationale": "No direct evidence",
            "used_citations": [],
            "retrieved_evidence": [{"preview": "secret" * 1_000}],
            "model": None,
        }
        for index in range(8)
    ]
    audit.write_text(json.dumps({"items": items}), encoding="utf-8")

    result = mcp_server.list_audit_items(str(audit), offset=2, limit=3)

    assert result["total"] == 8
    assert result["returned"] == 3
    assert result["has_more"] is True
    assert [item["requirement"]["id"] for item in result["items"]] == [
        "REQ-2",
        "REQ-3",
        "REQ-4",
    ]
    assert all("retrieved_evidence" not in item for item in result["items"])
    assert len(json.dumps(result["items"])) <= result["max_total_chars"]


def test_read_document_page_uses_character_windows(tmp_path):
    artifacts = tmp_path / "artifacts"
    document_dir = artifacts / "doc"
    document_dir.mkdir(parents=True)
    (document_dir / "document.json").write_text(
        json.dumps(
            {
                "id": "doc",
                "filename": "private.txt",
                "pages": [
                    {
                        "number": 1,
                        "label": "unit 1",
                        "kind": "text",
                        "source": "private.txt",
                        "blocks": [
                            {
                                "id": "doc:p1:b1",
                                "page": 1,
                                "order": 1,
                                "role": "body",
                                "source": "private.txt",
                                "text": "a" * 3_000,
                                "attrs": {"location": "line 1"},
                            },
                            {
                                "id": "doc:p1:b2",
                                "page": 1,
                                "order": 2,
                                "role": "body",
                                "source": "private.txt",
                                "text": "b" * 3_000,
                                "attrs": {"location": "line 2"},
                            },
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    (document_dir / "manifest.json").write_text(
        json.dumps({"source_path": "/private/private.txt"}), encoding="utf-8"
    )

    first = mcp_server.read_document_page(
        "doc", 1, artifacts_root=str(artifacts), offset=0, max_chars=1_000
    )
    second = mcp_server.read_document_page(
        "doc",
        1,
        artifacts_root=str(artifacts),
        offset=first["next_offset"],
        max_chars=1_000,
    )

    assert sum(len(block["text"]) for block in first["page"]["blocks"]) <= 1_000
    assert first["truncated"] is True
    assert first["next_offset"] == 1_000
    assert second["offset"] == 1_000


def test_graph_neighbors_packs_edges_under_total_budget(tmp_path):
    artifacts = tmp_path / "artifacts"
    document_dir = artifacts / "doc"
    document_dir.mkdir(parents=True)
    nodes = [
        {"id": "heading", "kind": "block", "attrs": {"text": "Heading"}},
        *[
            {
                "id": f"block-{index}",
                "kind": "block",
                "attrs": {"text": "x" * 2_000, "location": f"line {index}"},
            }
            for index in range(30)
        ],
    ]
    edges = [
        {"source": f"block-{index}", "target": "heading", "kind": "in_section"}
        for index in range(30)
    ]
    (document_dir / "graph.json").write_text(
        json.dumps({"nodes": nodes, "edges": edges}), encoding="utf-8"
    )

    result = mcp_server.graph_neighbors(
        "heading",
        artifacts_root=str(artifacts),
        limit=99,
        max_total_chars=4_000,
    )

    packed = {"nodes": result["nodes"], "edges": result["edges"]}
    assert 1 <= result["returned"] <= mcp_server.MAX_GRAPH_EDGE_LIMIT
    assert len(json.dumps(packed)) <= result["max_total_chars"]
    assert result["has_more"] is True
