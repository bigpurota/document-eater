from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

_TOKEN = re.compile(r"[^\W_]+", re.UNICODE)


@dataclass(frozen=True)
class Chunk:
    id: str
    document_id: str
    page_start: int
    page_end: int
    heading: str
    text: str
    block_ids: list[str]
    location_start: str = ""
    location_end: str = ""


@dataclass(frozen=True)
class SearchHit:
    chunk_id: str
    document_id: str
    page_start: int
    page_end: int
    heading: str
    text: str
    block_ids: list[str]
    score: float
    location_start: str = ""
    location_end: str = ""
    retrieval_scores: dict[str, float] = field(default_factory=dict)


def format_hit_location(hit: SearchHit) -> str:
    if hit.location_start:
        location = hit.location_start
        if hit.location_end and hit.location_end != hit.location_start:
            location += f"–{hit.location_end}"
        return f"{hit.chunk_id} {location}"
    location = f"{hit.chunk_id} p.{hit.page_start}"
    if hit.page_end != hit.page_start:
        location += f"-{hit.page_end}"
    return location


def _connect(database: str | Path) -> sqlite3.Connection:
    path = Path(database).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA foreign_keys=ON")
    return connection


def _create_schema(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS chunks (
            id TEXT PRIMARY KEY,
            document_id TEXT NOT NULL,
            page_start INTEGER NOT NULL,
            page_end INTEGER NOT NULL,
            heading TEXT NOT NULL,
            text TEXT NOT NULL,
            block_ids TEXT NOT NULL,
            location_start TEXT NOT NULL DEFAULT '',
            location_end TEXT NOT NULL DEFAULT ''
        );
        CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
            chunk_id UNINDEXED,
            heading,
            text,
            tokenize='unicode61 remove_diacritics 2'
        );
        CREATE TABLE IF NOT EXISTS indexed_documents (
            document_id TEXT PRIMARY KEY,
            source_sha256 TEXT NOT NULL,
            indexed_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );
        """
    )
    columns = {row[1] for row in connection.execute("PRAGMA table_info(chunks)")}
    if "location_start" not in columns:
        connection.execute("ALTER TABLE chunks ADD COLUMN location_start TEXT NOT NULL DEFAULT ''")
    if "location_end" not in columns:
        connection.execute("ALTER TABLE chunks ADD COLUMN location_end TEXT NOT NULL DEFAULT ''")


def _chunks_from_document(document: dict, max_chars: int = 1800) -> Iterable[Chunk]:
    doc_id = document["id"]
    heading = ""
    pending: list[dict] = []
    length = 0
    sequence = 0

    def flush() -> Chunk | None:
        nonlocal pending, length, sequence
        if not pending:
            return None
        sequence += 1
        result = Chunk(
            id=f"{doc_id}:c{sequence}",
            document_id=doc_id,
            page_start=pending[0]["page"],
            page_end=pending[-1]["page"],
            heading=heading,
            text="\n\n".join(block["text"] for block in pending),
            block_ids=[block["id"] for block in pending],
            location_start=pending[0]["_location"],
            location_end=pending[-1]["_location"],
        )
        pending = []
        length = 0
        return result

    for page in document["pages"]:
        for raw_block in page["blocks"]:
            block = dict(raw_block)
            block["_location"] = str(
                block.get("attrs", {}).get("location") or page.get("label") or f"p.{page['number']}"
            )
            if block.get("role") == "heading":
                ready = flush()
                if ready:
                    yield ready
                heading = block["text"]
                pending.append(block)
                length = len(block["text"])
                continue
            projected = length + len(block["text"]) + 2
            if pending and projected > max_chars:
                ready = flush()
                if ready:
                    yield ready
            pending.append(block)
            length += len(block["text"]) + 2
    ready = flush()
    if ready:
        yield ready


def index_artifacts(
    artifacts: str | Path,
    database: str | Path,
    *,
    max_chars: int = 1800,
    reset: bool = False,
) -> dict[str, int]:
    root = Path(artifacts).expanduser().resolve()
    documents = sorted(root.glob("*/document.json"))
    indexed_documents = 0
    indexed_chunks = 0
    with _connect(database) as connection:
        if reset:
            connection.executescript(
                """
                DROP TABLE IF EXISTS learned_sparse_embeddings;
                DROP TABLE IF EXISTS dense_embeddings;
                DROP TABLE IF EXISTS retrieval_metadata;
                DROP TABLE IF EXISTS chunks_fts;
                DROP TABLE IF EXISTS chunks;
                DROP TABLE IF EXISTS indexed_documents;
                """
            )
        _create_schema(connection)
        for document_path in documents:
            document = json.loads(document_path.read_text(encoding="utf-8"))
            doc_id = document["id"]
            connection.execute(
                "DELETE FROM chunks_fts WHERE chunk_id IN (SELECT id FROM chunks WHERE document_id = ?)",
                (doc_id,),
            )
            connection.execute("DELETE FROM chunks WHERE document_id = ?", (doc_id,))
            for chunk in _chunks_from_document(document, max_chars=max_chars):
                connection.execute(
                    """INSERT INTO chunks(
                        id, document_id, page_start, page_end, heading, text, block_ids,
                        location_start, location_end
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        chunk.id,
                        chunk.document_id,
                        chunk.page_start,
                        chunk.page_end,
                        chunk.heading,
                        chunk.text,
                        json.dumps(chunk.block_ids),
                        chunk.location_start,
                        chunk.location_end,
                    ),
                )
                connection.execute(
                    "INSERT INTO chunks_fts(chunk_id, heading, text) VALUES (?, ?, ?)",
                    (chunk.id, chunk.heading, chunk.text),
                )
                indexed_chunks += 1
            connection.execute(
                "INSERT OR REPLACE INTO indexed_documents(document_id, source_sha256) VALUES (?, ?)",
                (doc_id, document["sha256"]),
            )
            indexed_documents += 1
    return {"documents": indexed_documents, "chunks": indexed_chunks}


def _fts_query(query: str) -> str:
    tokens = _TOKEN.findall(query.casefold())
    if not tokens:
        raise ValueError("Search query contains no searchable terms")
    return " OR ".join(f'"{token.replace(chr(34), chr(34) * 2)}"' for token in tokens)


def search(database: str | Path, query: str, *, limit: int = 10) -> list[SearchHit]:
    if limit < 1:
        raise ValueError("limit must be positive")
    statement = """
        SELECT c.*, bm25(chunks_fts, 0.0, 1.6, 1.0) AS rank
        FROM chunks_fts
        JOIN chunks c ON c.id = chunks_fts.chunk_id
        WHERE chunks_fts MATCH ?
        ORDER BY rank
        LIMIT ?
    """
    with _connect(database) as connection:
        _create_schema(connection)
        rows = connection.execute(statement, (_fts_query(query), limit)).fetchall()
    return [
        SearchHit(
            chunk_id=row["id"],
            document_id=row["document_id"],
            page_start=row["page_start"],
            page_end=row["page_end"],
            heading=row["heading"],
            text=row["text"],
            block_ids=json.loads(row["block_ids"]),
            score=-float(row["rank"]),
            location_start=row["location_start"],
            location_end=row["location_end"],
            retrieval_scores={"bm25": -float(row["rank"])},
        )
        for row in rows
    ]


def list_chunks(database: str | Path) -> list[Chunk]:
    """Load indexed chunks in stable order for embedding and dense retrieval."""
    with _connect(database) as connection:
        _create_schema(connection)
        rows = connection.execute("SELECT * FROM chunks ORDER BY id").fetchall()
    return [
        Chunk(
            id=row["id"],
            document_id=row["document_id"],
            page_start=row["page_start"],
            page_end=row["page_end"],
            heading=row["heading"],
            text=row["text"],
            block_ids=json.loads(row["block_ids"]),
            location_start=row["location_start"],
            location_end=row["location_end"],
        )
        for row in rows
    ]
