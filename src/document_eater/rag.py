from __future__ import annotations

import json
import os
import sqlite3
from dataclasses import replace
from pathlib import Path
from typing import Callable, Protocol, Sequence

import numpy as np

from .index import Chunk, SearchHit, list_chunks, search

DEFAULT_EMBEDDING_MODEL = "intfloat/multilingual-e5-large"
DEFAULT_MODEL_CACHE = Path("models/retrieval")
QUALITY_EMBEDDING_MODEL = "BAAI/bge-m3"
QUALITY_EMBEDDING_REVISION = "5617a9f61b028005a4858fdac845db406aefb181"
QUALITY_RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
QUALITY_RERANKER_REVISION = "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"


class Encoder(Protocol):
    model_name: str

    def encode_documents(self, texts: Sequence[str]) -> np.ndarray: ...

    def encode_query(self, text: str) -> np.ndarray: ...


class Reranker(Protocol):
    model_name: str

    def score(self, query: str, documents: Sequence[str]) -> np.ndarray: ...


class FastEmbedEncoder:
    """Lazy local ONNX encoder. Model files never need a hosted inference API."""

    def __init__(
        self,
        model_name: str = DEFAULT_EMBEDDING_MODEL,
        cache_dir: str | Path = DEFAULT_MODEL_CACHE,
    ) -> None:
        os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
        from fastembed import TextEmbedding

        self.model_name = model_name
        cache = Path(cache_dir).expanduser().resolve()
        cache.mkdir(parents=True, exist_ok=True)
        self._model = TextEmbedding(model_name=model_name, cache_dir=str(cache))

    def encode_documents(self, texts: Sequence[str]) -> np.ndarray:
        vectors = list(self._model.passage_embed(texts))
        return _normalize(np.asarray(vectors, dtype=np.float32))

    def encode_query(self, text: str) -> np.ndarray:
        vector = np.asarray(list(self._model.query_embed(text))[0], dtype=np.float32)
        return _normalize(vector.reshape(1, -1))[0]


class BgeM3Encoder:
    """High-quality BGE-M3 dense + learned-sparse encoder pinned to one revision."""

    def __init__(self, cache_dir: str | Path = "models/retrieval") -> None:
        os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
        import torch
        from FlagEmbedding import BGEM3FlagModel

        cache = Path(cache_dir).expanduser().resolve()
        cache.mkdir(parents=True, exist_ok=True)
        use_mps = bool(torch.backends.mps.is_available())
        self.model_name = QUALITY_EMBEDDING_MODEL
        self._model = BGEM3FlagModel(
            QUALITY_EMBEDDING_MODEL,
            revision=QUALITY_EMBEDDING_REVISION,
            cache_dir=str(cache),
            devices="mps" if use_mps else None,
            use_fp16=use_mps,
            passage_max_length=1024,
            query_max_length=512,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False,
        )

    def encode_documents_hybrid(
        self, texts: Sequence[str]
    ) -> tuple[np.ndarray, list[dict[str, float]]]:
        result = self._model.encode(
            list(texts),
            max_length=1024,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False,
        )
        dense = _normalize(np.asarray(result["dense_vecs"], dtype=np.float32))
        sparse = [
            {str(key): float(value) for key, value in weights.items()}
            for weights in result["lexical_weights"]
        ]
        return dense, sparse

    def encode_query_hybrid(self, text: str) -> tuple[np.ndarray, dict[str, float]]:
        dense, sparse = self.encode_documents_hybrid([text])
        return dense[0], sparse[0]

    def encode_documents(self, texts: Sequence[str]) -> np.ndarray:
        return self.encode_documents_hybrid(texts)[0]

    def encode_query(self, text: str) -> np.ndarray:
        return self.encode_query_hybrid(text)[0]


class BgeM3Reranker:
    """Pinned multilingual cross-encoder, loaded only for the final candidate pool."""

    def __init__(self, cache_dir: str | Path = "models/retrieval") -> None:
        os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
        import torch
        from FlagEmbedding import FlagReranker

        cache = Path(cache_dir).expanduser().resolve()
        cache.mkdir(parents=True, exist_ok=True)
        use_mps = bool(torch.backends.mps.is_available())
        self.model_name = QUALITY_RERANKER_MODEL
        self._model = FlagReranker(
            QUALITY_RERANKER_MODEL,
            revision=QUALITY_RERANKER_REVISION,
            cache_dir=str(cache),
            devices="mps" if use_mps else None,
            use_fp16=use_mps,
            max_length=1024,
            normalize=True,
        )

    def score(self, query: str, documents: Sequence[str]) -> np.ndarray:
        pairs = [(query, document) for document in documents]
        values = self._model.compute_score(pairs, normalize=True, max_length=1024)
        return np.atleast_1d(np.asarray(values, dtype=np.float32))


def _normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.maximum(norms, 1e-12)


def _connect(database: str | Path) -> sqlite3.Connection:
    connection = sqlite3.connect(Path(database).expanduser().resolve())
    connection.row_factory = sqlite3.Row
    return connection


def _create_dense_schema(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS dense_embeddings (
            chunk_id TEXT PRIMARY KEY REFERENCES chunks(id) ON DELETE CASCADE,
            model_name TEXT NOT NULL,
            dimensions INTEGER NOT NULL,
            vector BLOB NOT NULL
        );
        CREATE TABLE IF NOT EXISTS retrieval_metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS learned_sparse_embeddings (
            chunk_id TEXT PRIMARY KEY REFERENCES chunks(id) ON DELETE CASCADE,
            model_name TEXT NOT NULL,
            weights TEXT NOT NULL
        );
        """
    )


def index_dense(
    database: str | Path,
    encoder: Encoder,
    *,
    batch_size: int = 16,
    progress: Callable[[str], None] | None = None,
) -> dict[str, int | str]:
    chunks = list_chunks(database)
    notify = progress or (lambda _message: None)
    dimensions = 0
    with _connect(database) as connection:
        connection.execute("PRAGMA foreign_keys=ON")
        _create_dense_schema(connection)
        connection.execute("DELETE FROM dense_embeddings")
        connection.execute("DELETE FROM learned_sparse_embeddings")
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start : start + batch_size]
            notify(f"Dense embeddings: {min(start + len(batch), len(chunks))}/{len(chunks)}")
            texts = [chunk.text for chunk in batch]
            if hasattr(encoder, "encode_documents_hybrid"):
                vectors, sparse_weights = encoder.encode_documents_hybrid(texts)  # type: ignore[attr-defined]
            else:
                vectors = encoder.encode_documents(texts)
                sparse_weights = None
            if len(vectors) != len(batch):
                raise RuntimeError("Embedding model returned the wrong vector count")
            if len(vectors):
                dimensions = int(vectors.shape[1])
            for offset, (chunk, vector) in enumerate(zip(batch, vectors, strict=True)):
                raw = np.asarray(vector, dtype=np.float32)
                connection.execute(
                    "INSERT INTO dense_embeddings VALUES (?, ?, ?, ?)",
                    (chunk.id, encoder.model_name, int(raw.size), raw.tobytes()),
                )
                if sparse_weights is not None:
                    connection.execute(
                        "INSERT INTO learned_sparse_embeddings VALUES (?, ?, ?)",
                        (
                            chunk.id,
                            encoder.model_name,
                            json.dumps(sparse_weights[offset], separators=(",", ":")),
                        ),
                    )
        connection.execute(
            "INSERT OR REPLACE INTO retrieval_metadata VALUES ('embedding_model', ?)",
            (encoder.model_name,),
        )
    return {"chunks": len(chunks), "dimensions": dimensions, "model": encoder.model_name}


def _hit_from_chunk(chunk: Chunk, score: float, scores: dict[str, float]) -> SearchHit:
    return SearchHit(
        chunk_id=chunk.id,
        document_id=chunk.document_id,
        page_start=chunk.page_start,
        page_end=chunk.page_end,
        heading=chunk.heading,
        text=chunk.text,
        block_ids=chunk.block_ids,
        score=score,
        retrieval_scores=scores,
    )


class HybridRetriever:
    """BM25 + dense retrieval fused by reciprocal rank fusion (RRF)."""

    def __init__(
        self, database: str | Path, encoder: Encoder, reranker: Reranker | None = None
    ) -> None:
        self.database = Path(database).expanduser().resolve()
        self.encoder = encoder
        self.reranker = reranker
        chunks = list_chunks(self.database)
        by_id = {chunk.id: chunk for chunk in chunks}
        ids: list[str] = []
        vectors: list[np.ndarray] = []
        with _connect(self.database) as connection:
            _create_dense_schema(connection)
            rows = connection.execute(
                "SELECT chunk_id, model_name, dimensions, vector FROM dense_embeddings ORDER BY chunk_id"
            ).fetchall()
            sparse_rows = connection.execute(
                "SELECT chunk_id, model_name, weights FROM learned_sparse_embeddings"
            ).fetchall()
        for row in rows:
            if row["model_name"] != encoder.model_name or row["chunk_id"] not in by_id:
                continue
            vector = np.frombuffer(row["vector"], dtype=np.float32)
            if vector.size != row["dimensions"]:
                raise RuntimeError(f"Corrupt dense vector for {row['chunk_id']}")
            ids.append(row["chunk_id"])
            vectors.append(vector)
        if chunks and not vectors:
            raise RuntimeError(
                "Dense index is missing for this embedding model. Re-index with hybrid mode."
            )
        self._chunks = by_id
        self._ids = ids
        self._matrix = np.vstack(vectors) if vectors else np.empty((0, 0), dtype=np.float32)
        self._sparse = {
            row["chunk_id"]: {
                str(key): float(value) for key, value in json.loads(row["weights"]).items()
            }
            for row in sparse_rows
            if row["model_name"] == encoder.model_name and row["chunk_id"] in by_id
        }

    def search(
        self,
        query: str,
        *,
        limit: int = 10,
        candidate_pool: int = 40,
        rerank_pool: int = 20,
    ) -> list[SearchHit]:
        if limit < 1:
            raise ValueError("limit must be positive")
        lexical = search(self.database, query, limit=candidate_pool)
        lexical_rank = {hit.chunk_id: rank for rank, hit in enumerate(lexical, 1)}
        lexical_hits = {hit.chunk_id: hit for hit in lexical}

        if self._sparse and hasattr(self.encoder, "encode_query_hybrid"):
            query_vector, query_sparse = self.encoder.encode_query_hybrid(query)  # type: ignore[attr-defined]
        else:
            query_vector = self.encoder.encode_query(query)
            query_sparse = {}
        if self._matrix.size and query_vector.size != self._matrix.shape[1]:
            raise RuntimeError("Query embedding dimensions do not match the dense index")
        dense_scores = self._matrix @ query_vector if self._matrix.size else np.array([])
        dense_order = np.argsort(-dense_scores)[:candidate_pool]
        dense_rank = {self._ids[int(index)]: rank for rank, index in enumerate(dense_order, 1)}
        raw_dense = {
            self._ids[int(index)]: float(dense_scores[int(index)]) for index in dense_order
        }

        sparse_scores = {
            chunk_id: sum(
                query_sparse.get(token, 0.0) * weight for token, weight in weights.items()
            )
            for chunk_id, weights in self._sparse.items()
        }
        sparse_order = [
            chunk_id
            for chunk_id in sorted(sparse_scores, key=sparse_scores.get, reverse=True)
            if sparse_scores[chunk_id] > 0
        ][:candidate_pool]
        sparse_rank = {chunk_id: rank for rank, chunk_id in enumerate(sparse_order, 1)}

        candidate_ids = set(lexical_rank) | set(dense_rank) | set(sparse_rank)
        fused = []
        for chunk_id in candidate_ids:
            rrf = 0.0
            if chunk_id in lexical_rank:
                rrf += 1.0 / (60 + lexical_rank[chunk_id])
            if chunk_id in dense_rank:
                rrf += 1.0 / (60 + dense_rank[chunk_id])
            if chunk_id in sparse_rank:
                rrf += 1.0 / (60 + sparse_rank[chunk_id])
            bm25 = lexical_hits.get(chunk_id)
            scores = {
                "rrf": rrf,
                "dense_cosine": raw_dense.get(chunk_id, 0.0),
                "learned_sparse": sparse_scores.get(chunk_id, 0.0),
                "bm25": bm25.retrieval_scores.get("bm25", 0.0) if bm25 else 0.0,
            }
            fused.append(_hit_from_chunk(self._chunks[chunk_id], rrf, scores))
        fused.sort(key=lambda hit: (-hit.score, hit.chunk_id))
        if self.reranker and fused:
            pool = fused[: max(limit, rerank_pool)]
            scores = self.reranker.score(query, [hit.text for hit in pool])
            if len(scores) != len(pool):
                raise RuntimeError("Reranker returned the wrong score count")
            reranked = [
                replace(
                    hit,
                    score=float(score),
                    retrieval_scores={
                        **hit.retrieval_scores,
                        "reranker": float(score),
                    },
                )
                for hit, score in zip(pool, scores, strict=True)
            ]
            reranked.sort(key=lambda hit: (-hit.score, hit.chunk_id))
            return reranked[:limit]
        return fused[:limit]
