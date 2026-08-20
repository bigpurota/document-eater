from __future__ import annotations

import numpy as np
import pymupdf

from document_eater.index import index_artifacts
from document_eater.pdf import ingest_pdf
from document_eater.rag import HybridRetriever, index_dense


class FakeEncoder:
    model_name = "fake-multilingual"

    def _vector(self, text: str) -> np.ndarray:
        lowered = text.casefold()
        if any(word in lowered for word in ("automobile", "vehicle", "машина")):
            return np.array([1.0, 0.0], dtype=np.float32)
        return np.array([0.0, 1.0], dtype=np.float32)

    def encode_documents(self, texts):
        return np.stack([self._vector(text) for text in texts])

    def encode_query(self, text):
        return self._vector(text)


def _pdf(path, text):
    document = pymupdf.open()
    page = document.new_page()
    page.insert_text((72, 90), text, fontsize=11)
    document.save(path)
    document.close()


def test_hybrid_retrieval_recovers_semantic_match_missing_from_bm25(tmp_path):
    artifacts = tmp_path / "artifacts"
    _pdf(tmp_path / "cars.pdf", "The automobile inspection was completed successfully.")
    _pdf(tmp_path / "weather.pdf", "Rain and clouds are expected tomorrow morning.")
    ingest_pdf(tmp_path / "cars.pdf", artifacts, ocr="never")
    ingest_pdf(tmp_path / "weather.pdf", artifacts, ocr="never")
    database = tmp_path / "index.sqlite3"
    index_artifacts(artifacts, database)
    encoder = FakeEncoder()
    index_dense(database, encoder)

    hits = HybridRetriever(database, encoder).search("vehicle", limit=2)

    assert "automobile" in hits[0].text
    assert hits[0].retrieval_scores["dense_cosine"] == 1.0
