from __future__ import annotations

import io
import json
import shutil
import urllib.error

import pymupdf
import pytest

from document_eater.index import index_artifacts, search
from document_eater.llm import QwenClient, answer_question, build_evidence
from document_eater.pdf import ingest_pdf, inspect_pdf


def _digital_pdf(path) -> None:
    document = pymupdf.open()
    page = document.new_page(width=595, height=842)
    page.insert_text((72, 90), "1 Introduction", fontsize=18)
    page.insert_text(
        (72, 130),
        "This document contains enough native text for reliable extraction and testing.",
        fontsize=11,
    )
    document.save(path)
    document.close()


def _scanned_pdf(path, temp_path) -> None:
    _digital_pdf(temp_path)
    source = pymupdf.open(temp_path)
    pix = source[0].get_pixmap(dpi=220, alpha=False)
    source.close()
    scan = pymupdf.open()
    page = scan.new_page(width=595, height=842)
    page.insert_image(page.rect, stream=pix.tobytes("png"))
    scan.save(path)
    scan.close()


def test_native_pdf_keeps_provenance_and_graph(tmp_path):
    source = tmp_path / "digital.pdf"
    _digital_pdf(source)

    report = inspect_pdf(source)
    assert report["pages"][0]["recommended_source"] == "native"

    destination = ingest_pdf(source, tmp_path / "out")
    document = json.loads((destination / "document.json").read_text())
    graph = json.loads((destination / "graph.json").read_text())
    manifest = json.loads((destination / "manifest.json").read_text())

    assert document["pages"][0]["source"] == "native"
    assert "Introduction" in document["pages"][0]["blocks"][0]["text"]
    assert any(edge["kind"] == "next" for edge in graph["edges"])
    assert manifest["source_pdf_copied"] is False
    assert not (destination / source.name).exists()


@pytest.mark.skipif(not shutil.which("tesseract"), reason="tesseract is not installed")
def test_scanned_pdf_uses_ocr(tmp_path):
    source = tmp_path / "scan.pdf"
    _scanned_pdf(source, tmp_path / "source.pdf")

    destination = ingest_pdf(source, tmp_path / "out", languages="eng", dpi=220)
    document = json.loads((destination / "document.json").read_text())
    text = " ".join(block["text"] for block in document["pages"][0]["blocks"])

    assert document["pages"][0]["source"] == "ocr"
    assert "Introduction" in text


def test_local_index_returns_page_and_block_citations(tmp_path):
    source = tmp_path / "digital.pdf"
    _digital_pdf(source)
    artifacts = tmp_path / "artifacts"
    ingest_pdf(source, artifacts)

    stats = index_artifacts(artifacts, tmp_path / "index.sqlite3")
    hits = search(tmp_path / "index.sqlite3", "reliable extraction")

    assert stats == {"documents": 1, "chunks": 1}
    assert len(hits) == 1
    assert hits[0].page_start == 1
    assert hits[0].block_ids
    assert "reliable extraction" in hits[0].text

    evidence, citations = build_evidence(hits)
    assert f"[SOURCE {hits[0].chunk_id} p.1]" in evidence
    assert citations[0]["block_ids"] == hits[0].block_ids


def test_qwen_client_requires_local_or_tunnelled_endpoint():
    with pytest.raises(ValueError, match="non-loopback"):
        QwenClient("https://public.example.com/v1", "qwen")
    client = QwenClient("http://127.0.0.1:8080/v1", "qwen")
    assert client.base_url == "http://127.0.0.1:8080/v1"
    with pytest.raises(ValueError, match="must use HTTPS"):
        QwenClient(
            "http://inference.example.com/v1",
            "qwen",
            allow_nonlocal_endpoint=True,
        )


def test_qwen_client_remote_opt_in_sends_bearer_key(monkeypatch):
    captured = {}

    def fake_urlopen(request, timeout):
        captured["authorization"] = request.get_header("Authorization")
        captured["timeout"] = timeout
        return io.BytesIO(b'{"choices":[{"message":{"content":"ok"}}]}')

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    client = QwenClient(
        "https://inference.example.com/v1",
        "Qwen/Qwen3.8-27B",
        allow_nonlocal_endpoint=True,
        api_key="secret-token",
        timeout_seconds=45,
    )

    assert client.chat([{"role": "user", "content": "test"}]) == "ok"
    assert captured == {"authorization": "Bearer secret-token", "timeout": 45}


def test_qwen_client_retries_serverless_rate_limit(monkeypatch):
    calls = []

    def fake_urlopen(request, timeout):
        calls.append((request.full_url, timeout))
        if len(calls) == 1:
            raise urllib.error.HTTPError(request.full_url, 429, "busy", {}, None)
        return io.BytesIO(b'{"choices":[{"message":{"content":"recovered"}}]}')

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    monkeypatch.setattr("time.sleep", lambda _seconds: None)
    client = QwenClient(
        "https://inference.example.com/v1",
        "Qwen/Qwen3.8-27B",
        allow_nonlocal_endpoint=True,
        max_retries=2,
    )

    assert client.chat([{"role": "user", "content": "test"}]) == "recovered"
    assert len(calls) == 2


def test_base_and_abliterated_prompt_profiles_are_explicit(tmp_path):
    source = tmp_path / "digital.pdf"
    _digital_pdf(source)
    artifacts = tmp_path / "artifacts"
    database = tmp_path / "index.sqlite3"
    ingest_pdf(source, artifacts)
    index_artifacts(artifacts, database)

    class RecordingClient:
        model = "recording-model"

        def __init__(self, use_system_prompt):
            self.use_system_prompt = use_system_prompt
            self.messages = None

        def chat(self, messages):
            self.messages = messages
            return "recorded"

    base = RecordingClient(use_system_prompt=True)
    answer_question(str(database), "reliable extraction", base)
    assert [message["role"] for message in base.messages] == ["system", "user"]

    fallback = RecordingClient(use_system_prompt=False)
    answer_question(str(database), "reliable extraction", fallback)
    assert [message["role"] for message in fallback.messages] == ["user"]
