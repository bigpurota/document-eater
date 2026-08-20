from __future__ import annotations

import json

import pymupdf

from document_eater.audit import audit_corpus, extract_requirements, verify_requirement
from document_eater.index import SearchHit


def _pdf(path, lines: list[str]) -> None:
    document = pymupdf.open()
    page = document.new_page(width=595, height=842)
    y = 80
    for line in lines:
        page.insert_text((72, y), line, fontsize=11)
        y += 32
    document.save(path)
    document.close()


def test_audit_builds_human_and_machine_reports(tmp_path):
    source = tmp_path / "private"
    source.mkdir()
    _pdf(
        source / "requirements.pdf",
        [
            "1 Requirements",
            "The supplier must submit the signed acceptance certificate.",
            "The report shall contain the completion date.",
        ],
    )
    _pdf(
        source / "evidence.pdf",
        [
            "2 Evidence",
            "The signed acceptance certificate was submitted by the supplier.",
            "The report contains the completion date 2026-08-20.",
        ],
    )

    run = tmp_path / "run"
    report = audit_corpus(source, run, ocr="never", languages="eng", retrieval_mode="lexical")

    assert len(report.items) == 2
    assert report.summary["UNKNOWN"] == 2
    assert (run / "report.html").is_file()
    assert (run / "requirements.csv").is_file()
    payload = json.loads((run / "audit.json").read_text(encoding="utf-8"))
    assert payload["verification_mode"] == "candidate_only"
    assert payload["items"][0]["requirement"]["source_path"].endswith("requirements.pdf")


def test_qwen_verdict_requires_a_valid_evidence_citation(tmp_path):
    source = tmp_path / "requirement.pdf"
    _pdf(source, ["The supplier must submit the signed acceptance certificate."])
    run = tmp_path / "run"
    audit_corpus(source, run, ocr="never", languages="eng", retrieval_mode="lexical")
    requirement = extract_requirements(run / "artifacts")[0]
    hit = SearchHit(
        chunk_id="evidence:c1",
        document_id="evidence",
        page_start=2,
        page_end=2,
        heading="Evidence",
        text="The signed acceptance certificate was submitted.",
        block_ids=["evidence:p2:b1"],
        score=1.0,
    )

    class FakeClient:
        model = "fake-qwen"
        use_system_prompt = True

        def chat(self, messages):
            return json.dumps(
                {
                    "status": "PASS",
                    "rationale": "Есть прямое подтверждение.",
                    "citations": ["evidence:c1 p.2"],
                }
            )

    item = verify_requirement(requirement, [hit], FakeClient())
    assert item.status == "PASS"
    assert item.used_citations == ["evidence:c1 p.2"]


def test_uncited_qwen_verdict_is_downgraded_to_unknown(tmp_path):
    source = tmp_path / "requirement.pdf"
    _pdf(source, ["The supplier must submit the signed acceptance certificate."])
    run = tmp_path / "run"
    audit_corpus(source, run, ocr="never", languages="eng", retrieval_mode="lexical")
    requirement = extract_requirements(run / "artifacts")[0]

    class FakeClient:
        model = "fake-qwen"
        use_system_prompt = True

        def chat(self, messages):
            return '{"status":"FAIL","rationale":"No proof","citations":[]}'

    item = verify_requirement(requirement, [], FakeClient())
    assert item.status == "UNKNOWN"


def test_reusing_output_does_not_mix_old_corpus_into_new_audit(tmp_path):
    first = tmp_path / "first.pdf"
    second = tmp_path / "second.pdf"
    _pdf(first, ["The supplier must submit one signed certificate."])
    _pdf(second, ["The client shall approve a different final report."])
    run = tmp_path / "run"

    audit_corpus(first, run, ocr="never", languages="eng", retrieval_mode="lexical")
    report = audit_corpus(second, run, ocr="never", languages="eng", retrieval_mode="lexical")

    assert len(report.items) == 1
    assert report.items[0].requirement.filename == "second.pdf"
