from __future__ import annotations

import io
import json

import pytest

from document_eater.cli import _parser
from document_eater.tui import DocumentTUI, Terminal, TUISettings, is_remote_endpoint


def _terminal(answers: list[str]):
    values = iter(answers)
    output = io.StringIO()
    terminal = Terminal(input_fn=lambda _prompt: next(values), output=output, color=False)
    return terminal, output


def test_tui_starts_in_document_folder_and_exits(tmp_path):
    terminal, output = _terminal(["0"])
    settings = TUISettings(
        source=tmp_path,
        workspace=tmp_path / ".document-eater-workspace",
        strict_offline=False,
        color=False,
    )

    DocumentTUI(settings, terminal).run()

    rendered = output.getvalue()
    assert "DOCUMENT EATER" in rendered
    assert "Проверить требования через Qwen" in rendered
    assert "корпус ещё не подготовлен" in rendered


def test_remote_endpoint_requires_explicit_tui_confirmation(tmp_path):
    settings = TUISettings(
        source=tmp_path,
        workspace=tmp_path / ".document-eater-workspace",
        strict_offline=False,
    )
    terminal, output = _terminal(["https://inference.example.com/v1", "no"])
    app = DocumentTUI(settings, terminal)

    app._configure_endpoint()

    assert settings.remote is False
    assert settings.allow_remote is False
    assert "отменено" in output.getvalue()

    terminal, _output = _terminal(["https://inference.example.com/v1", "REMOTE"])
    app.terminal = terminal
    app._configure_endpoint()
    assert settings.remote is True
    assert settings.allow_remote is True


def test_tui_cli_exposes_lean_remote_profile(tmp_path, monkeypatch):
    cache = tmp_path / "shared-model-cache"
    monkeypatch.setenv("DOCUMENT_EATER_MODEL_CACHE", str(cache))
    args = _parser().parse_args(
        [
            "tui",
            str(tmp_path),
            "--base-url",
            "https://inference.example.com/v1",
            "--model",
            "Qwen/Qwen3.8-27B",
            "--allow-remote",
            "--retrieval",
            "hybrid",
        ]
    )

    assert args.command == "tui"
    assert args.allow_remote is True
    assert args.model == "Qwen/Qwen3.8-27B"
    assert args.retrieval == "hybrid"
    assert args.embedding_cache == cache
    assert is_remote_endpoint(args.base_url) is True


def test_strict_offline_tui_refuses_remote_endpoint(tmp_path):
    settings = TUISettings(
        source=tmp_path,
        workspace=tmp_path / ".document-eater-workspace",
        strict_offline=True,
    )
    terminal, _output = _terminal(["https://inference.example.com/v1"])

    with pytest.raises(ValueError, match="offline"):
        DocumentTUI(settings, terminal)._configure_endpoint()


def test_strict_offline_tui_refuses_remote_endpoint_at_startup(tmp_path):
    settings = TUISettings(
        source=tmp_path,
        workspace=tmp_path / ".document-eater-workspace",
        base_url="https://inference.example.com/v1",
        allow_remote=True,
        strict_offline=True,
    )

    with pytest.raises(ValueError, match="offline"):
        DocumentTUI(settings)._validate_paths()


def test_tui_reads_cached_summary_without_document_text(tmp_path):
    workspace = tmp_path / ".document-eater-workspace"
    workspace.mkdir()
    (workspace / "audit.json").write_text(
        json.dumps(
            {
                "verification_mode": "qwen",
                "summary": {"PASS": 2, "FAIL": 1, "UNKNOWN": 3},
                "items": [{"requirement": {"text": "private text"}}],
            }
        ),
        encoding="utf-8",
    )
    terminal, output = _terminal(["0"])

    DocumentTUI(TUISettings(tmp_path, workspace), terminal).run()

    rendered = output.getvalue()
    assert "PASS:2" in rendered
    assert "FAIL:1" in rendered
    assert "private text" not in rendered


def test_tui_candidate_audit_runs_end_to_end_without_opencode(tmp_path):
    (tmp_path / "requirement.txt").write_text(
        "The supplier must submit the signed report before delivery.", encoding="utf-8"
    )
    workspace = tmp_path / ".document-eater-workspace"
    terminal, output = _terminal(["1", "", "0"])
    settings = TUISettings(
        source=tmp_path,
        workspace=workspace,
        retrieval="lexical",
        color=False,
    )

    DocumentTUI(settings, terminal).run()

    audit = json.loads((workspace / "audit.json").read_text(encoding="utf-8"))
    assert audit["verification_mode"] == "candidate_only"
    assert audit["summary"]["UNKNOWN"] == 1
    assert (workspace / "report.html").is_file()
    assert "Готово: 1 требований" in output.getvalue()
