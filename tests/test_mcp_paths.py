from __future__ import annotations

import json

from document_eater.mcp_server import _resolve_artifacts_root, _resolve_run_file


def test_latest_timestamped_run_is_resolved_from_default_style_path(tmp_path):
    workspace = tmp_path / "audit-run"
    older = workspace / "20260820-100000"
    newer = workspace / "20260820-110000"
    older.mkdir(parents=True)
    newer.mkdir(parents=True)
    (older / "audit.json").write_text(json.dumps({"run": "older"}))
    (newer / "audit.json").write_text(json.dumps({"run": "newer"}))
    (newer / "artifacts").mkdir()

    resolved = _resolve_run_file(str(workspace / "audit.json"), "audit.json")

    assert resolved == newer / "audit.json"
    assert _resolve_artifacts_root(str(workspace / "artifacts")) == newer / "artifacts"


def test_exact_run_file_wins_over_latest_discovery(tmp_path):
    exact = tmp_path / "selected" / "index.sqlite3"
    exact.parent.mkdir()
    exact.write_bytes(b"sqlite")

    assert _resolve_run_file(str(exact), "index.sqlite3") == exact
