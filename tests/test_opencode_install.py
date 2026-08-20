from __future__ import annotations

import json

from document_eater.install_opencode import install_workspace_launchers


def test_workspace_launchers_use_application_project_but_document_cwd(tmp_path):
    project = tmp_path / "application code"
    project.mkdir()
    (project / "opencode.json").write_text(
        json.dumps(
            {
                "model": "local-docs/qwen-27b",
                "mcp": {
                    "servers": {
                        "document-eater": {
                            "type": "local",
                            "command": ["uv", "run", "document-eater-mcp"],
                            "cwd": ".",
                        }
                    }
                },
            }
        )
    )
    config_dir = tmp_path / "config"
    bin_dir = tmp_path / "bin"

    installed = install_workspace_launchers(
        project,
        config_dir=config_dir,
        bin_dir=bin_dir,
    )

    config = json.loads((config_dir / "document-eater.json").read_text())
    server = config["mcp"]["servers"]["document-eater"]
    assert server["cwd"] == "."
    assert server["command"] == [
        "uv",
        "run",
        "--project",
        str(project.resolve()),
        "--no-sync",
        "document-eater-mcp",
    ]
    assert installed["document-opencode"] == str(bin_dir / "document-opencode")
    launcher = (bin_dir / "document-opencode").read_text()
    assert "OPENCODE_CONFIG=" in launcher
    assert launcher.endswith('exec opencode "$@"\n')
    assert (bin_dir / "document-opencode").stat().st_mode & 0o111
    assert (bin_dir / "document-qwen").stat().st_mode & 0o111
    assert (bin_dir / "document-qwen-smoke").stat().st_mode & 0o111
