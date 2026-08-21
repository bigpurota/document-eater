from __future__ import annotations

import json
from pathlib import Path

from document_eater.install_opencode import (
    build_workspace_config,
    install_workspace_launchers,
)


def test_checked_in_template_uses_supported_opencode_shape():
    project = Path(__file__).resolve().parents[1]
    config = build_workspace_config(project)

    assert "provider" in config
    assert "providers" not in config
    assert config["model"] in {
        f"local-docs/{model_id}" for model_id in config["provider"]["local-docs"]["models"]
    }
    assert "servers" not in config["mcp"]
    server = config["mcp"]["document-eater"]
    assert server["enabled"] is True
    assert "codemode" not in server
    assert config["default_agent"] == "document-auditor"
    agent = config["agent"]["document-auditor"]
    assert agent["mode"] == "primary"
    assert agent["steps"] == 12
    assert "Never emit routine progress" in agent["prompt"]
    assert "Never repeat a tool call" in agent["prompt"]
    assert agent["permission"]["*"] == "deny"
    assert agent["permission"]["document_eater_*"] == "allow"


def test_workspace_launchers_use_application_project_but_document_cwd(tmp_path):
    project = tmp_path / "application code"
    project.mkdir()
    (project / "opencode.json").write_text(
        json.dumps(
            {
                "model": "local-docs/models/Qwen3.8-27B-4bit",
                "mcp": {
                    "document-eater": {
                        "type": "local",
                        "command": ["uv", "run", "document-eater-mcp"],
                        "cwd": ".",
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
    server = config["mcp"]["document-eater"]
    assert server["cwd"] == "."
    assert server["enabled"] is True
    assert server["command"] == [
        "uv",
        "run",
        "--project",
        str(project.resolve()),
        "--no-sync",
        "document-eater-mcp",
    ]
    assert server["environment"]["DOCUMENT_EATER_MODEL_CACHE"] == str(
        project.resolve() / "models" / "retrieval"
    )
    assert installed["document-opencode"] == str(bin_dir / "document-opencode")
    launcher = (bin_dir / "document-opencode").read_text()
    assert "OPENCODE_CONFIG=" in launcher
    assert launcher.endswith('exec opencode "$@"\n')
    assert (bin_dir / "document-opencode").stat().st_mode & 0o111
    assert (bin_dir / "document-qwen").stat().st_mode & 0o111
    assert (bin_dir / "document-qwen-smoke").stat().st_mode & 0o111


def test_workspace_config_migrates_short_lived_v2_mcp_shape(tmp_path):
    project = tmp_path / "application"
    project.mkdir()
    (project / "opencode.json").write_text(
        json.dumps(
            {
                "mcp": {
                    "servers": {
                        "document-eater": {
                            "type": "local",
                            "command": ["old"],
                            "codemode": False,
                        }
                    }
                }
            }
        )
    )

    install_workspace_launchers(
        project,
        config_dir=tmp_path / "config",
        bin_dir=tmp_path / "bin",
    )

    config = json.loads((tmp_path / "config" / "document-eater.json").read_text())
    assert "servers" not in config["mcp"]
    server = config["mcp"]["document-eater"]
    assert server["enabled"] is True
    assert "codemode" not in server
