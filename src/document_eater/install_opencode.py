from __future__ import annotations

import argparse
import json
import os
import shlex
from pathlib import Path
from typing import Any


def build_workspace_config(project_root: Path) -> dict[str, Any]:
    """Convert the checked-in project config into a workspace-independent config."""
    root = project_root.expanduser().resolve()
    source = root / "opencode.json"
    if not source.is_file():
        raise ValueError(f"opencode.json not found under project root: {root}")
    config = json.loads(source.read_text(encoding="utf-8"))
    server = config["mcp"]["servers"]["document-eater"]
    server["command"] = [
        "uv",
        "run",
        "--project",
        str(root),
        "--no-sync",
        "document-eater-mcp",
    ]
    # A relative MCP cwd resolves from the OpenCode workspace. This deliberately
    # makes '.', relative input paths, and relative audit outputs refer to the folder
    # from which document-opencode was launched, not the application source tree.
    server["cwd"] = "."
    return config


def _launcher(environment: dict[str, str], command: list[str]) -> str:
    exports = "\n".join(
        f"export {name}={shlex.quote(value)}" for name, value in environment.items()
    )
    rendered = " ".join(shlex.quote(value) for value in command)
    return f'#!/bin/zsh\nset -euo pipefail\n{exports}\nexec {rendered} "$@"\n'


def install_workspace_launchers(
    project_root: Path,
    *,
    config_dir: Path,
    bin_dir: Path,
) -> dict[str, str]:
    root = project_root.expanduser().resolve()
    resolved_config_dir = config_dir.expanduser().resolve()
    resolved_bin_dir = bin_dir.expanduser().resolve()
    resolved_config_dir.mkdir(parents=True, exist_ok=True)
    resolved_bin_dir.mkdir(parents=True, exist_ok=True)

    config_path = resolved_config_dir / "document-eater.json"
    config_path.write_text(
        json.dumps(build_workspace_config(root), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    config_path.chmod(0o600)

    launchers = {
        "document-opencode": _launcher(
            {"OPENCODE_CONFIG": str(config_path)},
            ["opencode"],
        ),
        "document-qwen": _launcher(
            {},
            ["zsh", str(root / "scripts" / "start-qwen-macos.sh")],
        ),
        "document-qwen-smoke": _launcher(
            {},
            [
                "uv",
                "run",
                "--project",
                str(root),
                "--no-sync",
                "python",
                str(root / "scripts" / "smoke-mlx-tools.py"),
            ],
        ),
    }
    for name, content in launchers.items():
        path = resolved_bin_dir / name
        path.write_text(content, encoding="utf-8")
        path.chmod(0o755)

    return {
        "config": str(config_path),
        **{name: str(resolved_bin_dir / name) for name in launchers},
    }


def _default_config_dir() -> Path:
    xdg = os.environ.get("XDG_CONFIG_HOME")
    base = Path(xdg).expanduser() if xdg else Path.home() / ".config"
    return base / "opencode"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Install workspace-independent Document Eater launchers for OpenCode."
    )
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--config-dir", type=Path, default=_default_config_dir())
    parser.add_argument("--bin-dir", type=Path, required=True)
    args = parser.parse_args()
    installed = install_workspace_launchers(
        args.project_root,
        config_dir=args.config_dir,
        bin_dir=args.bin_dir,
    )
    print(json.dumps(installed, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
