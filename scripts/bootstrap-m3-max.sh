#!/bin/zsh
set -euo pipefail

SCRIPT_DIR="${0:A:h}"
PROJECT_ROOT="${SCRIPT_DIR:h}"
QWEN_DIR="$PROJECT_ROOT/models/Qwen3.8-27B-4bit"
RETRIEVAL_CACHE="$PROJECT_ROOT/models/retrieval"

QWEN_REPO="mlx-community/Qwen3.8-27B-4bit"
QWEN_REVISION="3e6447f082e89cc7f0bc6e5441afd38dfce760ff"
BGE_REPO="BAAI/bge-m3"
BGE_REVISION="5617a9f61b028005a4858fdac845db406aefb181"
RERANKER_REPO="BAAI/bge-reranker-v2-m3"
RERANKER_REVISION="953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"
HF_CLI_VERSION="0.36.2"
MLX_LM_VERSION="0.31.3"

if [[ "$(uname -s)" != "Darwin" || "$(uname -m)" != "arm64" ]]; then
  print -u2 "This bootstrap requires macOS on Apple Silicon (arm64)."
  exit 1
fi
if ! command -v brew >/dev/null 2>&1; then
  print -u2 "Homebrew is required. Install it from https://brew.sh and rerun."
  exit 1
fi
if ! xcode-select -p >/dev/null 2>&1; then
  print -u2 "Xcode Command Line Tools are required. Run: xcode-select --install"
  exit 1
fi

MEMORY_BYTES="$(sysctl -n hw.memsize)"
if (( MEMORY_BYTES < 34000000000 )); then
  print -u2 "WARNING: less than 34 GB unified memory detected."
  print -u2 "The supported target is an M3 Max with 36 GB."
fi

AVAILABLE_KB="$(df -Pk "$PROJECT_ROOT" | awk 'NR == 2 {print $4}')"
if (( AVAILABLE_KB < 41943040 )); then
  print -u2 "WARNING: less than 40 GiB free disk space. Model downloads may fail."
fi

cd "$PROJECT_ROOT"
mkdir -p "$QWEN_DIR" "$RETRIEVAL_CACHE"

print "[1/8] Installing required Homebrew packages"
if ! command -v uv >/dev/null 2>&1; then
  brew install uv
fi
if ! command -v tesseract >/dev/null 2>&1; then
  brew install tesseract
fi
if ! brew list --versions tesseract-lang >/dev/null 2>&1; then
  brew install tesseract-lang
fi
if ! command -v opencode >/dev/null 2>&1; then
  brew install anomalyco/tap/opencode
fi

print "[2/8] Installing Python 3.12 and the document pipeline"
uv python install 3.12
uv sync \
  --python 3.12 \
  --extra quality \
  --no-editable \
  --reinstall-package document-eater

print "[3/8] Installing workspace-independent OpenCode launchers"
BREW_PREFIX="$(brew --prefix)"
if [[ -z "$BREW_PREFIX" || ! -d "$BREW_PREFIX/bin" ]]; then
  print -u2 "Could not resolve the Homebrew bin directory."
  exit 1
fi
uv run --no-sync document-eater-install-opencode \
  --project-root "$PROJECT_ROOT" \
  --bin-dir "$BREW_PREFIX/bin"

export HF_HUB_DISABLE_TELEMETRY=1
export DO_NOT_TRACK=1

print "[4/8] Downloading pinned Qwen3.8 27B MLX 4-bit weights (~16.05 GB)"
uvx --from "huggingface-hub==$HF_CLI_VERSION" hf download \
  "$QWEN_REPO" \
  --revision "$QWEN_REVISION" \
  --local-dir "$QWEN_DIR"

print "[5/8] Downloading pinned BGE-M3 retrieval model"
uvx --from "huggingface-hub==$HF_CLI_VERSION" hf download \
  "$BGE_REPO" \
  --revision "$BGE_REVISION" \
  --cache-dir "$RETRIEVAL_CACHE"

print "[6/8] Downloading pinned multilingual reranker"
uvx --from "huggingface-hub==$HF_CLI_VERSION" hf download \
  "$RERANKER_REPO" \
  --revision "$RERANKER_REVISION" \
  --cache-dir "$RETRIEVAL_CACHE"

print "[7/8] Prefetching isolated MLX runtime"
uvx --from "mlx-lm==$MLX_LM_VERSION" mlx_lm.server --help >/dev/null

print "[8/8] Running the local automated checks"
uv run --no-sync python -m pytest

print
print "Bootstrap complete. No private documents were read or copied."
print
print "Next, start the model from any directory in terminal 1:"
print "  document-qwen"
print
print "Then verify tool calling in terminal 2:"
print "  document-qwen-smoke"
print
print "Finally, open the folder containing the PDFs and start the document agent:"
print "  cd '/absolute/path/to/my/documents'"
print "  document-opencode"
