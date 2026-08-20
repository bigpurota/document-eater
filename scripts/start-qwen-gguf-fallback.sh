#!/bin/zsh
set -euo pipefail

SCRIPT_DIR="${0:A:h}"
PROJECT_ROOT="${SCRIPT_DIR:h}"
cd "$PROJECT_ROOT"

MODEL_PATH="${1:-models/Qwen3.8-27B-OBLITERATED-Q4_K_M.gguf}"
MODEL_ALIAS="${2:-OBLITERATUS/Qwen3.8-27B-OBLITERATED}"

if ! command -v llama-server >/dev/null 2>&1; then
  print -u2 "llama-server not found. Install it with: brew install llama.cpp"
  exit 1
fi
if [[ ! -f "$MODEL_PATH" ]]; then
  print -u2 "GGUF fallback not found: $MODEL_PATH"
  exit 1
fi

exec llama-server \
  --model "$MODEL_PATH" \
  --alias "$MODEL_ALIAS" \
  --host 127.0.0.1 \
  --port 8080 \
  --ctx-size 8192 \
  --n-gpu-layers 999 \
  --flash-attn on \
  --jinja \
  --reasoning off
