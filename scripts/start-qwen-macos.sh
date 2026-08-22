#!/bin/zsh
set -euo pipefail

SCRIPT_DIR="${0:A:h}"
PROJECT_ROOT="${SCRIPT_DIR:h}"
cd "$PROJECT_ROOT"

# Runtime is deliberately fail-closed: uv/Transformers/Hugging Face may use only
# artifacts downloaded by bootstrap, and the model server binds to loopback below.
export DOCUMENT_EATER_STRICT_OFFLINE=1
export UV_OFFLINE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export DO_NOT_TRACK=1
# mlx-lm runs in an isolated uvx environment. Loading this project through
# PYTHONPATH makes sitecustomize install the same loopback-only socket guard there.
export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

MODEL_PATH="${1:-models/Qwen3.8-27B-4bit}"

if [[ ! -f "$MODEL_PATH/config.json" ]]; then
  print -u2 "MLX model not found: $MODEL_PATH"
  print -u2 "Download the pinned model first; see README.md -> Local Qwen setup."
  exit 1
fi

# 4 GB caps the persistent prompt-cache pool. The OpenCode model declaration
# separately limits individual requests to 12288 tokens.
# Keep mlx-lm isolated from the document environment: mlx-lm 0.31 uses
# Transformers 5, while the pinned FlagEmbedding reranker needs Transformers 4.
exec uvx --from 'mlx-lm==0.31.3' mlx_lm.server \
  --model "$MODEL_PATH" \
  --host 127.0.0.1 \
  --port 8080 \
  --temp 0.1 \
  --top-p 0.8 \
  --top-k 20 \
  --max-tokens 2048 \
  --chat-template-args '{"enable_thinking":false}' \
  --prompt-cache-size 2 \
  --prompt-cache-bytes 4GB
