#!/usr/bin/env python3
"""Verify that the local MLX server returns OpenAI-shaped tool calls."""

from __future__ import annotations

import json
import sys
import urllib.request

from document_eater.privacy import enable_strict_offline, strict_offline_requested

if strict_offline_requested():
    enable_strict_offline()

MODEL = "models/Qwen3.8-27B-4bit"
URL = "http://127.0.0.1:8080/v1/chat/completions"


payload = {
    "model": MODEL,
    "messages": [
        {
            "role": "user",
            "content": (
                "Call echo exactly once with value mlx-tool-smoke. Do not answer in plain text."
            ),
        }
    ],
    "tools": [
        {
            "type": "function",
            "function": {
                "name": "echo",
                "description": "Return a supplied value.",
                "parameters": {
                    "type": "object",
                    "properties": {"value": {"type": "string"}},
                    "required": ["value"],
                    "additionalProperties": False,
                },
            },
        }
    ],
    "tool_choice": "required",
    "temperature": 0.0,
    "max_tokens": 256,
    "chat_template_kwargs": {"enable_thinking": False},
}

request = urllib.request.Request(
    URL,
    data=json.dumps(payload).encode("utf-8"),
    headers={"Content-Type": "application/json"},
    method="POST",
)

try:
    with urllib.request.urlopen(request, timeout=300) as response:
        result = json.load(response)
    call = result["choices"][0]["message"]["tool_calls"][0]
    arguments = json.loads(call["function"]["arguments"])
    assert call["function"]["name"] == "echo"
    assert arguments == {"value": "mlx-tool-smoke"}
except Exception as exc:
    print(f"MLX tool-call smoke test FAILED: {exc}", file=sys.stderr)
    sys.exit(1)

print("MLX tool-call smoke test PASSED")
