# Document Eater

Local-first pipeline for private documents:

1. extract the original PDF text layer when it is usable;
2. OCR only pages that do not contain enough usable text;
3. preserve page and bounding-box provenance;
4. build a deterministic document graph;
5. retrieve locally and send only selected evidence to a Qwen endpoint.

The implemented slice includes ingestion, the structural graph, a high-quality
multilingual retrieval profile, an OpenCode MCP agent interface, and a localhost-only
Qwen client. Retrieval keeps its component scores visible so BM25, BGE-M3 dense,
learned-sparse, fusion, and reranking can be evaluated independently.

## Privacy boundary

- Source PDFs, extracted artifacts, embeddings, and graph data stay under local
  `data/`, `artifacts/`, and `models/` directories, all ignored by git.
- The remote LLM adapter accepts localhost endpoints and is intended for an SSH tunnel. It
  will send retrieved blocks, not full documents.
- A Vast.ai instance is still a third-party processor. Use it only when the data
  owner's policy permits external processing. Encryption in transit does not make
  a third party equivalent to local execution.
- No telemetry or hosted API is part of the ingestion path.

## Hardware profiles

Primary model: **`Qwen/Qwen3.8-27B`** as the pinned
**`mlx-community/Qwen3.8-27B-4bit`** conversion on the local Mac.
**`OBLITERATUS/Qwen3.8-27B-OBLITERATED`** remains an explicitly selected fallback,
not an automatic replacement.

| Profile | Quantization | Approx. weights | Intended use |
|---|---:|---:|---|
| M3 Max 36 GB primary | MLX 4-bit | 16.05 GB weights | local private inference; 8k initial context |
| Vast fallback | base Q4_K_M | 16.46 GB file | 24 GB GPU, only when data policy permits |
| Vast primary reference | official BF16 | ~56 GB | 80 GB GPU, evaluation/reference runs |
| Vast manual fallback | OBLITERATUS Q4_K_M | 16.81 GB file | separate 24 GB profile, loaded only when requested |

The published 262k context length is not a single-GPU operating target. RAG will use
a small, explicit evidence budget; the initial Vast Q4 profile starts at 8k context.
Community abliterated checkpoints are untrusted supply-chain inputs until hashes,
lineage, license, and a small document QA evaluation are recorded.

The local MLX primary is pinned to revision
`3e6447f082e89cc7f0bc6e5441afd38dfce760ff`. The official BF16 reference is pinned
to `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`; its Vast/GGUF derivative is pinned
to `unsloth/Qwen3.8-27B-GGUF@4ca7207`. The OBLITERATUS fallback remains pinned to
`46c3c40f`. Its model card reports a six-point MMLU drop (87.4% to 81.4%), which is
why it is never selected automatically.

Use the fallback only for an explicit policy refusal. Do not retry merely because
the primary says the retrieved evidence is insufficient, cannot support a claim, or
contains conflicting facts; those are desirable document-QA outcomes.

PyMuPDF is used by the ingestion MVP and is AGPL/commercial dual-licensed. Internal
use must be checked against the data owner's software policy; before proprietary
distribution, buy an appropriate license or replace this adapter with a permissive
PDF backend.

## Try it now (Qwen is optional)

Requirements: Python 3.12+, `uv`, and Tesseract with `rus` and `eng` language data.

```bash
uv sync --extra quality --no-editable --reinstall-package document-eater
uv run --no-sync document-eater audit /path/to/pdf-folder --output audit-run
open audit-run/report.html
```

This first command is deliberately useful without an LLM: it performs native-text
extraction/OCR, finds explicit requirement candidates, retrieves possible evidence,
and writes `report.html`, `requirements.csv`, `audit.json`, and a run manifest. All
verdicts remain `UNKNOWN` until Qwen checks the evidence. Missing evidence is never
silently converted into `FAIL`.

The explicit non-editable reinstall works around a macOS/Python issue where `.pth`
files under a folder marked hidden can be skipped. Subsequent `--no-sync` commands use
that exact installed build and do not silently remove the `quality` extra.

The default retrieval profile is:

```text
BM25 + BGE-M3 dense + BGE-M3 learned sparse -> RRF -> bge-reranker-v2-m3
```

Both BGE models are local and pinned to exact Hugging Face revisions. The first run
downloads their weights. `--retrieval hybrid` selects the lighter
`BM25 + multilingual-e5-large` profile; `--retrieval lexical` is the BM25 control.

## Local Qwen setup on an M3 Max 36 GB

For a new Mac that already has Homebrew and Xcode Command Line Tools, the complete
bootstrap can be started by double-clicking `INSTALL-M3-MAX.command` in Finder, or
with one terminal command:

```bash
cd /path/to/document_eater
zsh scripts/bootstrap-m3-max.sh
```

It installs missing `uv`, Tesseract language data, and OpenCode; creates the pinned
Python environment; downloads the pinned Qwen MLX weights, BGE-M3, and the reranker;
prefetches the isolated MLX runtime; and runs the tests. Downloads are resumable. It
does not inspect, copy, or move private documents. Expect more than 20 GB of model
downloads and keep at least 40 GiB of free disk space.

The equivalent manual setup is below.

Install OCR and download the pinned MLX 4-bit model (16.05 GB of weight shards):

```bash
brew install tesseract tesseract-lang
uv sync --extra quality --no-editable \
  --reinstall-package document-eater
HF_HUB_DISABLE_TELEMETRY=1 uvx --from huggingface-hub hf download \
  mlx-community/Qwen3.8-27B-4bit \
  --revision 3e6447f082e89cc7f0bc6e5441afd38dfce760ff \
  --local-dir models/Qwen3.8-27B-4bit
zsh scripts/start-qwen-macos.sh
```

Keep that terminal open. In a second terminal, first verify the exact capability
that OpenCode needs:

```bash
uv run --no-sync python scripts/smoke-mlx-tools.py
```

Only a `PASSED` result establishes that this checkpoint/runtime pair is returning
OpenAI-shaped tool calls. Then run a document audit:

```bash
uv run --no-sync document-eater audit /path/to/pdf-folder \
  --output audit-run --use-llm
open audit-run/report.html
```

The launcher binds only to `127.0.0.1`, disables thinking, and caps the persistent
prompt-cache pool at 4 GB. OpenCode separately caps requests at 8192 tokens. This is
an initial operating profile, not a measured throughput or memory optimum on your
exact M3 Max. It launches `mlx-lm==0.31.3` in an isolated `uvx` environment because
that runtime requires Transformers 5 while the local BGE reranker is deliberately
pinned to Transformers 4.x.

The abliterated fallback remains a separate llama.cpp/GGUF path via
`scripts/start-qwen-gguf-fallback.sh`; pass `--profile abliterated` to the audit CLI.
It is never selected automatically. The Vast profiles remain in
`config/model-profiles.toml` for cases where local latency is unacceptable and the
data owner allows third-party processing.

## Use it as an agent from OpenCode

The checked-in `opencode.json` is for the current OpenCode V2 configuration. It:

- selects the local MLX Qwen endpoint as the OpenCode model;
- starts `document-eater-mcp` over stdio;
- exposes `audit_documents`, `prepare_corpus`, `search_corpus`,
  `list_audit_items`, `get_audit_summary`, `read_document_page`, and
  `graph_neighbors` directly to the model.

Start Qwen, pass the tool-call smoke test, then run `opencode` from this repository
and ask:

```text
Use document-eater to audit /absolute/path/to/my/pdfs with local Qwen.
Show UNKNOWN and CONFLICT first, and cite the source pages.
```

Do not switch OpenCode to a cloud model for this task: MCP returns extracted document
text to the controlling model. A cloud OpenCode model therefore breaks the local-only
privacy boundary even though OCR and the index still run locally.

OpenCode V2 and older releases use different config shapes. The included file targets
V2; see the current [OpenCode MCP documentation](https://opencode.ai/v2/docs/mcp-servers)
if your installed version rejects `mcp.servers`.

## Lower-level commands

```bash
uv run --no-sync document-eater inspect path/to/document.pdf
uv run --no-sync document-eater ingest path/to/document.pdf --output artifacts
uv run --no-sync document-eater index artifacts --quality --database data/index.sqlite3
uv run --no-sync document-eater search "условия расторжения" --database data/index.sqlite3
uv run --no-sync document-eater ask "каков срок уведомления?" --database data/index.sqlite3
uv run --no-sync document-eater ask "..." --profile abliterated --database data/index.sqlite3
uv run --no-sync python -m pytest
```

`inspect` performs no OCR and reports which pages would use the native text layer or
OCR. `ingest` defaults to `--ocr auto --languages rus+eng` and writes:

- `<document-id>/document.json` — pages and blocks with provenance;
- `<document-id>/graph.json` — `contains`, `next`, and section membership edges;
- `<document-id>/manifest.json` — source hash and pipeline settings.

The source PDF is not copied into the artifact directory.

## What this build does and does not prove

Implemented and tested: page-level native/OCR routing, provenance, structural graph,
BM25+dense RRF, BGE-M3 dense+sparse generation on Apple Metal, multilingual BGE
reranking, rule-based requirement candidates, conservative Qwen JSON verdict
validation, HTML/CSV/JSON reports, and an MCP stdio handshake. A quality-RAG
end-to-end synthetic English PDF run found the expected requirement and evidence;
without an LLM it correctly retained `UNKNOWN`. The automated suite does not yet
establish recall on your documents, full Qwen tool use, or M3 Max throughput and peak
memory. Run the included MLX tool-call smoke test on the target Mac before relying on
OpenCode automation.

The current requirement extractor recognizes explicit modal language (`должен`,
`необходимо`, `требуется`, `must`, `shall`, and similar markers). Implicit obligations,
complex tables, signatures, handwritten marks, and visual checkbox state are not yet
validated. The search index is an auditable FTS5 baseline, not yet the final "good
RAG" merely because the components are present: promotion still needs a private
held-out set with expected evidence pages, Recall@k, citation precision, and latency.
