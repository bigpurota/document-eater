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

## End-to-end: fresh M3 Max to first document audit

This is the supported path for an **Apple M3 Max with 36 GB unified memory**. Keep at
least 40 GiB free; 50 GiB or more is recommended for the Python environment, Qwen,
the retrieval models, and working artifacts. Source PDFs may live anywhere on the
Mac and are never required to be inside this repository.

### 1. Prepare the Mac and clone the private repository

The Mac needs Homebrew and Xcode Command Line Tools. Verify them:

```bash
xcode-select -p
brew --version
```

If `xcode-select` fails, run `xcode-select --install`. To clone with GitHub CLI:

```bash
brew install gh
gh auth login
cd ~/Documents
gh repo clone bigpurota/document-eater
cd document-eater
```

Alternatively, download the repository ZIP while signed into GitHub, unpack it, and
open Terminal in the unpacked `document-eater` folder.

### 2. Run the complete installer

From Terminal:

```bash
zsh scripts/bootstrap-m3-max.sh
```

Or double-click `INSTALL-M3-MAX.command` in Finder. If Gatekeeper blocks it,
Control-click the file, choose **Open**, and confirm once.

The installer is resumable and idempotent. It:

- installs missing `uv`, Tesseract, Russian OCR data, and OpenCode through Homebrew;
- installs Python 3.12 and the locked project environment;
- downloads the pinned Qwen3.8 27B MLX 4-bit checkpoint (~16.05 GB);
- downloads pinned BGE-M3 and `bge-reranker-v2-m3` retrieval models;
- prefetches the isolated `mlx-lm==0.31.3` runtime;
- runs the project test suite.

It does not search for, read, copy, or upload private documents. A successful finish
ends with `Bootstrap complete` and a passing test summary.

Optional installation checks:

```bash
test -f models/Qwen3.8-27B-4bit/config.json && echo "Qwen model: OK"
tesseract --list-langs | grep -E '^(eng|rus)$'
uv run --no-sync python -m pytest
opencode --version
```

### 3. Start local Qwen

Keep this first terminal open:

```bash
cd ~/Documents/document-eater
zsh scripts/start-qwen-macos.sh
```

The server listens only on `127.0.0.1:8080`. Initial model loading can take a while.
After it starts, this request must succeed from a second terminal:

```bash
curl -fsS http://127.0.0.1:8080/v1/models
```

### 4. Verify agent tool calling

In the second terminal:

```bash
cd ~/Documents/document-eater
uv run --no-sync python scripts/smoke-mlx-tools.py
```

Continue only after it prints:

```text
MLX tool-call smoke test PASSED
```

This is a required gate: a server that generates normal text but fails this test
cannot reliably drive the OpenCode/MCP agent loop.

### 5. Keep documents outside the code repository

For example:

```text
~/Documents/PrivateDocuments/project-1/    # source PDFs
~/Documents/DocumentAudits/project-1/      # generated audit data
~/Documents/document-eater/                # application code and local models
```

Directories on an encrypted external SSD work as well, for example
`/Volumes/PrivateSSD/Documents/project-1`. Terminal and OpenCode must have permission
to access the selected folders under **System Settings → Privacy & Security**.

### 6. Start OpenCode and run the first audit

Always launch OpenCode from the repository so it loads the checked-in
`opencode.json` and MCP server:

```bash
cd ~/Documents/document-eater
opencode
```

Confirm that the selected model is `local-docs/qwen-27b`, then send a prompt with
absolute input and output paths:

```text
Use the document-eater tools to audit every PDF under
/Users/YOUR_NAME/Documents/PrivateDocuments/project-1.

Store generated artifacts under
/Users/YOUR_NAME/Documents/DocumentAudits/project-1.

Use local Qwen verification. Extract requirements, obligations, deadlines,
required documents, exceptions, and conflicting clauses. For every result use
PASS, PARTIAL, FAIL, UNKNOWN, CONFLICT, or NOT_APPLICABLE. Never convert missing
evidence into FAIL. Show FAIL, CONFLICT, and UNKNOWN first and cite exact source
documents and pages. Return the absolute report path when finished.
```

OpenCode should call `audit_documents`, wait for OCR/indexing/retrieval/model
verification, and return a summary plus an absolute `report_path`.

### 7. Inspect and continue working with the corpus

Open the path returned by the agent, normally:

```bash
open /Users/YOUR_NAME/Documents/DocumentAudits/project-1/report.html
```

The output directory contains:

| File | Purpose |
|---|---|
| `report.html` | human-readable requirement audit |
| `requirements.csv` | sortable/exportable requirement table |
| `audit.json` | complete machine-readable results and citations |
| `run-manifest.json` | input, model, retrieval mode, and artifact provenance |
| `index.sqlite3` | local lexical/dense/sparse retrieval index |
| `artifacts/<run-id>/` | extracted pages, blocks, bounding boxes, and graphs |

Follow-up prompts reuse those artifacts. Examples:

```text
Show every UNKNOWN item and explain which evidence is missing.
```

```text
Find all requirements concerning final payment and show the original pages.
```

```text
Check whether the main agreement and its appendices contain conflicting deadlines.
```

### CLI-only end-to-end alternative

With the Qwen terminal still running:

```bash
cd ~/Documents/document-eater
uv run --no-sync document-eater audit \
  /Users/YOUR_NAME/Documents/PrivateDocuments/project-1 \
  --output /Users/YOUR_NAME/Documents/DocumentAudits/project-1 \
  --use-llm
open /Users/YOUR_NAME/Documents/DocumentAudits/project-1/report.html
```

Omit `--use-llm` for extraction, requirement candidates, and retrieval without model
verification. In that mode, unresolved verdicts deliberately remain `UNKNOWN`.

### Updating an installed copy

Stop OpenCode and the MLX server, then run:

```bash
cd ~/Documents/document-eater
git pull --ff-only
zsh scripts/bootstrap-m3-max.sh
```

Existing private documents and audit directories outside the repository are not
touched.

### Troubleshooting

| Symptom | Action |
|---|---|
| `rus` is absent from `tesseract --list-langs` | Run `brew install tesseract-lang`, then retry. |
| Port 8080 is unavailable | Stop the old model server with `Ctrl+C`; do not expose a replacement on `0.0.0.0`. |
| `Local Qwen endpoint is unavailable` | Start `scripts/start-qwen-macos.sh` and verify `/v1/models`. |
| MLX tool-call smoke test fails | Do not use OpenCode automation; keep the server log and checkpoint/runtime versions for diagnosis. |
| OpenCode cannot see document tools | Start it from the repository and run `uv run --no-sync document-eater-mcp` once to inspect startup errors. |
| macOS denies access to PDFs or an external SSD | Grant Terminal/OpenCode access under Privacy & Security → Files and Folders. |
| Memory pressure becomes high | Close other large applications, restart the MLX server, and keep the configured 8k context/4 GB prompt-cache limits. |
| Report contains only `UNKNOWN` | Confirm that model verification was requested; otherwise this is the intentional candidate-only behavior. |

If the MLX server or smoke test fails, do not automatically switch to a cloud model:
that would change the privacy boundary for every document fragment returned to the
agent.

## Privacy boundary

- Source PDFs stay at the path you choose. Extracted artifacts, embeddings, and graph
  data stay in the selected local output directory or the ignored `data/`,
  `artifacts/`, and `models/` defaults.
- The LLM adapter accepts loopback endpoints. The primary MLX server is local; an
  optional remote server must be reached through an SSH tunnel and receives only
  retrieved evidence blocks rather than entire PDFs.
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

The published 262k context length is not the operating target for this laptop. RAG
uses a small, explicit evidence budget; the initial local MLX profile starts at 8k
context.
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

## Retrieval and runtime profiles

The default quality retriever is fully local:

```text
BM25 + BGE-M3 dense + BGE-M3 learned sparse -> RRF -> bge-reranker-v2-m3
```

Both BGE models are pinned to exact Hugging Face revisions. `--retrieval hybrid`
selects the lighter `BM25 + multilingual-e5-large` profile; `--retrieval lexical`
is the BM25 control.

The Qwen launcher binds only to `127.0.0.1`, disables thinking, and caps the
persistent prompt-cache pool at 4 GB. OpenCode separately caps requests at 8192
tokens. `mlx-lm==0.31.3` runs in an isolated `uvx` environment because that runtime
requires Transformers 5 while the local BGE reranker is deliberately pinned to
Transformers 4.x.

The abliterated fallback remains a separate llama.cpp/GGUF path through
`scripts/start-qwen-gguf-fallback.sh`; pass `--profile abliterated` to the CLI when
using that server. It is never selected automatically. Vast profiles remain in
`config/model-profiles.toml` for cases where local latency is unacceptable and the
data owner explicitly permits third-party processing.

## OpenCode integration details

The checked-in `opencode.json` is for the current OpenCode V2 configuration. It:

- selects the local MLX Qwen endpoint as the OpenCode model;
- starts `document-eater-mcp` over stdio;
- exposes `audit_documents`, `prepare_corpus`, `search_corpus`,
  `list_audit_items`, `get_audit_summary`, `read_document_page`, and
  `graph_neighbors` directly to the model.

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
