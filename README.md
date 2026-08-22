# Document Eater

Local-first pipeline for private PDF, Word, Excel, Markdown, XML, CSV, and text documents:

1. route each file to a format-aware local parser;
2. use the original PDF text layer when usable and OCR only weak/scanned pages;
3. preserve page, paragraph, table, sheet, row, cell, XML-path, and line provenance;
4. build a deterministic document graph;
5. retrieve locally and send only selected evidence to a Qwen endpoint.

The implemented slice includes ingestion, the structural graph, a high-quality
multilingual retrieval profile, an OpenCode MCP agent interface, and a localhost-only
Qwen client. Retrieval keeps its component scores visible so BM25, BGE-M3 dense,
learned-sparse, fusion, and reranking can be evaluated independently.

## End-to-end: fresh M3 Max to first document audit

This is the supported path for an **Apple M3 Max with 36 GB unified memory**. Keep at
least 40 GiB free; 50 GiB or more is recommended for the Python environment, Qwen,
the retrieval models, and working artifacts. Source documents may live anywhere on the
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
- installs `document-qwen`, `document-qwen-smoke`, and `document-opencode` commands;
- runs the project test suite.

It does not search for, read, copy, or upload private documents. A successful finish
ends with `Bootstrap complete` and a passing test summary.

Optional installation checks:

```bash
test -f models/Qwen3.8-27B-4bit/config.json && echo "Qwen model: OK"
tesseract --list-langs | grep -E '^(eng|rus)$'
uv run --no-sync python -m pytest
opencode --version
command -v document-qwen document-qwen-smoke document-opencode
```

### 3. Start local Qwen

Run this from any directory and keep the first terminal open:

```bash
document-qwen
```

The server listens only on `127.0.0.1:8080`. Initial model loading can take a while.
After it starts, this request must succeed from a second terminal:

```bash
curl -fsS http://127.0.0.1:8080/v1/models
```

### 4. Verify agent tool calling

In the second terminal, also from any directory:

```bash
document-qwen-smoke
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
~/Documents/PrivateDocuments/project-1/                 # source documents
~/Documents/PrivateDocuments/project-1/.document-eater-workspace/ # local RAG and reports
~/Documents/document-eater/                             # application code and local models
```

Directories on an encrypted external SSD work as well, for example
`/Volumes/PrivateSSD/Documents/project-1`. Terminal and OpenCode must have permission
to access the selected folders under **System Settings → Privacy & Security**.

The hidden `.document-eater-workspace` directory is created automatically. It contains
the index, extracted text, graph, and reports, but is excluded from source discovery.
The same protection also recognizes older `audit-run` directories by their
`run-manifest.json`, so generated CSV/Markdown files cannot be ingested back into the
next audit. You may instead pass an absolute output directory elsewhere on the disk.
Never use the source directory itself as the output directory.
The BGE model cache does not move with the documents: the installed MCP configuration
points it back to `<application>/models/retrieval` with an absolute path.

### 6. Start OpenCode and run the first audit

Change into the folder that contains the documents and launch the installed document
profile there:

```bash
cd ~/Documents/PrivateDocuments/project-1
document-opencode
```

`document-opencode` loads a generated OpenCode configuration containing the absolute
application path. The MCP process uses the current document folder as its working
directory, so `.` and relative output paths refer to this folder rather than the code
repository.

Confirm that the selected model is
`local-docs/models/Qwen3.8-27B-4bit`, then send:

```text
Audit every supported document in this folder with local Qwen. Extract requirements,
obligations, deadlines, required documents, exceptions, and conflicting clauses.
Show FAIL, CONFLICT, and UNKNOWN first and cite exact source locations.
```

The installed `document-auditor` is the default primary agent. It calls
`audit_documents` once, remains silent while OCR/indexing/retrieval/model verification
runs, and returns one final summary plus the absolute `report_path`. It only interrupts
with a message when a real error requires action. OpenCode may still render its own
tool activity indicator; that UI is not an extra model response.

Follow-up retrieval is deliberately context-bounded. Search returns at most six short
snippets, audit items are paginated, page/unit reads use character windows, and graph
neighbors are paginated with truncated node text. These limits do not reduce the local
index or the generated report; they only prevent one MCP result from consuming the
12K agent context. The agent has eight total steps and at most four follow-up tool calls.

Repeating the same request does not repeat OCR, embeddings, or Qwen verification.
The corpus fingerprint, audit settings, and model are compared with the completed
manifest; unchanged work is reused. Changed source files rebuild automatically. Use
`force_rebuild=true` only when you intentionally want to discard that cache.

### 7. Inspect and continue working with the corpus

Open the path returned by the agent, normally:

```bash
open ~/Documents/PrivateDocuments/project-1/.document-eater-workspace/report.html
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

This refreshes the absolute application path stored in the generated OpenCode profile
and reinstalls the three launcher commands. Existing private documents and audit
directories outside the repository are not touched.

Start a new OpenCode session after updating. An already-open session still contains its
old oversized tool results and cannot recover that context merely because the profile on
disk changed.

### Troubleshooting

| Symptom | Action |
|---|---|
| `rus` is absent from `tesseract --list-langs` | Run `brew install tesseract-lang`, then retry. |
| Port 8080 is unavailable | Stop the old model server with `Ctrl+C`; do not expose a replacement on `0.0.0.0`. |
| `Local Qwen endpoint is unavailable` | Start `scripts/start-qwen-macos.sh` and verify `/v1/models`. |
| MLX tool-call smoke test fails | Do not use OpenCode automation; keep the server log and checkpoint/runtime versions for diagnosis. |
| `document-opencode` is not found | Rerun the bootstrap; it installs launchers into the Homebrew `bin` directory. |
| OpenCode cannot see document tools | Rerun the bootstrap, then start with `document-opencode` instead of plain `opencode`. |
| The agent repeats searches or page reads | Stop that session, update/bootstrap the project, then launch a fresh `document-opencode` session. The profile now paginates and caps every text-returning MCP tool. |
| macOS denies access to documents or an external SSD | Grant Terminal/OpenCode access under Privacy & Security → Files and Folders. |
| Memory pressure becomes high | Close other large applications and restart the MLX server. If pressure persists on the 36 GB Mac, temporarily restore the OpenCode model limit from 12k to 8k; keep the 4 GB prompt-cache cap. |
| Report contains only `UNKNOWN` | Confirm that model verification was requested; otherwise this is the intentional candidate-only behavior. |

If the MLX server or smoke test fails, do not automatically switch to a cloud model:
that would change the privacy boundary for every document fragment returned to the
agent.

## Privacy boundary

- Source documents stay at the path you choose. Extracted artifacts, embeddings, and graph
  data stay in the selected local output directory or the ignored `data/`,
  `artifacts/`, and `models/` defaults.
- The LLM adapter accepts loopback endpoints. The primary MLX server is local; an
  optional remote server must be reached through an SSH tunnel and receives only
  retrieved evidence blocks rather than entire documents.
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
| M3 Max 36 GB primary | MLX 4-bit | 16.05 GB weights | local private inference; 12k bounded context |
| Vast fallback | base Q4_K_M | 16.46 GB file | 24 GB GPU, only when data policy permits |
| Vast primary reference | official BF16 | ~56 GB | 80 GB GPU, evaluation/reference runs |
| Vast manual fallback | OBLITERATUS Q4_K_M | 16.81 GB file | separate 24 GB profile, loaded only when requested |

The published 262k context length is not the operating target for this laptop. RAG
uses a small, explicit evidence budget; the local MLX profile starts at 12k context.
An experimental 16k limit is plausible on 36 GB because this hybrid model uses full
attention in only 16 of 64 layers, but it remains opt-in until memory pressure and
latency are measured on the target Mac with Qwen plus both retrieval models loaded.
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

PyMuPDF is used by the PDF ingestion adapter and is AGPL/commercial dual-licensed. Internal
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
persistent prompt-cache pool at 4 GB. OpenCode separately caps requests at 12288
tokens. `mlx-lm==0.31.3` runs in an isolated `uvx` environment because that runtime
requires Transformers 5 while the local BGE reranker is deliberately pinned to
Transformers 4.x.

The abliterated fallback remains a separate llama.cpp/GGUF path through
`scripts/start-qwen-gguf-fallback.sh`; pass `--profile abliterated` to the CLI when
using that server. It is never selected automatically. Vast profiles remain in
`config/model-profiles.toml` for cases where local latency is unacceptable and the
data owner explicitly permits third-party processing.

## OpenCode integration details

The checked-in `opencode.json` is the source template for the stable OpenCode
configuration installed by Homebrew. During bootstrap, an absolute-path copy is written to
`~/.config/opencode/document-eater.json`. The `document-opencode` launcher selects
that copy through `OPENCODE_CONFIG`. It:

- selects the local MLX Qwen endpoint as the OpenCode model;
- selects the quiet, bounded `document-auditor` primary agent;
- starts `document-eater-mcp` over stdio;
- exposes `audit_documents`, `prepare_corpus`, `search_corpus`,
  `list_audit_items`, `get_audit_summary`, `read_document_page`, and
  `graph_neighbors` directly to the model.

The model sees compact views rather than whole machine artifacts: search defaults to
four snippets (six maximum), audit lists default to three items, page/unit reads default
to 4,000 characters, and graph queries default to twelve edges. The complete
`audit.json`, `index.sqlite3`, extracted artifacts, and HTML report remain unchanged on
disk and can be paged through when more evidence is genuinely needed.

The generated MCP command selects the application environment with `uv --project`
while leaving MCP `cwd` as the OpenCode workspace. This is what allows code and model
files to stay under `~/Documents/document-eater` while OpenCode is launched from any
unrelated document folder.

The agent denies every non-document tool, including shell, file edits, web access, and
tools inherited from unrelated global MCP configurations. Only `document-eater_*`
tools are allowed, and identical tool calls are not retried. This both protects private
documents and avoids wasting the 12K local context on progress chatter or loops.

Do not switch OpenCode to a cloud model for this task: MCP returns extracted document
text to the controlling model. A cloud OpenCode model therefore breaks the local-only
privacy boundary even though OCR and the index still run locally.

The MCP entry follows the supported direct `mcp.document-eater` shape and is enabled
explicitly. See the current [OpenCode MCP documentation](https://opencode.ai/docs/mcp-servers).

## Lower-level commands

```bash
uv run --no-sync document-eater inspect path/to/document.pdf
uv run --no-sync document-eater ingest path/to/document.docx --output artifacts
uv run --no-sync document-eater index artifacts --quality --database data/index.sqlite3
uv run --no-sync document-eater search "условия расторжения" --database data/index.sqlite3
uv run --no-sync document-eater ask "каков срок уведомления?" --database data/index.sqlite3
uv run --no-sync document-eater ask "..." --profile abliterated --database data/index.sqlite3
uv run --no-sync python -m pytest
```

`inspect` is PDF-specific: it performs no OCR and reports which pages would use the
native text layer or OCR. `ingest` accepts `.pdf`, `.docx`, `.xlsx`, `.xml`, `.csv`,
`.md`, and `.txt`; the OCR flags affect only PDFs. It writes:

- `<document-id>/document.json` — normalized logical units and blocks with provenance;
- `<document-id>/graph.json` — `contains`, `next`, and section membership edges;
- `<document-id>/manifest.json` — source hash and pipeline settings.

The source document is not copied into the artifact directory.

### Supported source formats

| Format | Extraction and provenance |
|---|---|
| PDF | native text or local Tesseract OCR; page and bounding box |
| DOCX | paragraphs, heading styles, and table rows; paragraph/table-row locator |
| XLSX | worksheets, rows, cells, number formats, formulas, and cached values when present |
| XML | entity-safe structural traversal; XML element path and attributes |
| CSV | detected delimiter/header; row, column label, and raw value |
| Markdown | heading-aware blocks with original line ranges |
| TXT | UTF-8/UTF-16/Windows-1251 text; original line ranges |

Excel formulas are preserved as evidence but never executed. Macro-enabled workbooks,
legacy `.xls`, Word pagination, embedded charts/images, and handwritten content are
not interpreted in this version. DOCX has no reliable page number without rendering,
so reports cite the exact paragraph or table row instead of inventing a page.

## What this build does and does not prove

Implemented and tested: mixed PDF/DOCX/XLSX/XML/CSV/Markdown/TXT ingestion, page-level
native/OCR routing, format-aware provenance, structural graph,
BM25+dense RRF, BGE-M3 dense+sparse generation on Apple Metal, multilingual BGE
reranking, rule-based requirement candidates, conservative Qwen JSON verdict
validation, HTML/CSV/JSON reports, and an MCP stdio handshake. A quality-RAG
end-to-end mixed Word/Excel/text runs found the expected requirements and evidence;
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
