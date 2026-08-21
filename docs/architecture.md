# Architecture contract

## Data flow

```text
private source
  -> format router
     -> PDF: native page extraction or local OCR
     -> DOCX: paragraphs, heading styles, table rows
     -> XLSX: sheets, rows, cells, formulas and cached values
     -> Markdown/XML/CSV/TXT: structure-aware native extraction
  -> provenance-preserving blocks
  -> deterministic structural graph
  -> local BM25 + BGE-M3 dense + BGE-M3 learned-sparse candidates
  -> reciprocal-rank fusion
  -> local bge-reranker-v2-m3 on the top candidate pool
  -> evidence budget with page/paragraph/table/sheet/cell/path/line citations
  -> Qwen/Qwen3.8-27B MLX 4-bit primary (local M3 Max, 36 GB)
     -> optional SSH-tunnelled Vast server when policy permits
     -> optional manual OBLITERATUS fallback on explicit policy refusal
  -> answer + cited evidence + run manifest
```

No complete source file needs to cross the LLM boundary. Only retrieved text blocks
are passed to the controlling model.

## Multimodal extension boundary

Images and audio belong behind the same `ingest_document` boundary but require
separate local models and evidence types:

```text
PNG/JPEG -> OCR blocks + local vision captions/objects -> text and image vectors
audio    -> local ASR segments + speaker labels        -> text vectors
                                                        + timestamp graph edges
```

The first multimodal implementation must keep pixel boxes or time ranges on every
derived block, store the extractor/model revision in the manifest, and evaluate OCR,
caption, and transcription evidence separately. The current Qwen3.8 27B profile is
text-only; adding image tensors to its prompt is not a supported shortcut. A vision
or audio adapter must be optional so the 36 GB laptop can unload it before Qwen and
the BGE reranker are started. Sending rendered pages, images, or recordings to Vast
is a separate privacy decision from sending selected text blocks.

The fallback is never triggered by phrases such as "insufficient evidence" or by a
failed citation check. Automatic refusal classification is disabled until a held-out
set can distinguish policy refusal from correct uncertainty and data-quality errors.

## OpenCode workspace boundary

Application code and document workspaces are deliberately independent. Bootstrap
generates a custom OpenCode configuration with an absolute application path and
installs the `document-opencode` launcher. Its MCP command uses
`uv run --project <application-root>` to select the locked Python environment while
leaving MCP `cwd` equal to the OpenCode workspace. Therefore relative document and
output paths resolve from the folder in which `document-opencode` was launched, not
from this source repository.

The generated configuration is stored separately from the user's ordinary global
OpenCode configuration and selected through `OPENCODE_CONFIG`; bootstrap does not
overwrite unrelated providers or preferences.

## Graph layers

The graph is deliberately split into evidence levels:

1. **Structural, deterministic (implemented):** documents contain logical units
   (PDF pages, Excel sheets, or a Word/text document); units contain blocks, blocks
   follow one another, and body blocks belong to headings.
2. **Explicit references (next):** numbered cross-references, footnotes, table and
   figure references resolved by parser rules.
3. **Semantic dependencies (later):** Qwen proposes typed edges such as
   `requires`, `contradicts`, `defines`, and `exception_to`. Every edge must carry
   source block IDs, a confidence score, extractor version, and validation state.

An LLM-proposed semantic edge is never treated as ground truth merely because it
was emitted as JSON.

## Retrieval evaluation gate

The target retriever is hybrid: SQLite FTS5/BM25 plus multilingual dense vectors,
reciprocal-rank fusion, local cross-encoder reranking, then graph-neighbor expansion.
Promotion requires a private held-out set with questions, expected evidence block
IDs, and answer constraints. Report at least Recall@5/10/20, MRR@10, nDCG@10,
citation precision, answer faithfulness, latency, and peak memory.

The lexical index is the initial control. Dense or graph expansion stays disabled
if it does not improve held-out evidence recall.

## Remote execution boundary

- The GPU process binds to `127.0.0.1`, never `0.0.0.0`.
- The laptop connects with an SSH local port forward.
- Only selected text blocks and the question are transmitted.
- Instance disks are ephemeral; no cloud sync, model UI, request logs, or telemetry.
- A teardown checklist must verify process stop, volume deletion, and instance
  destruction. These controls reduce exposure but do not override a policy that
  prohibits third-party processing.
