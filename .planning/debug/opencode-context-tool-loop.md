---
status: resolved
trigger: "OpenCode document agent still appears to loop; suspected context exhaustion"
created: 2026-08-22T22:05:00+03:00
updated: 2026-08-22T22:32:00+03:00
---

## Current Focus

hypothesis: CONFIRMED AND FIXED — unbounded model-facing MCP results exhausted the local context and made repeated changed-argument retrieval calls likely.
test: Completed against the source tree, reinstalled runtime package, OpenCode 1.18.21 config parsing, adversarial synthetic payloads, and the full local regression suite.
expecting: Every result-heavy tool now stays near an 8,000-character envelope with continuation metadata; the installed agent has eight steps, bounded follow-ups, and a 12K context declaration.
next_action: On the reporting M3 Mac, pull the fix, rerun bootstrap, start a fresh session, and verify one broad non-sensitive request reaches a final response.

## Symptoms

expected: One audit call followed by one final answer, or a small number of bounded retrieval calls followed by one final answer.
actual: The OpenCode agent appears to repeat document operations and consume local context.
errors: No shareable transcript; documents are private. The local machine used for development has no matching recent Document Eater session, so diagnosis uses code/config evidence rather than document contents.
reproduction: Run the installed `document-opencode` from a document folder, ask a broad corpus question, and observe repeated MCP calls after large retrieval/audit-detail results.
started: Reported after the previous cache/quiet-agent hardening.

## Eliminated

- hypothesis: The source corpus recursively ingests `.document-eater-workspace` and necessarily rebuilds forever.
  evidence: The existing implementation excludes the active workspace and reuses unchanged audit fingerprints; tests cover unchanged reuse.
  timestamp: 2026-08-22T22:05:00+03:00
- hypothesis: The installed MCP runs stale Python code that differs from the current checkout.
  evidence: Byte-for-byte comparisons show `.venv` and checkout copies of `mcp_server.py`, `index.py`, `audit.py`, and `install_opencode.py` all match; the generated OpenCode config also matches the checked-in 8192/2048 limits and 12-step agent contract.
  timestamp: 2026-08-22T22:07:00+03:00

## Evidence

- timestamp: 2026-08-22T22:05:00+03:00
  checked: `opencode.json`
  found: Model context is 8192, output allowance is 2048, agent allows 12 steps and all seven Document Eater MCP tools.
  implication: Only about 6K tokens remain for prompt, tool schemas, conversation, and tool results.
- timestamp: 2026-08-22T22:05:00+03:00
  checked: `src/document_eater/mcp_server.py` and `src/document_eater/index.py`
  found: `search_corpus` defaults to ten chunks of up to roughly 1800 characters each; `list_audit_items` returns the entire filtered audit; `read_document_page` returns a whole page/unit.
  implication: A single normal tool result can dominate or exceed the useful context budget.
- timestamp: 2026-08-22T22:05:00+03:00
  checked: OpenCode 1.18.21 local database/log metadata
  found: No recent matching Document Eater session exists on this development Mac; the reported run happened on the other Mac and cannot be inspected without its private transcript.
  implication: We can prove the context hazard and harden it, but cannot attribute the exact visible repetition to a named tool yet.
- timestamp: 2026-08-22T22:05:00+03:00
  checked: Current OpenCode permissions documentation
  found: `doom_loop` triggers after three identical tool calls; the current agent sets it to deny.
  implication: It is a last-resort stop, not a substitute for bounded responses and deterministic tool use.
- timestamp: 2026-08-22T22:00:30+03:00
  checked: Complete `list_audit_items` and audit serialization path
  found: `list_audit_items` returns every matching item with no `limit`, cursor, projection, or truncation. Each serialized item includes the full requirement plus up to ten `retrieved_evidence` records, whose previews alone can total 5,000 characters per requirement.
  implication: The response grows with the number of requirements and can exceed the model input budget after only a few items.
- timestamp: 2026-08-22T22:01:15+03:00
  checked: Complete `search_corpus`, chunking, and `read_document_page` paths
  found: Search defaults to ten roughly 1,800-character chunks but accepts an unrestricted caller-supplied limit; a single source block can also exceed the nominal chunk size. `read_document_page` returns the selected unit with every block and has no text budget.
  implication: Multiple independent MCP tools permit responses larger than the configured context headroom; this is a cross-tool output-contract defect rather than one bad document parser.
- timestamp: 2026-08-22T22:02:00+03:00
  checked: `tests/test_mcp_paths.py`, `tests/test_opencode_install.py`, and `tests/test_audit.py`
  found: Tests cover path resolution, agent permissions, caching, and audit semantics but do not assert response byte/token budgets, pagination metadata, or bounded page/search/list results.
  implication: Existing CI could pass while oversized MCP results remained possible.
- timestamp: 2026-08-22T22:04:30+03:00
  checked: Synthetic `list_audit_items` responses containing no private data
  found: One item serialized to 6,465 characters, five to 32,325, and ten to 64,650. Growth was linear and the function returned all items unchanged.
  implication: Five routine audit-detail items can consume roughly the entire 6,144-token theoretical input headroom under common text-token ratios, and ten clearly exceed it; the response contract has no safety margin for prompts, schemas, or conversation.
- timestamp: 2026-08-22T22:05:00+03:00
  checked: Python module path used by the synthetic probe
  found: `uv run --no-sync` imported `document_eater` from `.venv/lib/python3.12/site-packages`, matching the same launch mode used by the installed MCP command.
  implication: The installed runtime copy must be compared with the checkout before source-level conclusions are treated as runtime facts.
- timestamp: 2026-08-22T22:07:00+03:00
  checked: Installed runtime package and generated OpenCode config
  found: Runtime and checkout files match byte-for-byte for all relevant modules. The installed config uses OpenCode 1.18.21, context 8192, output 2048, 12 steps, and the same seven-tool prompt/permission surface.
  implication: The oversized-response defect is active in the installed workflow and is not explained by stale installation state.
- timestamp: 2026-08-22T22:08:15+03:00
  checked: Valid synthetic search and page artifacts containing no private data
  found: Default `search_corpus` returned ten hits as 20,265 JSON characters. `read_document_page` returned a single 100,000-character block as 100,199 JSON characters. Neither result was truncated and the search limit accepts caller values without an upper bound.
  implication: Oversized results are directly reproducible through the actual installed MCP functions, independent of private corpus content or the unavailable transcript.
- timestamp: 2026-08-22T22:13:30+03:00
  checked: OpenCode 1.18.21 primary documentation and tagged source
  found: Model `limit.context` is the maximum input context and `limit.output` reserves generation room; `steps` is only a maximum agent-iteration count; `doom_loop` only triggers after three identical-input calls. Tagged source sends MCP results through a generic limit of 50 KiB/2,000 lines, while compaction summaries retain at most the first 2,000 characters of each tool result.
  implication: OpenCode has a transport guard, but it is not context-aware for this 8K model and compaction is lossy. Calls with changed arguments evade `doom_loop`, and a 12-step ceiling permits many context-consuming calls before forcing text.
- timestamp: 2026-08-22T22:16:30+03:00
  checked: Concurrent candidate patch tests `test_mcp_context_budget.py` and `test_opencode_install.py`
  found: Six targeted tests passed. The uncommitted candidate bounds search count/text, paginates compact audit items, windows page text, reduces steps, and adds an explicit tool allowlist.
  implication: The proposed application-level mitigation is mechanically viable, but this is candidate verification only; the patch was authored concurrently and the full regression/config checks remain pending.
- timestamp: 2026-08-22T22:03:30+03:00
  checked: Stable concurrent candidate patch with full local validation
  found: `pytest -q` passed 26 tests; `ruff check .` passed; `ruff format --check .` reported 26 files formatted. OpenCode 1.18.21 parsed the candidate config with eight steps and the Document Eater tool allowlist.
  implication: The candidate does not regress the tested pipeline/config contracts and its configuration is accepted by the installed OpenCode version.
- timestamp: 2026-08-22T22:04:00+03:00
  checked: Stable candidate response envelopes using synthetic non-private worst-case text
  found: Audit output was 7,593 characters without evidence and 6,105 with evidence; search was 7,161; page output was 5,398 with 5,000 returned text characters. `graph_neighbors` still returned 15,582 characters for its maximum 20-edge page.
  implication: Audit/search/page now have effective context budgets and continuation metadata. Graph results are count-bounded but remain outside the shared 8,000-character target, so the candidate is not yet uniformly context-safe.
- timestamp: 2026-08-22T22:04:30+03:00
  checked: Final graph budget candidate and regression suite
  found: After adding total-budget packing, the same 100-edge synthetic graph returned 10 edges plus continuation metadata in 8,069 serialized characters instead of 15,582. The stable candidate passes 27 tests, `ruff check .`, and `ruff format --check .`.
  implication: All result-heavy tools now have approximately 8K-character model-facing envelopes. Remaining verification is an end-to-end run on the reporting M3 Mac, not a known unbounded code path.
- timestamp: 2026-08-22T22:32:00+03:00
  checked: Reinstalled `.venv` runtime and generated `~/.config/opencode/document-eater.json`
  found: The non-editable runtime package now matches `src/document_eater/mcp_server.py` byte-for-byte; runtime signatures expose all pagination/budget parameters; the installed profile resolves to 12,288 context tokens, 2,048 output tokens, eight steps, and the document-only tool allowlist.
  implication: The fix is active on this development Mac rather than merely present in the checkout. The reporting Mac must rerun bootstrap because an already-open session and old non-editable runtime cannot adopt it automatically.

## Resolution

root_cause: The installed HEAD implementation exposes unpaginated or insufficiently bounded MCP results (`search_corpus`, `list_audit_items`, `read_document_page`, and graph detail) to an OpenCode agent configured for only 8,192 context tokens with 2,048 reserved output tokens. Reproduced results range from 20,265 to 100,199 JSON characters; OpenCode 1.18.21 only truncates at 50 KiB and compaction preserves just a 2,000-character tool preview. This context-contract mismatch, not recursive ingestion or a stale install, causes loss of retrieval state and makes changed-argument follow-up calls evade the identical-call doom-loop guard. The exact tool sequence seen on the reporting Mac remains unavailable, so attribution of each repeated call is inferential, but the causal capacity defect is directly reproduced.
fix: Added application-level structured pagination and deterministic total response budgets instead of relying on OpenCode truncation. Audit/search/page/graph results stay near an 8,000-character envelope with compact projections and explicit continuation metadata; the agent is limited to eight steps and at most four follow-up calls; non-document tools are removed from its visible tool surface. The local M3 Max profile now declares 12,288 context tokens with 2,048 output tokens. The package and generated OpenCode profile were reinstalled locally.
verification: Baseline synthetic probes reproduced 64,650-character audit lists, 20,265-character default searches, and 100,199-character page reads without private data. The fixed implementation passes 27 tests plus ruff lint/format and OpenCode config parsing. Synthetic envelopes measured 7,593 characters for audit, 7,161 for search, 5,398 for page, and 8,069 for graph with continuation metadata. The installed runtime matches source byte-for-byte and exposes the new signatures. Mechanical verification is complete; a live Qwen/OpenCode run remains unavailable because the model server is not running on this development Mac.
files_changed: [README.md, opencode.json, config/model-profiles.toml, scripts/start-qwen-macos.sh, src/document_eater/mcp_server.py, tests/test_mcp_context_budget.py, tests/test_opencode_install.py]
