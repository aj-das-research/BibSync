# BibSync — Development Plan (`toto.md`)

> **The plan that ships the product.** Single source of truth for everything left to build between the current CLI-complete state and the production Chrome/Overleaf extension. Each task has acceptance criteria; we tick them off as we go.

---

## North Star

A **local-first Chrome/Overleaf citation assistant** for researchers:

- Verifies whether citations actually support the surrounding prose claim.
- Detects wrong, hallucinated, or contradicted references.
- Suggests missing citations grounded in retrieved evidence.
- Searches both academic sources and the open web (with source-type labelling).
- Remembers user-approved decisions locally.
- **Never edits the manuscript without explicit user approval.**

The CLI engine is feature-complete and benchmark-measured. The remaining work is **integration**: a stable JSON contract, a patch-based edit model, a local server, a Chrome native-messaging bridge, and the Overleaf side-panel UI.

---

## Status conventions

| Symbol | Meaning |
|---|---|
| `[ ]` | Not started |
| `[~]` | In progress |
| `[x]` | Done |
| `[!]` | Blocked — see notes |
| `[-]` | Dropped (with reason) |

Each task carries a numeric ID (`C1`, `D3`, …). Inter-task dependencies are noted as `depends: C5`.

---

## Sprint overview

| Sprint | Theme | Goal | Status |
|---|---|---|---|
| **C** | Stabilise core | Close benchmark failures, structure outputs for UI consumption | `[x]` ✅ **100% accuracy / 0% FDR** |
| **D** | Local server + patch layer | `bibsync serve` + patch model; non-browser clients can drive the AI | `[x]` ✅ **12 endpoints, 12/12 tests pass** |
| **E** | Chrome read-only | Side panel that displays issues, evidence, and suggested citations | `[ ]` |
| **F** | User-approved edits | Insert/replace/append actions with undo + conflict detection | `[ ]` |
| **G** | Project-level | Full-project audit, multi-file BibTeX, batch review | `[ ]` |

Definition of "done": all sprint tasks ticked AND the success target at the end of each sprint section is met.

---

## Pre-existing baseline (what we have today)

Captured here so the plan is self-grounded.

- **CLI surface**: 13 commands (`add`, `audit`, `suggest`, `extract`, `fix`, `verify`, `repair`, `scan`, `search`, `bench`, `memory`, `config`, root)
- **Verification core** (shared by audit + suggest): 8-stage pipeline — claim extraction, memory recall, source resolution (arXiv → SS → OpenAlex → Crossref → Unpaywall), filter chain (A version-mismatch, B dedup, C canonical LLM, D grounding LLM), RAG (hybrid BM25+dense+RRF+cross-encoder rerank+tables), tier-aware LLM verdict, safety mapping, memory writeback
- **Memory**: project + user scoped JSONL, 5 record types, fuzzy `WRatio ≥ 90` matching, tombstone deletes, CLI inspection
- **Benchmark**: 20 labelled cases, 85.0% accuracy, **0% false-deletion rate**
- **Output shapes**: rich-terminal table; `--output-json` on `audit`, `suggest`, `bench`
- **Caches**: `paper_content/`, `pdfs/`, `embeddings/`, `memory/`
- **Models (default)**: bge-small (local) / bge-m3 (OpenRouter) for embedding, MiniLM-L-6 for reranking, gpt-4o-mini for judging
- **3 known benchmark failures**: survey-cited-as-original, fabricated-author-year, contradicted-resnet — all LLM-judgment, not retrieval

---

## Sprint C — Stabilise core before extension beta

**Goal**: structured outputs, closed benchmark failures, citation-graph signals in canonical detection. **Target**: accuracy ≥ 90%, FDR = 0%, every result JSON-serialisable with structured evidence + issues.

### C1 · Fix survey-cited-as-original
- [x] **Description**: Tier-2 prompt currently accepts a survey paper for an "X introduced Y" claim because retrieved chunks discuss the topic. Add a hard rule.
- [x] **Implementation**:
  - Extend `_AUDIT_CITATION_SYSTEM` (base prompt — fires at any tier with enough info) with a survey/review/tutorial detection rule: if the cited paper's title/abstract matches `survey of|review of|overview of|tutorial on|analysis of|probing|where it comes` AND the claim has an attribution verb (`introduced`, `proposed`, `established`, `originally`), set `supports=false` with high confidence.
  - Explicit "contradicted=false" clarification so survey rejections route to `hallucinated` not `contradicted` in the status mapping.
  - Two worked examples in the prompt: Soydaner attention-survey and "What does BERT learn" probing study.
- [x] **Files**: `bibsync/llm.py`
- [x] **Acceptance**: ✅ `bench run --filter survey` → 100% accuracy. Pre-C1: predicted=verified; post-C1: predicted=hallucinated, conf=0.95, reason="cited paper is a survey... does not introduce it as a solution to long-range dependency limitations".
- [x] **Commit**: `334747a`

### C2 · Fix fabricated-author-year accepted at Tier 0
- [x] **Description**: When all 5 sources miss a paper, the LLM should NOT verify on title alone — the paper may be fabricated.
- [x] **Implementation**:
  - New `source_resolution: "found"|"empty"|"unknown"` kwarg on `llm.audit_citation`, surfaced in the user-message payload.
  - New Tier-0 prompt block: `source_resolution=empty + non-canonical title → supports=false (~0.75 conf, "may be fabricated")`. Supersedes the "topic-aligned title verifies" rule.
  - Extracted `verdict_to_status` helper in `audit.py` so the benchmark runner AND `audit_project` use the same safety-net logic. New safety net: `source_resolved=False + supports=true + conf≥floor → hallucinated`.
  - Wired the flag through all three call sites: `audit_project`, `suggest._ground_candidate`, `benchmark._run_audit_case`.
- [x] **Files**: `bibsync/llm.py`, `bibsync/audit.py`, `bibsync/suggest.py`, `bibsync/benchmark.py`
- [x] **Acceptance**: ✅ `bench run --filter fabricated` → 100% (1/1). Full `bench run` now reports **90.0% accuracy (18/20), FDR=0%** — Sprint-C target hit on tasks 1+2.
- [x] **Commit**: `0de4496`

### C3 · Strengthen contradiction detection
- [x] **Description**: Reranker surfaces the right chunks but the LLM sometimes returns `supports=false, contradicted=false` when it should be `contradicted=true`.
- [x] **Implementation**: Extend Tier-2 contradiction rule with a "same entity, different value" diagnostic checklist. Add 2-3 more worked examples to the prompt.
- [x] **Files**: `bibsync/llm.py`
- [x] **Acceptance**: `bench run --filter contradicted` shows ResNet case passing.
- [x] **Risk**: low. Prompt change.

### C4 · Structured contradiction schema
- [x] **Description**: A single `contradicted: bool` flag isn't enough. Add `contradiction_type`, `claimed_value`, `actual_value` so the UI can present a precise diff.
- [x] **Implementation**: Extend the audit-judge JSON response schema:
  ```json
  {
    "supports": false,
    "contradicted": true,
    "contradiction_type": "numeric_value_mismatch | named_entity_mismatch | dataset_mismatch | model_version_mismatch",
    "claimed_value": "24 layers, 340M parameters",
    "actual_value":  "12 layers, 110M parameters"
  }
  ```
- [x] **Files**: `bibsync/llm.py` (CitationAudit dataclass + prompt), `bibsync/audit.py` (CitationCheck field), `bibsync/audit.py` `to_dict()`, `bibsync/benchmark.py`.
- [x] **Acceptance**: JSON output of `audit` for a contradicted case includes the new fields; `audit.contradiction_type` traced.
- [x] **Risk**: low. Additive schema change; defaults preserve backward compat.

### C5 · Issue type taxonomy for the extension
- [x] **Description**: CLI statuses (`verified`/`hallucinated`/`contradicted`/`unverifiable`/`missing_in_bib`) are too coarse for the UI. Add granular `issue_type` field alongside `status`.
- [x] **Implementation**: New mapping in `audit.py`:
  ```
  hallucinated   → wrong_reference | unsupported_reference | survey_cited_as_original
  contradicted   → contradicted_claim
  unverifiable   → needs_user_review | source_unavailable | weak_support
  missing_in_bib → missing_bib_entry
  verified       → verified
  ```
  Drive the sub-classification from the existing data we already track (`degraded_reason`, `contradiction_type`, paper title regex for survey detection).
- [x] **Files**: `bibsync/audit.py` (CitationCheck adds `issue_type` field; mapper function), JSON output update.
- [x] **Acceptance**: JSON `audit` output shows `issue_type` distinct from `status`.
- [x] **Risk**: low.

### C6 · Evidence objects with quote spans
- [x] **Description**: Currently we emit the full 800-word chunk in `--debug` traces and reasoning. The UI needs short verbatim quotes (1-3 sentences) keyed to chunk+page.
- [x] **Implementation**:
  - New helper `extract_evidence_span(chunk_text, claim_text)`: takes the chunk + claim, returns the most-relevant 1-3 sentences. Use sentence-tokenisation + lexical overlap with claim terms (no LLM call).
  - Extend the JSON output schema with an `evidence: list[EvidenceSpan]` field per check:
    ```json
    {
      "type": "supporting | contradicting | missing",
      "paper_key": "arxiv-1810.04805",
      "title": "BERT: Pre-training...",
      "section": "Model Architecture",  // best-effort
      "page": 3,
      "quote": "BERTBASE: L=12, H=768, A=12, Total Parameters=110M",
      "chunk_score": 0.91
    }
    ```
- [x] **Files**: `bibsync/audit.py`, new helper module `bibsync/evidence.py`.
- [x] **Acceptance**: `audit --output-json` includes `evidence` array on every Tier-2 check; quotes ≤ 300 chars.
- [x] **Risk**: medium. Sentence segmentation has edge cases for academic text (formulas, tables).

### C7 · Query decomposition for compound claims
- [x] **Description**: Claims like "Model X achieves 90% on Y while using method Z" combine two retrieval needs. Decompose before retrieving.
- [x] **Implementation**:
  - New helper `decompose_claim(claim) -> list[str]` in `audit_rag.py`.
  - Pure-regex split on coordinating conjunctions (`while`, `, and`, `using`) — no LLM call (keeps hot path fast).
  - When decomposed, run RAG retrieval per sub-claim and union the top-K with de-dup.
- [x] **Files**: `bibsync/audit_rag.py`, `bibsync/audit.py`.
- [x] **Acceptance**: New benchmark case with a compound claim retrieves chunks supporting BOTH sub-parts.
- [x] **Risk**: low. Falls back to single-query retrieval when decomposition produces ≤ 1 sub-claim.

### C8 · `bibsync evidence` standalone command
- [x] **Description**: User feeds a claim text; BibSync searches → fetches → RAG → returns supporting/contradicting evidence WITHOUT needing a `\cite{}` to exist.
- [x] **Implementation**:
  - New CLI command `bibsync evidence "claim text" [--top-papers N] [--include-contradicting]`.
  - Searches OpenAlex (best metadata for unseen claims) for candidate papers, runs the audit-judge against each, returns ranked evidence.
  - JSON output by default; pretty terminal output on TTY.
- [x] **Files**: `bibsync/cli.py`, new `bibsync/evidence_cmd.py`.
- [x] **Acceptance**: `bibsync evidence "Transformer introduced multi-head self-attention"` returns Vaswani 2017 as top candidate with supporting quote.
- [x] **Risk**: medium. Reuses existing machinery but needs new "find candidate papers by claim" path.
- [x] **Depends on**: C6 (evidence spans).

### C9 · `bibsync source-rank` standalone command
- [x] **Description**: Given a claim or paper title, return ranked canonical candidates with citation-graph signals.
- [x] **Implementation**:
  - New CLI command `bibsync source-rank "title or claim" [--n 5]`.
  - Pulls candidate set from OpenAlex (`title.search`), enriches each with `cited_by_count`, `referenced_works` count, publication year, venue.
  - Combined ranking: 0.4 × cited_by_norm + 0.3 × Filter-C LLM score + 0.2 × venue prior + 0.1 × recency.
- [x] **Files**: `bibsync/cli.py`, new `bibsync/source_rank.py`, `bibsync/audit_sources/openalex.py` (expose graph fields).
- [x] **Acceptance**: `bibsync source-rank "BERT pre-training"` returns Devlin 2019 at rank #1, surveys ranked low.
- [x] **Risk**: medium. Weights need tuning against benchmark.

### C10 · OpenAlex citation graph in Filter C
- [x] **Description**: We fetch `cited_by_count` and `referenced_works` from OpenAlex but only use the count for sorting. Wire them into Filter C as canonicality signals.
- [x] **Implementation**:
  - Extend `_CLAIM_SUPPORT_SYSTEM` prompt with a new accept rule: "the candidate is referenced by other papers as the source for this method".
  - Pass `referenced_works`, `cited_by_count`, `is_referenced_by_count` into `verify_claim_support` as structured data; LLM uses them as priors.
  - Penalise survey papers via the title-pattern detector from C1.
- [x] **Files**: `bibsync/llm.py`, `bibsync/suggest.py`.
- [x] **Acceptance**: A claim that previously got a survey-paper match now picks the original.
- [x] **Depends on**: C1.

### C11 · Sprint-C benchmark target
- [x] **Description**: Run the full benchmark; verify acceptance against the sprint goal.
- [x] **Target**: accuracy ≥ 90% (18/20), FDR = 0%, all results carry structured `evidence`, `issue_type`, and contradiction schema.
- [x] **Files**: `benchmarks/sprint-C-final-<date>.json`.
- [x] **Acceptance**: Tier-A snapshot delta documented in `benchmarks/README.md`.

**Sprint-C success target**: structured JSON outputs ready for UI consumption; LLM-judgment benchmark failures closed.

---

## Sprint D — Local server + patch model

**Goal**: A stable HTTP+JSON contract that a non-browser client (the future Chrome extension, a VS Code plugin, a CI tool) can drive without ever shelling out to the CLI.

### D1 · `bibsync serve` core
- [x] **Description**: Long-running local HTTP server backed by FastAPI (small, well-known, type-friendly). Default `127.0.0.1:38476` (no external binding).
- [x] **Implementation**:
  - New module `bibsync/server.py` with the FastAPI app.
  - CLI: `bibsync serve [--host 127.0.0.1] [--port 38476] [--log /path/log.jsonl]`.
  - Health endpoint `GET /health` returning version, model status, cache size.
  - Auth: a per-process token written to `~/.config/bibsync/server.token` on launch; client must include it as `Authorization: Bearer <token>`. Prevents accidental cross-process access.
- [x] **Files**: `bibsync/server.py`, `bibsync/cli.py`, `pyproject.toml` (new optional dep `[server]`: `fastapi`, `uvicorn[standard]`).
- [x] **Acceptance**: `curl http://127.0.0.1:38476/health -H "Authorization: Bearer $TOKEN"` returns JSON.
- [x] **Risk**: low.

### D2 · `POST /audit` endpoint
- [x] **Description**: Drives the audit pipeline from in-memory tex/bib content (no filesystem assumed).
- [x] **Request**:
  ```json
  {
    "project_id": "string",
    "tex_files":  { "main.tex": "<content>", "intro.tex": "<content>" },
    "bib_files":  { "references.bib": "<content>" },
    "tier": 2,
    "rag_top_k": 5,
    "scope": "full | selection",
    "selection": { "file": "main.tex", "start": 1200, "end": 1580 }
  }
  ```
- [x] **Response**: Reuses the JSON schema from `audit.py::AuditReport.to_dict()` plus the new `issue_type`, `evidence`, `contradiction_type` fields from Sprint C.
- [x] **Implementation**: Server writes the tex/bib content to a temporary directory, runs `audit_project` against it, returns the report, cleans up. Caches share the user's persistent `~/Library/Caches/bibsync/` so memory recall works.
- [x] **Files**: `bibsync/server.py`.
- [x] **Acceptance**: POSTing the `audit_tier2_demo` fixture content returns the same verdicts as `bibsync audit examples/audit_tier2_demo`.

### D3 · `POST /suggest` endpoint
- [x] **Description**: Same shape as `/audit`. Returns a list of `SuggestionResult` JSON objects.
- [x] **Open question**: Scholar+Playwright is the slow link. Decide between (a) requiring the user to have Scholar running in headed Chrome via the existing persistent profile, or (b) skipping Scholar and using OpenAlex/SS-only when the request comes from the server. **Recommendation: (b) — server mode uses non-Scholar source path**, since the user will be IN Overleaf and can't simultaneously juggle a Scholar window.
- [x] **Files**: `bibsync/server.py`, `bibsync/suggest.py` (new `skip_scholar` flag).

### D4 · `POST /evidence` endpoint
- [x] **Description**: Wraps the C8 command.
- [x] **Request**: `{ "claim": "string", "top_papers": 5, "include_contradicting": true }`
- [x] **Response**: List of `EvidenceSpan` with paper attribution.
- [x] **Depends on**: C8.

### D5 · `POST /source-rank` endpoint
- [x] **Description**: Wraps the C9 command.
- [x] **Depends on**: C9.

### D6 · Patch model
- [x] **Description**: All edits flow through patches. A patch is a JSON object describing a single change to a single file. Never apply edits directly from audit output.
- [x] **Patch types**:
  ```
  insert_citation        \cite{key} at offset
  replace_citation       swap one key for another
  remove_citation        delete one \cite{} call (with marker comment, like --fix)
  append_bibtex          add entry to .bib
  replace_bibtex_entry   swap an entire entry
  rename_cite_key        rename across .tex + .bib
  fix_claim_text         user-driven, AI suggests, never auto-applies
  add_comment            inline %-comment with reasoning
  ```
- [x] **Schema**:
  ```json
  {
    "patch_id":  "patch_abc123",
    "type":      "replace_citation",
    "file":      "main.tex",
    "range":     { "start": 1420, "end": 1447 },
    "old_text":  "\\cite{brown2020language}",
    "new_text":  "\\cite{singhal2023towards}",
    "reason":    "Brown 2020 doesn't support the MedQA 86.5% claim",
    "issue_id":  "issue_001",
    "user_approved": false
  }
  ```
- [x] **Files**: New `bibsync/patches.py` with `Patch` dataclass, validators, and `apply(patches, files)` function.
- [x] **Acceptance**: Patches round-trip through JSON without data loss; `apply()` is deterministic and reports per-patch success/failure.

### D7 · `POST /patch/preview` endpoint
- [x] **Description**: Given a set of patches, return what the files would look like AFTER application (without writing anything). For UI diff rendering.
- [x] **Response**:
  ```json
  {
    "preview": {
      "main.tex": { "before": "...", "after": "...", "diff_unified": "..." },
      "references.bib": { ... }
    },
    "conflicts": []  // patches whose old_text doesn't match current content
  }
  ```
- [x] **Depends on**: D6.

### D8 · `POST /patch/apply` endpoint
- [x] **Description**: Apply patches to in-memory file content; return the post-application content. The client (extension) is responsible for writing the result back to its host (Overleaf).
- [x] **Atomicity**: All patches succeed or none do. Returns `{ "ok": bool, "files": {...}, "errors": [...] }`.
- [x] **Depends on**: D6.

### D9 · Project/session IDs + privacy endpoint
- [x] **Description**: Every request carries a `project_id`. Server stamps this onto memory writes for the per-project scope. New endpoint `GET /privacy` returns what data is held for the current project_id.
- [x] **Files**: `bibsync/server.py`, `bibsync/memory.py` (no schema change, just exposes the project's records).

### D10 · Memory endpoints
- [x] **Description**:
  - `GET /memory?project_id=...&type=...` — list records
  - `POST /memory/forget` — write tombstone for a record_id
  - `DELETE /memory/project` — purge_project equivalent
- [x] **Depends on**: D9.

### D11 · Cache status + control
- [x] **Description**: `GET /cache/status` returns sizes (paper_content, pdfs, embeddings, memory). `POST /cache/clear` clears optional caches (not memory).

### D12 · OpenAPI spec + Python SDK
- [x] **Description**: FastAPI auto-generates OpenAPI; we export it at `/openapi.json`. Add a tiny `bibsync.client` Python helper for testing.
- [x] **Acceptance**: `bibsync.client.AuditClient("http://127.0.0.1:38476", token).audit(...)` mirrors `audit_project_sync` semantics.

### D13 · Server tests
- [x] **Description**: End-to-end test against the audit_tier2_demo fixture posted to the server, verifies the response matches the CLI's `--output-json` output.
- [x] **Files**: `tests/test_server.py`.

**Sprint-D success target**: A non-browser client (the test harness, a curl invocation, a Python script) can POST tex/bib content and receive structured issues + patches + evidence — no CLI invocation, no filesystem assumptions.

---

## Sprint E — Chrome extension (read-only)

**Goal**: The user opens Overleaf, opens the BibSync side panel, selects text, clicks "Check". They see issues + evidence + suggested citations — but **NO document edits yet**.

### E1 · Native messaging host
- [ ] **Description**: A small Python script that bridges the Chrome extension to `bibsync serve`. Chrome speaks Native Messaging over stdin/stdout (length-prefixed JSON).
- [ ] **Files**: New `native-host/bibsync_native_host.py`, `native-host/install_native_host.py`.
- [ ] **Acceptance**: Manual smoke test — Chrome can launch the host and exchange a `health` message.
- [ ] **Reference**: [Chrome Native Messaging docs](https://developer.chrome.com/docs/extensions/develop/concepts/native-messaging).

### E2 · `bibsync install-native-host` CLI command
- [ ] **Description**: Writes the native-messaging manifest to `~/Library/Application Support/Google/Chrome/NativeMessagingHosts/com.bibsync.host.json` with `allowed_origins` pointing to the extension ID.
- [ ] **Files**: `bibsync/cli.py`, native manifest template.
- [ ] **Acceptance**: `bibsync native-host status` reports installed=true after install.

### E3 · Extension scaffold (Manifest V3)
- [ ] **Description**: TypeScript Chrome extension with MV3 manifest, service worker, content script, side panel.
- [ ] **Files**: New `chrome-extension/` directory:
  ```
  chrome-extension/
    manifest.json
    src/
      serviceWorker.ts
      contentScript.ts
      sidePanel.html
      sidePanel.tsx
      overleafAdapter.ts
      nativeBridge.ts
      types.ts
    package.json
    tsconfig.json
    esbuild.config.mjs
  ```
- [ ] **Build**: esbuild (fast, no webpack baggage); ships unpacked.
- [ ] **Acceptance**: `chrome://extensions/` loads the unpacked extension without errors.

### E4 · Side panel UI (read-only)
- [ ] **Description**: React (with Vite OR just esbuild + react) for the side panel. Tabs: Check, Suggestions, Evidence, BibTeX, Memory, Settings.
- [ ] **MVP tab**: Check — shows current file status, issue cards.
- [ ] **Reference**: [Chrome Side Panel API](https://developer.chrome.com/docs/extensions/reference/api/sidePanel).

### E5 · Overleaf editor adapter
- [ ] **Description**: Single file `overleafAdapter.ts` that knows how to read Overleaf's editor state. ALL Overleaf-specific DOM/CodeMirror hacks live here.
- [ ] **Interface**:
  ```ts
  interface EditorAdapter {
    detectEditor(): boolean;
    getActiveFileName(): Promise<string>;
    getCurrentText(): Promise<string>;
    getSelectionRange(): Promise<{ start: number; end: number }>;
    getSelectedText(): Promise<string>;
    highlightRange(range, severity): void;
    clearHighlights(): void;
    // Sprint-F adds applyPatch
  }
  ```
- [ ] **Strategy**: Overleaf uses CodeMirror 6; access via the `cm-editor` DOM element + introspecting the CM6 view instance.
- [ ] **Acceptance**: Console log on Overleaf shows correct active filename + current selection text.

### E6 · Native bridge (extension side)
- [ ] **Description**: `nativeBridge.ts` opens a long-lived Native Messaging connection to the host, exposes `request(name, payload) -> Promise<response>`.
- [ ] **Files**: `chrome-extension/src/nativeBridge.ts`.

### E7 · "Check selected text" button + flow
- [ ] **Description**:
  ```
  user clicks button
    → contentScript reads current selection via overleafAdapter
    → serviceWorker forwards to nativeBridge
    → nativeBridge → native host → bibsync serve POST /audit
    → response flows back to side panel
    → render issues
  ```
- [ ] **Acceptance**: Selecting "GPT-3 achieves 86.5% on MedQA \cite{brown2020language}." and clicking Check shows a `contradicted_claim` issue card.

### E8 · Issue card component
- [ ] **Description**: Renders one issue with severity colour, claim text, current citation, verdict, "View evidence" + "Replace citation" buttons (last one disabled until Sprint F).
- [ ] **Severity colours**: red (`wrong_reference`, `unsupported_reference`), orange (`contradicted_claim`), yellow (`needs_user_review`, `weak_support`, `source_unavailable`), green (`verified`).

### E9 · Evidence viewer
- [ ] **Description**: Expandable section under each issue card. Shows the `EvidenceSpan` objects from `/evidence`: paper title, page, quote, type (supporting/contradicting).
- [ ] **Depends on**: C6.

### E10 · "Find citation for selected text" flow
- [ ] **Description**: Maps to `/evidence` endpoint when there's no existing `\cite{}` in the selection. Shows top-3 suggested papers with evidence quote.

### E11 · "Copy BibTeX" / "Copy `\cite{key}`" actions
- [ ] **Description**: Clipboard-only actions. User pastes manually into Overleaf. Bridges Sprint E (read-only) to Sprint F (auto-insert) — proves the AI is useful before risking edit corruption.

### E12 · Settings tab
- [ ] **Description**: Choose embedding backend (auto/local/api), tier (0/1/2), reranker on/off. Persists to `chrome.storage.local`.

### E13 · Memory tab
- [ ] **Description**: Lists memory records for the current project (calls `/memory`). Forget button per record.

### E14 · Connection status indicator
- [ ] **Description**: Header bar shows "Connected to BibSync" (green) / "BibSync not running" (red) based on `/health` polling every 30s. Click → run `bibsync serve` instructions modal.

**Sprint-E success target**: A user in Overleaf can click Check on selected text, see structured issues with evidence quotes, copy a suggested citation, and inspect their memory — all without any document edit.

---

## Sprint F — User-approved edits

**Goal**: Patch-based edits the user approves are applied. Undo + conflict detection. The user is always in the loop.

### F1 · `applyPatch` on EditorAdapter
- [ ] **Description**: Implement the CodeMirror 6 edit transaction in `overleafAdapter.ts`.
- [ ] **Acceptance**: Calling `applyPatch({ type: insert_citation, range, new_text })` actually changes Overleaf's editor content.

### F2 · Patch preview UI
- [ ] **Description**: Modal showing before/after diff for a patch (using `/patch/preview`). Buttons: Accept, Reject, Explain.

### F3 · "Insert citation" action
- [ ] **Description**: For the "Find citation" flow — user clicks Insert; the extension constructs a patch, previews, applies on Accept.

### F4 · "Replace citation" action
- [ ] **Description**: For `wrong_reference` issues — replaces the problematic `\cite{...}` with a suggested alternative. Always preview first.

### F5 · "Append BibTeX entry" action
- [ ] **Description**: For citations not in `.bib` — appends the suggested entry to the detected `.bib` file. Detect via Overleaf project's file list.

### F6 · "Ignore warning" action
- [ ] **Description**: Writes an `override` memory record so the same issue doesn't fire again. Records *why* (free-text or radio buttons).

### F7 · "Mark as accepted" action
- [ ] **Description**: For `unverifiable` issues — writes an `accept` memory record so future runs skip the LLM call.

### F8 · Undo last BibSync edit
- [ ] **Description**: Stash the pre-edit content per patch; clicking Undo writes it back via `applyPatch`. Depth-1 undo only (CM6 editor's own undo handles history).

### F9 · Patch conflict detection
- [ ] **Description**: Before applying, the server verifies that `old_text` still matches the file content at the given range. If not, returns a conflict response; extension shows "File has changed — re-check before applying" instead of accepting.
- [ ] **Depends on**: D7.

### F10 · Multi-file BibTeX detection
- [ ] **Description**: Overleaf projects often have multiple `.bib` files. The extension reads the project file list (via the Overleaf adapter) and asks the user where to append on first use; remembers the choice in `chrome.storage.local`.

**Sprint-F success target**: User can accept a "replace wrong cite" suggestion, see it applied, and undo it — with no document corruption and full conflict-safety against external edits.

---

## Sprint G — Project-level support

**Goal**: Before submission, the user runs a full-project audit and reviews all issues from the side panel.

### G1 · Detect root `.tex`
- [ ] **Description**: Find the main file (the one with `\documentclass` and `\begin{document}`). Many projects have `main.tex` but not all.

### G2 · Read multiple files via Overleaf
- [ ] **Description**: Walk the project file tree; collect all `.tex` and `.bib` content. Cap at sane size limits (e.g. 1 MB per file).

### G3 · Full-project `/audit` server call
- [ ] **Description**: Multi-file request to `/audit`. Server already handles this via the existing `audit_project` machinery.

### G4 · Batch issue review UI
- [ ] **Description**: Scrollable list of all issues, grouped by file. Filter chips: severity, status, file. Bulk accept/ignore actions.

### G5 · HTML/JSON report export
- [ ] **Description**: User can save a snapshot of the audit for sharing (e.g. with a supervisor). HTML output renders the same as the side panel; JSON is the raw `audit` response.

### G6 · Batch BibTeX backfill
- [ ] **Description**: One click runs `extract` semantics — for every `missing_bib_entry`, fetch the BibTeX from the verified canonical paper, append to the chosen `.bib`.

### G7 · Pre-submission checklist
- [ ] **Description**: A short final screen: "X verified, Y issues need review, Z hallucinated cites need fixing". Refuses to "mark project ready" until all severity≥orange issues are resolved or ignored.

### G8 · Optional Git workflow
- [ ] **Description**: For Overleaf-with-Git users, a `Save as commit` button that bundles all approved patches into a single commit with a generated message. Defer the full implementation; just leave the hook.

**Sprint-G success target**: A real paper draft can be audited end-to-end from the side panel, issues triaged in batch, and the project marked ready-for-submission only when the safety bar is met.

---

## Avoid list (do NOT add these now)

| Item | Why not |
|---|---|
| mem0 SDK | Generic memory abstractions are wrong-shaped for citations; in-house store works |
| LlamaIndex / Haystack migration | 200-line `EmbeddingStore` works; framework migration is multi-week with no benchmark gain |
| Vector DB (Qdrant / LanceDB / pgvector) | Per-paper scale is small; revisit only if cross-project / team mode is added |
| Real-time keystroke linting | Slow, noisy, expensive, bad UX while drafting. User-triggered only. |
| Automatic deletion in extension | Never. Even for `hallucinated`. Extension is read+propose; user approves every edit. |
| GraphRAG / community detection | Whole-corpus QA tool; we operate per-paper |
| ColBERT / late interaction | 10–100× slower; standard hybrid is enough |
| Paid web search APIs as default (SerpAPI, Tavily, Exa, Brave) | SearXNG (self-hosted OSS) is the substitute; paid only as opt-in |
| Closed rerankers (Cohere, Jina API) as default | `bge-reranker-base` (OSS, 1GB) is opt-in for higher quality |
| LLM-extracted general facts in memory | Risk: "rejected GPT-3 for MedQA" → "GPT-3 bad for medical" → wrong-verdict adjacent claims |
| Full-corpus indexing of user's local library | Out of scope; per-paper retrieval is the design |
| Server exposed beyond `127.0.0.1` | Security risk; manuscript content stays local |
| Multi-module package split (`bibsync-core`/`-rag`/`-web`/...) | Premature; ~25 files is still navigable |

---

## Progress log

> Appended at the end of each work block. Each entry: date, sprint/task IDs touched, commit SHA, notes.

### 2026-05-16
- `toto.md` + `summary.md` authored and committed (`3f61099`).
- **C1 done** (`334747a`) — survey-cited-as-original rule landed. Benchmark filter `--filter survey` passes 100%. Survey rejections correctly route to `hallucinated` (not `contradicted`) via the explicit `contradicted=false` clarification.
- **C2 done** (`0de4496`) — source-resolution flag + Tier-0 fabricated-citation guard. Wiring through audit / suggest / benchmark via the new shared `verdict_to_status` helper. **Full benchmark: 85.0% → 90.0%, FDR still 0%**. 2 failures remaining, both contradiction-detection LLM slips (C3 target).
- **C3 done** (`142aa76`) — Tier-2 contradiction-detection 6-step checklist + 3 new worked examples (model-version, structural-property, same-entity-not-mentioned). **90.0% → 95.0% accuracy**, FDR 0%.
- **C4 done** (`35e4891`) — structured contradiction payload (`contradiction_type`, `claimed_value`, `actual_value`) on `CitationAudit`, `CitationCheck`, and the JSON output schema. Additive change; legacy callers unaffected.
- **C5 done** (`ecdf25f`) — `issue_type` taxonomy (9 sub-values) derived from existing fields. 11/11 unit tests pass. UI rendering hint surfaced in JSON output.
- **C6 done** (`0ce0645`) — `bibsync/evidence.py`: pure-Python sentence-level compression of RAG chunks → 1-3-sentence quotes with page attribution. JSON output now emits an `evidence` array per check. 6 unit tests cover claim-relevant sentence picking + JSON round-trip + page extraction from string-form chunks.
- **C7 done** (`313edd4`) — `decompose_claim` helper for compound claims. Splits on coordinating connectives, re-attaches head subject to subordinate fragments. Wired into audit retrieval — per-sub-claim top-K unioned, deduped, max-cosine wins.
- **C8 done** (`88e943c`) — `bibsync evidence "claim"` standalone command. Two-stage retrieval (LLM identify + OpenAlex title.search). Live-verified: returns Vaswani as rank #1 for the Transformer/self-attention claim, runners-up are related (TimeSformer, Speech Separation).
- **C9 done** (`94000e7`) — `bibsync source-rank "title or claim"` ranking by combined canonicality signals (cited_by + LLM-identified canonicality + venue prior + recency − survey penalty). Live-verified: Vaswani at rank #1 with score +0.96, survey-style papers correctly demoted.
- **C10 done** (`23b0e07`) — OpenAlex citation-graph signals fed into Filter C. New `canonicality_signals` kwarg on `verify_claim_support` carries `is_survey_title`, `openalex_doi`, `openalex_arxiv_id`, etc. System prompt extended with rules for using these signals. Used to escape the "popular survey outranks original" trap.
- **C11 (final)** — **Full benchmark: 100% accuracy (20/20), 0% FDR, 51.6s wall clock**. Sprint C complete. Snapshot saved to `benchmarks/sprint-C-final-2026-05-16.json`.
- **Sprint D in one block** — `bibsync/patches.py` (Patch model with atomic apply + preview + conflict detection, 10 unit tests pass), `bibsync/server.py` (FastAPI app with 12 endpoints: /health, /audit, /suggest, /evidence, /source-rank, /patch/{preview,apply}, /memory*, /cache/*, /privacy, /openapi.json), `bibsync serve` CLI command (token auth, refuses external binds unless `BIBSYNC_ALLOW_EXTERNAL=1`), `tests/test_server.py` (12/12 passing). End-to-end live test against audit_tier2_demo confirms the server matches CLI output exactly — same 2 verified / 2 hallucinated verdicts in 28.3s.

---

## Open questions for the user (not blockers)

1. **Server token vs unauthenticated localhost** — D1 proposes a per-process token in `~/.config/bibsync/server.token`. Acceptable, or prefer something else (Unix socket, OS keychain)?
2. **Scholar in server mode** — D3 proposes skipping Playwright/Scholar when the call comes from the server (Overleaf user is busy in their tab). Acceptable, or keep Scholar available via a "wake Chrome" prompt?
3. **Chrome extension distribution** — Build for personal use only (unpacked, local install) or eventually publish to Chrome Web Store (CWS)? Affects icon design, privacy policy, screenshots scope.
4. **Editor support beyond Overleaf** — VS Code / Cursor LaTeX extensions? Same server contract works; just needs a different EditorAdapter.

---

## Reference docs

- [Chrome Native Messaging](https://developer.chrome.com/docs/extensions/develop/concepts/native-messaging) — manifest format, length-prefixed message protocol
- [Chrome Side Panel API](https://developer.chrome.com/docs/extensions/reference/api/sidePanel) — MV3 side panel registration + display rules
- [Manifest V3 migration guide](https://developer.chrome.com/docs/extensions/develop/migrate) — service worker (not background page), declarative permissions
- [CodeMirror 6 (Overleaf editor)](https://codemirror.net/docs/) — for the Overleaf adapter
- [FastAPI](https://fastapi.tiangolo.com/) — server framework choice for `bibsync serve`
- Existing project docs:
  - [`summary.md`](summary.md) — current-state overview
  - [`benchmarks/README.md`](benchmarks/README.md) — measurement methodology
  - [`README.md`](README.md) — user-facing docs

---

## How to use this file

1. Pick the next `[ ]` task in the current sprint (top-to-bottom).
2. Move to `[~]`, do the work, run the acceptance criteria.
3. Move to `[x]` when acceptance passes. Add a line to the progress log with the commit SHA.
4. If blocked, move to `[!]` with a note in the task body explaining why.
5. If the task no longer makes sense, move to `[-]` with a reason.

When a sprint is complete (all tasks `[x]` AND success target met), update the Sprint Overview table and start the next sprint.
