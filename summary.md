# BibSync — Project Summary

**A local-first AI assistant for managing citations in LaTeX papers.** It detects hallucinated citations, adds missing ones, verifies factual claims against the actual cited papers' content, and remembers decisions across runs — all from a single CLI, all working without paid APIs.

---

## 1. Problem it solves

Researchers writing papers (often with LLM assistance) routinely end up with citations that:

- **Don't exist** — the LLM fabricated plausible-looking author/year/title combinations.
- **Are misattributed** — the cited paper is real but reports a different number, evaluates a different benchmark, or covers a different topic than the prose claims.
- **Are surveys cited as originals** — a review paper about method X cited as the introduction of method X.
- **Have wrong values** — e.g. "GPT-3 achieves 86.5% on MedQA" — the GPT-3 paper exists, but the 86.5% number is from Med-PaLM 2, not GPT-3.

BibSync verifies every `\cite{}` against the actual paper's full text (via PDF + RAG retrieval) and an LLM judge, before the paper goes out. It also goes the other way — given prose that lacks citations, it proposes and inserts the right ones, grounded in the candidate paper's actual content.

---

## 2. What it does (user-facing commands)

Eleven Click subcommands, grouped by purpose:

### Citation lifecycle
| Command | Purpose |
|---|---|
| `bibsync suggest <tex>` | Read prose; for each paragraph without a `\cite{}`, identify what needs citing, find the canonical paper, verify against its content, insert `\cite{}` + append BibTeX. |
| `bibsync audit <project>` | For each existing `\cite{key}`, fetch the paper, run RAG retrieval against the surrounding prose claim, get an LLM verdict. With `--fix`, automatically remove confidently-hallucinated cites. |
| `bibsync extract <tex>` | For every `\cite{key}` that has no `.bib` entry, infer the paper from the cite key + claim, search, fetch BibTeX, fill the gap. |
| `bibsync add <title>` | Manual: search Scholar for a title, write its BibTeX into `references.bib`. |
| `bibsync search <title>` | Search-only; doesn't write anything. Useful for inspection. |

### Bibliography hygiene
| Command | Purpose |
|---|---|
| `bibsync verify` | Re-check each `.bib` entry against Scholar; flag entries whose canonical Scholar match disagrees. |
| `bibsync fix` | Re-fetch every `.bib` entry from Scholar, replace stale/wrong metadata, propagate cite-key renames into the `.tex` files. |
| `bibsync repair` | Convert legacy `\bibitem{}`-style references into structured BibTeX. |
| `bibsync scan` | Find every `\cite{}` in a project, reconcile against `.bib`, report missing keys. |

### Project memory
| Command | Purpose |
|---|---|
| `bibsync memory show --project .` | Inspect what BibSync remembers (verdicts, user decisions, preferences). |
| `bibsync memory forget <id>` / `purge-project` | Manage / clear memory. |

### Measurement
| Command | Purpose |
|---|---|
| `bibsync bench run` | Run a labeled citation-verification benchmark; report accuracy + false-deletion rate. |
| `bibsync bench show` | Inspect benchmark cases without running them. |

### Configuration
| Command | Purpose |
|---|---|
| `bibsync config {show,set,unset,path,reset-profile}` | Manage API keys (OpenRouter / OpenAI), default LLM model. |

---

## 3. Architecture overview

```
┌────────────────────────────────────────────────────────────────────────┐
│  USER · writes a LaTeX paper                                           │
│        paper.tex (prose, with or without \cite{})  +  references.bib   │
└────────────────────────────────┬───────────────────────────────────────┘
                                 │
       ┌─────────────────────────┼──────────────────────────┐
       ▼                         ▼                          ▼
 ┌───────────┐            ┌──────────────┐            ┌──────────────┐
 │  suggest  │            │    audit     │            │  extract /   │
 │  ADD      │            │  VERIFY +    │            │   fix /      │
 │  missing  │            │  optional    │            │   repair     │
 │  \cite{}  │            │  --fix       │            │  CLEAN UP    │
 └─────┬─────┘            └──────┬───────┘            └──────────────┘
       │                         │
       └────────────┬────────────┘
                    ▼
╔════════════════════════════════════════════════════════════════════════╗
║                  VERIFICATION CORE  —  shared pipeline                  ║
║                                                                        ║
║  ① CLAIM EXTRACTION                                                    ║
║  ② MEMORY RECALL  ◄─ short-circuits the rest if hit                   ║
║  ③ SOURCE RESOLUTION  (5-source fallback chain + PDF resolver)         ║
║  ④ FILTER CHAIN  (suggest only — version, dedup, canonical, grounding) ║
║  ⑤ RAG RETRIEVAL  (claim-type routing → hybrid + reranker + tables)    ║
║  ⑥ LLM VERDICT  (tier-aware prompt, contradiction detection)           ║
║  ⑦ STATUS MAPPING + SAFETY NETS                                        ║
║  ⑧ MEMORY WRITEBACK                                                    ║
║                                                                        ║
╚═══════════════════════════════╤════════════════════════════════════════╝
                                ▼
   ┌────────────────────────────────────────────────────────────────┐
   │  OUTPUTS: paper.tex (rewritten) + references.bib (appended) +  │
   │            terminal report + --output-json + persistent memory  │
   └────────────────────────────────────────────────────────────────┘
```

The two heaviest commands (`suggest` and `audit`) **share one verification core**. Different inputs — prose claims vs `\cite{}` calls — but identical 8-stage reasoning afterward. This unification was a deliberate architecture choice: every retrieval / prompt / judgment improvement benefits both paths automatically.

---

## 4. The Verification Core (the AI engine)

Each step is short-circuiting: if step ② hits, steps ③-⑦ are skipped. The pipeline is designed so the cheapest checks happen first.

### Step 1 — Claim extraction
- **`suggest`**: an LLM (`suggest_citations`) reads each paragraph and identifies "anchors" — phrases that need a citation — plus a search query and a one-line rationale.
- **`audit`**: walks the `.tex`, finds each `\cite{}`, extracts the surrounding sentence (with LaTeX comments stripped to avoid mis-routing the claim into author notes).

### Step 2 — Memory recall
JSONL store of past decisions, keyed by `(claim_hash, paper_key)`:
- **Exact match** on normalised claim hash (sub-millisecond lookup).
- **Fuzzy match** via rapidfuzz `WRatio ≥ 90` for paraphrases (e.g. "GPT-3 achieves 86.5% on MedQA" recalls "GPT-3 achieves 86.5% **accuracy** on MedQA").
- **Conservative threshold** — rejects "GPT-3 86.5% MedQA" vs "GPT-4 89% MedQA" (89% score, below threshold) to prevent wrong-verdict carryover.
- Recall **short-circuits the entire LLM pipeline** for already-judged pairs.

Verified live: a 4-citation audit goes from **4 LLM calls / ~80 seconds (cold)** to **0 LLM calls / ~3 seconds (warm)** on repeat runs.

### Step 3 — Source resolution
Five paper-content sources in fallback order:

```
cache → arXiv → Semantic Scholar → OpenAlex → Crossref → Unpaywall (PDF only)
```

Each source is gated by a **title-match guard** (rapidfuzz `token_sort_ratio ≥ 75`) to reject wrong-paper hits before they enter the pipeline. The suggest path additionally drives **Google Scholar via Playwright** (headed Chrome with persistent profile for CAPTCHA avoidance) to find papers when the user hasn't specified one.

OpenAlex closes the gap for Nature / IEEE / non-arXiv papers (300M+ works). Unpaywall resolves DOI → open-access PDF URL for the green-OA-archived 60% of paywalled papers, enabling Tier-2 RAG on otherwise-stuck citations.

### Step 4 — Filter chain (suggest only)
Four filters evaluated in order; first to reject ends the candidate:

| Filter | Logic |
|---|---|
| A — Version mismatch | Catches "GPT-3" anchor matched to a "GPT-4" paper, "Med-PaLM 2" anchor matched to "Med-PaLM 1", etc. Deterministic regex. |
| B — Cluster-id dedup | If the same paper is already used by an earlier citation in this run, reject. |
| C — Canonical detector (LLM) | "Is this the right canonical paper to cite for this claim?" Tuned with explicit accept rules (introduces the system, cited ≥ 200, etc.) and reject rules (survey, evaluation, replication, etc.). |
| D — Grounding gate (LLM, Tier 1 or 2) | "Does this paper actually SUPPORT the specific claim — not just match its topic?" Runs the full audit-tier prompt against the paper's abstract / retrieved chunks. |

The D gate is critical: it catches the failure mode where Filter C says "yes, this is the GPT-3 paper" but the specific claim ("GPT-3 achieves 86.5% on MedQA") isn't supported by the actual paper.

### Step 5 — RAG retrieval (Tier 2 only)

A **claim-type classifier** routes each claim to one of four retrieval strategies:

| Claim type | top-k | Reranker | Contradiction-query |
|---|---:|---|---|
| Quantitative ("X achieves Y% on Z") | 8 | ✓ | ✓ |
| Named-method ("The Transformer is...") | 5 | ✓ | ✗ |
| Attribution ("X introduced Y") | 3 | ✓ | ✗ |
| Generic (topic-level) | 3 | ✗ | ✗ |

Classification is pure regex (sub-microsecond per claim). The routing makes easy claims fast and hard claims thorough.

Within the chosen strategy:

```
PDF
 ├─ pypdf → page-aware text → 800-word overlapping prose chunks
 └─ PyMuPDF → page.find_tables() → quality-filtered table chunks
                                    (caption + headers + rows preserved
                                     as markdown-ish text)
                  │
                  ▼
       Embeddings  (fastembed bge-small local  OR  OpenRouter bge-m3 API)
                  │
                  ▼
       HYBRID RETRIEVAL
        ├─ Dense cosine over the embeddings    →  top-20
        ├─ BM25 (Okapi, rank_bm25) over tokens →  top-20
        └─ Reciprocal Rank Fusion (k=60)       →  fused top-20
                  │
                  ▼
       CROSS-ENCODER RERANK
       (Xenova/ms-marco-MiniLM-L-6-v2, 80MB fastembed ONNX)
                  │
                  ▼
       TOP-K chunks
                  │
                  ▼
       CONTRADICTION RETRIEVAL  (quantitative only)
       Value-stripped re-query → top-3 extra chunks → merged
                  │
                  ▼
       Final evidence pack → LLM judge
```

This is the standard recall-then-precision design pattern: dense captures semantic paraphrases (low recall ceiling on exact tokens), BM25 captures verbatim tokens (low recall ceiling on paraphrases), RRF fuses them, the cross-encoder reranks for precision.

**Tables get extracted as standalone chunks** because quantitative claims usually reference values that live in result tables, not prose. A table chunk preserves "(model=Med-PaLM 2, MedQA=86.5%)" as a single retrievable unit instead of scattering those tokens across multiple prose chunks.

### Step 6 — LLM verdict
The audit judge (`llm.audit_citation`) takes the claim + paper metadata + evidence pack and emits:

```json
{
  "supports": true | false,
  "confidence": 0.0 - 1.0,
  "reasoning": "one-sentence explanation",
  "contradicted": true | false
}
```

The system prompt is **tier-aware** — three suffixes (`_AUDIT_TIER0/1/2_SUFFIX`) extend the base prompt with rules specific to the evidence depth available:

- **Tier 0** — refuse to high-confidence verify quantitative claims (numbers, named benchmarks) from metadata alone. Worked example in the prompt: "GPT-3 achieves 86.5% on MedQA" cited to Brown 2020 — Tier 0 must NOT verify, because the claim's specific number can't be checked from a title.
- **Tier 1** — abstract is primary evidence. Quote a phrase from it in the reasoning.
- **Tier 2** — HARD RULES requiring a verbatim quote from a retrieved chunk for quantitative / named-entity claims. Worked examples cover the BERT-110M case (must quote `"BERTBASE: L=12 ... Total Parameters=110M"`) and the GPT-3/MedQA case (must explicitly state "no chunk mentions 'MedQA' or any number near 86.5%").

The **contradiction-detection rule** is the strongest negative signal: when chunks contain the same entity reporting a DIFFERENT value than the claim, the LLM sets `contradicted=true`. Distinct from `hallucinated` because it has different actionability — fix the prose number, don't delete the citation.

### Step 7 — Status mapping + safety nets
The `(supports, confidence, contradicted)` triple maps to one of five statuses:

| Status | Meaning | User action |
|---|---|---|
| `verified` | Paper supports the claim. | Keep cite. |
| `hallucinated` | Wrong paper / topic mismatch. | Delete cite (or auto-removed with `--fix`). |
| `contradicted` | Right paper, wrong value in prose. | **Fix the prose**, not the cite. |
| `unverifiable` | LLM lacks confidence; evidence missing. | Re-run at higher tier, or accept manually. |
| `missing_in_bib` | The cite key has no `.bib` entry. | Fix `.bib`. |

Two safety nets layered on top:
- **Tier-0 quantitative-claim safety net** — if the LLM returns `supports=true` at Tier 0 on a claim containing a number / named benchmark, downgrade to `unverifiable` (don't accept on title-alone for specific factual claims).
- **Confidence floor** (default 0.7) — weak `supports=false` verdicts become `unverifiable`, not `hallucinated`. Prevents auto-deleting good citations on shaky LLM rejections.

The **headline safety metric** is **false-deletion rate** — wrongly flagging a real citation as hallucinated. It's been **0% throughout development** across 20 benchmark cases.

### Step 8 — Memory writeback
Every verdict is persisted to JSONL:
```jsonc
{
  "id":         "mem_abc123",       // for forget-references
  "type":       "verdict",
  "scope":      "project",
  "claim_text": "GPT-3 achieves 86.5% on MedQA",
  "claim_hash": "b7897b5eab92",     // normalised for fast lookup
  "paper_key":  "arxiv-2005.14165", // arxiv → doi → title-hash
  "decision":   "contradicted",
  "tier":       2,
  "confidence": 0.95,
  "source":     "audit",
  "rationale":  "no chunk mentions 'MedQA' or '86.5%' — likely misattribution",
  "ts":         "2026-05-16T09:28:30Z"
}
```

The next run's step ② picks this up; the LLM call is skipped entirely.

---

## 5. Memory layer

Architecturally inspired by [mem0](https://github.com/mem0ai/mem0)'s **scoped + queryable + persistent** design — but built without the mem0 SDK and tightly scoped to citation work, so the failure mode "LLM generalises a memory into a wrong fact" cannot happen.

### Record types
| Type | When written | When read |
|---|---|---|
| `verdict` | Audit produced a verdict for (claim, paper) | Audit short-circuits the LLM call if memory holds the same pair at same-or-higher tier |
| `accept` | User approved a suggest proposal | Suggest fast-accepts the same (claim, paper) pair |
| `reject` | User rejected, OR Filter D rejected | Suggest skips the candidate next run |
| `preference` | Inferred from user accept patterns | Tie-breaker in Filter C ranking |
| `override` | User kept a cite an earlier run flagged hallucinated | Suppress repeat hallucinated verdicts |

### Storage layout
```
~/Library/Caches/bibsync/memory/
  user.jsonl                                  # user-scoped preferences
  projects/
    <sha1(realpath(project_root))>.jsonl      # per-project decisions
```

- **Append-only JSONL** — atomic writes, no torn files, no race conditions.
- **Tombstone deletes** — `forget` writes a tombstone record; reads filter it out. Original line stays in the file (auditable).
- **SHA1 project namespace** — files are anonymous unless you correlate them with the directory yourself.

### Matching strategy
- **Exact hash match** on normalised claim (lowercase + LaTeX `\cite{...}` stripped + hyphen-split + whitespace-collapsed).
- **Fuzzy match** via rapidfuzz `WRatio ≥ 90` — `WRatio` combines Levenshtein + partial_ratio + token_sort, which separates same-claim-paraphrase from different-claim-same-topic correctly (token_set_ratio would over-recall on "GPT-3 86.5% MedQA" vs "GPT-4 89% MedQA").
- **Paper identity** via stable_key: `arxiv-<id>` → `doi-<id>` → `title-<hash>`.

### Why not mem0 SDK
- mem0's flagship feature — "LLM extracts atomic facts and reconciles conflicts" — is **the wrong abstraction for citations**. We store *exact* (claim, paper, verdict) tuples. Generalising "rejected GPT-3 for MedQA claim" into "GPT-3 is bad for medical claims" would silently wrong-verdict adjacent claims.
- mem0 Cloud is a hosted service; this project is local-first by design.
- Vector / graph memory storage (Qdrant, Neo4j) is overkill for ~10K records per project max.

### CLI surface
```bash
bibsync memory show --project .         # rich table view
bibsync memory list                     # compact one-line-per-record
bibsync memory forget <record_id>       # write tombstone
bibsync memory purge-project --project .  # wipe project memory (user preserved)
bibsync memory path                     # show on-disk file paths
```

`--no-memory` is available on both `audit` and `suggest` to disable recall + writeback for one run.

---

## 6. RAG stack — techniques and models

### Models
| Role | Default | Alternative | Rationale |
|---|---|---|---|
| Embedding (local) | `BAAI/bge-small-en-v1.5` via fastembed (~80 MB ONNX) | — | Free, fully offline |
| Embedding (API, OpenRouter) | `baai/bge-m3` | `text-embedding-3-small` for OpenAI keys | Routable via OpenRouter, open-source, $0.01/M tokens |
| Reranker | `Xenova/ms-marco-MiniLM-L-6-v2` (~80 MB) | `BAAI/bge-reranker-base` (1 GB) opt-in | Standard cross-encoder, fastembed-native |
| Judge LLM | `openai/gpt-4o-mini` via OpenRouter | any OpenAI-compatible | User-selectable; provider auto-detected by key prefix |
| Table extraction | PyMuPDF (`pymupdf>=1.23`) | — | Pure-ish wheel, no Java |
| PDF text | `pypdf>=4.0` | — | Page-aware, no Java |

### Retrieval techniques (in order of impact)
1. **Hybrid BM25 + dense + Reciprocal Rank Fusion** — recall-then-precision standard.
2. **Cross-encoder rerank** as top-20 → top-5 polish pass — full-attention scoring of (query, chunk) pairs.
3. **Table-aware chunking** — quantitative claims hinge on values in result tables, not prose.
4. **Claim-type-aware routing** — generic claims get top-3 + no rerank; quantitative claims get top-8 + rerank + contradiction query.
5. **Contradiction retrieval** — for quantitative claims, value-stripped re-query surfaces conflicting numbers for the same entity.
6. **Tier-aware prompts with verbatim-grounding rules** — Tier 2 requires the LLM to quote a chunk verbatim for any quantitative claim.

### Techniques deliberately skipped
| Technique | Reason |
|---|---|
| LlamaIndex / Haystack framework | 200-line in-house `EmbeddingStore` works; adoption is a 2-week refactor for no measurable gain |
| Vector DB (Qdrant, LanceDB) | Per-paper scale (≤500 chunks) doesn't justify the dependency |
| ColBERT / late interaction | 10-100× slower; standard hybrid is good enough for short academic claims |
| GraphRAG / community detection | Designed for whole-corpus QA; we operate per-paper |
| HyDE / query decomposition | Possible Sprint-C items; not core to current failure modes |
| Closed rerankers (Cohere, Jina API) | Paid; violates open-source-first stance |
| LLM-extracted memory facts (mem0 core) | Wrong abstraction for citations — see Memory section above |

---

## 7. Source-of-truth integrations

The five sources, in fallback order. Each is HTTP-only (no SDKs), follows redirects, uses informative error logging.

| Source | What it provides | Auth |
|---|---|---|
| **arXiv** | ML/CS preprints, abstracts, PDF URLs | None |
| **Semantic Scholar** | 200M+ papers, abstracts, openAccessPdf URLs | Optional API key for higher rate limits |
| **OpenAlex** | 300M+ works (Nature, IEEE, ACL Anthology, biomed), citation graph, OA PDF URLs | None (polite-pool email) |
| **Crossref** | DOI-keyed metadata (last-ditch for traditional journals) | None |
| **Unpaywall** | DOI → green-OA PDF URL | None (mailto required) |
| **Google Scholar** (suggest only) | Citation count, cluster IDs, BibTeX export | Persistent Chrome via Playwright |

Each source's adapter runs through a **rapidfuzz title-match guard** (`token_sort_ratio ≥ 75`) before returning. Wrong-paper hits — e.g. Crossref returning "Spectrum-BERT" for a "BERT" query, or OpenAlex returning a Japanese commentary paper for "BERT" — get rejected before contaminating the judge.

---

## 8. Safety mechanisms

| Layer | Failure caught |
|---|---|
| **Title-match guard** (source adapters) | Wrong-paper hits (Crossref returning unrelated papers with topic overlap) |
| **Quantitative-claim heuristic** (audit, Tier 0) | LLM rubber-stamping a number on title alone |
| **Tier-aware system prompts** (llm.py) | Tier 0: refuse to verify quantitative claims; Tier 2: require verbatim quote |
| **Filter D grounding gate** (suggest) | Canonical paper exists but doesn't support YOUR specific claim |
| **Confidence floor** (audit) | Weak `supports=false` → `unverifiable` not `hallucinated` |
| **Per-tier degradation tracking** | Loud warning when ≥25% of citations couldn't reach the requested tier |
| **Conservative on LLM/network failure** | All errors → `supports=True, conf=0` → routes to `unverifiable`. Never auto-delete a citation because a network call failed. |
| **`--fix` targets only `hallucinated`** | `contradicted` and `unverifiable` are NEVER auto-removed — user must act manually |
| **Atomic writes** (bib + tex) | Tempfile-then-rename; no torn files on `Ctrl+C` |
| **Incremental `.bib` persistence** | Each new entry written immediately; mid-run abort never leaves dangling `\cite{}` |

---

## 9. Evaluation — benchmark harness

`bibsync bench run` runs a labeled set of (claim, paper, expected verdict) cases through the production audit pipeline. The benchmark file is JSONL, append-only, git-diff-friendly.

### Coverage (20 cases, 12 categories)
- Topical method attribution (positive controls)
- Quantitative architecture spec (BERT 12 layers / 110M params)
- Field cross-contamination (NLP paper cited for CV claim)
- Version conflict (GPT-3 claim cited to GPT-4 paper)
- Survey-vs-original (probing paper cited as the introduction)
- Quantitative misattribution (GPT-3 → MedQA 86.5% — the headline test case)
- Contradicted-quantitative (BERT-base 24 layers → paper actually says 12)
- Fabricated citation (paper doesn't exist)
- Missing-in-bib
- Generic background

### Headline metrics
| Metric | Baseline | Final |
|---|---:|---:|
| **Accuracy** | 77.8% (14/18) | **85.0%** (17/20) |
| **False-deletion rate** | 0.0% | **0.0%** (preserved across all changes) |
| **Wall clock (cold cache)** | 133.5s | 68.8s |
| **Wall clock (warm cache + memory)** | n/a | ~3s (full short-circuit) |

False-deletion rate is the **safety-critical metric**: it measures how often a real citation got flagged as hallucinated. It's been 0% throughout development.

### Remaining failures (3/20, all LLM-judgment, not retrieval)
| Case | Why it fails |
|---|---|
| `hallucinated-survey-cited-as-original` | LLM at Tier 2 sees attention-mechanism chunks (it's a survey ABOUT attention) and verifies. Needs prompt: "surveys are not introductions." |
| `hallucinated-fabricated-author-year` | All sources correctly miss → Tier 0 LLM accepts on title-plausibility. Needs prompt: "source-fetch-empty → unverifiable." |
| `contradicted-resnet-50-100m-params` | LLM said `supports=false` (correct rejection) but didn't set `contradicted=true`. Needs prompt strengthening on contradiction detection. |

All three are Sprint-C prompt-engineering items, not retrieval / source / memory bugs.

### Reproducibility
Two on-disk snapshots committed alongside the cases:
- `benchmarks/baseline-2026-05-15.json` (start)
- `benchmarks/sprint-B-final-2026-05-16.json` (current)

```bash
bibsync bench run --output-json /tmp/r.json    # full run + JSON output
bibsync bench show                              # inspect cases without LLM cost
bibsync bench run --filter contradicted         # category subset
```

---

## 10. Storage layout

All on-disk artifacts live in the XDG cache directory (`~/Library/Caches/bibsync/` on macOS).

```
~/Library/Caches/bibsync/
  paper_content/          # JSON, per paper, 30-day TTL
    arxiv-2005.14165.json
    doi-10.1038_...json
    ...
  pdfs/
    arxiv-2005.14165.pdf          # downloaded PDFs
    arxiv-2005.14165.txt          # extracted text (page-tagged)
  embeddings/
    arxiv-2005.14165.json         # {model, backend, chunks, vectors}
  memory/
    user.jsonl                    # user-scoped preferences
    projects/
      <sha1(project_root)>.jsonl  # per-project decisions
  fastembed_cache/                # ONNX models for local embedding/rerank
```

Embedding caches are keyed by both `(model, backend)` so switching between `bge-small (local)` and `bge-m3 (OpenRouter)` never silently mixes vector spaces. Config (API keys) lives separately at `~/.config/bibsync/` via `platformdirs`.

---

## 11. Technology stack

### Core dependencies (always installed)
| Package | Role |
|---|---|
| `click` | CLI |
| `rich` | Terminal rendering (tables, prompts) |
| `bibtexparser` (1.x) | BibTeX I/O |
| `rapidfuzz` | Title + claim fuzzy matching |
| `playwright` | Google Scholar (suggest only) |
| `platformdirs` | XDG paths |
| `httpx` | HTTP source clients |

### Optional `[audit-rag]` extras (~200 MB total)
| Package | Role |
|---|---|
| `pypdf` | PDF text extraction |
| `fastembed` | Local embeddings + cross-encoder (ONNX) |
| `rank_bm25` | BM25 (pure Python) |
| `pymupdf` | Table extraction |

### Optional `[openai]` extras
| Package | Role |
|---|---|
| `openai` | OpenAI-compatible API client (OpenRouter and OpenAI native) |

### Deliberate non-dependencies
- **No PyTorch** — fastembed uses ONNX Runtime instead
- **No mem0** — citation-specific memory built in-house
- **No LlamaIndex / Haystack** — 200-line `EmbeddingStore` does the job
- **No vector DB** — JSON files at our scale
- **No Java** (Camelot, Tabula, GROBID) — PyMuPDF for tables
- **No paid APIs by default** — OpenRouter (open-source-friendly) for LLM, fastembed (local) for embeddings

---

## 12. Out of scope (deliberately deferred)

These were considered and deferred during planning. Each has a documented reason.

| Item | Why deferred |
|---|---|
| Chrome / Overleaf extension | Multi-week project; CLI is faster than network round-trips for the researcher workflow. VS Code / Cursor extension would fit better if editor integration becomes a priority. |
| Local `bibsync serve` HTTP server | No downstream consumer yet; the JSON output flag is the cheaper integration surface. |
| LlamaIndex / Haystack adoption | 2-week refactor with no measurable benchmark gain; revisit only if abstraction need emerges. |
| mem0 SDK | Wrong abstraction (LLM-extracted facts) for citation memory; in-house module is more robust. |
| `mem0ai/memory-benchmarks` integration | Generic memory eval; would not measure citation-specific behaviour. |
| Ragas evaluation harness | Requires a labeled benchmark — our custom JSONL set is the equivalent at the right scope. |
| Web search beyond papers (Brave, SerpAPI, Tavily) | All paid APIs; SearXNG (self-hosted OSS) is the substitute if needed. |
| Multi-module package split (`bibsync-core`, `-rag`, `-web`, ...) | 12-module flat layout is still navigable; premature splits cause internal API churn. |
| Cohere / Jina paid rerankers | Open-source `bge-reranker-base` available as opt-in covers the quality use case. |

---

## 13. Roadmap

### Sprint-C candidates (in priority order)
1. **Tier-2 prompt strengthening** (~4h total) — closes the 3 remaining benchmark failures via prompt rules: surveys aren't originals, source-fetch-empty → unverifiable, stronger contradiction detection.
2. **OpenAlex citation-graph signals in Filter C** (~2 days) — `referenced_works` + `cited_by` data already fetched but unused for canonical detection.
3. **`bibsync source-rank "title"` standalone command** (~1 day) — exposes Filter C + citation-graph as a user query.
4. **`bibsync evidence "claim"` standalone command** (~1 day) — search → fetch → RAG → return supporting/contradicting chunks for any claim, not just a cited one.
5. **GROBID integration for hard PDFs** (~2 days) — closes the GPT-3-paper-style table extraction gap that PyMuPDF can't handle.

### Architecturally non-trivial (1+ week, would change project posture)
- VS Code / Cursor extension wrapping the CLI
- HyDE retrieval (hypothetical document embeddings) for hard paraphrased claims
- SearXNG web-search adapter for non-academic-paper sources

---

## 14. Repository

**Branch**: `main`
**Lines of code**: ~6,500 across 20 Python files in `bibsync/` + 8 source adapters
**Test coverage**: 11 unit tests for memory (round-trip, fuzzy, scope isolation, tombstones); 20 end-to-end benchmark cases for audit pipeline; offline integration tests for suggest grounding gate
**Commits on this product line**: ~25 commits across two named sprints (A: retrieval / sources / contradiction; B: memory / table-aware / reranker / claim-type routing)

The benchmark snapshots (`benchmarks/baseline-2026-05-15.json`, `benchmarks/sprint-B-final-2026-05-16.json`) document the measured progress; the per-commit messages document the design decisions and tradeoffs.

---

## 15. One-line summary

BibSync is a **local-first, open-source-first citation AI assistant for LaTeX papers**: it verifies citations against the actual papers' full-text content via a hybrid RAG pipeline (BM25 + dense + cross-encoder rerank + table-aware chunks), catches hallucinated and misattributed cites via tier-aware LLM judging, persists decisions across runs via a citation-specific memory layer adapted from mem0's architecture (no SDK), and exposes the whole thing through 13 Click subcommands measured against a labeled benchmark with a 0% false-deletion-rate safety invariant.
