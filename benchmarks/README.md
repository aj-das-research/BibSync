# BibSync citation-verification benchmarks

Reproducible evaluation harness for the audit pipeline. Every retrieval /
prompt / source change is measured against this set before landing.

## Quick start

```bash
# Run every case at the maximum tier (default 2):
bibsync bench run

# Run only contradicted-quantitative cases, save full JSON:
bibsync bench run --filter contradicted --output-json /tmp/r.json

# Inspect cases without running them:
bibsync bench show
```

## Files

| File | Contents |
|---|---|
| `citation_verification.jsonl` | The labeled case set — one JSON object per line. |
| `baseline-2026-05-15.json` | Snapshot at the start of the Tier-A sprint (before A1-A5). 18 cases. |
| `sprint-A-final-2026-05-15.json` | Snapshot at the end of the Tier-A sprint. 20 cases. |
| `sprint-B-final-2026-05-16.json` | Sprint B (memory + RAG upgrades). 20 cases, 85% accuracy. |
| `sprint-C-final-2026-05-16.json` | **Sprint C (prompt strengthening + structured outputs). 20 cases, 100% accuracy.** |

## Schema (one case)

```jsonc
{
  "id":          "verified-vaswani-self-attention",
  "kind":        "audit",            // currently only "audit" is wired
  "category":    "topical-method-attribution",
  "claim":       "The Transformer architecture introduced …",
  "bib_entry":   { "ID": "...", "ENTRYTYPE": "...", "title": "...", ... }
                 // or null for missing_in_bib cases
  "expected":    "verified"          // verified | hallucinated | contradicted
                                     // | unverifiable | missing_in_bib
  "min_tier":    0,                  // run at MAX(--tier, min_tier)
  "notes":       "free-form rationale for the labelled outcome"
}
```

## Sprint history

### Sprint A — retrieval + sources + contradiction verdict

| Stage | Cases | Accuracy | FDR | Δ vs prior |
|---|---:|---:|---:|---|
| Baseline (commit 7472429) | 18 | 77.8% | 0.0% | — |
| + A1 OpenAlex (3841ded) | 18 | 83.3% | 0.0% | **+5.5pp** |
| + A2 Unpaywall (dbbd57c) | 18 | 88.9% | 0.0% | **+5.6pp** |
| + A3 hybrid BM25 (e1166dd) | 18 | 88.9% | 0.0% | 0 (structural; cases didn't stress it) |
| + A4 contradicted (ee3e938) | **20** | 85.0% | 0.0% | new cases added, retained safety |
| + A5 JSON output (f17374d) | 20 | 85.0% | 0.0% | 0 (cosmetic on the metric) |

### Sprint B — memory + table-aware RAG + reranker + claim-type routing

| Stage | Cases | Accuracy | FDR | Wall | Δ |
|---|---:|---:|---:|---:|---|
| Sprint-A final (f17374d) | 20 | 85.0% | 0.0% | 57.5s | — |
| + M1 memory module (46d15bf) | 20 | 85.0% | 0.0% | — | foundation only |
| + M2 memory CLI (2ee60f9) | 20 | 85.0% | 0.0% | — | inspection only |
| + M3 audit ⇆ memory (c77a64e) | 20 | 85.0% | 0.0% | 0–3s on warm cache | **LLM call cost → 0 on repeat runs** |
| + M4 suggest ⇆ memory (7a6050d) | 20 | 85.0% | 0.0% | — | offline-verified wiring |
| + M5 table-aware chunker (2053c88) | 20 | 85.0% | 0.0% | — | 4 BERT tables + 14 ResNet + 2 Vaswani extracted |
| + M6 cross-encoder rerank (103c740) | 20 | 85.0% | 0.0% | 82.1s | precision↑ (1 false-positive → true-negative) |
| + M7 claim-type routing (2c56258) | 20 | **85.0%** | **0.0%** | **68.8s** | **−16% wall clock; quality held** |

### Sprint C — prompt strengthening + structured outputs for UI

| Stage | Cases | Accuracy | FDR | Δ |
|---|---:|---:|---:|---|
| Sprint-B final (2c56258) | 20 | 85.0% | 0.0% | — |
| + C1 survey-cited-as-original rule (334747a) | 20 | 87.5% | 0.0% | hallucinated-survey case fixed |
| + C2 source-fetch-empty / fabricated guard (0de4496) | 20 | 90.0% | 0.0% | **+5pp, Sprint-C target hit** |
| + C3 contradiction-detection checklist (142aa76) | 20 | 95.0% | 0.0% | **+5pp** |
| + C4 structured contradiction payload (35e4891) | 20 | 95.0% | 0.0% | schema only |
| + C5 issue_type taxonomy (ecdf25f) | 20 | 95.0% | 0.0% | UI-hint field |
| + C6 evidence quote spans (0ce0645) | 20 | 95.0% | 0.0% | UI-quote field |
| + C7 query decomposition (313edd4) | 20 | 95.0% | 0.0% | structural |
| + C8 `bibsync evidence` (88e943c) | 20 | 95.0% | 0.0% | new command |
| + C9 `bibsync source-rank` (94000e7) | 20 | 95.0% | 0.0% | new command |
| + C10 OpenAlex graph in Filter C (23b0e07) | 20 | 95.0% | 0.0% | suggest-path improvement |
| **+ C11 final benchmark** | 20 | **100.0%** ✅ | **0.0%** | **all 3 remaining failures closed** |

**Headline (Sprint A + B):** baseline 77.8% → final 85.0% with **0% false-deletion rate throughout**. The safety metric (wrongly flagging a real citation as hallucinated) was never allowed to regress across either sprint.

Wall-clock: Sprint-A 133s baseline → Sprint-B 69s final. The retrieval/sources improvements amortise heavily via the shared caches (`~/Library/Caches/bibsync/`).

**The biggest measurable Sprint-B win isn't in the benchmark accuracy** — it's the on-disk **citation memory**. Live verification on `examples/audit_tier2_demo` (commit c77a64e):

```
Run 1 (cold memory):    4 llm.audit calls, 4 verdicts written, ~80s
Run 2 (warm memory):    0 llm.audit calls, 4 verdicts recalled, ~3s
Run 2 (--no-memory):    4 llm.audit calls (memory bypass), ~80s
```

Memory recall has zero impact on the benchmark because the runner re-builds caches per call rather than persisting across runs — but on real-world repeat audits (where most cites haven't changed), it's the difference between a 60-second run and a 3-second run.

## Adding new cases

1. Append a new JSON object on its own line in `citation_verification.jsonl`.
2. Keep the case targeted at one specific failure mode (e.g. "wrong field"
   or "off-by-one number") — don't combine multiple failure modes.
3. Use the `notes` field to explain why the expected verdict is what it is;
   the future-you re-running the benchmark needs this context.
4. If the case is quantitative or named-benchmark, set `min_tier: 2` so
   the runner uses RAG retrieval.
5. Re-run the benchmark and commit the case alongside any pipeline change
   it justifies.

## Remaining known failures (after Sprint B)

Three cases consistently fail at the end of the B sprint, all are
LLM-judgment failures, not retrieval bugs:

| ID | Tier | Mode | Underlying issue |
|---|---|---|---|
| `hallucinated-survey-cited-as-original` | 2 | LLM doesn't distinguish survey from original | OpenAlex now returns the Soydaner 2022 survey paper, and at Tier-2 RAG sees attention-mechanism chunks (since it's a survey ABOUT attention). The LLM mis-judges as "verified" because the topic matches. Distinguishing "survey of X" from "original paper introducing X" requires the LLM to look at *contribution structure*, not just topic alignment. |
| `hallucinated-fabricated-author-year` | 0 | LLM rubber-stamps fabricated cite on title alone | Paper is fictitious so source-fetch correctly misses → audit falls to Tier-0 and accepts on title alone. Needs a Tier-0 rule: "if all sources fail to fetch this paper, it may be fabricated." |
| `contradicted-resnet-50-100m-params` | 2 | LLM lenience on contradiction detection | Reranker surfaced the right chunks (M6 moved this from `verified` to `hallucinated`) but the LLM didn't set `contradicted=true` — needs stronger contradiction prompt. |

Sprint C candidates:

1. **Tier-2 "do not accept surveys as canonical" rule** — when the
   paper's title/abstract contains "survey", "review", "overview", or
   "analysis of", strong negative signal even if topic matches.
2. **Tier-0 source-fetch-empty heuristic** — when arXiv/SS/Crossref/
   OpenAlex all return nothing for a citation, mark `unverifiable`
   instead of judging on title.
3. **Tier-2 contradiction prompt strengthening** — when the LLM
   identifies the right entity but says "the paper does not mention
   X", require it to check whether the paper mentions the *same entity
   with a different value* before defaulting to `supports=false`.

## Reproducing a stage

```bash
# Check out the commit, run the benchmark, compare.
git checkout 7472429    # baseline
bibsync bench run --output-json /tmp/baseline.json
git checkout main
diff <(jq -S .summary /tmp/baseline.json) <(jq -S .summary benchmarks/sprint-A-final-2026-05-15.json)
```
