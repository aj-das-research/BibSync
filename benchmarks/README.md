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

## Tier-A sprint result snapshot

| Stage | Cases | Accuracy | FDR | Δ vs prior |
|---|---:|---:|---:|---|
| Baseline (commit 7472429) | 18 | 77.8% | 0.0% | — |
| + A1 OpenAlex (3841ded) | 18 | 83.3% | 0.0% | **+5.5pp** |
| + A2 Unpaywall (dbbd57c) | 18 | 88.9% | 0.0% | **+5.6pp** |
| + A3 hybrid BM25 (e1166dd) | 18 | 88.9% | 0.0% | 0 (structural; cases didn't stress it) |
| + A4 contradicted (ee3e938) | **20** | 85.0% | 0.0% | new cases added, retained safety |
| + A5 JSON output (f17374d) | 20 | 85.0% | 0.0% | 0 (cosmetic on the metric) |

**Headline:** baseline 77.8% → final 85.0% with **0% false-deletion rate
throughout** — the headline safety metric (wrongly flagging a real citation
as hallucinated) was never allowed to regress.

Wall-clock on a warm cache: 133s → 58s — the extra sources fail-fast on
cache hits, and hybrid BM25 + contradiction retrieval add only local
computation, no extra LLM calls.

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

## Remaining known failures

Three cases consistently fail at the end of the Tier-A sprint:

| ID | Tier | Mode | Underlying issue |
|---|---|---|---|
| `hallucinated-survey-cited-as-original` | 0 | LLM topical-only verify | Source fetch returned nothing → judge sees title only ("Attention mechanism in neural networks…") and rubber-stamps |
| `hallucinated-fabricated-author-year` | 0 | LLM topical-only verify | Paper is fictitious so source-fetch correctly misses → judge falls to Tier-0 and accepts on title alone |
| `contradicted-resnet-50-100m-params` | 2 | LLM lenience | Hybrid retrieved the right chunks but judge said "widely accepted that ResNet-50 has around 100M parameters" — overruled the retrieved evidence with world knowledge |

These are LLM-judgment failures at Tier-0 (the first two) or LLM
prior-overrides-evidence at Tier-2 (the third), not retrieval bugs.
Sprint B should focus on:
1. Tier-0 prompt strengthening for source-fetch-failed cases
   ("if no abstract is fetchable, the citation may be fabricated").
2. Tier-2 prompt strengthening against "but it's widely known…"
   priors that override retrieved evidence.

## Reproducing a stage

```bash
# Check out the commit, run the benchmark, compare.
git checkout 7472429    # baseline
bibsync bench run --output-json /tmp/baseline.json
git checkout main
diff <(jq -S .summary /tmp/baseline.json) <(jq -S .summary benchmarks/sprint-A-final-2026-05-15.json)
```
