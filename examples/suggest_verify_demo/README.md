# `suggest --verify-tier` test fixture

Demonstrates the **Filter D grounding gate** — the Tier-1 / Tier-2 verification
layer for the `bibsync suggest` command. Mirrors `audit_tier2_demo/` but tests
the **citation-addition** path instead of the audit path.

## What's in here

| File | What |
|---|---|
| `intro.tex` | Two prose claims: one topical (Transformer / self-attention) and one quantitative-trap (GPT-3 → 86.5% on MedQA, which is actually a Med-PaLM 2 number) |
| `references.bib` | Starts empty — `suggest` will populate it |

## Pipeline gates being tested

```
For each anchor (sentence needing a citation):
  Filter A: title-similarity sanity check          (deterministic)
  Filter B: cluster_id dedup                       (deterministic)
  Filter C: verify_claim_support()                 (LLM canonical-paper detector)
  Filter D: _ground_candidate()                    ← THIS is what we're testing
       ├─ tier 1: fetch abstract → audit_citation(abstract=...)
       └─ tier 2: + PDF + RAG retrieval → audit_citation(retrieved_chunks=...)
```

## Expected outcomes per tier

| Tier | Claim A (Transformer / self-attention) | Claim B (GPT-3 → 86.5% MedQA) |
|---|---|---|
| `--verify-tier 0` (today's default) | **added** (Vaswani 2017) | **added** (Brown 2020) ⚠️ wrong-attribution, since Filter D never ran |
| `--verify-tier 1` (abstract grounding) | **added** (Vaswani 2017) | **no_grounded_match** — abstract doesn't mention MedQA |
| `--verify-tier 2` (PDF-RAG grounding) | **added** (Vaswani 2017) | **no_grounded_match** with verbatim reasoning ("no chunk mentions 'MedQA' or '86.5%'") |

**Headline test: Claim B at `--verify-tier 2`** — the canonical GPT-3 paper IS
the right paper for the topic (Filter C accepts), but the specific number in
the claim is wrong (Filter D rejects). This is exactly the misattribution that
LLM-assisted writing produces and that the audit pipeline was originally built
to catch — now caught at *write* time, not just at audit time.

## Run it

```bash
# Tier 0 — baseline (today's behaviour). Likely accepts both, including the bad one.
bibsync --debug suggest examples/suggest_verify_demo/intro.tex \
                        --bib examples/suggest_verify_demo/references.bib \
                        --auto \
                        2> /tmp/suggest-t0.log

# Reset the fixture (suggest mutates intro.tex + references.bib)
git checkout examples/suggest_verify_demo

# Tier 2 — full RAG grounding. Should accept Claim A, reject Claim B.
bibsync --debug suggest examples/suggest_verify_demo/intro.tex \
                        --bib examples/suggest_verify_demo/references.bib \
                        --auto \
                        --verify-tier 2 --verify-rag-top-k 5 \
                        2> /tmp/suggest-t2.log

# What you should see at the bottom of the Tier-2 run:
#   ⚠  1 citation(s) rejected by Filter D (grounding) — the canonical paper
#       was found, but it doesn't actually support the claim:
#       • Para 1 anchor 'GPT-3 achieves 86.5\\% ...': grounded rejection
#         (tier 2): The abstract and retrieved chunks do not mention 'MedQA'
#         or '86.5%'
```

## Inspect the trace

```bash
# Filter C decisions (canonical-paper detection)
grep "suggest.verify"  /tmp/suggest-t2.log

# Filter D decisions (grounding gate) — this is the new pipeline
grep "suggest.ground"  /tmp/suggest-t2.log

# Final per-suggestion status
grep "suggest.commit\|suggest.resolve" /tmp/suggest-t2.log
```

## Reset between runs

```bash
git checkout examples/suggest_verify_demo
# OR manually: truncate references.bib + revert intro.tex
```
