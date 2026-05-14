# Tier-2 audit test fixture

Exercises the full RAG audit pipeline against real arXiv papers:

  arXiv lookup → PDF download → text extraction → chunking → embedding →
  semantic retrieval → LLM judges with retrieved passages as evidence.

## What's in here

| File | What it contains |
|---|---|
| `paper.tex` | 4 academic claims, each with a `\cite{}`. **2 true, 2 hallucinated.** |
| `references.bib` | 4 real arXiv papers (Vaswani 2017, BERT, GPT-3, ResNet) |

## Expected audit verdicts at each tier

| Claim | Tier 0 (metadata) | Tier 1 (abstract) | Tier 2 (RAG) |
|---|---|---|---|
| 1. Transformer / multi-head self-attention / Vaswani 2017 | verified | verified | **verified** (RAG retrieves the self-attention passage) |
| 2. BERT-base has 12 layers + 110M params / Devlin 2019 | verified | verified (abstract mentions BERT-base) | **verified** (RAG retrieves the exact architecture spec) |
| 3. GPT-3 → 86.5% on MedQA / Brown 2020 | likely verified (wrong; topic-ish match) | likely verified (GPT-3 abstract is on-topic) | **hallucinated** (RAG retrieves few-shot eval passages — no MedQA, no 86.5%) |
| 4. ResNet-50 → 100B params → protein structure / He 2016 | hallucinated (topic mismatch obvious) | hallucinated | **hallucinated** |

Claim 3 is the headline test — Tier 2 should catch it where lower tiers don't,
because the GPT-3 paper is topic-ish for any LM claim but only the actual
retrieved passages can disambiguate the *specific* MedQA/86.5% misattribution.

## Run it end-to-end

See [`docs/tier2-testing.md`](../../docs/tier2-testing.md) (or the snippet
in the project root README under "Try Tier 2") for the full commands.
Short version:

```bash
pip install -e ".[openai,audit-rag]"

# Force a fresh fetch (no caches)
rm -rf "$HOME/Library/Caches/bibsync"   # macOS path; see XDG cache on Linux

bibsync --debug audit examples/audit_tier2_demo \
                      --bib examples/audit_tier2_demo/references.bib \
                      --tier 2 --rag-top-k 5 \
                      2> /tmp/tier2.log

# Inspect what the pipeline actually did
grep audit.fetch /tmp/tier2.log   # source lookups (arxiv/SS/crossref)
grep audit.pdf   /tmp/tier2.log   # PDF download + extract events
grep audit.rag   /tmp/tier2.log   # chunking + embeddings + retrieval
grep llm.audit   /tmp/tier2.log   # LLM verdicts (tier-aware)

# Inspect on-disk caches (built on first run, reused on re-runs)
ls -la "$HOME/Library/Caches/bibsync/paper_content/"  # arXiv/SS/Crossref hits
ls -la "$HOME/Library/Caches/bibsync/pdfs/"           # downloaded PDFs + extracted .txt
ls -la "$HOME/Library/Caches/bibsync/embeddings/"     # per-paper embedding caches
```

## Reset between runs

`paper.tex` is read-only unless you pass `--fix`. Even with `--fix` only the
hallucinated `\cite{}` calls are rewritten — `git checkout examples/audit_tier2_demo`
puts everything back.

## Note on embeddings

Tier 2 calls the OpenAI-compatible `embeddings` endpoint on whichever LLM
provider you've configured. **OpenRouter does not currently route to all
embedding models** — if your `bibsync config show` resolves to an OpenRouter
key and Tier 2 silently degrades to Tier 1 (look for `audit.rag embeddings
request failed` in the trace), set an OpenAI key just for embeddings:

```bash
bibsync config set openai_key sk-...      # OpenAI native
# or use a Together / DeepInfra / Fireworks key configured for embeddings
```

The system will use OpenAI for embeddings and your existing OpenRouter (or
whatever) key for LLM completions, as configured.
