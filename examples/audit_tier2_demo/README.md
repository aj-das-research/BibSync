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
# Local-first install: open-source embeddings (fastembed), no API key needed
# for the embedding step. You still need an LLM key for completions — any
# OpenRouter / OpenAI / compatible key works.
pip install -e ".[audit-rag]"

# (Optional) If you also want the OpenAI-compatible embeddings backend as a
# fallback, install the [openai] extra too:
#   pip install -e ".[openai,audit-rag]"

# Force a fresh fetch (no caches)
rm -rf "$HOME/Library/Caches/bibsync"   # macOS path; see XDG cache on Linux

bibsync --debug audit examples/audit_tier2_demo \
                      --bib examples/audit_tier2_demo/references.bib \
                      --tier 2 --rag-top-k 5 \
                      2> /tmp/tier2.log

# Explicit-backend variants:
#   --embedding-backend local   # force fastembed (BAAI/bge-small-en-v1.5)
#   --embedding-backend api     # force OpenAI-compatible endpoint
#   --embedding-backend auto    # default: local-first, API as fallback

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

**Default: fully local, open-source, no API key needed.** Tier 2 ships with
[`fastembed`](https://github.com/qdrant/fastembed) under `[audit-rag]`, which
loads [`BAAI/bge-small-en-v1.5`](https://huggingface.co/BAAI/bge-small-en-v1.5)
via ONNX Runtime. The first run downloads the model (~80 MB) into the fastembed
cache; subsequent runs load it from disk in under a second. Retrieval quality
is on par with OpenAI's `text-embedding-3-small` for short academic claims.

If you'd rather use a hosted API endpoint, pass `--embedding-backend api`.
The system issues OpenAI-compatible `/embeddings` calls and picks a
**provider-aware default model** based on your configured LLM key:

| Configured key | Default API embedding model | Why |
|---|---|---|
| `openrouter_key` (`sk-or-...`) | **`baai/bge-m3`** | Open-source BGE family, 8K ctx, ~$0.01 / 1M tokens, routable via OpenRouter's `/v1/embeddings` (unlike `text-embedding-*` which OpenRouter currently does not relay). |
| `openai_key` (`sk-...`) | **`text-embedding-3-small`** | OpenAI native default. |

Override the default with `--embedding-model <id>` (e.g. `--embedding-model
qwen/qwen3-embedding-8b` for max-quality OpenRouter retrieval). Cache
invalidates automatically when the effective model OR backend changes, so
switching never mixes vector spaces.
