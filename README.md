# BibSync

Take a `.bib` file, replace every entry with its canonical version from
Google Scholar, and keep your `\cite{}` calls in `.tex` files in sync —
even when cite keys change. LLM-verified at every match to prevent the
hallucination class of failures.

## The one workflow

```bash
bibsync fix --bib references.bib --project .
```

For each entry in `references.bib`:

1. **Search Google Scholar** by title.
2. **Heuristic filter** — discard candidates whose first-author surname or
   year disagree with your entry.
3. **LLM-as-judge verifies identity** — for each remaining candidate, an LLM
   call decides "is this the same paper?". Different first author, different
   topic, or "derivative work" (chapter ABOUT vs. the original) → rejected.
   Only matches with confidence ≥ 0.7 are accepted.
4. **Fetch official BibTeX** from Scholar for the verified match.
5. **Regenerate the cite key** from the corrected metadata
   (`firstauthor + year + firsttitleword`). If your old entry had the year
   wrong (`wang2019` → corrected to `wang2021`), the new key reflects it.
6. **Propagate the rename** — every `\cite{wang2019}` in every `.tex` under
   `--project` becomes `\cite{wang2021}`. The `.bib` and the prose stay in
   sync, atomically.

Entries that fail any step are reported as **`unverified`** and **left
untouched**. The default stance is "never silently corrupt your data".

## Why LLM verification is the safety net

Google Scholar's title search is noisy. A query for `Attention is all you
need` can return *"Is Attention All You Need?"* — a 2025 book chapter by a
different author — as the top result. Pure heuristics (title fuzz, author
fuzz) catch some of these, but not the close calls. The LLM judge is the
final gate: given the original `.bib` entry and a candidate, it reasons
about whether they're the *same paper* with rules like:

* First-author surname agrees (allowing transliteration / abbreviation)
* Title matches semantically (allowing case / punctuation drift)
* Year within ±2 (preprint → proceedings drift is normal)
* The candidate is not a derivative work (review, follow-up, chapter ABOUT)

Each verdict is shown in the report:

```
mineault2025attention → unverified
LLM verdict: conf=0.95 — "Different first author (Mineault vs Vaswani);
                          a 2025 book chapter ABOUT the original."
```

## Install

```bash
git clone https://github.com/<you>/BibSync.git && cd BibSync
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[openai]"
playwright install chromium       # one-time, ~150 MB
```

## OpenAI or OpenRouter API key

`fix` requires an LLM key — match verification is what makes the pipeline
safe. Provider is auto-detected from the key prefix:

| Key prefix | Provider | Default model |
|---|---|---|
| `sk-or-...` | OpenRouter | `openai/gpt-4o-mini` |
| `sk-...` | OpenAI | `gpt-4o-mini` |

Set it once:

```bash
bibsync config set openrouter_key sk-or-v1-...
# or
bibsync config set openai_key sk-...

bibsync config show     # prints redacted key + which provider was picked up
```

Resolution order: `OPENROUTER_API_KEY` / `OPENAI_API_KEY` env vars → `.env`
file in cwd → config file at
`~/Library/Application Support/bibsync/config.json` (macOS) or
`~/.config/bibsync/config.json` (Linux). Config file is `chmod 600`.

You can pick a stronger model (Claude, Gemini, GPT-4o) per command:

```bash
bibsync fix --bib refs.bib --project . --model anthropic/claude-3.5-sonnet
```

## Quick demo

The bundled `examples/fix_demo/` has a `.bib` with intentional errors plus
a `paper.tex` that cites those keys.

```bash
git checkout examples/         # reset to pristine state
bibsync fix --bib examples/fix_demo/references.bib --project examples/fix_demo
```

After the run, `git diff examples/fix_demo/` shows exactly what changed in
both the `.bib` and `paper.tex`.

## Run every command (against bundled examples)

Each command has a self-contained fixture under `examples/`. Reset state with
`git checkout examples/` before each run.

```bash
# 1. fix      — verify + rewrite .bib, propagate \cite renames to .tex
bibsync fix --bib examples/fix_demo/references.bib --project examples/fix_demo

# 2. extract  — decode placeholder \cite{key} keys, populate .bib from Scholar
bibsync extract examples/extract_demo/intro.tex --bib examples/extract_demo/references.bib

# 3. repair   — convert legacy \bibitem{} blocks into verified BibTeX
bibsync repair examples/repair_demo/old_bibliography.tex --bib examples/repair_demo/repaired.bib

# 4. suggest  — read prose, suggest + insert citations interactively
bibsync suggest examples/suggest_demo/intro.tex --bib examples/suggest_demo/references.bib

# 5. scan     — diagnose missing + orphan cite keys across a project (read-only)
bibsync scan examples/scan_demo

# 6. verify   — read-only audit of a .bib against Scholar
bibsync verify --bib examples/fix_demo/references.bib

# 7. search   — preview Scholar results for a title
bibsync search "Attention Is All You Need"

# 8. add      — add a single paper by title to a .bib
bibsync add "BERT pre-training of deep bidirectional transformers" --bib /tmp/scratch.bib

# 9. config   — manage stored API key + model
bibsync config show
bibsync config set llm_model anthropic/claude-3.5-sonnet
```

See [`examples/README.md`](examples/README.md) for a per-demo description
of what each fixture contains and what output to expect.

## CLI flags

```bash
bibsync fix --bib <file>          # required: the .bib to fix
            --project <dir>       # optional: scan this dir for .tex files to update
            --model <model_id>    # optional: override the LLM (e.g. anthropic/claude-3.5-sonnet)
            --preserve-keys       # optional: keep original cite keys; do not regenerate
            --headless            # optional: run Chrome headlessly (CAPTCHAs more likely)
            --delay 1.5           # optional: seconds between Scholar lookups
```

## Browser opens visibly — why?

Google Scholar aggressively CAPTCHAs headless browsers. BibSync uses a
**persistent Chrome profile** (stored under your OS user data dir) so the
first solved CAPTCHA carries over to subsequent runs. The window is
visible by default; solve any CAPTCHA once and continue.

`--headless` is supported but expect more frequent challenges.

## Additional commands (less used)

These exist for completeness; you typically only need `fix`:

| Command | What it does |
|---|---|
| `bibsync add "<paper title>"` | Add a single paper by title. |
| `bibsync verify --bib <file>` | Read-only audit; no rewrites. |
| `bibsync scan <dir>` | Diagnostic — list missing/orphan cite keys. |
| `bibsync search "<query>"` | Preview Scholar results without writing. |
| `bibsync extract <paper.tex>` | Given LLM-drafted prose with `\cite{key}` placeholders, decode each key from context and populate the `.bib`. |
| `bibsync repair <bibitems.tex>` | Convert legacy `\bibitem{...}` blocks to BibTeX. |
| `bibsync suggest <paper.tex>` | Read prose without citations, propose `\cite{}` per claim (LLM identifies the canonical paper from world knowledge, Scholar confirms). |
| `bibsync audit <dir> [--tier N]` | Verify every existing `\cite{}` actually supports its claim. Three evidence tiers; pass `--fix` to remove hallucinated citations. See below. |

### `bibsync audit` — citation verification engine

For every `\cite{key}` in your project, audit checks whether the cited paper actually
supports the surrounding prose claim. Three evidence tiers, configurable per run:

| Tier | What it does | Catches | Extra latency |
|---|---|---|---|
| **0** | metadata only — title + authors + year + venue from the `.bib` entry | gross topic mismatches (e.g. a CV paper cited for an NLP claim) | one LLM call per (claim, paper) |
| **1** (default) | also fetches the paper's **abstract** from arXiv → Semantic Scholar → Crossref (with on-disk cache) | misattributions where the title is on-topic but the actual contribution differs | + one HTTP call per unique paper, cached |
| **2** | also downloads the open-access **PDF**, chunks it, embeds the chunks, and retrieves the **top-K most claim-relevant passages** for the LLM to ground its verdict in | specific numerical / factual mismatches the abstract doesn't disambiguate | + PDF download + embeddings call per paper, both cached |

Higher tiers gracefully degrade per-citation: a paper not on arXiv/SS/Crossref still
gets a Tier-0 audit; a paper with no open-access PDF still gets a Tier-1 audit.

```bash
# Tier 1 (default): abstract-grounded verdicts for everything findable
bibsync audit . --bib references.bib

# Tier 2: full RAG against the PDFs (needs pypdf and an embeddings-capable API key)
pip install -e ".[openai,audit-rag]"
bibsync audit . --bib references.bib --tier 2 --rag-top-k 5

# Auto-remove hallucinated cites (replaced with a marker comment preserving the LLM's reasoning)
bibsync audit . --bib references.bib --fix

# Demo: 4 verified, 2 hallucinated, 1 missing_in_bib, 1 commented-out (ignored)
bibsync audit examples/audit_demo --bib examples/audit_demo/references.bib
```

The report shows the evidence tier each citation was actually verified at (`meta`,
`abs`, or `RAG×N` where N is the retrieved chunk count) so you can see at a glance
whether the verdict was based on the abstract or the actual paper text.

## File layout

```
BibSync/
├── bibsync/
│   ├── cli.py          # Click subcommands
│   ├── fix.py          # bib repair + .tex sync (LLM-verified per-paper match)
│   ├── audit.py        # citation verification engine (tiers 0/1/2)
│   ├── audit_sources/  # arXiv / Semantic Scholar / Crossref / PDF / cache
│   ├── audit_rag.py    # chunking + embeddings + retrieval for Tier-2 audit
│   ├── llm.py          # OpenAI/OpenRouter client, prompts, verify_match agent
│   ├── scholar.py      # Playwright scraper (persistent profile, CAPTCHA-aware)
│   ├── picker.py       # Canonical-version selection
│   ├── bibtex.py       # Parse / dedupe / derive_cite_key / atomic write
│   ├── tex_rewrite.py  # Comment-aware \cite{} rename + insertion
│   ├── scanner.py      # .tex citation parser + paragraph-context extraction
│   ├── config.py       # API-key resolution chain
│   ├── verify.py       # Read-only .bib audit (legacy)
│   ├── extract.py      # \cite{placeholder} → LLM-decode → Scholar add (legacy)
│   ├── repair.py       # \bibitem{} → BibTeX (legacy)
│   ├── suggest.py      # prose → suggested citations (experimental)
│   └── models.py       # PaperHit dataclass
└── pyproject.toml
```

## Status

Alpha. `fix` is the focus and is tested end-to-end with mocks for the
adversarial cases (wrong-paper rejection by LLM, year-driven key
rename + .tex propagation). Other commands work but are less load-bearing.

## License

MIT.
