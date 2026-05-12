# BibSync examples

Self-contained demos — one per command. Run from the project root. Every demo
modifies the files in its folder, so use `git checkout examples/` to reset
between runs.

```
examples/
├── fix_demo/         # bibsync fix       — verify + rewrite .bib + propagate \cite to .tex
├── extract_demo/     # bibsync extract   — decode placeholder \cite{key} keys into a .bib
├── repair_demo/      # bibsync repair    — convert \bibitem{} blocks into BibTeX
├── suggest_demo/     # bibsync suggest   — read prose, suggest + insert citations
├── scan_demo/        # bibsync scan      — diagnose missing + orphan cite keys
└── (verify reuses fix_demo's .bib)
```

---

## 1. `fix_demo/` — the main workflow

The starting state: a `references.bib` with intentional errors plus a
`paper.tex` that cites all four keys.

What's wrong in `references.bib`:

| Key | What's wrong |
|---|---|
| `vaswani2017attention` | Year says `2016` (should be 2017). Journal `arXiv preprint` (should be NeurIPS, type `@inproceedings`). |
| `he2016resnet` | Correct — should come back `unchanged`. |
| `devlin2018bert` | Year `2018` is the arXiv date; venue is NAACL 2019. |
| `goodfellow2014gan` | Correct — should come back `unchanged`. |

```bash
# Reset to pristine state first
git checkout examples/

# Run with LLM-verified match + .tex propagation
bibsync fix --bib examples/fix_demo/references.bib --project examples/fix_demo

# See what changed
git diff examples/fix_demo/
```

The report shows the LLM's verdict per entry (`conf=0.98 — "Same paper: Vaswani 2017 NeurIPS"`).
Wrong-paper Scholar hits (e.g. "Is Attention All You Need?" by Mineault 2025) are
rejected by the LLM judge — those entries fall to `unverified` and stay untouched.

### Variant: preserve cite keys

```bash
bibsync fix --bib examples/fix_demo/references.bib \
            --project examples/fix_demo \
            --preserve-keys
```

Fields are still corrected, but the original cite keys are kept and the `.tex` is not modified.

---

## 2. `extract_demo/` — placeholder `\cite{}` keys → populated `.bib`

The starting state: a paragraph with `\cite{}` calls referencing placeholder
keys (no `.bib` entries yet — what an LLM-drafted paper looks like). The cite
key itself encodes hints (`moor2023gmai` → Moor + 2023 + acronym "GMAI").

```bash
bibsync extract examples/extract_demo/intro.tex \
                --bib examples/extract_demo/references.bib
```

The LLM decodes each key using the surrounding prose, searches Scholar, fetches
BibTeX, and writes it to the `.bib` — preserving your cite keys so the
`.tex` keeps working.

---

## 3. `repair_demo/` — legacy `\bibitem{}` → BibTeX

The starting state: a `\begin{thebibliography}` block with hand-typed
entries in the old (pre-BibTeX) format.

```bash
bibsync repair examples/repair_demo/old_bibliography.tex \
               --bib examples/repair_demo/repaired.bib
```

The LLM parses each block, cross-checks against Scholar, and writes verified
BibTeX to `repaired.bib`. Omit `--bib` to preview the output on stdout.

---

## 4. `suggest_demo/` — text → suggested citations

The starting state: a paragraph of prose with **no** citations at all.

```bash
bibsync suggest examples/suggest_demo/intro.tex \
                --bib examples/suggest_demo/references.bib
```

For each paragraph that has no `\cite{}`, the LLM identifies what should be
cited (named methods like `Med-PaLM`, attributed claims, foundational
works). Each suggestion is shown interactively — accept (`y`), reject (`n`),
or quit (`q`). On accept, the entry is appended to the `.bib` and a
`\cite{}` is inserted into the `.tex` at the anchor phrase.

> **Note:** `suggest` is experimental — it uses heuristic Scholar matching, not the
> LLM-judge verification that `fix` uses, so accept suggestions critically and
> follow up with `bibsync fix` to clean up any wrong matches.

---

## 5. `scan_demo/` — diagnose missing + orphan cite keys (read-only)

The starting state: a `paper.tex` that cites four keys (two of them are made
up — likely hallucinated by an LLM) and a `references.bib` that defines four
keys (two of them are never cited).

```bash
bibsync scan examples/scan_demo
```

Expected output:

```
Missing (cited but not defined in any .bib — possible hallucinations):
  • another_missing_2023
  • ghost_paper_2024

Orphan entries (defined in .bib but never cited):
  • another_orphan
  • unused_orphan_entry
```

The `\cite{commented_out_paper}` inside a LaTeX comment is correctly ignored.

---

## 6. Verify (read-only audit, no new fixture)

`verify` reads the same input shape as `fix` but only **reports** discrepancies —
it never modifies files. Use it for a dry-run before running `fix`:

```bash
bibsync verify --bib examples/fix_demo/references.bib
```

You'll see a table listing year/venue/author mismatches per entry, with no
changes to disk.

---

## Reset between runs

```bash
git checkout examples/
```

This restores every demo to its pristine starting state so you can re-run.
