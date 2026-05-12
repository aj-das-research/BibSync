# BibSync examples

Four self-contained demos — one per workflow. Run from the project root.

> **Tip:** each demo modifies the files in its folder. To re-run a demo from
> scratch, reset with `git checkout examples/`.

---

## 1. `suggest_demo/` — text → citations

The starting state: a paragraph of prose with no citations.

```bash
bibsync suggest examples/suggest_demo/intro.tex \
                --bib examples/suggest_demo/references.bib
```

You'll be prompted (interactively) for each suggested citation. Accept the
ones that look right. When you're done:

* `references.bib` will contain BibTeX entries fetched from Google Scholar
  for each accepted suggestion.
* `intro.tex` will now contain `\cite{...}` calls inserted right after the
  relevant phrases (e.g. `~\cite{moor2023foundation}` after `"vision in 2023"`).

Use `--auto` to accept everything without prompting (not recommended for first runs).

---

## 2. `extract_demo/` — placeholder `\cite{}` keys → populated `.bib`

The starting state: a paragraph that already has `\cite{}` calls with
placeholder keys (no `.bib` entries yet — what you'd get from an LLM-drafted
paper). The cite keys themselves carry hints (`moor2023gmai` = Moor 2023,
topic acronym GMAI).

```bash
bibsync extract examples/extract_demo/intro.tex \
                --bib examples/extract_demo/references.bib
```

BibSync decodes each key (using the surrounding prose as context), searches
Scholar, fetches BibTeX, and writes it to the `.bib` — preserving your cite
keys exactly as the `.tex` uses them.

---

## 3. `repair_demo/` — legacy `\bibitem{}` → BibTeX

The starting state: a `\begin{thebibliography}` block with hand-typed
entries in the old (pre-BibTeX) format.

```bash
bibsync repair examples/repair_demo/old_bibliography.tex \
               --bib examples/repair_demo/repaired.bib
```

BibSync LLM-parses each `\bibitem{...}` block, cross-checks the parsed
metadata against Google Scholar (catching the case where the original
bibitem was itself hallucinated), and writes verified BibTeX to
`repaired.bib`. Omit `--bib` to preview the output on stdout without
writing anything.

---

## 4. `fix_demo/` — verify `.bib`, propagate key renames

The starting state: a `references.bib` with **intentional errors**, plus a
`paper.tex` that cites them.

What's wrong in `references.bib`:

| Key | What's wrong |
|---|---|
| `vaswani2017attention` | Year says `2016` (should be 2017). Journal says `arXiv preprint` (should be NeurIPS). |
| `he2016resnet` | Correct — should come back `unchanged`. |
| `devlin2018bert` | Year `2018` is the arXiv date; the venue is NAACL 2019. Journal type vs. inproceedings. |
| `goodfellow2014gan` | Correct — should come back `unchanged`. |

Run the verifier:

```bash
bibsync fix --bib examples/fix_demo/references.bib \
            --project examples/fix_demo
```

After the run:
* `references.bib` has corrected year, venue, and entry-type fields.
* `paper.tex` is untouched (no key changes happened).

Now try the key-regeneration flag:

```bash
bibsync fix --bib examples/fix_demo/references.bib \
            --project examples/fix_demo \
            --regenerate-keys
```

This rebuilds every cite key from `firstauthor + year + firsttitleword`
(e.g., `vaswani2017attention` → same, but `devlin2018bert` may become
`devlin2019bert`). Every `\cite{...}` in `paper.tex` is rewritten to match.

---

## Reset between runs

```bash
git checkout examples/
```
