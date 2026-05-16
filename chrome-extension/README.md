# BibSync Chrome extension (Overleaf citation assistant)

Open Overleaf, open the BibSync side panel, click **Check** — see which
`\cite{}` calls are verified / hallucinated / contradicted, with evidence
quotes from the actual papers. Apply user-approved fixes (remove a
hallucinated cite, insert a suggested one) through a before/after diff
preview — nothing edits your manuscript without an explicit Accept click.

## Architecture

```
Overleaf page
  └─ content script  (reads editor selection / document via overleafAdapter)
       │  chrome.tabs message
       ▼
Side panel  (index.ts)
       │  fetch()  — direct, host_permissions grants localhost access
       ▼
bibsync serve   →   BibSync core (audit / evidence / RAG / memory)
```

The side panel is a `chrome-extension://` page; with `host_permissions`
for `http://127.0.0.1:38476/*` declared in the manifest it `fetch()`es
the local server directly — no native-messaging host, no wrapper script.

The extension never runs the AI. It ships the editor's text to `bibsync
serve`, which runs the verification pipeline locally.

## Prerequisites

1. **Python BibSync installed** with the server extras:
   ```bash
   pip install -e ".[server,audit-rag]"
   ```
2. **An LLM key configured** (`bibsync config set openrouter_key sk-or-...`).
3. **Node.js 18+** to build the extension.

## Setup

### 1. Build the extension
```bash
cd chrome-extension
npm install
npm run build        # → dist/
```

### 2. Load it in Chrome
1. Visit `chrome://extensions`.
2. Toggle **Developer mode** (top-right).
3. Click **Load unpacked** → select `chrome-extension/dist/`.

### 3. Start the server
```bash
bibsync serve
```
Leave it running in a terminal. By default it runs **without auth** on
`127.0.0.1:38476` — nothing else to configure.

> Shared machine? Use `bibsync serve --token`, then paste the printed
> token into the extension's **Settings → Server token** field.

### 4. Use it
1. Open an Overleaf project.
2. Click the BibSync toolbar icon → the side panel opens.
3. The header shows **connected** within ~2s.
4. Click **Check selected text** — audits every `\cite{}` in the current
   file and lists issues.
5. Select a sentence without a citation and click **Find citation for
   selection** — suggests candidate papers with evidence quotes.

## Issue colours

| Colour | Meaning |
|---|---|
| 🟢 green | verified — the cited paper supports the claim |
| 🔴 red | wrong / unsupported / survey-cited-as-original / missing bib entry |
| 🟠 orange | contradicted — the paper reports a *different* value |
| 🟡 yellow | needs review — low confidence or source unavailable |

## Troubleshooting

| Symptom | Fix |
|---|---|
| Header stuck on "BibSync not running" | Start `bibsync serve` in a terminal |
| Header shows "connection error" + you used `--token` | Paste the token into Settings → Server token (or restart `bibsync serve` without `--token`) |
| "Couldn't read the Overleaf editor" | Click into the editor pane first; reload the Overleaf tab |
| Still red after a code change | Rebuild (`npm run build`) and reload the extension at `chrome://extensions` |

## Tabs

| Tab | What it does |
|---|---|
| **Check** | Audit the current file's `\cite{}` calls; find citations for a selection |
| **Memory** | Inspect / forget the decisions BibSync has stored for this Overleaf project |
| **Settings** | Evidence tier (0/1/2), embedding backend, RAG top-k — persisted in `chrome.storage.local` |

Memory is scoped per Overleaf project: the project ID in the URL
(`overleaf.com/project/<id>`) is the memory namespace key, so re-running
Check on the same project recalls prior verdicts instead of re-calling
the LLM.

## Editing (Sprint F)

The extension can now apply user-approved edits:

| Action | Where | What it does |
|---|---|---|
| **Remove citation** | issue card (hallucinated / wrong) | strikes the `\cite{}`; writes a `reject` memory record |
| **Insert \cite** | candidate card (Find flow) | inserts `~\cite{key}` after your selection |
| **Ignore** | any problem card | writes an `override` memory record — won't be re-flagged |
| **Mark verified** | unverifiable card | writes an `accept` memory record |
| **Undo** | banner after any edit | reverts the most recent edit |

Every edit shows a **before/after diff modal** first — nothing touches
the manuscript without an explicit Accept click. If the file changed
since the issue was computed, the modal shows a conflict warning and
disables Accept (re-run Check first).

## Project review (Sprint G)

After a Check, the panel shows:

- **Pre-submission checklist** — a ready / not-ready verdict.
  "Ready" only when nothing is hallucinated, contradicted, or missing
  from the `.bib`.
- **Filter chips** — All / Problems / Verified, to triage many
  citations quickly.
- **Export report** — downloads a self-contained HTML report (shareable
  with a supervisor) plus a JSON report (for tooling / CI).

## Multi-file projects

The extension audits **the file currently open in the Overleaf editor**
— Overleaf only keeps one file's content in the page at a time. To
audit an entire multi-file project, use the CLI:

```bash
bibsync audit /path/to/project --tier 2
```

## What's NOT in this build

- Append-BibTeX-entry edits (the `\cite` insert places the marker;
  the bib entry is copied in manually for now).
- Multi-`.bib`-file target selection.
- Browser-side whole-project audit (use the CLI — see above).

## Development

```bash
npm run watch       # rebuild on change
npm run typecheck   # tsc --noEmit
```

After a rebuild, click the refresh icon on the extension card in
`chrome://extensions`.
