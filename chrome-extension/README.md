# BibSync Chrome extension (Overleaf citation assistant)

Read-only MVP: open Overleaf, open the BibSync side panel, click **Check** —
see which `\cite{}` calls are verified / hallucinated / contradicted, with
evidence quotes from the actual papers.

> **Sprint E scope.** This is the read-only build — it displays issues and
> evidence but does **not** edit your manuscript. User-approved editing
> (insert / replace / append) lands in Sprint F.

## Architecture

```
Overleaf page
  └─ content script  (reads editor selection / document via overleafAdapter)
       │  chrome.tabs message
       ▼
Side panel  (index.ts)
       │  chrome.runtime message  { kind: "native", request }
       ▼
Service worker  (serviceWorker.ts)
       │  chrome.runtime.connectNative
       ▼
Native messaging host  (../native-host/bibsync_native_host.py)
       │  HTTP + Bearer token
       ▼
bibsync serve   →   BibSync core (audit / evidence / RAG / memory)
```

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
4. Copy the extension's **ID** (a 32-char string under its name).

### 3. Install the native messaging host
```bash
bibsync native-host install --extension-id <THE-ID-FROM-STEP-2>
```
(For quick local dev you can use `--extension-id "*"`, which lets any
extension launch the host — fine on a personal machine, not for a
shared one.)

### 4. Start the server
```bash
bibsync serve
```
Leave this running in a terminal. It writes a bearer token the native
host reads automatically.

### 5. Use it
1. Open an Overleaf project.
2. Click the BibSync toolbar icon → the side panel opens.
3. The header shows **connected** once it reaches `bibsync serve`.
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
| Header shows "connection error" | Re-run `bibsync native-host install` with the correct extension ID |
| "Couldn't read the Overleaf editor" | Click into the editor pane first; reload the Overleaf tab |
| Native host log | `~/Library/Logs/bibsync-native-host.log` (macOS) |

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

## What's NOT in this build (Sprint F / G)

- Applying edits (insert / replace / append BibTeX) — patch model exists
  server-side; the UI gate is Sprint F.
- Full-project audit + batch review (Sprint G).

## Development

```bash
npm run watch       # rebuild on change
npm run typecheck   # tsc --noEmit
```

After a rebuild, click the refresh icon on the extension card in
`chrome://extensions`.
