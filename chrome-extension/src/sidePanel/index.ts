/**
 * BibSync side panel — main UI controller.
 *
 * Vanilla TypeScript + DOM templating. Three tabs (Check / Memory /
 * Settings). Sprint F adds user-approved editing:
 *   • Insert citation   — additive \cite{} at the selection (Find flow)
 *   • Remove citation   — strike a hallucinated \cite{} (issue cards)
 *   • Ignore warning    — write an `override` memory record
 *   • Mark verified     — write an `accept` memory record
 *   • Undo last edit    — inverse-patch the most recent edit
 *
 * Every edit goes through a preview modal: the user sees the exact
 * before/after diff and clicks Accept. Nothing touches the manuscript
 * without that explicit click. Conflicts (file changed since the patch
 * was built) are detected server-side and surfaced in the modal.
 */
import {
  applyOverleafEdit,
  auditSelection,
  checkHealth,
  findEvidence,
  forgetMemory,
  getMemory,
  getOverleafDocument,
  getOverleafProjectId,
  getOverleafSelection,
  previewPatch,
  rememberMemory,
} from "./api";
import type { Patch } from "./api";
import { DEFAULT_SETTINGS } from "../types";
import type {
  CitationCheck,
  ConnectionState,
  EvidenceCandidate,
  EvidenceSpan,
  MemoryRecord,
  Settings,
} from "../types";

// ── DOM handles ─────────────────────────────────────────────────────────────
const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;
const $$ = <T extends HTMLElement>(sel: string) =>
  Array.from(document.querySelectorAll(sel)) as T[];

const connEl = $("conn");
const connLabel = $("conn-label");
const btnCheck = $<HTMLButtonElement>("btn-check");
const btnEvidence = $<HTMLButtonElement>("btn-evidence");
const statusLine = $("status-line");
const results = $("results");

// ── settings (chrome.storage.local) ─────────────────────────────────────────

let settings: Settings = { ...DEFAULT_SETTINGS };
const SETTINGS_KEY = "bibsync.settings";

async function loadSettings(): Promise<void> {
  const stored = await chrome.storage.local.get(SETTINGS_KEY);
  if (stored[SETTINGS_KEY]) {
    settings = { ...DEFAULT_SETTINGS, ...stored[SETTINGS_KEY] };
  }
  ($("set-tier") as HTMLSelectElement).value = String(settings.tier);
  ($("set-backend") as HTMLSelectElement).value = settings.embeddingBackend;
  ($("set-topk") as HTMLInputElement).value = String(settings.ragTopK);
  ($("set-token") as HTMLInputElement).value = settings.serverToken;
}

async function saveSettings(): Promise<void> {
  settings = {
    tier: Number(($("set-tier") as HTMLSelectElement).value) as Settings["tier"],
    embeddingBackend: ($("set-backend") as HTMLSelectElement)
      .value as Settings["embeddingBackend"],
    ragTopK: Math.max(
      1,
      Math.min(20, Number(($("set-topk") as HTMLInputElement).value) || 5),
    ),
    serverToken: ($("set-token") as HTMLInputElement).value.trim(),
  };
  await chrome.storage.local.set({ [SETTINGS_KEY]: settings });
  const el = $("set-status");
  el.textContent = "Saved.";
  setTimeout(() => (el.textContent = ""), 1500);
}

// ── tab switching ────────────────────────────────────────────────────────────

function switchTab(name: string): void {
  $$(".tab").forEach((b) =>
    b.classList.toggle("tab-active", b.dataset.tab === name),
  );
  $$<HTMLElement>(".tab-view").forEach((v) => {
    v.hidden = v.dataset.view !== name;
  });
  if (name === "memory") void loadMemory();
}

// ── connection status (E14) ─────────────────────────────────────────────────

function setConnection(state: ConnectionState, detail = ""): void {
  connEl.className = `conn conn-${state}`;
  const labels: Record<ConnectionState, string> = {
    connecting: "connecting…",
    connected: "connected",
    disconnected: "BibSync not running",
    error: "connection error",
  };
  connLabel.textContent = labels[state];
  connEl.title = detail || labels[state];
  const offline = state !== "connected";
  btnCheck.disabled = offline;
  btnEvidence.disabled = offline;
}

async function pollHealth(): Promise<void> {
  try {
    const h = await checkHealth();
    setConnection("connected", `bibsync ${h.version}`);
  } catch (e) {
    const msg = e instanceof Error ? e.message : String(e);
    setConnection(
      msg.includes("could not reach") || msg.includes("disconnected")
        ? "disconnected"
        : "error",
      msg,
    );
  }
}

// ── helpers ──────────────────────────────────────────────────────────────────

function esc(s: string): string {
  const d = document.createElement("div");
  d.textContent = s ?? "";
  return d.innerHTML;
}
function truncate(s: string, n: number): string {
  return s.length > n ? s.slice(0, n) + "…" : s;
}
function severityClass(issueType: string, status: string): string {
  if (status === "verified") return "sev-green";
  switch (issueType) {
    case "wrong_reference":
    case "unsupported_reference":
    case "survey_cited_as_original":
    case "missing_bib_entry":
      return "sev-red";
    case "contradicted_claim":
      return "sev-orange";
    case "source_unavailable":
    case "weak_support":
    case "needs_user_review":
      return "sev-yellow";
    default:
      return status === "hallucinated" ? "sev-red" : "sev-yellow";
  }
}
function setStatus(text: string, isError = false): void {
  statusLine.textContent = text;
  statusLine.className = isError ? "status-line error" : "status-line";
}
function setBusy(text: string): void {
  statusLine.innerHTML = `<span class="spinner"></span> ${esc(text)}`;
  statusLine.className = "status-line";
}
function emptyState(text: string): void {
  results.innerHTML = `<div class="empty">${esc(text)}</div>`;
}
function uid(): string {
  return "patch_" + Math.random().toString(36).slice(2, 12);
}

// ── module state ─────────────────────────────────────────────────────────────

/** The most-recent applied edit, kept so Undo can apply the inverse. */
let lastEdit: { file: string; start: number; newText: string; oldText: string } | null =
  null;
/** Last selection from the Find-citation flow — Insert needs the insert point. */
let lastSelection: { start: number; end: number; text: string } | null = null;
/** Current audit checks, keyed by an index so card buttons can look them up. */
let currentChecks: CitationCheck[] = [];
let currentCandidates: EvidenceCandidate[] = [];

// ── evidence + issue cards (E8, E9) ─────────────────────────────────────────

function evidenceListHtml(spans: EvidenceSpan[] | undefined): string {
  if (!spans || spans.length === 0) return "";
  return `<div class="evidence-list">${spans
    .map((s) => {
      const page = s.page != null ? `p.${s.page}` : "—";
      return `<div class="evidence-span ev-${s.type}">
        <span class="ev-meta">[${esc(s.type)}] ${esc(page)}</span>
        <div class="ev-quote">${esc(s.quote ?? "")}</div></div>`;
    })
    .join("")}</div>`;
}

/** Action buttons for an issue card, depending on its status. */
function issueActionsHtml(c: CitationCheck, idx: number): string {
  const buttons: string[] = [];
  const problem = ["hallucinated", "contradicted", "unverifiable"].includes(c.status);
  if (
    c.status === "hallucinated" ||
    c.issue_type === "wrong_reference" ||
    c.issue_type === "unsupported_reference" ||
    c.issue_type === "survey_cited_as_original"
  ) {
    buttons.push(`<button data-remove="${idx}">Remove citation</button>`);
  }
  if (c.status === "unverifiable") {
    buttons.push(`<button data-mark-verified="${idx}">Mark verified</button>`);
  }
  if (problem) {
    buttons.push(`<button data-ignore="${idx}">Ignore</button>`);
  }
  return buttons.length
    ? `<div class="card-actions">${buttons.join("")}</div>`
    : "";
}

function issueCardHtml(c: CitationCheck, idx: number): string {
  const sev = severityClass(c.issue_type, c.status);
  const evToggle =
    c.evidence && c.evidence.length > 0
      ? `<div class="evidence-toggle" data-toggle>▸ ${c.evidence.length} evidence quote(s)</div>`
      : "";
  let diff = "";
  if (c.status === "contradicted" && (c.claimed_value || c.actual_value)) {
    diff = `<div class="issue-diff">
      <div class="claimed">claim: ${esc(c.claimed_value ?? "")}</div>
      <div class="actual">paper: ${esc(c.actual_value ?? "")}</div></div>`;
  }
  return `<div class="issue-card ${sev}" data-card>
    <div class="issue-head">
      <span class="issue-status ${sev}">${esc(c.issue_type || c.status)}</span>
      <span class="issue-key">\\cite{${esc(c.cite_key)}}</span>
    </div>
    <div class="issue-claim">${esc(truncate(c.claim_text, 220))}</div>
    <div class="issue-reason">${esc(c.reasoning)}</div>
    ${diff}${evToggle}
    <div class="evidence-collapsed" style="display:none">${evidenceListHtml(c.evidence)}</div>
    ${issueActionsHtml(c, idx)}
  </div>`;
}

function candidateCardHtml(cand: EvidenceCandidate, idx: number): string {
  const author = cand.first_author ? esc(cand.first_author) + " " : "";
  const year = cand.year ? `(${cand.year})` : "";
  const cited = cand.cited_by ? `· cited ${cand.cited_by}` : "";
  const tier = ["meta", "abstract", "RAG"][cand.evidence_tier] ?? "?";
  const spans = cand.spans
    .slice(0, 2)
    .map((s) => {
      const page = s.page != null ? `p.${s.page}` : "—";
      return `<div class="evidence-span ev-${s.type}">
        <span class="ev-meta">${esc(page)}</span>
        <div class="ev-quote">${esc(s.quote ?? "")}</div></div>`;
    })
    .join("");
  const key = cand.cite_key || "untitled";
  return `<div class="cand-card">
    <div class="cand-title">#${idx + 1} ${esc(cand.title)}</div>
    <div class="cand-meta">${author}${year} · ${esc(cand.venue || "—")} ${cited} · [${tier}]</div>
    <div class="cand-key">\\cite{${esc(key)}}</div>
    ${spans}
    <div class="cand-actions">
      <button data-insert="${idx}">Insert \\cite</button>
      <button data-copy-bibtex="${idx}">Copy BibTeX</button>
    </div>
  </div>`;
}

// ── preview modal (F2) ──────────────────────────────────────────────────────

/**
 * Show a before/after diff modal for a patch. Resolves true if the
 * user clicks Accept, false otherwise. Surfaces server-detected
 * conflicts (stale old_text) and disables Accept when conflicted.
 */
function showPreviewModal(args: {
  title: string;
  diff: string;
  conflict?: string;
}): Promise<boolean> {
  return new Promise((resolve) => {
    const overlay = document.createElement("div");
    overlay.className = "modal-overlay";
    const conflictHtml = args.conflict
      ? `<div class="modal-conflict">⚠ ${esc(args.conflict)}</div>`
      : "";
    const diffHtml = args.diff
      .split("\n")
      .map((ln) => {
        const cls = ln.startsWith("+")
          ? "diff-add"
          : ln.startsWith("-")
            ? "diff-del"
            : ln.startsWith("@")
              ? "diff-hunk"
              : "";
        return `<div class="${cls}">${esc(ln)}</div>`;
      })
      .join("");
    overlay.innerHTML = `
      <div class="modal">
        <div class="modal-title">${esc(args.title)}</div>
        ${conflictHtml}
        <div class="modal-diff">${diffHtml}</div>
        <div class="modal-actions">
          <button class="modal-reject">Reject</button>
          <button class="modal-accept primary" ${args.conflict ? "disabled" : ""}>Accept</button>
        </div>
      </div>`;
    document.body.appendChild(overlay);

    const close = (accepted: boolean) => {
      overlay.remove();
      resolve(accepted);
    };
    overlay.querySelector(".modal-accept")?.addEventListener("click", () => close(true));
    overlay.querySelector(".modal-reject")?.addEventListener("click", () => close(false));
    overlay.addEventListener("click", (e) => {
      if (e.target === overlay) close(false);
    });
  });
}

// ── undo banner (F8) ────────────────────────────────────────────────────────

function showUndoBanner(): void {
  let banner = document.getElementById("undo-banner");
  if (!banner) {
    banner = document.createElement("div");
    banner.id = "undo-banner";
    banner.className = "undo-banner";
    statusLine.after(banner);
  }
  banner.innerHTML = `Edit applied. <button id="undo-btn">Undo</button>`;
  banner.style.display = "block";
  document.getElementById("undo-btn")?.addEventListener("click", () => void undoLastEdit());
}

function hideUndoBanner(): void {
  const banner = document.getElementById("undo-banner");
  if (banner) banner.style.display = "none";
}

async function undoLastEdit(): Promise<void> {
  if (!lastEdit) return;
  // Inverse patch: the newText now occupies [start, start+newText.length);
  // replacing that span with the original oldText reverts the edit.
  const invStart = lastEdit.start;
  const invEnd = lastEdit.start + lastEdit.newText.length;
  const res = await applyOverleafEdit(invStart, invEnd, lastEdit.oldText);
  if (res.ok) {
    setStatus("Edit undone.");
    lastEdit = null;
    hideUndoBanner();
  } else {
    setStatus(`Undo failed: ${res.reason}`, true);
  }
}

// ── edit application (F1, F3, F4) ───────────────────────────────────────────

/**
 * Run the full propose→preview→apply cycle for a single text edit.
 *   1. Build a patch and POST /patch/preview (server renders diff +
 *      checks the old_text still matches → conflict detection).
 *   2. Show the diff modal.
 *   3. On Accept, apply the edit into Overleaf via the content script.
 *   4. Stash the inverse for Undo.
 */
async function proposeEdit(args: {
  title: string;
  file: string;
  fileContent: string;
  start: number;
  end: number;
  oldText: string;
  newText: string;
  reason: string;
}): Promise<boolean> {
  const patch: Patch = {
    patch_id: uid(),
    type: "raw",
    file: args.file,
    range: { start: args.start, end: args.end },
    old_text: args.oldText,
    new_text: args.newText,
    reason: args.reason,
    issue_id: "",
    user_approved: false,
  };

  let diff = "";
  let conflict: string | undefined;
  try {
    const pv = await previewPatch([patch], { [args.file]: args.fileContent });
    diff = pv.preview[args.file]?.diff_unified ?? "(no change)";
    if (pv.conflicts.length > 0) {
      conflict =
        "The file changed since this was computed — re-run Check before applying. " +
        pv.conflicts[0].reason;
    }
  } catch (e) {
    setStatus(e instanceof Error ? e.message : String(e), true);
    return false;
  }

  const accepted = await showPreviewModal({ title: args.title, diff, conflict });
  if (!accepted || conflict) return false;

  const res = await applyOverleafEdit(args.start, args.end, args.newText);
  if (!res.ok) {
    if (res.reason === "offscreen") {
      setStatus(
        "That location isn't currently rendered — scroll near it in Overleaf and retry.",
        true,
      );
    } else {
      setStatus(`Could not apply edit: ${res.reason}`, true);
    }
    return false;
  }
  lastEdit = {
    file: args.file,
    start: args.start,
    newText: args.newText,
    oldText: args.oldText,
  };
  showUndoBanner();
  setStatus("Edit applied.");
  return true;
}

/** Span of the `\cite{...}` call starting at `charOffset` in `docText`. */
function citeSpan(docText: string, charOffset: number): { start: number; end: number } | null {
  // Expect `\cite` (or \citep/\citet/...) starting at or near charOffset.
  // Find the opening brace, then the matching close.
  const open = docText.indexOf("{", charOffset);
  if (open < 0) return null;
  const close = docText.indexOf("}", open);
  if (close < 0) return null;
  return { start: charOffset, end: close + 1 };
}

// ── Check flow (E7) ─────────────────────────────────────────────────────────

let currentDoc: { file: string; text: string } | null = null;

async function runCheck(): Promise<void> {
  results.innerHTML = "";
  hideUndoBanner();
  setBusy("Reading Overleaf editor…");

  const doc = await getOverleafDocument();
  if (!doc) {
    setStatus(
      "Couldn't read the Overleaf editor. Open a project and click into the editor.",
      true,
    );
    return;
  }
  currentDoc = doc;
  const projectId = await getOverleafProjectId();

  setBusy(`Auditing ${doc.file} (tier ${settings.tier})…`);
  try {
    const report = await auditSelection({
      file: doc.file,
      texContent: doc.text,
      tier: settings.tier,
      ragTopK: settings.ragTopK,
      embeddingBackend: settings.embeddingBackend,
      projectId,
    });
    renderAudit(report.checks, report.summary);
  } catch (e) {
    setStatus(e instanceof Error ? e.message : String(e), true);
  }
}

/** Active batch-review filter (G4). */
let currentFilter: "all" | "problems" | "verified" = "all";

const STATUS_ORDER = [
  "hallucinated", "contradicted", "missing_in_bib", "unverifiable", "verified",
];

function isProblem(c: CitationCheck): boolean {
  return c.status !== "verified";
}

/**
 * Pre-submission checklist (G7). Computes a ready / not-ready verdict:
 * a project is "ready" only when nothing is hallucinated or contradicted
 * and nothing is still missing from the .bib. Unverifiable is allowed
 * (the user may have judged those manually) but surfaced.
 */
function checklistHtml(checks: CitationCheck[]): string {
  const count = (s: string) => checks.filter((c) => c.status === s).length;
  const verified = count("verified");
  const hallucinated = count("hallucinated");
  const contradicted = count("contradicted");
  const missing = count("missing_in_bib");
  const unverifiable = count("unverifiable");
  const blockers = hallucinated + contradicted + missing;
  const ready = blockers === 0;

  const verdict = ready
    ? `<div class="checklist-verdict ready">✓ Citations look ready${
        unverifiable ? ` — ${unverifiable} unverifiable, review optional` : ""
      }</div>`
    : `<div class="checklist-verdict not-ready">✗ ${blockers} blocking issue(s) — not ready</div>`;

  const row = (label: string, n: number, cls: string) =>
    n > 0
      ? `<span class="cl-stat ${cls}">${n} ${label}</span>`
      : "";

  return `<div class="checklist">
    ${verdict}
    <div class="cl-stats">
      ${row("verified", verified, "cl-green")}
      ${row("hallucinated", hallucinated, "cl-red")}
      ${row("contradicted", contradicted, "cl-orange")}
      ${row("missing in bib", missing, "cl-red")}
      ${row("unverifiable", unverifiable, "cl-yellow")}
    </div>
    <button id="btn-export" class="export-btn">Export report</button>
  </div>`;
}

/** Filter chip bar (G4). */
function filterBarHtml(checks: CitationCheck[]): string {
  const nAll = checks.length;
  const nProb = checks.filter(isProblem).length;
  const nVer = nAll - nProb;
  const chip = (key: string, label: string, n: number) =>
    `<button class="chip ${currentFilter === key ? "chip-active" : ""}" ` +
    `data-filter="${key}">${label} (${n})</button>`;
  return `<div class="filter-bar">
    ${chip("all", "All", nAll)}
    ${chip("problems", "Problems", nProb)}
    ${chip("verified", "Verified", nVer)}
  </div>`;
}

function renderAudit(checks: CitationCheck[], summary: Record<string, number>): void {
  currentChecks = checks;
  currentFilter = "all";
  if (checks.length === 0) {
    setStatus("");
    emptyState("No \\cite{} calls found in this file.");
    return;
  }
  const parts = Object.entries(summary)
    .map(([k, v]) => `${v} ${k}`)
    .join(", ");
  setStatus(`${checks.length} citation(s): ${parts}`);
  results.innerHTML =
    checklistHtml(checks) +
    filterBarHtml(checks) +
    `<div id="issue-cards"></div>`;
  renderCards();
}

/** Render the issue cards for the active filter (G4). */
function renderCards(): void {
  const host = document.getElementById("issue-cards");
  if (!host) return;
  // Keep original indices stable for the action buttons.
  const indexed = currentChecks.map((c, i) => ({ c, i }));
  indexed.sort(
    (a, b) =>
      STATUS_ORDER.indexOf(a.c.status) - STATUS_ORDER.indexOf(b.c.status),
  );
  const visible = indexed.filter(({ c }) => {
    if (currentFilter === "problems") return isProblem(c);
    if (currentFilter === "verified") return c.status === "verified";
    return true;
  });
  host.innerHTML = visible.length
    ? visible.map(({ c, i }) => issueCardHtml(c, i)).join("")
    : `<div class="empty">No citations in this filter.</div>`;
}

// ── report export (G5) ──────────────────────────────────────────────────────

/** Build a self-contained HTML report from the current audit. */
function buildHtmlReport(checks: CitationCheck[], file: string): string {
  const ts = new Date().toISOString();
  const rows = checks
    .map((c) => {
      const colour =
        c.status === "verified"
          ? "#16a34a"
          : c.status === "contradicted"
            ? "#ea580c"
            : c.status === "unverifiable"
              ? "#ca8a04"
              : "#dc2626";
      return `<tr>
        <td><code>\\cite{${escHtml(c.cite_key)}}</code></td>
        <td style="color:${colour};font-weight:600">${escHtml(c.status)}</td>
        <td>${escHtml(c.issue_type || "")}</td>
        <td>${escHtml(c.claim_text)}</td>
        <td>${escHtml(c.reasoning)}</td>
      </tr>`;
    })
    .join("\n");
  return `<!doctype html>
<html><head><meta charset="utf-8"><title>BibSync audit — ${escHtml(file)}</title>
<style>
  body{font-family:-apple-system,Segoe UI,sans-serif;margin:32px;color:#1c1c1e}
  h1{font-size:18px} .meta{color:#6b6b70;font-size:13px;margin-bottom:18px}
  table{border-collapse:collapse;width:100%;font-size:13px}
  th,td{border:1px solid #e3e3e6;padding:6px 8px;text-align:left;vertical-align:top}
  th{background:#f7f7f8} code{font-family:ui-monospace,Menlo,monospace}
</style></head><body>
<h1>BibSync citation audit</h1>
<div class="meta">File: ${escHtml(file)} · ${checks.length} citation(s) · generated ${ts}</div>
<table><thead><tr>
  <th>Citation</th><th>Status</th><th>Issue type</th><th>Claim</th><th>Reasoning</th>
</tr></thead><tbody>
${rows}
</tbody></table>
</body></html>`;
}

function escHtml(s: string): string {
  return (s ?? "").replace(/[&<>"]/g, (ch) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[ch] ?? ch,
  );
}

/** Trigger a browser download of `content` as `filename`. */
function downloadFile(filename: string, content: string, mime: string): void {
  const blob = new Blob([content], { type: mime });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

function exportReport(): void {
  if (currentChecks.length === 0) return;
  const file = currentDoc?.file ?? "audit";
  const stamp = new Date().toISOString().slice(0, 10);
  // Offer both: HTML for humans, JSON for tools. Default to HTML.
  const html = buildHtmlReport(currentChecks, file);
  downloadFile(`bibsync-audit-${stamp}.html`, html, "text/html");
  const json = JSON.stringify(
    { file, generated: new Date().toISOString(), checks: currentChecks },
    null,
    2,
  );
  downloadFile(`bibsync-audit-${stamp}.json`, json, "application/json");
  setStatus("Report exported (HTML + JSON).");
}

// ── Find-citation flow (E10) ────────────────────────────────────────────────

async function runFindCitation(): Promise<void> {
  results.innerHTML = "";
  hideUndoBanner();
  setBusy("Reading selection…");

  const sel = await getOverleafSelection();
  if (!sel || !sel.text.trim()) {
    setStatus("Select a sentence in Overleaf first, then click again.", true);
    return;
  }
  lastSelection = { start: sel.start, end: sel.end, text: sel.text };

  setBusy(`Searching for evidence: "${truncate(sel.text, 60)}"…`);
  try {
    const report = await findEvidence({
      claim: sel.text,
      topPapers: 5,
      tier: Math.min(settings.tier, 1) as 0 | 1,
    });
    currentCandidates = report.candidates;
    if (report.candidates.length === 0) {
      setStatus("No candidate papers found for this claim.");
      emptyState("Try selecting a more specific claim.");
      return;
    }
    setStatus(`${report.candidates.length} candidate paper(s)`);
    results.innerHTML = report.candidates.map(candidateCardHtml).join("");
  } catch (e) {
    setStatus(e instanceof Error ? e.message : String(e), true);
  }
}

// ── Memory tab (E13) ─────────────────────────────────────────────────────────

const memoryList = $("memory-list");
const memoryCount = $("memory-count");

function memoryRecordHtml(r: MemoryRecord): string {
  const claim = r.claim_text ? truncate(r.claim_text, 90) : "";
  const decision = r.decision ? ` · ${esc(r.decision)}` : "";
  return `<div class="mem-record" data-rec="${esc(r.id)}" data-scope="${esc(r.scope)}">
    <div class="mem-head">
      <span class="mem-type t-${esc(r.type)}">${esc(r.type)}${decision}</span>
      <button class="mem-forget" data-forget="${esc(r.id)}" data-scope="${esc(r.scope)}">forget</button>
    </div>
    ${claim ? `<div class="mem-claim">${esc(claim)}</div>` : ""}
    <div class="mem-ts">${esc(r.scope)} · tier ${r.tier} · ${esc(r.ts)}</div>
  </div>`;
}

async function loadMemory(): Promise<void> {
  memoryList.innerHTML = `<div class="empty"><span class="spinner"></span> loading…</div>`;
  try {
    const projectId = await getOverleafProjectId();
    const { records, total } = await getMemory(projectId);
    if (records.length === 0) {
      memoryCount.textContent = "";
      memoryList.innerHTML = `<div class="empty">No memory records yet.<br/>Run Check on this project to build memory.</div>`;
      return;
    }
    memoryCount.textContent = `${total} record(s)`;
    memoryList.innerHTML = records.map(memoryRecordHtml).join("");
  } catch (e) {
    memoryCount.textContent = "";
    memoryList.innerHTML = `<div class="empty">${esc(
      e instanceof Error ? e.message : String(e),
    )}</div>`;
  }
}

// ── card action handlers (F3-F7) ────────────────────────────────────────────

async function handleRemove(idx: number): Promise<void> {
  const c = currentChecks[idx];
  if (!c || !currentDoc) return;
  const span = citeSpan(currentDoc.text, c.char_offset);
  if (!span) {
    setStatus(`Couldn't locate \\cite{${c.cite_key}} in the document.`, true);
    return;
  }
  const oldText = currentDoc.text.slice(span.start, span.end);
  // Strike the cite, leave a trail in a LaTeX comment the user will see.
  const newText = "";
  const ok = await proposeEdit({
    title: `Remove \\cite{${c.cite_key}}`,
    file: currentDoc.file,
    fileContent: currentDoc.text,
    start: span.start,
    end: span.end,
    oldText,
    newText,
    reason: c.reasoning,
  });
  if (ok) {
    // Persist the rejection so a future Check doesn't re-flag the same pair.
    const projectId = await getOverleafProjectId();
    await rememberMemory({
      projectId,
      type: "reject",
      claimText: c.claim_text,
      paperKey: c.paper_arxiv_id || c.paper_doi || `cite-${c.cite_key}`,
      citeKey: c.cite_key,
      decision: "user_removed",
      rationale: c.reasoning,
    });
  }
}

async function handleIgnore(idx: number): Promise<void> {
  const c = currentChecks[idx];
  if (!c) return;
  const projectId = await getOverleafProjectId();
  try {
    await rememberMemory({
      projectId,
      type: "override",
      claimText: c.claim_text,
      paperKey: c.paper_arxiv_id || c.paper_doi || `cite-${c.cite_key}`,
      citeKey: c.cite_key,
      decision: "user_ignored",
      rationale: `ignored ${c.issue_type}`,
    });
    setStatus(`Ignored — \\cite{${c.cite_key}} won't be re-flagged.`);
  } catch (e) {
    setStatus(e instanceof Error ? e.message : String(e), true);
  }
}

async function handleMarkVerified(idx: number): Promise<void> {
  const c = currentChecks[idx];
  if (!c) return;
  const projectId = await getOverleafProjectId();
  try {
    await rememberMemory({
      projectId,
      type: "accept",
      claimText: c.claim_text,
      paperKey: c.paper_arxiv_id || c.paper_doi || `cite-${c.cite_key}`,
      citeKey: c.cite_key,
      decision: "verified",
      rationale: "user marked verified",
    });
    setStatus(`Marked \\cite{${c.cite_key}} as verified.`);
  } catch (e) {
    setStatus(e instanceof Error ? e.message : String(e), true);
  }
}

/**
 * Compute where a `\cite{}` should be inserted relative to a selection.
 *
 * Rules (so the cite lands inline, not on a broken line):
 *   1. Walk the offset BACK over trailing whitespace + newlines — the
 *      cite must attach to the last visible character of the clause,
 *      never jump to the next line.
 *   2. Walk back over ONE trailing sentence punctuation mark (. , ; :)
 *      so the result is the LaTeX-conventional "text~\cite{k}." and
 *      not "text.~\cite{k}".
 *   3. Pick the separator: a "~" (non-breaking space) unless the
 *      preceding char is already whitespace / ~.
 *   4. Flag if a \cite is already adjacent — so we don't double-cite.
 */
function citeInsertionPoint(
  text: string,
  selStart: number,
  selEnd: number,
): { offset: number; separator: string; alreadyCited: boolean } {
  let p = Math.min(selEnd, text.length);
  while (p > selStart && /\s/.test(text[p - 1])) p--;
  if (p > selStart && /[.,;:]/.test(text[p - 1])) p--;
  const around = text.slice(Math.max(0, p - 16), p + 16);
  const alreadyCited = /\\(?:no)?cite\w*\s*\{/.test(around);
  const prev = p > 0 ? text[p - 1] : "";
  const separator = prev === "" || /\s/.test(prev) || prev === "~" ? "" : "~";
  return { offset: p, separator, alreadyCited };
}

/**
 * Detect the project's bibliography file from a .tex's content.
 *   \addbibresource{references.bib}  → biblatex
 *   \bibliography{references}        → bibtex / natbib
 */
function findBibResource(
  text: string,
): { name: string; command: string } | null {
  let m = text.match(/\\addbibresource\s*\{([^}]+)\}/);
  if (m) return { name: m[1].trim(), command: "addbibresource" };
  m = text.match(/\\bibliography\s*\{([^}]+)\}/);
  if (m) {
    const raw = m[1].trim();
    return { name: raw.endsWith(".bib") ? raw : raw + ".bib", command: "bibliography" };
  }
  return null;
}

async function handleInsert(idx: number): Promise<void> {
  const cand = currentCandidates[idx];
  if (!cand || !lastSelection) {
    setStatus("Re-run Find citation, then Insert.", true);
    return;
  }
  // Re-read the document so offsets are current.
  const doc = await getOverleafDocument();
  if (!doc) {
    setStatus("Couldn't read the editor.", true);
    return;
  }
  // Re-locate the selected text in the live document — more reliable
  // than trusting the offset captured at selection time.
  const selStart = doc.text.indexOf(lastSelection.text);
  if (selStart < 0) {
    setStatus(
      "The selected text moved since Find citation — re-select it and try again.",
      true,
    );
    return;
  }
  const selEnd = selStart + lastSelection.text.length;

  const { offset, separator, alreadyCited } = citeInsertionPoint(
    doc.text,
    selStart,
    selEnd,
  );
  if (alreadyCited) {
    setStatus("There's already a \\cite here — skipped to avoid a duplicate.", true);
    return;
  }

  // Proper cite key (firstauthor+year+titleword) — never the cache id.
  const citeKey = cand.cite_key || "untitled";
  const ok = await proposeEdit({
    title: `Insert \\cite{${citeKey}}`,
    file: doc.file,
    fileContent: doc.text,
    start: offset,
    end: offset,
    oldText: "",
    newText: `${separator}\\cite{${citeKey}}`,
    reason: `cite ${cand.title}`,
  });
  if (!ok) return;

  // The \cite is in the .tex — the .bib entry must exist too. Detect the
  // project's bib file so we can tell the user exactly where it goes.
  if (cand.bibtex) await navigator.clipboard.writeText(cand.bibtex);
  const bib = findBibResource(doc.text);
  if (bib) {
    setStatus(
      `Inserted \\cite{${citeKey}}. BibTeX entry copied — paste it into ${bib.name}.`,
    );
  } else {
    setStatus(
      `Inserted \\cite{${citeKey}}. BibTeX entry copied. ⚠ This file has no ` +
        "\\addbibresource / \\bibliography — add one (and a references.bib) " +
        "so the citation resolves.",
    );
  }
}

// ── event wiring ─────────────────────────────────────────────────────────────

$$(".tab").forEach((b) =>
  b.addEventListener("click", () => switchTab(b.dataset.tab ?? "check")),
);
btnCheck.addEventListener("click", () => void runCheck());
btnEvidence.addEventListener("click", () => void runFindCitation());
$("btn-memory-refresh").addEventListener("click", () => void loadMemory());
for (const id of ["set-tier", "set-backend", "set-topk", "set-token"]) {
  $(id).addEventListener("change", () => void saveSettings());
}

results.addEventListener("click", (e) => {
  const t = e.target as HTMLElement;
  // G4 — filter chips.
  const filter = t.getAttribute("data-filter");
  if (filter) {
    currentFilter = filter as typeof currentFilter;
    document
      .querySelectorAll(".chip")
      .forEach((c) =>
        c.classList.toggle(
          "chip-active",
          c.getAttribute("data-filter") === filter,
        ),
      );
    renderCards();
    return;
  }
  // G5 — export button.
  if (t.id === "btn-export") {
    exportReport();
    return;
  }
  if (t.matches("[data-toggle]")) {
    const card = t.closest("[data-card]");
    const panel = card?.querySelector<HTMLElement>(".evidence-collapsed");
    if (panel) {
      const open = panel.style.display !== "none";
      panel.style.display = open ? "none" : "block";
      t.textContent = (open ? "▸ " : "▾ ") + (t.textContent ?? "").slice(2);
    }
    return;
  }
  const copyBib = t.getAttribute("data-copy-bibtex");
  if (copyBib) {
    const cand = currentCandidates[Number(copyBib)];
    if (cand?.bibtex) {
      void navigator.clipboard.writeText(cand.bibtex);
      t.textContent = "Copied!";
      setTimeout(() => (t.textContent = "Copy BibTeX"), 1500);
    }
    return;
  }
  const remove = t.getAttribute("data-remove");
  if (remove) { void handleRemove(Number(remove)); return; }
  const ignore = t.getAttribute("data-ignore");
  if (ignore) { void handleIgnore(Number(ignore)); return; }
  const markV = t.getAttribute("data-mark-verified");
  if (markV) { void handleMarkVerified(Number(markV)); return; }
  const insert = t.getAttribute("data-insert");
  if (insert) { void handleInsert(Number(insert)); return; }
});

memoryList.addEventListener("click", (e) => {
  const t = e.target as HTMLElement;
  const recId = t.getAttribute("data-forget");
  if (!recId) return;
  const scope = (t.getAttribute("data-scope") ?? "project") as "project" | "user";
  t.textContent = "…";
  void (async () => {
    try {
      const projectId = await getOverleafProjectId();
      await forgetMemory(recId, scope, projectId);
      t.closest("[data-rec]")?.remove();
    } catch {
      t.textContent = "failed";
    }
  })();
});

// ── init ─────────────────────────────────────────────────────────────────────

void loadSettings();
emptyState("Select text in Overleaf, then click an action above.");
void pollHealth();
setInterval(() => void pollHealth(), 30_000);
