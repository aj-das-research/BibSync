/**
 * BibSync side panel — main UI controller.
 *
 * Vanilla TypeScript + DOM templating. Three tabs:
 *   • Check    — audit the current file, find citations for a selection
 *   • Memory   — inspect / forget the project's stored decisions
 *   • Settings — tier / embedding backend / RAG top-k (chrome.storage.local)
 *
 * Pipeline (every server call):
 *   side panel → service worker → native host → bibsync serve
 */
import {
  auditSelection,
  checkHealth,
  findEvidence,
  forgetMemory,
  getMemory,
  getOverleafDocument,
  getOverleafProjectId,
  getOverleafSelection,
} from "./api";
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
  // Reflect into the form controls.
  ($("set-tier") as HTMLSelectElement).value = String(settings.tier);
  ($("set-backend") as HTMLSelectElement).value = settings.embeddingBackend;
  ($("set-topk") as HTMLInputElement).value = String(settings.ragTopK);
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

function issueCardHtml(c: CitationCheck): string {
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
  return `<div class="cand-card">
    <div class="cand-title">#${idx + 1} ${esc(cand.title)}</div>
    <div class="cand-meta">${author}${year} · ${esc(cand.venue || "—")} ${cited} · [${tier}]</div>
    ${spans}
    <div class="cand-actions">
      <button data-copy-cite="${esc(cand.paper_key)}">Copy \\cite key</button>
    </div>
  </div>`;
}

// ── Check flow (E7) ─────────────────────────────────────────────────────────

async function runCheck(): Promise<void> {
  results.innerHTML = "";
  setBusy("Reading Overleaf editor…");

  const doc = await getOverleafDocument();
  if (!doc) {
    setStatus(
      "Couldn't read the Overleaf editor. Open a project and click into the editor.",
      true,
    );
    return;
  }
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

function renderAudit(checks: CitationCheck[], summary: Record<string, number>): void {
  if (checks.length === 0) {
    setStatus("");
    emptyState("No \\cite{} calls found in this file.");
    return;
  }
  const parts = Object.entries(summary)
    .map(([k, v]) => `${v} ${k}`)
    .join(", ");
  setStatus(`${checks.length} citation(s): ${parts}`);
  const order = ["hallucinated", "contradicted", "missing_in_bib", "unverifiable", "verified"];
  const sorted = [...checks].sort(
    (a, b) => order.indexOf(a.status) - order.indexOf(b.status),
  );
  results.innerHTML = sorted.map(issueCardHtml).join("");
}

// ── Find-citation flow (E10) ────────────────────────────────────────────────

async function runFindCitation(): Promise<void> {
  results.innerHTML = "";
  setBusy("Reading selection…");

  const sel = await getOverleafSelection();
  if (!sel || !sel.text.trim()) {
    setStatus("Select a sentence in Overleaf first, then click again.", true);
    return;
  }

  setBusy(`Searching for evidence: "${truncate(sel.text, 60)}"…`);
  try {
    const report = await findEvidence({
      claim: sel.text,
      topPapers: 5,
      tier: Math.min(settings.tier, 1) as 0 | 1,
    });
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

// ── event wiring ─────────────────────────────────────────────────────────────

$$(".tab").forEach((b) =>
  b.addEventListener("click", () => switchTab(b.dataset.tab ?? "check")),
);

btnCheck.addEventListener("click", () => void runCheck());
btnEvidence.addEventListener("click", () => void runFindCitation());
$("btn-memory-refresh").addEventListener("click", () => void loadMemory());

for (const id of ["set-tier", "set-backend", "set-topk"]) {
  $(id).addEventListener("change", () => void saveSettings());
}

// Delegated clicks in the Check results pane.
results.addEventListener("click", (e) => {
  const target = e.target as HTMLElement;
  if (target.matches("[data-toggle]")) {
    const card = target.closest("[data-card]");
    const panel = card?.querySelector<HTMLElement>(".evidence-collapsed");
    if (panel) {
      const open = panel.style.display !== "none";
      panel.style.display = open ? "none" : "block";
      target.textContent =
        (open ? "▸ " : "▾ ") + (target.textContent ?? "").slice(2);
    }
  }
  const copyKey = target.getAttribute("data-copy-cite");
  if (copyKey) {
    void navigator.clipboard.writeText(`\\cite{${copyKey}}`);
    target.textContent = "Copied!";
    setTimeout(() => (target.textContent = "Copy \\cite key"), 1500);
  }
});

// Delegated clicks in the Memory pane (forget buttons).
memoryList.addEventListener("click", (e) => {
  const target = e.target as HTMLElement;
  const recId = target.getAttribute("data-forget");
  if (!recId) return;
  const scope = (target.getAttribute("data-scope") ?? "project") as
    | "project"
    | "user";
  target.textContent = "…";
  void (async () => {
    try {
      const projectId = await getOverleafProjectId();
      await forgetMemory(recId, scope, projectId);
      target.closest("[data-rec]")?.remove();
    } catch {
      target.textContent = "failed";
    }
  })();
});

// ── init ─────────────────────────────────────────────────────────────────────

void loadSettings();
emptyState("Select text in Overleaf, then click an action above.");
void pollHealth();
setInterval(() => void pollHealth(), 30_000);
