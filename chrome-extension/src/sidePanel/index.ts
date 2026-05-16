/**
 * BibSync side panel — main UI controller.
 *
 * Vanilla TypeScript + DOM templating (no framework — the panel has
 * three views and ~5KB of logic; React would be more weight than win).
 *
 * Flows:
 *   • Connection status (E14) — polls /health every 30s.
 *   • Check selected text (E7) — reads the Overleaf selection, sends
 *     the current file to /audit, renders issue cards (E8) with
 *     expandable evidence (E9).
 *   • Find citation (E10) — sends the selection text to /evidence,
 *     renders candidate papers.
 */
import {
  auditSelection,
  checkHealth,
  findEvidence,
  getOverleafDocument,
  getOverleafSelection,
} from "./api";
import type {
  CitationCheck,
  ConnectionState,
  EvidenceCandidate,
  EvidenceSpan,
} from "../types";

// ── DOM handles ─────────────────────────────────────────────────────────────
const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;

const connEl = $("conn");
const connLabel = $("conn-label");
const btnCheck = $<HTMLButtonElement>("btn-check");
const btnEvidence = $<HTMLButtonElement>("btn-evidence");
const statusLine = $("status-line");
const results = $("results");

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
    // "could not reach" → server down; anything else → host/extension issue.
    setConnection(
      msg.includes("could not reach") || msg.includes("disconnected")
        ? "disconnected"
        : "error",
      msg,
    );
  }
}

// ── severity mapping (issue_type → colour) ──────────────────────────────────

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

// ── HTML escaping ───────────────────────────────────────────────────────────

function esc(s: string): string {
  const d = document.createElement("div");
  d.textContent = s ?? "";
  return d.innerHTML;
}

// ── evidence rendering (E9) ─────────────────────────────────────────────────

function evidenceListHtml(spans: EvidenceSpan[] | undefined): string {
  if (!spans || spans.length === 0) return "";
  const items = spans
    .map((s) => {
      const page = s.page != null ? `p.${s.page}` : "—";
      const cls = `evidence-span ev-${s.type}`;
      return `
        <div class="${cls}">
          <span class="ev-meta">[${esc(s.type)}] ${esc(page)}</span>
          <div class="ev-quote">${esc(s.quote ?? "")}</div>
        </div>`;
    })
    .join("");
  return `<div class="evidence-list">${items}</div>`;
}

// ── issue card (E8) ─────────────────────────────────────────────────────────

function issueCardHtml(c: CitationCheck): string {
  const sev = severityClass(c.issue_type, c.status);
  const evidence = evidenceListHtml(c.evidence);
  const evToggle =
    c.evidence && c.evidence.length > 0
      ? `<div class="evidence-toggle" data-toggle>▸ ${c.evidence.length} evidence quote(s)</div>`
      : "";

  let diff = "";
  if (c.status === "contradicted" && (c.claimed_value || c.actual_value)) {
    diff = `
      <div class="issue-diff">
        <div class="claimed">claim: ${esc(c.claimed_value ?? "")}</div>
        <div class="actual">paper: ${esc(c.actual_value ?? "")}</div>
      </div>`;
  }

  return `
    <div class="issue-card ${sev}" data-card>
      <div class="issue-head">
        <span class="issue-status ${sev}">${esc(c.issue_type || c.status)}</span>
        <span class="issue-key">\\cite{${esc(c.cite_key)}}</span>
      </div>
      <div class="issue-claim">${esc(truncate(c.claim_text, 220))}</div>
      <div class="issue-reason">${esc(c.reasoning)}</div>
      ${diff}
      ${evToggle}
      <div class="evidence-collapsed" style="display:none">${evidence}</div>
    </div>`;
}

function truncate(s: string, n: number): string {
  return s.length > n ? s.slice(0, n) + "…" : s;
}

// ── candidate card (E10 — evidence command) ─────────────────────────────────

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
  return `
    <div class="cand-card">
      <div class="cand-title">#${idx + 1} ${esc(cand.title)}</div>
      <div class="cand-meta">${author}${year} · ${esc(cand.venue || "—")} ${cited} · [${tier}]</div>
      ${spans}
      <div class="cand-actions">
        <button data-copy-cite="${esc(cand.paper_key)}">Copy \\cite key</button>
      </div>
    </div>`;
}

// ── status line helpers ─────────────────────────────────────────────────────

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

// ── Check flow (E7) ─────────────────────────────────────────────────────────

async function runCheck(): Promise<void> {
  results.innerHTML = "";
  setBusy("Reading Overleaf selection…");

  const doc = await getOverleafDocument();
  if (!doc) {
    setStatus(
      "Couldn't read the Overleaf editor. Open a project and click into the editor.",
      true,
    );
    return;
  }

  setBusy(`Auditing ${doc.file} — this can take 30-60s on first run…`);
  try {
    const report = await auditSelection({
      file: doc.file,
      texContent: doc.text,
      tier: 2,
      ragTopK: 5,
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
  // Sort: problems first (red, orange, yellow), verified last.
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
      tier: 1,
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

// ── event wiring ────────────────────────────────────────────────────────────

btnCheck.addEventListener("click", () => void runCheck());
btnEvidence.addEventListener("click", () => void runFindCitation());

// Evidence toggle (delegated — cards are re-rendered on each run).
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
    setTimeout(() => {
      target.textContent = "Copy \\cite key";
    }, 1500);
  }
});

// ── init ────────────────────────────────────────────────────────────────────

emptyState("Select text in Overleaf, then click an action above.");
void pollHealth();
setInterval(() => void pollHealth(), 30_000);
