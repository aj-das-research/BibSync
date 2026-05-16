/**
 * Side-panel API helper.
 *
 * The side panel is a chrome-extension:// page. With `host_permissions`
 * declared for http://127.0.0.1:38476/* in the manifest, it can
 * `fetch()` the local `bibsync serve` directly — no native messaging,
 * no service-worker hop, no CORS problem.
 *
 *   side panel  ──fetch──▶  bibsync serve
 *
 * The optional bearer token (only when the server was started with
 * `--token`) is read from chrome.storage.local, where the Settings tab
 * stores it.
 */
import type {
  AuditReport,
  EditorSelection,
  EvidenceReport,
  HealthInfo,
  MemoryRecord,
} from "../types";

/** Base URL of the local server — matches `bibsync serve`'s default. */
const SERVER_URL = "http://127.0.0.1:38476";

/** Normalised response shape (kept stable for all callers). */
interface ServerResponse {
  ok: boolean;
  status: number;
  body?: unknown;
  error?: string;
}

/** Read the optional bearer token the Settings tab persisted. */
async function getServerToken(): Promise<string> {
  try {
    const stored = await chrome.storage.local.get("bibsync.settings");
    return (stored["bibsync.settings"]?.serverToken as string) ?? "";
  } catch {
    return "";
  }
}

/** Fetch the local server directly. Never throws — failures come back
 *  as { ok:false }. A network failure (server not running) yields an
 *  error containing "could not reach" so the UI can show the right
 *  "BibSync not running" state. */
async function serverFetch(request: {
  method: "GET" | "POST" | "DELETE";
  path: string;
  body?: unknown;
  query?: Record<string, string>;
}): Promise<ServerResponse> {
  const token = await getServerToken();
  const headers: Record<string, string> = { "Content-Type": "application/json" };
  if (token) headers["Authorization"] = `Bearer ${token}`;

  let url = SERVER_URL + request.path;
  if (request.query && Object.keys(request.query).length > 0) {
    url += "?" + new URLSearchParams(request.query).toString();
  }

  try {
    const resp = await fetch(url, {
      method: request.method,
      headers,
      body: request.body !== undefined ? JSON.stringify(request.body) : undefined,
    });
    const text = await resp.text();
    let parsed: unknown;
    try {
      parsed = text ? JSON.parse(text) : undefined;
    } catch {
      parsed = text;
    }
    if (!resp.ok) {
      const detail =
        (parsed as { detail?: string })?.detail ?? `HTTP ${resp.status}`;
      return { ok: false, status: resp.status, body: parsed, error: detail };
    }
    return { ok: true, status: resp.status, body: parsed };
  } catch (e) {
    // fetch() rejects only on network-level failure — server not running.
    return {
      ok: false,
      status: 0,
      error: `could not reach bibsync serve at ${SERVER_URL} — is it running?`,
    };
  }
}

/** Back-compat alias — existing call sites use viaWorker(). */
const viaWorker = serverFetch;

/** GET /health — also doubles as the connection probe. */
export async function checkHealth(): Promise<HealthInfo> {
  const resp = await viaWorker({ method: "GET", path: "/health" });
  if (!resp.ok) {
    throw new Error(resp.error ?? `health check failed (status ${resp.status})`);
  }
  return resp.body as HealthInfo;
}

/**
 * POST /audit — verify the citations in the supplied tex content.
 *
 * The side panel ships the whole current file as a single-file project;
 * the bib file is optional (Overleaf projects usually have one, but the
 * MVP audits whatever \cite{} keys it finds and reports missing_in_bib
 * for unresolved ones).
 */
export async function auditSelection(args: {
  file: string;
  texContent: string;
  bibContent?: string;
  tier: number;
  ragTopK: number;
  embeddingBackend: string;
  projectId: string;
}): Promise<AuditReport> {
  const resp = await viaWorker({
    method: "POST",
    path: "/audit",
    body: {
      project_id: args.projectId,
      tex_files: { [args.file]: args.texContent },
      bib_files: args.bibContent ? { "references.bib": args.bibContent } : {},
      tier: args.tier,
      rag_top_k: args.ragTopK,
      embedding_backend: args.embeddingBackend,
      use_memory: true,
    },
  });
  if (!resp.ok) {
    throw new Error(resp.error ?? `audit failed (status ${resp.status})`);
  }
  return resp.body as AuditReport;
}

// ── memory ──────────────────────────────────────────────────────────────────

/** GET /memory — list records for the given Overleaf project. */
export async function getMemory(projectId: string): Promise<{
  records: MemoryRecord[];
  total: number;
}> {
  const resp = await viaWorker({
    method: "GET",
    path: "/memory",
    query: { project_id: projectId, scope: "all", limit: "200" },
  });
  if (!resp.ok) {
    throw new Error(resp.error ?? `memory query failed (status ${resp.status})`);
  }
  return resp.body as { records: MemoryRecord[]; total: number };
}

/** POST /memory/forget — tombstone a record by id. */
export async function forgetMemory(
  recordId: string,
  scope: "project" | "user",
  projectId: string,
): Promise<boolean> {
  const resp = await viaWorker({
    method: "POST",
    path: "/memory/forget",
    body: { record_id: recordId, scope, project_id: projectId },
  });
  if (!resp.ok) {
    throw new Error(resp.error ?? `forget failed (status ${resp.status})`);
  }
  return Boolean((resp.body as { ok?: boolean })?.ok);
}

/** POST /memory/remember — write a record (Ignore / Mark-accepted actions). */
export async function rememberMemory(args: {
  projectId: string;
  type: "accept" | "reject" | "verdict" | "preference" | "override";
  claimText?: string;
  paperKey?: string;
  citeKey?: string;
  decision?: string;
  rationale?: string;
}): Promise<boolean> {
  const resp = await viaWorker({
    method: "POST",
    path: "/memory/remember",
    body: {
      project_id: args.projectId,
      type: args.type,
      claim_text: args.claimText ?? "",
      paper_key: args.paperKey ?? "",
      cite_key: args.citeKey ?? "",
      decision: args.decision ?? "",
      rationale: args.rationale ?? "",
      source: "extension",
      scope: "project",
    },
  });
  if (!resp.ok) {
    throw new Error(resp.error ?? `remember failed (status ${resp.status})`);
  }
  return Boolean((resp.body as { ok?: boolean })?.ok);
}

// ── patches (Sprint F) ──────────────────────────────────────────────────────

/** A patch as the server expects it (mirror of bibsync/patches.py Patch). */
export interface Patch {
  patch_id: string;
  type: string;
  file: string;
  range: { start: number; end: number };
  old_text: string;
  new_text: string;
  reason: string;
  issue_id: string;
  user_approved: boolean;
}

/** POST /patch/preview — render a diff without applying. */
export async function previewPatch(
  patches: Patch[],
  files: Record<string, string>,
): Promise<{
  ok: boolean;
  preview: Record<string, { before: string; after: string; diff_unified: string }>;
  conflicts: Array<{ patch_id: string; expected: string; actual: string; reason: string }>;
}> {
  const resp = await viaWorker({
    method: "POST",
    path: "/patch/preview",
    body: { patches, files },
  });
  if (!resp.ok) {
    throw new Error(resp.error ?? `preview failed (status ${resp.status})`);
  }
  return resp.body as never;
}

// ── Overleaf editing (Sprint F) ─────────────────────────────────────────────

/**
 * Apply a text edit into the live Overleaf editor via the content
 * script (which calls overleafAdapter.applyEdit → execCommand on CM6).
 *
 * Returns { ok, reason }. reason="offscreen" means the target range
 * isn't rendered (CM6 virtualised it) — the user must scroll near it.
 */
export async function applyOverleafEdit(
  start: number,
  end: number,
  newText: string,
): Promise<{ ok: boolean; reason: string }> {
  const tab = await activeTab();
  if (!tab?.id) return { ok: false, reason: "no active tab" };
  try {
    const resp = await chrome.tabs.sendMessage(tab.id, {
      kind: "ol-apply",
      start,
      end,
      newText,
    });
    return resp ?? { ok: false, reason: "no response from content script" };
  } catch (e) {
    return { ok: false, reason: e instanceof Error ? e.message : String(e) };
  }
}

/** POST /evidence — find supporting papers for a free-form claim. */
export async function findEvidence(args: {
  claim: string;
  topPapers: number;
  tier: number;
}): Promise<EvidenceReport> {
  const resp = await viaWorker({
    method: "POST",
    path: "/evidence",
    body: {
      claim: args.claim,
      top_papers: args.topPapers,
      tier: args.tier,
    },
  });
  if (!resp.ok) {
    throw new Error(resp.error ?? `evidence lookup failed (status ${resp.status})`);
  }
  return resp.body as EvidenceReport;
}

// ── content-script messaging (read Overleaf editor state) ──────────────────

/** Ask the active Overleaf tab's content script for the current selection. */
export async function getOverleafSelection(): Promise<EditorSelection | null> {
  const tab = await activeTab();
  if (!tab?.id) return null;
  try {
    const resp = await chrome.tabs.sendMessage(tab.id, { kind: "ol-selection" });
    return (resp?.selection as EditorSelection) ?? null;
  } catch {
    return null; // content script not present on this tab
  }
}

/** Ask the active Overleaf tab for its full document text. */
export async function getOverleafDocument(): Promise<{ file: string; text: string } | null> {
  const tab = await activeTab();
  if (!tab?.id) return null;
  try {
    const resp = await chrome.tabs.sendMessage(tab.id, { kind: "ol-document" });
    if (resp?.text) return { file: resp.file, text: resp.text };
    return null;
  } catch {
    return null;
  }
}

/** True when the active tab is an Overleaf editor. */
export async function isOverleafTab(): Promise<boolean> {
  const tab = await activeTab();
  if (!tab?.id) return false;
  try {
    const resp = await chrome.tabs.sendMessage(tab.id, { kind: "ol-detect" });
    return Boolean(resp?.editor);
  } catch {
    return false;
  }
}

async function activeTab(): Promise<chrome.tabs.Tab | undefined> {
  const tabs = await chrome.tabs.query({ active: true, currentWindow: true });
  return tabs[0];
}

/**
 * Stable project identity for memory scoping.
 *
 * Overleaf project URLs are `overleaf.com/project/<24-hex-id>`. That hex
 * ID is a perfect stable key — the same project across sessions yields
 * the same `project_id`, so memory recall works. Falls back to a
 * generic id when the URL isn't an Overleaf project page (e.g. the
 * panel is open on a non-Overleaf tab).
 */
export async function getOverleafProjectId(): Promise<string> {
  const tab = await activeTab();
  const url = tab?.url ?? "";
  const m = url.match(/overleaf\.com\/project\/([a-f0-9]+)/i);
  return m ? `overleaf:${m[1]}` : "overleaf:unknown";
}
