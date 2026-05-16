/**
 * Side-panel API helper.
 *
 * The side panel can't open native-messaging ports itself (only the
 * service worker can). So every server call is wrapped as a
 * `chrome.runtime.sendMessage` to the worker, which forwards to the
 * native host, which forwards to `bibsync serve`.
 *
 *   side panel → worker → native host → bibsync serve
 *
 * This module hides that chain behind plain async functions.
 */
import type {
  AuditReport,
  EditorSelection,
  EvidenceReport,
  HealthInfo,
  NativeResponse,
} from "../types";

/** Forward a native-host request through the service worker. */
async function viaWorker(request: {
  method: "GET" | "POST" | "DELETE";
  path: string;
  body?: unknown;
  query?: Record<string, string>;
}): Promise<NativeResponse> {
  const resp = (await chrome.runtime.sendMessage({
    kind: "native",
    request,
  })) as NativeResponse;
  if (!resp) {
    throw new Error("no response from service worker");
  }
  return resp;
}

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
}): Promise<AuditReport> {
  const resp = await viaWorker({
    method: "POST",
    path: "/audit",
    body: {
      project_id: "overleaf-extension",
      tex_files: { [args.file]: args.texContent },
      bib_files: args.bibContent ? { "references.bib": args.bibContent } : {},
      tier: args.tier,
      rag_top_k: args.ragTopK,
      use_memory: true,
    },
  });
  if (!resp.ok) {
    throw new Error(resp.error ?? `audit failed (status ${resp.status})`);
  }
  return resp.body as AuditReport;
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
