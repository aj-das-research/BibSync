/**
 * Shared types across the extension contexts.
 *
 * These mirror the JSON shapes produced by `bibsync serve`. They are
 * intentionally hand-written (not generated from OpenAPI) so the extension
 * has no build-time dependency on a running server — the shapes are stable
 * because they're the same dataclass `.to_dict()` outputs the CLI uses.
 */

// ── audit result shapes (mirror bibsync/audit.py AuditReport.to_dict) ───────

export interface EvidenceSpan {
  type: "supporting" | "contradicting" | "missing" | "neutral";
  paper_key?: string;
  paper_title?: string;
  section?: string;
  page?: number | null;
  chunk_idx?: number | null;
  chunk_score?: number;
  quote?: string;
}

export interface CitationCheck {
  cite_key: string;
  file: string;
  line: number;
  char_offset: number;
  claim_text: string;
  status: "verified" | "hallucinated" | "contradicted" | "unverifiable" | "missing_in_bib";
  issue_type: string;
  confidence: number;
  reasoning: string;
  evidence_tier: number;
  n_chunks: number;
  degraded_reason: string;
  fixed: boolean;
  paper_source?: string | null;
  paper_doi?: string | null;
  paper_arxiv_id?: string | null;
  contradiction_type?: string;
  claimed_value?: string;
  actual_value?: string;
  evidence?: EvidenceSpan[];
}

export interface AuditReport {
  project_root: string;
  bib_file: string;
  tex_files_scanned: number;
  summary: Record<string, number>;
  checks: CitationCheck[];
}

// ── evidence command shapes (mirror bibsync/evidence_cmd.py) ────────────────

export interface EvidenceCandidate {
  paper_key: string;
  title: string;
  first_author: string;
  authors: string[];
  year: number | null;
  venue: string;
  doi: string;
  arxiv_id: string;
  cited_by: number;
  evidence_tier: number;
  has_abstract: boolean;
  has_pdf: boolean;
  spans: EvidenceSpan[];
  note: string;
  /** Proper BibTeX cite key (e.g. "das2024confidence"). */
  cite_key: string;
  /** Ready-to-paste @type{...} BibTeX entry. */
  bibtex: string;
}

export interface EvidenceReport {
  claim: string;
  candidates: EvidenceCandidate[];
  elapsed_sec: number;
}

// ── memory record (mirrors bibsync/memory.py MemoryRecord) ─────────────────

export interface MemoryRecord {
  id: string;
  type: "accept" | "reject" | "verdict" | "preference" | "override" | "forgotten";
  scope: "project" | "user";
  ts: string;
  claim_text: string;
  claim_hash: string;
  paper_key: string;
  cite_key: string;
  decision: string;
  tier: number;
  confidence: number;
  source: string;
  rationale: string;
  tags: string[];
  forgets: string;
}

// ── connection status ──────────────────────────────────────────────────────

export type ConnectionState = "connecting" | "connected" | "disconnected" | "error";

export interface HealthInfo {
  ok: boolean;
  version: string;
  ts: string;
}

// ── editor adapter ──────────────────────────────────────────────────────────

export interface EditorSelection {
  file: string;
  start: number;
  end: number;
  text: string;
}

// ── settings (persisted in chrome.storage.local) ───────────────────────────

export interface Settings {
  tier: 0 | 1 | 2;
  embeddingBackend: "auto" | "local" | "api";
  ragTopK: number;
  /** Bearer token — only needed when `bibsync serve --token` was used.
   *  Empty for the default (no-auth, localhost-only) server. */
  serverToken: string;
}

export const DEFAULT_SETTINGS: Settings = {
  tier: 2,
  embeddingBackend: "auto",
  ragTopK: 5,
  serverToken: "",
};
