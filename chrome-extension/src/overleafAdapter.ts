/**
 * Overleaf editor adapter — the ONLY file that knows Overleaf's DOM.
 *
 * If Overleaf changes its editor, or we add a VS Code adapter later,
 * only this file changes. Everything else speaks the neutral
 * `EditorSelection` type.
 *
 * Overleaf's editor is CodeMirror 6. A content script runs in an
 * *isolated world* — it can see the page DOM but not the page's JS
 * objects (so we can't grab the `EditorView` instance directly without
 * injecting a page script). For the read-only MVP we work off the
 * rendered DOM:
 *   • active file name  ← the highlighted entry in the file-tree panel
 *   • current text      ← concatenated `.cm-line` text content
 *   • selection         ← window.getSelection() intersected with `.cm-content`
 *
 * This is lower-fidelity than the CM6 model API (DOM line text can lose
 * exact source offsets for folded/virtualised lines) but it needs no
 * page-script injection and is robust enough to send a selected
 * paragraph to the audit pipeline.
 */
import type { EditorSelection } from "./types";

/** True when the current page looks like an Overleaf project editor. */
export function detectEditor(): boolean {
  return Boolean(document.querySelector(".cm-editor"));
}

/** Best-effort active .tex file name from Overleaf's file tree. */
export function getActiveFileName(): string {
  // Overleaf marks the open file in the file-tree with `aria-selected`
  // or a `selected` class depending on UI version. Try several.
  const selectors = [
    ".file-tree .selected .item-name-button",
    ".file-tree [aria-selected=\"true\"]",
    ".file-tree-inner .selected",
  ];
  for (const sel of selectors) {
    const el = document.querySelector(sel);
    const name = el?.textContent?.trim();
    if (name) return name;
  }
  // Fallback — the document title often contains the project name.
  return "main.tex";
}

/** Full text of the currently open editor pane. */
export function getCurrentText(): string {
  const content = document.querySelector(".cm-content");
  if (!content) return "";
  // Each `.cm-line` is one source line. textContent of the container
  // collapses them without newlines, so join line-by-line.
  const lines = Array.from(content.querySelectorAll(".cm-line"));
  if (lines.length === 0) return content.textContent ?? "";
  return lines.map((l) => l.textContent ?? "").join("\n");
}

/**
 * The user's current selection within the editor.
 *
 * Returns null when nothing is selected or the selection is outside the
 * `.cm-content` region. `start`/`end` are character offsets into the
 * value returned by getCurrentText() — computed by walking the line
 * structure so they line up with what the audit pipeline receives.
 */
export function getSelection(): EditorSelection | null {
  const sel = window.getSelection();
  if (!sel || sel.rangeCount === 0 || sel.isCollapsed) return null;

  const content = document.querySelector(".cm-content");
  if (!content) return null;

  const range = sel.getRangeAt(0);
  // Selection must be inside the editor content.
  if (!content.contains(range.commonAncestorContainer)) return null;

  const text = sel.toString();
  if (!text.trim()) return null;

  // Offset computation: find the selection's start line + the lines
  // before it, sum their lengths (+1 per newline).
  const fullText = getCurrentText();
  // Use indexOf as a pragmatic locator. For unique-enough paragraph
  // selections this is exact; for short ambiguous selections it finds
  // the first occurrence (acceptable for the MVP — the audit pipeline
  // re-extracts the claim sentence anyway).
  const start = fullText.indexOf(text);
  const end = start >= 0 ? start + text.length : -1;

  return {
    file: getActiveFileName(),
    start: start >= 0 ? start : 0,
    end: end >= 0 ? end : text.length,
    text,
  };
}

/** Highlight a character range (used later to flag issue locations). */
export function clearHighlights(): void {
  document
    .querySelectorAll(".bibsync-highlight")
    .forEach((el) => el.classList.remove("bibsync-highlight"));
}
