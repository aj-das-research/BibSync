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

// ── editing (Sprint F) ──────────────────────────────────────────────────────

/**
 * Map a character offset (into getCurrentText()) to a DOM position
 * { node, offset } inside the rendered `.cm-line` nodes.
 *
 * Returns null when the offset lands on a line CodeMirror hasn't
 * rendered (CM6 virtualises — off-screen lines have no DOM). The
 * caller surfaces a "scroll to the location and retry" message.
 */
function offsetToDOMPosition(
  charOffset: number,
): { node: Node; offset: number } | null {
  const content = document.querySelector(".cm-content");
  if (!content) return null;
  const lines = Array.from(content.querySelectorAll(".cm-line"));
  if (lines.length === 0) return null;

  let consumed = 0;
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    const lineText = line.textContent ?? "";
    const lineEnd = consumed + lineText.length; // exclusive of the \n
    if (charOffset <= lineEnd) {
      // Target is within this line. Walk its descendant text nodes.
      const within = charOffset - consumed;
      let acc = 0;
      const walker = document.createTreeWalker(line, NodeFilter.SHOW_TEXT);
      let tn: Node | null = walker.nextNode();
      if (!tn) {
        // Empty line — position at the line element itself.
        return { node: line, offset: 0 };
      }
      while (tn) {
        const len = (tn.textContent ?? "").length;
        if (acc + len >= within) {
          return { node: tn, offset: within - acc };
        }
        acc += len;
        tn = walker.nextNode();
      }
      // Past the last text node — clamp to end of line.
      return { node: line, offset: line.childNodes.length };
    }
    consumed = lineEnd + 1; // +1 for the newline join
  }
  return null;
}

/**
 * Replace the text in `[start, end)` with `newText`, applied through
 * CodeMirror's contentEditable input handling.
 *
 * Mechanism: position a DOM Selection across the target range, then
 * `document.execCommand("insertText")`. CM6 processes the resulting
 * input event and updates its model — this is the only content-script-
 * safe way to edit CM6 without injecting a page script.
 *
 * Returns { ok, reason }. `ok=false` with reason="offscreen" means the
 * target range isn't rendered (virtualised) — the user must scroll
 * near it and retry.
 */
export function applyEdit(
  start: number,
  end: number,
  newText: string,
): { ok: boolean; reason: string } {
  const content = document.querySelector<HTMLElement>(".cm-content");
  if (!content) return { ok: false, reason: "editor not found" };

  const startPos = offsetToDOMPosition(start);
  const endPos = offsetToDOMPosition(end);
  if (!startPos || !endPos) {
    return { ok: false, reason: "offscreen" };
  }

  const sel = window.getSelection();
  if (!sel) return { ok: false, reason: "no selection api" };

  try {
    const range = document.createRange();
    range.setStart(startPos.node, startPos.offset);
    range.setEnd(endPos.node, endPos.offset);
    sel.removeAllRanges();
    sel.addRange(range);
    content.focus();
    // execCommand is deprecated but remains the working path for
    // programmatic contentEditable edits that CM6 will observe.
    const ok = document.execCommand("insertText", false, newText);
    return ok
      ? { ok: true, reason: "" }
      : { ok: false, reason: "execCommand rejected" };
  } catch (e) {
    return { ok: false, reason: String(e) };
  }
}

