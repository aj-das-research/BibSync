/**
 * Content script — injected into Overleaf pages.
 *
 * Thin bridge: the side panel can't read the Overleaf DOM (different
 * context), so it asks the content script. The content script owns the
 * `overleafAdapter` and answers three message kinds:
 *
 *   { kind: "ol-detect" }     → { editor: boolean }
 *   { kind: "ol-selection" }  → EditorSelection | null
 *   { kind: "ol-document" }   → { file, text }
 *
 * No audit logic here — the content script only reads editor state and
 * hands it to the side panel, which routes it through the worker to the
 * native host.
 */
import {
  applyEdit,
  detectEditor,
  getActiveFileName,
  getCurrentText,
  getSelection,
} from "./overleafAdapter";

chrome.runtime.onMessage.addListener((msg, _sender, sendResponse) => {
  if (!msg || typeof msg !== "object") return false;

  switch (msg.kind) {
    case "ol-detect":
      sendResponse({ editor: detectEditor() });
      return false;

    case "ol-selection":
      sendResponse({ selection: getSelection() });
      return false;

    case "ol-document":
      sendResponse({
        file: getActiveFileName(),
        text: getCurrentText(),
      });
      return false;

    case "ol-apply": {
      // { kind: "ol-apply", start, end, newText }
      const result = applyEdit(
        Number(msg.start),
        Number(msg.end),
        String(msg.newText ?? ""),
      );
      sendResponse(result);
      return false;
    }

    default:
      return false;
  }
});

// Announce readiness — useful for the side panel to know the content
// script is live on this tab.
console.debug("[bibsync] content script ready on", location.href);
