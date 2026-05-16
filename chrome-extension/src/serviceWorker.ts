/**
 * MV3 service worker — the extension's privileged message hub.
 *
 * Responsibilities:
 *   1. Open the side panel when the toolbar icon is clicked.
 *   2. Route messages from the side panel + content script to the native
 *      host (via nativeBridge) and relay responses back.
 *
 * The worker is EPHEMERAL — Chrome kills it after ~30s idle. We keep no
 * state in worker globals; every message handler is self-contained and
 * re-derives whatever it needs. The native-messaging port is opened
 * per-request inside callNative(), so there's nothing to leak on respawn.
 */
import { callNative } from "./nativeBridge";
import type { NativeRequest } from "./types";

// Open the side panel when the toolbar icon is clicked.
chrome.action.onClicked.addListener(async (tab) => {
  if (tab.windowId !== undefined) {
    await chrome.sidePanel.open({ windowId: tab.windowId });
  }
});

// Make the side panel available on Overleaf tabs automatically.
chrome.runtime.onInstalled.addListener(() => {
  chrome.sidePanel
    .setPanelBehavior({ openPanelOnActionClick: true })
    .catch((e) => console.error("[bibsync] setPanelBehavior failed", e));
});

/**
 * Message contract (side panel / content script → worker):
 *   { kind: "native", request: NativeRequest }   → forwards to native host
 *   { kind: "ping" }                              → worker liveness probe
 *
 * Every handler returns true to keep the sendResponse channel open for
 * the async reply (MV3 requirement).
 */
chrome.runtime.onMessage.addListener((msg, _sender, sendResponse) => {
  if (!msg || typeof msg !== "object") {
    sendResponse({ ok: false, error: "malformed message" });
    return false;
  }

  if (msg.kind === "ping") {
    sendResponse({ ok: true, pong: true });
    return false;
  }

  if (msg.kind === "native") {
    const request = msg.request as Omit<NativeRequest, "id">;
    callNative(request)
      .then((resp) => sendResponse(resp))
      .catch((err) =>
        sendResponse({
          ok: false,
          status: 0,
          error: err instanceof Error ? err.message : String(err),
        }),
      );
    return true; // keep the channel open for the async reply
  }

  sendResponse({ ok: false, error: `unknown message kind: ${msg.kind}` });
  return false;
});
