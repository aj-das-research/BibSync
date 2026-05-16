/**
 * Native-messaging bridge — runs in the SERVICE WORKER context.
 *
 * The side panel and content script never talk to the native host
 * directly; they post messages to the service worker, which owns the
 * native-messaging connection. This keeps the privileged channel in one
 * place and works with MV3's ephemeral-worker model: each request opens
 * a short-lived connectNative port, sends one message, awaits one reply,
 * and disconnects. No long-lived port to leak when the worker is killed.
 */
import type { NativeRequest, NativeResponse } from "./types";

const HOST_NAME = "com.bibsync.host";

let _counter = 0;
function nextId(): string {
  _counter += 1;
  return `req-${Date.now()}-${_counter}`;
}

/**
 * Send one request to the native host and resolve with its response.
 *
 * Opens a fresh `chrome.runtime.connectNative` port per call. The host
 * (a Python script) forwards to `bibsync serve` over HTTP and replies.
 * Rejects if the host can't be reached (native host not installed) or
 * the port disconnects before replying.
 */
export function callNative(
  req: Omit<NativeRequest, "id">,
  timeoutMs = 200_000,
): Promise<NativeResponse> {
  return new Promise((resolve, reject) => {
    let settled = false;
    let port: chrome.runtime.Port;
    try {
      port = chrome.runtime.connectNative(HOST_NAME);
    } catch (e) {
      reject(new Error(`could not connect to native host: ${String(e)}`));
      return;
    }

    const id = nextId();
    const timer = setTimeout(() => {
      if (settled) return;
      settled = true;
      try { port.disconnect(); } catch { /* ignore */ }
      reject(new Error(`native host request timed out after ${timeoutMs}ms`));
    }, timeoutMs);

    port.onMessage.addListener((msg: NativeResponse) => {
      if (settled) return;
      // Ignore messages that aren't our correlation id (defensive — the
      // per-call port should only ever carry one reply).
      if (msg && msg.id && msg.id !== id) return;
      settled = true;
      clearTimeout(timer);
      try { port.disconnect(); } catch { /* ignore */ }
      resolve(msg);
    });

    port.onDisconnect.addListener(() => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      const err = chrome.runtime.lastError?.message ?? "native host disconnected";
      reject(new Error(err));
    });

    const payload: NativeRequest = { id, ...req };
    try {
      port.postMessage(payload);
    } catch (e) {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      reject(new Error(`failed to post to native host: ${String(e)}`));
    }
  });
}
