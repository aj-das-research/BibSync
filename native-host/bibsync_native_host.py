#!/usr/bin/env python3
"""BibSync Chrome native messaging host.

Bridges Chrome's native-messaging protocol (length-prefixed JSON over
stdin/stdout) to the local ``bibsync serve`` HTTP API.

Why this layer exists:

  1. Chrome content scripts CAN'T reliably fetch ``http://127.0.0.1:*`` —
     CORS + private-network-access policies block or warn aggressively.
  2. Native messaging isolates the browser-page code (which runs in the
     web origin sandbox) from the local privileged process.
  3. Same host works on macOS / Linux / Windows with no per-OS branching.
  4. The host stays SMALL — it doesn't run the AI. It just forwards
     requests to ``bibsync serve``, which the user runs separately.

Protocol (https://developer.chrome.com/docs/extensions/develop/concepts/native-messaging):

  • 4-byte little-endian length prefix
  • Followed by length bytes of UTF-8 JSON
  • Bidirectional; messages can be sent in either direction asynchronously

Wire format (extension → host):
  { "id": "<correlation-id>", "method": "GET|POST|DELETE",
    "path": "/audit", "body": <json>, "query": { ... } }

Wire format (host → extension):
  { "id": "<correlation-id>", "ok": true,  "status": 200, "body": <json> }
  { "id": "<correlation-id>", "ok": false, "error": "..." }

Token discovery:
  Reads ``~/.config/bibsync/server.token`` on every request (so a
  newly-launched ``bibsync serve`` is picked up without restarting Chrome).
"""
from __future__ import annotations

import json
import logging
import os
import struct
import sys
from pathlib import Path
from typing import Any, Optional

# Standard-library HTTP — avoids any third-party dep that may not be
# available in the user's interpreter (Chrome launches the host with
# whatever python3 is in PATH, NOT necessarily inside the project venv).
import urllib.error
import urllib.parse
import urllib.request

# Default server URL — matches ``bibsync serve``'s defaults.
SERVER_URL = os.environ.get("BIBSYNC_SERVER_URL", "http://127.0.0.1:38476")
REQUEST_TIMEOUT = float(os.environ.get("BIBSYNC_NATIVE_TIMEOUT", "180"))

# Log to ~/Library/Logs/bibsync-native-host.log (macOS convention).
# Chrome captures stderr; we use a file so the user can inspect history
# without re-running with chrome --enable-logging.
def _log_path() -> Path:
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Logs" / "bibsync-native-host.log"
    if sys.platform == "win32":
        return Path.home() / "AppData" / "Local" / "bibsync" / "native-host.log"
    return Path.home() / ".local" / "share" / "bibsync" / "native-host.log"


def _setup_logging() -> None:
    p = _log_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=str(p),
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )


def _token_path() -> Path:
    """Same token file ``bibsync serve`` writes."""
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "bibsync" / "server.token"
    if sys.platform == "win32":
        return Path.home() / "AppData" / "Local" / "bibsync" / "server.token"
    return Path.home() / ".config" / "bibsync" / "server.token"


def _read_token() -> Optional[str]:
    p = _token_path()
    try:
        return p.read_text(encoding="utf-8").strip()
    except (OSError, UnicodeDecodeError):
        return None


# ── Chrome native messaging framing ────────────────────────────────────────


def _read_message() -> Optional[dict]:
    """Read one length-prefixed JSON message from stdin. Returns None on EOF."""
    raw_length = sys.stdin.buffer.read(4)
    if len(raw_length) != 4:
        return None
    msg_length = struct.unpack("<I", raw_length)[0]
    if msg_length > 1024 * 1024 * 64:  # 64 MB safety cap
        logging.error("incoming message too large: %d bytes", msg_length)
        return None
    raw = sys.stdin.buffer.read(msg_length)
    if len(raw) != msg_length:
        return None
    try:
        return json.loads(raw.decode("utf-8"))
    except json.JSONDecodeError as e:
        logging.error("decode error: %s", e)
        return None


def _send_message(msg: dict) -> None:
    """Write one length-prefixed JSON message to stdout."""
    raw = json.dumps(msg, ensure_ascii=False).encode("utf-8")
    sys.stdout.buffer.write(struct.pack("<I", len(raw)))
    sys.stdout.buffer.write(raw)
    sys.stdout.buffer.flush()


# ── HTTP bridge ────────────────────────────────────────────────────────────


def _forward(msg: dict) -> dict:
    """Forward one extension message to the local bibsync server."""
    msg_id = msg.get("id", "")
    method = (msg.get("method") or "GET").upper()
    path = msg.get("path") or "/"
    body = msg.get("body")
    query = msg.get("query") or {}

    if path.startswith("/"):
        url = SERVER_URL.rstrip("/") + path
    else:
        url = path  # absolute URL, rare

    if query:
        url += ("&" if "?" in url else "?") + urllib.parse.urlencode(query)

    token = _read_token()
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    data: Optional[bytes] = None
    if body is not None and method in ("POST", "PUT", "PATCH", "DELETE"):
        data = json.dumps(body).encode("utf-8")

    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as resp:
            payload_bytes = resp.read()
            status = resp.status
            try:
                payload = json.loads(payload_bytes.decode("utf-8")) if payload_bytes else None
            except json.JSONDecodeError:
                payload = {"raw": payload_bytes.decode("utf-8", errors="replace")}
            return {
                "id": msg_id, "ok": True, "status": status, "body": payload,
            }
    except urllib.error.HTTPError as e:
        try:
            err_payload = json.loads(e.read().decode("utf-8")) if e.fp else None
        except Exception:
            err_payload = None
        logging.warning("HTTPError %s on %s %s", e.code, method, url)
        return {
            "id": msg_id, "ok": False, "status": e.code,
            "error": f"HTTP {e.code}: {e.reason}",
            "body": err_payload,
        }
    except urllib.error.URLError as e:
        logging.warning("URLError on %s %s: %s", method, url, e)
        return {
            "id": msg_id, "ok": False, "status": 0,
            "error": f"could not reach {url} — is `bibsync serve` running? ({e.reason})",
        }
    except Exception as e:
        logging.exception("unexpected error")
        return {"id": msg_id, "ok": False, "status": 0, "error": str(e)}


# ── main loop ──────────────────────────────────────────────────────────────


def main() -> None:
    _setup_logging()
    logging.info("native host started (server=%s)", SERVER_URL)
    try:
        while True:
            msg = _read_message()
            if msg is None:
                logging.info("EOF — exiting")
                return
            resp = _forward(msg)
            _send_message(resp)
    except BrokenPipeError:
        logging.info("Chrome closed the pipe — exiting")
    except KeyboardInterrupt:
        logging.info("interrupted — exiting")


if __name__ == "__main__":
    main()
