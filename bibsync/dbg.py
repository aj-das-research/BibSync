"""Lightweight per-step trace logger for the BibSync pipelines.

Enable by passing ``--debug`` on the CLI, or by setting ``BIBSYNC_DEBUG=1`` in the
environment. When enabled, every important step in scholar / fix / extract / LLM
emits one line to stderr in a paste-friendly ``key=value`` format::

    [bibsync:scholar.search] query='Attention is all you need' max_results=10
    [bibsync:scholar.parse] hit=0 title='Attention is all you need' year=2017 cited=142000
    [bibsync:fix.heuristic] passed=2 rejected=3
    [bibsync:llm.verify] same=False conf=0.95 reason='R1: author mismatch'

Output goes to stderr so the trace doesn't pollute piped stdout.
"""

from __future__ import annotations

import os
import sys
from typing import Any

_ENABLED = False
_MAX_FIELD_LEN = 80


def enable() -> None:
    global _ENABLED
    _ENABLED = True


def is_enabled() -> bool:
    return _ENABLED or os.environ.get("BIBSYNC_DEBUG") == "1"


def _truncate(v: Any) -> Any:
    if isinstance(v, str) and len(v) > _MAX_FIELD_LEN:
        return v[: _MAX_FIELD_LEN - 3] + "..."
    return v


def trace(stage: str, message: str = "", **fields: Any) -> None:
    """Emit a single debug line. ``stage`` is a short tag like ``scholar.search`` or
    ``fix.heuristic``. ``message`` is optional free-text; ``fields`` are kv pairs.

    No-op if debug is disabled (so it's safe to leave traces in hot paths).
    """
    if not is_enabled():
        return
    parts = [f"[bibsync:{stage}]"]
    if message:
        parts.append(message)
    for k, v in fields.items():
        parts.append(f"{k}={_truncate(v)!r}")
    print(" ".join(parts), file=sys.stderr, flush=True)
