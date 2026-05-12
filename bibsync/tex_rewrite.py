"""Shared utilities for rewriting LaTeX source files.

Two operations:
  * :func:`rename_keys_in_project` — replace `\\cite{old}` with `\\cite{new}` across every
    .tex file under a project root. Used by ``fix`` to propagate .bib key renames.
  * :func:`insert_cite_after_anchor` — find a verbatim anchor phrase in a .tex file and
    insert ``\\cite{key}`` right after it. Used by ``suggest`` to add new citations.

Both operations:
  * Honour LaTeX comments — never edit content after an unescaped ``%``.
  * Write atomically (write to a tempfile, then ``os.replace``).
"""

from __future__ import annotations

import os
import re
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

_COMMENT_RE = re.compile(r"(?<!\\)%[^\n]*")
_CITE_KEY_RE = re.compile(r"\\(?P<cmd>(?:no)?cite\w*)\s*(?P<opts>(?:\[[^\]]*\])*)\s*\{(?P<keys>[^}]*)\}")


@dataclass
class TexRewriteSummary:
    files_scanned: int = 0
    files_changed: int = 0
    edits_by_file: dict[Path, int] = field(default_factory=dict)


def _iter_tex_files(project_root: Path) -> Iterable[Path]:
    for p in project_root.rglob("*.tex"):
        if any(part in {".git", "node_modules", ".venv", "venv"} for part in p.parts):
            continue
        yield p


def _split_comment_safe(line: str) -> tuple[str, str]:
    """Split a single line into (code_part, comment_part). The comment_part starts at the
    first unescaped ``%`` and is preserved verbatim (including the ``%`` itself)."""
    m = _COMMENT_RE.search(line)
    if not m:
        return line, ""
    return line[: m.start()], line[m.start() :]


def _rename_keys_in_text(text: str, renames: dict[str, str]) -> tuple[str, int]:
    """Replace cite keys in ``text`` according to ``renames``. Returns (new_text, num_edits).

    Operates line-by-line so we can preserve comment regions verbatim.
    """
    if not renames:
        return text, 0

    out_lines: list[str] = []
    edits = 0
    # Preserve original line endings: use splitlines(keepends=True) and join.
    for line in text.splitlines(keepends=True):
        code, comment = _split_comment_safe(line)

        def replace(match: re.Match) -> str:
            nonlocal edits
            keys_blob = match.group("keys")
            new_keys: list[str] = []
            changed = False
            for raw_key in keys_blob.split(","):
                key = raw_key.strip()
                if key in renames:
                    new_keys.append(renames[key])
                    changed = True
                else:
                    new_keys.append(key)
            if changed:
                edits += 1
                # Preserve any whitespace-around-commas style by joining with ", ".
                new_blob = ", ".join(new_keys)
                return f"\\{match.group('cmd')}{match.group('opts')}{{{new_blob}}}"
            return match.group(0)

        new_code = _CITE_KEY_RE.sub(replace, code)
        out_lines.append(new_code + comment)

    return "".join(out_lines), edits


def rename_keys_in_project(
    project_root: Path, renames: dict[str, str]
) -> TexRewriteSummary:
    """Apply key renames to every .tex file under ``project_root``."""
    summary = TexRewriteSummary()
    if not renames:
        return summary
    for tex in _iter_tex_files(project_root):
        summary.files_scanned += 1
        try:
            text = tex.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        new_text, edits = _rename_keys_in_text(text, renames)
        if edits:
            _atomic_write_text(tex, new_text)
            summary.files_changed += 1
            summary.edits_by_file[tex] = edits
    return summary


def _normalize(s: str) -> str:
    """Collapse whitespace for fuzzy anchor matching. Does NOT remove LaTeX commands —
    if the LLM stripped \\textit{...} we have to handle that separately."""
    return re.sub(r"\s+", " ", s).strip()


def _find_anchor(haystack: str, anchor: str) -> int:
    """Return the offset *just after* ``anchor`` in ``haystack``, or -1 if not found.

    Tries:
      1. Exact substring match.
      2. Whitespace-normalized match (treats ``\\n``, ``~``, and runs of spaces as one space).
      3. Anchor with LaTeX inline commands stripped.
    """
    idx = haystack.find(anchor)
    if idx >= 0:
        return idx + len(anchor)

    norm_hay = _normalize(haystack)
    norm_anchor = _normalize(anchor)
    idx = norm_hay.find(norm_anchor)
    if idx >= 0:
        # Map normalized index back to original by walking character-by-character.
        return _map_norm_to_orig(haystack, idx + len(norm_anchor))

    stripped_anchor = _normalize(re.sub(r"\\[a-zA-Z]+\*?\{([^}]*)\}", r"\1", anchor))
    if stripped_anchor and stripped_anchor != norm_anchor:
        idx = norm_hay.find(stripped_anchor)
        if idx >= 0:
            return _map_norm_to_orig(haystack, idx + len(stripped_anchor))

    return -1


def _map_norm_to_orig(orig: str, norm_pos: int) -> int:
    """Walk ``orig`` until we have emitted ``norm_pos`` characters of normalized output."""
    emitted = 0
    in_ws = False
    for i, ch in enumerate(orig):
        if ch.isspace():
            if not in_ws and emitted > 0:
                emitted += 1
                in_ws = True
        else:
            emitted += 1
            in_ws = False
        if emitted >= norm_pos:
            return i + 1
    return len(orig)


def insert_cite_after_anchor(
    tex_file: Path, anchor: str, cite_key: str, *, cite_cmd: str = "cite"
) -> bool:
    """Find ``anchor`` in ``tex_file`` and insert ``\\cite{cite_key}`` right after it.
    Returns True if a substitution was made, False if the anchor was not found.

    The inserted call is preceded by ``~`` (non-breaking space) — the LaTeX convention
    for binding a citation to the preceding word so they don't get split at a line break.
    """
    text = tex_file.read_text(encoding="utf-8", errors="replace")

    # Strip comments for searching, but keep the original text for editing.
    code_only = _COMMENT_RE.sub(lambda m: " " * len(m.group(0)), text)
    insert_at = _find_anchor(code_only, anchor)
    if insert_at < 0:
        return False

    insertion = f"~\\{cite_cmd}{{{cite_key}}}"
    new_text = text[:insert_at] + insertion + text[insert_at:]
    _atomic_write_text(tex_file, new_text)
    return True


def append_cite_to_paragraph(
    tex_file: Path, paragraph_text: str, cite_key: str, *, cite_cmd: str = "cite"
) -> bool:
    """Fallback when no anchor is found: append ``\\cite{key}`` to the END of the paragraph
    containing ``paragraph_text``. The paragraph is identified by finding any substantial
    substring of paragraph_text in the file."""
    text = tex_file.read_text(encoding="utf-8", errors="replace")
    # Use first ~80 chars of the paragraph as a recognition seed.
    seed = _normalize(paragraph_text)[:80]
    code_only = _COMMENT_RE.sub(lambda m: " " * len(m.group(0)), text)
    norm_code = _normalize(code_only)
    if seed not in norm_code:
        return False

    # Find the paragraph boundary (next blank line) in the original.
    idx = code_only.find(paragraph_text.strip()[:40])
    if idx < 0:
        # Try with normalized
        norm_idx = norm_code.find(seed)
        if norm_idx < 0:
            return False
        idx = _map_norm_to_orig(code_only, norm_idx)

    # Walk forward to the end of the paragraph (next blank line, or end of file).
    end = text.find("\n\n", idx)
    if end < 0:
        end = len(text)
    # Walk back past trailing whitespace within the paragraph.
    while end > idx and text[end - 1].isspace():
        end -= 1

    insertion = f"~\\{cite_cmd}{{{cite_key}}}"
    new_text = text[:end] + insertion + text[end:]
    _atomic_write_text(tex_file, new_text)
    return True


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(content)
        os.replace(tmp, path)
    except Exception:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise
