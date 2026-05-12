"""Scan a LaTeX project for \\cite{} keys and reconcile with .bib files.

Reports:
  - keys cited in .tex but missing from any .bib  (potential hallucinations or typos)
  - keys defined in .bib but never cited           (orphan entries — candidate for cleanup)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from . import bibtex

# Match any \cite-family command: \cite, \citep, \citet, \autocite, \textcite, \nocite, ...
_CITE_RE = re.compile(r"\\(?:no)?cite\w*\s*(?:\[[^\]]*\])?\s*(?:\[[^\]]*\])?\s*\{([^}]*)\}")

# Strip line comments: % is a comment unless preceded by a backslash (\%).
_COMMENT_RE = re.compile(r"(?<!\\)%[^\n]*")


@dataclass
class CitationUse:
    key: str
    file: Path
    line: int
    context: str = ""  # surrounding paragraph, populated when scan(..., with_context=True)


@dataclass
class ScanReport:
    project_root: Path
    tex_files: list[Path] = field(default_factory=list)
    bib_files: list[Path] = field(default_factory=list)
    uses: list[CitationUse] = field(default_factory=list)
    defined_keys: set[str] = field(default_factory=set)

    @property
    def cited_keys(self) -> set[str]:
        return {u.key for u in self.uses}

    @property
    def missing_keys(self) -> set[str]:
        """Cited but not defined in any .bib — these are likely hallucinations or typos."""
        return self.cited_keys - self.defined_keys

    @property
    def orphan_keys(self) -> set[str]:
        """Defined but never cited — safe to remove from .bib."""
        return self.defined_keys - self.cited_keys

    def uses_of(self, key: str) -> list[CitationUse]:
        return [u for u in self.uses if u.key == key]


def _strip_comments(text: str) -> str:
    return _COMMENT_RE.sub("", text)


_PARA_BOUNDARY = re.compile(r"\n\s*\n")


def _paragraph_around(text: str, offset: int) -> str:
    """Return the paragraph (blank-line-delimited block) containing ``offset``."""
    # Walk backwards/forwards to the nearest blank-line boundary.
    start = 0
    for m in _PARA_BOUNDARY.finditer(text, 0, offset):
        start = m.end()
    end_match = _PARA_BOUNDARY.search(text, offset)
    end = end_match.start() if end_match else len(text)
    return text[start:end].strip()


def _find_cite_uses_in_file(path: Path, *, with_context: bool = False) -> list[CitationUse]:
    uses: list[CitationUse] = []
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return uses

    cleaned = _strip_comments(text)
    # Build a line-number lookup. Comment stripping preserves newlines so positions
    # in `cleaned` still line up with the original line numbers.
    line_starts = [0]
    for i, ch in enumerate(cleaned):
        if ch == "\n":
            line_starts.append(i + 1)

    def offset_to_line(offset: int) -> int:
        lo, hi = 0, len(line_starts) - 1
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if line_starts[mid] <= offset:
                lo = mid
            else:
                hi = mid - 1
        return lo + 1  # 1-indexed

    for m in _CITE_RE.finditer(cleaned):
        keys_blob = m.group(1)
        line_no = offset_to_line(m.start())
        ctx = _paragraph_around(cleaned, m.start()) if with_context else ""
        for raw_key in keys_blob.split(","):
            key = raw_key.strip()
            if key:
                uses.append(CitationUse(key=key, file=path, line=line_no, context=ctx))
    return uses


def scan(project_root: Path, *, with_context: bool = False) -> ScanReport:
    """Walk ``project_root`` for .tex and .bib files and produce a report."""
    project_root = project_root.resolve()
    report = ScanReport(project_root=project_root)

    for p in project_root.rglob("*.tex"):
        if any(part in {".git", "node_modules", ".venv", "venv"} for part in p.parts):
            continue
        report.tex_files.append(p)
        report.uses.extend(_find_cite_uses_in_file(p, with_context=with_context))

    for p in project_root.rglob("*.bib"):
        if any(part in {".git", "node_modules", ".venv", "venv"} for part in p.parts):
            continue
        report.bib_files.append(p)
        db = bibtex.load(p)
        for entry in db.entries:
            key = entry.get("ID")
            if key:
                report.defined_keys.add(key)

    return report
