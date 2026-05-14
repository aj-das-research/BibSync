"""Audit existing citations in a LaTeX project for hallucination / misattribution.

For every ``\\cite{key}`` call in the project's .tex files, this verifies that the
cited paper (per its .bib entry) actually supports the surrounding prose claim.
Hallucinated citations — typically introduced by LLM-assisted paper drafting where
the model fabricated plausible-looking BibTeX entries — are flagged and optionally
removed with ``--fix``.

Pipeline:
  1. Scan all .tex files for ``\\cite{...}`` matches; for each, extract the
     surrounding sentence (the claim).
  2. Group occurrences by cite key — one LLM call per unique paper/claim pair
     instead of per textual occurrence.
  3. Look up each key in the .bib; missing keys are flagged ``missing_in_bib``.
  4. For each (claim, paper) pair, call ``llm.audit_citation``.
  5. Build a report with: verified / hallucinated / unverifiable / missing_in_bib
     buckets.
  6. With ``--fix``, replace every hallucinated ``\\cite{...}`` with a marker
     comment so the user can see exactly what was removed.
"""

from __future__ import annotations

import asyncio
import bisect
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from . import bibtex, dbg, llm

# Match any \cite-family command — same regex as scanner.py.
_CITE_RE = re.compile(r"\\(?:no)?cite\w*\s*(?:\[[^\]]*\])*\s*\{([^}]+)\}")

# LaTeX line comment: % to end-of-line, but \% is an escaped percent (not a comment).
_COMMENT_RE = re.compile(r"(?<!\\)%[^\n]*")


def _strip_comments(text: str) -> str:
    """Replace LaTeX line comments with spaces of equal length so character offsets
    and line numbers stay correct. Commented-out ``\\cite{}`` calls must not be
    audited; they're notes-to-self, not real citations."""
    return _COMMENT_RE.sub(lambda m: " " * len(m.group(0)), text)


@dataclass
class CitationCheck:
    """One audited use of a ``\\cite{key}`` somewhere in the project."""

    cite_key: str
    file: Path
    line: int
    char_offset: int  # offset of the \cite{...} call start in the source file
    claim_text: str  # the surrounding sentence (with the \cite{} call stripped)
    bib_entry: Optional[dict] = None
    status: str = "pending"  # "verified" | "hallucinated" | "unverifiable" | "missing_in_bib"
    confidence: float = 0.0
    reasoning: str = ""
    fixed: bool = False  # True if --fix replaced the \cite{} in the .tex


@dataclass
class AuditReport:
    project_root: Path
    bib_file: Path
    tex_files_scanned: int = 0
    checks: list[CitationCheck] = field(default_factory=list)

    def summary(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for c in self.checks:
            out[c.status] = out.get(c.status, 0) + 1
        return out


# --- helpers ---------------------------------------------------------------


def _strip_cite_calls(text: str) -> str:
    """Remove ``\\cite{...}`` calls from a fragment so the claim reads cleanly when
    shown to the LLM."""
    out = re.sub(
        r"~?\\(?:no)?cite\w*\s*(?:\[[^\]]*\])*\s*\{[^}]+\}",
        "",
        text,
    )
    return re.sub(r"\s+", " ", out).strip()


def _extract_claim(text: str, cite_start: int, cite_end: int) -> str:
    """Return the sentence containing the cite, with the cite call stripped out.

    A "sentence" boundary is the nearest preceding ``[.!?]\\s+`` (or paragraph
    start) and the nearest following ``[.!?]\\s`` (or end). For LaTeX prose this
    is good enough — math periods rarely surround \\cite{}.
    """
    # Walk back to start of sentence (or paragraph).
    start = 0
    for m in re.finditer(r"(?:[.!?]\s+)|(?:\n\s*\n)", text[:cite_start]):
        start = m.end()
    # Walk forward to next sentence end.
    after = text[cite_end:]
    end_m = re.search(r"[.!?](?:\s|\n|$)", after)
    end = cite_end + (end_m.end() if end_m else len(after))
    return _strip_cite_calls(text[start:end])


def _entry_to_audit_inputs(entry: dict) -> tuple[str, str, Optional[int], str]:
    """Extract (title, authors_str, year, venue) from a bibtex entry — cleaned up."""
    title = re.sub(r"[{}]", "", entry.get("title", "") or "").strip()
    authors = re.sub(r"[{}]", "", entry.get("author", "") or "").strip()
    year: Optional[int] = None
    ystr = entry.get("year", "") or ""
    m = re.search(r"\d{4}", ystr)
    if m:
        year = int(m.group(0))
    venue = (
        entry.get("booktitle") or entry.get("journal") or entry.get("publisher") or ""
    )
    venue = re.sub(r"[{}]", "", venue).strip()
    return title, authors, year, venue


def _gather_citations(project_root: Path) -> tuple[list[CitationCheck], int]:
    """Walk project_root for .tex files and build a CitationCheck for every
    ``\\cite{}`` occurrence (one per key per location).

    Returns ``(checks, num_tex_files_scanned)``.
    """
    checks: list[CitationCheck] = []
    tex_files = [
        p
        for p in project_root.rglob("*.tex")
        if not any(part in {".git", ".venv", "node_modules", "venv"} for part in p.parts)
    ]
    for tex in tex_files:
        try:
            raw = tex.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        # Strip LaTeX line comments first (preserves offsets via space-padding) so
        # commented-out \cite{} calls are not audited as real citations.
        text = _strip_comments(raw)
        # Precompute line-start offsets for fast line-number lookup.
        line_starts = [0]
        for i, ch in enumerate(text):
            if ch == "\n":
                line_starts.append(i + 1)

        for m in _CITE_RE.finditer(text):
            line_no = bisect.bisect_right(line_starts, m.start())
            claim = _extract_claim(text, m.start(), m.end())
            for raw_key in m.group(1).split(","):
                key = raw_key.strip()
                if not key:
                    continue
                checks.append(
                    CitationCheck(
                        cite_key=key,
                        file=tex,
                        line=line_no,
                        char_offset=m.start(),
                        claim_text=claim,
                    )
                )
    return checks, len(tex_files)


# --- main pipeline ---------------------------------------------------------


async def audit_project(
    project_root: Path,
    bib_file: Path,
    *,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    delay_seconds: float = 0.5,
    fix: bool = False,
    confidence_floor: float = 0.7,
) -> AuditReport:
    """Audit every ``\\cite{}`` in ``project_root`` against the ``bib_file``.

    Confidence floor: only mark a citation as ``hallucinated`` when the LLM
    confidently judges supports=false. Weaker no-support verdicts fall into
    ``unverifiable`` so we don't trigger ``--fix`` on uncertain calls.
    """
    project_root = project_root.resolve()
    bib_file = bib_file.resolve()
    dbg.trace("audit.start", project=str(project_root), bib=str(bib_file), fix=fix)

    report = AuditReport(project_root=project_root, bib_file=bib_file)
    db = bibtex.load(bib_file)
    bib_by_key = {e.get("ID"): e for e in db.entries}

    checks, tex_count = _gather_citations(project_root)
    report.tex_files_scanned = tex_count
    report.checks = checks

    dbg.trace(
        "audit.scan",
        f"{len(checks)} citation occurrences across {tex_count} .tex files",
        unique_keys=len({c.cite_key for c in checks}),
    )

    # Cache LLM verdicts by (cite_key, claim_text) so the same paper cited for the
    # same claim in two files only spends one LLM call.
    verdict_cache: dict[tuple[str, str], llm.CitationAudit] = {}
    judged_keys: set[str] = set()

    for check in checks:
        # Step 1 — bib lookup.
        entry = bib_by_key.get(check.cite_key)
        if entry is None:
            check.status = "missing_in_bib"
            check.confidence = 1.0
            check.reasoning = "no entry with this key in the .bib"
            dbg.trace("audit.check", "missing_in_bib", key=check.cite_key)
            continue
        check.bib_entry = entry

        title, authors, year, venue = _entry_to_audit_inputs(entry)
        cache_key = (check.cite_key, check.claim_text)

        # Step 2 — LLM audit (cached).
        if cache_key in verdict_cache:
            verdict = verdict_cache[cache_key]
            dbg.trace("audit.check", "cache hit", key=check.cite_key)
        else:
            verdict = llm.audit_citation(
                claim_text=check.claim_text,
                cited_paper_title=title,
                cited_paper_authors=authors,
                cited_paper_year=year,
                cited_paper_venue=venue,
                model=model,
                api_key=api_key,
            )
            verdict_cache[cache_key] = verdict
            # Polite pacing only when we actually issue a new LLM call.
            if check.cite_key not in judged_keys and delay_seconds > 0:
                await asyncio.sleep(delay_seconds)
            judged_keys.add(check.cite_key)

        check.confidence = verdict.confidence
        check.reasoning = verdict.reasoning
        if verdict.supports:
            check.status = "verified"
        elif verdict.confidence >= confidence_floor:
            check.status = "hallucinated"
        else:
            check.status = "unverifiable"

        dbg.trace(
            "audit.check",
            check.status,
            key=check.cite_key,
            conf=round(check.confidence, 2),
            line=check.line,
        )

    # Step 3 — Optional --fix: replace hallucinated cite calls with marker comments.
    if fix:
        _apply_fixes(report)

    return report


def _apply_fixes(report: AuditReport) -> None:
    """Per-occurrence rewrite of hallucinated ``\\cite{}`` calls.

    KEY DETAIL: the same cite key can be used correctly in one location and
    hallucinated in another (e.g., ``vaswani2017attention`` for a real
    self-attention claim AND for a fabricated LibriSpeech claim). We must only
    rewrite the occurrence at the hallucinated location, not every occurrence of
    that key. This requires tracking the character offset of each citation —
    set in :func:`_gather_citations` and used here to scope the rewrite.

    Multi-key calls (``\\cite{a,b}``) where only one key is bad keep the good
    keys intact in the rewritten brace list.
    """
    by_file: dict[Path, list[CitationCheck]] = {}
    for c in report.checks:
        if c.status == "hallucinated":
            by_file.setdefault(c.file, []).append(c)

    for file, bad_checks in by_file.items():
        try:
            text = file.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue

        # Group hallucinated checks by the char_offset of the cite call they belong
        # to. A single ``\\cite{a,b,c}`` with two hallucinated keys is one offset
        # with two CitationChecks; we want to rewrite that one cite call.
        bad_by_offset: dict[int, list[CitationCheck]] = {}
        for c in bad_checks:
            bad_by_offset.setdefault(c.char_offset, []).append(c)

        # Build a list of (start, end, replacement) edits by re-finding the cite
        # calls in the comment-stripped text and matching offsets we recorded.
        text_stripped = _strip_comments(text)
        edits: list[tuple[int, int, str, list[CitationCheck]]] = []
        for m in _CITE_RE.finditer(text_stripped):
            if m.start() not in bad_by_offset:
                continue
            bad_here = bad_by_offset[m.start()]
            bad_keys_here = {c.cite_key for c in bad_here}
            reasons = {c.cite_key: c.reasoning for c in bad_here}

            keys = [k.strip() for k in m.group(1).split(",") if k.strip()]
            remaining = [k for k in keys if k not in bad_keys_here]
            removed = [k for k in keys if k in bad_keys_here]
            if not removed:
                continue

            comment_parts = [
                f"\\cite{{{k}}} — {reasons.get(k, 'topic mismatch')}" for k in removed
            ]
            comment = (
                "  % [bibsync audit] removed hallucinated: "
                + "; ".join(comment_parts)
            )
            if remaining:
                # Preserve the command (e.g. ``\\citep`` vs ``\\cite``) and any
                # bracket options before the brace.
                head = m.group(0).split("{", 1)[0]
                replacement = f"{head}{{{', '.join(remaining)}}}{comment}"
            else:
                replacement = comment.lstrip()

            edits.append((m.start(), m.end(), replacement, bad_here))

        if not edits:
            continue

        # Apply edits in reverse offset order so earlier offsets stay valid as we
        # rewrite the file.
        edits.sort(key=lambda t: t[0], reverse=True)
        new_text = text
        for start, end, repl, _ in edits:
            new_text = new_text[:start] + repl + new_text[end:]

        file.write_text(new_text, encoding="utf-8")
        for _, _, _, bad_here in edits:
            for c in bad_here:
                c.fixed = True
        dbg.trace(
            "audit.fix", "rewritten",
            file=str(file), edits=len(edits),
        )


def audit_project_sync(*args, **kwargs) -> AuditReport:
    return asyncio.run(audit_project(*args, **kwargs))
