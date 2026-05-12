"""Suggest and insert citations into a LaTeX file that currently lacks them.

Workflow:
  1. Split the .tex file into paragraphs.
  2. For each paragraph WITHOUT existing \\cite{} calls, ask the LLM what should be cited.
  3. For each suggestion, search Google Scholar, pick the canonical version, fetch BibTeX.
  4. If interactive, ask the user to confirm before writing.
  5. On confirmation:
       a. Insert \\cite{key} after the LLM-provided anchor phrase in the .tex.
       b. Append the BibTeX entry to the .bib file.

Single .bib file assumption: the user maintains one .bib per project.
"""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from . import bibtex, llm, picker, scholar, tex_rewrite
from .models import PaperHit

_PARA_BOUNDARY = re.compile(r"\n\s*\n")
_CITE_PRESENCE_RE = re.compile(r"\\(?:no)?cite\w*\s*(?:\[[^\]]*\])*\s*\{")


@dataclass
class SuggestionResult:
    paragraph_index: int
    paragraph_preview: str
    query: str
    anchor: str
    reason: str
    scholar_hit: Optional[PaperHit] = None
    cite_key: Optional[str] = None
    status: str = "pending"  # "added" | "skipped" | "no_scholar_hit" | "anchor_not_found" | "duplicate" | "error"
    note: str = ""


@dataclass
class SuggestReport:
    tex_file: Path
    bib_file: Path
    paragraphs_scanned: int = 0
    paragraphs_with_existing_cites: int = 0
    results: list[SuggestionResult] = field(default_factory=list)

    def summary(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for r in self.results:
            out[r.status] = out.get(r.status, 0) + 1
        return out


def _split_paragraphs(text: str) -> list[str]:
    return [p for p in _PARA_BOUNDARY.split(text) if p.strip()]


def _has_existing_cite(paragraph: str) -> bool:
    return bool(_CITE_PRESENCE_RE.search(paragraph))




async def _resolve_suggestion(
    suggestion: llm.CitationSuggestion,
    paragraph: str,
    paragraph_idx: int,
    *,
    headless: bool,
) -> tuple[SuggestionResult, Optional[dict]]:
    result = SuggestionResult(
        paragraph_index=paragraph_idx,
        paragraph_preview=paragraph[:80] + ("…" if len(paragraph) > 80 else ""),
        query=suggestion.query,
        anchor=suggestion.anchor,
        reason=suggestion.reason,
    )

    try:
        hits = await scholar.search(suggestion.query, headless=headless, max_results=5)
    except Exception as e:
        result.status = "error"
        result.note = f"scholar search failed: {e}"
        return result, None

    if not hits:
        result.status = "no_scholar_hit"
        result.note = f"no Scholar results for {suggestion.query!r}"
        return result, None

    top = hits[0]
    candidates = [top]
    if top.versions_url:
        try:
            versions = await scholar.fetch_versions(top.versions_url, headless=headless)
            if versions:
                candidates = versions
        except Exception:
            pass

    canonical = picker.pick_canonical(candidates)
    result.scholar_hit = canonical

    if not canonical.cluster_id:
        result.status = "error"
        result.note = "canonical hit had no cluster id"
        return result, None

    try:
        bib_text = await scholar.fetch_bibtex_for_cluster(
            canonical.cluster_id, headless=headless
        )
    except Exception as e:
        result.status = "error"
        result.note = f"fetch bibtex failed: {e}"
        return result, None

    db = bibtex.parse_string(bib_text)
    if not db.entries:
        result.status = "error"
        result.note = "scholar returned bibtex but it failed to parse"
        return result, None

    return result, db.entries[0]


async def suggest_for_file(
    tex_file: Path,
    bib_file: Path,
    *,
    headless: bool = False,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    only_paragraphs_without_cites: bool = True,
    auto_approve: bool = False,
    approve_fn=None,  # callable(SuggestionResult, candidate_bibtex_entry) -> bool
    delay_seconds: float = 1.5,
) -> SuggestReport:
    """Scan ``tex_file`` paragraph-by-paragraph for missing citations.

    When ``auto_approve`` is False (default) and ``approve_fn`` is provided, the function
    calls ``approve_fn`` for each suggestion to ask the user whether to commit it.
    """
    text = tex_file.read_text(encoding="utf-8", errors="replace")
    paragraphs = _split_paragraphs(text)

    report = SuggestReport(tex_file=tex_file, bib_file=bib_file)
    report.paragraphs_scanned = len(paragraphs)

    db = bibtex.load(bib_file)

    for p_idx, paragraph in enumerate(paragraphs):
        if only_paragraphs_without_cites and _has_existing_cite(paragraph):
            report.paragraphs_with_existing_cites += 1
            continue

        try:
            suggestions = llm.suggest_citations(paragraph, model=model, api_key=api_key)
        except Exception as e:
            r = SuggestionResult(
                paragraph_index=p_idx,
                paragraph_preview=paragraph[:80],
                query="",
                anchor="",
                reason="",
                status="error",
                note=f"llm suggest failed: {e}",
            )
            report.results.append(r)
            continue

        for sugg in suggestions:
            r, entry = await _resolve_suggestion(
                sugg, paragraph, p_idx, headless=headless
            )

            if entry is None:
                report.results.append(r)
                continue

            # Set the cite key. Prefer a clean derived key over Scholar's auto-generated one.
            cite_key = bibtex.derive_cite_key(entry)
            cite_key = bibtex.ensure_unique_key(db, cite_key)
            entry["ID"] = cite_key
            r.cite_key = cite_key

            # Approval gate.
            approved = auto_approve
            if not approved and approve_fn is not None:
                approved = approve_fn(r, entry)
            elif not approved and approve_fn is None:
                approved = True  # no approval callback provided; default to True

            if not approved:
                r.status = "skipped"
                r.note = "user rejected"
                report.results.append(r)
                continue

            # Check duplicate against current .bib state.
            stored, was_added = bibtex.append_entry(db, entry)
            if not was_added:
                r.status = "duplicate"
                r.cite_key = stored["ID"]
                r.note = f"already in .bib as {stored['ID']}"
            # Insert \cite{} into the .tex.
            inserted = tex_rewrite.insert_cite_after_anchor(
                tex_file, r.anchor, r.cite_key
            )
            if not inserted:
                inserted = tex_rewrite.append_cite_to_paragraph(
                    tex_file, paragraph, r.cite_key
                )
            if inserted and r.status != "duplicate":
                r.status = "added"
            elif not inserted:
                r.status = "anchor_not_found"
                r.note = "could not locate insertion point in .tex"
            report.results.append(r)

            await asyncio.sleep(delay_seconds)

    # Single .bib write at the end.
    if any(r.status in ("added", "duplicate") for r in report.results):
        bibtex.dump(db, bib_file)

    return report


def suggest_for_file_sync(*args, **kwargs) -> SuggestReport:
    return asyncio.run(suggest_for_file(*args, **kwargs))
