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

from . import bibtex, dbg, llm, picker, scholar, tex_rewrite
from .models import PaperHit

# Try at most this many queries per claim, and at most this many top candidates
# (sorted by cited_by) per claim when asking the LLM "does this support the claim?".
MAX_QUERIES_PER_CLAIM = 3
MAX_CANDIDATES_PER_CLAIM = 4
CLAIM_SUPPORT_CONFIDENCE_FLOOR = 0.7

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
    model: Optional[str],
    api_key: Optional[str],
) -> tuple[SuggestionResult, Optional[dict]]:
    """Agentic pipeline for one suggested citation:

      1. Run ALL of the LLM's proposed queries on Scholar; merge & dedupe candidates.
      2. Sort merged candidates by ``cited_by`` (most-cited first — the original
         paper is almost always vastly more cited than any derivative).
      3. For the top N candidates, ask the LLM-as-judge whether each candidate
         actually supports the prose claim. Accept the first ``supports=True`` with
         confidence ≥ ``CLAIM_SUPPORT_CONFIDENCE_FLOOR``.
      4. Fetch official BibTeX for the accepted candidate.

    Returns ``(result, bibtex_entry_or_None)``. If the LLM rejects every candidate
    across every query, the result is marked ``no_supporting_match`` and the caller
    leaves both .tex and .bib untouched for this suggestion.
    """
    result = SuggestionResult(
        paragraph_index=paragraph_idx,
        paragraph_preview=paragraph[:80] + ("…" if len(paragraph) > 80 else ""),
        query=suggestion.query,
        anchor=suggestion.anchor,
        reason=suggestion.reason,
    )

    dbg.trace(
        "suggest.resolve",
        "start",
        anchor=suggestion.anchor,
        queries=suggestion.queries,
        para=paragraph_idx,
    )

    # Step 1: run each query, merge results, dedupe by cluster_id.
    merged: dict[str, PaperHit] = {}
    for q_idx, query in enumerate(suggestion.queries[:MAX_QUERIES_PER_CLAIM]):
        dbg.trace("suggest.query", f"attempt #{q_idx + 1}", query=query)
        try:
            hits = await scholar.search(query, headless=headless, max_results=6)
        except Exception as e:
            dbg.trace("suggest.query", "ERR", error=str(e))
            continue
        dbg.trace("suggest.query", f"got {len(hits)} hits")
        for h in hits:
            if h.cluster_id and h.cluster_id not in merged:
                merged[h.cluster_id] = h
        if len(merged) >= 6:
            # Already have a healthy pool; stop spending queries.
            break

    if not merged:
        result.status = "no_scholar_hit"
        result.note = f"no Scholar results across {len(suggestion.queries)} querie(s)"
        dbg.trace("suggest.resolve", "no candidates from any query")
        return result, None

    # Step 2: sort by cited_by DESC. Original foundational papers crush derivatives
    # in citation count, so the most-cited heuristically-plausible candidate is the
    # one to ask the LLM about first.
    candidates = sorted(merged.values(), key=lambda h: (h.cited_by or 0), reverse=True)
    dbg.trace("suggest.merge", f"{len(candidates)} unique candidates after dedupe")
    for i, c in enumerate(candidates[:MAX_CANDIDATES_PER_CLAIM]):
        dbg.trace(
            "suggest.candidate",
            f"#{i+1}",
            title=c.title,
            authors=(c.authors[0] if c.authors else ""),
            year=c.year,
            cited=c.cited_by,
        )

    # Step 3: LLM judges each top candidate against the claim.
    accepted: Optional[PaperHit] = None
    last_reason = ""
    last_confidence = 0.0
    for i, candidate in enumerate(candidates[:MAX_CANDIDATES_PER_CLAIM]):
        verdict = llm.verify_claim_support(
            claim_text=suggestion.anchor,
            context=paragraph,
            candidate=candidate,
            model=model,
            api_key=api_key,
        )
        last_reason = verdict.reasoning
        last_confidence = verdict.confidence
        if verdict.supports and verdict.confidence >= CLAIM_SUPPORT_CONFIDENCE_FLOOR:
            accepted = candidate
            dbg.trace(
                "suggest.verify",
                "ACCEPTED",
                candidate_idx=i + 1,
                conf=round(verdict.confidence, 2),
                title=candidate.title,
            )
            break
        dbg.trace(
            "suggest.verify",
            f"rejected #{i+1}",
            conf=round(verdict.confidence, 2),
            reason=verdict.reasoning,
        )

    if accepted is None:
        result.status = "no_supporting_match"
        result.note = (
            f"LLM rejected all {min(len(candidates), MAX_CANDIDATES_PER_CLAIM)} "
            f"top candidate(s) — last reason: {last_reason}"
        )
        result.scholar_hit = candidates[0]
        return result, None

    result.scholar_hit = accepted

    if not accepted.cluster_id:
        result.status = "error"
        result.note = "accepted hit had no cluster id"
        return result, None

    # Step 4: fetch BibTeX for the LLM-verified canonical paper.
    try:
        bib_text = await scholar.fetch_bibtex_for_cluster(
            accepted.cluster_id, headless=headless
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

    dbg.trace(
        "suggest.start",
        tex=str(tex_file),
        bib=str(bib_file),
        paragraphs=len(paragraphs),
    )

    # Pool all Scholar calls in this run into ONE browser context.
    async with scholar.shared_session(headless=headless):
        for p_idx, paragraph in enumerate(paragraphs):
            if only_paragraphs_without_cites and _has_existing_cite(paragraph):
                report.paragraphs_with_existing_cites += 1
                dbg.trace("suggest.paragraph", "skip (already has \\cite)", idx=p_idx)
                continue

            dbg.trace(
                "suggest.paragraph",
                "scanning",
                idx=p_idx,
                preview=paragraph[:80],
            )
            try:
                suggestions = llm.suggest_citations(
                    paragraph, model=model, api_key=api_key
                )
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
            dbg.trace(
                "suggest.paragraph",
                "got suggestions",
                idx=p_idx,
                n=len(suggestions),
            )

            for sugg in suggestions:
                r, entry = await _resolve_suggestion(
                    sugg, paragraph, p_idx,
                    headless=headless, model=model, api_key=api_key,
                )

                if entry is None:
                    report.results.append(r)
                    continue

                # Set cite key from corrected metadata.
                cite_key = bibtex.derive_cite_key(entry)
                cite_key = bibtex.ensure_unique_key(db, cite_key)
                entry["ID"] = cite_key
                r.cite_key = cite_key
                dbg.trace(
                    "suggest.commit",
                    "proposing",
                    cite_key=cite_key,
                    anchor=r.anchor,
                )

                # Approval gate.
                approved = auto_approve
                if not approved and approve_fn is not None:
                    approved = approve_fn(r, entry)
                elif not approved and approve_fn is None:
                    approved = True  # default to True when no callback provided

                if not approved:
                    r.status = "skipped"
                    r.note = "user rejected"
                    dbg.trace("suggest.commit", "user rejected", cite_key=cite_key)
                    report.results.append(r)
                    continue

                # Check duplicate against current .bib state.
                stored, was_added = bibtex.append_entry(db, entry)
                if not was_added:
                    r.status = "duplicate"
                    r.cite_key = stored["ID"]
                    r.note = f"already in .bib as {stored['ID']}"
                    dbg.trace("suggest.commit", "duplicate", existing_key=stored["ID"])

                # Insert \cite{} into the .tex at the LLM-provided anchor.
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
                dbg.trace(
                    "suggest.commit",
                    "done",
                    cite_key=r.cite_key,
                    status=r.status,
                )
                report.results.append(r)

                await asyncio.sleep(delay_seconds)

    # Single .bib write at the end.
    if any(r.status in ("added", "duplicate") for r in report.results):
        bibtex.dump(db, bib_file)
        dbg.trace("suggest.done", "bib written", path=str(bib_file))

    return report


def suggest_for_file_sync(*args, **kwargs) -> SuggestReport:
    return asyncio.run(suggest_for_file(*args, **kwargs))
