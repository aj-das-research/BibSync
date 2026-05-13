"""Repair legacy `\\bibitem{...}` bibliographies into proper BibTeX.

A `\\bibitem` block looks like:

    \\bibitem{tu2023medpalmm}
    T. Tu et al.
    Towards generalist biomedical AI.
    \\textit{arXiv:2307.14334}, 2023.
    \\url{https://arxiv.org/abs/2307.14334}

This module:
  1. Splits a .tex / .bbl file into individual bibitem blocks.
  2. Uses the LLM to extract structured metadata from each.
  3. Searches Scholar to find the real paper.
  4. Cross-checks the LLM's extracted year/author against Scholar (catches the case
     where the *original* bibitem was itself hallucinated).
  5. Emits BibTeX entries — preserving the original cite key so existing
     `\\cite{tu2023medpalmm}` references keep working.
"""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from rapidfuzz import fuzz

from . import bibtex, llm, picker, scholar
from .models import PaperHit

_BIBITEM_RE = re.compile(
    r"\\bibitem(?:\[[^\]]*\])?\s*\{([^}]+)\}(.*?)(?=\\bibitem(?:\[|\{)|\\end\{thebibliography\}|\Z)",
    re.DOTALL,
)


@dataclass
class RepairResult:
    cite_key: str
    parsed: Optional[llm.ParsedBibitem] = None
    scholar_hit: Optional[PaperHit] = None
    new_bibtex_entry: Optional[dict] = None
    status: str = "pending"  # "repaired" | "verified_match" | "discrepancy" | "no_scholar_hit" | "error"
    discrepancies: list[str] = field(default_factory=list)
    note: str = ""


@dataclass
class RepairReport:
    source: Path
    results: list[RepairResult] = field(default_factory=list)

    def summary(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for r in self.results:
            out[r.status] = out.get(r.status, 0) + 1
        return out


def split_bibitems(text: str) -> list[tuple[str, str]]:
    """Return ``[(cite_key, block_body), ...]`` parsed out of a file containing
    `\\bibitem{...}` entries."""
    return [(m.group(1).strip(), m.group(2).strip()) for m in _BIBITEM_RE.finditer(text)]


def _crosscheck(parsed: llm.ParsedBibitem, hit: PaperHit) -> list[str]:
    """Return a list of human-readable discrepancy notes between parsed bibitem and hit."""
    notes: list[str] = []
    if parsed.year and hit.year and abs(parsed.year - hit.year) > 1:
        notes.append(f"year: bibitem={parsed.year} vs scholar={hit.year}")
    if parsed.authors and hit.authors:
        # First author surname
        bib_first = parsed.authors[0]
        if "," in bib_first:
            bib_surname = bib_first.split(",")[0].strip()
        else:
            bib_surname = bib_first.split()[-1] if bib_first.split() else ""
        hit_surname = hit.authors[0].split()[-1] if hit.authors[0].split() else ""
        if bib_surname and hit_surname and fuzz.ratio(bib_surname.lower(), hit_surname.lower()) < 80:
            notes.append(f"first author: bibitem={bib_surname!r} vs scholar={hit_surname!r}")
    if parsed.title and hit.title:
        score = fuzz.ratio(
            re.sub(r"[^\w\s]", " ", parsed.title.lower()),
            re.sub(r"[^\w\s]", " ", hit.title.lower()),
        )
        if score < 70:
            notes.append(f"title: low similarity ({score:.0f}/100)")
    return notes


async def _repair_one(
    cite_key: str,
    block: str,
    *,
    headless: bool,
    openai_model: str,
    api_key: Optional[str],
) -> RepairResult:
    result = RepairResult(cite_key=cite_key)

    try:
        parsed = llm.parse_bibitem(
            block, cite_key=cite_key, model=openai_model, api_key=api_key
        )
    except Exception as e:
        result.status = "error"
        result.note = f"llm parse failed: {e}"
        return result
    result.parsed = parsed

    query = parsed.search_query()
    if not query:
        result.status = "error"
        result.note = "could not derive a search query from the bibitem"
        return result

    try:
        hits = await scholar.search(query, headless=headless, max_results=5)
    except Exception as e:
        result.status = "error"
        result.note = f"scholar search failed: {e}"
        return result

    if not hits:
        result.status = "no_scholar_hit"
        result.note = f"no Scholar results for {query!r}"
        return result

    # Pick best title match among returned hits (not just the top result — Scholar
    # ranking sometimes puts a less-relevant hit first).
    best, best_score = hits[0], 0.0
    for h in hits:
        s = fuzz.ratio(
            re.sub(r"[^\w\s]", " ", (parsed.title or "").lower()),
            re.sub(r"[^\w\s]", " ", h.title.lower()),
        )
        if s > best_score:
            best, best_score = h, s

    # Expand versions to pick canonical (e.g., arXiv → AAAI).
    candidates = [best]
    if best.versions_url:
        try:
            versions = await scholar.fetch_versions(best.versions_url, headless=headless)
            if versions:
                candidates = versions
        except Exception:
            pass
    # Add the other top Scholar hits as additional candidates for the LLM judge so
    # we recover gracefully when the title-fuzz best isn't the true match.
    for h in hits[:3]:
        if h not in candidates:
            candidates.append(h)

    # LLM-as-judge: verify against the metadata parsed from the original \bibitem block.
    expected = {
        "title": parsed.title or "",
        "author": " and ".join(parsed.authors[:3]) if parsed.authors else "",
        "year": str(parsed.year) if parsed.year else "",
    }
    verification = llm.pick_verified_match(
        expected,
        candidates,
        model=openai_model,
        api_key=api_key,
    )
    if verification.hit is None:
        result.status = "no_verified_match"
        result.scholar_hit = candidates[0] if candidates else None
        result.note = (
            f"LLM rejected Scholar top {verification.candidates_considered} "
            f"hit(s): {verification.reasoning}"
        )
        return result
    canonical = verification.hit
    result.scholar_hit = canonical

    # Cross-check what the original bibitem said vs what Scholar found (still useful as
    # a per-field diff for the report, even though the LLM has already endorsed identity).
    result.discrepancies = _crosscheck(parsed, canonical)

    if not canonical.cluster_id:
        result.status = "error"
        result.note = "canonical hit had no cluster id"
        return result

    try:
        bib_text = await scholar.fetch_bibtex_for_cluster(
            canonical.cluster_id, headless=headless
        )
    except Exception as e:
        result.status = "error"
        result.note = f"fetch bibtex failed: {e}"
        return result

    db = bibtex.parse_string(bib_text)
    if not db.entries:
        result.status = "error"
        result.note = "scholar returned bibtex but it failed to parse"
        return result

    new_entry = db.entries[0]
    new_entry["ID"] = cite_key  # preserve user's existing cite key
    result.new_bibtex_entry = new_entry

    result.status = "discrepancy" if result.discrepancies else "repaired"
    return result


async def repair_file(
    source: Path,
    *,
    bib_output: Optional[Path] = None,
    headless: bool = False,
    openai_model: str = "gpt-4o-mini",
    api_key: Optional[str] = None,
    delay_seconds: float = 1.5,
) -> RepairReport:
    """Repair every `\\bibitem{...}` in ``source``. Optionally append/merge results into
    ``bib_output``."""
    text = source.read_text(encoding="utf-8", errors="replace")
    blocks = split_bibitems(text)
    report = RepairReport(source=source)

    if not blocks:
        return report

    repaired_entries: list[dict] = []
    async with scholar.shared_session(headless=headless):
        for i, (key, body) in enumerate(blocks):
            r = await _repair_one(
                key,
                body,
                headless=headless,
                openai_model=openai_model,
                api_key=api_key,
            )
            report.results.append(r)
            if r.new_bibtex_entry is not None:
                repaired_entries.append(r.new_bibtex_entry)
            if i < len(blocks) - 1:
                await asyncio.sleep(delay_seconds)

    if bib_output is not None and repaired_entries:
        db = bibtex.load(bib_output)
        for entry in repaired_entries:
            bibtex.append_entry(db, entry)
        bibtex.dump(db, bib_output)

    return report


def repair_file_sync(*args, **kwargs) -> RepairReport:
    return asyncio.run(repair_file(*args, **kwargs))
