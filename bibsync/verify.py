"""Verify existing .bib entries against Google Scholar.

Goal: detect LLM-hallucinated citations — entries whose title looks real but whose
year, venue, authors, or DOI don't match the actual paper.

We never overwrite without user approval. This module returns a structured report; the
CLI decides how to surface it.
"""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass, field
from typing import Optional

from rapidfuzz import fuzz

from . import scholar
from .models import PaperHit

# When the title fuzz ratio is below this, we treat the lookup as "no match found"
# rather than "entry is wrong" — could be a Scholar miss or an unindexed venue.
TITLE_FOUND_THRESHOLD = 80

# When the title matches but a field disagrees, we report it as a discrepancy.
YEAR_TOLERANCE = 1  # off-by-one is fine (preprint vs. proceedings date)


@dataclass
class FieldDiscrepancy:
    field: str
    bib_value: str
    scholar_value: str
    severity: str  # "minor" | "major"


@dataclass
class VerifyResult:
    entry_id: str
    bib_title: str
    status: str  # "verified" | "discrepancy" | "not_found" | "no_title"
    matched_hit: Optional[PaperHit] = None
    title_similarity: float = 0.0
    discrepancies: list[FieldDiscrepancy] = field(default_factory=list)
    note: str = ""


def _norm_title(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", s.lower())).strip()


def _norm_year(s: str) -> Optional[int]:
    m = re.search(r"(19|20)\d{2}", s or "")
    return int(m.group(0)) if m else None


def _compare_authors(bib_authors: str, hit_authors: list[str]) -> Optional[FieldDiscrepancy]:
    """Return a discrepancy if the first author's surname differs."""
    if not bib_authors or not hit_authors:
        return None
    # BibTeX author field: "Last1, First1 and Last2, First2" or "First1 Last1 and ..."
    first_bib = bib_authors.split(" and ")[0].strip()
    if "," in first_bib:
        bib_surname = first_bib.split(",")[0].strip()
    else:
        bib_surname = first_bib.split()[-1] if first_bib.split() else ""

    first_hit = hit_authors[0]
    hit_surname = first_hit.split()[-1] if first_hit.split() else ""

    if not bib_surname or not hit_surname:
        return None
    if fuzz.ratio(bib_surname.lower(), hit_surname.lower()) < 80:
        return FieldDiscrepancy(
            field="author",
            bib_value=bib_authors,
            scholar_value=" and ".join(hit_authors),
            severity="major",
        )
    return None


def _verify_one(entry: dict, hits: list[PaperHit]) -> VerifyResult:
    entry_id = entry.get("ID", "?")
    bib_title = entry.get("title", "")
    if not bib_title:
        return VerifyResult(
            entry_id=entry_id, bib_title="", status="no_title", note="entry has no title field"
        )

    norm = _norm_title(bib_title)
    best_hit: Optional[PaperHit] = None
    best_score = 0.0
    for h in hits:
        score = fuzz.ratio(norm, _norm_title(h.title))
        if score > best_score:
            best_score = score
            best_hit = h

    if best_hit is None or best_score < TITLE_FOUND_THRESHOLD:
        return VerifyResult(
            entry_id=entry_id,
            bib_title=bib_title,
            status="not_found",
            title_similarity=best_score,
            note="no Scholar hit with similar title in top results",
        )

    discrepancies: list[FieldDiscrepancy] = []

    bib_year = _norm_year(entry.get("year", ""))
    if bib_year and best_hit.year:
        if abs(bib_year - best_hit.year) > YEAR_TOLERANCE:
            discrepancies.append(
                FieldDiscrepancy(
                    field="year",
                    bib_value=str(bib_year),
                    scholar_value=str(best_hit.year),
                    severity="major",
                )
            )

    # Venue: a "minor" discrepancy if BibTeX has a venue and it disagrees.
    bib_venue = (
        entry.get("booktitle") or entry.get("journal") or entry.get("publisher") or ""
    ).strip()
    if bib_venue and best_hit.venue:
        if fuzz.partial_ratio(bib_venue.lower(), best_hit.venue.lower()) < 50:
            discrepancies.append(
                FieldDiscrepancy(
                    field="venue",
                    bib_value=bib_venue,
                    scholar_value=best_hit.venue,
                    severity="minor",
                )
            )

    author_disc = _compare_authors(entry.get("author", ""), best_hit.authors)
    if author_disc:
        discrepancies.append(author_disc)

    status = "verified" if not discrepancies else "discrepancy"
    return VerifyResult(
        entry_id=entry_id,
        bib_title=bib_title,
        status=status,
        matched_hit=best_hit,
        title_similarity=best_score,
        discrepancies=discrepancies,
    )


async def verify_entries(
    entries: list[dict], *, headless: bool = False, delay_seconds: float = 2.0
) -> list[VerifyResult]:
    """Verify a list of bibtex entries. Searches Scholar once per entry, sequentially,
    with a small delay to be polite (and avoid rate-limit triggers)."""
    results: list[VerifyResult] = []
    for i, entry in enumerate(entries):
        title = entry.get("title", "")
        if not title:
            results.append(_verify_one(entry, []))
            continue
        # Clean BibTeX braces from the search query.
        query = re.sub(r"[{}\\]", "", title)
        try:
            hits = await scholar.search(query, headless=headless, max_results=5)
        except Exception as e:
            results.append(
                VerifyResult(
                    entry_id=entry.get("ID", "?"),
                    bib_title=title,
                    status="not_found",
                    note=f"search failed: {e}",
                )
            )
            continue
        results.append(_verify_one(entry, hits))
        if i < len(entries) - 1:
            await asyncio.sleep(delay_seconds)
    return results


def verify_entries_sync(entries: list[dict], **kwargs) -> list[VerifyResult]:
    return asyncio.run(verify_entries(entries, **kwargs))
