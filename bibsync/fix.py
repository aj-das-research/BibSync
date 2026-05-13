"""Fix a .bib file by re-fetching each entry from Google Scholar (matched by title)
and propagating any citation-key renames to .tex files in the project.

Workflow:
  1. Load all entries from the .bib.
  2. For each entry, search Scholar by title; pick the best title-similarity match.
  3. Fetch the canonical BibTeX for that match.
  4. Merge: replace the user's entry fields with Scholar's, BUT preserve the user's
     citation key by default (unless --regenerate-keys is set, in which case use the
     LLM-derived key built from author/year/title).
  5. After all entries are processed, atomically write the new .bib.
  6. If any keys changed, walk the project tree and rewrite \\cite{old} → \\cite{new}
     across every .tex file.

The verification step never deletes entries — if Scholar has no hit, the original entry
is kept untouched and reported as ``unverified``.
"""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from rapidfuzz import fuzz

from . import bibtex, llm, scholar, tex_rewrite
from .models import PaperHit

LLM_CONFIDENCE_FLOOR = 0.7  # LLM verdict must be at least this confident to accept a match
MAX_LLM_CANDIDATES = 3      # check at most this many top heuristic-plausible hits with the LLM


@dataclass
class FixResult:
    original_key: str
    new_key: str
    status: str  # "rewritten" | "key_renamed" | "unchanged" | "unverified" | "error"
    title_similarity: float = 0.0
    scholar_hit: Optional[PaperHit] = None
    field_changes: list[str] = field(default_factory=list)
    note: str = ""
    llm_reasoning: str = ""  # the LLM judge's reasoning (accepted or rejected)
    llm_confidence: float = 0.0


@dataclass
class FixReport:
    bib_file: Path
    project_root: Optional[Path]
    results: list[FixResult] = field(default_factory=list)
    tex_summary: Optional[tex_rewrite.TexRewriteSummary] = None

    def renames(self) -> dict[str, str]:
        return {r.original_key: r.new_key for r in self.results if r.original_key != r.new_key}

    def summary(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for r in self.results:
            out[r.status] = out.get(r.status, 0) + 1
        return out


def _normalize_title(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[{}\\]", "", s)).strip()


# Plausibility thresholds — used to reject Scholar hits whose author/year disagree
# with the original .bib entry even if the title matches. Tuned to allow legitimate
# small drift (arXiv preprint year vs. proceedings year, author-name abbreviation)
# while rejecting obvious different-paper matches.
_PLAUSIBLE_AUTHOR_FUZZ = 70   # 0-100; below this, surnames disagree
_PLAUSIBLE_YEAR_TOLERANCE = 3  # years; arXiv→conference→journal lag is at most ~2 years


def _first_author_surname(author_field: str) -> str:
    """Extract first author's surname from a BibTeX-style author field.

    Handles both "Last, First and Last2, First2" and "First Last and First2 Last2".
    """
    if not author_field:
        return ""
    first = author_field.split(" and ")[0].strip()
    if "," in first:
        return first.split(",")[0].strip()
    parts = first.split()
    return parts[-1] if parts else ""


def _bib_year(entry: dict) -> Optional[int]:
    m = re.search(r"\d{4}", entry.get("year") or "")
    return int(m.group(0)) if m else None


def _is_plausible_match(entry: dict, hit: PaperHit) -> tuple[bool, str]:
    """Sanity-check that a Scholar hit could plausibly be the same paper as the .bib entry.

    Returns ``(True, "")`` if plausible, ``(False, reason)`` otherwise.

    Title similarity alone is NOT enough — different papers can share titles. We require
    that whatever metadata both sides have (author surname, year) is consistent.
    """
    bib_surname = _first_author_surname(entry.get("author", ""))
    hit_surname = ""
    if hit.authors:
        hit_first = hit.authors[0]
        parts = hit_first.split()
        hit_surname = parts[-1] if parts else ""

    if bib_surname and hit_surname:
        surname_score = fuzz.ratio(bib_surname.lower(), hit_surname.lower())
        if surname_score < _PLAUSIBLE_AUTHOR_FUZZ:
            return (
                False,
                f"first author mismatch ({bib_surname!r} vs Scholar {hit_surname!r})",
            )

    bib_year = _bib_year(entry)
    if bib_year and hit.year:
        if abs(bib_year - hit.year) > _PLAUSIBLE_YEAR_TOLERANCE:
            return (
                False,
                f"year off by {abs(bib_year - hit.year)} ({bib_year} vs Scholar {hit.year})",
            )

    return True, ""


def _build_refined_query(entry: dict) -> Optional[str]:
    """Combine title + first-author surname + year for a tighter Scholar search."""
    title = re.sub(r"[{}\\]", "", entry.get("title", "")).strip()
    if not title:
        return None
    parts = [title]
    surname = _first_author_surname(entry.get("author", ""))
    if surname:
        parts.append(surname)
    year = _bib_year(entry)
    if year:
        parts.append(str(year))
    return " ".join(parts)


def _merge_entry(original: dict, scholar_entry: dict, preserve_key: bool) -> tuple[dict, list[str]]:
    """Return (merged_entry, list_of_field_change_descriptions)."""
    changes: list[str] = []
    merged = dict(original)

    # Scholar's BibTeX gives us ENTRYTYPE, ID, title, author, year, booktitle/journal, publisher.
    # We replace these in the original, but preserve the cite ID (if requested).
    for field_name in ("title", "author", "year", "booktitle", "journal", "publisher", "volume", "number", "pages", "doi"):
        old_val = (original.get(field_name) or "").strip()
        new_val = (scholar_entry.get(field_name) or "").strip()
        if new_val and new_val != old_val:
            merged[field_name] = new_val
            preview_old = (old_val[:40] + "…") if len(old_val) > 40 else old_val
            preview_new = (new_val[:40] + "…") if len(new_val) > 40 else new_val
            changes.append(f"{field_name}: {preview_old!r} → {preview_new!r}")

    if (scholar_entry.get("ENTRYTYPE") and
        scholar_entry["ENTRYTYPE"] != original.get("ENTRYTYPE")):
        old_t = original.get("ENTRYTYPE") or "?"
        merged["ENTRYTYPE"] = scholar_entry["ENTRYTYPE"]
        changes.append(f"type: @{old_t} → @{scholar_entry['ENTRYTYPE']}")

    if not preserve_key:
        # The caller will set the new ID separately.
        pass
    else:
        merged["ID"] = original["ID"]
    return merged, changes


@dataclass
class _PickResult:
    hit: Optional[PaperHit]
    title_similarity: float
    rejection_reason: str
    llm_reasoning: str = ""
    llm_confidence: float = 0.0


async def _search_and_pick(
    entry: dict,
    query: str,
    *,
    headless: bool,
    model: Optional[str],
    api_key: Optional[str],
) -> _PickResult:
    """Run a Scholar search, filter heuristically, then ask the LLM to verify identity.

    Pipeline (each layer eliminates non-matches before the next, cheaper-to-most-expensive):
      1. Scholar search → up to 10 candidate hits.
      2. Cheap heuristic filter: title sim ≥ 60 AND author/year plausibility.
      3. LLM-as-judge verdict on up to MAX_LLM_CANDIDATES top candidates.
      4. Accept the first candidate the LLM marks ``same_paper=True`` with confidence
         ≥ LLM_CONFIDENCE_FLOOR. Otherwise return ``hit=None``.
    """
    try:
        hits = await scholar.search(query, headless=headless, max_results=10)
    except Exception as e:
        return _PickResult(None, 0.0, f"scholar search failed: {e}")
    if not hits:
        return _PickResult(None, 0.0, "no Scholar hits")

    norm = _normalize_title(entry.get("title", "")).lower()

    # Heuristic filter — title similarity + author/year plausibility.
    scored: list[tuple[float, PaperHit, tuple[bool, str]]] = []
    for h in hits:
        title_score = fuzz.ratio(norm, _normalize_title(h.title).lower())
        plausible = _is_plausible_match(entry, h)
        scored.append((title_score, h, plausible))

    plausible_hits = [(s, h) for (s, h, (ok, _)) in scored if ok and s >= 60]
    if not plausible_hits:
        scored.sort(key=lambda t: t[0], reverse=True)
        top_score, _, (_, reason) = scored[0]
        return _PickResult(None, top_score, reason or f"title sim only {top_score:.0f}%")

    plausible_hits.sort(key=lambda t: t[0], reverse=True)

    # LLM verification — the safety net that catches title-similar-but-different papers.
    # Delegated to the universal pick_verified_match agent so fix shares one verification
    # path with extract, repair, and add.
    pick = llm.pick_verified_match(
        entry,
        [h for _, h in plausible_hits],
        confidence_floor=LLM_CONFIDENCE_FLOOR,
        max_candidates=MAX_LLM_CANDIDATES,
        model=model,
        api_key=api_key,
    )
    if pick.hit is not None:
        # Find the title score for the accepted hit.
        accepted_score = next(s for s, h in plausible_hits if h is pick.hit)
        return _PickResult(
            pick.hit,
            accepted_score,
            "",
            llm_reasoning=pick.reasoning,
            llm_confidence=pick.confidence,
        )

    best_score = plausible_hits[0][0]
    return _PickResult(
        None,
        best_score,
        f"LLM rejected all {pick.candidates_considered} top candidate(s)",
        llm_reasoning=pick.reasoning,
        llm_confidence=pick.confidence,
    )


async def _fix_one(
    entry: dict,
    *,
    headless: bool,
    regenerate_keys: bool,
    used_keys: set[str],
    model: Optional[str] = None,
    api_key: Optional[str] = None,
) -> tuple[FixResult, Optional[dict]]:
    """Returns (FixResult, merged_entry_or_None). If merged_entry is None, keep the original."""
    original_key = entry.get("ID", "?")
    title = entry.get("title", "")
    result = FixResult(original_key=original_key, new_key=original_key, status="pending")

    if not title:
        result.status = "unverified"
        result.note = "entry has no title field"
        return result, None

    base_query = re.sub(r"[{}\\]", "", title)
    pick = await _search_and_pick(
        entry, base_query, headless=headless, model=model, api_key=api_key
    )

    # If title-only search yielded no LLM-verified match, retry with refined query.
    if pick.hit is None:
        refined = _build_refined_query(entry)
        if refined and refined != base_query:
            pick2 = await _search_and_pick(
                entry, refined, headless=headless, model=model, api_key=api_key
            )
            if pick2.hit is not None:
                pick = pick2
            else:
                # Keep the better diagnostic between the two attempts.
                pick = pick2 if pick2.rejection_reason else pick

    result.title_similarity = pick.title_similarity
    result.llm_reasoning = pick.llm_reasoning
    result.llm_confidence = pick.llm_confidence

    if pick.hit is None:
        result.status = (
            "error" if pick.rejection_reason.startswith("scholar search failed") else "unverified"
        )
        result.note = pick.rejection_reason or "no plausible Scholar match"
        return result, None

    best = pick.hit
    result.scholar_hit = best

    if not best.cluster_id:
        result.status = "error"
        result.note = "best hit had no cluster id"
        return result, None

    try:
        bib_text = await scholar.fetch_bibtex_for_cluster(
            best.cluster_id, headless=headless
        )
    except Exception as e:
        result.status = "error"
        result.note = f"fetch bibtex failed: {e}"
        return result, None

    db = bibtex.parse_string(bib_text)
    if not db.entries:
        result.status = "error"
        result.note = "scholar bibtex failed to parse"
        return result, None
    scholar_entry = db.entries[0]

    merged, changes = _merge_entry(entry, scholar_entry, preserve_key=not regenerate_keys)
    result.field_changes = changes

    if regenerate_keys:
        new_key = bibtex.derive_cite_key(merged)
        # ensure uniqueness w.r.t. keys we've already assigned this run
        candidate = new_key
        suffix = 0
        while candidate in used_keys and candidate != original_key:
            suffix += 1
            candidate = f"{new_key}{chr(ord('a') + suffix - 1)}"
        merged["ID"] = candidate
        result.new_key = candidate
        used_keys.add(candidate)
    else:
        used_keys.add(original_key)

    if not changes and result.new_key == result.original_key:
        result.status = "unchanged"
    elif result.new_key != result.original_key:
        result.status = "key_renamed"
    else:
        result.status = "rewritten"

    return result, merged


async def fix_bib(
    bib_file: Path,
    *,
    project_root: Optional[Path] = None,
    headless: bool = False,
    regenerate_keys: bool = True,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    delay_seconds: float = 3.0,
) -> FixReport:
    """Verify and rewrite each entry in ``bib_file``.

    The pipeline for each entry:
      1. Scholar search by title.
      2. Heuristic plausibility filter (author surname + year tolerance).
      3. LLM-as-judge verification — only accept matches the LLM confirms are the same
         paper with confidence ≥ ``LLM_CONFIDENCE_FLOOR``.
      4. If accepted, merge Scholar's fields into the entry. With ``regenerate_keys``,
         build a new cite key from the corrected author+year+title and propagate any
         rename to .tex files under ``project_root``.

    Entries that fail any layer are reported as ``unverified`` and left untouched —
    a deliberate "never silently corrupt" stance.
    """
    db = bibtex.load(bib_file)
    report = FixReport(bib_file=bib_file, project_root=project_root)

    if not db.entries:
        return report

    used_keys: set[str] = set()
    new_entries: list[dict] = []

    # Pool all Scholar calls for this run into a single browser context so we don't
    # trigger Scholar's anti-bot heuristics on the 8th-12th launch.
    soft_block_warned = False
    async with scholar.shared_session(headless=headless):
        for i, entry in enumerate(db.entries):
            result, merged = await _fix_one(
                entry,
                headless=headless,
                regenerate_keys=regenerate_keys,
                used_keys=used_keys,
                model=model,
                api_key=api_key,
            )
            report.results.append(result)
            new_entries.append(merged if merged is not None else entry)

            # If we see 3 consecutive empty Scholar responses, the IP/profile is almost
            # certainly soft-blocked. Print a one-time warning and keep going (so the user
            # can see the full unverified report rather than aborting mid-way).
            if not soft_block_warned and scholar.consecutive_empty_count() >= 3:
                print(
                    "\n[bibsync] Detected 3+ consecutive empty Scholar responses — "
                    "your IP/profile is almost certainly soft-blocked by Google Scholar.\n"
                    "[bibsync] Fix: run `bibsync config reset-profile` and re-run, "
                    "or wait ~30 minutes for the rate-limit window to clear.\n"
                )
                soft_block_warned = True

            if i < len(db.entries) - 1:
                await asyncio.sleep(delay_seconds)

    # Skip the dump entirely on a no-op run. Otherwise BibTexWriter reorders the file
    # alphabetically and the user sees diff noise even when nothing semantically changed.
    any_changed = any(
        r.status in ("rewritten", "key_renamed") for r in report.results
    )
    if any_changed:
        db.entries = new_entries
        bibtex.dump(db, bib_file)

    renames = report.renames()
    if project_root is not None and renames:
        report.tex_summary = tex_rewrite.rename_keys_in_project(project_root, renames)

    return report


def fix_bib_sync(*args, **kwargs) -> FixReport:
    return asyncio.run(fix_bib(*args, **kwargs))
