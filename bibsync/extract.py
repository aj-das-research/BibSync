"""Extract papers referenced in a LaTeX file's `\\cite{...}` calls and add them to a .bib.

Workflow:
  1. Scan the .tex file for `\\cite{key}` uses + the paragraph they appear in.
  2. Group by key; for each unique key, ask the LLM to infer paper metadata from
     ``(key, surrounding_text)``.
  3. Query Scholar with the inferred title, pick the canonical version, fetch
     official BibTeX, append to the .bib file.
  4. Replace the LLM-inferred citation key with the official one if Scholar's
     differs — keeping the user's existing `\\cite{}` calls working.

A confidence floor is applied: low-confidence inferences are skipped and reported
so the user can resolve them manually.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from . import bibtex, llm, picker, scanner, scholar
from .models import PaperHit

DEFAULT_CONFIDENCE_FLOOR = 0.6


@dataclass
class ExtractResult:
    cite_key: str
    inferred: Optional[llm.InferredPaper] = None
    scholar_hit: Optional[PaperHit] = None
    bibtex_key: Optional[str] = None  # the key we ended up with in the .bib
    status: str = "pending"  # "added" | "duplicate" | "low_confidence" | "no_scholar_hit" | "error"
    note: str = ""


@dataclass
class ExtractReport:
    tex_file: Path
    bib_file: Path
    results: list[ExtractResult] = field(default_factory=list)

    def summary(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for r in self.results:
            out[r.status] = out.get(r.status, 0) + 1
        return out


def _gather_unique_keys(tex_file: Path) -> dict[str, scanner.CitationUse]:
    """Return ``{cite_key: first_use_with_context}`` for each unique key in the file."""
    out: dict[str, scanner.CitationUse] = {}
    # Walk a single file via the scanner's internal helper.
    for use in scanner._find_cite_uses_in_file(tex_file, with_context=True):
        if use.key not in out:
            out[use.key] = use
    return out


async def _resolve_one(
    cite_key: str,
    use: scanner.CitationUse,
    *,
    headless: bool,
    confidence_floor: float,
    openai_model: str,
    api_key: Optional[str],
) -> tuple[ExtractResult, Optional[dict]]:
    """Resolve one key. Returns (result, parsed_bibtex_entry_or_None)."""
    result = ExtractResult(cite_key=cite_key)

    try:
        inferred = llm.infer_paper_from_cite_key(
            cite_key, use.context, model=openai_model, api_key=api_key
        )
    except Exception as e:
        result.status = "error"
        result.note = f"llm inference failed: {e}"
        return result, None
    result.inferred = inferred

    if not inferred.title or inferred.confidence < confidence_floor:
        result.status = "low_confidence"
        result.note = (
            f"confidence {inferred.confidence:.2f} below floor {confidence_floor:.2f}; "
            f"title guess: {inferred.title!r}"
        )
        return result, None

    try:
        hits = await scholar.search(inferred.search_query(), headless=headless, max_results=5)
    except Exception as e:
        result.status = "error"
        result.note = f"scholar search failed: {e}"
        return result, None

    if not hits:
        result.status = "no_scholar_hit"
        result.note = f"no Scholar results for {inferred.search_query()!r}"
        return result, None

    # Expand top hit to all versions if available.
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

    entry = db.entries[0]
    # Try to preserve the user's existing citation key when reasonable — only swap
    # to Scholar's auto-generated key if the user's key looks placeholder-y.
    return result, entry


async def extract_from_file(
    tex_file: Path,
    bib_file: Path,
    *,
    only_missing: bool = True,
    headless: bool = False,
    confidence_floor: float = DEFAULT_CONFIDENCE_FLOOR,
    openai_model: str = "gpt-4o-mini",
    api_key: Optional[str] = None,
    delay_seconds: float = 1.5,
) -> ExtractReport:
    """Resolve every `\\cite{}` key in ``tex_file`` and append to ``bib_file``.

    If ``only_missing`` (default), keys already defined in the .bib are skipped.
    """
    report = ExtractReport(tex_file=tex_file, bib_file=bib_file)

    keys_with_context = _gather_unique_keys(tex_file)
    if not keys_with_context:
        return report

    db = bibtex.load(bib_file)
    existing_keys = {e.get("ID") for e in db.entries}

    resolved_entries: list[tuple[ExtractResult, dict]] = []

    for i, (cite_key, use) in enumerate(keys_with_context.items()):
        if only_missing and cite_key in existing_keys:
            r = ExtractResult(cite_key=cite_key, status="duplicate", note="already in .bib")
            report.results.append(r)
            continue

        r, entry = await _resolve_one(
            cite_key,
            use,
            headless=headless,
            confidence_floor=confidence_floor,
            openai_model=openai_model,
            api_key=api_key,
        )
        if entry is not None:
            # Force the BibTeX key to match the user's existing \cite{} so we don't
            # break the LaTeX source.
            entry["ID"] = cite_key
            resolved_entries.append((r, entry))
        report.results.append(r)

        if i < len(keys_with_context) - 1:
            await asyncio.sleep(delay_seconds)

    # Commit additions as a single write at the end.
    if resolved_entries:
        for r, entry in resolved_entries:
            stored, was_added = bibtex.append_entry(db, entry)
            r.bibtex_key = stored["ID"]
            r.status = "added" if was_added else "duplicate"
            if not was_added:
                r.note = f"fuzzy-matched existing entry {stored['ID']}"
        bibtex.dump(db, bib_file)

    return report


def extract_from_file_sync(*args, **kwargs) -> ExtractReport:
    return asyncio.run(extract_from_file(*args, **kwargs))
