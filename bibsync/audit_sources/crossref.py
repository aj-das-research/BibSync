"""Crossref REST API client.

Free, no auth (but supplying a User-Agent with a contact email puts you in the
"polite pool" with better latency). DOI-based metadata — useful for traditional
journal papers where the .bib entry has a DOI but no arXiv ID, and Semantic
Scholar doesn't have the abstract.

Abstracts come wrapped in JATS XML; we strip the tags before returning.
"""

from __future__ import annotations

import re
import urllib.parse
from typing import Optional

from .. import dbg
from .types import PaperContent

CROSSREF_API = "https://api.crossref.org/works"


async def search_crossref(
    title: str, doi: Optional[str] = None, *, timeout: float = 15.0
) -> Optional[PaperContent]:
    """Query Crossref by DOI (if provided) or by bibliographic title."""
    try:
        import httpx
    except ImportError:
        dbg.trace("audit.source.crossref", "httpx not installed; skipping Crossref lookup")
        return None

    if doi:
        url = f"{CROSSREF_API}/{urllib.parse.quote(doi, safe='/')}"
    else:
        params = {"query.bibliographic": title, "rows": "1"}
        url = f"{CROSSREF_API}?{urllib.parse.urlencode(params)}"

    dbg.trace("audit.source.crossref", "query", title=title, doi=doi)
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.get(
                url,
                headers={"User-Agent": "bibsync/0.1 (mailto:noreply@bibsync.dev)"},
            )
            resp.raise_for_status()
            data = resp.json()
    except Exception as e:
        dbg.trace("audit.source.crossref", "request failed", error=str(e))
        return None

    if doi:
        msg = data.get("message") or {}
    else:
        items = (data.get("message") or {}).get("items") or []
        if not items:
            dbg.trace("audit.source.crossref", "miss")
            return None
        msg = items[0]

    title_list = msg.get("title") or []
    title_out = title_list[0] if title_list else ""

    authors: list[str] = []
    for a in msg.get("author", []) or []:
        family = (a.get("family") or "").strip()
        given = (a.get("given") or "").strip()
        if family:
            authors.append(f"{given} {family}".strip())

    year: Optional[int] = None
    for key in ("published-print", "published-online", "issued"):
        parts = (msg.get(key) or {}).get("date-parts") or []
        if parts and parts[0]:
            try:
                year = int(parts[0][0])
                break
            except (ValueError, IndexError, TypeError):
                continue

    venue = None
    ct = msg.get("container-title") or []
    if ct:
        venue = ct[0]

    abstract = msg.get("abstract")
    if abstract:
        # Crossref returns abstracts wrapped in JATS XML; strip tags + collapse whitespace.
        abstract = re.sub(r"<[^>]+>", "", abstract)
        abstract = re.sub(r"\s+", " ", abstract).strip()

    result = PaperContent(
        title=title_out,
        abstract=abstract or None,
        authors=authors,
        year=year,
        venue=venue,
        doi=msg.get("DOI"),
        source="crossref",
    )
    dbg.trace(
        "audit.source.crossref",
        "hit",
        title=result.title,
        doi=result.doi,
        has_abstract=bool(result.abstract),
    )
    return result
