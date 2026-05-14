"""Semantic Scholar Academic Graph API client.

Free tier rate-limited to 100 req / 5 min without an API key. Broadest coverage
of any free academic search API (~200M papers including non-arXiv journals).
The ``abstract`` field is the key reason we use it — many arXiv preprints lack
abstracts in their Atom feed, but Semantic Scholar has them.
"""

from __future__ import annotations

import urllib.parse
from typing import Optional

from .. import dbg
from ._match import titles_match
from .types import PaperContent

SS_API = "https://api.semanticscholar.org/graph/v1/paper/search"
SS_FIELDS = "title,authors,year,venue,abstract,externalIds,openAccessPdf"


async def search_semantic_scholar(
    title: str,
    *,
    api_key: Optional[str] = None,
    timeout: float = 15.0,
) -> Optional[PaperContent]:
    """Query Semantic Scholar by title. Returns the top match or ``None``."""
    try:
        import httpx
    except ImportError:
        dbg.trace("audit.source.ss", "httpx not installed; skipping SS lookup")
        return None

    params = {"query": title, "limit": "1", "fields": SS_FIELDS}
    url = f"{SS_API}?{urllib.parse.urlencode(params)}"
    headers = {"User-Agent": "bibsync/0.1"}
    if api_key:
        headers["x-api-key"] = api_key

    dbg.trace("audit.source.ss", "query", title=title, has_api_key=bool(api_key))
    try:
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
            resp = await client.get(url, headers=headers)
            if resp.status_code == 429:
                dbg.trace("audit.source.ss", "rate limited")
                return None
            resp.raise_for_status()
            data = resp.json()
    except Exception as e:
        dbg.trace(
            "audit.source.ss",
            "request failed",
            error_type=type(e).__name__,
            error=str(e) or repr(e),
        )
        return None

    items = data.get("data") or []
    if not items:
        dbg.trace("audit.source.ss", "miss")
        return None
    item = items[0]

    authors = [a.get("name", "") for a in (item.get("authors") or []) if a.get("name")]
    ext = item.get("externalIds") or {}
    oapdf = item.get("openAccessPdf") or {}

    result = PaperContent(
        title=item.get("title", ""),
        abstract=item.get("abstract") or None,
        authors=authors,
        year=item.get("year"),
        venue=item.get("venue") or None,
        doi=ext.get("DOI"),
        arxiv_id=ext.get("ArXiv"),
        pdf_url=oapdf.get("url"),
        source="semantic_scholar",
    )
    if not titles_match(title, result.title, source="semantic_scholar"):
        return None
    dbg.trace(
        "audit.source.ss",
        "hit",
        title=result.title,
        doi=result.doi,
        arxiv=result.arxiv_id,
        has_pdf=bool(result.pdf_url),
    )
    return result
