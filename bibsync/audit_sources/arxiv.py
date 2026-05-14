"""arXiv API client. Free, no auth required. Best coverage for ML/CS preprints
and the venue where most LLM-drafted citations land (arXiv preprints are the
single most-cited venue from LLM-generated bibliographies)."""

from __future__ import annotations

import re
import urllib.parse
from typing import Optional

from .. import dbg
from ._match import titles_match
from .types import PaperContent

# arXiv issues a 301 from http→https and httpx.AsyncClient does NOT follow
# redirects by default (unlike requests). Use https directly and pass
# follow_redirects=True as belt-and-suspenders against future redirect changes.
ARXIV_API = "https://export.arxiv.org/api/query"


async def search_arxiv(
    title: str, first_author: Optional[str] = None, *, timeout: float = 15.0
) -> Optional[PaperContent]:
    """Query arXiv by title (and optionally first author surname).

    Returns the first match's :class:`PaperContent` (with ``source='arxiv'``) or
    ``None`` if no match, the API errored, or ``httpx`` isn't installed.
    """
    try:
        import httpx
    except ImportError:
        dbg.trace("audit.source.arxiv", "httpx not installed; skipping arXiv lookup")
        return None

    parts = [f'ti:"{title}"']
    if first_author:
        parts.append(f"au:{first_author}")
    query = "+AND+".join(parts)
    url = f"{ARXIV_API}?search_query={urllib.parse.quote_plus(query)}&max_results=1"

    dbg.trace("audit.source.arxiv", "query", title=title, author=first_author)
    try:
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
            resp = await client.get(url, headers={"User-Agent": "bibsync/0.1"})
            resp.raise_for_status()
            xml = resp.text
    except Exception as e:
        # Some httpx errors (e.g. ReadTimeout) have empty str(e) but informative
        # repr(e); use the latter so the trace is actually debuggable.
        dbg.trace(
            "audit.source.arxiv",
            "request failed",
            error_type=type(e).__name__,
            error=str(e) or repr(e),
        )
        return None

    result = _parse_arxiv_atom(xml)
    if not result:
        dbg.trace("audit.source.arxiv", "miss")
        return None
    if not titles_match(title, result.title, source="arxiv"):
        # arXiv accepted the request but returned a different paper. Reject so
        # the orchestrator falls through to the next source.
        return None
    dbg.trace("audit.source.arxiv", "hit", title=result.title, arxiv_id=result.arxiv_id)
    return result


def _parse_arxiv_atom(xml: str) -> Optional[PaperContent]:
    """Parse the first <entry> from an arXiv Atom XML feed.

    We use loose regex rather than lxml/feedparser to avoid the extra dep.
    arXiv's Atom format is stable enough for this.
    """
    entry_m = re.search(r"<entry>(.*?)</entry>", xml, re.DOTALL)
    if not entry_m:
        return None
    entry = entry_m.group(1)

    def _grab(tag: str) -> Optional[str]:
        m = re.search(rf"<{tag}>(.*?)</{tag}>", entry, re.DOTALL)
        return m.group(1).strip() if m else None

    title = re.sub(r"\s+", " ", _grab("title") or "").strip()
    summary = re.sub(r"\s+", " ", _grab("summary") or "").strip()  # abstract

    authors: list[str] = []
    for am in re.finditer(r"<author>.*?<name>(.*?)</name>", entry, re.DOTALL):
        authors.append(am.group(1).strip())

    year: Optional[int] = None
    pub = _grab("published")
    if pub:
        y = re.search(r"(\d{4})", pub)
        if y:
            year = int(y.group(1))

    arxiv_id: Optional[str] = None
    pdf_url: Optional[str] = None
    arxiv_url = _grab("id")
    if arxiv_url:
        m = re.search(r"arxiv\.org/abs/([\w\.\-/]+?)(v\d+)?$", arxiv_url)
        if m:
            arxiv_id = m.group(1)
            pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"

    return PaperContent(
        title=title,
        abstract=summary or None,
        authors=authors,
        year=year,
        venue="arXiv",
        arxiv_id=arxiv_id,
        pdf_url=pdf_url,
        source="arxiv",
    )
