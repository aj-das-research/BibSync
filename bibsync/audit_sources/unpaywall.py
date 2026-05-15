"""Unpaywall DOI → open-access PDF URL resolver.

Unpaywall (https://unpaywall.org/products/api) maps a DOI to the best
known open-access PDF URL across publisher OA, repositories, and
preprint servers. Unlike arXiv / Semantic Scholar / OpenAlex / Crossref,
Unpaywall is NOT a paper-content source — it doesn't return titles,
abstracts, or authors. It's a single-purpose enrichment service for
"this paper has a DOI but the source I just fetched didn't give me a
PDF URL — does Unpaywall know where the OA copy lives?".

Used in the audit_sources fallback chain as a *post-merge* enrichment:
after the main sources have populated a PaperContent, if the record has
a DOI but no pdf_url, we query Unpaywall and set pdf_url. This unlocks
Tier-2 RAG for papers that would otherwise stall at Tier 1 (Nature /
IEEE / ACM papers whose publishers don't expose OA URLs but where a
green-OA repository copy exists).

Free, no auth — only requires a contact email parameter. Adding a real
mailto puts you in the polite pool.
"""

from __future__ import annotations

import urllib.parse
from typing import Optional

from .. import dbg

UNPAYWALL_API = "https://api.unpaywall.org/v2"
_POLITE_MAILTO = "bibsync@noreply.dev"


async def resolve_pdf_url(doi: str, *, timeout: float = 10.0) -> Optional[str]:
    """Look up ``doi`` on Unpaywall; return its best open-access PDF URL.

    Returns the URL string or ``None`` if:
      * the DOI is empty / malformed
      * httpx is missing
      * Unpaywall has no OA copy
      * the request fails for any reason

    Errors are traced rather than raised — Unpaywall is enrichment, not
    a required source, so a failure here just means "no Tier-2 for this
    paper this run".
    """
    if not doi:
        return None
    try:
        import httpx
    except ImportError:
        dbg.trace("audit.source.unpaywall", "httpx not installed; skipping")
        return None

    # Normalize: strip leading "https://doi.org/" if a URL was passed.
    if doi.startswith("https://doi.org/"):
        doi = doi[len("https://doi.org/"):]
    elif doi.startswith("doi:"):
        doi = doi[len("doi:"):]
    doi = doi.strip()

    url = f"{UNPAYWALL_API}/{urllib.parse.quote(doi, safe='/')}?email={urllib.parse.quote(_POLITE_MAILTO)}"
    dbg.trace("audit.source.unpaywall", "query", doi=doi)
    try:
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
            resp = await client.get(url, headers={"User-Agent": "bibsync/0.1"})
            if resp.status_code == 404:
                dbg.trace("audit.source.unpaywall", "miss (DOI not found)", doi=doi)
                return None
            resp.raise_for_status()
            data = resp.json()
    except Exception as e:
        dbg.trace(
            "audit.source.unpaywall",
            "request failed",
            error_type=type(e).__name__,
            error=str(e) or repr(e),
        )
        return None

    # Unpaywall's response structure:
    #   best_oa_location.url_for_pdf  — direct PDF link if available
    #   best_oa_location.url          — landing page (may be HTML, fall back)
    #   oa_locations[*]               — full list of OA copies; we walk if
    #                                   best_oa_location doesn't have a PDF
    best = data.get("best_oa_location") or {}
    pdf = best.get("url_for_pdf")
    if pdf:
        dbg.trace("audit.source.unpaywall", "hit best_oa_location.pdf", doi=doi)
        return pdf

    for loc in data.get("oa_locations") or []:
        pdf = loc.get("url_for_pdf")
        if pdf:
            dbg.trace(
                "audit.source.unpaywall",
                "hit oa_locations.pdf",
                doi=doi,
                host_type=loc.get("host_type"),
            )
            return pdf

    # No PDF, but a landing page might be a PDF if the publisher serves
    # one directly. We don't return non-PDF URLs — the audit pipeline's
    # downloader assumes PDF content.
    dbg.trace("audit.source.unpaywall", "miss (no OA PDF in response)", doi=doi)
    return None
