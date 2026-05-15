"""Paper-content fetching for audit tiers 1+.

Public API:

    from bibsync.audit_sources import fetch_paper_content, PaperContent

    content = await fetch_paper_content(
        title="Attention is all you need",
        first_author="Vaswani",
        year=2017,
        cache=PaperContentCache(...),
    )
    if content:
        print(content.abstract)
        print(content.pdf_url)  # may be None if no open-access PDF

Fallback chain (first non-None wins; results are merged across sources):

    cache  →  arXiv  →  Semantic Scholar  →  OpenAlex  →  Crossref

arXiv is tried first because it has the broadest coverage of ML / CS preprints
and the cleanest API. Semantic Scholar fills in missing abstracts (arXiv often
has them, but not always) and provides ``openAccessPdf.url``. OpenAlex is the
third hop — it's free, fast, has 200M+ works, supplies cleaner metadata than
Crossref, and adds open-access PDF URLs for many non-arXiv papers. Crossref
remains the last-ditch backstop for traditional-journal papers.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from .. import dbg
from .arxiv import search_arxiv
from .cache import PaperContentCache
from .crossref import search_crossref
from .openalex import search_openalex
from .semantic_scholar import search_semantic_scholar
from .types import PaperContent
from .unpaywall import resolve_pdf_url as unpaywall_pdf_url

__all__ = [
    "PaperContent",
    "PaperContentCache",
    "fetch_paper_content",
]


def _merge(base: PaperContent, extra: PaperContent) -> PaperContent:
    """Merge ``extra`` into ``base`` — fill missing fields without overwriting.
    Source string keeps the first (arXiv-preferring) source; we don't track multi-source."""
    if not base.abstract and extra.abstract:
        base.abstract = extra.abstract
    if not base.doi and extra.doi:
        base.doi = extra.doi
    if not base.arxiv_id and extra.arxiv_id:
        base.arxiv_id = extra.arxiv_id
    if not base.pdf_url and extra.pdf_url:
        base.pdf_url = extra.pdf_url
    if not base.venue and extra.venue:
        base.venue = extra.venue
    if not base.year and extra.year:
        base.year = extra.year
    if not base.authors and extra.authors:
        base.authors = extra.authors
    return base


async def fetch_paper_content(
    title: str,
    *,
    first_author: Optional[str] = None,
    year: Optional[int] = None,
    doi: Optional[str] = None,
    cache: Optional[PaperContentCache] = None,
    no_cache: bool = False,
    ss_api_key: Optional[str] = None,
) -> Optional[PaperContent]:
    """Fetch paper content via the cache → arXiv → SS → Crossref fallback chain.

    Returns the merged :class:`PaperContent` (with the first hit's ``source`` tag)
    or ``None`` if every source missed.
    """
    # 1. Cache hit?
    if cache and not no_cache:
        cached = cache.get(title, year)
        if cached:
            cached.source = "cache"
            dbg.trace("audit.fetch", "cache hit", title=title)
            return cached

    result: Optional[PaperContent] = None

    # 2. arXiv (preferred for ML/CS)
    ax = await search_arxiv(title, first_author)
    if ax:
        result = ax

    # 3. Semantic Scholar — best for missing abstracts and openAccessPdf URLs.
    #    Skip if arXiv already gave us everything we need.
    needs_more = (not result) or (not result.abstract) or (not result.pdf_url)
    if needs_more:
        ss = await search_semantic_scholar(title, api_key=ss_api_key)
        if ss:
            result = _merge(result, ss) if result else ss

    # 4. OpenAlex — covers non-arXiv papers (Nature, IEEE, ACL Anthology, etc.)
    #    Free, no auth, 200M+ works. Adds open-access PDF URLs and citation-
    #    graph metadata that Crossref doesn't have. Run before Crossref because
    #    its title search is much stricter than Crossref's bibliographic search.
    needs_more = (not result) or (not result.abstract) or (not result.pdf_url)
    if needs_more:
        oa = await search_openalex(title, first_author=first_author, year=year)
        if oa:
            result = _merge(result, oa) if result else oa

    # 5. Crossref — last-ditch for traditional journals when a DOI is known
    #    (DOI-keyed lookups are exact; title-search lookups are noisy).
    needs_more = (not result) or (not result.abstract)
    if needs_more:
        cr = await search_crossref(title, doi=doi)
        if cr:
            result = _merge(result, cr) if result else cr

    # 6. Unpaywall PDF enrichment — DOI → open-access PDF URL. Runs AFTER all
    #    the content sources so it can use whichever DOI got resolved
    #    (Crossref-by-DOI, OpenAlex's `ids.doi`, etc.). Closes the Tier-2 gap
    #    for papers that have a known DOI but no PDF URL from the primary
    #    sources — common for Nature / IEEE / ACM papers where the publisher
    #    doesn't expose OA URLs but a green-OA repository copy exists.
    if result and result.doi and not result.pdf_url:
        oa_pdf = await unpaywall_pdf_url(result.doi)
        if oa_pdf:
            result.pdf_url = oa_pdf

    # Cache the merged result so subsequent runs skip the API hops.
    if result and cache:
        cache.put(result)

    if result:
        dbg.trace(
            "audit.fetch",
            "result",
            title=result.title,
            source=result.source,
            has_abstract=bool(result.abstract),
            has_pdf=bool(result.pdf_url),
            arxiv=result.arxiv_id,
            doi=result.doi,
        )
    else:
        dbg.trace("audit.fetch", "all sources missed", title=title)
    return result
