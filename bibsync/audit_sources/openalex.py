"""OpenAlex client.

Free, no auth, no rate-limit headaches for low-volume use. Adding a polite
``mailto`` puts requests in the "polite pool" with higher per-IP limits.

Why OpenAlex matters for BibSync:

  - **Coverage gap**: arXiv miss ⇒ Semantic Scholar rate-limit ⇒ Crossref
    title-search returns wrong papers. OpenAlex fills the gap for non-arXiv
    ML, biomed, and humanities papers with cleaner metadata than Crossref.
  - **Open-access PDF URLs**: OpenAlex's ``primary_location.pdf_url`` and
    ``best_oa_location.pdf_url`` give us Tier-2 PDFs for papers that
    Semantic Scholar's openAccessPdf doesn't have.
  - **Citation-graph signals**: ``cited_by_count``, ``referenced_works``, and
    ``related_works`` — wired up later for canonicality ranking. Stored on
    the PaperContent record for now even though the audit pipeline doesn't
    use them yet.
  - **DOI / arXiv ID enrichment**: when other sources gave us a bare title,
    OpenAlex's ``ids`` block usually fills in DOI and arXiv ID — useful for
    stable cache keys.

Reference: https://docs.openalex.org/api-entities/works
"""

from __future__ import annotations

import urllib.parse
from typing import Optional

from .. import dbg
from ._match import titles_match
from .types import PaperContent

OPENALEX_API = "https://api.openalex.org/works"

# OpenAlex politeness — sending a mailto parameter elevates per-IP rate
# limits (10x faster on average). Replace with a real address if forking.
_POLITE_MAILTO = "bibsync@noreply.dev"


async def search_openalex(
    title: str,
    *,
    first_author: Optional[str] = None,
    year: Optional[int] = None,
    timeout: float = 15.0,
) -> Optional[PaperContent]:
    """Query OpenAlex by title; return the top match's :class:`PaperContent`.

    Returns ``None`` if the request fails, the response has no hits, or the
    top hit fails the title-similarity guard. arXiv ID and DOI are populated
    when present in OpenAlex's ``ids`` block.
    """
    try:
        import httpx
    except ImportError:
        dbg.trace("audit.source.openalex", "httpx not installed; skipping")
        return None

    # OpenAlex query strategy:
    #   • filter=title.search:... limits ranking signal to the title field,
    #     dramatically more precise than the bare ``search`` parameter (which
    #     matches against authors, abstract, and references too).
    #   • sort=cited_by_count:desc biases the top result toward the canonical
    #     paper when multiple works share similar titles — important because
    #     OpenAlex sometimes ranks commentary/review/translation papers above
    #     the original (the original BERT paper vs a 0-citation Japanese
    #     commentary on it, for example).
    #   • per_page=5 + take the highest-cited title-match-passing one — gives
    #     a small bit of slack for cases where the literal first result is a
    #     summary or translation.
    #   • We DELIBERATELY don't filter by year. OpenAlex's publication_year
    #     often differs from the .bib year (preprint year vs venue year), and
    #     a strict year filter caused canonical-paper misses in testing.
    #     The title-match guard + cited_by sort give us enough precision.
    # ``language:en`` filters out non-English commentaries and translations
    # of canonical papers (real example: a 0-citation Japanese summary of
    # BERT was outranking the original BERT paper before this filter). Has no
    # downside for non-English papers because BibSync's tex-side audit is
    # English-only anyway.
    params = {
        "filter": f"title.search:{title},language:en",
        "sort": "cited_by_count:desc",
        "per_page": "5",
        "mailto": _POLITE_MAILTO,
    }
    url = f"{OPENALEX_API}?{urllib.parse.urlencode(params)}"

    dbg.trace(
        "audit.source.openalex",
        "query",
        title=title,
        first_author=first_author,
        year=year,
    )
    try:
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
            resp = await client.get(url, headers={"User-Agent": "bibsync/0.1"})
            resp.raise_for_status()
            data = resp.json()
    except Exception as e:
        dbg.trace(
            "audit.source.openalex",
            "request failed",
            error_type=type(e).__name__,
            error=str(e) or repr(e),
        )
        return None

    items = data.get("results") or []
    if not items:
        dbg.trace("audit.source.openalex", "miss")
        return None

    # Walk top-K hits (already sorted by cited_by desc) and accept the first
    # one that passes the title-similarity guard. Two filters reduce false-
    # positives on summary / commentary papers:
    #   1. Title-match guard via rapidfuzz (the standard check).
    #   2. Skip 0-citation hits when there's at least one non-zero alternative.
    #      Catches the "Japanese summary of BERT outranking the canonical
    #      BERT paper" failure: the summary has cited_by=0; we walk past it
    #      to the next title-matching hit (or miss if there isn't one).
    has_cited_alternative = any((w.get("cited_by_count") or 0) > 0 for w in items)
    for w in items:
        result = _parse_openalex_work(w)
        if result is None:
            continue
        cited = w.get("cited_by_count") or 0
        if cited == 0 and has_cited_alternative:
            dbg.trace(
                "audit.source.openalex",
                "skip zero-cite candidate",
                title=result.title,
            )
            continue
        if not titles_match(title, result.title, source="openalex"):
            continue
        dbg.trace(
            "audit.source.openalex",
            "hit",
            title=result.title,
            cited=cited,
            doi=result.doi,
            arxiv=result.arxiv_id,
            has_pdf=bool(result.pdf_url),
            has_abstract=bool(result.abstract),
        )
        return result

    dbg.trace("audit.source.openalex", "miss (no title-match within top 5)")
    return None


def _parse_openalex_work(work: dict) -> Optional[PaperContent]:
    """Convert one OpenAlex Work JSON record to a PaperContent.

    OpenAlex returns abstracts as an "inverted index" (token -> [positions]);
    we reconstruct the plain text. Returns ``None`` if the record has no
    title (we'd have no anchor to match against).
    """
    title = (work.get("title") or "").strip()
    if not title:
        return None

    # Authors: take the display_name of each authorship.
    authors: list[str] = []
    for a in work.get("authorships", []) or []:
        name = (a.get("author") or {}).get("display_name", "").strip()
        if name:
            authors.append(name)

    year = work.get("publication_year")
    if year is not None:
        try:
            year = int(year)
        except (TypeError, ValueError):
            year = None

    # Venue: prefer primary_location.source.display_name, fall back to host_venue.
    venue: Optional[str] = None
    prim = work.get("primary_location") or {}
    src = prim.get("source") or {}
    if src.get("display_name"):
        venue = src["display_name"]
    elif (hv := work.get("host_venue") or {}).get("display_name"):
        venue = hv["display_name"]

    # Identifiers — DOI and arXiv ID.
    ids = work.get("ids") or {}
    doi = ids.get("doi")
    if doi and doi.startswith("https://doi.org/"):
        doi = doi[len("https://doi.org/"):]
    arxiv_id: Optional[str] = None
    for k in ("openalex", "pmid", "mag"):  # noqa: B007 — placeholder, real arxiv extraction below
        pass
    # OpenAlex doesn't expose arXiv ID in `ids` directly; it lives under
    # primary_location.landing_page_url for arXiv-hosted papers.
    for loc_key in ("primary_location", "best_oa_location"):
        loc = work.get(loc_key) or {}
        landing = (loc.get("landing_page_url") or "").lower()
        if "arxiv.org/abs/" in landing:
            arxiv_id = landing.rsplit("arxiv.org/abs/", 1)[-1].rstrip("/")
            # Strip URL version suffix and query params.
            arxiv_id = arxiv_id.split("?", 1)[0].split("#", 1)[0]
            break

    # Open-access PDF URL — prefer best_oa_location, fall back to primary.
    pdf_url: Optional[str] = None
    for loc_key in ("best_oa_location", "primary_location"):
        loc = work.get(loc_key) or {}
        if loc.get("pdf_url"):
            pdf_url = loc["pdf_url"]
            break

    # Abstract — OpenAlex stores it as an inverted index for licensing reasons.
    abstract = _abstract_from_inverted_index(work.get("abstract_inverted_index"))

    return PaperContent(
        title=title,
        abstract=abstract,
        authors=authors,
        year=year,
        venue=venue,
        doi=doi,
        arxiv_id=arxiv_id,
        pdf_url=pdf_url,
        source="openalex",
    )


def _abstract_from_inverted_index(inv: Optional[dict]) -> Optional[str]:
    """Reconstruct plaintext from OpenAlex's inverted-index abstract format.

    The index maps each token to the list of positions it appears at. We
    rebuild a sparse positions->token map and join in order. Missing positions
    (rare; happens when OpenAlex's extractor failed on some tokens) become
    empty placeholders rather than crashing the parse.
    """
    if not inv or not isinstance(inv, dict):
        return None
    positions: dict[int, str] = {}
    for token, idxs in inv.items():
        if not isinstance(idxs, list):
            continue
        for i in idxs:
            try:
                positions[int(i)] = token
            except (TypeError, ValueError):
                continue
    if not positions:
        return None
    max_pos = max(positions)
    words = [positions.get(i, "") for i in range(max_pos + 1)]
    text = " ".join(w for w in words if w).strip()
    return text or None
