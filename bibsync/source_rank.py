"""``bibsync source-rank "title or claim"`` — return ranked canonical candidates.

Different from ``evidence``: this command doesn't extract supporting/
contradicting quotes. It just ranks the candidate papers themselves by
canonicality signals so the user can pick "the right paper to cite".

Ranking signals (combined linearly):

  0.40 × cited_by_norm   — log-scaled citation count (the canonicality
                            signal; the original BERT paper has ~80K cites
                            while derivatives have ≤5K)
  0.30 × is_canonical    — whether the paper title pattern matches
                            "introduced X", "X paper", "first / original",
                            or matches the LLM's identified expected_title
  0.20 × venue_prior     — top-tier venue bonus (NeurIPS, ICML, ICLR, ACL,
                            EMNLP, CVPR, ICCV, ECCV, Nature, Science,
                            arXiv preprint)
  0.10 × recency         — newer publication bonus, capped (avoids
                            promoting fresh-but-derivative work)

  - 0.50 × is_survey     — strong negative signal for survey/review papers
                            (matches the C1 prompt rule's intent)

The output is JSON-serialisable so the future server endpoint
POST /source-rank can return it directly.
"""

from __future__ import annotations

import asyncio
import math
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

from . import dbg


# Top-tier venue allowlist for the venue_prior signal. Substring-matched
# (case-insensitive) against the OpenAlex venue display_name.
_TOP_VENUES = (
    "NeurIPS", "Neural Information Processing Systems",
    "ICML", "International Conference on Machine Learning",
    "ICLR", "International Conference on Learning Representations",
    "ACL", "Association for Computational Linguistics",
    "EMNLP",
    "NAACL",
    "CVPR", "Computer Vision and Pattern Recognition",
    "ICCV", "International Conference on Computer Vision",
    "ECCV", "European Conference on Computer Vision",
    "Nature", "Science",
    "PNAS", "Proceedings of the National Academy",
    "arXiv",
)

# Title patterns that indicate a survey / review / tutorial paper. Used as
# a NEGATIVE signal in source-rank (mirrors the C1 prompt rule).
_SURVEY_TITLE_RE = re.compile(
    r"\b(survey|review|overview|tutorial|comprehensive (?:analysis|evaluation)|"
    r"what does .* learn|where it comes and where it goes|revisiting|probing)\b",
    re.IGNORECASE,
)


@dataclass
class RankedCandidate:
    """One ranked candidate paper with its score breakdown."""

    rank: int = 0
    paper_key: str = ""
    title: str = ""
    first_author: str = ""
    year: Optional[int] = None
    venue: str = ""
    doi: str = ""
    arxiv_id: str = ""
    cited_by: int = 0
    score: float = 0.0
    score_breakdown: dict = field(default_factory=dict)  # signal → contribution
    is_survey: bool = False
    notes: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class SourceRankReport:
    query: str
    candidates: list = field(default_factory=list)  # list[RankedCandidate]
    elapsed_sec: float = 0.0

    def to_dict(self) -> dict:
        return {
            "query": self.query,
            "candidates": [c.to_dict() for c in self.candidates],
            "elapsed_sec": self.elapsed_sec,
        }


# ── scoring ────────────────────────────────────────────────────────────────


def _cited_by_norm(cited: int) -> float:
    """Log-scale the citation count to [0, 1]. log10(50K cites) ≈ 4.7 →
    cap at log10(10^5) = 5 for normalisation."""
    if cited <= 0:
        return 0.0
    return min(math.log10(cited + 1) / 5.0, 1.0)


def _venue_prior(venue: str) -> float:
    """1.0 if the venue is a top-tier ML / CS / Nature / arXiv venue,
    0.5 if it's reputable-but-unrecognised, 0.0 if no venue."""
    if not venue:
        return 0.0
    for top in _TOP_VENUES:
        if top.lower() in venue.lower():
            return 1.0
    return 0.5


def _recency_score(year: Optional[int]) -> float:
    """Newer papers get a small bonus, capped to avoid promoting derivatives.
    Year < 2010 → 0.0, 2010 → 0.2, 2020 → 0.7, 2024+ → 1.0."""
    if not year:
        return 0.0
    if year < 2010:
        return 0.0
    return min((year - 2010) / 14.0, 1.0)


def _is_canonical_title(title: str, expected_title: str) -> float:
    """1.0 when the title fuzzy-matches the LLM-identified expected title,
    0.0 otherwise. Uses the same rapidfuzz threshold (75) as the existing
    title-match guard."""
    if not expected_title or not title:
        return 0.0
    try:
        from rapidfuzz import fuzz
        score = fuzz.token_sort_ratio(title.lower(), expected_title.lower())
        return 1.0 if score >= 75 else 0.0
    except ImportError:
        return 0.0


def _score_candidate(
    *, cited: int, venue: str, year: Optional[int], title: str,
    expected_title: str,
) -> tuple[float, dict, bool]:
    """Combine signals into a single score in [-0.5, +1.0]. Returns
    (score, breakdown_dict, is_survey_flag)."""
    cb = _cited_by_norm(cited)
    canon = _is_canonical_title(title, expected_title)
    venue_p = _venue_prior(venue)
    recency = _recency_score(year)
    is_survey = bool(_SURVEY_TITLE_RE.search(title or ""))

    score = (0.40 * cb) + (0.30 * canon) + (0.20 * venue_p) + (0.10 * recency)
    if is_survey:
        score -= 0.5
    breakdown = {
        "cited_by_norm": round(0.40 * cb, 3),
        "is_canonical": round(0.30 * canon, 3),
        "venue_prior": round(0.20 * venue_p, 3),
        "recency": round(0.10 * recency, 3),
        "survey_penalty": -0.5 if is_survey else 0.0,
    }
    return score, breakdown, is_survey


# ── retrieval ───────────────────────────────────────────────────────────────


async def _fetch_candidates(
    query: str, *, top_papers: int, api_key: Optional[str] = None,
    timeout: float = 15.0,
) -> tuple[list[dict], str]:
    """Two-stage retrieval (mirrors evidence_cmd):

      1. LLM identifies expected title from world knowledge.
      2. OpenAlex title.search + cited_by_count:desc (titles found by
         the LLM) UNION broader search= over keywords.

    Returns (candidates, expected_title). The expected_title is used
    by the scoring layer.
    """
    try:
        import httpx
    except ImportError:
        return [], ""
    import urllib.parse

    from .llm import identify_canonical_paper
    ident = identify_canonical_paper(
        claim=query, paragraph=query, document_context="", api_key=api_key,
    )
    expected_title = ident.expected_title if ident.confidence >= 0.4 else ""

    queries: list[tuple[str, dict]] = []
    if expected_title:
        queries.append((
            "title.search",
            {
                "filter": f"title.search:{expected_title},language:en",
                "sort": "cited_by_count:desc",
                "per_page": str(top_papers * 2),
                "mailto": "bibsync@noreply.dev",
            },
        ))
    keywords = re.findall(r"[A-Z][A-Za-z0-9-]+|\b[a-z]{4,}\b", query)[:6]
    if keywords:
        queries.append((
            "search-keywords",
            {
                "search": " ".join(keywords),
                "filter": "language:en",
                "sort": "cited_by_count:desc",
                "per_page": str(top_papers * 2),
                "mailto": "bibsync@noreply.dev",
            },
        ))

    results: list[dict] = []
    seen_ids: set[str] = set()
    for label, params in queries:
        url = "https://api.openalex.org/works?" + urllib.parse.urlencode(params)
        dbg.trace("source_rank.search", "openalex", strategy=label,
                  expected_title=expected_title)
        try:
            async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as c:
                r = await c.get(url, headers={"User-Agent": "bibsync/0.1"})
                r.raise_for_status()
                data = r.json()
        except Exception as e:
            dbg.trace("source_rank.search", "request failed",
                      error_type=type(e).__name__, error=str(e) or repr(e))
            continue
        for w in data.get("results") or []:
            wid = w.get("id") or w.get("title", "")
            if wid and wid not in seen_ids:
                seen_ids.add(wid)
                results.append(w)
                if len(results) >= top_papers * 2:
                    return results, expected_title
    return results[:top_papers * 2], expected_title


# ── public entry ────────────────────────────────────────────────────────────


async def rank_sources(
    query: str,
    *,
    top_papers: int = 5,
    api_key: Optional[str] = None,
) -> SourceRankReport:
    """Search OpenAlex for candidate papers matching ``query`` (a claim or
    a title), score them by combined canonicality signals, return ranked."""
    import time
    from .audit_sources.openalex import _parse_openalex_work

    t0 = time.monotonic()
    raw, expected_title = await _fetch_candidates(
        query, top_papers=top_papers, api_key=api_key,
    )
    report = SourceRankReport(query=query)

    scored: list[tuple[float, RankedCandidate]] = []
    for w in raw:
        pc = _parse_openalex_work(w)
        if pc is None:
            continue
        cited = int(w.get("cited_by_count") or 0)
        score, breakdown, is_survey = _score_candidate(
            cited=cited, venue=pc.venue or "", year=pc.year,
            title=pc.title, expected_title=expected_title,
        )
        scored.append((score, RankedCandidate(
            paper_key=pc.stable_key(),
            title=pc.title,
            first_author=(pc.authors[0] if pc.authors else ""),
            year=pc.year,
            venue=pc.venue or "",
            doi=pc.doi or "",
            arxiv_id=pc.arxiv_id or "",
            cited_by=cited,
            score=round(score, 3),
            score_breakdown=breakdown,
            is_survey=is_survey,
        )))

    scored.sort(key=lambda t: t[0], reverse=True)
    for i, (_, c) in enumerate(scored[:top_papers], 1):
        c.rank = i
        report.candidates.append(c)

    report.elapsed_sec = time.monotonic() - t0
    return report


def rank_sources_sync(*args, **kwargs) -> SourceRankReport:
    return asyncio.run(rank_sources(*args, **kwargs))
