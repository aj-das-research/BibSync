"""``bibsync evidence "claim"`` — return supporting / contradicting evidence
for a free-form claim, without needing a pre-existing ``\\cite{}``.

Pipeline:

  user claim
    → search OpenAlex for top-K candidate papers (title.search,
      sorted by cited_by descending — biases toward canonical sources)
    → fetch each via the standard source-fallback chain
      (arXiv → SS → OpenAlex → Crossref → Unpaywall PDF)
    → for each candidate that resolved with a PDF, run the same hybrid
      RAG retrieval audit uses (BM25 + dense + RRF + cross-encoder)
    → for each top-K chunk per candidate, compress to a 1-3-sentence
      EvidenceSpan
    → group spans by candidate paper, return ranked output

The command is the read-only twin of ``suggest``: it surfaces evidence
without modifying any file. Used by the future server's ``POST /evidence``
endpoint and the Chrome extension's "Find citation" flow.

Cost: 1 OpenAlex API call + N paper fetches (cached) + N PDF downloads
(cached) + N embedding calls (cached). After first run, repeats are
~free.
"""

from __future__ import annotations

import asyncio
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

from . import dbg


@dataclass
class EvidenceCandidate:
    """One candidate paper + its retrieved evidence spans for the claim."""

    paper_key: str = ""
    title: str = ""
    first_author: str = ""
    year: Optional[int] = None
    venue: str = ""
    doi: str = ""
    arxiv_id: str = ""
    pdf_url: str = ""
    cited_by: int = 0
    source: str = ""           # which adapter produced this candidate
    evidence_tier: int = 0     # 0 = metadata, 1 = abstract, 2 = chunks
    has_abstract: bool = False
    has_pdf: bool = False
    spans: list = field(default_factory=list)  # list[dict] of EvidenceSpan
    note: str = ""             # error / status detail when no spans extracted

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class EvidenceReport:
    claim: str
    candidates: list = field(default_factory=list)  # list[EvidenceCandidate]
    elapsed_sec: float = 0.0

    def to_dict(self) -> dict:
        return {
            "claim": self.claim,
            "candidates": [c.to_dict() for c in self.candidates],
            "elapsed_sec": self.elapsed_sec,
        }


# ── core ────────────────────────────────────────────────────────────────────


async def _candidates_for_claim(
    claim: str, *, top_papers: int,
    api_key: Optional[str] = None, timeout: float = 15.0,
) -> list[dict]:
    """Find candidate papers for a free-form claim.

    Two-stage retrieval:

      1. Ask the LLM (via ``identify_canonical_paper``) to NAME the most
         likely canonical paper from world knowledge. This is one cheap
         LLM call but it dramatically narrows the search — Vaswani 2017
         for "the Transformer introduced self-attention" is a fact in
         the LLM's training data; we just need it to surface that.

      2. Run an OpenAlex ``title.search`` against the identified title.
         Title-search is precise (unlike full-text search which dilutes
         on common terms). cited_by_count:desc biases toward the
         canonical version when there are paper-with-paper title hits.

    Falls back to broad ``search=`` over keywords when the LLM step
    returns low confidence. Worst case: returns 0 candidates and the
    user sees "No candidate papers found."
    """
    try:
        import httpx
    except ImportError:
        return []
    import urllib.parse
    import re

    # Stage 1 — LLM identifies the canonical paper from world knowledge.
    from .llm import identify_canonical_paper
    ident = identify_canonical_paper(
        claim=claim, paragraph=claim, document_context="", api_key=api_key,
    )
    queries: list[tuple[str, dict]] = []
    if ident.confidence >= 0.4 and ident.expected_title:
        # Title-search against the LLM-identified title.
        queries.append((
            f"title:{ident.expected_title}",
            {
                "filter": f"title.search:{ident.expected_title},language:en",
                "sort": "cited_by_count:desc",
                "per_page": str(top_papers * 2),
                "mailto": "bibsync@noreply.dev",
            },
        ))
    # Stage 2 — broad keyword search as fallback / supplement.
    keywords = re.findall(r"[A-Z][A-Za-z0-9-]+|\b[a-z]{4,}\b", claim)[:6]
    if keywords:
        queries.append((
            f"keywords:{' '.join(keywords)}",
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
        dbg.trace("audit.evidence.search", "openalex", strategy=label)
        try:
            async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as c:
                r = await c.get(url, headers={"User-Agent": "bibsync/0.1"})
                r.raise_for_status()
                data = r.json()
        except Exception as e:
            dbg.trace("audit.evidence.search", "request failed",
                      error_type=type(e).__name__, error=str(e) or repr(e))
            continue
        for w in data.get("results") or []:
            wid = w.get("id") or w.get("title", "")
            if wid and wid not in seen_ids:
                seen_ids.add(wid)
                results.append(w)
                if len(results) >= top_papers:
                    return results
    return results[:top_papers]


async def find_evidence_for_claim(
    claim: str,
    *,
    top_papers: int = 5,
    tier: int = 2,
    cache_dir: Optional[Path] = None,
    rag_top_k: int = 5,
    embedding_backend: str = "auto",
    api_key: Optional[str] = None,
    model: Optional[str] = None,
) -> EvidenceReport:
    """Search → fetch → RAG → return evidence spans for ``claim``.

    Reuses the existing audit infrastructure (PaperContentCache,
    EmbeddingStore, evidence.build_evidence_spans) so this command
    shares caches and behaviour with `bibsync audit`.
    """
    import time
    from platformdirs import user_cache_dir
    from .audit_sources import fetch_paper_content, PaperContentCache
    from .audit_sources.openalex import _parse_openalex_work
    from .audit_sources._match import titles_match
    from .audit_rag import EmbeddingStore, chunk_text
    from .audit_sources.pdf import PdfCache, get_paper_text
    from .audit_sources.tables import extract_tables_from_pdf
    from .evidence import build_evidence_spans

    t0 = time.monotonic()
    if cache_dir is None:
        cache_dir = Path(user_cache_dir("bibsync", "bibsync"))
    cache_dir.mkdir(parents=True, exist_ok=True)
    paper_cache = PaperContentCache(cache_dir)
    pdf_cache = PdfCache(cache_dir) if tier >= 2 else None
    embed_store = (
        EmbeddingStore(cache_dir, model="auto", api_key=api_key, backend=embedding_backend)
        if tier >= 2 else None
    )

    raw = await _candidates_for_claim(claim, top_papers=top_papers, api_key=api_key)
    report = EvidenceReport(claim=claim)

    for work in raw:
        # Parse the OpenAlex work into a PaperContent-shaped record.
        pc = _parse_openalex_work(work)
        if pc is None:
            continue
        cand = EvidenceCandidate(
            paper_key=pc.stable_key(),
            title=pc.title,
            first_author=(pc.authors[0] if pc.authors else ""),
            year=pc.year,
            venue=pc.venue or "",
            doi=pc.doi or "",
            arxiv_id=pc.arxiv_id or "",
            pdf_url=pc.pdf_url or "",
            cited_by=int(work.get("cited_by_count") or 0),
            source="openalex",
            has_abstract=bool(pc.abstract),
            has_pdf=bool(pc.pdf_url),
        )

        # If we don't have abstract OR pdf, try the full fallback chain to
        # enrich (Unpaywall often provides PDF URLs for DOI-keyed papers).
        if not pc.abstract or not pc.pdf_url:
            enriched = await fetch_paper_content(
                title=pc.title, first_author=cand.first_author or None,
                year=pc.year, doi=pc.doi, cache=paper_cache,
            )
            if enriched:
                if not pc.abstract and enriched.abstract:
                    pc.abstract = enriched.abstract
                    cand.has_abstract = True
                if not pc.pdf_url and enriched.pdf_url:
                    pc.pdf_url = enriched.pdf_url
                    cand.pdf_url = enriched.pdf_url
                    cand.has_pdf = True

        # Run RAG against the PDF if we have it AND tier >= 2.
        spans = []
        if tier >= 2 and pc.pdf_url and pdf_cache and embed_store:
            text = await get_paper_text(cand.paper_key, pc.pdf_url, pdf_cache)
            if text:
                prose_chunks = chunk_text(text, cand.paper_key)
                pdf_path = pdf_cache.pdf_path(cand.paper_key)
                table_chunks = (
                    extract_tables_from_pdf(pdf_path, cand.paper_key)
                    if pdf_path.exists() else []
                )
                all_chunks = table_chunks + prose_chunks
                if all_chunks:
                    top = await embed_store.retrieve(
                        query=claim, paper_key=cand.paper_key,
                        chunks=all_chunks, top_k=rag_top_k,
                    )
                    if top:
                        cand.evidence_tier = 2
                        spans = build_evidence_spans(
                            [c for c, _ in top], claim,
                            paper_key=cand.paper_key, paper_title=pc.title,
                            chunk_scores=[s for _, s in top],
                            evidence_type="supporting",
                            max_spans=rag_top_k,
                        )
                        cand.spans = [s.to_dict() for s in spans]
        if not spans and pc.abstract:
            # Tier-1 fallback: use the abstract as evidence.
            from .evidence import EvidenceSpan, extract_evidence_span
            quote = extract_evidence_span(pc.abstract, claim)
            if quote:
                cand.evidence_tier = 1
                cand.spans = [EvidenceSpan(
                    type="supporting", paper_key=cand.paper_key,
                    paper_title=pc.title, page=None, chunk_idx=None,
                    chunk_score=0.0, quote=quote,
                ).to_dict()]
        if not cand.spans:
            cand.note = "no evidence spans (no PDF, no abstract overlap)"
        report.candidates.append(cand)

    report.elapsed_sec = time.monotonic() - t0
    return report


def find_evidence_sync(*args, **kwargs) -> EvidenceReport:
    return asyncio.run(find_evidence_for_claim(*args, **kwargs))
