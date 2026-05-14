"""Suggest and insert citations into a LaTeX file that currently lacks them.

Workflow:
  1. Split the .tex file into paragraphs.
  2. For each paragraph WITHOUT existing \\cite{} calls, ask the LLM what should be cited.
  3. For each suggestion, search Google Scholar, pick the canonical version, fetch BibTeX.
  4. If interactive, ask the user to confirm before writing.
  5. On confirmation:
       a. Insert \\cite{key} after the LLM-provided anchor phrase in the .tex.
       b. Append the BibTeX entry to the .bib file.

Single .bib file assumption: the user maintains one .bib per project.
"""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from . import bibtex, dbg, llm, picker, scholar, tex_rewrite
from .models import PaperHit

# Grounding-gate (Filter D) imports are LAZY — only loaded when --verify-tier ≥ 1,
# so a default suggest run doesn't pay the cost of importing audit_rag / pypdf /
# fastembed when the user hasn't asked for grounding.

# Per-claim caps: includes both deterministic canonical queries (see
# _canonical_paper_queries) and the LLM-proposed queries. 5 = 2 canonical + 3 LLM.
MAX_QUERIES_PER_CLAIM = 5
MAX_CANDIDATES_PER_CLAIM = 5
CLAIM_SUPPORT_CONFIDENCE_FLOOR = 0.7

# Anchors longer than this (word count) are NOT used as a quoted-phrase query.
# Quoting "the attention mechanism that displaced recurrence" yields too few results;
# quoting short named-system anchors like "MedSAM" or "Med-PaLM 2" is exactly right.
MAX_ANCHOR_WORDS_FOR_QUOTED_QUERY = 6


def _canonical_paper_queries(anchor: str) -> list[str]:
    """Return deterministic Scholar queries that target the CANONICAL paper for a
    short named-system anchor.

    The pattern ``"<name>" original paper`` is a well-known manual technique
    (quotes force literal match on the system name; "original paper" downranks
    surveys / reviews / applications). We prepend these in front of whatever
    queries the LLM proposed.
    """
    anchor = anchor.strip()
    if not anchor:
        return []
    word_count = len(anchor.split())
    if word_count == 0 or word_count > MAX_ANCHOR_WORDS_FOR_QUOTED_QUERY:
        return []
    quoted = f'"{anchor}"'
    return [
        f"{quoted} original paper",
        quoted,
    ]

_PARA_BOUNDARY = re.compile(r"\n\s*\n")
_CITE_PRESENCE_RE = re.compile(r"\\(?:no)?cite\w*\s*(?:\[[^\]]*\])*\s*\{")

# Pattern matches "Name <version-token>" in a claim. Name is a hyphenated or
# camel-case word starting with a capital. Version token is digits, a lone
# capital letter (M / N / X), or a small set of version words (v2, Pro, Plus).
_VERSIONED_NAME_RE = re.compile(
    r"\b([A-Z][A-Za-z]+(?:[-][A-Za-z]+)*)\s+(\d+(?:\.\d+)?|[A-Z](?=\b)|v\d+|Pro|Plus)\b"
)


def _version_mismatch(claim: str, candidate_title: str) -> Optional[str]:
    """Return a rejection reason if the claim names a versioned system but the
    candidate title contains only the base (unversioned) name.

    Catches the gpt-4o-mini blind spot where "Med-PaLM 2" got matched to the original
    "Med-PaLM" paper. The candidate must contain ``<Name>[\\s\\-]<Version>`` together;
    presence of only ``<Name>`` is treated as a different paper (almost always the
    earlier version).
    """
    for m in _VERSIONED_NAME_RE.finditer(claim):
        name = m.group(1)
        version = m.group(2)
        cand = candidate_title.lower()
        name_re = re.escape(name.lower())
        ver_re = re.escape(version.lower())
        if re.search(rf"\b{name_re}[\s\-]*{ver_re}\b", cand):
            continue  # candidate contains the versioned form — fine
        if re.search(rf"\b{name_re}\b", cand):
            return (
                f"claim names {name!r} {version!r}, but candidate title contains "
                f"only base {name!r} — different version, almost certainly the "
                f"prior paper"
            )
    return None


@dataclass
class SuggestionResult:
    paragraph_index: int
    paragraph_preview: str
    query: str
    anchor: str
    reason: str
    scholar_hit: Optional[PaperHit] = None
    cite_key: Optional[str] = None
    status: str = "pending"  # "added" | "skipped" | "no_scholar_hit" | "anchor_not_found" | "duplicate" | "error" | "no_supporting_match" | "no_grounded_match"
    note: str = ""
    identification: Optional["llm.IdentifiedPaper"] = None
    # Filter-D grounding evidence (None when --verify-tier 0). Used by the CLI
    # approval prompt to show the user what evidence backs the accepted citation.
    grounding: Optional["_GroundingVerdict"] = None


@dataclass
class SuggestReport:
    tex_file: Path
    bib_file: Path
    paragraphs_scanned: int = 0
    paragraphs_with_existing_cites: int = 0
    results: list[SuggestionResult] = field(default_factory=list)

    def summary(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for r in self.results:
            out[r.status] = out.get(r.status, 0) + 1
        return out


def _split_paragraphs(text: str) -> list[str]:
    return [p for p in _PARA_BOUNDARY.split(text) if p.strip()]


def _has_existing_cite(paragraph: str) -> bool:
    return bool(_CITE_PRESENCE_RE.search(paragraph))


def _claim_sentence_around_anchor(paragraph: str, anchor: str) -> str:
    """Return the sentence in ``paragraph`` that contains ``anchor``.

    This is the prose claim Filter D should ground against — NOT the anchor
    itself. Passing just the anchor (often a 1-2 word phrase like 'GPT-3') to
    ``audit_citation`` collapses the question to "does this paper support the
    term 'GPT-3'?" — trivially true and useless for grounding. The actual
    misattribution we want to catch lives in the full sentence (e.g.
    "GPT-3 achieves 86.5% on MedQA" → the GPT-3 paper does NOT support that).

    Mirrors ``audit._extract_claim``'s sentence-boundary logic, but uses the
    anchor's position in the paragraph as the pivot. Falls back to the whole
    paragraph if the anchor isn't found (e.g. case-folded or stripped).
    """
    if not anchor:
        return paragraph.strip()
    idx = paragraph.find(anchor)
    if idx < 0:
        # Try a case-insensitive search before giving up — anchors sometimes
        # come back lower-cased from the LLM.
        lower = paragraph.lower()
        idx = lower.find(anchor.lower())
        if idx < 0:
            return paragraph.strip()
    # Walk back to start of previous sentence (or paragraph start).
    start = 0
    for m in re.finditer(r"(?:[.!?]\s+)|(?:\n\s*\n)", paragraph[:idx]):
        start = m.end()
    # Walk forward to next sentence end.
    after = paragraph[idx:]
    end_m = re.search(r"[.!?](?:\s|\n|$)", after)
    end = idx + (end_m.end() if end_m else len(after))
    return paragraph[start:end].strip()




def _queries_from_identification(ident: llm.IdentifiedPaper) -> list[str]:
    """Build targeted Scholar queries from the LLM's paper identification.

    The strongest query is the verbatim paper title (Scholar is title-search-friendly).
    Adding the first-author surname narrows further when the title is generic.
    arXiv id is its own deterministic hit when known.
    """
    out: list[str] = []
    if not ident or not ident.expected_title:
        return out
    title = ident.expected_title.strip().strip('"').strip("'")
    if not title:
        return out
    if ident.expected_first_author:
        out.append(f"{title} {ident.expected_first_author}")
    out.append(title)
    if ident.arxiv_id:
        out.append(f"arXiv:{ident.arxiv_id}")
    return out


@dataclass
class _GroundingVerdict:
    """Result of the Tier-1/2 grounding gate (Filter D) on a single Scholar candidate.

    ``passed=True`` means: the LLM, given abstract (tier 1) or retrieved chunks
    (tier 2), confirmed the candidate paper actually supports the SPECIFIC claim
    — not just the topic. ``passed=False`` is one of:
      * the LLM said supports=false with conf ≥ floor (real grounding rejection)
      * we couldn't fetch the source / PDF / chunks (infra failure)

    ``evidence_summary`` is a short human-readable string used in the approval
    prompt UI: e.g. ``"abstract grounded"``, ``"5 chunks retrieved (top score 0.71)"``.
    """
    passed: bool
    achieved_tier: int  # 0 = no grounding done, 1 = abstract, 2 = chunks
    confidence: float
    reason: str
    evidence_summary: str = ""


async def _ground_candidate(
    *,
    candidate: PaperHit,
    claim_text: str,
    verify_tier: int,
    rag_top_k: int,
    grounding_floor: float,
    paper_cache,
    pdf_cache,
    embed_store,
    no_cache: bool,
    model: Optional[str],
    api_key: Optional[str],
    ss_api_key: Optional[str] = None,
) -> _GroundingVerdict:
    """Filter D — grounded verification of a Scholar candidate against a prose claim.

    Reuses the audit pipeline end-to-end:
      tier 1: ``fetch_paper_content(title)`` → audit_citation(abstract=...)
      tier 2: + PDF download + chunking + embed + retrieve → audit_citation(chunks=...)

    Returns a ``_GroundingVerdict``. The candidate-evaluation loop in
    :func:`_resolve_suggestion` treats ``passed=False`` exactly like a Filter C
    rejection — try the next candidate.
    """
    if verify_tier < 1:
        return _GroundingVerdict(
            passed=True, achieved_tier=0, confidence=0.0,
            reason="grounding disabled (--verify-tier 0)",
        )

    # Lazy imports — only when grounding is actually requested.
    from .audit_sources import fetch_paper_content

    first_author = ""
    if candidate.authors:
        a = candidate.authors[0]
        if "," in a:
            first_author = a.split(",", 1)[0].strip()
        else:
            parts = a.split()
            first_author = parts[-1] if parts else ""

    dbg.trace(
        "suggest.ground",
        "fetching source",
        title=candidate.title,
        first_author=first_author,
        tier=verify_tier,
    )

    content = await fetch_paper_content(
        title=candidate.title,
        first_author=first_author or None,
        year=candidate.year,
        cache=paper_cache,
        no_cache=no_cache,
        ss_api_key=ss_api_key,
    )

    if content is None:
        # arXiv/SS/Crossref all missed (or title-match guard rejected every hit).
        # Don't pay for an LLM call we already know will degrade to Tier 0.
        dbg.trace(
            "suggest.ground",
            "REJECT no source",
            title=candidate.title,
            reason="arXiv/SS/Crossref all missed or rejected by title-match guard",
        )
        return _GroundingVerdict(
            passed=False, achieved_tier=0, confidence=1.0,
            reason="could not fetch paper abstract/PDF — Scholar canonical match "
                   "may be the wrong paper, or paper is not open-access",
        )

    retrieved_chunk_texts: Optional[list[str]] = None
    evidence_summary = ""

    if verify_tier >= 2 and pdf_cache is not None and embed_store is not None:
        if not content.pdf_url:
            dbg.trace(
                "suggest.ground",
                "no open-access PDF — degrading to abstract-only grounding",
                title=candidate.title,
            )
        else:
            from .audit_rag import chunk_text
            from .audit_sources.pdf import get_paper_text

            paper_key = content.stable_key()
            text = await get_paper_text(paper_key, content.pdf_url, pdf_cache)
            chunks = chunk_text(text, paper_key) if text else []
            if chunks:
                top = await embed_store.retrieve(
                    query=claim_text,
                    paper_key=paper_key,
                    chunks=chunks,
                    top_k=rag_top_k,
                )
                if top:
                    retrieved_chunk_texts = [
                        f"[p.{c.page or '?'}] {c.text}" for c, _ in top
                    ]
                    top_score = round(top[0][1], 3)
                    evidence_summary = (
                        f"{len(retrieved_chunk_texts)} chunks (top cosine {top_score})"
                    )

    achieved_tier = 2 if retrieved_chunk_texts else (1 if content.abstract else 0)
    if achieved_tier == 0:
        # Source returned but no abstract AND no chunks — nothing to ground against.
        dbg.trace(
            "suggest.ground",
            "REJECT no evidence",
            title=candidate.title,
            has_abstract=False,
            has_chunks=False,
        )
        return _GroundingVerdict(
            passed=False, achieved_tier=0, confidence=0.8,
            reason="paper found but no abstract or open-access PDF — cannot ground",
        )

    if not evidence_summary:
        evidence_summary = "abstract grounded" if achieved_tier == 1 else "chunks grounded"

    # Now run the strengthened audit prompt with whatever evidence we got.
    audit = llm.audit_citation(
        claim_text=claim_text,
        cited_paper_title=content.title or candidate.title,
        cited_paper_authors=", ".join(content.authors or candidate.authors or []),
        cited_paper_year=content.year or candidate.year,
        cited_paper_venue=content.venue or (candidate.venue or ""),
        abstract=content.abstract,
        retrieved_chunks=retrieved_chunk_texts,
        model=model,
        api_key=api_key,
    )

    passed = audit.supports and audit.confidence >= grounding_floor
    dbg.trace(
        "suggest.ground",
        "ACCEPTED" if passed else "REJECTED",
        tier=achieved_tier,
        supports=audit.supports,
        conf=round(audit.confidence, 2),
        reason=audit.reasoning,
        evidence=evidence_summary,
    )
    return _GroundingVerdict(
        passed=passed,
        achieved_tier=achieved_tier,
        confidence=audit.confidence,
        reason=audit.reasoning,
        evidence_summary=evidence_summary,
    )


async def _resolve_suggestion(
    suggestion: llm.CitationSuggestion,
    paragraph: str,
    paragraph_idx: int,
    *,
    headless: bool,
    model: Optional[str],
    api_key: Optional[str],
    used_cluster_ids: Optional[set] = None,
    document_context: str = "",
    # Filter-D grounding gate (Tier-1/2 verification). Default 0 = today's
    # behaviour (Filter C only).
    verify_tier: int = 0,
    rag_top_k: int = 5,
    grounding_floor: float = 0.6,
    paper_cache=None,
    pdf_cache=None,
    embed_store=None,
    no_cache: bool = False,
    ss_api_key: Optional[str] = None,
) -> tuple[SuggestionResult, Optional[dict]]:
    """Agentic pipeline for one suggested citation:

      1. Run ALL of the LLM's proposed queries on Scholar; merge & dedupe candidates.
      2. Sort merged candidates by ``cited_by`` (most-cited first — the original
         paper is almost always vastly more cited than any derivative).
      3. For the top N candidates, ask the LLM-as-judge whether each candidate
         actually supports the prose claim. Accept the first ``supports=True`` with
         confidence ≥ ``CLAIM_SUPPORT_CONFIDENCE_FLOOR``.
      4. Fetch official BibTeX for the accepted candidate.

    Returns ``(result, bibtex_entry_or_None)``. If the LLM rejects every candidate
    across every query, the result is marked ``no_supporting_match`` and the caller
    leaves both .tex and .bib untouched for this suggestion.
    """
    result = SuggestionResult(
        paragraph_index=paragraph_idx,
        paragraph_preview=paragraph[:80] + ("…" if len(paragraph) > 80 else ""),
        query=suggestion.query,
        anchor=suggestion.anchor,
        reason=suggestion.reason,
    )

    # === Step 0: LLM world-knowledge paper identification ===
    # The most reliable signal we have. For famous papers (UNI, BERT, Vaswani, etc.)
    # the LLM just *knows* the canonical paper from training data. We use its named
    # title + first-author as a targeted Scholar query — far more reliable than
    # topic-guessing.
    try:
        identification = llm.identify_canonical_paper(
            claim=suggestion.anchor,
            paragraph=paragraph,
            document_context=document_context,
            model=model,
            api_key=api_key,
        )
    except Exception as e:
        dbg.trace("suggest.identify", "ERR", error=str(e))
        identification = llm.IdentifiedPaper(expected_title="", confidence=0.0)

    targeted_qs: list[str] = []
    if identification.confidence >= 0.4:  # accept even moderately confident IDs
        targeted_qs = _queries_from_identification(identification)
    canonical_qs = _canonical_paper_queries(suggestion.anchor)

    # Query order: LLM-identified-title first (most precise), then deterministic
    # canonical pattern, then LLM-proposed topical queries.
    all_queries: list[str] = []
    for q in targeted_qs + canonical_qs + list(suggestion.queries):
        if q and q not in all_queries:
            all_queries.append(q)

    dbg.trace(
        "suggest.resolve",
        "start",
        anchor=suggestion.anchor,
        identified_title=identification.expected_title,
        identified_author=identification.expected_first_author,
        identified_year=identification.expected_year,
        identified_conf=round(identification.confidence, 2),
        targeted_queries=targeted_qs,
        canonical_queries=canonical_qs,
        llm_queries=suggestion.queries,
        para=paragraph_idx,
    )
    # Stash identification on the result so the report can show it.
    result.identification = identification

    # Pipeline shape (query-by-query early exit):
    #
    #   for each query (canonical first, then LLM-proposed):
    #     1. run Scholar search; merge new candidates by cluster_id
    #     2. sort the WHOLE pool so far by cited_by DESC
    #     3. for each top candidate not yet evaluated:
    #          a. cheap deterministic filters (version mismatch, dedup)
    #          b. LLM-as-judge claim support
    #          c. if accepted → STOP (don't run more queries)
    #
    # This is dramatically faster than run-all-queries-then-judge: when query #1
    # already returns the canonical paper (the common case), we issue exactly one
    # Scholar search and one LLM call before moving on.
    merged: dict[str, PaperHit] = {}
    evaluated: set[str] = set()  # cluster IDs we've already LLM-judged or filtered
    accepted: Optional[PaperHit] = None
    # Track whether at least one candidate passed Filter C (topical canonicality)
    # but failed Filter D (grounding). When this is True at end-of-loop, the
    # rejection mode is "we found the right paper but it doesn't support YOUR
    # specific claim" — a much more actionable user signal than "no canonical
    # match found at all".
    grounding_blocked: bool = False
    last_reason = ""
    last_confidence = 0.0
    used = used_cluster_ids if used_cluster_ids is not None else set()

    for q_idx, query in enumerate(all_queries[:MAX_QUERIES_PER_CLAIM]):
        dbg.trace("suggest.query", f"attempt #{q_idx + 1}", query=query)
        try:
            hits = await scholar.search(query, headless=headless, max_results=8)
        except Exception as e:
            dbg.trace("suggest.query", "ERR", error=str(e))
            continue
        new = 0
        for h in hits:
            if h.cluster_id and h.cluster_id not in merged:
                merged[h.cluster_id] = h
                new += 1
        dbg.trace(
            "suggest.query",
            f"got {len(hits)} hits",
            new_candidates=new,
            total_pool=len(merged),
        )

        if not merged:
            continue

        # Sort the whole pool so far by cited_by — canonical papers (typically
        # 1000+ cites) rise to the top across all queries' results.
        sorted_pool = sorted(merged.values(), key=lambda h: (h.cited_by or 0), reverse=True)

        # Walk the top candidates we haven't yet evaluated.
        for cand_i, candidate in enumerate(sorted_pool[:MAX_CANDIDATES_PER_CLAIM]):
            if candidate.cluster_id in evaluated:
                continue

            # Filter A — deterministic version-mismatch check (catches Med-PaLM 2 → Med-PaLM).
            mismatch = _version_mismatch(suggestion.anchor, candidate.title)
            if mismatch:
                dbg.trace(
                    "suggest.version_check",
                    "REJECT",
                    candidate_idx=cand_i + 1,
                    title=candidate.title,
                    reason=mismatch,
                )
                last_reason = f"version mismatch: {mismatch}"
                last_confidence = 1.0
                evaluated.add(candidate.cluster_id)
                continue

            # Filter B — cluster-id dedup (catches "same paper cited for two claims").
            if candidate.cluster_id and candidate.cluster_id in used:
                dbg.trace(
                    "suggest.dedupe",
                    "REJECT",
                    candidate_idx=cand_i + 1,
                    title=candidate.title,
                    cluster=candidate.cluster_id,
                )
                last_reason = (
                    "candidate cluster already used by an earlier citation — "
                    "would create a duplicate"
                )
                last_confidence = 1.0
                evaluated.add(candidate.cluster_id)
                continue

            # Filter C — LLM-as-judge (canonical-paper detection).
            verdict = llm.verify_claim_support(
                claim_text=suggestion.anchor,
                context=paragraph,
                candidate=candidate,
                model=model,
                api_key=api_key,
            )
            evaluated.add(candidate.cluster_id)
            last_reason = verdict.reasoning
            last_confidence = verdict.confidence
            if not (
                verdict.supports
                and verdict.confidence >= CLAIM_SUPPORT_CONFIDENCE_FLOOR
            ):
                dbg.trace(
                    "suggest.verify",
                    f"rejected (after q{q_idx + 1}, cand{cand_i + 1})",
                    conf=round(verdict.confidence, 2),
                    reason=verdict.reasoning,
                )
                continue

            # Filter D — grounding gate (Tier-1/2 verification against the actual
            # paper's abstract / full-text chunks). Only runs when --verify-tier ≥ 1.
            # A no-op when verify_tier=0, so Filter C remains the only gate by default.
            #
            # CRITICAL: ground against the FULL SENTENCE containing the anchor,
            # not the anchor itself. The anchor is a 1-2 word phrase ('GPT-3');
            # the misattribution the gate is supposed to catch lives in the
            # surrounding prose ('GPT-3 achieves 86.5% on MedQA'). Mirrors how
            # the audit pipeline grounds on the sentence containing the \cite{}.
            grounded_claim = _claim_sentence_around_anchor(paragraph, suggestion.anchor)
            grounding = await _ground_candidate(
                candidate=candidate,
                claim_text=grounded_claim,
                verify_tier=verify_tier,
                rag_top_k=rag_top_k,
                grounding_floor=grounding_floor,
                paper_cache=paper_cache,
                pdf_cache=pdf_cache,
                embed_store=embed_store,
                no_cache=no_cache,
                model=model,
                api_key=api_key,
                ss_api_key=ss_api_key,
            )
            if not grounding.passed:
                # The candidate is the right canonical paper, but the SPECIFIC
                # claim isn't supported by the paper's actual content. Reject and
                # try the next candidate. Surface the grounding reason — it's
                # usually more actionable than Filter-C reasons ("the paper does
                # not mention 'MedQA' or '86.5%'").
                grounding_blocked = True
                last_reason = (
                    f"grounded rejection (tier {grounding.achieved_tier}): "
                    f"{grounding.reason}"
                )
                last_confidence = grounding.confidence
                dbg.trace(
                    "suggest.ground",
                    f"rejected (after q{q_idx + 1}, cand{cand_i + 1})",
                    achieved_tier=grounding.achieved_tier,
                    conf=round(grounding.confidence, 2),
                    reason=grounding.reason,
                )
                continue

            # Both Filter C (canonicality) AND Filter D (grounding) accept.
            accepted = candidate
            if candidate.cluster_id:
                used.add(candidate.cluster_id)
            # Stash grounding evidence on the result so the approval prompt
            # and end-of-run report can show it.
            result.grounding = grounding
            dbg.trace(
                "suggest.verify",
                "ACCEPTED",
                after_query=q_idx + 1,
                candidate_idx=cand_i + 1,
                conf=round(verdict.confidence, 2),
                ground_tier=grounding.achieved_tier,
                ground_conf=round(grounding.confidence, 2),
                title=candidate.title,
                cluster=candidate.cluster_id,
            )
            break

        if accepted is not None:
            # Stop — no need to run the remaining queries.
            dbg.trace(
                "suggest.resolve",
                "early-exit",
                queries_used=q_idx + 1,
                queries_skipped=max(0, min(len(all_queries), MAX_QUERIES_PER_CLAIM) - (q_idx + 1)),
            )
            break

    if accepted is None and not merged:
        result.status = "no_scholar_hit"
        result.note = (
            f"no Scholar results across {min(len(all_queries), MAX_QUERIES_PER_CLAIM)} "
            f"queries (incl. canonical-paper pattern)"
        )
        dbg.trace("suggest.resolve", "no candidates from any query")
        return result, None

    if accepted is None:
        # Build a single-pass sorted view for the report's "top candidate" surface.
        sorted_pool = sorted(merged.values(), key=lambda h: (h.cited_by or 0), reverse=True)
        if grounding_blocked:
            # The canonical-detection filter (C) accepted at least one candidate,
            # but the grounding filter (D) rejected all of them. This means we
            # FOUND the right paper, but the paper doesn't actually support the
            # specific prose claim. Distinct from no_supporting_match because
            # the user likely needs to fix their CLAIM, not search for a different
            # paper.
            result.status = "no_grounded_match"
            result.note = (
                f"canonical paper found, but grounding (Tier-1/2) rejected — "
                f"{last_reason}"
            )
        else:
            result.status = "no_supporting_match"
            result.note = (
                f"LLM rejected all {min(len(sorted_pool), MAX_CANDIDATES_PER_CLAIM)} "
                f"top candidate(s) — last reason: {last_reason}"
            )
        if sorted_pool:
            result.scholar_hit = sorted_pool[0]
        return result, None

    result.scholar_hit = accepted

    if not accepted.cluster_id:
        result.status = "error"
        result.note = "accepted hit had no cluster id"
        return result, None

    # Step 4: fetch BibTeX for the LLM-verified canonical paper.
    try:
        bib_text = await scholar.fetch_bibtex_for_cluster(
            accepted.cluster_id, headless=headless
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

    return result, db.entries[0]


async def suggest_for_file(
    tex_file: Path,
    bib_file: Path,
    *,
    headless: bool = False,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    only_paragraphs_without_cites: bool = True,
    auto_approve: bool = False,
    approve_fn=None,  # callable(SuggestionResult, candidate_bibtex_entry) -> bool
    delay_seconds: float = 1.5,
    # Filter-D grounding gate. verify_tier=0 → today's behaviour (Filter C only).
    # verify_tier=1 → add abstract grounding. verify_tier=2 → add PDF-RAG grounding.
    verify_tier: int = 0,
    rag_top_k: int = 5,
    grounding_floor: float = 0.6,
    embedding_backend: str = "auto",
    embedding_model: str = "auto",
    cache_dir: Optional[Path] = None,
    no_cache: bool = False,
    ss_api_key: Optional[str] = None,
) -> SuggestReport:
    """Scan ``tex_file`` paragraph-by-paragraph for missing citations.

    When ``auto_approve`` is False (default) and ``approve_fn`` is provided, the function
    calls ``approve_fn`` for each suggestion to ask the user whether to commit it.
    """
    text = tex_file.read_text(encoding="utf-8", errors="replace")
    paragraphs = _split_paragraphs(text)

    report = SuggestReport(tex_file=tex_file, bib_file=bib_file)
    report.paragraphs_scanned = len(paragraphs)

    db = bibtex.load(bib_file)

    # Document context for the LLM identifier — pass the whole tex file (capped in
    # the identifier itself) so anchors like "UNI" can be disambiguated by the
    # surrounding paper topic.
    document_context = text

    dbg.trace(
        "suggest.start",
        tex=str(tex_file),
        bib=str(bib_file),
        paragraphs=len(paragraphs),
        doc_chars=len(document_context),
        verify_tier=verify_tier,
    )

    # ── Filter-D grounding-gate setup ────────────────────────────────────────
    # All initialised once and shared across every anchor in this run so a paper
    # that grounds 3 different claims is fetched/embedded once.
    paper_cache = None
    pdf_cache = None
    embed_store = None
    if verify_tier >= 1:
        from platformdirs import user_cache_dir
        from .audit_sources import PaperContentCache
        if cache_dir is None:
            cache_dir = Path(user_cache_dir("bibsync", "bibsync"))
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        paper_cache = PaperContentCache(cache_dir)
        dbg.trace("suggest.ground", "Tier-1 cache initialised", dir=str(cache_dir))
    if verify_tier >= 2:
        from .audit_rag import EmbeddingStore
        from .audit_sources.pdf import PdfCache
        pdf_cache = PdfCache(cache_dir)
        embed_store = EmbeddingStore(
            cache_dir,
            model=embedding_model,
            api_key=api_key,
            backend=embedding_backend,
        )
        dbg.trace(
            "suggest.ground",
            "Tier-2 RAG pipeline initialised",
            top_k=rag_top_k,
            backend=embedding_backend,
        )

    # Track which Scholar cluster_ids we've already cited in THIS run. If two
    # different anchors both resolve to the same paper, the second is almost
    # certainly the wrong match (since we already wrote it for the first claim).
    used_cluster_ids: set[str] = set()
    # Pre-seed with cluster IDs we can already infer from existing .bib entries
    # (so a re-run doesn't duplicate-cite). We can't fully recover cluster_id from
    # a stored entry, so this is best-effort — handled at append time below.

    # Pool all Scholar calls in this run into ONE browser context.
    async with scholar.shared_session(headless=headless):
        for p_idx, paragraph in enumerate(paragraphs):
            if only_paragraphs_without_cites and _has_existing_cite(paragraph):
                report.paragraphs_with_existing_cites += 1
                dbg.trace("suggest.paragraph", "skip (already has \\cite)", idx=p_idx)
                continue

            dbg.trace(
                "suggest.paragraph",
                "scanning",
                idx=p_idx,
                preview=paragraph[:80],
            )
            try:
                suggestions = llm.suggest_citations(
                    paragraph, model=model, api_key=api_key
                )
            except Exception as e:
                r = SuggestionResult(
                    paragraph_index=p_idx,
                    paragraph_preview=paragraph[:80],
                    query="",
                    anchor="",
                    reason="",
                    status="error",
                    note=f"llm suggest failed: {e}",
                )
                report.results.append(r)
                continue
            dbg.trace(
                "suggest.paragraph",
                "got suggestions",
                idx=p_idx,
                n=len(suggestions),
            )

            for sugg in suggestions:
                r, entry = await _resolve_suggestion(
                    sugg, paragraph, p_idx,
                    headless=headless, model=model, api_key=api_key,
                    used_cluster_ids=used_cluster_ids,
                    document_context=document_context,
                    verify_tier=verify_tier,
                    rag_top_k=rag_top_k,
                    grounding_floor=grounding_floor,
                    paper_cache=paper_cache,
                    pdf_cache=pdf_cache,
                    embed_store=embed_store,
                    no_cache=no_cache,
                    ss_api_key=ss_api_key,
                )

                if entry is None:
                    report.results.append(r)
                    continue

                # Set cite key from corrected metadata.
                cite_key = bibtex.derive_cite_key(entry)
                cite_key = bibtex.ensure_unique_key(db, cite_key)
                entry["ID"] = cite_key
                r.cite_key = cite_key
                dbg.trace(
                    "suggest.commit",
                    "proposing",
                    cite_key=cite_key,
                    anchor=r.anchor,
                )

                # Approval gate.
                approved = auto_approve
                if not approved and approve_fn is not None:
                    approved = approve_fn(r, entry)
                elif not approved and approve_fn is None:
                    approved = True  # default to True when no callback provided

                if not approved:
                    r.status = "skipped"
                    r.note = "user rejected"
                    dbg.trace("suggest.commit", "user rejected", cite_key=cite_key)
                    report.results.append(r)
                    continue

                # Check duplicate against current .bib state.
                stored, was_added = bibtex.append_entry(db, entry)
                if not was_added:
                    r.status = "duplicate"
                    r.cite_key = stored["ID"]
                    r.note = f"already in .bib as {stored['ID']}"
                    dbg.trace("suggest.commit", "duplicate", existing_key=stored["ID"])

                # Persist the .bib BEFORE touching the .tex. This guarantees that
                # if anything later (Ctrl+C, exception, anchor-not-found) interrupts
                # us, we never leave the .tex with a \cite{} pointing to a missing
                # .bib entry. bibtex.dump is atomic (write-tempfile-then-rename).
                if was_added:
                    bibtex.dump(db, bib_file)
                    dbg.trace(
                        "suggest.commit", "bib persisted",
                        cite_key=r.cite_key, path=str(bib_file),
                    )

                # Insert \cite{} into the .tex at the LLM-provided anchor.
                inserted = tex_rewrite.insert_cite_after_anchor(
                    tex_file, r.anchor, r.cite_key
                )
                if not inserted:
                    inserted = tex_rewrite.append_cite_to_paragraph(
                        tex_file, paragraph, r.cite_key
                    )
                if inserted and r.status != "duplicate":
                    r.status = "added"
                elif not inserted:
                    r.status = "anchor_not_found"
                    r.note = "could not locate insertion point in .tex"
                dbg.trace(
                    "suggest.commit",
                    "done",
                    cite_key=r.cite_key,
                    status=r.status,
                )
                report.results.append(r)

                await asyncio.sleep(delay_seconds)

    # Final defensive dump in case any duplicate-only paths skipped the per-step
    # write above. Idempotent: same content, no harm.
    if any(r.status in ("added", "duplicate") for r in report.results):
        bibtex.dump(db, bib_file)
        dbg.trace("suggest.done", "bib finalised", path=str(bib_file))

    return report


def suggest_for_file_sync(*args, **kwargs) -> SuggestReport:
    return asyncio.run(suggest_for_file(*args, **kwargs))
