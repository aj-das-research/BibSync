"""Audit existing citations in a LaTeX project for hallucination / misattribution.

For every ``\\cite{key}`` call in the project's .tex files, this verifies that the
cited paper (per its .bib entry) actually supports the surrounding prose claim.
Hallucinated citations — typically introduced by LLM-assisted paper drafting where
the model fabricated plausible-looking BibTeX entries — are flagged and optionally
removed with ``--fix``.

Pipeline:
  1. Scan all .tex files for ``\\cite{...}`` matches; for each, extract the
     surrounding sentence (the claim).
  2. Group occurrences by cite key — one LLM call per unique paper/claim pair
     instead of per textual occurrence.
  3. Look up each key in the .bib; missing keys are flagged ``missing_in_bib``.
  4. For each (claim, paper) pair, call ``llm.audit_citation``.
  5. Build a report with: verified / hallucinated / unverifiable / missing_in_bib
     buckets.
  6. With ``--fix``, replace every hallucinated ``\\cite{...}`` with a marker
     comment so the user can see exactly what was removed.
"""

from __future__ import annotations

import asyncio
import bisect
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from . import bibtex, dbg, llm

if TYPE_CHECKING:
    from .audit_sources import PaperContent

# Match any \cite-family command — same regex as scanner.py.
_CITE_RE = re.compile(r"\\(?:no)?cite\w*\s*(?:\[[^\]]*\])*\s*\{([^}]+)\}")

# LaTeX line comment: % to end-of-line, but \% is an escaped percent (not a comment).
_COMMENT_RE = re.compile(r"(?<!\\)%[^\n]*")


def _strip_comments(text: str) -> str:
    """Replace LaTeX line comments with spaces of equal length so character offsets
    and line numbers stay correct. Commented-out ``\\cite{}`` calls must not be
    audited; they're notes-to-self, not real citations."""
    return _COMMENT_RE.sub(lambda m: " " * len(m.group(0)), text)


@dataclass
class CitationCheck:
    """One audited use of a ``\\cite{key}`` somewhere in the project."""

    cite_key: str
    file: Path
    line: int
    char_offset: int  # offset of the \cite{...} call start in the source file
    claim_text: str  # the surrounding sentence (with the \cite{} call stripped)
    bib_entry: Optional[dict] = None
    status: str = "pending"  # verified | hallucinated | contradicted | unverifiable | missing_in_bib
    confidence: float = 0.0
    reasoning: str = ""
    fixed: bool = False  # True if --fix replaced the \cite{} in the .tex
    # Tier-1+ enrichment
    evidence_tier: int = 0  # 0 = metadata only, 1 = abstract used, 2 = chunks used
    paper_content: Optional["PaperContent"] = None  # what audit_sources returned
    n_chunks: int = 0  # how many retrieved chunks were sent to the LLM (tier 2)
    # If the user requested a higher tier but we couldn't deliver it, record WHY
    # so the end-of-run summary can surface a loud diagnostic. One of:
    #   "" (no degradation)
    #   "source_not_found"   — arXiv/SS/Crossref all missed
    #   "no_open_access_pdf" — paper found but no pdf_url (Tier 2 only)
    #   "pdf_download_failed", "pdf_extract_failed"
    #   "embedding_failed"   — fastembed not installed AND API embeddings unusable
    degraded_reason: str = ""


@dataclass
class AuditReport:
    project_root: Path
    bib_file: Path  # the primary .bib (or the literal --bib arg when --per-dir-bib is on)
    tex_files_scanned: int = 0
    checks: list[CitationCheck] = field(default_factory=list)
    # Per-tex bib resolution (only set when per_dir_bib was True):
    bib_files_used: dict[Path, Path] = field(default_factory=dict)

    def summary(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for c in self.checks:
            out[c.status] = out.get(c.status, 0) + 1
        return out

    def to_dict(self) -> dict:
        """Serialisable view of the report for ``--output-json``.

        Drops in-memory-only fields (parsed bib entries, paper_content
        objects) and keeps what an external tool actually needs: file
        locations, statuses, reasoning, confidence, evidence tier, and
        the original prose claim.
        """
        return {
            "project_root": str(self.project_root),
            "bib_file": str(self.bib_file),
            "tex_files_scanned": self.tex_files_scanned,
            "summary": self.summary(),
            "bib_files_used": {
                str(t): str(b) for t, b in (self.bib_files_used or {}).items()
            },
            "checks": [
                {
                    "cite_key": c.cite_key,
                    "file": str(c.file),
                    "line": c.line,
                    "char_offset": c.char_offset,
                    "claim_text": c.claim_text,
                    "status": c.status,
                    "confidence": round(c.confidence, 3),
                    "reasoning": c.reasoning,
                    "evidence_tier": c.evidence_tier,
                    "n_chunks": c.n_chunks,
                    "degraded_reason": c.degraded_reason,
                    "fixed": c.fixed,
                    "paper_source": (
                        c.paper_content.source if c.paper_content else None
                    ),
                    "paper_doi": (
                        c.paper_content.doi if c.paper_content else None
                    ),
                    "paper_arxiv_id": (
                        c.paper_content.arxiv_id if c.paper_content else None
                    ),
                }
                for c in self.checks
            ],
        }


# --- helpers ---------------------------------------------------------------


def _strip_cite_calls(text: str) -> str:
    """Remove ``\\cite{...}`` calls from a fragment so the claim reads cleanly when
    shown to the LLM."""
    out = re.sub(
        r"~?\\(?:no)?cite\w*\s*(?:\[[^\]]*\])*\s*\{[^}]+\}",
        "",
        text,
    )
    return re.sub(r"\s+", " ", out).strip()


def _extract_claim(text: str, cite_start: int, cite_end: int) -> str:
    """Return the sentence containing the cite, with the cite call stripped out.

    A "sentence" boundary is the nearest preceding ``[.!?]\\s+`` (or paragraph
    start) and the nearest following ``[.!?]\\s`` (or end). For LaTeX prose this
    is good enough — math periods rarely surround \\cite{}.
    """
    # Walk back to start of sentence (or paragraph).
    start = 0
    for m in re.finditer(r"(?:[.!?]\s+)|(?:\n\s*\n)", text[:cite_start]):
        start = m.end()
    # Walk forward to next sentence end.
    after = text[cite_end:]
    end_m = re.search(r"[.!?](?:\s|\n|$)", after)
    end = cite_end + (end_m.end() if end_m else len(after))
    return _strip_cite_calls(text[start:end])


def _entry_to_audit_inputs(entry: dict) -> tuple[str, str, Optional[int], str]:
    """Extract (title, authors_str, year, venue) from a bibtex entry — cleaned up."""
    title = re.sub(r"[{}]", "", entry.get("title", "") or "").strip()
    authors = re.sub(r"[{}]", "", entry.get("author", "") or "").strip()
    year: Optional[int] = None
    ystr = entry.get("year", "") or ""
    m = re.search(r"\d{4}", ystr)
    if m:
        year = int(m.group(0))
    venue = (
        entry.get("booktitle") or entry.get("journal") or entry.get("publisher") or ""
    )
    venue = re.sub(r"[{}]", "", venue).strip()
    return title, authors, year, venue


def _nearest_bib(tex_file: Path, project_root: Path) -> Optional[Path]:
    """Walk up from ``tex_file``'s directory looking for the closest ``.bib`` file,
    stopping at ``project_root``. Returns ``None`` if no ``.bib`` is found in the
    chain. Used by ``--per-dir-bib`` so a project tree with multiple subprojects
    (each with their own bibliography) audits correctly against each subproject's
    own .bib.
    """
    tex_dir = tex_file.parent.resolve()
    root = project_root.resolve()
    current = tex_dir
    while True:
        bib_candidates = sorted(current.glob("*.bib"))
        if bib_candidates:
            # Prefer "references.bib" if it exists, else first alphabetically.
            for c in bib_candidates:
                if c.name == "references.bib":
                    return c
            return bib_candidates[0]
        if current == root or current.parent == current:
            return None
        current = current.parent


def _gather_citations(project_root: Path) -> tuple[list[CitationCheck], int]:
    """Walk project_root for .tex files and build a CitationCheck for every
    ``\\cite{}`` occurrence (one per key per location).

    Returns ``(checks, num_tex_files_scanned)``.
    """
    checks: list[CitationCheck] = []
    tex_files = [
        p
        for p in project_root.rglob("*.tex")
        if not any(part in {".git", ".venv", "node_modules", "venv"} for part in p.parts)
    ]
    for tex in tex_files:
        try:
            raw = tex.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        # Strip LaTeX line comments first (preserves offsets via space-padding) so
        # commented-out \cite{} calls are not audited as real citations.
        text = _strip_comments(raw)
        # Precompute line-start offsets for fast line-number lookup.
        line_starts = [0]
        for i, ch in enumerate(text):
            if ch == "\n":
                line_starts.append(i + 1)

        for m in _CITE_RE.finditer(text):
            line_no = bisect.bisect_right(line_starts, m.start())
            claim = _extract_claim(text, m.start(), m.end())
            for raw_key in m.group(1).split(","):
                key = raw_key.strip()
                if not key:
                    continue
                checks.append(
                    CitationCheck(
                        cite_key=key,
                        file=tex,
                        line=line_no,
                        char_offset=m.start(),
                        claim_text=claim,
                    )
                )
    return checks, len(tex_files)


# Heuristic: does this claim contain quantitative / named-benchmark content
# that metadata alone (Tier 0) cannot reasonably verify? Used as a safety net
# to downgrade Tier-0 high-confidence "supports=true" verdicts to
# "unverifiable" — preventing the LLM from rubber-stamping fabricated numbers
# on the basis of a topic-ish title.
_QUANTITATIVE_CLAIM_RE = re.compile(
    r"""
    \b\d+\s*%                          # "86.5%"
    | \b\d[\d,.]*\s*(?:M|B|K)?\s*(?:parameters|params|layers|epochs|steps)\b
    | \b(?:F1|BLEU|ROUGE|MMLU|MedQA|HumanEval|GLUE|SuperGLUE|ImageNet|
          LibriSpeech|SQuAD|WMT|COCO|CIFAR|MNIST)\b   # named benchmarks
    | \bstate[- ]of[- ]the[- ]art\b
    | \b\d+\s*(?:billion|million|trillion)\s+parameters?\b
    """,
    re.IGNORECASE | re.VERBOSE,
)


def _is_quantitative_claim(claim: str) -> bool:
    """True if the claim contains a specific number, benchmark name, or other
    quantitative content that metadata alone cannot verify."""
    return bool(_QUANTITATIVE_CLAIM_RE.search(claim or ""))


# Pattern matches the specific *value* in a quantitative claim. Used to build
# a "contradiction query" — same entities and benchmark names but stripped of
# the claim's specific number, so retrieval surfaces chunks reporting the
# paper's ACTUAL value for the same entity.
_QUANTITATIVE_VALUE_RE = re.compile(
    r"""
    \b\d+(?:\.\d+)?\s*%                            # "86.5%"
    | \b\d[\d,.]*\s*(?:M|B|K)?\s*(?:billion|million|trillion)?\s*  # "100 billion"
      (?:parameters|params|layers|epochs|steps)\b
    | \b\d+(?:\.\d+)?\s*(?:billion|million|trillion)\s+parameters?\b
    """,
    re.IGNORECASE | re.VERBOSE,
)


def _make_contradiction_query(claim: str) -> Optional[str]:
    """Build a retrieval query that targets contradicting evidence for a
    quantitative claim.

    Strategy: strip the specific *value* (e.g. "86.5%", "100 billion params")
    while keeping the discriminating entities (benchmark name, model name).
    The resulting query retrieves chunks discussing the same benchmark/entity
    with whatever value the paper actually reports — if that differs from the
    claim's value, the LLM judge sees both and can output contradicted=true.

    Returns ``None`` if the claim isn't quantitative or has nothing to strip
    (in which case the regular hybrid retrieval is sufficient).

    Example::

        "GPT-3 achieves 86.5% on MedQA"   →   "GPT-3 achieves on MedQA"
        "ResNet-50 with 100 billion params" →  "ResNet-50 with"
    """
    if not _is_quantitative_claim(claim):
        return None
    stripped = _QUANTITATIVE_VALUE_RE.sub("", claim).strip()
    # Collapse any double spaces created by the substitution.
    stripped = re.sub(r"\s{2,}", " ", stripped)
    # If stripping removed substantively all content, retrieval would be too
    # broad — skip the contradiction query and rely on the normal retrieval.
    if len(stripped.split()) < 3:
        return None
    if stripped == claim:
        return None
    return stripped


# --- main pipeline ---------------------------------------------------------


async def audit_project(
    project_root: Path,
    bib_file: Path,
    *,
    tier: int = 1,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    delay_seconds: float = 0.5,
    fix: bool = False,
    confidence_floor: float = 0.7,
    cache_dir: Optional[Path] = None,
    no_cache: bool = False,
    rag_top_k: int = 5,
    embedding_model: str = "auto",
    embedding_backend: str = "auto",
    ss_api_key: Optional[str] = None,
    per_dir_bib: bool = False,
    use_memory: bool = True,
) -> AuditReport:
    """Audit every ``\\cite{}`` in ``project_root`` against the ``bib_file``.

    ``tier`` selects the evidence depth fed to the LLM judge:

      * 0 — metadata only (title + authors + year + venue from the .bib entry).
        Cheap, catches gross topic mismatches.
      * 1 — also fetch the paper's abstract from arXiv / Semantic Scholar /
        Crossref and include it in the prompt. Catches misattributions where
        the title is on-topic but the abstract reveals a different actual
        contribution. Adds ~1 HTTP call per unique paper (cached).
      * 2 — also download the paper PDF (if open-access) via the source's
        ``pdf_url``, extract text, chunk it, embed the chunks, retrieve the
        top-K most-claim-relevant chunks, and include them as evidence.
        Catches specific numerical / factual mismatches. Adds PDF download
        plus an embedding call per paper (both cached).

    Higher tiers gracefully degrade when content isn't available — e.g., a
    paper not on arXiv/SS/Crossref still gets a Tier-0 audit.
    """
    project_root = project_root.resolve()
    bib_file = bib_file.resolve()
    dbg.trace(
        "audit.start",
        project=str(project_root),
        bib=str(bib_file),
        tier=tier,
        fix=fix,
    )

    report = AuditReport(project_root=project_root, bib_file=bib_file)

    checks, tex_count = _gather_citations(project_root)
    report.tex_files_scanned = tex_count
    report.checks = checks

    # Build a per-cite-check lookup dict: each check resolves against EITHER the
    # global ``bib_file`` (default) or the nearest .bib in its .tex's directory
    # chain (when per_dir_bib=True). We pre-load every .bib we'll need so we
    # don't re-parse the same file for every cite that uses it.
    bib_dbs: dict[Path, dict] = {}

    def _load_bib(path: Path) -> dict:
        path = path.resolve()
        if path not in bib_dbs:
            try:
                db = bibtex.load(path)
                bib_dbs[path] = {e.get("ID"): e for e in db.entries}
            except Exception as e:
                dbg.trace("audit.bib", "load failed", path=str(path), error=str(e))
                bib_dbs[path] = {}
        return bib_dbs[path]

    # Tex-to-bib mapping
    tex_to_bib: dict[Path, Path] = {}
    if per_dir_bib:
        for check in checks:
            tex = check.file.resolve()
            if tex in tex_to_bib:
                continue
            nearest = _nearest_bib(tex, project_root)
            tex_to_bib[tex] = nearest if nearest else bib_file
        report.bib_files_used = tex_to_bib
        dbg.trace(
            "audit.per_dir_bib",
            "resolved",
            mappings={str(k): str(v) for k, v in tex_to_bib.items()},
        )
    # Default load of the explicit --bib if not per_dir_bib (or as a fallback).
    if not per_dir_bib:
        _load_bib(bib_file)

    dbg.trace(
        "audit.scan",
        f"{len(checks)} citation occurrences across {tex_count} .tex files",
        unique_keys=len({c.cite_key for c in checks}),
        per_dir_bib=per_dir_bib,
    )

    # Cache directories (used for Tier 1+ paper content, Tier 2 PDFs/embeddings).
    if cache_dir is None:
        from platformdirs import user_cache_dir
        cache_dir = Path(user_cache_dir("bibsync", "bibsync"))
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Memory — per-project decision recall. Enabled by default; --no-memory
    # makes it inert without changing the call shape. Reads short-circuit
    # LLM calls when we've previously judged the same (claim, paper) pair
    # at the same-or-higher tier. Writes capture each new verdict so the
    # next run can recall.
    from . import memory as memory_mod
    mem = memory_mod.open_memory(
        project_root=project_root, enabled=use_memory, cache_dir=cache_dir,
    )
    dbg.trace(
        "audit.memory",
        "enabled" if use_memory else "disabled",
        project_id=mem.project_id,
    )

    paper_cache = None
    if tier >= 1:
        from .audit_sources import PaperContentCache  # lazy: keeps tier-0 cheap
        paper_cache = PaperContentCache(cache_dir)

    pdf_cache = None
    embed_store = None
    if tier >= 2:
        from .audit_sources.pdf import PdfCache
        from .audit_rag import EmbeddingStore
        pdf_cache = PdfCache(cache_dir)
        embed_store = EmbeddingStore(
            cache_dir,
            model=embedding_model,
            api_key=api_key,
            backend=embedding_backend,
        )

    # Cache enrichment + verdicts per cite-key so multiple usages of the same key
    # in different claims still only fetch the paper / embed it / get its abstract
    # once per run.
    paper_content_by_key: dict[str, Optional["PaperContent"]] = {}
    paper_chunks_by_key: dict[str, list] = {}
    verdict_cache: dict[tuple[str, str], llm.CitationAudit] = {}
    paced_keys: set[str] = set()

    for check in checks:
        # Step 1 — bib lookup. Resolve which .bib applies to this check.
        if per_dir_bib:
            this_bib = tex_to_bib.get(check.file.resolve(), bib_file)
        else:
            this_bib = bib_file
        bib_by_key = _load_bib(this_bib)
        entry = bib_by_key.get(check.cite_key)
        if entry is None:
            check.status = "missing_in_bib"
            check.confidence = 1.0
            check.reasoning = (
                f"no entry with this key in {this_bib.name}"
                if per_dir_bib
                else "no entry with this key in the .bib"
            )
            dbg.trace(
                "audit.check",
                "missing_in_bib",
                key=check.cite_key,
                bib=str(this_bib),
            )
            continue
        check.bib_entry = entry

        title, authors, year, venue = _entry_to_audit_inputs(entry)

        # Step 2 — Tier 1+: fetch abstract via the source fallback chain.
        abstract: Optional[str] = None
        content = None
        if tier >= 1:
            if check.cite_key in paper_content_by_key:
                content = paper_content_by_key[check.cite_key]
            else:
                from .audit_sources import fetch_paper_content
                first_author = ""
                if authors:
                    first = authors.split(" and ")[0]
                    first_author = (
                        first.split(",")[0] if "," in first else (first.split()[-1] if first.split() else "")
                    ).strip()
                content = await fetch_paper_content(
                    title=title,
                    first_author=first_author or None,
                    year=year,
                    doi=(entry.get("doi") or None),
                    cache=paper_cache,
                    no_cache=no_cache,
                    ss_api_key=ss_api_key,
                )
                paper_content_by_key[check.cite_key] = content
            if content and content.abstract:
                abstract = content.abstract
        check.paper_content = content

        # Step 3 — Tier 2: PDF → chunks → retrieve top-K relevant chunks for THIS claim.
        retrieved_chunk_texts: Optional[list[str]] = None
        tier2_failure_reason = ""
        if tier >= 2 and pdf_cache and embed_store:
            if not content:
                tier2_failure_reason = "source_not_found"
            elif not content.pdf_url:
                tier2_failure_reason = "no_open_access_pdf"
            else:
                from .audit_sources.pdf import get_paper_text
                from .audit_rag import chunk_text

                paper_key = content.stable_key()
                if check.cite_key in paper_chunks_by_key:
                    chunks = paper_chunks_by_key[check.cite_key]
                else:
                    text = await get_paper_text(paper_key, content.pdf_url, pdf_cache)
                    chunks = chunk_text(text, paper_key) if text else []
                    paper_chunks_by_key[check.cite_key] = chunks

                if not chunks:
                    tier2_failure_reason = "pdf_download_or_extract_failed"
                else:
                    top = await embed_store.retrieve(
                        query=check.claim_text,
                        paper_key=paper_key,
                        chunks=chunks,
                        top_k=rag_top_k,
                    )
                    if not top:
                        tier2_failure_reason = "embedding_failed"
                    else:
                        retrieved_chunk_texts = [
                            f"[p.{c.page or '?'}] {c.text}" for c, _score in top
                        ]
                        check.n_chunks = len(retrieved_chunk_texts)

                        # Contradiction retrieval — only when the claim is
                        # quantitative AND a value-stripped query is non-trivial.
                        # Retrieves a SMALL extra pool of chunks (top-3) using a
                        # query that targets the same entity without the claim's
                        # specific value, so the LLM can spot conflicting numbers.
                        contradiction_q = _make_contradiction_query(check.claim_text)
                        if contradiction_q:
                            extra = await embed_store.retrieve(
                                query=contradiction_q,
                                paper_key=paper_key,
                                chunks=chunks,
                                top_k=3,
                            )
                            # De-dupe: only add chunks not already in `top`.
                            already = {id(c) for c, _ in top}
                            extra_texts = [
                                f"[p.{c.page or '?'}] {c.text}"
                                for c, _ in extra
                                if id(c) not in already
                            ]
                            if extra_texts:
                                retrieved_chunk_texts.extend(extra_texts)
                                check.n_chunks = len(retrieved_chunk_texts)
                                dbg.trace(
                                    "audit.contradict",
                                    "added chunks",
                                    n_extra=len(extra_texts),
                                    query=contradiction_q[:80],
                                )

        # Record what tier we actually achieved (may be lower than the requested
        # ``tier`` if the paper wasn't on arXiv/SS/Crossref or had no PDF).
        if retrieved_chunk_texts:
            check.evidence_tier = 2
        elif abstract:
            check.evidence_tier = 1
        else:
            check.evidence_tier = 0

        # Record WHY we degraded so the end-of-run summary can surface it.
        if tier >= 1 and check.evidence_tier < 1 and not content:
            check.degraded_reason = "source_not_found"
        if tier >= 2 and check.evidence_tier < 2 and tier2_failure_reason:
            # Only overwrite if Tier 2 actually had a reason — keep tier-1 reason otherwise.
            check.degraded_reason = tier2_failure_reason

        # Step 4 — Memory recall + LLM audit.
        #
        # Three-tier short-circuit (cheapest first):
        #   a. In-run verdict_cache keyed by (cite_key, claim_text) — already
        #      judged in THIS run. O(1) lookup, no I/O.
        #   b. CROSS-RUN memory recall keyed by (claim_text, paper_key) — judged
        #      in a previous run at the same-or-higher tier. Reads JSONL.
        #   c. LLM call to llm.audit_citation.
        cache_key = (check.cite_key, check.claim_text)
        memory_hit: Optional[llm.CitationAudit] = None
        # Identify the paper for cross-run memory keying. Prefer the
        # PaperContent stable_key (arxiv → doi → title-hash) when we have
        # it; fall back to a doi/arxiv from the bib entry; finally the
        # cite_key (project-local, less stable but better than nothing).
        paper_key_for_mem = ""
        if content is not None:
            paper_key_for_mem = content.stable_key()
        elif entry.get("doi"):
            paper_key_for_mem = "doi-" + entry["doi"].replace("/", "_")
        elif entry.get("eprint") or entry.get("arxiv"):
            paper_key_for_mem = "arxiv-" + (entry.get("eprint") or entry["arxiv"])
        else:
            paper_key_for_mem = "cite-" + check.cite_key

        if cache_key in verdict_cache and not retrieved_chunk_texts:
            verdict = verdict_cache[cache_key]
            dbg.trace("audit.check", "in-run cache hit", key=check.cite_key)
        else:
            # Memory recall — only at the SAME-or-higher achieved tier so we
            # don't carry a tier-0 verdict into a tier-2 request that has
            # better evidence available.
            recalls = mem.recall(
                check.claim_text,
                paper_key=paper_key_for_mem,
                types=["verdict", "override"],
            )
            best_recall = None
            for r in recalls:
                if r.tier >= check.evidence_tier:
                    best_recall = r
                    break
            if best_recall is not None:
                # Reconstruct a CitationAudit shape from the stored decision.
                supports = best_recall.decision == "verified"
                contradicted = best_recall.decision == "contradicted"
                memory_hit = llm.CitationAudit(
                    supports=supports,
                    confidence=best_recall.confidence,
                    reasoning=(
                        f"[memory] {best_recall.rationale} "
                        f"(from {best_recall.ts}, tier={best_recall.tier})"
                    ),
                    contradicted=contradicted,
                )
                verdict = memory_hit
                dbg.trace(
                    "audit.memory",
                    "RECALL — skipped LLM call",
                    key=check.cite_key,
                    paper_key=paper_key_for_mem,
                    decision=best_recall.decision,
                    age=best_recall.ts,
                )
            else:
                verdict = llm.audit_citation(
                    claim_text=check.claim_text,
                    cited_paper_title=title,
                    cited_paper_authors=authors,
                    cited_paper_year=year,
                    cited_paper_venue=venue,
                    abstract=abstract,
                    retrieved_chunks=retrieved_chunk_texts,
                    model=model,
                    api_key=api_key,
                )
                if check.cite_key not in paced_keys and delay_seconds > 0:
                    await asyncio.sleep(delay_seconds)
                paced_keys.add(check.cite_key)
            verdict_cache[cache_key] = verdict

        check.confidence = verdict.confidence
        check.reasoning = verdict.reasoning

        # SAFETY NET — if we only had metadata (Tier 0) and the claim contains
        # quantitative / named-benchmark content, refuse to high-confidence
        # verify on title-alone. This catches the case where the LLM ignores
        # the Tier-0 prompt rule and still returns supports=true on a number-
        # bearing claim. We don't FLIP to "hallucinated" — that would risk
        # auto-deleting good citations — we route to "unverifiable" so the
        # user sees the gap and can re-run at higher tier.
        if (
            check.evidence_tier == 0
            and verdict.supports
            and _is_quantitative_claim(check.claim_text)
        ):
            check.status = "unverifiable"
            check.reasoning = (
                "metadata-only evidence cannot verify a quantitative / "
                "named-benchmark claim; re-run with --tier 1 or --tier 2"
            )
            dbg.trace(
                "audit.safety",
                "downgraded Tier-0 supports=true on quantitative claim",
                key=check.cite_key,
                claim=check.claim_text[:80],
            )
        elif verdict.supports:
            check.status = "verified"
        elif verdict.contradicted and verdict.confidence >= confidence_floor:
            # Distinct from `hallucinated`: the paper exists, IS the right
            # canonical source for the topic, but reports a DIFFERENT value
            # than the claim states. User action: fix the prose, not the
            # citation. --fix DOES NOT auto-edit these (different from
            # hallucinated) because the cite itself is correct.
            check.status = "contradicted"
        elif verdict.confidence >= confidence_floor:
            check.status = "hallucinated"
        else:
            check.status = "unverifiable"

        dbg.trace(
            "audit.check",
            check.status,
            key=check.cite_key,
            tier=check.evidence_tier,
            conf=round(check.confidence, 2),
            line=check.line,
        )

        # Persist this verdict to memory — but only when we actually ran the
        # LLM (not when the verdict came from memory itself). Skip
        # missing_in_bib because there's no paper to key on.
        if (
            memory_hit is None
            and check.status != "missing_in_bib"
            and paper_key_for_mem
        ):
            mem.remember(
                type_="verdict",
                claim_text=check.claim_text,
                paper_key=paper_key_for_mem,
                cite_key=check.cite_key,
                decision=check.status,
                tier=check.evidence_tier,
                confidence=check.confidence,
                source="audit",
                rationale=check.reasoning,
                scope="project",
            )

    # Step 5 — Optional --fix: replace hallucinated cite calls with marker comments.
    if fix:
        _apply_fixes(report)

    return report


def _apply_fixes(report: AuditReport) -> None:
    """Per-occurrence rewrite of hallucinated ``\\cite{}`` calls.

    KEY DETAIL: the same cite key can be used correctly in one location and
    hallucinated in another (e.g., ``vaswani2017attention`` for a real
    self-attention claim AND for a fabricated LibriSpeech claim). We must only
    rewrite the occurrence at the hallucinated location, not every occurrence of
    that key. This requires tracking the character offset of each citation —
    set in :func:`_gather_citations` and used here to scope the rewrite.

    Multi-key calls (``\\cite{a,b}``) where only one key is bad keep the good
    keys intact in the rewritten brace list.
    """
    by_file: dict[Path, list[CitationCheck]] = {}
    for c in report.checks:
        if c.status == "hallucinated":
            by_file.setdefault(c.file, []).append(c)

    for file, bad_checks in by_file.items():
        try:
            text = file.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue

        # Group hallucinated checks by the char_offset of the cite call they belong
        # to. A single ``\\cite{a,b,c}`` with two hallucinated keys is one offset
        # with two CitationChecks; we want to rewrite that one cite call.
        bad_by_offset: dict[int, list[CitationCheck]] = {}
        for c in bad_checks:
            bad_by_offset.setdefault(c.char_offset, []).append(c)

        # Build a list of (start, end, replacement) edits by re-finding the cite
        # calls in the comment-stripped text and matching offsets we recorded.
        text_stripped = _strip_comments(text)
        edits: list[tuple[int, int, str, list[CitationCheck]]] = []
        for m in _CITE_RE.finditer(text_stripped):
            if m.start() not in bad_by_offset:
                continue
            bad_here = bad_by_offset[m.start()]
            bad_keys_here = {c.cite_key for c in bad_here}
            reasons = {c.cite_key: c.reasoning for c in bad_here}

            keys = [k.strip() for k in m.group(1).split(",") if k.strip()]
            remaining = [k for k in keys if k not in bad_keys_here]
            removed = [k for k in keys if k in bad_keys_here]
            if not removed:
                continue

            comment_parts = [
                f"\\cite{{{k}}} — {reasons.get(k, 'topic mismatch')}" for k in removed
            ]
            comment = (
                "  % [bibsync audit] removed hallucinated: "
                + "; ".join(comment_parts)
            )
            if remaining:
                # Preserve the command (e.g. ``\\citep`` vs ``\\cite``) and any
                # bracket options before the brace.
                head = m.group(0).split("{", 1)[0]
                replacement = f"{head}{{{', '.join(remaining)}}}{comment}"
            else:
                replacement = comment.lstrip()

            edits.append((m.start(), m.end(), replacement, bad_here))

        if not edits:
            continue

        # Apply edits in reverse offset order so earlier offsets stay valid as we
        # rewrite the file.
        edits.sort(key=lambda t: t[0], reverse=True)
        new_text = text
        for start, end, repl, _ in edits:
            new_text = new_text[:start] + repl + new_text[end:]

        file.write_text(new_text, encoding="utf-8")
        for _, _, _, bad_here in edits:
            for c in bad_here:
                c.fixed = True
        dbg.trace(
            "audit.fix", "rewritten",
            file=str(file), edits=len(edits),
        )


def audit_project_sync(*args, **kwargs) -> AuditReport:
    return asyncio.run(audit_project(*args, **kwargs))
