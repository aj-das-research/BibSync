"""Citation-verification benchmark runner.

Loads JSONL test cases of the form

    {
      "id":         "...",
      "kind":       "audit" | "suggest",
      "category":   "...",         # free-form grouping label
      "claim":      "<prose claim text>",
      "bib_entry":  { "ID": "...", "ENTRYTYPE": "...", "title": "...", ... }
                    | null,        # null = expected "missing_in_bib"
      "expected":   "verified" | "hallucinated" | "unverifiable"
                    | "contradicted" | "missing_in_bib",
      "min_tier":   0 | 1 | 2,     # run case at MAX(--tier, min_tier)
      "notes":      "..."          # free-form rationale
    }

For each ``audit``-kind case, the runner constructs a synthetic CitationCheck,
runs the same machinery that ``bibsync audit`` uses (source fetch → optional
RAG → ``llm.audit_citation``), records the verdict, and tallies:

  - per-status confusion matrix
  - precision / recall / F1 per ``expected`` label
  - **false-deletion rate** — verified-expected cases that got predicted as
    hallucinated. This is the most important number for a tool that auto-
    rewrites .tex files; we surface it on every run.

Designed to share the production pipeline's caches (``~/Library/Caches/bibsync``)
so subsequent runs are mostly LLM cost.
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from . import config, dbg, llm

# Status values the runner understands. New values added as the pipeline grows
# new verdict types (e.g. ``contradicted`` from A4).
_KNOWN_STATUSES = ("verified", "hallucinated", "unverifiable", "contradicted", "missing_in_bib")


# ── case loading ────────────────────────────────────────────────────────────


@dataclass
class BenchmarkCase:
    """One labeled (claim, bib_entry, expected_verdict) triple."""

    id: str
    kind: str  # "audit" | "suggest"
    category: str
    claim: str
    bib_entry: Optional[dict]
    expected: str
    min_tier: int
    notes: str = ""

    @classmethod
    def from_dict(cls, d: dict) -> "BenchmarkCase":
        return cls(
            id=d["id"],
            kind=d.get("kind", "audit"),
            category=d.get("category", ""),
            claim=d.get("claim", ""),
            bib_entry=d.get("bib_entry"),
            expected=d.get("expected", "verified"),
            min_tier=int(d.get("min_tier", 0)),
            notes=d.get("notes", ""),
        )


def load_cases(path: Path) -> list[BenchmarkCase]:
    """Parse a JSONL benchmark file. One case per non-empty, non-comment line."""
    cases: list[BenchmarkCase] = []
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        s = raw.strip()
        if not s or s.startswith("//") or s.startswith("#"):
            continue
        try:
            obj = json.loads(s)
        except json.JSONDecodeError as e:
            raise ValueError(f"{path}:{line_no}: invalid JSON — {e}") from e
        case = BenchmarkCase.from_dict(obj)
        if case.expected not in _KNOWN_STATUSES:
            raise ValueError(
                f"{path}:{line_no}: case {case.id!r} has unknown expected "
                f"status {case.expected!r}; must be one of {_KNOWN_STATUSES}"
            )
        cases.append(case)
    return cases


# ── result + metrics ────────────────────────────────────────────────────────


@dataclass
class CaseResult:
    """One predicted vs expected outcome for a benchmark case."""

    case: BenchmarkCase
    predicted: str
    confidence: float
    evidence_tier: int  # what tier did the pipeline ACTUALLY achieve
    reasoning: str
    elapsed_sec: float

    @property
    def correct(self) -> bool:
        return self.predicted == self.case.expected


@dataclass
class BenchmarkReport:
    """Aggregate metrics across all run cases."""

    results: list[CaseResult] = field(default_factory=list)
    elapsed_sec: float = 0.0

    def confusion(self) -> dict[tuple[str, str], int]:
        """Return ``{(expected, predicted): count}`` across all results."""
        m: dict[tuple[str, str], int] = {}
        for r in self.results:
            k = (r.case.expected, r.predicted)
            m[k] = m.get(k, 0) + 1
        return m

    def accuracy(self) -> float:
        if not self.results:
            return 0.0
        return sum(1 for r in self.results if r.correct) / len(self.results)

    def false_deletion_rate(self) -> float:
        """Fraction of cases that were really ``verified`` but predicted
        ``hallucinated`` — i.e. cases where the auditor would auto-delete a
        valid citation if ``--fix`` were enabled. This is THE headline safety
        metric for a citation tool; we minimise this above all else.
        """
        ver_expected = [r for r in self.results if r.case.expected == "verified"]
        if not ver_expected:
            return 0.0
        wrong = sum(1 for r in ver_expected if r.predicted == "hallucinated")
        return wrong / len(ver_expected)

    def per_label_metrics(self) -> dict[str, dict[str, float]]:
        """Precision / recall / F1 per expected-status label."""
        out: dict[str, dict[str, float]] = {}
        all_labels = {r.case.expected for r in self.results} | {
            r.predicted for r in self.results
        }
        for label in sorted(all_labels):
            tp = sum(
                1 for r in self.results
                if r.case.expected == label and r.predicted == label
            )
            fp = sum(
                1 for r in self.results
                if r.case.expected != label and r.predicted == label
            )
            fn = sum(
                1 for r in self.results
                if r.case.expected == label and r.predicted != label
            )
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
            out[label] = {
                "support": tp + fn,
                "precision": prec,
                "recall": rec,
                "f1": f1,
            }
        return out

    def to_dict(self) -> dict:
        """Serialisable summary — for JSON output and CI baselines."""
        return {
            "n_cases": len(self.results),
            "accuracy": self.accuracy(),
            "false_deletion_rate": self.false_deletion_rate(),
            "elapsed_sec": self.elapsed_sec,
            "per_label": self.per_label_metrics(),
            "confusion": {f"{e}->{p}": c for (e, p), c in self.confusion().items()},
            "cases": [
                {
                    "id": r.case.id,
                    "category": r.case.category,
                    "expected": r.case.expected,
                    "predicted": r.predicted,
                    "correct": r.correct,
                    "confidence": round(r.confidence, 3),
                    "evidence_tier": r.evidence_tier,
                    "elapsed_sec": round(r.elapsed_sec, 2),
                    "reasoning": r.reasoning,
                }
                for r in self.results
            ],
        }


# ── runner ──────────────────────────────────────────────────────────────────


async def _run_audit_case(
    case: BenchmarkCase,
    *,
    tier: int,
    paper_cache,
    pdf_cache,
    embed_store,
    no_cache: bool,
    rag_top_k: int,
    model: Optional[str],
    api_key: Optional[str],
    confidence_floor: float,
) -> CaseResult:
    """Run a single ``audit``-kind case through the production pipeline."""
    t0 = time.monotonic()

    # missing_in_bib short-circuit — no LLM call, no fetch.
    if case.bib_entry is None:
        return CaseResult(
            case=case,
            predicted="missing_in_bib",
            confidence=1.0,
            evidence_tier=0,
            reasoning="no bib_entry provided",
            elapsed_sec=time.monotonic() - t0,
        )

    entry = case.bib_entry
    title = (entry.get("title") or "").strip("{}")
    authors = (entry.get("author") or "").strip("{}")
    year_str = entry.get("year", "")
    year: Optional[int] = None
    if year_str:
        try:
            year = int(year_str)
        except ValueError:
            pass
    venue = (
        entry.get("booktitle") or entry.get("journal") or entry.get("publisher") or ""
    ).strip("{}")

    abstract: Optional[str] = None
    content = None
    retrieved_chunks: Optional[list[str]] = None
    achieved_tier = 0

    # Tier 1 — fetch the paper's abstract.
    if tier >= 1 and paper_cache is not None:
        from .audit_sources import fetch_paper_content

        first_author = ""
        if authors:
            first = authors.split(" and ")[0]
            first_author = (
                first.split(",")[0]
                if "," in first
                else (first.split()[-1] if first.split() else "")
            ).strip()
        content = await fetch_paper_content(
            title=title,
            first_author=first_author or None,
            year=year,
            doi=entry.get("doi") or None,
            cache=paper_cache,
            no_cache=no_cache,
        )
        if content and content.abstract:
            abstract = content.abstract
            achieved_tier = 1

    # Tier 2 — PDF download → chunks → retrieve top-K against the claim.
    if (
        tier >= 2
        and content is not None
        and content.pdf_url
        and pdf_cache is not None
        and embed_store is not None
    ):
        from .audit_rag import chunk_text
        from .audit_sources.pdf import get_paper_text

        paper_key = content.stable_key()
        text = await get_paper_text(paper_key, content.pdf_url, pdf_cache)
        chunks = chunk_text(text, paper_key) if text else []
        if chunks:
            top = await embed_store.retrieve(
                query=case.claim, paper_key=paper_key, chunks=chunks, top_k=rag_top_k
            )
            if top:
                retrieved_chunks = [
                    f"[p.{c.page or '?'}] {c.text}" for c, _ in top
                ]
                achieved_tier = 2

    # source_resolution flag tells the judge whether any adapter returned the
    # paper. For fabricated citations all adapters miss, and the Tier-0 rule
    # in the system prompt refuses to verify on plausible-title-alone.
    source_resolution = "found" if content is not None else "empty"
    verdict = llm.audit_citation(
        claim_text=case.claim,
        cited_paper_title=title,
        cited_paper_authors=authors,
        cited_paper_year=year,
        cited_paper_venue=venue,
        abstract=abstract,
        retrieved_chunks=retrieved_chunks,
        source_resolution=source_resolution,
        model=model,
        api_key=api_key,
    )

    # Translate (supports, confidence, contradicted) → status using the
    # SHARED helper so the benchmark scores the same predictions a real
    # ``bibsync audit`` run would produce — including both safety nets.
    from .audit import verdict_to_status
    predicted, _reasoning = verdict_to_status(
        verdict,
        evidence_tier=achieved_tier,
        claim_text=case.claim,
        confidence_floor=confidence_floor,
        requested_tier=tier,
        source_resolved=(content is not None),
    )

    return CaseResult(
        case=case,
        predicted=predicted,
        confidence=verdict.confidence,
        evidence_tier=achieved_tier,
        reasoning=verdict.reasoning,
        elapsed_sec=time.monotonic() - t0,
    )


async def run_benchmark(
    cases: list[BenchmarkCase],
    *,
    tier: int = 2,
    cache_dir: Optional[Path] = None,
    no_cache: bool = False,
    rag_top_k: int = 5,
    embedding_backend: str = "auto",
    embedding_model: str = "auto",
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    confidence_floor: float = 0.7,
    only_kind: Optional[str] = None,
    progress_cb=None,
) -> BenchmarkReport:
    """Run every case in ``cases`` at ``tier`` (or each case's ``min_tier`` if higher).

    ``progress_cb(i, total, case, result)`` is called after each case so the
    CLI can stream progress.
    """
    if only_kind:
        cases = [c for c in cases if c.kind == only_kind]

    # Initialise caches ONCE — same pattern as audit.audit_project.
    paper_cache = None
    pdf_cache = None
    embed_store = None
    if any(c.kind == "audit" for c in cases) and tier >= 1:
        from platformdirs import user_cache_dir
        from .audit_sources import PaperContentCache

        if cache_dir is None:
            cache_dir = Path(user_cache_dir("bibsync", "bibsync"))
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        paper_cache = PaperContentCache(cache_dir)
    if any(c.kind == "audit" for c in cases) and tier >= 2:
        from .audit_rag import EmbeddingStore
        from .audit_sources.pdf import PdfCache

        pdf_cache = PdfCache(cache_dir)
        embed_store = EmbeddingStore(
            cache_dir,
            model=embedding_model,
            api_key=api_key,
            backend=embedding_backend,
        )

    report = BenchmarkReport()
    t0 = time.monotonic()
    for i, case in enumerate(cases, 1):
        case_tier = max(tier, case.min_tier)
        dbg.trace(
            "bench.case",
            "start",
            id=case.id,
            kind=case.kind,
            expected=case.expected,
            case_tier=case_tier,
        )
        if case.kind == "audit":
            r = await _run_audit_case(
                case,
                tier=case_tier,
                paper_cache=paper_cache,
                pdf_cache=pdf_cache,
                embed_store=embed_store,
                no_cache=no_cache,
                rag_top_k=rag_top_k,
                model=model,
                api_key=api_key,
                confidence_floor=confidence_floor,
            )
        else:
            # 'suggest'-kind cases not yet wired up — they need Scholar access
            # which can't run in CI. Future work.
            r = CaseResult(
                case=case, predicted="unverifiable", confidence=0.0,
                evidence_tier=0,
                reasoning=f"kind={case.kind!r} not yet supported by runner",
                elapsed_sec=0.0,
            )
        report.results.append(r)
        dbg.trace(
            "bench.case",
            "done",
            id=case.id,
            predicted=r.predicted,
            correct=r.correct,
            tier=r.evidence_tier,
            elapsed_sec=round(r.elapsed_sec, 2),
        )
        if progress_cb is not None:
            progress_cb(i, len(cases), case, r)
    report.elapsed_sec = time.monotonic() - t0
    return report


def run_benchmark_sync(*args, **kwargs) -> BenchmarkReport:
    return asyncio.run(run_benchmark(*args, **kwargs))
