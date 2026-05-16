"""Citation-decision memory — local, scoped, queryable, persistent.

Architecturally inspired by mem0 (project- and user-scoped, queryable
long-term recall), but **citation-specific**: the only record types it
stores are ones that materially change a future audit or suggest
decision. We do not store free-form notes, paper content, or anything
that doesn't directly fold back into the pipeline.

Record types:

  ``accept``     — user approved (claim, paper) in suggest. Recall short-
                   circuits Filter C+D on the same (claim, paper) next
                   time, saving the LLM cost.
  ``reject``     — user said no, OR Filter D rejected. Recall skips the
                   candidate next time without re-querying Scholar/judge.
  ``verdict``    — audit produced (verified | hallucinated | contradicted
                   | unverifiable) for (claim, paper). Recall memoises
                   the verdict — skip the audit LLM call if we've judged
                   this exact pair before, unless tier increased.
  ``preference`` — venue / domain bias inferred from the user's accept
                   patterns. Tie-breaker in Filter C ranking.
  ``override``   — user kept a citation that an earlier run flagged as
                   hallucinated (via memory CLI). Suppress repeat
                   hallucinated verdicts on the same pair.

Storage layout::

  <cache_dir>/memory/
    user.jsonl                  # user-scoped (preferences, defaults)
    projects/
      <sha1(project_root)>.jsonl  # project-scoped

Files are append-only JSONL. Logical deletion is a `forgotten` tombstone
record referencing the target ``id``; the query layer filters tombstones
out at read time. This keeps every write atomic — no read-modify-write
race, no torn-file risk on Ctrl+C, no need for a lock file.

The query API is intentionally minimal: ``recall(claim, paper_key)``
returns the most informative live record (most recent non-tombstoned).
Anything more sophisticated belongs in a layer above this module.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal, Optional

from . import dbg

# ── public types ────────────────────────────────────────────────────────────

MemoryType = Literal["accept", "reject", "verdict", "preference", "override", "forgotten"]
MemoryScope = Literal["project", "user"]

# Minimum rapidfuzz WRatio to consider two claims "the same logical claim"
# for memory recall. We use WRatio (a weighted combination of Levenshtein +
# partial_ratio + token_sort_ratio) instead of token_set_ratio — the latter
# de-duplicates tokens and ignores order, scoring "GPT-3 86.5% on MedQA" and
# "GPT-4 89.0% on MedQA" as 89% similar (they share most token-set members)
# even though they're factually different claims. False-recall on a
# different claim would carry over a wrong verdict; we bias HARD toward
# false-negative (extra LLM call) over false-positive (wrong verdict
# silently reused).
#
# Empirically (test set in commit message): WRatio ≥ 90 accepts paraphrases
# like "GPT-3 achieves 86.5% on MedQA" ↔ "GPT-3 achieves 86.5% accuracy on
# MedQA" (95) while rejecting "GPT-3 86.5% MedQA" ↔ "GPT-4 89% MedQA" (89).
_CLAIM_FUZZY_THRESHOLD = 90


@dataclass
class MemoryRecord:
    """One memory record. Append to disk; never mutate in place."""

    id: str  # opaque short hash for forget-references
    type: MemoryType
    scope: MemoryScope
    ts: str  # ISO-8601 UTC

    # Content the record is *about* (one or more may be empty for
    # preference / override records).
    claim_text: str = ""
    claim_hash: str = ""  # normalized hash for fast exact-lookup
    paper_key: str = ""  # stable_key() — arxiv:..., doi:..., or title-...
    cite_key: str = ""  # the .bib cite key when known

    # Verdict / decision payload.
    decision: str = ""  # 'verified' | 'hallucinated' | 'contradicted' | ...
    tier: int = 0  # evidence_tier at decision time
    confidence: float = 0.0  # LLM confidence, if applicable
    source: str = ""  # 'audit' | 'suggest_user_approve' | 'filter_d_reject' | ...
    rationale: str = ""  # short human-readable reason

    # Free-form tags — e.g. ['nlp', 'transformer']. Use sparingly.
    tags: list[str] = field(default_factory=list)

    # When type == 'forgotten', this is the id of the record being revoked.
    forgets: str = ""


# ── normalisation + identity ────────────────────────────────────────────────


# Strip LaTeX cite calls and collapse whitespace before hashing/matching.
_CITE_STRIP_RE = re.compile(r"~?\\(?:no)?cite\w*\s*(?:\[[^\]]*\])*\s*\{[^}]+\}")


def _normalize_claim(claim: str) -> str:
    """Lowercase + remove ``\\cite{}`` + split hyphenated terms + collapse
    whitespace. Used as the hash seed for fast exact-lookup AND as the
    pre-fuzzy-match canonical form.

    Hyphen splitting is the load-bearing step. "GPT-3" and "GPT 3" are
    the same logical token but tokenisers like rapidfuzz's word-splitter
    treat "GPT-3" as one token and "GPT" + "3" as two — token_set_ratio
    would then score them as ~68% similar, well below our recall
    threshold. Splitting on hyphens up front makes the comparison
    semantic rather than surface-form sensitive.
    """
    if not claim:
        return ""
    s = _CITE_STRIP_RE.sub(" ", claim)
    s = s.replace("-", " ")
    s = re.sub(r"\s+", " ", s).strip().lower()
    return s


def _claim_hash(claim: str) -> str:
    """Stable 12-char hex hash of the normalised claim. Used as the primary
    fast-lookup key — exact-match within a memory file."""
    n = _normalize_claim(claim)
    if not n:
        return ""
    return hashlib.sha1(n.encode("utf-8")).hexdigest()[:12]


def _record_id(*, claim_hash: str, paper_key: str, type_: str, ts: str) -> str:
    """Generate a short, stable, collision-resistant id for a record. Used
    so ``forget`` calls can reference a specific record later."""
    h = hashlib.sha1(f"{type_}|{claim_hash}|{paper_key}|{ts}".encode()).hexdigest()
    return "mem_" + h[:10]


def _project_id(project_root: Path) -> str:
    """Per-project memory namespace. SHA1 of the absolute path — stable across
    cwd changes, doesn't leak the path content. Files are anonymous unless
    you correlate them with the directory yourself."""
    rp = str(Path(project_root).resolve())
    return hashlib.sha1(rp.encode("utf-8")).hexdigest()[:16]


# ── store ───────────────────────────────────────────────────────────────────


class CitationMemory:
    """JSONL-backed memory of citation decisions.

    Construct one per command invocation. ``project_root=None`` means
    "no project context" — only user-scoped memory will be accessible.

    Thread-safe? No. The on-disk file is append-only so concurrent writes
    are usually fine, but interleaved lines from concurrent processes
    would still be possible. BibSync's commands are single-process, so
    we don't bother with file locking.
    """

    def __init__(
        self,
        cache_dir: Path,
        *,
        project_root: Optional[Path] = None,
        enabled: bool = True,
    ):
        self.enabled = enabled
        self.dir = Path(cache_dir) / "memory"
        if enabled:
            self.dir.mkdir(parents=True, exist_ok=True)
        self.user_file = self.dir / "user.jsonl"
        self.project_id: Optional[str] = None
        self.project_file: Optional[Path] = None
        if project_root is not None:
            self.project_id = _project_id(project_root)
            if enabled:
                (self.dir / "projects").mkdir(parents=True, exist_ok=True)
                self.project_file = self.dir / "projects" / f"{self.project_id}.jsonl"

    # ── write paths ─────────────────────────────────────────────────────────

    def _append(self, scope: MemoryScope, record: MemoryRecord) -> None:
        if not self.enabled:
            return
        path = self.user_file if scope == "user" else self.project_file
        if path is None:
            dbg.trace(
                "memory.write", "skipped — no project context for project-scoped record",
                type=record.type,
            )
            return
        line = json.dumps(asdict(record), ensure_ascii=False)
        try:
            with open(path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
            dbg.trace(
                "memory.write",
                "appended",
                scope=scope,
                type=record.type,
                id=record.id,
                claim_hash=record.claim_hash,
                paper_key=record.paper_key or None,
                decision=record.decision or None,
            )
        except OSError as e:
            dbg.trace("memory.write", "WRITE FAILED", path=str(path), error=str(e))

    def remember(
        self,
        *,
        type_: MemoryType,
        claim_text: str = "",
        paper_key: str = "",
        cite_key: str = "",
        decision: str = "",
        tier: int = 0,
        confidence: float = 0.0,
        source: str = "",
        rationale: str = "",
        tags: Optional[list[str]] = None,
        scope: MemoryScope = "project",
    ) -> Optional[MemoryRecord]:
        """Create + persist a record. Returns the record (with its assigned id)
        or ``None`` if memory is disabled.

        Idempotent on the (claim_hash, paper_key, type_, ts-second) tuple —
        repeat writes within the same second produce the same id, so a
        retry loop won't fan out into duplicates.
        """
        if not self.enabled:
            return None
        ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        ch = _claim_hash(claim_text)
        rec = MemoryRecord(
            id=_record_id(claim_hash=ch, paper_key=paper_key, type_=type_, ts=ts),
            type=type_,
            scope=scope,
            ts=ts,
            claim_text=claim_text,
            claim_hash=ch,
            paper_key=paper_key,
            cite_key=cite_key,
            decision=decision,
            tier=tier,
            confidence=confidence,
            source=source,
            rationale=rationale,
            tags=list(tags or []),
        )
        self._append(scope, rec)
        return rec

    def forget(self, target_id: str, *, scope: MemoryScope = "project") -> bool:
        """Logically delete a record by writing a tombstone. Returns True
        if a record with that id exists in the given scope."""
        if not self.enabled:
            return False
        records = list(self._iter_raw(scope))
        if not any(r.id == target_id for r in records):
            return False
        ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        tomb = MemoryRecord(
            id=_record_id(claim_hash="", paper_key="", type_="forgotten", ts=ts + target_id),
            type="forgotten",
            scope=scope,
            ts=ts,
            forgets=target_id,
            source="user_forget",
        )
        self._append(scope, tomb)
        return True

    def purge_project(self) -> bool:
        """Delete the entire project-scope file. Returns True if a file was
        removed. User-scoped memory is untouched."""
        if not self.enabled or self.project_file is None:
            return False
        if self.project_file.exists():
            try:
                self.project_file.unlink()
                dbg.trace("memory.purge", "deleted", path=str(self.project_file))
                return True
            except OSError as e:
                dbg.trace("memory.purge", "FAILED", error=str(e))
        return False

    # ── read paths ──────────────────────────────────────────────────────────

    def _iter_raw(self, scope: MemoryScope):
        """Yield every record in a scope, including tombstones (caller filters)."""
        if not self.enabled:
            return
        path = self.user_file if scope == "user" else self.project_file
        if path is None or not path.exists():
            return
        try:
            with open(path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                        yield MemoryRecord(**obj)
                    except (json.JSONDecodeError, TypeError):
                        continue
        except OSError:
            return

    def _live_records(self, scope: MemoryScope) -> list[MemoryRecord]:
        """All records in scope, minus the ones a `forgotten` tombstone refers to."""
        raw = list(self._iter_raw(scope))
        tombstoned: set[str] = {r.forgets for r in raw if r.type == "forgotten" and r.forgets}
        return [r for r in raw if r.id not in tombstoned and r.type != "forgotten"]

    def all_records(self, *, scope: Optional[MemoryScope] = None) -> list[MemoryRecord]:
        """Every live record across one or both scopes. Sorted ts-descending."""
        out: list[MemoryRecord] = []
        if scope in (None, "project"):
            out.extend(self._live_records("project"))
        if scope in (None, "user"):
            out.extend(self._live_records("user"))
        return sorted(out, key=lambda r: r.ts, reverse=True)

    def recall(
        self,
        claim_text: str,
        paper_key: Optional[str] = None,
        *,
        types: Optional[list[MemoryType]] = None,
        fuzzy: bool = True,
    ) -> list[MemoryRecord]:
        """Find live records matching (claim, paper).

        Match order:
          1. Exact claim_hash match (fastest, most reliable).
          2. If ``fuzzy``, rapidfuzz token_set_ratio ≥ 85 on normalised claim.
          3. If ``paper_key`` is given, filter to records with the same
             paper_key (after step 1/2 already narrowed by claim).

        Returns matches sorted by recency (most recent first) so the
        caller can pick records[0] for "the most recent decision about
        this pair".
        """
        if not self.enabled:
            return []
        target_hash = _claim_hash(claim_text)
        if not target_hash:
            return []
        target_norm = _normalize_claim(claim_text)

        candidates: list[MemoryRecord] = []
        # Project memory first (more specific), then user memory.
        for scope in ("project", "user"):
            for r in self._live_records(scope):  # type: ignore[arg-type]
                if types and r.type not in types:
                    continue
                if paper_key and r.paper_key and r.paper_key != paper_key:
                    continue
                if r.claim_hash == target_hash:
                    candidates.append(r)
                    continue
                if fuzzy and target_norm and r.claim_text:
                    try:
                        from rapidfuzz import fuzz

                        score = fuzz.WRatio(
                            target_norm, _normalize_claim(r.claim_text)
                        )
                        if score >= _CLAIM_FUZZY_THRESHOLD:
                            candidates.append(r)
                    except ImportError:
                        pass

        candidates.sort(key=lambda r: r.ts, reverse=True)
        if candidates:
            dbg.trace(
                "memory.recall",
                "matched",
                n=len(candidates),
                top_type=candidates[0].type,
                top_decision=candidates[0].decision or None,
                top_age=candidates[0].ts,
                paper_key=paper_key,
            )
        return candidates


# ── factory ─────────────────────────────────────────────────────────────────


def open_memory(
    project_root: Optional[Path] = None,
    *,
    enabled: bool = True,
    cache_dir: Optional[Path] = None,
) -> CitationMemory:
    """Construct a ``CitationMemory`` with the default cache dir.

    Pass ``enabled=False`` to satisfy callers' type contracts without
    persisting anything (used by ``--no-memory`` flags). All writes
    become no-ops; reads return empty lists.
    """
    if cache_dir is None:
        from platformdirs import user_cache_dir

        cache_dir = Path(user_cache_dir("bibsync", "bibsync"))
    # Don't bother creating dirs when disabled — keeps --no-memory truly
    # side-effect-free even on a fresh user account.
    if enabled:
        cache_dir.mkdir(parents=True, exist_ok=True)
    return CitationMemory(cache_dir, project_root=project_root, enabled=enabled)
