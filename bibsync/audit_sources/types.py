"""Shared data types for the audit-sources fallback chain."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class PaperContent:
    """Paper metadata + abstract (and optionally a PDF URL) fetched from an
    open-access source.

    Returned by :func:`bibsync.audit_sources.fetch_paper_content`. ``source``
    records which backend produced the record so the trace makes the fallback
    chain visible.
    """

    title: str
    abstract: Optional[str] = None
    authors: list[str] = field(default_factory=list)
    year: Optional[int] = None
    venue: Optional[str] = None
    doi: Optional[str] = None
    arxiv_id: Optional[str] = None
    pdf_url: Optional[str] = None
    source: str = ""  # 'arxiv' | 'semantic_scholar' | 'crossref' | 'cache' | ''
    fetched_at: Optional[str] = None  # ISO 8601 timestamp

    def first_author_surname(self) -> str:
        if not self.authors:
            return ""
        a = self.authors[0]
        if "," in a:
            return a.split(",", 1)[0].strip()
        parts = a.split()
        return parts[-1] if parts else ""

    def stable_key(self) -> str:
        """A stable identifier we can use as a filename for PDF/embedding caches.
        Prefers arxiv_id, then doi, then a hash of the normalized title."""
        if self.arxiv_id:
            return f"arxiv-{self.arxiv_id.replace('/', '_')}"
        if self.doi:
            return f"doi-{self.doi.replace('/', '_')}"
        import hashlib
        return "title-" + hashlib.sha256(
            (self.title or "").lower().strip().encode()
        ).hexdigest()[:16]
