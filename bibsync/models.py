"""Shared data model for paper search results.

Decouples the source (Google Scholar today, possibly Semantic Scholar / DBLP later)
from downstream consumers (canonical-version picker, BibTeX writer, verifier).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class PaperHit:
    """One result row from a scholar search.

    Multi-version groups (preprint + conference of the same paper) are represented
    by populating ``versions`` on a single PaperHit rather than producing separate hits.
    """

    title: str
    authors: list[str] = field(default_factory=list)
    year: Optional[int] = None
    venue: Optional[str] = None
    cited_by: int = 0
    cluster_id: Optional[str] = None
    cite_popup_url: Optional[str] = None
    bibtex_url: Optional[str] = None
    versions_url: Optional[str] = None
    versions: list["PaperHit"] = field(default_factory=list)
    raw_snippet: Optional[str] = None

    def short(self) -> str:
        authors = ", ".join(self.authors[:3]) + (" et al." if len(self.authors) > 3 else "")
        year = f" ({self.year})" if self.year else ""
        venue = f" — {self.venue}" if self.venue else ""
        return f"{self.title}{year} — {authors}{venue}"
