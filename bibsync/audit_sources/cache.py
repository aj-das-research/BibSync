"""On-disk JSON cache for fetched paper content.

Survives across runs so a 50-citation .bib doesn't re-hit Semantic Scholar /
Crossref / arXiv every time. Keyed by ``hash(normalized_title + year)`` so
records from different sources for the same paper collapse to one cache entry.
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

from .types import PaperContent


def _normalize_title(title: str) -> str:
    s = re.sub(r"[^\w\s]", " ", title.lower())
    return re.sub(r"\s+", " ", s).strip()


class PaperContentCache:
    """Per-paper JSON cache. Entries older than ``ttl_days`` are treated as stale
    and re-fetched (so abstracts get refreshed if a paper changes)."""

    def __init__(self, cache_dir: Path, ttl_days: int = 30):
        self.dir = cache_dir / "paper_content"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.ttl_days = ttl_days

    def _key(self, title: str, year: Optional[int]) -> str:
        norm = _normalize_title(title)
        salt = f"{norm}|{year or ''}"
        return hashlib.sha256(salt.encode()).hexdigest()[:16]

    def _path(self, key: str) -> Path:
        return self.dir / f"{key}.json"

    def get(self, title: str, year: Optional[int]) -> Optional[PaperContent]:
        path = self._path(self._key(title, year))
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        fetched = data.get("fetched_at")
        if fetched and self.ttl_days > 0:
            try:
                ts = datetime.fromisoformat(fetched)
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
                age = datetime.now(timezone.utc) - ts
                if age > timedelta(days=self.ttl_days):
                    return None
            except ValueError:
                pass
        try:
            return PaperContent(**data)
        except TypeError:
            # Schema changed since this entry was written — treat as stale.
            return None

    def put(self, content: PaperContent) -> None:
        if content.fetched_at is None:
            content.fetched_at = datetime.now(timezone.utc).isoformat()
        path = self._path(self._key(content.title, content.year))
        try:
            path.write_text(
                json.dumps(content.__dict__, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except OSError:
            pass
