"""Chunking, embedding, and retrieval for Tier-2 RAG-based audit.

Pipeline:
  raw PDF text (with [Page N] markers from audit_sources.pdf)
     │
     ▼  chunk_text()
  list[Chunk]    — overlapping ~800-word windows, page-tagged
     │
     ▼  EmbeddingStore.index_paper()
  embeddings cached at ~/Library/Caches/bibsync/embeddings/<key>.json
     │
     ▼  EmbeddingStore.retrieve(query, ...)
  top-K most-similar chunks for a specific claim

Embeddings via the same OpenAI/OpenRouter client we already use elsewhere — no
extra SDK. Cosine similarity in pure Python (no numpy). Cache is per-paper and
keyed by embedding model, so changing the model invalidates the cache.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

from . import config, dbg


@dataclass
class Chunk:
    """One retrievable chunk of a paper's full text."""

    paper_key: str
    text: str
    page: Optional[int] = None
    chunk_idx: int = 0


_PAGE_BLOCK_RE = re.compile(r"\[Page (\d+)\]\n(.*?)(?=\[Page \d+\]|\Z)", re.DOTALL)


def chunk_text(
    text: str, paper_key: str, *, chunk_size: int = 800, overlap: int = 100
) -> list[Chunk]:
    """Split paper text into overlapping word-windows, preserving page numbers.

    ``chunk_size`` is in words; ~800 words ≈ ~1100 tokens, well inside the
    ``text-embedding-3-small`` 8k-token window with room to spare.
    """
    page_blocks: list[tuple[Optional[int], str]] = []
    for m in _PAGE_BLOCK_RE.finditer(text):
        page_blocks.append((int(m.group(1)), m.group(2).strip()))
    if not page_blocks:
        # No page markers (extractor failed to preserve them) — treat as one big page.
        page_blocks = [(None, text)]

    chunks: list[Chunk] = []
    idx = 0
    for page_num, block in page_blocks:
        words = block.split()
        if not words:
            continue
        step = max(1, chunk_size - overlap)
        i = 0
        while i < len(words):
            window = words[i : i + chunk_size]
            chunks.append(
                Chunk(
                    paper_key=paper_key,
                    text=" ".join(window),
                    page=page_num,
                    chunk_idx=idx,
                )
            )
            idx += 1
            if i + chunk_size >= len(words):
                break
            i += step
    return chunks


def _cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity in pure Python — no numpy dependency."""
    if not a or not b:
        return 0.0
    num = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        num += x * y
        na += x * x
        nb += y * y
    if na == 0 or nb == 0:
        return 0.0
    return num / (math.sqrt(na) * math.sqrt(nb))


class EmbeddingStore:
    """Compute + cache OpenAI/OpenRouter embeddings, retrieve top-K chunks.

    Cache layout: ``<cache_dir>/embeddings/<paper_key>.json`` containing the
    chunks and their embedding vectors plus the model name. Switching models
    invalidates the cache automatically.
    """

    def __init__(
        self,
        cache_dir: Path,
        *,
        model: str = "text-embedding-3-small",
        api_key: Optional[str] = None,
    ):
        self.dir = cache_dir / "embeddings"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.model = model
        self.api_key = api_key

    # ── cache I/O ───────────────────────────────────────────────────────────

    def _path(self, paper_key: str) -> Path:
        safe = paper_key.replace("/", "_")
        return self.dir / f"{safe}.json"

    def _load(self, paper_key: str) -> Optional[tuple[list[Chunk], list[list[float]]]]:
        p = self._path(paper_key)
        if not p.exists():
            return None
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if data.get("model") != self.model:
            dbg.trace(
                "audit.rag",
                "cache invalidated (model mismatch)",
                key=paper_key,
                cached_model=data.get("model"),
                want_model=self.model,
            )
            return None
        try:
            chunks = [Chunk(**c) for c in data["chunks"]]
            vectors = data["vectors"]
        except (KeyError, TypeError):
            return None
        return chunks, vectors

    def _save(
        self, paper_key: str, chunks: list[Chunk], vectors: list[list[float]]
    ) -> None:
        try:
            self._path(paper_key).write_text(
                json.dumps(
                    {
                        "model": self.model,
                        "chunks": [asdict(c) for c in chunks],
                        "vectors": vectors,
                    }
                ),
                encoding="utf-8",
            )
        except OSError:
            pass

    # ── embedding ───────────────────────────────────────────────────────────

    async def _embed(self, texts: list[str]) -> Optional[list[list[float]]]:
        """Call the embeddings endpoint via the configured LLM provider.

        Works with both OpenAI and OpenRouter (both expose the same
        embeddings API shape via the openai SDK).
        """
        if not texts:
            return []
        try:
            from openai import AsyncOpenAI
        except ImportError:
            dbg.trace("audit.rag", "openai SDK not installed")
            return None

        cfg = config.resolve_llm_config(self.api_key)
        if not cfg:
            dbg.trace("audit.rag", "no LLM API key resolved")
            return None

        client_kwargs: dict = {"api_key": cfg.api_key}
        if cfg.base_url:
            client_kwargs["base_url"] = cfg.base_url

        try:
            client = AsyncOpenAI(**client_kwargs)
            resp = await client.embeddings.create(model=self.model, input=texts)
            return [d.embedding for d in resp.data]
        except Exception as e:
            dbg.trace("audit.rag", "embeddings request failed", error=str(e))
            return None

    # ── public API ──────────────────────────────────────────────────────────

    async def index_paper(
        self, paper_key: str, chunks: list[Chunk]
    ) -> Optional[list[list[float]]]:
        """Return embedding vectors for ``chunks``, computing+caching if needed."""
        cached = self._load(paper_key)
        if cached:
            cached_chunks, cached_vecs = cached
            if len(cached_chunks) == len(chunks):
                dbg.trace(
                    "audit.rag", "embedding cache hit", key=paper_key, n=len(chunks)
                )
                return cached_vecs

        dbg.trace(
            "audit.rag", "embedding chunks", key=paper_key, n=len(chunks), model=self.model
        )
        vectors = await self._embed([c.text for c in chunks])
        if vectors is None:
            return None
        self._save(paper_key, chunks, vectors)
        return vectors

    async def retrieve(
        self,
        query: str,
        paper_key: str,
        chunks: list[Chunk],
        *,
        top_k: int = 5,
    ) -> list[tuple[Chunk, float]]:
        """Embed ``query`` and return top-K chunks by cosine similarity."""
        vectors = await self.index_paper(paper_key, chunks)
        if not vectors:
            return []
        q_emb = await self._embed([query])
        if not q_emb:
            return []
        q = q_emb[0]
        scored: list[tuple[Chunk, float]] = []
        for i, c in enumerate(chunks):
            if i < len(vectors):
                scored.append((c, _cosine(q, vectors[i])))
        scored.sort(key=lambda t: t[1], reverse=True)
        result = scored[:top_k]
        dbg.trace(
            "audit.rag",
            "retrieved",
            key=paper_key,
            top_k=top_k,
            scores=[round(s, 3) for _, s in result],
        )
        return result
