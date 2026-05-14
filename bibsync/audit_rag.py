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

Two embedding backends, resolved local-first:

  1. LOCAL — via ``fastembed`` (BAAI/bge-small-en-v1.5 by default). Free,
     fully offline, ~80 MB model download cached after first use. Quality
     is on par with OpenAI's text-embedding-3-small for retrieval tasks.
     Requires ``pip install -e ".[audit-rag]"`` (which includes fastembed).

  2. API — via the OpenAI-compatible ``embeddings`` endpoint. Used when
     fastembed isn't installed, or when the user explicitly passes
     ``--embedding-backend api``. Default model is provider-aware:

       - OpenRouter (``sk-or-...`` keys) → ``baai/bge-m3``
         (open-source BGE, 8K ctx, ~$0.01 / 1M tokens, routable via
         OpenRouter's ``/v1/embeddings`` endpoint).
       - OpenAI native (``sk-...`` keys) → ``text-embedding-3-small``.

     Override either with ``--embedding-model <id>``.

Cache invalidates automatically on model change because the effective
model name is stored alongside the vectors. Cosine similarity in pure
Python — no numpy dep.
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


# Default models per backend. Both produce dense embeddings; cache is keyed by
# the actual model name, so switching backends invalidates the cache cleanly.
DEFAULT_LOCAL_MODEL = "BAAI/bge-small-en-v1.5"

# API backend defaults are provider-aware. Native OpenAI gets its own embedding
# model; OpenRouter gets baai/bge-m3 — an open-source BGE family model (same
# vendor as our local default), 8K context, ~$0.01/M tokens through OpenRouter,
# and routable via OpenRouter's /v1/embeddings endpoint (unlike OpenAI's
# text-embedding-* IDs which OpenRouter currently does not relay).
DEFAULT_API_MODEL_OPENAI = "text-embedding-3-small"
DEFAULT_API_MODEL_OPENROUTER = "baai/bge-m3"


def _default_api_model_for(base_url: Optional[str]) -> str:
    """Pick a sensible embedding model id for an OpenAI-compatible base URL.

    We use base_url as the provider signal because that's what's already
    resolved by ``config.resolve_llm_config`` from the user's key prefix
    (``sk-or-...`` → OpenRouter, ``sk-...`` → OpenAI native).
    """
    if base_url and "openrouter" in base_url.lower():
        return DEFAULT_API_MODEL_OPENROUTER
    return DEFAULT_API_MODEL_OPENAI


# ── backends ────────────────────────────────────────────────────────────────


class _LocalEmbeddingBackend:
    """fastembed-backed embedder. ONNX-based, no PyTorch, no API key needed.

    First use downloads the model (~80 MB for bge-small-en-v1.5) into the
    fastembed cache; subsequent uses load from disk in <1s.
    """

    kind = "local"

    def __init__(self, model_name: str):
        from fastembed import TextEmbedding  # type: ignore

        self.model_name = model_name
        self._embedder = TextEmbedding(model_name=model_name)

    async def embed(self, texts: list[str]) -> list[list[float]]:
        import asyncio

        def _run() -> list[list[float]]:
            # fastembed.embed returns a generator of numpy arrays
            return [list(map(float, v)) for v in self._embedder.embed(texts)]

        return await asyncio.to_thread(_run)


class _ApiEmbeddingBackend:
    """OpenAI-compatible embeddings endpoint (OpenAI native, or any provider
    that exposes the same shape, e.g. some OpenRouter routes)."""

    kind = "api"

    def __init__(self, model_name: str, client):
        self.model_name = model_name
        self._client = client

    async def embed(self, texts: list[str]) -> list[list[float]]:
        resp = await self._client.embeddings.create(model=self.model_name, input=texts)
        return [d.embedding for d in resp.data]


def _try_make_local_backend(model_hint: str) -> Optional[_LocalEmbeddingBackend]:
    """Return a local backend if fastembed is installed and the model loads."""
    try:
        import fastembed  # noqa: F401
    except ImportError:
        dbg.trace(
            "audit.rag",
            "fastembed not installed — install with `pip install -e \".[audit-rag]\"` "
            "to enable local Tier-2 embeddings",
        )
        return None
    # If the user asked for an OpenAI-only model name (text-embedding-*), route
    # to a sensible local default. Anything else, pass through as a local model id.
    if not model_hint or model_hint == "auto" or model_hint.startswith("text-embedding-"):
        local_model = DEFAULT_LOCAL_MODEL
    else:
        local_model = model_hint
    try:
        backend = _LocalEmbeddingBackend(local_model)
        dbg.trace("audit.rag", "local backend ready", model=local_model)
        return backend
    except Exception as e:
        dbg.trace("audit.rag", "local backend init failed", model=local_model, error=str(e))
        return None


def _try_make_api_backend(
    model_hint: str, api_key: Optional[str]
) -> Optional[_ApiEmbeddingBackend]:
    """Return an API backend if the openai SDK + a usable LLM provider are configured."""
    try:
        from openai import AsyncOpenAI
    except ImportError:
        dbg.trace("audit.rag", "openai SDK not installed; cannot use API embeddings")
        return None
    cfg = config.resolve_llm_config(api_key)
    if not cfg:
        dbg.trace("audit.rag", "no LLM API key resolved for embeddings")
        return None
    if not model_hint or model_hint == "auto":
        model = _default_api_model_for(cfg.base_url)
    else:
        model = model_hint
    client_kwargs: dict = {"api_key": cfg.api_key}
    if cfg.base_url:
        client_kwargs["base_url"] = cfg.base_url
    try:
        client = AsyncOpenAI(**client_kwargs)
        dbg.trace("audit.rag", "api backend ready", model=model, base_url=cfg.base_url)
        return _ApiEmbeddingBackend(model, client)
    except Exception as e:
        dbg.trace("audit.rag", "api backend init failed", error=str(e))
        return None


# ── store ──────────────────────────────────────────────────────────────────


class EmbeddingStore:
    """Compute + cache embeddings, retrieve top-K chunks.

    Backend resolution (local-first by default):

      ``backend='auto'`` (default) — try local (fastembed) first, then API.
      ``backend='local'``         — force fastembed; fail if not installed.
      ``backend='api'``           — force OpenAI-compatible endpoint;
                                    fail if no key / SDK / provider support.

    The chosen backend is resolved lazily on first use and cached. The cache
    layout is ``<cache_dir>/embeddings/<paper_key>.json`` containing the
    chunks, vectors, and ``effective_model`` (so different backends/models
    don't collide).
    """

    def __init__(
        self,
        cache_dir: Path,
        *,
        model: str = "auto",
        api_key: Optional[str] = None,
        backend: str = "auto",
    ):
        self.dir = cache_dir / "embeddings"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.requested_model = model
        self.api_key = api_key
        self.requested_backend = backend
        self._backend: Optional[object] = None
        self._resolution_attempted = False

    # ── backend resolution ─────────────────────────────────────────────────

    def _resolve_backend(self):
        if self._resolution_attempted:
            return self._backend
        self._resolution_attempted = True

        if self.requested_backend == "local":
            self._backend = _try_make_local_backend(self.requested_model)
        elif self.requested_backend == "api":
            self._backend = _try_make_api_backend(self.requested_model, self.api_key)
        else:  # 'auto' — local-first
            self._backend = _try_make_local_backend(self.requested_model)
            if self._backend is None:
                self._backend = _try_make_api_backend(
                    self.requested_model, self.api_key
                )

        if self._backend is None:
            dbg.trace(
                "audit.rag",
                "NO embedding backend available — Tier 2 will degrade to Tier 1 "
                "for this run. Install fastembed (`pip install -e \".[audit-rag]\"`) "
                "or configure an embeddings-capable LLM provider.",
            )
        return self._backend

    @property
    def effective_model(self) -> str:
        b = self._resolve_backend()
        return b.model_name if b else self.requested_model

    @property
    def effective_backend(self) -> str:
        b = self._resolve_backend()
        return b.kind if b else "none"

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
        # Match BOTH model name and backend kind — same model id under different
        # backends would produce different vectors.
        if data.get("model") != self.effective_model or data.get("backend") != self.effective_backend:
            dbg.trace(
                "audit.rag",
                "cache invalidated (model/backend mismatch)",
                key=paper_key,
                cached_model=data.get("model"),
                cached_backend=data.get("backend"),
                want_model=self.effective_model,
                want_backend=self.effective_backend,
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
                        "model": self.effective_model,
                        "backend": self.effective_backend,
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
        if not texts:
            return []
        backend = self._resolve_backend()
        if backend is None:
            return None
        try:
            return await backend.embed(texts)
        except Exception as e:
            dbg.trace(
                "audit.rag",
                "embed call failed",
                backend=getattr(backend, "kind", "?"),
                error=str(e),
            )
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
            "audit.rag",
            "embedding chunks",
            key=paper_key,
            n=len(chunks),
            backend=self.effective_backend,
            model=self.effective_model,
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
            backend=self.effective_backend,
            scores=[round(s, 3) for _, s in result],
        )
        return result
