"""PDF download + text extraction with on-disk cache.

For Tier-2 audit, we fetch the paper's open-access PDF (URL provided by
arXiv or Semantic Scholar's ``openAccessPdf``), extract text page-by-page,
and cache both the binary and the extracted text. The extracted text
preserves ``[Page N]`` markers so the RAG retriever can cite page numbers
in its evidence.

Lightweight by design: ``pypdf`` for extraction (no heavy ML deps).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from .. import dbg


class PdfCache:
    """Per-paper PDF + extracted-text cache under ``cache_dir/pdfs/``."""

    def __init__(self, cache_dir: Path):
        self.dir = cache_dir / "pdfs"
        self.dir.mkdir(parents=True, exist_ok=True)

    def pdf_path(self, paper_key: str) -> Path:
        safe = paper_key.replace("/", "_")
        return self.dir / f"{safe}.pdf"

    def text_path(self, paper_key: str) -> Path:
        safe = paper_key.replace("/", "_")
        return self.dir / f"{safe}.txt"

    def get_text(self, paper_key: str) -> Optional[str]:
        p = self.text_path(paper_key)
        if p.exists():
            try:
                return p.read_text(encoding="utf-8", errors="replace")
            except OSError:
                return None
        return None

    def put_text(self, paper_key: str, text: str) -> None:
        try:
            self.text_path(paper_key).write_text(text, encoding="utf-8")
        except OSError:
            pass


async def download_pdf(url: str, dest: Path, *, timeout: float = 60.0) -> bool:
    """Download a PDF to ``dest``. Returns ``True`` if the download succeeded
    AND the bytes start with the ``%PDF-`` magic (so a paywalled HTML
    redirect doesn't get cached as a fake PDF)."""
    try:
        import httpx
    except ImportError:
        dbg.trace("audit.pdf", "httpx not installed")
        return False
    try:
        async with httpx.AsyncClient(
            timeout=timeout, follow_redirects=True
        ) as client:
            resp = await client.get(
                url, headers={"User-Agent": "bibsync/0.1"}
            )
            resp.raise_for_status()
            if not resp.content[:5].startswith(b"%PDF-"):
                dbg.trace(
                    "audit.pdf",
                    "downloaded content is not a PDF (likely a paywall/HTML)",
                    url=url,
                    first_bytes=resp.content[:8],
                )
                return False
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(resp.content)
            dbg.trace("audit.pdf", "downloaded", url=url, bytes=len(resp.content))
            return True
    except Exception as e:
        dbg.trace("audit.pdf", "download failed", url=url, error=str(e))
        return False


def extract_pdf_text(pdf_path: Path) -> Optional[str]:
    """Extract text page-by-page using pypdf, preserving ``[Page N]`` markers.

    Returns the full text with one paragraph per page, or ``None`` if pypdf
    isn't installed or extraction failed.
    """
    try:
        import pypdf  # type: ignore
    except ImportError:
        dbg.trace(
            "audit.pdf",
            "pypdf not installed — install with `pip install pypdf` to enable Tier 2",
        )
        return None
    try:
        reader = pypdf.PdfReader(str(pdf_path))
    except Exception as e:
        dbg.trace("audit.pdf", "PdfReader failed", error=str(e))
        return None
    pages: list[str] = []
    for i, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""
        text = text.strip()
        if text:
            pages.append(f"[Page {i}]\n{text}")
    if not pages:
        return None
    return "\n\n".join(pages)


async def get_paper_text(
    paper_key: str, pdf_url: str, cache: PdfCache
) -> Optional[str]:
    """Idempotent: ensure the paper's text is cached and return it.

    Flow:
      1. If text already cached → return.
      2. Else if PDF cached → extract text, cache text, return.
      3. Else download PDF, extract text, cache both, return.
    """
    cached = cache.get_text(paper_key)
    if cached:
        dbg.trace("audit.pdf", "text cache hit", key=paper_key)
        return cached

    pdf_path = cache.pdf_path(paper_key)
    if not pdf_path.exists():
        ok = await download_pdf(pdf_url, pdf_path)
        if not ok:
            return None

    text = extract_pdf_text(pdf_path)
    if text:
        cache.put_text(paper_key, text)
        dbg.trace("audit.pdf", "extracted", key=paper_key, chars=len(text))
    return text
