"""PDF table extraction via PyMuPDF (``fitz``).

Used by the audit pipeline to surface result/benchmark tables as
first-class retrievable chunks. Most quantitative-claim verification
hinges on values that live in tables, not prose — splitting them out
makes them ranking-distinguishable from the surrounding text.

PyMuPDF's ``page.find_tables()`` (1.23+) handles the lattice-/stream-
based table detection internally. We serialise each detected table as
caption + headers + rows in a markdown-ish layout so:
  • BM25 retrieval sees the column header tokens and row labels as
    individual tokens (huge wins for "X on Y benchmark" queries).
  • The LLM judge can quote a specific cell at audit time
    (e.g. "Table 3 row 'Med-PaLM 2' col 'MedQA' = 86.5").
  • Cosine retrieval still works because the serialised form is
    natural-language-friendly.

Tables are extracted at ``get_paper_text`` time alongside text, then
chunk_text() can be given them separately and merge them into the
final chunk list. We DON'T put tables in the cached .txt because (a)
the .txt is text-flow only, and (b) re-running extraction on cached
PDFs is fast (~50ms per page).

Lazy-imports pymupdf at call time so the audit-rag extras stay
optional.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

from .. import dbg


def _looks_like_caption(line: str) -> bool:
    """Heuristic for whether a nearby line is a table caption.

    Captions almost always start with "Table N" or "TABLE N:" in
    academic papers. Some venues use "Tab. N" — we accept that too.
    """
    return bool(re.match(r"^\s*(?:table|tab\.)\s+\d+", line, re.IGNORECASE))


def _table_quality_ok(rows: list[list[str]]) -> bool:
    """Reject obviously-spurious table detections.

    PyMuPDF's find_tables() routinely misfires on figures, equation
    arrays, and badly-spaced two-column layouts, returning "tables"
    that are mostly empty cells or have only one meaningful column.
    Filter heuristics:

      • ≥ 2 non-empty rows (otherwise it's a header-only or single-row
        snippet, almost always a figure caption).
      • ≥ 2 columns with at least one non-empty cell each.
      • Fill rate ≥ 50% across the table — real result tables are
        densely populated; figure misfires have most cells blank.
    """
    if len(rows) < 2:
        return False
    non_empty_rows = [r for r in rows if any((c or "").strip() for c in r)]
    if len(non_empty_rows) < 2:
        return False
    max_cols = max((len(r) for r in non_empty_rows), default=0)
    if max_cols < 2:
        return False
    cols_with_content = 0
    for col in range(max_cols):
        if any(col < len(r) and (r[col] or "").strip() for r in non_empty_rows):
            cols_with_content += 1
    if cols_with_content < 2:
        return False
    total_cells = sum(len(r) for r in non_empty_rows)
    if total_cells == 0:
        return False
    filled = sum(1 for r in non_empty_rows for c in r if (c or "").strip())
    fill_rate = filled / total_cells
    if fill_rate < 0.5:
        return False
    return True


def _serialize_table(
    rows: list[list[str]],
    *,
    caption: str = "",
    page: int,
    table_idx: int,
) -> str:
    """Render a table as markdown-ish text for embedding + LLM consumption.

    Format::

        [Table {idx} · p.{page}] {caption}
        | col1 | col2 | col3 |
        | row1c1 | row1c2 | row1c3 |
        | row2c1 | row2c2 | row2c3 |

    Empty rows are dropped. Cell text is collapsed to single-space.
    The leading sentinel line gives the LLM something to quote
    ("Table 3 on p.7 says..."). Markdown pipes are not parsed by
    bge-m3 or BM25 but their presence is a strong signal to the LLM
    that this is tabular data.
    """
    if not rows:
        return ""
    # Drop wholly-empty rows.
    cleaned: list[list[str]] = []
    for r in rows:
        clean = [re.sub(r"\s+", " ", (c or "").strip()) for c in r]
        if any(clean):
            cleaned.append(clean)
    if not cleaned:
        return ""
    lines = [f"[Table {table_idx} · p.{page}] {caption}".rstrip()]
    for row in cleaned:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def extract_tables_from_pdf(
    pdf_path: Path,
    paper_key: str,
    *,
    max_tables_per_page: int = 6,
) -> list:
    """Return a list of ``Chunk`` (chunk_type='table') extracted from ``pdf_path``.

    Caps tables per page to ``max_tables_per_page`` so a pathological
    figure-heavy paper can't blow up memory. Reverses-on-error: any
    PyMuPDF exception during extraction returns an empty list with a
    trace line. Callers should treat tables as best-effort enrichment;
    the prose-chunk path is the load-bearing retrieval.
    """
    # Lazy import — keeps pymupdf optional under [audit-rag] extras.
    try:
        import pymupdf  # type: ignore
    except ImportError:
        dbg.trace(
            "audit.tables",
            "pymupdf not installed — install with `pip install -e \".[audit-rag]\"`",
        )
        return []

    # Defer the Chunk import to call-time to avoid the circular-import risk
    # if anyone re-exports tables from audit_rag.
    from ..audit_rag import Chunk

    try:
        doc = pymupdf.open(str(pdf_path))
    except Exception as e:
        dbg.trace("audit.tables", "open failed", path=str(pdf_path), error=str(e))
        return []

    chunks: list = []
    chunk_idx = 0
    try:
        for page_num, page in enumerate(doc, start=1):
            try:
                tables = page.find_tables()
            except Exception as e:
                dbg.trace(
                    "audit.tables", "find_tables failed",
                    page=page_num, error=str(e),
                )
                continue
            # PyMuPDF's TableFinder is iterable over its `.tables` attr.
            page_tables = list(getattr(tables, "tables", []) or [])
            if not page_tables:
                continue
            # Cache the page's text lines so we can sniff a caption near
            # the table bbox without re-walking the whole page.
            try:
                page_text_lines = [
                    l for l in (page.get_text("text") or "").splitlines() if l.strip()
                ]
            except Exception:
                page_text_lines = []

            for ti, tbl in enumerate(page_tables[:max_tables_per_page], start=1):
                try:
                    rows = tbl.extract() or []
                except Exception as e:
                    dbg.trace(
                        "audit.tables", "extract failed",
                        page=page_num, ti=ti, error=str(e),
                    )
                    continue
                # Quality gate — PyMuPDF over-detects on figures and badly-
                # spaced multi-column layouts; reject anything that doesn't
                # look like a real result table.
                if not _table_quality_ok(rows):
                    dbg.trace(
                        "audit.tables", "rejected low-quality detection",
                        page=page_num, ti=ti, rows=len(rows),
                    )
                    continue
                # Caption sniff — first line above or below the table's
                # vertical bbox that looks like "Table N: ...". A more
                # accurate version would use tbl.header.external (1.24+)
                # but the line-scan fallback is robust across versions.
                caption = ""
                try:
                    bbox = tbl.bbox  # (x0, y0, x1, y1)
                except Exception:
                    bbox = None
                # Walk all page lines, pick the one that's both close to
                # the table and matches the caption pattern.
                if bbox is not None:
                    for line in page_text_lines:
                        if _looks_like_caption(line):
                            caption = line.strip()
                            break

                text = _serialize_table(
                    rows, caption=caption, page=page_num, table_idx=chunk_idx + 1,
                )
                if not text:
                    continue
                chunks.append(
                    Chunk(
                        paper_key=paper_key,
                        text=text,
                        page=page_num,
                        chunk_idx=chunk_idx,
                        chunk_type="table",
                    )
                )
                chunk_idx += 1
    finally:
        try:
            doc.close()
        except Exception:
            pass

    dbg.trace(
        "audit.tables",
        "extracted",
        path=str(pdf_path),
        key=paper_key,
        n_tables=len(chunks),
    )
    return chunks
