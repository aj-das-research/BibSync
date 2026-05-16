"""BibTeX file utilities: parse, dedupe, append, atomic write."""

from __future__ import annotations

import os
import re
import tempfile
import unicodedata
from pathlib import Path
from typing import Optional

import bibtexparser
from bibtexparser.bibdatabase import BibDatabase
from bibtexparser.bparser import BibTexParser
from bibtexparser.bwriter import BibTexWriter
from rapidfuzz import fuzz

# Match threshold for title-based deduplication (0-100; higher = stricter).
TITLE_MATCH_THRESHOLD = 92


_KEY_STOPWORDS = {
    "the", "and", "for", "with", "from", "this", "that", "into",
    "using", "based", "via", "towards", "novel", "approach",
}


def derive_cite_key(entry: dict) -> str:
    """Build a citation key from a BibTeX entry: firstauthorsurname + year + firsttitleword."""
    author = (entry.get("author") or "").split(" and ")[0]
    if "," in author:
        surname = author.split(",")[0].strip()
    else:
        parts = author.split()
        surname = parts[-1] if parts else "anon"
    surname = re.sub(r"[^A-Za-z]", "", surname).lower() or "anon"
    year = (entry.get("year") or "").strip() or "nd"
    title = re.sub(r"[^A-Za-z\s]", " ", entry.get("title") or "")
    words = [w for w in title.split() if len(w) > 2 and w.lower() not in _KEY_STOPWORDS]
    first_word = (words[0] if words else "untitled").lower()
    return f"{surname}{year}{first_word}"


def build_entry_from_metadata(
    *,
    title: str,
    authors: list,
    year: Optional[int] = None,
    venue: str = "",
    doi: str = "",
    arxiv_id: str = "",
) -> tuple[str, dict, str]:
    """Build a proper BibTeX entry from resolved paper metadata.

    Returns ``(cite_key, entry_dict, bibtex_text)``:
      • ``cite_key`` — firstauthorsurname+year+firsttitleword
        (e.g. ``das2024confidence``), NOT a mangled DOI.
      • ``entry_dict`` — the parsed-entry shape used elsewhere.
      • ``bibtex_text`` — the formatted ``@type{...}`` block, ready to
        paste into a .bib file.

    Entry type heuristic: arXiv-only → ``@article``; a venue containing
    "conference"/"proceedings"/"lecture notes"/"workshop" → ``@inproceedings``
    (LNCS / MICCAI / NeurIPS proceedings); otherwise ``@article``.
    """
    # authors → "Surname, Given and Surname, Given" BibTeX form.
    author_field = " and ".join(a.strip() for a in (authors or []) if a.strip())

    venue_l = (venue or "").lower()
    is_proceedings = any(
        kw in venue_l
        for kw in ("conference", "proceedings", "lecture notes", "workshop", "symposium")
    )
    entry_type = "inproceedings" if is_proceedings else "article"

    entry: dict = {
        "ENTRYTYPE": entry_type,
        "title": title or "",
        "author": author_field,
    }
    if year:
        entry["year"] = str(year)
    if venue:
        entry["booktitle" if entry_type == "inproceedings" else "journal"] = venue
    if doi:
        entry["doi"] = doi
    if arxiv_id:
        entry["eprint"] = arxiv_id
        entry["archivePrefix"] = "arXiv"

    cite_key = derive_cite_key(entry)
    entry["ID"] = cite_key

    # Format the @type{...} block via the standard writer (one-entry db).
    db = BibDatabase()
    db.entries = [entry]
    bibtex_text = bibtexparser.dumps(db, writer=_writer()).strip()
    return cite_key, entry, bibtex_text


def _normalize_title(title: str) -> str:
    """Strip punctuation, lowercase, collapse whitespace — for fuzzy comparison only."""
    title = unicodedata.normalize("NFKD", title)
    title = re.sub(r"[{}\\]", "", title)
    title = re.sub(r"[^\w\s]", " ", title)
    return re.sub(r"\s+", " ", title).strip().lower()


def _make_parser() -> BibTexParser:
    parser = BibTexParser(common_strings=True)
    parser.ignore_nonstandard_types = False
    parser.homogenize_fields = False
    return parser


def load(path: Path) -> BibDatabase:
    """Load a .bib file. Returns an empty database if the file does not exist."""
    if not path.exists():
        db = BibDatabase()
        db.entries = []
        return db
    with path.open(encoding="utf-8") as f:
        return bibtexparser.load(f, parser=_make_parser())


def parse_string(text: str) -> BibDatabase:
    return bibtexparser.loads(text, parser=_make_parser())


def _writer() -> BibTexWriter:
    w = BibTexWriter()
    w.indent = "  "
    w.align_values = True
    w.order_entries_by = ("ID",)
    return w


def dump(db: BibDatabase, path: Path) -> None:
    """Atomic write: write to a sibling tempfile then os.replace, so a crash mid-write
    never leaves a partial .bib file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    text = bibtexparser.dumps(db, writer=_writer())
    fd, tmp = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=str(path.parent or ".")
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
        os.replace(tmp, path)
    except Exception:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def find_duplicate(db: BibDatabase, new_entry: dict) -> Optional[dict]:
    """Return an existing entry that matches the new one, or None.

    Matching priority:
      1. DOI exact match (case-insensitive).
      2. Citation key exact match.
      3. Fuzzy title match >= TITLE_MATCH_THRESHOLD AND same year (if both have years).
    """
    new_doi = (new_entry.get("doi") or "").strip().lower()
    new_id = (new_entry.get("ID") or "").strip()
    new_title_norm = _normalize_title(new_entry.get("title", ""))
    new_year = (new_entry.get("year") or "").strip()

    for entry in db.entries:
        if new_doi and (entry.get("doi") or "").strip().lower() == new_doi:
            return entry
        if new_id and entry.get("ID", "") == new_id:
            return entry
        if not new_title_norm:
            continue
        existing_title_norm = _normalize_title(entry.get("title", ""))
        if not existing_title_norm:
            continue
        ratio = fuzz.ratio(new_title_norm, existing_title_norm)
        if ratio >= TITLE_MATCH_THRESHOLD:
            existing_year = (entry.get("year") or "").strip()
            # If both have years, they must match. If either is missing, accept the title match.
            if not new_year or not existing_year or new_year == existing_year:
                return entry
    return None


def ensure_unique_key(db: BibDatabase, desired_key: str) -> str:
    """Return a citation key not already used in db. Appends 'a', 'b', ... if needed."""
    existing = {e.get("ID", "") for e in db.entries}
    if desired_key not in existing:
        return desired_key
    for suffix in "abcdefghijklmnopqrstuvwxyz":
        candidate = f"{desired_key}{suffix}"
        if candidate not in existing:
            return candidate
    # Extremely unlikely fallback: numeric suffixes.
    i = 1
    while f"{desired_key}{i}" in existing:
        i += 1
    return f"{desired_key}{i}"


def append_entry(db: BibDatabase, entry: dict) -> tuple[dict, bool]:
    """Append entry to db unless a duplicate exists. Returns (entry_in_db, was_added)."""
    dup = find_duplicate(db, entry)
    if dup is not None:
        return dup, False
    entry = dict(entry)
    entry["ID"] = ensure_unique_key(db, entry.get("ID") or "entry")
    db.entries.append(entry)
    return entry, True
