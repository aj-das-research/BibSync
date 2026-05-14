"""Title-match guard for source adapters.

Each source (arXiv, Semantic Scholar, Crossref) returns the *first* result for a
title query — none of them strictly require their result to actually correspond
to the queried paper. In practice, when arXiv is down and the title is generic,
SS / Crossref readily return wrong-paper hits like:

  query  "Language Models are Few-Shot Learners"
  → Crossref returns "Adapting Language-Audio Models as Few-Shot Audio Learners"

The downstream audit pipeline then judges the user's claim against the wrong
paper's abstract and rubber-stamps it. We need a similarity gate that rejects
results whose title is too far from the query before they enter the merge step
in ``fetch_paper_content``.

We use ``rapidfuzz.fuzz.token_sort_ratio`` (not ``token_set_ratio``) because
``token_set_ratio`` deduplicates overlapping words and rates "Spectrum-BERT…
Spectral Chemometrics" as 82-similar to the BERT query — too lenient.
``token_sort_ratio`` is order-aware: it tolerates punctuation and subtitle
expansions but separates real matches (~79+) from wrong-paper hits (~74 and
below). Threshold ``DEFAULT_TITLE_THRESHOLD = 75`` is the lowest value that
passes our fixture cases while rejecting the known wrong-paper hits from
real OpenRouter/Crossref traces.
"""

from __future__ import annotations

import re

from .. import dbg

DEFAULT_TITLE_THRESHOLD = 75  # 0–100, rapidfuzz scale


def _normalize_title(s: str) -> str:
    """Lowercase, strip braces/punct, collapse whitespace — for fuzzy matching."""
    s = re.sub(r"[{}\\]", " ", s or "")
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip().lower()


def titles_match(
    query: str,
    candidate: str,
    *,
    threshold: int = DEFAULT_TITLE_THRESHOLD,
    source: str = "?",
) -> bool:
    """Return True iff ``candidate`` is similar enough to ``query`` to trust.

    Logs a rejection trace with the score so a failing match is debuggable.
    Falls open (returns True) if rapidfuzz isn't installed — we don't want this
    guard to silently drop all results in a broken-deps environment.
    """
    if not query or not candidate:
        return False
    try:
        from rapidfuzz import fuzz
    except ImportError:  # rapidfuzz is in core deps but be defensive
        dbg.trace(
            "audit.match",
            "rapidfuzz not installed — title guard skipped",
            source=source,
        )
        return True

    q = _normalize_title(query)
    c = _normalize_title(candidate)
    score = int(fuzz.token_sort_ratio(q, c))
    if score < threshold:
        dbg.trace(
            "audit.match",
            "REJECTED low title similarity",
            source=source,
            score=score,
            threshold=threshold,
            query=query[:80],
            candidate=candidate[:80],
        )
        return False
    dbg.trace(
        "audit.match",
        "accepted",
        source=source,
        score=score,
    )
    return True
