"""Choose the canonical version among multiple versions of the same paper.

Preference order (highest score wins):
  1. Major peer-reviewed venues (top-tier conferences, journals).
  2. Workshops, symposia, technical reports.
  3. Preprint servers (arXiv, bioRxiv, SSRN, HAL).
  4. Unknown venues — fallback by citation count.

Tie-breaks: more recent year > higher cited_by count > longer venue string (proxy for
"this hit had richer metadata, so probably came from a real publisher page").
"""

from __future__ import annotations

import re
from typing import Optional

from .models import PaperHit

# Lowercase substrings -> score. Order matters: we use the highest-matching score per hit.
_VENUE_SCORE_TABLE: list[tuple[str, int]] = [
    # Top-tier ML / AI conferences
    ("neurips", 100), ("advances in neural", 100), ("nips", 95),
    ("icml", 100), ("iclr", 100),
    ("aaai", 100), ("ijcai", 100),
    ("cvpr", 100), ("iccv", 100), ("eccv", 100),
    ("acl", 100), ("emnlp", 100), ("naacl", 100),
    ("kdd", 100), ("www", 95),
    ("sigir", 100), ("sigmod", 100), ("vldb", 100), ("icde", 100),
    ("siggraph", 100),
    ("uai", 95), ("aistats", 95),
    # Generic high-quality signals
    ("proceedings of", 80),
    ("conference on", 75),
    ("international conference", 75),
    ("transactions on", 90),
    ("journal of", 85),
    ("nature", 95), ("science", 95),
    # Lower-tier / workshop signals
    ("workshop", 40),
    ("symposium", 50),
    ("technical report", 25),
    ("thesis", 20),
    # Preprint servers
    ("arxiv", 10), ("arxiv preprint", 10),
    ("biorxiv", 10), ("medrxiv", 10),
    ("ssrn", 10),
    ("hal", 10),
    ("openreview", 30),  # could be preprint or reviewed — middle ground
]


def venue_score(venue: Optional[str]) -> int:
    if not venue:
        return 0
    v = venue.lower()
    best = 0
    for needle, score in _VENUE_SCORE_TABLE:
        if needle in v:
            best = max(best, score)
    return best


def _sort_key(hit: PaperHit) -> tuple[int, int, int, int]:
    return (
        venue_score(hit.venue),
        hit.year or 0,
        hit.cited_by,
        len(hit.venue or ""),
    )


def pick_canonical(versions: list[PaperHit]) -> PaperHit:
    """Pick the canonical version from a list. Caller must ensure list is non-empty."""
    if not versions:
        raise ValueError("pick_canonical called with empty list")
    if len(versions) == 1:
        return versions[0]
    return sorted(versions, key=_sort_key, reverse=True)[0]


def is_ambiguous(versions: list[PaperHit]) -> bool:
    """True if the top two candidates have the same heuristic score — LLM tiebreak helpful."""
    if len(versions) < 2:
        return False
    ranked = sorted(versions, key=_sort_key, reverse=True)
    return _sort_key(ranked[0]) == _sort_key(ranked[1])


# Optional OpenAI tiebreaker ---------------------------------------------------


def pick_canonical_with_llm(
    versions: list[PaperHit],
    *,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
) -> PaperHit:
    """Use the configured LLM (OpenAI or OpenRouter) to pick among ambiguous versions.
    Falls back to heuristic on any error."""
    if len(versions) <= 1:
        return pick_canonical(versions)
    try:
        from .llm import _get_client_and_model  # lazy
        client, model_id = _get_client_and_model(api_key, model)
    except Exception:
        return pick_canonical(versions)

    lines = [
        f"{i}. title={v.title!r}, venue={v.venue!r}, year={v.year}, cited_by={v.cited_by}"
        for i, v in enumerate(versions)
    ]
    prompt = (
        "You are an academic-citation expert. Given these versions of the same paper, "
        "pick the one a researcher should cite (peer-reviewed venue beats preprint; newer "
        "official version beats older; canonical conference/journal beats workshop). "
        "Respond with ONLY the integer index.\n\n" + "\n".join(lines)
    )
    try:
        resp = client.chat.completions.create(
            model=model_id,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=4,
        )
        text = (resp.choices[0].message.content or "").strip()
        m = re.search(r"\d+", text)
        if m:
            idx = int(m.group(0))
            if 0 <= idx < len(versions):
                return versions[idx]
    except Exception:
        pass
    return pick_canonical(versions)
