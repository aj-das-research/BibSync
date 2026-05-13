"""OpenAI-powered extraction helpers.

Two prompts, same shape — free text in, structured JSON out:

* :func:`infer_paper_from_cite_key` — given a LaTeX citation key + the prose around it,
  guess the paper's title / first author / year. Used by ``bibsync extract`` to
  resolve `\\cite{moor2023gmai}` into a real Scholar query.

* :func:`parse_bibitem` — given a free-text ``\\bibitem{...}`` block, extract the
  structured metadata (title, author list, year, arXiv id). Used by
  ``bibsync repair`` to convert legacy bibliographies into BibTeX.

Both calls use JSON mode and ``temperature=0``: deterministic structured extraction,
not generation.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Optional

from . import config
from .models import PaperHit


@dataclass
class InferredPaper:
    title: str
    first_author: Optional[str] = None
    year: Optional[int] = None
    confidence: float = 0.0  # 0.0 - 1.0
    reasoning: str = ""

    def search_query(self) -> str:
        """A Scholar-friendly query string built from inferred fields."""
        parts = [self.title]
        if self.first_author:
            parts.append(self.first_author)
        return " ".join(parts).strip()


@dataclass
class MatchVerification:
    """Verdict from the LLM-as-judge that decides if a Scholar hit is the same paper
    as a .bib entry. Used by the ``fix`` pipeline as the final gate before rewriting."""

    same_paper: bool
    confidence: float  # 0.0 - 1.0
    reasoning: str


@dataclass
class CitationSuggestion:
    """One paper that should be cited in a paragraph of prose."""

    query: str  # Google Scholar search query (paper title or "first-author year topic")
    anchor: str  # verbatim substring of the paragraph to insert \cite{} after
    reason: str  # one sentence: why this needs citation
    expected_first_author: Optional[str] = None
    expected_year: Optional[int] = None


@dataclass
class ParsedBibitem:
    key: str
    title: Optional[str] = None
    authors: list[str] = None  # type: ignore[assignment]
    year: Optional[int] = None
    arxiv_id: Optional[str] = None
    doi: Optional[str] = None
    raw: str = ""

    def __post_init__(self) -> None:
        if self.authors is None:
            self.authors = []

    def search_query(self) -> str:
        parts = []
        if self.title:
            parts.append(self.title)
        if self.authors:
            parts.append(self.authors[0])
        return " ".join(parts).strip() or self.raw[:80]


def _safe_extract_content(resp) -> Optional[str]:
    """Return the first choice's message content, or None if the response is malformed.

    OpenRouter occasionally returns responses where ``choices`` is missing or empty when
    the underlying provider errors. Indexing into a None or empty list raises a confusing
    'NoneType' or 'IndexError' — this helper folds those into a clean None.
    """
    try:
        choices = getattr(resp, "choices", None) or []
        if not choices:
            return None
        msg = getattr(choices[0], "message", None)
        if msg is None:
            return None
        return getattr(msg, "content", None)
    except Exception:
        return None


def _get_client_and_model(
    api_key: Optional[str] = None, model_override: Optional[str] = None
):
    """Return ``(OpenAI client, model_id)``. Works for OpenAI or OpenRouter transparently.

    OpenRouter is OpenAI-API-compatible — we just point the same client at a different
    base_url. The provider is auto-detected from the key prefix.
    """
    try:
        from openai import OpenAI  # type: ignore
    except ImportError as e:
        raise RuntimeError(
            "openai package not installed. Run: pip install -e '.[openai]'"
        ) from e
    cfg = config.resolve_llm_config(api_key)
    if not cfg:
        raise RuntimeError(
            "No LLM API key found. Set one with `bibsync config set openrouter_key sk-or-...`"
            " or `bibsync config set openai_key sk-...`."
        )
    client_kwargs: dict = {"api_key": cfg.api_key}
    if cfg.base_url:
        client_kwargs["base_url"] = cfg.base_url
    if cfg.provider == "openrouter":
        # Optional but recommended by OpenRouter for usage tracking.
        client_kwargs["default_headers"] = {
            "HTTP-Referer": "https://github.com/bibsync/bibsync",
            "X-Title": "BibSync",
        }
    client = OpenAI(**client_kwargs)
    model = model_override or cfg.model
    return client, model


_INFER_SYSTEM = """\
You decode LaTeX citation keys into paper metadata.

A citation key like `moor2023gmai` usually decomposes to:
  - first author surname  (moor)
  - publication year      (2023)
  - title acronym/word    (gmai = "Generalist Medical AI")

You receive a citation key and the surrounding LaTeX prose. Use BOTH signals.
The prose often states the paper's topic explicitly ("Generalist medical AI was
articulated as a vision in 2023" → confirms title and year).

Respond with a single JSON object:
  {
    "title": "best guess at full paper title",
    "first_author": "Last name only, or null if unknown",
    "year": 2023,
    "confidence": 0.0 to 1.0,
    "reasoning": "one sentence on how you decomposed the key"
  }

If you genuinely cannot infer anything, return null for the unknowable fields.
Do not invent specific titles you are not sure about — lower the confidence instead.
"""


def infer_paper_from_cite_key(
    cite_key: str,
    context: str,
    *,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
) -> InferredPaper:
    """Infer paper metadata from a `\\cite{key}` plus surrounding LaTeX text."""
    client, model_id = _get_client_and_model(api_key, model)
    user_msg = (
        f"Citation key: {cite_key}\n\n"
        f"Surrounding LaTeX text:\n---\n{context}\n---\n\n"
        "Return JSON only."
    )
    resp = client.chat.completions.create(
        model=model_id,
        messages=[
            {"role": "system", "content": _INFER_SYSTEM},
            {"role": "user", "content": user_msg},
        ],
        temperature=0,
        response_format={"type": "json_object"},
    )
    content = _safe_extract_content(resp)
    try:
        data = json.loads(content or "{}")
    except json.JSONDecodeError:
        data = {}
    if not isinstance(data, dict):
        data = {}
    return InferredPaper(
        title=(data.get("title") or "").strip(),
        first_author=(data.get("first_author") or None),
        year=int(data["year"]) if data.get("year") else None,
        confidence=float(data.get("confidence") or 0.0),
        reasoning=str(data.get("reasoning") or ""),
    )


_SUGGEST_SYSTEM = """\
You are an academic citation assistant. The user has a paragraph of LaTeX prose with
NO citations yet. Your job is to identify factual claims, named methods, named systems,
or attributions that should be cited, and propose what to cite.

Respond with a single JSON object of the form:
  {"suggestions": [
      {
        "query": "Google Scholar search query (paper title, or '<first author> <year> <topic>')",
        "anchor": "verbatim substring (5-20 words) from the input paragraph that the citation should be inserted RIGHT AFTER; must be an exact substring of the input",
        "reason": "one sentence explaining why this needs a citation",
        "expected_first_author": "optional surname guess (or null)",
        "expected_year": 2023
      },
      ...
  ]}

Rules:
  * Only flag genuine attributions (named methods like "Med-PaLM", specific systems
    like "RETFound", quantitative claims, foundational works). Do NOT cite generic
    methodology phrases like "we propose" or "we evaluate".
  * If the paragraph names multiple distinct systems / methods, propose ONE citation per
    distinct system. The anchor can be the system name itself if it's the natural place
    for the citation in LaTeX style (e.g., anchor="Med-PaLM 2").
  * If the paragraph has NO claims needing citation, return JSON {"suggestions": []}.
"""


def suggest_citations(
    paragraph: str,
    *,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
) -> list[CitationSuggestion]:
    """Given a paragraph of LaTeX prose, return a list of citations that should be added."""
    client, model_id = _get_client_and_model(api_key, model)
    resp = client.chat.completions.create(
        model=model_id,
        messages=[
            {"role": "system", "content": _SUGGEST_SYSTEM},
            {"role": "user", "content": f"Paragraph (return JSON):\n---\n{paragraph}\n---"},
        ],
        temperature=0,
        response_format={"type": "json_object"},
    )
    content = _safe_extract_content(resp)
    try:
        data = json.loads(content or "{}")
    except json.JSONDecodeError:
        return []
    if not isinstance(data, dict):
        return []
    out: list[CitationSuggestion] = []
    for s in data.get("suggestions") or []:
        if not isinstance(s, dict):
            continue
        query = (s.get("query") or "").strip()
        anchor = (s.get("anchor") or "").strip()
        if not query or not anchor:
            continue
        year_val = s.get("expected_year")
        if isinstance(year_val, str):
            m = re.search(r"\d{4}", year_val)
            year_val = int(m.group(0)) if m else None
        elif not isinstance(year_val, int):
            year_val = None
        out.append(
            CitationSuggestion(
                query=query,
                anchor=anchor,
                reason=str(s.get("reason") or ""),
                expected_first_author=s.get("expected_first_author") or None,
                expected_year=year_val,
            )
        )
    return out


_BIBITEM_SYSTEM = """\
You extract structured citation metadata from free-text LaTeX bibliography entries
(the old `\\bibitem{...} ... ` format used before BibTeX).

Input is one bibitem block, e.g.:
  \\bibitem{tu2023medpalmm}
  T. Tu et al.
  Towards generalist biomedical AI.
  \\textit{arXiv:2307.14334}, 2023.
  \\url{https://arxiv.org/abs/2307.14334}

Respond with a single JSON object:
  {
    "title": "Towards generalist biomedical AI",
    "authors": ["Tu, T.", ...],   # use BibTeX "Last, First" format if possible
    "year": 2023,
    "arxiv_id": "2307.14334",      # or null
    "doi": "10.x/x",               # or null
  }

Preserve "et al." as a literal author if the original had it. Do not invent authors
you don't see in the block.
"""


def parse_bibitem(
    bibitem_text: str,
    *,
    cite_key: str = "",
    model: Optional[str] = None,
    api_key: Optional[str] = None,
) -> ParsedBibitem:
    """Extract structured metadata from a `\\bibitem{...}` block."""
    client, model_id = _get_client_and_model(api_key, model)
    resp = client.chat.completions.create(
        model=model_id,
        messages=[
            {"role": "system", "content": _BIBITEM_SYSTEM},
            {"role": "user", "content": bibitem_text},
        ],
        temperature=0,
        response_format={"type": "json_object"},
    )
    content = _safe_extract_content(resp)
    try:
        data = json.loads(content or "{}")
    except json.JSONDecodeError:
        data = {}
    if not isinstance(data, dict):
        data = {}

    # Year may come back as a string ("2023") or int — coerce.
    year_val = data.get("year")
    if isinstance(year_val, str):
        m = re.search(r"\d{4}", year_val)
        year_val = int(m.group(0)) if m else None
    elif isinstance(year_val, int):
        year_val = year_val
    else:
        year_val = None

    return ParsedBibitem(
        key=cite_key,
        title=(data.get("title") or None),
        authors=list(data.get("authors") or []),
        year=year_val,
        arxiv_id=(data.get("arxiv_id") or None),
        doi=(data.get("doi") or None),
        raw=bibitem_text,
    )


_VERIFY_MATCH_SYSTEM = """\
You are a STRICT academic citation matcher. Your DEFAULT verdict is REJECT
(same_paper=false). Only ACCEPT (same_paper=true) when ALL strict criteria hold.

You compare an ORIGINAL paper to a CANDIDATE from Google Scholar.

REJECT IMMEDIATELY (same_paper=false) if ANY of these is true:

  R1. The CANDIDATE's first-author surname does not match the ORIGINAL's first-author
      surname (case-insensitive, allowing transliteration like "Müller"≈"Mueller").
      Examples of REJECT: "Vaswani" vs "Mineault", "Moor" vs "Di", "Goodfellow" vs
      "Labaca-Castro", "Singhal" vs "Kanjilal".

  R2. The CANDIDATE's year is more than ±2 from the ORIGINAL's year (preprint→proceedings
      drift is at most ~2 years; anything beyond is a different paper).

  R3. The CANDIDATE is a DERIVATIVE work, not the ORIGINAL itself. Reject if the
      CANDIDATE title contains words like:
        - "Review", "Survey", "Overview" of the topic
        - "Applications of <X>", "<X> in <domain>" when X is the original system
        - "Towards", "Will", "Is" — often signal commentary/position papers
        - "Chapter ABOUT", "Foundations of", encyclopedia entries
        - Same title but a much later year suggesting a textbook chapter

  R4. The CANDIDATE's title describes a DIFFERENT subject matter from the ORIGINAL,
      even if some keywords overlap. Example: ORIGINAL "Foundation models for
      generalist medical AI" vs CANDIDATE "Will generalist medical AI be the future
      path for NLP models" — these are different papers despite shared phrase.

ACCEPT (same_paper=true) ONLY if ALL of these hold:

  A1. First-author surnames match (per R1).
  A2. Years agree within ±2 (per R2).
  A3. CANDIDATE title is semantically the same paper as ORIGINAL (minor
      punctuation/case/order differences OK; substantive title changes are NOT OK).
  A4. CANDIDATE is the ORIGINAL paper, not a derivative work (per R3).

Bias HEAVILY toward rejection. If you are not >90% certain it's the same paper,
return same_paper=false with confidence reflecting your uncertainty. A wrong accept
permanently corrupts the user's bibliography; a wrong reject just means the caller
retries with a different query.

Return a single JSON object:
  {
    "same_paper": true | false,
    "confidence": 0.0 to 1.0,
    "reasoning": "one short sentence naming the specific rule (R1-R4 or A1-A4) that applied"
  }
"""


def verify_match(
    bib_entry: dict,
    candidate: PaperHit,
    *,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
) -> MatchVerification:
    """Use the LLM as a judge to decide whether a Scholar hit is the same paper as the
    given .bib entry.

    On any failure (network, JSON decode, missing fields) returns a conservative
    ``same_paper=False`` verdict — better to leave an entry alone than to risk a wrong
    match. The caller decides what to do with the verdict.
    """
    try:
        client, model_id = _get_client_and_model(api_key, model)
    except Exception as e:
        return MatchVerification(False, 0.0, f"LLM client unavailable: {e}")

    bib_venue = (bib_entry.get("booktitle") or bib_entry.get("journal") or "").strip()
    user_msg = (
        "Are these the same paper? Return JSON.\n\n"
        "ORIGINAL .bib entry:\n"
        f"  title:   {bib_entry.get('title', '')!r}\n"
        f"  authors: {bib_entry.get('author', '')!r}\n"
        f"  year:    {bib_entry.get('year', '')!r}\n"
        f"  venue:   {bib_venue!r}\n\n"
        "CANDIDATE from Google Scholar:\n"
        f"  title:    {candidate.title!r}\n"
        f"  authors:  {', '.join(candidate.authors)!r}\n"
        f"  year:     {candidate.year}\n"
        f"  venue:    {(candidate.venue or '')!r}\n"
        f"  cited_by: {candidate.cited_by}\n"
    )

    try:
        resp = client.chat.completions.create(
            model=model_id,
            messages=[
                {"role": "system", "content": _VERIFY_MATCH_SYSTEM},
                {"role": "user", "content": user_msg},
            ],
            temperature=0,
            response_format={"type": "json_object"},
        )
    except Exception as e:
        return MatchVerification(False, 0.0, f"LLM call failed: {e}")

    content = _safe_extract_content(resp)
    if not content:
        return MatchVerification(False, 0.0, "no LLM response content")
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        return MatchVerification(False, 0.0, "LLM did not return valid JSON")
    if not isinstance(data, dict):
        return MatchVerification(False, 0.0, "LLM response was not a JSON object")

    return MatchVerification(
        same_paper=bool(data.get("same_paper", False)),
        confidence=float(data.get("confidence") or 0.0),
        reasoning=str(data.get("reasoning") or ""),
    )


@dataclass
class VerifiedPick:
    """Outcome of :func:`pick_verified_match` — either a Scholar hit the LLM endorsed
    as the same paper, or ``None`` with a human-readable rejection reason."""

    hit: Optional[PaperHit]
    confidence: float
    reasoning: str  # accepted-or-rejected explanation from the last LLM call
    candidates_considered: int = 0


def pick_verified_match(
    expected: dict,
    candidates: list[PaperHit],
    *,
    confidence_floor: float = 0.7,
    max_candidates: int = 3,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
) -> VerifiedPick:
    """Universal LLM-verified match picker, shared by every command that searches Scholar.

    Walks the top ``max_candidates`` of ``candidates`` and asks the LLM judge whether
    each is the same paper as ``expected`` (a dict with ``title``, ``author``, ``year``
    keys — only ``title`` is required). Returns the first candidate the LLM accepts with
    ``confidence >= confidence_floor``, otherwise ``hit=None`` with the LAST rejection
    reasoning so the caller can surface it to the user.

    Used by ``fix``, ``extract``, ``repair``, and ``add`` to ensure no wrong-paper
    Scholar hit ever reaches the .bib.
    """
    if not candidates:
        return VerifiedPick(None, 0.0, "no candidates supplied", 0)

    last_reasoning = ""
    last_confidence = 0.0
    considered = 0
    for candidate in candidates[:max_candidates]:
        considered += 1
        verdict = verify_match(expected, candidate, model=model, api_key=api_key)
        last_reasoning = verdict.reasoning
        last_confidence = verdict.confidence
        if verdict.same_paper and verdict.confidence >= confidence_floor:
            return VerifiedPick(candidate, verdict.confidence, verdict.reasoning, considered)

    return VerifiedPick(None, last_confidence, last_reasoning, considered)
