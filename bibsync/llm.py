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

from . import config, dbg
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
    """One claim in the prose that should be cited, plus search angles to find the paper.

    ``queries`` holds 2-3 alternative Google Scholar searches: usually one specific
    (title-like) and one or two broader (topic / author + year / system name).
    Searching multiple queries and merging candidates dramatically improves recall
    when Scholar's relevance ranking doesn't surface the canonical paper for the
    most obvious query.
    """

    queries: list[str]  # 2-3 Google Scholar search queries (most-specific first)
    anchor: str  # verbatim substring of the paragraph to insert \cite{} after
    reason: str  # one sentence: why this needs citation
    expected_first_author: Optional[str] = None
    expected_year: Optional[int] = None

    @property
    def query(self) -> str:
        """Back-compat: the first (most-specific) query."""
        return self.queries[0] if self.queries else ""


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
You identify citation targets in LaTeX prose and produce Google Scholar search
queries that will surface the CANONICAL ORIGINAL paper for each.

═══════════════════════════════════════════════════════════════════════════════
RULE 1 — One suggestion PER NAMED SYSTEM. ABSOLUTELY DO NOT GROUP.
═══════════════════════════════════════════════════════════════════════════════
When prose lists multiple named systems/methods/models — e.g. "Med-PaLM, Med-PaLM 2,
Med-PaLM M, and Med-Gemini" — you MUST emit ONE suggestion FOR EACH NAMED SYSTEM.
The anchor MUST be just the system name itself, not the surrounding phrase.

WRONG (a single grouped suggestion):
    {"anchor": "Med-PaLM, Med-PaLM 2, Med-PaLM M, and Med-Gemini approached", ...}

RIGHT (one suggestion per system, anchor = system name only):
    {"anchor": "Med-PaLM", "queries": [...]},
    {"anchor": "Med-PaLM 2", "queries": [...]},
    {"anchor": "Med-PaLM M", "queries": [...]},
    {"anchor": "Med-Gemini", "queries": [...]}

Same rule for "MedSAM, RETFound, UNI, and Prov-GigaPath" → four suggestions.
Same rule for "chess, Go, and poker" — if each implies the AI paper(s) for that
game, emit one suggestion per game with anchor "chess", "Go", "poker".

IMPORTANT: the anchor must be PLAIN TEXT only — DO NOT include any LaTeX commands
(no \\cite, no \\nocite, no \\textit, no curly braces). Just the verbatim text that
already exists in the input paragraph. The caller adds \\cite{} separately.

═══════════════════════════════════════════════════════════════════════════════
RULE 1b — What counts as a citation target
═══════════════════════════════════════════════════════════════════════════════
Cite when the prose attributes any of:
  • A named SYSTEM / MODEL ("Med-PaLM 2", "RETFound", "Prov-GigaPath", "BERT")
  • A named ALGORITHM / TECHNIQUE ("attention mechanism", "equilibrium search
    methods", "RLHF", "Monte Carlo tree search")
  • A specific QUANTITATIVE CLAIM that comes from a particular paper
    (e.g., "exceed 10^64 possible choices per turn" → cite the source paper)
  • A named GAME / DOMAIN being studied at the AI-research level
    ("chess" implies AlphaZero / Deep Blue; "Go" implies AlphaGo;
     "poker" implies Libratus / Pluribus; "Diplomacy" implies Cicero / DeepMind no-press)
  • A NAMED PARADIGM or finding ("pretraining-then-finetuning paradigm" → BERT;
    "scale alone could unlock emergent capabilities" → GPT-3 / Wei et al. emergent)
  • A foundational result attributed to specific authors / years

When in doubt for a domain-establishing claim, EMIT a suggestion — the caller
will reject candidates that don't actually support the claim. Missing a
citable claim is worse than proposing one that gets filtered.

═══════════════════════════════════════════════════════════════════════════════
RULE 2 — Queries must look like ACTUAL PAPER TITLES, not concept paraphrases.
═══════════════════════════════════════════════════════════════════════════════
Scholar's relevance ranking rewards keyword overlap with paper titles. A
conceptual query like "BERT pretraining-then-finetuning paradigm" finds surveys.
A title-like query like "BERT Pre-training Deep Bidirectional Transformers Devlin"
finds the actual BERT paper.

For each system/claim, produce 2-3 queries:
  - Q1: A title-like search that includes the SYSTEM NAME plus likely keywords
        from the canonical paper title.
  - Q2: SURNAME + YEAR + SYSTEM NAME (if you can guess the first author).
  - Q3: Distinctive descriptive phrase from the original paper (fallback).

GOOD query examples (find canonical papers):
  - "Attention Is All You Need Vaswani transformer"
  - "BERT Pre-training Deep Bidirectional Transformers Devlin"
  - "Language Models are Few-Shot Learners Brown GPT-3"
  - "Segment Anything in Medical Images Ma MedSAM"
  - "RETFound foundation model retinal images Zhou Nature"
  - "Prov-GigaPath whole-slide pathology foundation model"

BAD query examples (find surveys/reviews/derivatives — DON'T DO THIS):
  - "Attention mechanism transformer architecture"     ← too conceptual
  - "BERT pretraining-then-finetuning paradigm"        ← describes effect
  - "GPT-style decoder-only models emergent capabilities" ← paraphrase
  - "MedSAM imaging model 2023"                        ← too generic

═══════════════════════════════════════════════════════════════════════════════
Response format
═══════════════════════════════════════════════════════════════════════════════
Return a single JSON object:
  {"suggestions": [
      {
        "queries": ["title-like query", "<author> <year> <system>", "fallback"],
        "anchor": "EXACT substring of the input paragraph — usually just the
                   system name. \\cite{} is inserted RIGHT AFTER this substring.",
        "reason": "one sentence on why this needs a citation",
        "expected_first_author": "surname guess or null",
        "expected_year": 2023
      },
      ...
  ]}

If the paragraph has NO genuine attributions needing citation (pure narrative,
methods boilerplate), return JSON {"suggestions": []}.
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
        # Prefer ``queries`` (list) — fall back to a single ``query`` for back-compat.
        raw_queries = s.get("queries")
        if isinstance(raw_queries, list):
            queries = [str(q).strip() for q in raw_queries if str(q).strip()]
        else:
            single = (s.get("query") or "").strip()
            queries = [single] if single else []
        anchor = (s.get("anchor") or "").strip()
        # Sanitise the anchor: LLM sometimes hallucinates \nocite{}, \cite{}, \textit{},
        # or stray braces into the anchor. Strip any \xxx{...} command sequences plus
        # leftover backslashes and braces, so the anchor stays plain text we can find
        # as a substring of the user's actual paragraph.
        anchor = re.sub(r"\\[a-zA-Z]+\*?\s*(?:\[[^\]]*\])?\s*\{[^}]*\}", "", anchor)
        anchor = re.sub(r"\\[a-zA-Z]+\*?", "", anchor)
        anchor = anchor.replace("{", "").replace("}", "").strip()
        anchor = re.sub(r"\s+", " ", anchor).strip()
        if not queries or not anchor:
            continue
        year_val = s.get("expected_year")
        if isinstance(year_val, str):
            m = re.search(r"\d{4}", year_val)
            year_val = int(m.group(0)) if m else None
        elif not isinstance(year_val, int):
            year_val = None
        out.append(
            CitationSuggestion(
                queries=queries,
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
You are a semantic citation matcher. The caller has ALREADY verified, in code, that:
  - The CANDIDATE's first-author surname matches the ORIGINAL's first-author surname.
  - The CANDIDATE's year is within ±3 of the ORIGINAL's year.
  - The CANDIDATE's title is highly similar to the ORIGINAL's title.

DO NOT re-check year arithmetic. DO NOT re-check author surnames. Those have been
deterministically validated already; your math will be less reliable than the code's.

Your ONLY job: decide whether the CANDIDATE is the SAME work as the ORIGINAL, or a
DIFFERENT work (a derivative or unrelated paper) that happens to be structurally similar.

REJECT (same_paper=false) ONLY if the CANDIDATE is a DERIVATIVE work, such as:
  - "A review of <X>" / "A survey of <X>" / "<X>: an overview"
  - "Applications of <X> in <domain>"
  - "Towards <X>", "Is <X>?", "Will <X>?" — typically commentary/position pieces
  - A book chapter ABOUT or building on the original, not the original paper itself
  - A follow-up paper by similar authors but on a different topic

ACCEPT (same_paper=true) if the CANDIDATE is the same paper as the ORIGINAL — same
work, same topic, structural fields already match. Title and year drift between
arXiv preprints and published proceedings is normal and ACCEPTABLE.

When the structural fields already match and you see no derivative-work signal,
ACCEPT. Refusing the obvious original wastes user time without protecting them.

Return a single JSON object:
  {
    "same_paper": true | false,
    "confidence": 0.0 to 1.0,
    "reasoning": "one short sentence — 'same work, original paper' or naming the derivative-work signal"
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
        dbg.trace("llm.verify", "ERR client unavailable", error=str(e))
        return MatchVerification(False, 0.0, f"LLM client unavailable: {e}")

    bib_venue = (bib_entry.get("booktitle") or bib_entry.get("journal") or "").strip()

    # Pre-extract first-author surnames so the LLM sees normalized, single-token
    # comparison inputs (instead of "Goodfellow, Ian and Pouget-Abadie and ..." which
    # gpt-4o-mini has been observed to misread).
    def _surname_from(field: str) -> str:
        if not field:
            return ""
        first = field.split(" and ")[0].strip()
        if "," in first:
            return first.split(",")[0].strip()
        parts = first.split()
        return parts[-1] if parts else ""

    bib_surname = _surname_from(bib_entry.get("author", "") or "")
    cand_surname = ""
    if candidate.authors:
        parts = candidate.authors[0].split()
        cand_surname = parts[-1] if parts else ""

    dbg.trace(
        "llm.verify",
        "calling",
        model=model_id,
        bib_title=bib_entry.get("title", ""),
        bib_surname=bib_surname,
        bib_year=bib_entry.get("year", ""),
        cand_title=candidate.title,
        cand_surname=cand_surname,
        cand_year=candidate.year,
        cand_cited=candidate.cited_by,
    )
    user_msg = (
        "Structural fields already match (surname, year ±3, title similarity). "
        "Decide whether the CANDIDATE is the SAME WORK or a DERIVATIVE. Return JSON.\n\n"
        "ORIGINAL .bib entry:\n"
        f"  title:                {bib_entry.get('title', '')!r}\n"
        f"  first_author_surname: {bib_surname!r}\n"
        f"  year:                 {bib_entry.get('year', '')!r}\n"
        f"  venue:                {bib_venue!r}\n\n"
        "CANDIDATE from Google Scholar:\n"
        f"  title:                {candidate.title!r}\n"
        f"  first_author_surname: {cand_surname!r}\n"
        f"  year:                 {candidate.year}\n"
        f"  venue:                {(candidate.venue or '')!r}\n"
        f"  cited_by:             {candidate.cited_by}\n"
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

    verdict = MatchVerification(
        same_paper=bool(data.get("same_paper", False)),
        confidence=float(data.get("confidence") or 0.0),
        reasoning=str(data.get("reasoning") or ""),
    )
    dbg.trace(
        "llm.verify",
        "verdict",
        same=verdict.same_paper,
        conf=round(verdict.confidence, 2),
        reason=verdict.reasoning,
    )
    return verdict


@dataclass
class ClaimSupport:
    """Verdict from the LLM-as-judge that decides whether a Scholar hit is the right
    paper to cite for a particular prose claim. Used by ``suggest``."""

    supports: bool
    confidence: float  # 0.0 - 1.0
    reasoning: str


_CLAIM_SUPPORT_SYSTEM = """\
You are a citation expert. Decide whether a CANDIDATE paper from Google Scholar is
the RIGHT paper to cite for a CLAIM in academic prose.

ACCEPT (supports=true) when ANY of these hold (each is sufficient on its own):

  A. The CANDIDATE is the canonical work that INTRODUCED the named system / method
     / dataset the CLAIM attributes. The paper's title need NOT literally contain
     the system name — many canonical papers describe the system as "a foundation
     model for X" rather than naming it in the title. Example: the Prov-GigaPath
     paper is titled "A whole-slide foundation model for digital pathology" — that
     IS the right paper to cite for any claim about Prov-GigaPath.

  B. The CANDIDATE has VERY HIGH citation count (cited_by ≥ 500) AND topic-matches
     the claim. High citation count + topic match is overwhelming evidence of
     canonicality; derivatives almost never reach 500+ citations.

  C. The CANDIDATE clearly introduces a named result that the claim attributes,
     and the author / year / venue are plausible.

REJECT (supports=false) when ANY of these hold:

  X. The CANDIDATE is an EXPLICIT VERSION VARIANT. If the claim names "MedSAM" and
     the CANDIDATE title is "Medical SAM 2", "MedSAM 2", "MedSAM v2", or similar
     "X 2" / "X+" — REJECT. A sequel is a different paper. Same goes for
     "BERT" vs "RoBERTa", "GPT" vs "GPT-2", etc. UNLESS the claim explicitly
     references the variant.

  Y. The CANDIDATE is a SURVEY, REVIEW, retrospective, comprehensive overview, or
     book chapter, AND there is no indication this is the ORIGINAL work.
     Title signals: "A survey of", "A review of", "Comprehensive analysis of",
     "Applications of X in Y", "An overview of".

  Z. The CANDIDATE evaluates / replicates / extends another work and is NOT itself
     the foundational introduction. Title signals: "Independent evaluation of X",
     "Replicability of X", "Assessing X", "Extending X to Y".

  W. The CANDIDATE is on a clearly different topic (only superficial keyword overlap).

Heuristic priority: A and B together are STRONG accept signals — when a paper has
both high citation count and clear topic match, accept even if the title doesn't
contain the named system literally. When in doubt with cited_by ≥ 500, ACCEPT.

Return a single JSON object:
  {
    "supports": true | false,
    "confidence": 0.0 to 1.0,
    "reasoning": "one short sentence — name the specific accept (A/B/C) or reject (X/Y/Z/W) signal"
  }
"""


def verify_claim_support(
    claim_text: str,
    context: str,
    candidate: PaperHit,
    *,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
) -> ClaimSupport:
    """LLM-as-judge: does this Scholar candidate actually support the prose claim?

    Conservative on failure — any error returns ``supports=False`` so the caller falls
    through to the next candidate / query rather than writing a wrong citation.
    """
    try:
        client, model_id = _get_client_and_model(api_key, model)
    except Exception as e:
        dbg.trace("llm.claim_support", "ERR client unavailable", error=str(e))
        return ClaimSupport(False, 0.0, f"LLM client unavailable: {e}")

    cand_surname = ""
    if candidate.authors:
        parts = candidate.authors[0].split()
        cand_surname = parts[-1] if parts else ""

    dbg.trace(
        "llm.claim_support",
        "calling",
        model=model_id,
        claim=claim_text,
        cand_title=candidate.title,
        cand_surname=cand_surname,
        cand_year=candidate.year,
        cand_cited=candidate.cited_by,
    )

    user_msg = (
        "Does the CANDIDATE paper support the CLAIM? Return JSON.\n\n"
        f"CLAIM (anchor in prose):\n  {claim_text!r}\n\n"
        f"CONTEXT (surrounding paragraph):\n  {context!r}\n\n"
        "CANDIDATE from Google Scholar:\n"
        f"  title:                {candidate.title!r}\n"
        f"  first_author_surname: {cand_surname!r}\n"
        f"  year:                 {candidate.year}\n"
        f"  venue:                {(candidate.venue or '')!r}\n"
        f"  cited_by:             {candidate.cited_by}\n"
    )

    try:
        resp = client.chat.completions.create(
            model=model_id,
            messages=[
                {"role": "system", "content": _CLAIM_SUPPORT_SYSTEM},
                {"role": "user", "content": user_msg},
            ],
            temperature=0,
            response_format={"type": "json_object"},
        )
    except Exception as e:
        return ClaimSupport(False, 0.0, f"LLM call failed: {e}")

    content = _safe_extract_content(resp)
    if not content:
        return ClaimSupport(False, 0.0, "no LLM response content")
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        return ClaimSupport(False, 0.0, "LLM did not return valid JSON")
    if not isinstance(data, dict):
        return ClaimSupport(False, 0.0, "LLM response was not a JSON object")

    verdict = ClaimSupport(
        supports=bool(data.get("supports", False)),
        confidence=float(data.get("confidence") or 0.0),
        reasoning=str(data.get("reasoning") or ""),
    )
    dbg.trace(
        "llm.claim_support",
        "verdict",
        supports=verdict.supports,
        conf=round(verdict.confidence, 2),
        reason=verdict.reasoning,
    )
    return verdict


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
        dbg.trace("llm.pick", "no candidates supplied")
        return VerifiedPick(None, 0.0, "no candidates supplied", 0)

    dbg.trace(
        "llm.pick",
        "starting",
        candidates=len(candidates),
        will_check=min(len(candidates), max_candidates),
        floor=confidence_floor,
    )

    last_reasoning = ""
    last_confidence = 0.0
    considered = 0
    for candidate in candidates[:max_candidates]:
        considered += 1
        dbg.trace("llm.pick", f"checking candidate #{considered}", title=candidate.title)
        verdict = verify_match(expected, candidate, model=model, api_key=api_key)
        last_reasoning = verdict.reasoning
        last_confidence = verdict.confidence
        if verdict.same_paper and verdict.confidence >= confidence_floor:
            dbg.trace(
                "llm.pick",
                "ACCEPTED",
                index=considered,
                conf=round(verdict.confidence, 2),
                title=candidate.title,
            )
            return VerifiedPick(candidate, verdict.confidence, verdict.reasoning, considered)

    dbg.trace("llm.pick", "REJECTED all", considered=considered, last_reason=last_reasoning)
    return VerifiedPick(None, last_confidence, last_reasoning, considered)
