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

(The caller automatically prepends two deterministic queries — `"<anchor>" original
paper` and `"<anchor>"` — so do NOT include those literal patterns. Your queries
should add additional angles the deterministic queries miss.)

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
class IdentifiedPaper:
    """The LLM's best guess at the canonical paper for a citation anchor, using its
    training-data knowledge plus the surrounding document context.

    Returned by :func:`identify_canonical_paper`. The Scholar search uses
    ``expected_title`` + ``expected_first_author`` as a near-deterministic query —
    far more reliable than topic-guessing queries.
    """

    expected_title: str
    expected_first_author: Optional[str] = None
    expected_year: Optional[int] = None
    expected_venue: Optional[str] = None
    arxiv_id: Optional[str] = None
    confidence: float = 0.0  # 0.0 - 1.0
    reasoning: str = ""


_IDENTIFY_PAPER_SYSTEM = """\
You are an academic-citation expert with broad knowledge of ML, AI, medical AI,
NLP, vision, systems, and related research literature. The user is writing a paper
and needs to cite a specific named system / method / dataset / claim from their prose.

YOUR JOB: identify the SPECIFIC CANONICAL paper that should be cited.

You will receive:
  • CLAIM — a short anchor phrase (e.g. "UNI", "Med-PaLM 2", "Cicero", "attention mechanism")
  • PARAGRAPH — the local sentence/paragraph where the claim appears
  • DOCUMENT — an excerpt of the surrounding paper to disambiguate ambiguous anchors

Use your TRAINING-DATA WORLD KNOWLEDGE to name the canonical paper. The DOCUMENT
context helps disambiguate (e.g. "UNI" inside a pathology-foundation-models list
means Chen et al. 2024 Nature Medicine, not "University of Northern Iowa").

Return JSON:
  {
    "expected_title": "the actual paper title (your best guess from world knowledge)",
    "expected_first_author": "surname of the first author (e.g. 'Chen', 'Devlin', 'Vaswani')",
    "expected_year": 2024,
    "expected_venue": "Nature Medicine / NeurIPS / ICML / arXiv / ...",
    "arxiv_id": "2307.14334 if you know it, else null",
    "confidence": 0.0 to 1.0,
    "reasoning": "one sentence — what tells you this is the right paper"
  }

WORKED EXAMPLES:

  CLAIM "UNI" + pathology FM context →
    title    : "Towards a general-purpose foundation model for computational pathology"
    author   : "Chen"   year : 2024   venue : "Nature Medicine"
    conf     : 0.95
    reason   : "UNI is the foundation model from Mahmood lab introduced in Chen et al. 2024"

  CLAIM "BERT" + language model pretraining context →
    title    : "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding"
    author   : "Devlin"  year : 2019  venue : "NAACL"  conf : 1.0

  CLAIM "Med-Gemini" + medical LLM context →
    title    : "Capabilities of Gemini models in medicine"
    author   : "Saab"   year : 2024  venue : "arXiv"  arxiv_id : "2404.18416"  conf : 0.9
    reason   : "Saab et al. 2024 from Google DeepMind introduced the Med-Gemini family"

  CLAIM "attention mechanism" + transformer/sequence-modeling context →
    title    : "Attention Is All You Need"
    author   : "Vaswani"  year : 2017  venue : "NeurIPS"  conf : 0.95
    reason   : "Canonical paper for self-attention displacing recurrence is Vaswani 2017"

  CLAIM "Cicero" + Diplomacy AI context →
    title    : "Human-level play in the game of Diplomacy by combining language models with strategic reasoning"
    author   : "Bakhtin"  year : 2022  venue : "Science"  conf : 0.95

When you are NOT confident (very recent paper, ambiguous acronym you can't disambiguate
even with context, or a generic phrase that doesn't map to a specific paper), return
a low confidence (<0.4) and pick the BEST guess you have. The caller will verify via
Scholar search and may fall back to topic-only queries.

Do NOT make up titles or authors with high confidence. Better to return a guess with
confidence 0.3 than a hallucinated specific title with confidence 0.9.
"""


def identify_canonical_paper(
    claim: str,
    paragraph: str,
    document_context: str = "",
    *,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
) -> IdentifiedPaper:
    """Ask the LLM to name the canonical paper for a citation anchor from world knowledge.

    The returned ``expected_title`` is used to build a targeted Scholar search query;
    the other fields are used to cross-verify Scholar's top result.
    """
    try:
        client, model_id = _get_client_and_model(api_key, model)
    except Exception as e:
        dbg.trace("llm.identify", "ERR client unavailable", error=str(e))
        return IdentifiedPaper(expected_title="", confidence=0.0, reasoning=f"LLM unavailable: {e}")

    dbg.trace("llm.identify", "calling", model=model_id, claim=claim)

    # Cap context lengths to keep the prompt bounded.
    paragraph_excerpt = paragraph[:1200]
    doc_excerpt = document_context[:3500] if document_context else ""

    user_msg = (
        f"CLAIM (anchor phrase to cite): {claim!r}\n\n"
        f"PARAGRAPH (where the claim appears):\n{paragraph_excerpt}\n\n"
        f"DOCUMENT CONTEXT (excerpt of the surrounding paper):\n{doc_excerpt}\n\n"
        "Identify the canonical paper. Return JSON."
    )

    try:
        resp = client.chat.completions.create(
            model=model_id,
            messages=[
                {"role": "system", "content": _IDENTIFY_PAPER_SYSTEM},
                {"role": "user", "content": user_msg},
            ],
            temperature=0,
            response_format={"type": "json_object"},
        )
    except Exception as e:
        return IdentifiedPaper(
            expected_title="", confidence=0.0, reasoning=f"LLM call failed: {e}"
        )

    content = _safe_extract_content(resp)
    if not content:
        return IdentifiedPaper(expected_title="", confidence=0.0, reasoning="no LLM response")
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        return IdentifiedPaper(
            expected_title="", confidence=0.0, reasoning="LLM did not return valid JSON"
        )
    if not isinstance(data, dict):
        return IdentifiedPaper(
            expected_title="", confidence=0.0, reasoning="LLM response was not a JSON object"
        )

    year_val = data.get("expected_year")
    if isinstance(year_val, str):
        m = re.search(r"\d{4}", year_val)
        year_val = int(m.group(0)) if m else None
    elif not isinstance(year_val, int):
        year_val = None

    identified = IdentifiedPaper(
        expected_title=str(data.get("expected_title") or "").strip(),
        expected_first_author=(data.get("expected_first_author") or None),
        expected_year=year_val,
        expected_venue=(data.get("expected_venue") or None),
        arxiv_id=(data.get("arxiv_id") or None),
        confidence=float(data.get("confidence") or 0.0),
        reasoning=str(data.get("reasoning") or ""),
    )
    dbg.trace(
        "llm.identify",
        "result",
        title=identified.expected_title,
        author=identified.expected_first_author,
        year=identified.expected_year,
        venue=identified.expected_venue,
        arxiv=identified.arxiv_id,
        conf=round(identified.confidence, 2),
    )
    return identified


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
     the system name — canonical papers very commonly describe the system without
     naming it. Recognise these patterns as canonical introductions:
       - "A foundation model for <domain>" (e.g., introduces Prov-GigaPath, RETFound)
       - "<Capability> in/for <domain>" with high cited_by (e.g. "Capabilities of
         gemini models in medicine" IS the Med-Gemini paper)
       - "<Verb> anything in <domain>" (e.g. "Segment anything in medical images"
         IS the MedSAM paper)
       - Generally: the FIRST/EARLIEST paper from the originating research group
         that describes the system, regardless of title wording

  B. The CANDIDATE has NOTABLE citation count (cited_by ≥ 200) AND topic-matches
     the claim. Citations + topic match is strong evidence of canonicality;
     derivatives, follow-ups and surveys can also reach a few hundred cites, so this
     rule complements (does not replace) rule A and the rejection rules below.

  C. The CANDIDATE clearly introduces a named result that the claim attributes,
     and the author / year / venue are plausible.

REJECT (supports=false) when ANY of these hold:

  X. The CANDIDATE TITLE contains an EXPLICIT CONFLICTING VERSION of the system in
     the claim. Concretely: this rule fires ONLY when the candidate title literally
     mentions the system name followed by a version that DIFFERS from the claim.

     RULE X fires (REJECT):
       - claim "MedSAM" + candidate "Medical SAM 2" / "MedSAM 2" / "MedSAM v2"
       - claim "GPT-3" + candidate "GPT-2: ..." / "GPT-4 Technical Report"
       - claim "BERT" + candidate "RoBERTa: A Robustly Optimized BERT ..."

     RULE X DOES NOT FIRE (don't reject just on this — assess via A/B/C instead):
       - claim "MedSAM" + candidate "Segment Anything in Medical Images"
         (title doesn't mention any MedSAM version; could still BE the MedSAM paper
         — many canonical papers don't put the system name in the title; check A/B)
       - claim "Med-Gemini" + candidate "Capabilities of gemini models in medicine"
         (no conflicting version; topic + cited_by are the right signals)
       - claim "GPT-3" + candidate "Language Models are Few-Shot Learners"
         (no version conflict in title — this is the GPT-3 paper)

     KEY MENTAL TEST: if the candidate title doesn't mention any version number at
     all, you CANNOT reject on rule X — pivot to A (introduces the system) or
     B (high cited_by + topic match).

  Y. The CANDIDATE is a SURVEY, REVIEW, retrospective, comprehensive overview,
     ANALYSIS / PROBING study, BENCHMARK, EVALUATION, COMMENTARY, position paper,
     or book chapter — and there is no indication it is itself the ORIGINAL work.

     STRONG title signals (REJECT immediately if any match):
       - "A survey of X" / "A review of X" / "An overview of X"
       - "Comprehensive analysis of X" / "Comprehensive evaluation of X"
       - "Applications of X in <Y>" / "X in <domain>: a review"
       - "What does X learn" / "Understanding X" / "Analyzing X" / "Probing X"
         (these are analysis papers ABOUT X, never the original X paper)
       - "How does X work" / "Where it comes and where it goes" (descriptive
         survey phrasing)
       - "Revisiting X" / "Rethinking X" / "Beyond X" (commentary / reformulation)
       - "X: where it comes and where it goes" — verbose descriptive subtitle
       - "Towards X" without strong evidence of being the originating paper
       - Titles with ":" subtitle that is a definition/description rather than
         the system name itself

     EXAMPLE (must REJECT):
       - claim "BERT" + candidate "What does BERT learn about the structure of
         language?" → REJECT (probing study, not the BERT introduction)
       - claim "attention mechanism" + candidate "Attention mechanism in neural
         networks: where it comes and where it goes" → REJECT (survey)
       - claim "RETFound" + candidate "Independent evaluation of RETFound on
         optic nerve" → REJECT (evaluation, not original RETFound paper)

  Z. The CANDIDATE evaluates / replicates / extends another work and is NOT itself
     the foundational introduction. Title signals: "Independent evaluation of X",
     "Replicability of X", "Assessing X", "Extending X to Y", "Improving X via Y".

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
class CitationAudit:
    """Verdict from the audit agent that decides whether an EXISTING citation in
    LaTeX prose actually supports the claim it's attached to. Used by ``bibsync audit``.

    This is distinct from :class:`ClaimSupport`: that one asks "is this the canonical
    paper to cite?"; this one asks the broader "is this paper a reasonable citation
    for this claim, or is it a topic mismatch / hallucination?". A non-canonical
    paper can still be a fine citation.

    ``contradicted`` is the strongest negative signal: the retrieved evidence
    actively refutes the claim (e.g. claim says "90% on MedQA", paper chunk says
    "17.8% on MedQA"). The caller maps this to a distinct user-facing status so
    the user can FIX the prose rather than DELETE the citation.
    """

    supports: bool
    confidence: float  # 0.0 - 1.0
    reasoning: str
    contradicted: bool = False  # paper actively reports something different


_AUDIT_CITATION_SYSTEM = """\
You are auditing an EXISTING citation in academic prose. The user has a CLAIM and a
specific CITED PAPER. Decide whether the cited paper SUPPORTS the claim — i.e.,
whether it's a reasonable scholarly source for what the claim attributes.

This is DIFFERENT from picking the most canonical paper. A non-canonical paper can
still be a fine citation if its topic / contribution aligns with the claim. You are
auditing for misattribution and hallucination, not for citation perfection.

SUPPORTS (supports=true) if ANY of these hold:
  - The paper's TOPIC matches what the claim is about (e.g., the claim is about
    transformer attention and the paper is "Attention Is All You Need").
  - The paper's CONTRIBUTION is what the claim attributes to it.
  - A reviewer would not flag this citation as a misattribution.
  - The paper is a reasonable scholarly source for this claim, even if not the
    most canonical one.

DOES NOT SUPPORT (supports=false) — flag as hallucination/misattribution if:
  - The paper is on a CLEARLY DIFFERENT TOPIC than the claim. Examples:
      * claim about NLP / language model + paper about computer vision/image
      * claim about computational pathology + paper about retinal images
      * claim about speech recognition + paper about transformer translation
  - The paper is in a different field that doesn't address the claim's subject.
  - The paper's contribution is unrelated to or contradicts the claim.
  - The citation looks fabricated (paper title and claim subject have no overlap).
  - **SURVEY / REVIEW / TUTORIAL cited for an "introduced / proposed / originally
    developed" CLAIM.** If the claim attributes a method, system, or finding to
    being INTRODUCED, PROPOSED, ESTABLISHED, or ORIGINALLY developed in the
    cited paper, AND the cited paper is a survey/review/tutorial/analysis/
    probing study (title or abstract contains "survey of", "review of",
    "overview of", "tutorial on", "analysis of X", "what does X learn",
    "where it comes and where it goes", "Comprehensive evaluation",
    "Revisiting X" without strong evidence it's the original), then
    supports=false with high confidence. A survey paper documents a method
    that ALREADY EXISTS; it cannot be the introduction of that method even
    when topic alignment is perfect. Cite the ORIGINAL paper instead.

    EXAMPLES that MUST be rejected:
      claim: "The attention mechanism was introduced to address long-range
              dependency limitations"
      paper: "Attention mechanism in neural networks: where it comes and
              where it goes" (a survey)
      → supports=false, conf~0.85, reasoning="cited paper is a survey of the
        attention mechanism, not its introduction; cite Bahdanau 2014 /
        Vaswani 2017 instead"

      claim: "BERT introduced bidirectional masked language modeling"
      paper: "What does BERT learn about the structure of language?"
              (a probing study)
      → supports=false, conf~0.85, reasoning="cited paper is a probing
        analysis of BERT, not its introduction; cite Devlin 2019 instead"

    The topic-alignment rule does NOT override this. A survey of X is on
    the topic of X but is not the source for "X was introduced by ..." or
    "X was originally proposed in ...". Watch for attribution verbs.

    IMPORTANT: when you reject a citation because the paper is a survey,
    set ``"contradicted": false``. This is a misattribution
    (wrong-paper-for-the-attribution), NOT a value-contradiction. The
    survey paper doesn't report a DIFFERENT specific value for the same
    entity — it just isn't the original source. ``contradicted=true`` is
    reserved for the "claim says X=Y, paper says X=Z" pattern (see Tier-2
    suffix), not for "claim says paper introduced X, but paper is a
    survey of X".

BE CONSERVATIVE: when in doubt, return supports=true with lower confidence. A
wrong "this is hallucinated" verdict makes the user delete a good citation. A
missed hallucination is recoverable later.

But when topic mismatch is OBVIOUS (e.g., a CV paper for an NLP claim), flag it
confidently. False supports>0.85 should only be returned when you're certain.

Return a single JSON object:
  {
    "supports":    true | false,
    "confidence":  0.0 to 1.0,
    "reasoning":   "one short sentence — what tells you it does or doesn't support",
    "contradicted": true | false   // OPTIONAL — see Tier-2 suffix
  }

The ``contradicted`` field defaults to false. Set it to true ONLY when the
retrieved evidence actively REFUTES the claim with a specific conflicting
value (e.g. claim says "X reaches 90% on Y", retrieved chunk says "X reaches
17.8% on Y"). Mere absence of evidence is NOT contradiction — that's just
unsupported. See the Tier-2 suffix for the precise rule.
"""


_AUDIT_TIER0_SUFFIX = """\

═════════════════════════════════════════════════════════════════════
EVIDENCE: paper METADATA ONLY (title + authors + year + venue)
═════════════════════════════════════════════════════════════════════
You have NO abstract and NO full-text excerpts. You see only the cited paper's
bibliographic metadata plus your own training-data knowledge.

═════════════════════════════════════════════════════════════════════
SOURCE-RESOLUTION SIGNAL (read this first — it changes the rules)
═════════════════════════════════════════════════════════════════════
The user message includes ``source_resolution: found | empty | unknown``:

  • ``found``   — at least one of arXiv, Semantic Scholar, OpenAlex, or
                  Crossref returned a paper matching this metadata. The
                  paper exists. Apply the normal rules below.
  • ``empty``   — ALL source adapters MISSED. No public database has a
                  paper with this title + author combination. Most likely
                  causes: the citation is FABRICATED (LLM-generated
                  hallucination), the title is mangled, or it's a
                  very-recent / obscure / non-indexed work.
  • ``unknown`` — source resolution wasn't performed (rare; legacy code
                  path). Treat conservatively as if ``empty``.

WHEN source_resolution=empty:
  ── If the title matches a clearly canonical paper you know exists in
     your training data (e.g. "Attention Is All You Need" by Vaswani),
     return supports=true at the normal confidence level. The miss is
     just an adapter failure.
  ── If the title is plausibly-real-sounding but you do NOT recognise
     it as a canonical work, return supports=false with confidence
     ~0.75 and reasoning "source-fetch-empty; paper not indexed in any
     of arXiv/SS/OpenAlex/Crossref — citation may be fabricated". The
     caller will route this to ``unverifiable`` or ``hallucinated``
     depending on the confidence floor; never auto-deleting good cites.
  ── This rule supersedes "topic-level claims with on-topic title get
     supports=true." A fabricated paper has an on-topic title BY
     CONSTRUCTION (the LLM generated it to fit the claim).

Hard rules for metadata-only verdicts:

  • TOPIC-LEVEL claims (e.g. "the Transformer architecture introduced
    self-attention", "BERT pretrains bidirectional transformers") — if the
    paper title clearly names the topic/method, supports=true is appropriate.

  • QUANTITATIVE claims (a specific number, percentage, benchmark name, dataset
    name, parameter count, accuracy, F1, etc.) — DO NOT high-confidence verify
    on metadata alone. The title doesn't carry quantitative content. Even if
    the paper plausibly contains the number, you have no way to know from
    metadata. Return supports=false with MEDIUM confidence (~0.5–0.65) and say
    "metadata insufficient for quantitative claim" in reasoning. The caller
    treats medium-conf supports=false as `unverifiable`, not as
    `hallucinated`, so this won't delete the citation — it only refuses to
    rubber-stamp it.

  • NAMED-BENCHMARK claims ("X achieves Y on Z benchmark") — same rule. The
    title rarely names every benchmark a paper reports. Refuse to verify with
    high confidence.

  • OBVIOUS TOPIC MISMATCH (a CV paper for an NLP claim) — supports=false high
    confidence is still appropriate even with metadata only.

Examples:
  claim: "GPT-3 achieves 86.5% on MedQA"  →  cited: "Language Models are
  Few-Shot Learners" (Brown 2020). Topic plausible, BUT the claim is
  quantitative + named-benchmark. From metadata alone you cannot confirm the
  paper actually reports 86.5% on MedQA. Verdict: supports=false, conf~0.55,
  reasoning="quantitative MedQA accuracy claim cannot be verified from
  metadata alone; needs abstract or full-text".
"""


_AUDIT_TIER1_SUFFIX = """\

═════════════════════════════════════════════════════════════════════
EVIDENCE: paper ABSTRACT (use this as the primary source of truth)
═════════════════════════════════════════════════════════════════════
The user provides the cited paper's abstract. Use it as the PRIMARY evidence,
not just the title. An abstract is the most reliable single summary of what the
paper actually claims.

When the abstract is available:
  • If the abstract clearly addresses the claim → supports=true (high confidence).
  • If the abstract is on the same general topic but says nothing about the
    specific point the claim attributes → supports=false (medium confidence,
    flag as unsupported rather than hallucinated).
  • If the abstract describes a DIFFERENT topic entirely → supports=false (high
    confidence — clear misattribution).

Cite specific phrases from the abstract in your reasoning (one short quote is
enough). This makes the verdict auditable.
"""

_AUDIT_TIER2_SUFFIX = """\

═════════════════════════════════════════════════════════════════════
EVIDENCE: paper ABSTRACT + retrieved CHUNKS from the paper's full text
═════════════════════════════════════════════════════════════════════
You receive both the paper's abstract AND several excerpts retrieved by semantic
search against the prose claim. The excerpts are direct evidence; the abstract
is for context.

═════════════════════════════════════════════════════════════════════
HARD RULES — verbatim grounding (READ CAREFULLY)
═════════════════════════════════════════════════════════════════════

For QUANTITATIVE claims (a specific number, percentage, accuracy, parameter
count) OR NAMED-ENTITY claims (a named benchmark like MedQA / ImageNet / GLUE,
a named dataset, a named method): you MUST find a verbatim phrase in one of the
retrieved chunks that contains the SPECIFIC number / benchmark / entity from
the claim before returning supports=true.

  • If the claim says "GPT-3 achieves 86.5% on MedQA": you must quote a chunk
    containing both "MedQA" AND a number ≈ 86.5 (or close enough). It is NOT
    enough that the chunks mention "various benchmarks" or "question
    answering". Generic-topic alignment does NOT verify a specific number or
    named benchmark.

  • If the claim says "BERT-base has 110M parameters": you must quote a chunk
    containing "110M" (or "110 million", or "BERT-base ... parameters" with a
    matching number).

  • If you CANNOT find a verbatim quote with the specific entity/number,
    return supports=false with medium confidence (≤0.7) and put the missing
    entity/number in your reasoning: e.g. "no chunk mentions MedQA or 86.5%
    — likely misattribution".

For TOPIC-LEVEL claims (no specific number, no specific named benchmark) —
e.g. "introduced multi-head self-attention", "established pretraining-then-
finetuning paradigm" — a chunk that discusses the same general method or
contribution is sufficient. Quote it anyway.

In your reasoning field:
  • If supports=true: quote the chunk by page reference, e.g.
    "p.7 says 'BERT_BASE: L=12, H=768, A=12, Total Parameters=110M'".
  • If supports=false on a quantitative/named-entity claim: explicitly state
    what's MISSING from the chunks, e.g. "no chunk contains 'MedQA' or any
    number near 86.5%".

DO NOT paraphrase generously. DO NOT infer that because the paper "evaluates
many benchmarks" it must therefore evaluate the specific benchmark in the
claim. The retrieved chunks are the only evidence — if a number or named
entity isn't in them, you cannot verify it.

═════════════════════════════════════════════════════════════════════
CONTRADICTION DETECTION (contradicted=true) — read in full
═════════════════════════════════════════════════════════════════════
Set ``"contradicted": true`` whenever the retrieved chunks NAME the same
entity from the claim but report a DIFFERENT specific value, version, or
identity. This is a STRUCTURED checklist — walk it before deciding:

  ① Identify the entity from the claim (e.g. "ResNet-50", "BERT-base",
     "GPT-3", "Med-PaLM 2", "MedQA benchmark").
  ② Identify the specific value claimed (e.g. "100M parameters", "86.5%",
     "deepest residual network", "12 layers").
  ③ Scan the retrieved chunks for the SAME ENTITY by name. Look in:
       • table rows (`| ResNet-50 | 25.5M | ... |`)
       • architecture descriptions (`ResNet-50 has X parameters`)
       • figure captions / spec lines
  ④ If you find the entity in a chunk AND it specifies a different value
     than the claim → supports=false, contradicted=true. QUOTE the chunk
     line in the reasoning field.
  ⑤ If the chunks discuss the topic but never name the specific entity
     at all → supports=false, contradicted=false (just unsupported).
  ⑥ If the chunks discuss DIFFERENT entities (e.g. claim says ResNet-50,
     paper only details ResNet-152) AND the claim implies a relative
     property (e.g. "ResNet-50 is the deepest"), that's still contradicted
     when the paper makes a contradicting relative statement
     ("ResNet-152, the deepest of our models, achieves...").

Worked examples — read carefully and apply the same reasoning structure:

  EXAMPLE A · numerical value mismatch
    claim:   "GPT-3 reaches 90% on MedQA"
    chunk:   "p.18 reports GPT-3 17.8% on MedQA closed-book"
    verdict: supports=false, conf=0.95, contradicted=true,
             reasoning="paper reports GPT-3 17.8% on MedQA (p.18), not 90%"

  EXAMPLE B · spec value mismatch
    claim:   "BERT-base has 220M parameters"
    chunk:   "BERTBASE (L=12, H=768, A=12, Total Parameters=110M)"
    verdict: supports=false, conf=0.95, contradicted=true,
             reasoning="paper specifies BERT-base has 110M parameters, not 220M"

  EXAMPLE C · model-version mismatch (NEW)
    claim:   "BERT-base has 24 transformer layers and 340M parameters"
    chunk:   "We have two model sizes: BERTBASE (L=12, ... 110M) and
              BERTLARGE (L=24, ... 340M)"
    verdict: supports=false, conf=0.95, contradicted=true,
             reasoning="paper reports BERT-BASE has L=12, 110M params;
                        the claim's 24/340M values describe BERT-LARGE"

  EXAMPLE D · structural property mismatch (NEW)
    claim:   "ResNet-50 has 100 million parameters and was the deepest
              residual network in the He 2016 paper"
    chunk:   "We present ResNet-50, ResNet-101 and ResNet-152
              architectures... ResNet-152 ... is 8× deeper than VGG"
    chunk:   "ResNet-50 model size: 25.5M parameters"
    verdict: supports=false, conf=0.95, contradicted=true,
             reasoning="paper reports ResNet-50 has 25.5M params (not 100M),
                        and ResNet-152 (not ResNet-50) is the deepest"

  EXAMPLE E · same-entity-not-mentioned (NOT contradiction)
    claim:   "GPT-3 achieves 86.5% on MedQA"
    chunk:   "GPT-3 evaluates on TriviaQA, LAMBADA, and HellaSwag"
    verdict: supports=false, conf=0.9, contradicted=false,
             reasoning="no chunk mentions MedQA — unsupported, not
                        contradicted (paper doesn't make a competing claim)"

KEY DISTINCTION:
  • "paper says X is Y, claim says X is Z (Y ≠ Z)"  →  contradicted=true
  • "claim says X is Y, paper never mentions X's Y at all"  →  contradicted=false
  • "claim says property P of X; paper says property P of A DIFFERENT entity"
    →  contradicted=true IF the claim implies X uniquely has P
        (e.g. "X is the deepest" but paper says "Z is the deepest")

When you set contradicted=true you MUST quote (in reasoning) the chunk
phrase that shows the conflicting value/identity. Without a verbatim
quote, leave contradicted=false.
"""


def audit_citation(
    claim_text: str,
    cited_paper_title: str,
    cited_paper_authors: str = "",
    cited_paper_year: Optional[int] = None,
    cited_paper_venue: str = "",
    *,
    abstract: Optional[str] = None,
    retrieved_chunks: Optional[list[str]] = None,
    source_resolution: str = "unknown",  # "found" | "empty" | "unknown"
    model: Optional[str] = None,
    api_key: Optional[str] = None,
) -> CitationAudit:
    """Audit whether the cited paper supports the surrounding prose claim.

    Three evidence tiers, selected automatically by which arguments are populated:

      * Tier 0 (default — neither ``abstract`` nor ``retrieved_chunks``):
        metadata-only. LLM judges by title + authors + year + venue and its own
        training-data knowledge of famous papers.
      * Tier 1 (``abstract`` provided): LLM additionally grounds its verdict in
        the paper's abstract — catches misattributions where the title is on-topic
        but the abstract reveals a different actual contribution.
      * Tier 2 (``retrieved_chunks`` provided): LLM additionally grounds its
        verdict in passages retrieved from the paper's full text via semantic
        search against the claim. Catches specific numerical / factual mismatches
        that even the abstract doesn't disambiguate.

    The prompt is assembled tier-aware: extra evidence is added to the system
    message with explicit instructions on how to weight it. The user-message
    payload carries the evidence itself.

    Conservative on failure: any LLM/API error returns ``supports=True`` with
    confidence 0 so the caller falls back to "unverifiable" rather than
    deleting a real citation.
    """
    try:
        client, model_id = _get_client_and_model(api_key, model)
    except Exception as e:
        dbg.trace("llm.audit", "ERR client unavailable", error=str(e))
        return CitationAudit(True, 0.0, f"LLM client unavailable: {e}")

    tier = 2 if retrieved_chunks else (1 if abstract else 0)
    system = _AUDIT_CITATION_SYSTEM
    if tier == 0:
        system = _AUDIT_CITATION_SYSTEM + _AUDIT_TIER0_SUFFIX
    elif tier == 1:
        system = _AUDIT_CITATION_SYSTEM + _AUDIT_TIER1_SUFFIX
    elif tier == 2:
        system = _AUDIT_CITATION_SYSTEM + _AUDIT_TIER1_SUFFIX + _AUDIT_TIER2_SUFFIX

    dbg.trace(
        "llm.audit",
        "calling",
        model=model_id,
        tier=tier,
        claim=claim_text,
        paper_title=cited_paper_title,
        paper_authors=cited_paper_authors,
        paper_year=cited_paper_year,
        has_abstract=bool(abstract),
        n_chunks=len(retrieved_chunks or []),
    )

    user_parts = [
        "Does the cited paper SUPPORT the claim? Return JSON.\n",
        f"CLAIM (academic prose containing the citation):\n  {claim_text!r}\n",
        "CITED PAPER (from the user's .bib entry):",
        f"  title:   {cited_paper_title!r}",
        f"  authors: {cited_paper_authors!r}",
        f"  year:    {cited_paper_year}",
        f"  venue:   {cited_paper_venue!r}",
        f"  source_resolution: {source_resolution}",
    ]
    if abstract:
        user_parts.append("\nABSTRACT (Tier-1 evidence):\n" + abstract.strip())
    if retrieved_chunks:
        user_parts.append("\nRETRIEVED CHUNKS from the paper's full text (Tier-2 evidence):")
        for i, chunk in enumerate(retrieved_chunks, 1):
            user_parts.append(f"\n[chunk {i}]\n{chunk.strip()}")

    user_msg = "\n".join(user_parts)

    try:
        resp = client.chat.completions.create(
            model=model_id,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_msg},
            ],
            temperature=0,
            response_format={"type": "json_object"},
        )
    except Exception as e:
        return CitationAudit(True, 0.0, f"LLM call failed: {e}")

    content = _safe_extract_content(resp)
    if not content:
        return CitationAudit(True, 0.0, "no LLM response content")
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        return CitationAudit(True, 0.0, "LLM did not return valid JSON")
    if not isinstance(data, dict):
        return CitationAudit(True, 0.0, "LLM response was not a JSON object")

    verdict = CitationAudit(
        supports=bool(data.get("supports", True)),
        confidence=float(data.get("confidence") or 0.0),
        reasoning=str(data.get("reasoning") or ""),
        contradicted=bool(data.get("contradicted", False)),
    )
    dbg.trace(
        "llm.audit",
        "verdict",
        tier=tier,
        supports=verdict.supports,
        contradicted=verdict.contradicted,
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
