"""Evidence-span extraction — turn 800-word RAG chunks into 1-3-sentence quotes.

The audit pipeline's RAG retrieval returns chunks sized for the LLM judge
(~800 words). That's the right shape for grounding the verdict but too long
for UI rendering — a side-panel issue card needs a 1-3 sentence quote the
user can read in seconds.

This module does the chunk → span compression heuristically (no LLM call —
runs on every audit, must stay fast):

  1. Tokenise the claim into salient terms (lowercase + drop stopwords).
  2. Split the chunk into sentences (pure-Python regex, no NLTK).
  3. Score each sentence by overlap with the claim's salient terms PLUS
     a small bonus for sentences that contain numbers / named-benchmarks
     (often the most informative for quantitative claims).
  4. Pick the top 1-3 contiguous sentences capped at ~300 characters total.

Returns ``EvidenceSpan`` objects with the quote, the chunk's page number,
the chunk index it came from, and an evidence-type label (supporting /
contradicting / missing). The label is derived from the audit verdict:
when a verdict is ``contradicted``, the surfaced spans are the ones
containing the conflicting value; otherwise they're the supporting
evidence the LLM grounded its decision in.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Optional

from . import dbg


@dataclass
class EvidenceSpan:
    """One short quote with attribution, ready to render in a UI card."""

    type: str           # "supporting" | "contradicting" | "missing" | "neutral"
    paper_key: str = ""
    paper_title: str = ""
    section: str = ""   # best-effort, often empty
    page: Optional[int] = None
    chunk_idx: Optional[int] = None
    chunk_score: float = 0.0
    quote: str = ""     # the actual 1-3 sentence text

    def to_dict(self) -> dict:
        d = asdict(self)
        # Drop None for cleaner JSON output.
        return {k: v for k, v in d.items() if v not in (None, "")}


# ── tokenisation + scoring ──────────────────────────────────────────────────


# Small academic-English stopword list. Intentionally tight; the goal is to
# drop "the / and / of / a" without losing rare-but-content-bearing words.
_STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "at", "for",
    "with", "by", "from", "as", "is", "are", "was", "were", "be", "been",
    "this", "that", "these", "those", "it", "its", "their", "his", "her",
    "we", "our", "they", "them", "but", "not", "no", "yes", "do", "does",
    "did", "have", "has", "had", "can", "could", "may", "might", "will",
    "would", "should", "than", "then", "so", "if", "while", "also",
}


_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9\-]*|\d+(?:\.\d+)?%?")


def _salient_terms(text: str) -> set[str]:
    """Lowercase tokens with stopwords removed. Numbers + percentages
    pass through (they're discriminating signals for quantitative claims)."""
    if not text:
        return set()
    out: set[str] = set()
    for tok in _WORD_RE.findall(text):
        low = tok.lower()
        if low in _STOPWORDS:
            continue
        if len(low) < 2 and not low.isdigit():
            continue
        out.add(low)
    return out


# Number / benchmark-name regex — sentence-level bonus.
_INFORMATIVE_RE = re.compile(
    r"\d+(?:\.\d+)?%|\d+\s*(?:M|B|K)?\s*(?:parameters|params|layers|epochs)"
    r"|F1|BLEU|ROUGE|MedQA|MMLU|GLUE|SuperGLUE|ImageNet|LibriSpeech|SQuAD",
    re.IGNORECASE,
)


# Sentence splitter — same shape as audit._extract_claim's. Good enough for
# academic prose; table rows split on newlines (we treat each table row as
# its own "sentence").
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z\[])|(?:\n+)")


def _split_sentences(chunk_text: str) -> list[str]:
    """Coarse sentence splitter. Drops leading-page markers (e.g. "[p.3]")."""
    # Strip the chunk's leading "[p.N] " marker so we don't surface it as
    # part of the quote.
    text = re.sub(r"^\s*\[p\.[^\]]+\]\s*", "", chunk_text)
    text = re.sub(r"^\s*\[Table[^\]]+\]\s*", "", text)
    parts = _SENT_SPLIT_RE.split(text.strip())
    return [p.strip() for p in parts if p.strip()]


def _score_sentence(sentence: str, claim_terms: set[str]) -> float:
    """Overlap-based relevance score with bonus for informative tokens."""
    sent_terms = _salient_terms(sentence)
    if not sent_terms or not claim_terms:
        return 0.0
    overlap = len(sent_terms & claim_terms)
    # Normalise by claim term count so longer claims aren't biased.
    base = overlap / max(len(claim_terms), 1)
    # Bonus when the sentence carries quantitative / named-benchmark content.
    if _INFORMATIVE_RE.search(sentence):
        base += 0.25
    return base


# ── public API ──────────────────────────────────────────────────────────────


def extract_evidence_span(
    chunk_text: str,
    claim_text: str,
    *,
    max_chars: int = 300,
    max_sentences: int = 3,
) -> str:
    """Return the most-relevant 1-3 contiguous sentences from ``chunk_text``.

    The pure-Python heuristic:

      1. Split chunk into sentences.
      2. Score each by claim-term overlap + informative-token bonus.
      3. Pick the highest-scoring sentence, then extend with adjacent
         sentences (in either direction) as long as the total stays
         within ``max_chars`` and ``max_sentences``.

    Falls back to the first ``max_chars`` characters of the chunk if
    no sentence scored above zero (claim shares no terms with chunk).
    """
    if not chunk_text:
        return ""
    sentences = _split_sentences(chunk_text)
    if not sentences:
        return chunk_text[:max_chars]
    claim_terms = _salient_terms(claim_text)
    if not claim_terms:
        # No discriminating signal — return first sentence.
        return sentences[0][:max_chars]

    scores = [_score_sentence(s, claim_terms) for s in sentences]
    best_idx = max(range(len(scores)), key=lambda i: scores[i])

    if scores[best_idx] == 0.0:
        # No overlap anywhere. Return a small head of the chunk.
        return sentences[0][:max_chars]

    # Greedy contiguous expansion around the best sentence.
    selected = [best_idx]
    used_chars = len(sentences[best_idx])
    left, right = best_idx - 1, best_idx + 1
    while len(selected) < max_sentences and used_chars < max_chars:
        # Pick whichever neighbour (left or right) has the higher score.
        left_score = scores[left] if left >= 0 else -1
        right_score = scores[right] if right < len(sentences) else -1
        if left_score < 0 and right_score < 0:
            break
        if right_score >= left_score and right < len(sentences):
            extra = sentences[right]
            if used_chars + len(extra) + 1 > max_chars:
                break
            selected.append(right)
            used_chars += len(extra) + 1
            right += 1
        elif left >= 0:
            extra = sentences[left]
            if used_chars + len(extra) + 1 > max_chars:
                break
            selected.insert(0, left)
            used_chars += len(extra) + 1
            left -= 1
        else:
            break

    selected.sort()
    quote = " ".join(sentences[i] for i in selected).strip()
    # Hard cap for safety even when sentences are huge.
    if len(quote) > max_chars + 50:
        quote = quote[:max_chars] + "…"
    return quote


def build_evidence_spans(
    retrieved_chunks: list,
    claim_text: str,
    *,
    paper_key: str = "",
    paper_title: str = "",
    chunk_scores: Optional[list[float]] = None,
    evidence_type: str = "supporting",
    max_spans: int = 5,
) -> list[EvidenceSpan]:
    """Build a list of EvidenceSpan from RAG-retrieved chunks.

    ``retrieved_chunks`` is the list of ``Chunk`` objects from
    ``EmbeddingStore.retrieve``. We collapse each to a quote span and
    return them in retrieval-order. ``chunk_scores`` (optional) is the
    parallel list of cosine scores so the UI can sort/filter.
    """
    spans: list[EvidenceSpan] = []
    for i, c in enumerate(retrieved_chunks[:max_spans]):
        # Chunks come in two shapes: a dataclass with .text / .page /
        # .chunk_idx (the in-process form) OR a raw "[p.N] ..." string
        # (the form passed to the LLM). Handle both.
        if hasattr(c, "text"):
            text = c.text
            page = getattr(c, "page", None)
            chunk_idx = getattr(c, "chunk_idx", None)
        else:
            # Raw string — strip the page-marker prefix when present.
            text = str(c)
            page = None
            chunk_idx = i
            m = re.match(r"\[p\.(\d+)\]\s*(.*)$", text, re.DOTALL)
            if m:
                try:
                    page = int(m.group(1))
                except ValueError:
                    page = None
                text = m.group(2)

        quote = extract_evidence_span(text, claim_text)
        if not quote:
            continue
        spans.append(EvidenceSpan(
            type=evidence_type,
            paper_key=paper_key,
            paper_title=paper_title,
            page=page,
            chunk_idx=chunk_idx,
            chunk_score=float(chunk_scores[i]) if chunk_scores and i < len(chunk_scores) else 0.0,
            quote=quote,
        ))
    dbg.trace(
        "audit.evidence",
        "extracted spans",
        n=len(spans),
        paper_key=paper_key,
        type=evidence_type,
    )
    return spans
