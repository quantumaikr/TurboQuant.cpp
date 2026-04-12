"""Stage 2: LOCATOR.

Day 3 design: pure non-LLM keyword-overlap scoring as the primary signal,
with an LLM fallback only when the top-2 chunks are too close to call.

Why non-LLM works here:
  - The gist's full_text is the actual document text (no model rewrite),
    so question keywords like "CFO" / "revenue" / "Maria Santos" can be
    matched literally.
  - Multi-word capitalised phrases ("Maria Santos") are very high-signal.
  - Numbers ("2025", "847") are similarly high-signal.
  - Section-title position weighting: words in the first ~60 chars of a
    chunk get a 2x bonus, because section titles are highly distinctive
    (e.g., "Risk Factors" matches the question keyword "risk factor").
  - It's deterministic, parser-free, and ~1000x faster than an LLM call.

When the LLM fallback fires:
  - The top-2 chunks score within 0.5 of each other (genuinely ambiguous)
  - The fallback presents candidates as 1-indexed *choice numbers* (NOT
    chunk ids) so the parser doesn't get confused by document-internal
    integers in the model's reply.
"""
import re
from dataclasses import dataclass, field
from typing import List, Tuple

from . import _llm
from .gist import Gist


# Common English stopwords + interrogatives + low-information question fillers.
STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "of", "in", "on", "at", "to", "for", "from", "by", "with", "as",
    "and", "or", "but", "if", "then", "than", "that", "this", "these",
    "those", "what", "which", "who", "whom", "whose", "where", "when",
    "why", "how", "do", "does", "did", "done", "doing", "have", "has",
    "had", "having", "i", "you", "he", "she", "it", "we", "they",
    "me", "him", "her", "us", "them", "my", "your", "his", "its", "our",
    "their", "about", "into", "through", "during", "before", "after",
    "above", "below", "between", "out", "off", "over", "under",
    "again", "further", "once", "here", "there", "all", "any", "both",
    "each", "few", "more", "most", "other", "some", "such", "no", "nor",
    "not", "only", "own", "same", "so", "too", "very", "can", "will",
    "just", "would", "should", "could", "may", "might", "must",
    "much", "many", "long", "ago", "later", "well", "thing",
    "something", "anything", "nothing", "everything", "people",
    "person", "anyone", "someone",
}

# Common business/document filler that adds noise to the score
LOW_SIGNAL_TERMS = {
    "company", "year", "section", "report", "annual", "fiscal",
}

# Section title region weighting — words appearing in the first
# SECTION_TITLE_CHARS of a chunk get this multiplier (matches in headers
# are far more discriminating than matches in body text).
SECTION_TITLE_CHARS = 60
SECTION_TITLE_BONUS = 2.0


# LLM fallback prompt — only fires when keyword scoring is ambiguous.
# Day 3 design: present candidates as 1-indexed *choice* numbers (decoupled
# from chunk ids) so the parser never accidentally picks up "Section 3"
# from the model's reply as if it were a chunk id.
LOCATOR_LLM_PROMPT_TEMPLATE = """{outline}

Question: {question}

Which choice contains the answer? Look at the topic of each choice. Reply with one digit only: the choice number."""


@dataclass
class RegionPointer:
    chunk_id: int
    confidence: str  # "high" | "medium" | "low"
    candidates: List[int] = field(default_factory=list)
    char_start: int = 0
    char_end: int = 0
    score: float = 0.0
    method: str = ""  # "keyword" | "keyword+llm" | "llm" | "fallback"

    def to_dict(self):
        return {
            "chunk_id": self.chunk_id,
            "confidence": self.confidence,
            "candidates": self.candidates,
            "char_start": self.char_start,
            "char_end": self.char_end,
            "score": self.score,
            "method": self.method,
        }


# ----------------------------------------------------------------------------
# Non-LLM keyword overlap (primary signal)
# ----------------------------------------------------------------------------
def _normalize(s: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", " ", s.lower())


def _question_keywords(question: str) -> List[Tuple[str, float]]:
    """Extract weighted (term, weight) tuples from a question."""
    terms: List[Tuple[str, float]] = []
    seen = set()

    def add(term: str, weight: float) -> None:
        key = term.lower()
        if not key or key in seen:
            return
        if key in STOPWORDS or key in LOW_SIGNAL_TERMS:
            return
        seen.add(key)
        terms.append((term, weight))

    for m in re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b", question):
        add(m, 3.0)
    for m in re.findall(r"\b[A-Z]{2,5}\b", question):
        add(m, 3.0)
    for m in re.findall(r"\b[A-Z][a-z]{2,}\b", question):
        add(m, 1.5)
    for m in re.findall(r"\b\d{2,5}\b", question):
        add(m, 2.0)
    q_norm = _normalize(question)
    for w in q_norm.split():
        if len(w) >= 4 and w not in STOPWORDS and w not in LOW_SIGNAL_TERMS:
            add(w, 1.0)
    return terms


def _word_in_text(word: str, text_norm: str) -> bool:
    """Day 3 word-boundary-aware fuzzy match.

    A `word` matches a region word `rw` if:
      - exact: rw == word
      - shared prefix: ≥4 chars (≥3 for short ≤6-char words), with the
        shared prefix at least min(len(w), len(rw)) - 2.
    Word-by-word matching avoids the substring trap (e.g., "event" in
    "revenue" via "even").
    """
    if not word or len(word) < 3:
        return False
    w = word.lower()
    min_prefix = 4 if len(w) > 6 else 3
    for rw in text_norm.split():
        if not rw:
            continue
        if rw == w:
            return True
        shared = 0
        for a, b in zip(w, rw):
            if a == b:
                shared += 1
            else:
                break
        if shared >= min_prefix and shared >= min(len(w), len(rw)) - 2:
            return True
    return False


def _term_in_text(term: str, text_norm: str) -> bool:
    """Multi-word term match: ≥50% of the words must fuzzy-match.
    Whole-phrase substring is allowed as a fast path for multi-word terms."""
    t = _normalize(term)
    if not t:
        return False
    if " " in t and t in text_norm:
        return True
    words = [w for w in t.split() if len(w) >= 3]
    if not words:
        return False
    matched = sum(1 for w in words if _word_in_text(w, text_norm))
    return matched >= max(1, (len(words) + 1) // 2)


_HEADING_RE = re.compile(r"^(?:section|chapter|part|appendix)\s*[ivxlcdm\d]+\s*[:.\-]", re.IGNORECASE)


def _looks_like_heading(text: str) -> bool:
    """Day 4: detect whether the chunk's first line looks like a section
    heading. Only when this returns True does the SECTION_TITLE_BONUS
    apply — for continuous narrative wikitext (no headers), the bonus
    misled the locator on Day 4 iteration 1 because it boosted whatever
    keywords happened to land in the first 60 chars of an arbitrary
    char-based chunk."""
    if not text:
        return False
    first_line = text.split("\n", 1)[0].strip()
    if not first_line:
        return False
    if _HEADING_RE.match(first_line):
        return True
    # Short first line followed by a colon, e.g. "Risk Factors:" or "Topic: Foo"
    if 4 <= len(first_line) <= 50 and ":" in first_line[:50]:
        prefix = first_line.split(":", 1)[0].strip()
        if prefix and prefix[0].isupper():
            return True
    return False


def _score_chunk(weighted_terms: List[Tuple[str, float]], chunk) -> float:
    """Score a chunk against weighted question terms.

    Day 3: match against full_text and apply a SECTION_TITLE_BONUS to
    terms appearing in the first 60 chars (the section title region).

    Day 4: the title bonus is now CONDITIONAL on the chunk's first line
    actually looking like a heading. Continuous narrative docs without
    section headers don't get the bonus — every chunk just starts
    mid-paragraph, and an unconditional bonus boosts incidental words.
    """
    text = chunk.full_text or chunk.head_text
    text_norm = _normalize(text)
    has_heading = _looks_like_heading(text)
    title_norm = _normalize(text[:SECTION_TITLE_CHARS]) if (text and has_heading) else ""
    entities_norm = _normalize(" ".join(chunk.entities)) if chunk.entities else ""
    score = 0.0
    for term, weight in weighted_terms:
        if _term_in_text(term, text_norm):
            if title_norm and _term_in_text(term, title_norm):
                score += weight * SECTION_TITLE_BONUS
            else:
                score += weight
        elif entities_norm and _term_in_text(term, entities_norm):
            score += weight * 0.5
    return score


def _keyword_locate(
    question: str,
    gist: Gist,
    excluded: List[int],
) -> Tuple[int, float, List[Tuple[int, float]]]:
    """Score every (non-excluded) chunk by keyword overlap."""
    weighted = _question_keywords(question)
    scores: List[Tuple[int, float]] = []
    for chunk in gist.chunks:
        if chunk.chunk_id in excluded:
            continue
        s = _score_chunk(weighted, chunk)
        scores.append((chunk.chunk_id, s))
    scores.sort(key=lambda x: x[1], reverse=True)
    if not scores:
        return -1, 0.0, []
    return scores[0][0], scores[0][1], scores


# ----------------------------------------------------------------------------
# BM25 scoring (Strategy 1: TF-IDF weighted term matching)
# ----------------------------------------------------------------------------
import math

def _bm25_score_chunks(question: str, gist: Gist, excluded: List[int],
                        k1: float = 1.5, b: float = 0.75) -> List[Tuple[int, float]]:
    """BM25 scoring: TF-IDF weighted keyword matching.
    Unlike simple keyword overlap, BM25 penalizes common terms and
    rewards rare terms — 'Mercury Fur' gets a huge boost because it
    appears in very few chunks, while 'the' gets zero."""
    q_terms = [w for w in _normalize(question).split()
               if len(w) >= 3 and w not in STOPWORDS and w not in LOW_SIGNAL_TERMS]
    if not q_terms:
        return [(c.chunk_id, 0.0) for c in gist.chunks if c.chunk_id not in excluded]

    chunks = [c for c in gist.chunks if c.chunk_id not in excluded]
    N = len(chunks)
    if N == 0:
        return []

    # Document frequency for each term
    texts = [_normalize(c.full_text or c.head_text) for c in chunks]
    avg_dl = sum(len(t.split()) for t in texts) / max(N, 1)

    df = {}
    for term in q_terms:
        df[term] = sum(1 for t in texts if _word_in_text(term, t))

    scores = []
    for i, chunk in enumerate(chunks):
        doc_words = texts[i].split()
        dl = len(doc_words)
        score = 0.0
        for term in q_terms:
            tf = sum(1 for w in doc_words if w == term or
                     (len(w) >= 3 and len(term) >= 3 and
                      w[:min(4, len(w))] == term[:min(4, len(term))]))
            n = df.get(term, 0)
            # Standard BM25 IDF: log((N-n+0.5)/(n+0.5)+1). Terms appearing
            # in ALL chunks (n==N) get idf=0 (no discriminating power).
            idf = max(0.0, math.log((N - n + 0.5) / (n + 0.5) + 1.0)) if n < N else 0.0
            denom = tf + k1 * (1 - b + b * dl / max(avg_dl, 1))
            tf_norm = (tf * (k1 + 1)) / max(denom, 1e-9)
            score += idf * tf_norm
        scores.append((chunk.chunk_id, score))

    scores.sort(key=lambda x: x[1], reverse=True)
    return scores


# ----------------------------------------------------------------------------
# LLM classification (Strategy 2: always-on, not just fallback)
# ----------------------------------------------------------------------------
_NUM_RE = re.compile(r"\b(\d+)\b")


def _parse_locator_response(text: str, n_max: int) -> int:
    """Find the first integer in [0, n_max) in the response. Returns -1
    on parse failure. Strips '## Step 1:' reasoning preambles."""
    text = text.strip()
    if "## Step" in text:
        parts = [l for l in text.split("\n") if not l.strip().startswith("##")]
        text = " ".join(parts)
    for m in _NUM_RE.finditer(text):
        n = int(m.group(1))
        if 0 <= n < n_max:
            return n
    return -1


def _llm_locate(
    question: str,
    gist: Gist,
    excluded: List[int],
    candidate_ids: List[int],
    verbose: bool = False,
) -> int:
    """Ask the LLM to classify which chunk contains the answer.
    Day 5: always-on LLM classification (not just fallback).
    Shows first 2 sentences per chunk for better context."""
    available = [cid for cid in candidate_ids if cid not in excluded]
    if not available:
        return -1

    lines = []
    for choice_num, cid in enumerate(available, start=1):
        if cid >= len(gist.chunks):
            continue  # skip invalid chunk_id
        chunk = gist.chunks[cid]
        text = (chunk.full_text or chunk.head_text).replace("\n", " ").strip()
        # Show first 2 sentences (more context than just head)
        sents = re.split(r'(?<=[.!?])\s+', text)
        preview = " ".join(sents[:2])
        if len(preview) > 250:
            preview = preview[:250] + "..."
        lines.append(f"[{choice_num}] {preview}")
    outline = "\n".join(lines)
    prompt = LOCATOR_LLM_PROMPT_TEMPLATE.format(outline=outline, question=question)
    result = _llm.llm_call(prompt, max_tokens=8)
    if verbose:
        print(f"[locator-llm] response: {result.text!r}")
    # Parser accepts [0, n_max). Choices are 1-indexed, so n_max = N+1.
    # Post-filter: reject 0 (not a valid choice) and > N (out of bounds).
    choice = _parse_locator_response(result.text, len(available) + 1)
    if choice < 1 or choice > len(available):
        return -1  # parse failure or out-of-range → caller falls back to keyword winner
    return available[choice - 1]


# ----------------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------------
def locate(
    question: str,
    gist: Gist,
    *,
    excluded_chunks: list[int] = None,
    verbose: bool = False,
) -> RegionPointer:
    """Locate the chunk most likely to contain the answer.

    Day 3 design: non-LLM keyword overlap is the primary signal. The LLM
    fallback fires only when the top scores are ambiguous.
    """
    excluded = list(excluded_chunks or [])

    available = [c for c in gist.chunks if c.chunk_id not in excluded]
    if not available:
        if not gist.chunks:
            # Empty document — return a dummy pointer
            return RegionPointer(
                chunk_id=0, confidence="low", method="fallback",
                char_start=0, char_end=0, score=0.0,
            )
        chunk = gist.chunks[0]
        return RegionPointer(
            chunk_id=0, confidence="low", method="fallback",
            char_start=chunk.char_start, char_end=chunk.char_end, score=0.0,
        )
    if len(available) == 1:
        chunk = available[0]
        return RegionPointer(
            chunk_id=chunk.chunk_id, confidence="high", method="keyword",
            char_start=chunk.char_start, char_end=chunk.char_end, score=0.0,
        )

    # --- Step 1: Keyword scoring ---
    best_id, best_score, kw_scores = _keyword_locate(question, gist, excluded)

    # --- Step 2: BM25 scoring ---
    bm25_scores = _bm25_score_chunks(question, gist, excluded)

    if verbose:
        print(f"[locator] keyword top3: {kw_scores[:3]}")
        print(f"[locator] bm25   top3: {bm25_scores[:3]}")

    # --- Step 3: Reciprocal Rank Fusion (keyword + BM25) ---
    # RRF k parameter: controls how much the top ranks dominate.
    # Standard k=60 works for <100 chunks. For very large documents
    # (400+ chunks), increase k to preserve ranking discrimination.
    n_chunks = len(kw_scores)
    rrf_k = 60 if n_chunks < 100 else min(n_chunks, 200)
    rrf = {}
    for rank, (cid, _) in enumerate(kw_scores):
        rrf[cid] = rrf.get(cid, 0) + 1.0 / (rrf_k + rank)
    for rank, (cid, _) in enumerate(bm25_scores):
        rrf[cid] = rrf.get(cid, 0) + 1.0 / (rrf_k + rank)
    # Sort by (score DESC, chunk_id ASC) for deterministic tie-breaking
    rrf_ranked = sorted(rrf.items(), key=lambda x: (-x[1], x[0]))

    if verbose:
        print(f"[locator] rrf    top3: {rrf_ranked[:3]}")

    # Guard: if no chunks survived scoring, return first available
    if not rrf_ranked:
        chunk = available[0]
        return RegionPointer(
            chunk_id=chunk.chunk_id, confidence="low", method="fallback",
            candidates=[], char_start=chunk.char_start, char_end=chunk.char_end,
            score=0.0,
        )

    # --- Step 4: LLM classification on top candidates ---
    # Always run LLM on the top 5 RRF candidates (not just when ambiguous)
    top_candidates = [cid for cid, _ in rrf_ranked[:5]]
    llm_choice = _llm_locate(question, gist, excluded, top_candidates, verbose=verbose)

    rrf_top1 = rrf_ranked[0][0]
    rrf_top1_score = rrf_ranked[0][1]
    rrf_top2_score = rrf_ranked[1][1] if len(rrf_ranked) > 1 else 0.0
    rrf_margin = (rrf_top1_score - rrf_top2_score) / max(rrf_top1_score, 0.001)

    # Day 5: always trust RRF. LLM classification on small models is
    # unreliable — it consistently picks the wrong chunk. BM25+keyword
    # RRF is deterministic and more accurate for entity lookup queries.
    # LLM is only used as a tiebreaker when RRF margin is essentially zero.
    chosen = rrf_top1
    method = "rrf"
    confidence = "high" if rrf_margin > 0.05 else "medium"

    if llm_choice >= 0 and llm_choice == rrf_top1:
        method = "rrf+llm"
        confidence = "high"
    elif llm_choice >= 0 and rrf_margin < 0.005:
        # Dead tie — let LLM break it
        chosen = llm_choice
        method = "rrf+llm-tiebreak"
        confidence = "medium"

    if verbose:
        print(f"[locator] chosen: chunk {chosen} via {method} (confidence={confidence})")

    chunk = gist.chunks[chosen]
    return RegionPointer(
        chunk_id=chosen,
        confidence=confidence,
        candidates=[cid for cid, _ in rrf_ranked[:3]],
        char_start=chunk.char_start,
        char_end=chunk.char_end,
        score=rrf.get(chosen, 0.0),
        method=method,
    )
