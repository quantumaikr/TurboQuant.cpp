"""Stage 4: VERIFY (citation-grounded).

Day 1 lesson: verifying against the gist alone is too lossy. The gist
summaries are generic; the verifier can't tell if a hallucinated answer
is wrong because the gist also doesn't mention the entity.

Day 2 redesign: verify by **citation**. The verifier reads the actual
lookup region text (the same chunk Stage 3 read) and checks two things:

  1. Does the answer's key entity / fact appear *literally* in the
     region text? This is a fast non-LLM check using fuzzy substring
     matching that handles Q4 visual jitter.
  2. As a fallback when the literal check is ambiguous, ask the LLM:
     "Is this answer supported by the text below?" with a yes/no/unsure
     response.

The literal check is the fastest and most reliable hallucination filter
when the model is supposed to be quoting from a specific region. If the
answer mentions "John Williams" but the region text doesn't contain
"John" or "Williams" or "Williamlims" (jitter variants), the answer is
clearly fabricated.
"""
import re
from dataclasses import dataclass

from . import _llm
from .gist import Gist
from .lookup import LookupResult
from .locator import _question_keywords, _term_in_text, _keyword_locate, _normalize as _loc_normalize


# Day 3: model-side preamble tokens that show up in answer text but
# aren't content. Filter so they don't get extracted as "key terms".
ANSWER_NOISE_TOKENS = {
    "quote", "quotte", "quoted", "quotation", "citation", "citing",
    "section", "sentence", "answer", "below", "above", "following",
    "according", "context", "based", "text", "passage", "paragraph",
}


VERIFY_LLM_PROMPT_TEMPLATE = """{region_text}

Question: {question}
Answer given: {answer}

Is the answer supported by the text above? Reply with one word: yes, no, or unsure."""


@dataclass
class VerifyResult:
    verdict: str       # "CONFIDENT" | "UNSURE" | "CONTRADICTED"
    reason: str
    raw: str = ""
    method: str = ""   # "literal" | "llm" | "literal+llm"


# ----------------------------------------------------------------------------
# Literal (regex-based) citation check
# ----------------------------------------------------------------------------
def _normalize(s: str) -> str:
    """Lowercase and strip non-alphanum-or-space. Used for fuzzy matching
    against Q4 visual jitter."""
    return re.sub(r"[^a-z0-9 ]+", " ", s.lower())


def _extract_answer_key_terms(answer: str) -> tuple[list[str], list[str]]:
    """Day 3: returns (word_terms, number_terms) so the matcher can apply
    different rules — words use fuzzy matching for Q4 jitter, numbers
    use exact match (2002 must NOT fuzzy-match 2023). Filters
    ANSWER_NOISE_TOKENS so model preambles don't get extracted as facts."""
    multi_cap = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b", answer)
    single_cap = re.findall(r"\b[A-Z][a-z]{3,}\b", answer)
    nums = re.findall(r"\b\d{2,5}\b", answer)

    seen = set()
    word_terms: list[str] = []
    for term in multi_cap + single_cap:
        key = term.lower()
        if key in seen:
            continue
        # Exact word match (not substring) — "text" must not filter "context"
        if key in ANSWER_NOISE_TOKENS:
            continue
        seen.add(key)
        word_terms.append(term)

    seen_n = set()
    number_terms: list[str] = []
    for n in nums:
        if n in seen_n:
            continue
        seen_n.add(n)
        number_terms.append(n)

    return word_terms[:8], number_terms[:4]


def _fuzzy_word_in_region(word: str, region_norm: str) -> bool:
    """Day 3: word-boundary-aware fuzzy match. Iterates region words and
    checks shared-prefix similarity. Avoids cross-word substring traps
    like "event" matching "revenue" via "even"."""
    if not word or len(word) < 3:
        return False
    w = word.lower()
    min_prefix = 4 if len(w) > 6 else 3
    for rw in region_norm.split():
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


def _fuzzy_in_region(term: str, region_norm: str) -> bool:
    """Return True if `term` (possibly multi-word) appears in the region,
    tolerant of Q4 visual jitter on individual words.

    For multi-word terms (e.g., "John Williams"), require that ≥50% of the
    words match individually via _fuzzy_word_in_region. For single words,
    require that one word matches.
    """
    term_norm = _normalize(term)
    if not term_norm:
        return False
    if term_norm in region_norm:
        return True
    words = [w for w in term_norm.split() if len(w) >= 3]
    if not words:
        return False
    matched = sum(1 for w in words if _fuzzy_word_in_region(w, region_norm))
    return matched >= max(1, len(words) // 2 + (len(words) % 2))


def _question_grounded_via_locator(
    question: str,
    chunk_id: int | None,
    gist: Gist | None,
) -> tuple[bool, float, float, str]:
    """Day 3 architectural fix: re-use the locator's keyword scoring to
    confirm the chunk being verified is a near-top match for the question.

    Citation-grounding alone catches hallucination but NOT locator errors:
    if the locator picked the wrong chunk, the model dutifully extracts
    a sentence from it, and the answer's terms trivially match the region.
    Re-running the locator scoring asks the right question: "is this
    chunk really the best match for the question?"

    Returns (is_grounded, chunk_score, best_other_score, reason).
    """
    if gist is None or chunk_id is None or not gist.chunks:
        return True, 0.0, 0.0, "no gist; trust answer-grounding"
    best_id, best_score, all_scores = _keyword_locate(question, gist, [])
    chunk_score = next((s for c, s in all_scores if c == chunk_id), 0.0)
    if best_score < 1.0:
        return True, chunk_score, best_score, "weak signal; defer to answer check"
    if chunk_score >= 0.6 * best_score and chunk_score >= 1.0:
        return True, chunk_score, best_score, f"chunk_score={chunk_score:.1f}/best={best_score:.1f}"
    return False, chunk_score, best_score, (
        f"chunk_score={chunk_score:.1f} << best={best_score:.1f}"
    )


def _literal_verify(
    question: str,
    answer: str,
    region_text: str,
    *,
    chunk_id: int | None = None,
    gist: Gist | None = None,
) -> tuple[str, str]:
    """Fast non-LLM citation check. Day 3: two checks must both pass.
      1. Question-grounded (via locator scoring) — catches locator errors
      2. Answer-grounded — catches hallucination
    """
    if not answer.strip() or not region_text.strip():
        return "UNSURE", "empty answer or region"

    region_norm = _normalize(region_text)

    q_ok, q_score, q_best, q_reason = _question_grounded_via_locator(
        question, chunk_id, gist,
    )
    if not q_ok:
        return "CONTRADICTED", f"question not grounded ({q_reason}) — likely wrong chunk"

    # Day 5: detect "I don't know" / "not provided" refusal answers.
    # These should never be CONFIDENT — the model is saying it couldn't
    # find the answer, so send it back to RESEARCH for a different chunk.
    #
    # Production hardening: only detect refusal when the phrase appears
    # in the FIRST 120 chars of the answer (not embedded in a valid
    # quoted sentence like "The study does not provide evidence for...").
    # Also require the answer to be SHORT (< 200 chars) — long answers
    # that happen to contain a refusal phrase are likely real content.
    answer_lower = answer.lower()
    answer_head = answer_lower[:120]
    refusal_phrases = [
        "does not provide", "no information", "not contain the answer",
        "cannot determine", "unable to find", "unable to determine",
        "not specified in", "not stated in", "not available in",
        "i don't know", "i'm not sure", "no relevant information",
        "the text does not", "the passage does not",
    ]
    if len(answer) < 200 and any(p in answer_head for p in refusal_phrases):
        return "UNSURE", f"answer is a refusal ('{answer[:60]}...')"

    word_terms, number_terms = _extract_answer_key_terms(answer)
    if not word_terms and not number_terms:
        return "UNSURE", f"q-grounded ({q_reason}); no extractable answer entities"

    word_found = [t for t in word_terms if _fuzzy_in_region(t, region_norm)]
    num_found = [n for n in number_terms if n in region_norm]

    total_terms = len(word_terms) + len(number_terms)
    total_found = len(word_found) + len(num_found)

    if total_found >= 1 and total_found / total_terms >= 0.5:
        return "CONFIDENT", (
            f"q-grounded ({q_reason}); "
            f"a-matched={total_found}/{total_terms} ({(word_found + num_found)[:3]})"
        )
    elif total_found >= 1:
        return "UNSURE", (
            f"q-grounded ({q_reason}); "
            f"only {total_found}/{total_terms} answer terms found"
        )
    else:
        return "CONTRADICTED", (
            f"q-grounded ({q_reason}); "
            f"none of {(word_terms + number_terms)[:3]} in region — likely fabricated"
        )


def _parse_llm_verify_response(text: str) -> tuple[str, str]:
    """Tolerant yes/no/unsure parser for the LLM fallback."""
    text = text.strip().lower()
    if "## step" in text:
        parts = [l for l in text.split("\n") if not l.strip().startswith("##")]
        text = " ".join(parts)
    head = text[:120]
    if any(w in head[:30] for w in ("yes", "supported", "consistent", "correct")):
        return "CONFIDENT", head[:80]
    if any(w in head[:30] for w in ("no,", "no.", "not supported", "incorrect", "wrong")):
        return "CONTRADICTED", head[:80]
    if any(w in head[:30] for w in ("unsure", "uncertain", "cannot")):
        return "UNSURE", head[:80]
    return "UNSURE", head[:80]


# ----------------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------------
def verify(
    question: str,
    answer: str,
    gist: Gist,
    *,
    region_text: str = "",
    chunk_id: int | None = None,
    use_llm_fallback: bool = True,
    verbose: bool = False,
) -> VerifyResult:
    """Verify a tentative answer.

    Day 2 design: prefers literal citation grounding (no LLM call) when
    the lookup region is provided. Falls back to LLM verification only
    when the literal check is ambiguous.

    The `gist` parameter is kept for API stability but is no longer the
    primary signal — citation grounding against the actual region is
    much more reliable.
    """
    method = "literal"
    if region_text:
        verdict, reason = _literal_verify(
            question, answer, region_text,
            chunk_id=chunk_id, gist=gist,
        )
        if verdict != "UNSURE" or not use_llm_fallback:
            if verbose:
                print(f"[verifier] literal -> {verdict} ({reason})")
            return VerifyResult(verdict=verdict, reason=reason, method=method)

        # Day 5: if the answer is a refusal, don't let LLM override to CONFIDENT.
        # The literal check correctly flagged it as UNSURE — trust that.
        if "refusal" in reason:
            if verbose:
                print(f"[verifier] refusal detected, skipping LLM fallback -> UNSURE")
            return VerifyResult(verdict="UNSURE", reason=reason, method="literal(refusal)")

        # Ambiguous — fall back to LLM verification on the same region
        if verbose:
            print(f"[verifier] literal=UNSURE, falling back to LLM")
        prompt = VERIFY_LLM_PROMPT_TEMPLATE.format(
            region_text=region_text,
            question=question,
            answer=answer,
        )
        result = _llm.llm_call(prompt, max_tokens=24)
        v2, r2 = _parse_llm_verify_response(result.text)
        return VerifyResult(
            verdict=v2,
            reason=f"literal:UNSURE; llm:{r2}",
            raw=result.text,
            method="literal+llm",
        )

    # No region provided — pure LLM verify against gist (legacy path)
    if verbose:
        print(f"[verifier] no region_text, falling back to gist-only LLM verify")
    outline = gist.to_outline_text() if gist else ""
    prompt = VERIFY_LLM_PROMPT_TEMPLATE.format(
        region_text=outline,
        question=question,
        answer=answer,
    )
    result = _llm.llm_call(prompt, max_tokens=24)
    verdict, reason = _parse_llm_verify_response(result.text)
    return VerifyResult(verdict=verdict, reason=reason, raw=result.text, method="llm")
