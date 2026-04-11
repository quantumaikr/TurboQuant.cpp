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


def _extract_answer_key_terms(answer: str) -> list[str]:
    """Pull out the candidate "fact tokens" from an answer that we expect
    to find in the region text. Capitalized words, multi-word names,
    numbers."""
    # Names like "John Williams" or "Maria Santos"
    multi_cap = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b", answer)
    # Single capitalized words (in case the model only said "Williams")
    single_cap = re.findall(r"\b[A-Z][a-z]{3,}\b", answer)
    # Numbers (years, amounts)
    nums = re.findall(r"\b\d{2,5}\b", answer)
    # Combine, dedupe, prefer multi-word over single
    seen = set()
    out = []
    for term in multi_cap + single_cap + nums:
        key = term.lower()
        if key not in seen:
            seen.add(key)
            out.append(term)
    return out[:8]


def _fuzzy_word_in_region(word: str, region_norm: str) -> bool:
    """Return True if a single word appears in the normalized region,
    tolerant of Q4 jitter that inserts/duplicates characters mid-word.

    Strategy: try the full word, then progressively shorter prefixes
    down to 4 chars. This handles:
      - "Williams" → "williams" (exact)
      - "Williams" → "williamlims" (model output) where the first 5 chars
        ("willi") still match the region's "williams" via prefix
      - "Williamlims" → "williams" via reverse prefix (the answer prefix
        is in the region prefix)
    """
    if not word or len(word) < 3:
        return False
    if word in region_norm:
        return True
    # Forward prefix matching: any prefix of `word` (≥4 chars) found in region
    for length in range(len(word), 3, -1):
        if length < 4:
            break
        if word[:length] in region_norm:
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


def _literal_verify(answer: str, region_text: str) -> tuple[str, str]:
    """Fast non-LLM citation check. Returns (verdict, reason)."""
    if not answer.strip() or not region_text.strip():
        return "UNSURE", "empty answer or region"

    region_norm = _normalize(region_text)
    answer_norm = _normalize(answer)
    key_terms = _extract_answer_key_terms(answer)

    if not key_terms:
        # Answer has no extractable entities — can't do literal check
        return "UNSURE", "no entity terms in answer"

    found = [t for t in key_terms if _fuzzy_in_region(t, region_norm)]
    not_found = [t for t in key_terms if t not in found]

    # Decision rule: if at least one key term is found and the found
    # ratio is at least 50%, the answer is grounded in the region.
    if len(found) >= 1 and len(found) / len(key_terms) >= 0.5:
        return "CONFIDENT", f"found {len(found)}/{len(key_terms)} key terms in region: {found[:3]}"
    elif len(found) >= 1:
        return "UNSURE", f"only {len(found)}/{len(key_terms)} key terms found, ambiguous"
    else:
        return "CONTRADICTED", f"none of {key_terms[:3]} appear in the region — likely fabricated"


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
        verdict, reason = _literal_verify(answer, region_text)
        if verdict != "UNSURE" or not use_llm_fallback:
            if verbose:
                print(f"[verifier] literal -> {verdict} ({reason})")
            return VerifyResult(verdict=verdict, reason=reason, method=method)

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
