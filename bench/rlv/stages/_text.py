"""Shared text matching utilities for RLV stages.

Extracted from locator.py and verifier.py to eliminate code duplication
(audit issues I2, I3). All fuzzy matching, normalization, and keyword
extraction functions live here.
"""
import re
from typing import List, Tuple


# ============================================================
# Text normalization
# ============================================================
def normalize(s: str) -> str:
    """Lowercase and strip non-alphanum-or-space. Used for fuzzy matching
    against Q4 visual jitter."""
    return re.sub(r"[^a-z0-9 ]+", " ", s.lower())


# ============================================================
# Fuzzy word matching (Q4 jitter tolerant)
# ============================================================
def word_in_text(word: str, text_norm: str) -> bool:
    """Word-boundary-aware fuzzy match. Tolerates Q4 KV jitter by
    matching shared prefixes between the query word and each word in
    the normalized text.

    A `word` matches a region word `rw` if:
      - exact: rw == word
      - shared prefix: >= min_prefix chars, with shared length at least
        min(len(w), len(rw)) - 2
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


def term_in_text(term: str, text_norm: str) -> bool:
    """Multi-word term match: >= 50% of the words must fuzzy-match.
    Whole-phrase substring is allowed as a fast path for multi-word terms."""
    t = normalize(term)
    if not t:
        return False
    if " " in t and t in text_norm:
        return True
    words = [w for w in t.split() if len(w) >= 3]
    if not words:
        return False
    matched = sum(1 for w in words if word_in_text(w, text_norm))
    return matched >= max(1, (len(words) + 1) // 2)


def fuzzy_in_region(term: str, region_norm: str) -> bool:
    """Return True if `term` (possibly multi-word) appears in the region,
    tolerant of Q4 visual jitter on individual words."""
    return term_in_text(term, region_norm)


# ============================================================
# Stopwords & low-signal terms
# ============================================================
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

LOW_SIGNAL_TERMS = {
    "company", "year", "section", "report", "annual", "fiscal",
}
