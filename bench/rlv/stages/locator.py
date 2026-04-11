"""Stage 2: LOCATOR.

Given a question and a gist (Stage 1 output), return a region pointer
indicating where in the document the answer is most likely to be found.

The locator runs entirely on the gist's text outline (~500-2000 tokens),
so it always fits well below the cliff budget. This is what makes the
stage reliable.

Output: a RegionPointer with a chunk_id (or list of candidates) and a
confidence score in [0.0, 1.0].
"""
import re
from dataclasses import dataclass, field
from typing import List

from . import _llm
from .gist import Gist


LOCATOR_PROMPT_TEMPLATE = """Below is a list of sections from a document, each with a number and a short summary.

{outline}

Which section number is most likely to contain the answer to this question: {question}

Reply with just the section number."""


@dataclass
class RegionPointer:
    chunk_id: int
    confidence: str  # "high" | "medium" | "low"
    candidates: List[int] = field(default_factory=list)  # alternative chunks if needed
    char_start: int = 0
    char_end: int = 0

    def to_dict(self):
        return {
            "chunk_id": self.chunk_id,
            "confidence": self.confidence,
            "candidates": self.candidates,
            "char_start": self.char_start,
            "char_end": self.char_end,
        }


_NUM_RE = re.compile(r"\b(\d+)\b")


def _parse_locator_response(text: str, n_chunks: int) -> tuple[int, str]:
    """Find the first integer in the response that's a valid chunk id.
    Returns (chunk_id, confidence). If no valid id found, returns (-1, 'low')."""
    text = text.strip()
    # Strip any "## Step 1:" reasoning preamble
    if "## Step" in text:
        parts = [l for l in text.split("\n") if not l.strip().startswith("##")]
        text = " ".join(parts)
    # Find all integers in order; pick the first one that's a valid chunk id
    for m in _NUM_RE.finditer(text):
        n = int(m.group(1))
        if 0 <= n < n_chunks:
            return n, "high" if len(text) < 80 else "medium"
    return -1, "low"


def locate(
    question: str,
    gist: Gist,
    *,
    excluded_chunks: list[int] = None,
    verbose: bool = False,
) -> RegionPointer:
    """Run the locator stage. Optionally exclude chunks already tried (for re-search)."""
    excluded_chunks = excluded_chunks or []

    outline = gist.to_outline_text()
    if excluded_chunks:
        outline += f"\n\n(Note: chunks {excluded_chunks} were already checked and did not contain the answer. Pick a different one.)"

    prompt = LOCATOR_PROMPT_TEMPLATE.format(outline=outline, question=question)

    if verbose:
        from . import _llm
        within, est, budget = _llm.check_cliff_budget(prompt)
        print(f"[locator] prompt ~{est} tokens (budget {budget}), within={within}")

    result = _llm.llm_call(prompt, max_tokens=24)
    chunk_id, confidence = _parse_locator_response(result.text, len(gist.chunks))

    # If parse failed, fall back to first non-excluded chunk and mark low confidence
    if chunk_id < 0:
        if verbose:
            print(f"[locator] parse failed, raw text: {result.text!r}")
        for cid in range(len(gist.chunks)):
            if cid not in excluded_chunks:
                chunk_id = cid
                break
        else:
            chunk_id = 0
        confidence = "low"

    chunk = gist.chunks[chunk_id]
    return RegionPointer(
        chunk_id=chunk_id,
        confidence=confidence,
        candidates=[],
        char_start=chunk.char_start,
        char_end=chunk.char_end,
    )
