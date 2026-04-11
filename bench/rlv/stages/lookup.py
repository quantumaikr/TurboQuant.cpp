"""Stage 3: LOOKUP.

Given a region pointer (Stage 2 output), the original document, and a
question, run a single LLM call with ONLY that region as context. The
region must be sized below the cliff budget.

Output: a tentative answer string.
"""
from dataclasses import dataclass

from . import _llm
from .locator import RegionPointer


# EMPIRICAL: this exact format (doc + blank line + question) is what
# worked in the Phase 3 Day 1 isolation test against Llama-3.2-3B-Q4.
# Adding any wrap like "Based ONLY on the text above..." breaks the
# model and causes it to fall back to primacy-bias entity selection.
# Keep this prompt minimal — every word matters.
LOOKUP_PROMPT_TEMPLATE = """{region_text}

{question}"""


@dataclass
class LookupResult:
    answer: str
    region_text: str
    chunk_id: int
    raw_llm_output: str = ""


def lookup(
    question: str,
    region: RegionPointer,
    doc_text: str,
    *,
    verbose: bool = False,
) -> LookupResult:
    """Stage 3: read the targeted region and answer the question."""
    region_text = doc_text[region.char_start:region.char_end]

    prompt = LOOKUP_PROMPT_TEMPLATE.format(
        region_text=region_text,
        question=question,
    )

    if verbose:
        within, est, budget = _llm.check_cliff_budget(prompt)
        print(f"[lookup] chunk {region.chunk_id} ({len(region_text)} chars), "
              f"prompt ~{est} tokens (budget {budget}), within={within}")

    result = _llm.llm_call(prompt, max_tokens=64)

    return LookupResult(
        answer=result.text.strip(),
        region_text=region_text,
        chunk_id=region.chunk_id,
        raw_llm_output=result.text,
    )
