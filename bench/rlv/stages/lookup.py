"""Stage 3: LOOKUP.

Given a region pointer (Stage 2 output), the original document, and a
question, run a single LLM call with ONLY that region as context. The
region must be sized below the cliff budget.

Output: a tentative answer string.
"""
from dataclasses import dataclass

from . import _llm
from .locator import RegionPointer


# Day 2 redesign: reframe lookup as EXTRACTIVE ("find and quote the
# sentence that contains the answer") rather than GENERATIVE ("answer
# the question"). The extractive framing forces the model to do span
# selection, which sidesteps primacy bias — instead of summarising the
# region (which picks the first-mentioned entity) the model has to
# identify the specific sentence that matches the question's keywords.
LOOKUP_PROMPT_TEMPLATE = """{region_text}

Quote the single sentence from the text above that answers this question. Reply with only that sentence, no explanation.

Question: {question}"""


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
