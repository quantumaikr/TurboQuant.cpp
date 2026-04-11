"""Stage 4: VERIFY.

Cross-check a tentative answer (Stage 3 output) against the gist (Stage 1).
This is the hallucination filter — the stage that vector RAG and pure
long-context inference both lack.

Output: a verdict {confident, unsure, contradicted} and a brief reason.

The verifier asks: "Given my structural understanding of the document
(the gist), does this answer fit?" If it doesn't fit (contradicts a
known fact, mentions an entity not in the gist, etc.), the answer is
flagged for re-search or final-stage uncertainty.
"""
from dataclasses import dataclass

from . import _llm
from .gist import Gist
from .lookup import LookupResult


VERIFY_PROMPT_TEMPLATE = """Document index:
{outline}

Someone asked: "{question}"
And was told: "{answer}"

Based on the index above, is the answer plausible? Reply with one word: yes, no, or unsure."""


@dataclass
class VerifyResult:
    verdict: str   # "CONFIDENT" | "UNSURE" | "CONTRADICTED"
    reason: str
    raw: str = ""


def _parse_verify_response(text: str) -> tuple[str, str]:
    """Tolerant yes/no/unsure parser. Looks at the first 200 chars."""
    text = text.strip().lower()
    # Strip any "## Step 1:" reasoning preamble
    if "## step" in text:
        parts = [l for l in text.split("\n") if not l.strip().startswith("##")]
        text = " ".join(parts)
    head = text[:200]

    # Order matters: contradicted/no first (most specific), then yes
    if any(w in head for w in ("contradicted", "contradict", "wrong", "incorrect", "not consistent")):
        return "CONTRADICTED", head[:100]
    if "no" in head[:30] and "not" not in head[:5]:
        return "CONTRADICTED", head[:100]
    if any(w in head for w in ("unsure", "uncertain", "cannot", "don't know", "insufficient")):
        return "UNSURE", head[:100]
    if any(w in head for w in ("yes", "plausible", "consistent", "matches", "correct", "agree")):
        return "CONFIDENT", head[:100]
    return "UNSURE", head[:100]


def verify(
    question: str,
    answer: str,
    gist: Gist,
    *,
    verbose: bool = False,
) -> VerifyResult:
    """Stage 4: verify a tentative answer against the gist."""
    outline = gist.to_outline_text()
    prompt = VERIFY_PROMPT_TEMPLATE.format(
        outline=outline,
        question=question,
        answer=answer,
    )

    if verbose:
        within, est, budget = _llm.check_cliff_budget(prompt)
        print(f"[verifier] prompt ~{est} tokens (budget {budget}), within={within}")

    result = _llm.llm_call(prompt, max_tokens=24)
    verdict, reason = _parse_verify_response(result.text)

    return VerifyResult(verdict=verdict, reason=reason, raw=result.text)
