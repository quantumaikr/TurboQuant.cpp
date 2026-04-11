"""Stage 5: RE-SEARCH.

If Stage 4 verification fails (verdict != CONFIDENT), the researcher
asks the locator to try a different region. Capped at MAX_RETRIES
attempts. After that, return calibrated uncertainty.
"""
from dataclasses import dataclass
from typing import List

from . import locator, lookup, verifier
from .gist import Gist
from .lookup import LookupResult
from .verifier import VerifyResult


MAX_RETRIES = 3


@dataclass
class ResearchResult:
    final_answer: str
    final_verdict: str    # CONFIDENT | UNSURE | CONTRADICTED | EXHAUSTED
    final_chunk: int
    attempts: List[dict]  # one per attempt: {chunk, answer, verdict, reason}
    n_retries: int


def research(
    question: str,
    initial_lookup: LookupResult,
    initial_verify: VerifyResult,
    gist: Gist,
    doc_text: str,
    *,
    max_retries: int = MAX_RETRIES,
    verbose: bool = False,
) -> ResearchResult:
    """If verification didn't pass, retry with different regions."""
    attempts = [{
        "chunk": initial_lookup.chunk_id,
        "answer": initial_lookup.answer,
        "verdict": initial_verify.verdict,
        "reason": initial_verify.reason,
    }]

    if initial_verify.verdict == "CONFIDENT":
        return ResearchResult(
            final_answer=initial_lookup.answer,
            final_verdict="CONFIDENT",
            final_chunk=initial_lookup.chunk_id,
            attempts=attempts,
            n_retries=0,
        )

    excluded = [initial_lookup.chunk_id]
    for retry in range(max_retries):
        if verbose:
            print(f"[researcher] retry {retry+1}/{max_retries}, excluding chunks {excluded}")

        new_region = locator.locate(question, gist, excluded_chunks=excluded, verbose=verbose)
        # If locator picked a chunk we already excluded (parser failure or only-one-chunk doc), bail
        if new_region.chunk_id in excluded:
            if verbose:
                print(f"[researcher] locator returned excluded chunk {new_region.chunk_id}, stopping")
            break

        new_lookup = lookup.lookup(question, new_region, doc_text, verbose=verbose)
        new_verify = verifier.verify(question, new_lookup.answer, gist, verbose=verbose)

        attempts.append({
            "chunk": new_lookup.chunk_id,
            "answer": new_lookup.answer,
            "verdict": new_verify.verdict,
            "reason": new_verify.reason,
        })

        if new_verify.verdict == "CONFIDENT":
            return ResearchResult(
                final_answer=new_lookup.answer,
                final_verdict="CONFIDENT",
                final_chunk=new_lookup.chunk_id,
                attempts=attempts,
                n_retries=retry + 1,
            )

        excluded.append(new_lookup.chunk_id)

    # All retries exhausted. Return the best uncertain answer with explicit
    # uncertainty marker. The orchestrator will format the final output.
    last = attempts[-1]
    return ResearchResult(
        final_answer=last["answer"],
        final_verdict="EXHAUSTED",
        final_chunk=last["chunk"],
        attempts=attempts,
        n_retries=len(attempts) - 1,
    )
