"""Stage 3: LOOKUP.

Day 3 v3 architecture: SELECT-BY-INDEX lookup.

  Day 2 framing was "Quote the single sentence". This works for
  single-hop but breaks under Q4 KV jitter for multi-hop:
    - Q6: model paraphrased "...proposed by EVP James Park during the
      2023 strategic planning retreat in Kyoto" → "...proposed by EVP
      in 2002" (dropped James Park and Kyoto entirely)
    - Q7: model picked the wrong sentence inside the right chunk
      ("Supply chain disruptions" instead of "Currency fluctuations")

  Day 3 redesign: numbered-sentence selection. We split the chunk into
  sentences, present them as a numbered list, and ask the model to pick
  a single integer (the sentence index that contains the answer). Then
  we return the *verbatim* sentence from the original text — the model
  never has to QUOTE, only SELECT. Selection is something the 3B Q4
  model is good at (the locator showed this); quoting under Q4 jitter
  is what's broken.
"""
import re
from dataclasses import dataclass
from typing import List

from . import _llm
from .locator import RegionPointer


# Day 3 v3: numbered-sentence selection prompt. The model picks an
# integer; we map it back to a verbatim sentence.
# H1/H2: prompts use explicit delimiters (---BEGIN/END---) to separate
# user-provided text from instructions, reducing prompt injection risk.
# The model is told to treat content between delimiters as opaque data.
# Model-agnostic prompts: natural language, no rigid format requirements.
# Works with Phi-3.5 (concise), Qwen3.5 (verbose), SmolLM2, etc.

LOOKUP_PROMPT_TEMPLATE = """Sentences from a document:

{numbered_sentences}

Question: {question}

Which sentence number answers the question? Reply with the number."""

LOOKUP_QUOTE_FALLBACK_TEMPLATE = """Document:
{region_text}

Question: {question}

Answer the question using ONLY information from the document above.
If the document does not contain the answer, say "not found"."""


@dataclass
class LookupResult:
    answer: str
    region_text: str
    chunk_id: int
    raw_llm_output: str = ""
    method: str = ""  # "select" | "quote" | "select-fallback"


# Common abbreviations that end with a period but aren't sentence endings.
_ABBREVIATIONS = {"dr", "mr", "mrs", "ms", "jr", "sr", "st", "vs", "etc",
                  "prof", "rev", "gen", "corp", "inc", "ltd", "vol", "no",
                  "approx", "dept", "est", "govt"}


def _split_into_sentences(text: str) -> List[str]:
    """Split text into sentences. Snaps on period/!?/whitespace but avoids
    splitting on common abbreviations (Dr., Mr., etc.) and single-letter
    initials (J. K. Rowling).
    Filters out tiny fragments (< 8 chars) that aren't real sentences."""
    # Strategy: split on `. ` / `! ` / `? `, then re-join fragments that
    # ended with an abbreviation or single letter.
    raw = re.split(r"(?<=[.!?])\s+", text)
    merged: List[str] = []
    for frag in raw:
        frag = frag.strip()
        if not frag:
            continue
        if merged:
            prev = merged[-1]
            # Check if prev ended with an abbreviation or single initial
            last_word = prev.rsplit(None, 1)[-1].rstrip(".").lower() if prev else ""
            if last_word in _ABBREVIATIONS or (len(last_word) == 1 and last_word.isalpha()):
                merged[-1] = prev + " " + frag
                continue
        merged.append(frag)
    return [s for s in merged if len(s) >= 8]


def _parse_sentence_index(text: str, n_sentences: int) -> int:
    """Find the first integer 1..n_sentences in the model's reply.
    Returns -1 on parse failure."""
    text = text.strip()
    if "## Step" in text:
        text = " ".join(l for l in text.split("\n") if not l.strip().startswith("##"))
    for m in re.finditer(r"\b(\d+)\b", text):
        n = int(m.group(1))
        if 1 <= n <= n_sentences:
            return n
    return -1


def lookup(
    question: str,
    region: RegionPointer,
    doc_text: str,
    *,
    verbose: bool = False,
) -> LookupResult:
    """Stage 3: read the targeted region and answer the question.

    Day 3 v3: select-by-index. Split the chunk into numbered sentences,
    let the model pick an index, return the verbatim sentence from the
    original text. The model never has to QUOTE — only SELECT.
    """
    region_text = doc_text[region.char_start:region.char_end]

    # A9: guard empty region (e.g., fallback pointer with char_start=char_end=0)
    if not region_text.strip():
        return LookupResult(
            answer="[no text in selected region]", region_text=region_text,
            chunk_id=region.chunk_id, raw_llm_output="", method="error",
        )

    sentences = _split_into_sentences(region_text)

    # Day 4 adaptive lookup: select-by-index for small chunks (≤8 sentences,
    # typical of structured docs with section headers — Acme regime), and
    # direct-answer for large chunks (>8 sentences, typical of continuous
    # narrative — wikitext regime). Select-by-index is perfect for Acme
    # (the model's quoting under Q4 jitter is broken, but integer selection
    # works); direct-answer is better for wikitext (the model needs full
    # chunk context for cross-sentence pronoun resolution, and select fails
    # to pick the right sentence when there are 15+ candidates).
    MAX_SENTENCES_FOR_SELECT = 8

    if len(sentences) < 2 or len(sentences) > MAX_SENTENCES_FOR_SELECT:
        # Direct-answer mode: feed the chunk text + question to the model
        prompt = LOOKUP_QUOTE_FALLBACK_TEMPLATE.format(
            region_text=region_text, question=question,
        )
        if verbose:
            mode = "direct-answer" if len(sentences) > MAX_SENTENCES_FOR_SELECT else "single-sentence"
            print(f"[lookup] chunk {region.chunk_id} ({len(region_text)} chars), "
                  f"{len(sentences)} sentences -> {mode}")
        result = _llm.llm_call(prompt, max_tokens=24)
        if result.is_error:
            return LookupResult(
                answer=result.text, region_text=region_text,
                chunk_id=region.chunk_id, raw_llm_output=result.text, method="error",
            )
        text = result.text.strip()
        # Model-agnostic refusal detection: various ways models say "not found"
        text_lower = text.lower()[:120]
        refusal_signals = [
            "not found", "not contain", "does not", "no information",
            "cannot determine", "not mentioned", "not stated", "not available",
            "not specified", "unable to", "i don't know", "no answer",
            "[NONE]", "none",
        ]
        is_refusal = any(sig in text_lower for sig in refusal_signals)
        if is_refusal and len(text) < 200:
            text = f"[NONE] {text}"
        # Strip common answer prefixes (model-agnostic)
        for prefix in ["ANSWER:", "Answer:", "answer:", "A:", "**Answer:**", "**"]:
            if text.startswith(prefix):
                text = text[len(prefix):].strip()
                break
        return LookupResult(
            answer=text,
            region_text=region_text,
            chunk_id=region.chunk_id,
            raw_llm_output=result.text,
            method="direct",
        )

    # Select-by-index mode: chunks with 2-8 sentences (structured docs)
    numbered = "\n".join(f"[{i+1}] {s}" for i, s in enumerate(sentences))
    prompt = LOOKUP_PROMPT_TEMPLATE.format(
        numbered_sentences=numbered,
        question=question,
    )

    if verbose:
        within, est, budget = _llm.check_cliff_budget(prompt)
        print(f"[lookup] chunk {region.chunk_id} ({len(region_text)} chars), "
              f"{len(sentences)} sentences, prompt ~{est} tokens "
              f"(budget {budget}), within={within}")

    # Only need a single digit — minimize tokens for slow CPU models
    result = _llm.llm_call(prompt, max_tokens=8)
    if result.is_error:
        return LookupResult(
            answer=result.text, region_text=region_text,
            chunk_id=region.chunk_id, raw_llm_output=result.text, method="error",
        )
    idx = _parse_sentence_index(result.text, len(sentences))

    if idx < 1:
        if verbose:
            print(f"[lookup] index parse failed: {result.text!r} -> quote fallback")
        prompt = LOOKUP_QUOTE_FALLBACK_TEMPLATE.format(
            region_text=region_text, question=question,
        )
        result2 = _llm.llm_call(prompt, max_tokens=24)
        return LookupResult(
            answer=result2.text.strip(),
            region_text=region_text,
            chunk_id=region.chunk_id,
            raw_llm_output=result2.text,
            method="select-fallback",
        )

    # Day 4: return a 2-sentence window (selected sentence + previous
    # sentence). For continuous-narrative wikitext, the answer often
    # requires pronoun resolution across adjacent sentences ("He was cast
    # in Mercury Fur. He was directed by John Tiffany." — picking either
    # sentence alone loses the connection). For Acme-style structured
    # docs, the previous sentence is benign extra context.
    # Return a 3-sentence window centered on the selected sentence.
    # Multi-hop questions often require context from adjacent sentences
    # (e.g., "strategy proposed at what event?" spans sentences about
    # the strategy AND the event name in the next sentence).
    window = []
    for offset in range(-1, 2):  # prev, selected, next
        i = idx - 1 + offset
        if 0 <= i < len(sentences):
            window.append(sentences[i])
    answer = " ".join(window)
    if verbose:
        print(f"[lookup] selected sentence {idx}/{len(sentences)}: {sentences[idx-1][:80]!r}")
    return LookupResult(
        answer=answer,
        region_text=region_text,
        chunk_id=region.chunk_id,
        raw_llm_output=result.text,
        method="select",
    )
