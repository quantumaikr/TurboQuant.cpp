"""Stage 1: GIST.

Build a lightweight index of a document. Each chunk gets:
  - char_start, char_end          : where in the doc
  - head_text                     : first ~150 chars of the actual chunk text
  - entities                      : regex-extracted capitalized words + numbers
  - summary (optional, LLM)       : one-sentence model-written summary

Day 1 lesson: model-written gist summaries are too generic to discriminate
("This section is about Acme Robotics, a company that..." for every chunk).
The locator's primary signal is the actual chunk *content*, not a model
summary. We extract the first ~150 chars of each chunk and let the locator
match the question against real text.

The LLM-generated summary path is still available via use_llm=True for cases
where the chunk's head text isn't representative of the section content
(e.g., a chunk that starts mid-sentence). For the prototype, the no-LLM
path (use_llm=False) is the default — much faster, much more discriminating.
"""
import re
from dataclasses import dataclass, field, asdict
from typing import List

from . import _llm

# Chunk size in characters. Two constraints:
# 1. Cliff-safe: each chunk + lookup prompt template must fit below ~1024 tokens
# 2. Primacy-bias-safe: each chunk should be small enough that when stage 3
#    LOOKUP reads ONE chunk, the model doesn't pick the first-mentioned
#    entity over the question-relevant one. Phase 2B showed this bias kicks
#    in even well below the cliff. Empirically ~500 chars works.
# 500 chars ≈ 165 tokens — both constraints satisfied.
CHUNK_CHARS = 500

# How many leading characters of each chunk to include in the locator's
# index. Long enough to capture the section's topic, short enough that
# 10 chunks of head_text fit in one locator prompt below the cliff.
HEAD_TEXT_CHARS = 200


# Optional LLM-summary path (off by default in prototype)
GIST_SUMMARY_PROMPT = """Below is one section of a longer document.

{chunk}

In one short sentence, what is this section about? What are the main people, places, or facts mentioned?"""


@dataclass
class GistChunk:
    chunk_id: int
    char_start: int
    char_end: int
    head_text: str = ""           # first ~200 chars (used by LLM-fallback outline)
    full_text: str = ""           # complete chunk text (used by Day 3 non-LLM keyword scoring)
    full_text_norm: str = ""      # D6/D13: pre-normalized text (avoids re-normalizing per score call)
    entities: List[str] = field(default_factory=list)
    summary: str = ""             # optional LLM-generated summary

    def to_dict(self):
        d = asdict(self)
        d.pop("full_text_norm", None)  # don't serialize internal cache
        return d


@dataclass
class Gist:
    doc_id: str
    n_chars: int
    chunks: List[GistChunk]

    def to_outline_text(self) -> str:
        """Render the gist as a compact text outline that the locator
        will use to pick a chunk. Day 2 design: use head_text as the
        primary discriminator, not the model-written summary."""
        lines = []
        for c in self.chunks:
            # Compact one-line representation: chunk_id followed by the
            # head text (which contains real terms the locator can match
            # against the question).
            head = c.head_text.replace("\n", " ").strip()
            if len(head) > HEAD_TEXT_CHARS:
                head = head[:HEAD_TEXT_CHARS] + "…"
            lines.append(f"[{c.chunk_id}] {head}")
        return "\n".join(lines)


def _extract_entities(text: str) -> List[str]:
    """Regex-based entity extraction. No LLM call. Captures capitalized
    multi-word names + standalone numbers + dates. Tolerant; won't get
    everything but produces useful index terms cheaply."""
    # Capitalized 1-3 word sequences (names, places, orgs)
    cap_seq = re.findall(r"\b[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+){0,2}\b", text)
    # Standalone numbers (years, amounts, counts)
    nums = re.findall(r"\b\d{2,5}\b", text)
    # ALL-CAPS acronyms (CEO, CFO, CTO, etc.)
    acronyms = re.findall(r"\b[A-Z]{2,5}\b", text)
    # Combine, dedupe, cap at 12
    seen = set()
    out = []
    for item in cap_seq + acronyms + nums:
        item = item.strip()
        if item and item.lower() not in seen and len(item) > 1:
            seen.add(item.lower())
            out.append(item)
        if len(out) >= 12:
            break
    return out


MIN_CHUNK_CHARS = 100
MAX_CHUNK_CHARS = 800

# Day 4: char-based fallback uses larger chunks than paragraph-aware mode.
# For continuous narrative docs (no \\n\\n breaks) like wikitext, smaller
# chunks (~500 chars) made the locator's job too hard — 60+ chunks with
# overlapping topic words. Larger chunks (~1500 chars / ~500 tokens, well
# within the 1024-token cliff) give the locator fewer, more topically
# distinct candidates AND give the lookup model more cross-sentence
# context for pronoun resolution.
NARRATIVE_CHUNK_CHARS = 1500
NARRATIVE_MAX_CHUNK_CHARS = 2000


def chunk_document(doc_text: str, chunk_chars: int = CHUNK_CHARS) -> List[tuple]:
    """Day 3-4: paragraph-aware + char-based chunker.

    Strategy:
      1. If the doc has paragraph breaks (`\\n\\n`), use them as the
         primary split — preserves section structure (e.g., "Section 3:
         Growth Strategy" stays in one chunk). Default chunk_chars=500.
      2. Merge tiny adjacent paragraphs (< MIN_CHUNK_CHARS).
      3. Split any oversized chunk (> MAX_CHUNK_CHARS) at sentence boundaries.
      4. If there are no paragraph breaks, use the *narrative* path with
         larger chunks (NARRATIVE_CHUNK_CHARS=1500). The narrative path
         is for unstructured wikitext-style content where many small
         chunks would dilute the locator's discrimination power.

    Returns a list of (start, end, text) tuples.
    """
    n = len(doc_text)

    if "\n\n" in doc_text:
        raw_parts: List[tuple] = []
        pos = 0
        while pos < n:
            nxt = doc_text.find("\n\n", pos)
            if nxt < 0:
                raw_parts.append((pos, n))
                break
            raw_parts.append((pos, nxt))
            pos = nxt + 2

        merged: List[tuple] = []
        for start, end in raw_parts:
            if not merged:
                merged.append((start, end))
                continue
            prev_start, prev_end = merged[-1]
            if (prev_end - prev_start) < MIN_CHUNK_CHARS:
                merged[-1] = (prev_start, end)
            else:
                merged.append((start, end))

        out: List[tuple] = []
        for start, end in merged:
            length = end - start
            if length <= MAX_CHUNK_CHARS:
                out.append((start, end, doc_text[start:end]))
                continue
            sub_start = start
            while sub_start < end:
                sub_end = min(sub_start + chunk_chars, end)
                if sub_end < end:
                    sb = doc_text.find(". ", sub_end)
                    if sb > 0 and sb < end and sb - sub_end < 300:
                        sub_end = sb + 2
                out.append((sub_start, sub_end, doc_text[sub_start:sub_end]))
                sub_start = sub_end
        return out

    # Char-based narrative fallback (no paragraph structure) — uses
    # larger chunks for unstructured wikitext-style content.
    target = max(chunk_chars, NARRATIVE_CHUNK_CHARS)
    max_size = NARRATIVE_MAX_CHUNK_CHARS
    out = []
    pos = 0
    while pos < n:
        end = min(pos + target, n)
        if end < n:
            sb_next = doc_text.find(". ", end)
            if sb_next > 0 and sb_next - end < 400 and sb_next + 2 - pos <= max_size:
                end = sb_next + 2
        out.append((pos, end, doc_text[pos:end]))
        pos = end
    return out


def build_gist(
    doc_text: str,
    doc_id: str = "doc",
    *,
    chunk_chars: int = CHUNK_CHARS,
    use_llm: bool = False,
    verbose: bool = False,
) -> Gist:
    """Build a gist of a document.

    Default (use_llm=False): no LLM calls. Just chunk the text, store
    head_text and regex-extracted entities. Fast and discriminating.

    With use_llm=True: also generate a one-sentence summary per chunk
    via an LLM call. Costs N extra LLM calls per document but produces
    a richer index for cases where the chunk head text isn't
    representative of the section.
    """
    # Guard: empty or whitespace-only documents produce no chunks
    if not doc_text or not doc_text.strip():
        if verbose:
            print(f"[gist] doc_id={doc_id} — empty document, returning empty gist")
        return Gist(doc_id=doc_id, n_chars=len(doc_text or ""), chunks=[])

    chunks_raw = chunk_document(doc_text, chunk_chars=chunk_chars)
    if verbose:
        print(f"[gist] doc_id={doc_id} len={len(doc_text)} chars, {len(chunks_raw)} chunks "
              f"({'with LLM summary' if use_llm else 'no-LLM'})")

    out_chunks = []
    for i, (start, end, chunk_text) in enumerate(chunks_raw):
        head_text = chunk_text[:HEAD_TEXT_CHARS].strip()
        full_text = chunk_text.strip()
        entities = _extract_entities(chunk_text)

        summary = ""
        if use_llm:
            s_prompt = GIST_SUMMARY_PROMPT.format(chunk=chunk_text)
            s_result = _llm.llm_call(s_prompt, max_tokens=80)
            # B3: check for LLM errors before parsing summary
            if s_result.is_error:
                if verbose:
                    print(f"[gist] LLM error on chunk {i}: {s_result.text[:80]}")
                summary = ""  # fall back to no summary (head_text still available)
            else:
                summary = _parse_summary_response(s_result.text)

        # D6/D13: pre-normalize text once during gist build
        from ._text import normalize as _norm
        gc = GistChunk(
            chunk_id=i,
            char_start=start,
            char_end=end,
            head_text=head_text,
            full_text=full_text,
            full_text_norm=_norm(full_text),
            entities=entities,
            summary=summary,
        )
        out_chunks.append(gc)
        if verbose:
            print(f"[gist] chunk {i+1}/{len(chunks_raw)}: "
                  f"head={head_text[:60]!r}..., entities={entities[:4]}")

    return Gist(doc_id=doc_id, n_chars=len(doc_text), chunks=out_chunks)


def _parse_summary_response(text: str) -> str:
    """Take the first non-empty sentence as the summary."""
    text = text.strip()
    if "## Step" in text:
        parts = text.split("\n")
        non_step = [l for l in parts if not l.strip().startswith("##")]
        text = " ".join(non_step).strip()
    first_period = text.find(". ")
    if first_period > 0 and first_period < 200:
        return text[:first_period + 1]
    return text[:200]
