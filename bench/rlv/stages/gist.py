"""Stage 1: GIST.

Read the document chunk by chunk (each chunk sized below the cliff budget)
and produce a structured outline. The outline is small (~500-2000 tokens
for any-size document) and serves as the *index* for stages 2 and 4.

Output schema (one entry per chunk):
    [
      {
        "chunk_id": 0,
        "char_start": 0,
        "char_end": 3000,
        "topics": ["intro", "motivation"],
        "key_facts": ["released July 2023", "three sizes 7B/13B/70B"],
        "summary": "Introduction to Llama 2 and its motivation."
      },
      ...
    ]
"""
from dataclasses import dataclass, field, asdict
from typing import List

from . import _llm

# Chunk size in characters. Two constraints:
# 1. Cliff-safe: each chunk + gist prompt template must fit below ~1024 tokens
# 2. Primacy-bias-safe: each chunk should be small enough that when stage 3
#    LOOKUP reads ONE chunk, the model doesn't pick the first-mentioned
#    entity over the question-relevant one. Phase 2B showed this bias kicks
#    in even well below the cliff. Empirically ~500 chars works.
# 500 chars ≈ 165 tokens — both constraints satisfied.
CHUNK_CHARS = 500


# Use direct natural-language questions instead of structured format
# requests — Llama-3.2-3B-Q4 in chat mode emits reasoning chains when
# asked for structured output but answers cleanly to direct questions.
# We make TWO small calls per chunk (summary + entities) and parse the
# free-text responses with the tolerant extractor below.
GIST_SUMMARY_PROMPT = """Below is one section of a longer document.

{chunk}

In one short sentence, what is this section about? What are the main people, places, or facts mentioned?"""

GIST_ENTITIES_PROMPT = """Below is one section of a longer document.

{chunk}

List the named people, organizations, places, dates, and numbers mentioned in this section. Comma-separated, no other text."""


@dataclass
class GistChunk:
    chunk_id: int
    char_start: int
    char_end: int
    topics: List[str] = field(default_factory=list)
    entities: List[str] = field(default_factory=list)
    facts: List[str] = field(default_factory=list)
    summary: str = ""

    def to_dict(self):
        return asdict(self)


@dataclass
class Gist:
    doc_id: str
    n_chars: int
    chunks: List[GistChunk]

    def to_outline_text(self) -> str:
        """Render the gist as a compact text outline that fits in another
        LLM prompt (used by Stage 2 locator and Stage 4 verifier)."""
        lines = []
        for c in self.chunks:
            lines.append(f"[chunk {c.chunk_id}, chars {c.char_start}-{c.char_end}]")
            if c.topics:   lines.append(f"  topics: {', '.join(c.topics)}")
            if c.entities: lines.append(f"  entities: {', '.join(c.entities)}")
            if c.facts:    lines.append(f"  facts: {', '.join(c.facts)}")
            if c.summary:  lines.append(f"  summary: {c.summary}")
        return "\n".join(lines)


def _parse_summary_response(text: str) -> str:
    """Take the first non-empty sentence as the summary."""
    text = text.strip()
    # If model still emitted "## Step 1:" reasoning, take everything after the
    # last "##" line and treat as summary.
    if "## Step" in text:
        parts = text.split("\n")
        non_step = [l for l in parts if not l.strip().startswith("##")]
        text = " ".join(non_step).strip()
    # Take the first sentence (period-terminated)
    first_period = text.find(". ")
    if first_period > 0 and first_period < 200:
        return text[:first_period + 1]
    return text[:200]


def _parse_entities_response(text: str) -> list[str]:
    """Extract a comma-separated entity list from a free-text response."""
    # Strip any preamble like "Here are the entities:" etc.
    text = text.strip()
    if ":" in text and len(text.split(":", 1)[0]) < 60:
        text = text.split(":", 1)[1]
    # Take only the first line; some models wrap with extra explanation
    text = text.split("\n", 1)[0]
    items = [t.strip().rstrip(".,;") for t in text.split(",")]
    return [i for i in items if 1 < len(i) < 60][:12]


def chunk_document(doc_text: str, chunk_chars: int = CHUNK_CHARS) -> List[tuple]:
    """Split a document into cliff-safe chunks at sentence boundaries.
    Returns a list of (start, end, text) tuples."""
    chunks = []
    pos = 0
    n = len(doc_text)
    while pos < n:
        end = min(pos + chunk_chars, n)
        # Snap to next sentence boundary
        if end < n:
            sb_next = doc_text.find(". ", end)
            if sb_next > 0 and sb_next - end < 300:
                end = sb_next + 2
        chunks.append((pos, end, doc_text[pos:end]))
        pos = end
    return chunks


def build_gist(
    doc_text: str,
    doc_id: str = "doc",
    *,
    chunk_chars: int = CHUNK_CHARS,
    verbose: bool = False,
) -> Gist:
    """Build a gist of a document by running Stage 1 over each chunk."""
    chunks_raw = chunk_document(doc_text, chunk_chars=chunk_chars)
    if verbose:
        print(f"[gist] doc_id={doc_id} len={len(doc_text)} chars, {len(chunks_raw)} chunks")

    out_chunks = []
    for i, (start, end, chunk_text) in enumerate(chunks_raw):
        # Stage 1a: free-text summary
        s_prompt = GIST_SUMMARY_PROMPT.format(chunk=chunk_text)
        s_result = _llm.llm_call(s_prompt, max_tokens=80)
        summary = _parse_summary_response(s_result.text)

        # Stage 1b: entity list
        e_prompt = GIST_ENTITIES_PROMPT.format(chunk=chunk_text)
        e_result = _llm.llm_call(e_prompt, max_tokens=80)
        entities = _parse_entities_response(e_result.text)

        gc = GistChunk(
            chunk_id=i,
            char_start=start,
            char_end=end,
            topics=[],   # not used in current design — kept for schema stability
            entities=entities,
            facts=[],    # subsumed by summary + entities
            summary=summary,
        )
        out_chunks.append(gc)
        if verbose:
            print(f"[gist] chunk {i+1}/{len(chunks_raw)}: "
                  f"entities={entities[:3]}..., summary={summary[:60]!r}")

    return Gist(doc_id=doc_id, n_chars=len(doc_text), chunks=out_chunks)
