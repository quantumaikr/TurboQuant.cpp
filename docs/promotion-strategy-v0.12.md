# Promotion Strategy: v0.12 Document-Level RAG Breakthrough

## Date: 2026-04-11

## The Story (One Sentence)

> **"Chunk-RAG hallucinated 7/7 questions. Loading the full document with 6.4x KV compression got 7/7 correct — on a 16GB Mac."**

## Why This Story Resonates

1. **Concrete numbers**: 7/7 vs 0/7 is impossible to misread
2. **Real fear**: Hallucination is the #1 production RAG concern
3. **Counter-intuitive**: KV compression wasn't expected to enable this
4. **Reproducible**: Single-file benchmark, anyone can run it
5. **Actionable**: `pip install quantcpp` works today

## Three-Tier Audience Strategy

### Tier 1: r/LocalLLaMA (highest priority)

**Why first**: Our existing community, tech-savvy, RAG fatigue is high.

**Title options** (A/B test mentally):
- **A** (concrete): "We measured chunk-RAG vs full-document on a 3B model — 0/7 vs 7/7"
- **B** (provocative): "Your RAG hallucinates when retrieval fails. Here's the data."
- **C** (technical): "6.4x KV compression makes 'Document-Level RAG' practical on 16GB Macs"

**Recommend A** — concrete data wins on r/LocalLLaMA.

**Post structure**:
1. Hook: 7/7 vs 0/7 table
2. The hallucination examples (John Smith, $1M, 15%)
3. Methodology (Llama 3.2 3B, 7 questions, 3 methods)
4. Why it matters: chunking is the bug, not the model
5. CTA: `pip install quantcpp`, GitHub link, benchmark file
6. Honest disclaimer: "single synthetic doc, needs scale validation"

**Timing**: Tuesday or Wednesday, 9 AM ET (peak r/LocalLLaMA traffic)

**Avoid**:
- "Patent pending", "revolutionary", "patent us"
- Comparing to llama.cpp (we already covered this)
- Hiding limitations (community will dig them out anyway)

### Tier 2: HackerNews

**Why second**: Broader tech audience, RAG/AI is hot topic.

**Title** (HN style — concrete + intriguing):
- "Show HN: We compared chunk-RAG vs full-document QA — 0/7 vs 7/7"

**Post structure**:
1. Lead with the benchmark table
2. Brief on quant.cpp (16K LOC C, single header, KV compression)
3. The Document-Level RAG concept
4. Why this matters for production RAG
5. Repo link

**Avoid**:
- Marketing speak
- Vague claims
- Anything that sounds like a startup pitch

### Tier 3: Twitter/X

**Why third**: Amplification + screenshots.

**Thread structure** (5-7 tweets):

```
1/ Chunk-RAG: 0/7 ❌ (all hallucinated)
   Full Document: 7/7 ✅
   Same model. Same questions. Just different context approach.
   
   We measured this on Llama 3.2 3B Q8_0:
   [screenshot of benchmark table]

2/ When chunk-RAG retrieves the wrong section, the model doesn't say "I don't know."
   It generates plausible lies:
   • "CTO?" → "John Smith" (actually: Maria Santos)  
   • "Revenue?" → "$1M" (actually: $847M)
   • "R&D?" → "15% of net income" (actually: 14% of revenue)

3/ The fix isn't a smarter retriever. It's loading the full document.
   But that needs context windows that don't fit on consumer hardware.
   Until now.

4/ quant.cpp's 6.4x KV compression makes 128K context fit in 9.5 GB on a 16GB Mac.
   With Llama 3.2 3B, the full document fits → 7/7 accuracy → zero hallucinations.
   
   FP32 KV: 7/7
   6.4x compressed KV: 7/7 (zero quality loss)

5/ This isn't "RAG is dead." It's "chunking RAG is dangerous."
   RAG decides which documents to look at.
   Long-context decides how deeply to understand them.
   They complement each other.

6/ Open source, MIT-style: pip install quantcpp
   Single C header, 16K LOC, runs anywhere.
   
   Benchmark: github.com/quantumaikr/quant.cpp/blob/main/bench/results/document_level_rag_breakthrough.md
   
   /end
```

### Bonus: LinkedIn (selective)

**Audience**: Enterprise AI leads, ML engineers at companies with internal RAG.

**Tone**: Professional, focus on production risk.

**Key message**: "Your production RAG might be hallucinating without you knowing. Here's a measurable benchmark to find out."

## Defensive Preparation

### Anticipated criticism + responses

**Q: "5 sections, 7 questions, single model — that's not a benchmark."**
A: "Correct. This is a proof-of-concept, not a paper. Reproduce it in 5 minutes; we'd love to see results on LongBench/NIAH next."

**Q: "Of course full context beats wrong-chunk retrieval. Your retriever sucks."**
A: "Actually that's the point. Real production RAG fails silently when retrieval misses — and we showed exactly what 'silent failure' looks like (hallucination, not 'I don't know')."

**Q: "Why not just use Gemini 1.5 Pro / Claude 3 with native 1M context?"**
A: "Those run in cloud at $X/M tokens. quant.cpp runs locally for free on your laptop. Different use case (privacy, offline, cost)."

**Q: "Your model output has 'SanSannt' instead of 'Santos'. That's broken."**
A: "Q4 weight quantization artifact — semantically correct but visually noisy. For exact-string output use Q8 weights. Documented honestly in the report."

**Q: "Chunking has been a known problem for years. What's new?"**
A: "Two things: (1) We measured the failure mode quantitatively (silent hallucination). (2) We made the alternative practical on consumer hardware via KV compression."

## Honest Self-Assessment

**Strengths to lean into:**
- Concrete numbers, not vague claims
- Open source benchmark, instantly reproducible
- 11 prior self-corrections (track record of honesty)
- Real measurement on real hardware

**Weaknesses to acknowledge upfront:**
- Synthetic document (not real-world corpus)
- Single model size (3B)
- Single language (English)
- Q4 weight artifacts in output

**Don't lean into:**
- "Paradigm shift" language (premature, see paradigm-shift discussion)
- "RAG is dead" claims
- Comparing to specific commercial RAG products
- Anything that sounds like a startup pitch

## Success Metrics

**Tier 1 (r/LocalLLaMA)**:
- 200+ upvotes = good
- 500+ upvotes = great
- 1000+ upvotes = breakthrough
- Comments to track: meaningful technical discussion, reproductions, criticism

**Tier 2 (HN)**:
- Front page = good
- 100+ comments = great
- Thread depth (technical replies) > vote count

**Tier 3 (Twitter/X)**:
- 100+ retweets on lead tweet = good
- ML researcher engagement (Karpathy, Mikolov, etc.) = great

## Post-Launch Actions

**Day 1-3**: Active comment engagement, answer questions, fix typos found by community.

**Week 1**: Aggregate feedback into a follow-up "what we learned" post. Address top criticism transparently.

**Week 2-4**: Run the benchmark on:
- LongBench subset (real questions)
- 7B model (better instruction-following)
- 2-3 different document types (code, legal, novel)

**Month 2**: Write a more rigorous benchmark report based on what survives scrutiny. This becomes the "v2 evidence" for any future paradigm-shift claims.

## What Would Make This a Real Paradigm Shift (Future Work)

To upgrade from "interesting result" to "paradigm shift":
1. ✅ 0/7 vs 7/7 on synthetic data — done
2. ⏳ Same result on LongBench / NIAH (1000+ questions)
3. ⏳ Reproduced by independent team
4. ⏳ Featured in HuggingFace blog or paper citation
5. ⏳ Adopted by 1+ production system

We're at step 1. Steps 2-5 need months. The promotion now should reflect this honestly.
