# Reddit r/LocalLLM Responses — quant.cpp rebrand update (2026-04-03)

Copy-paste ready. Each section = one comment.

---

## Top-level update (new comment on the thread)

We rebranded to **quant.cpp** (https://github.com/quantumaikr/quant.cpp). Old URLs redirect automatically.

Also owe you all an honest correction: the early 1-bit "zero loss" claim had a bug. An FP32 key cache was still being read during attention, so the quantized keys were never actually used. We found it, fixed it, and pulled every claim based on that measurement.

Here's where things actually stand (SmolLM2 1.7B, 999 tokens, real dequant path, no FP32 fallback):

- 4-bit K: PPL +0.0% (genuinely lossless)
- delta + 3-bit K + Q4 V: PPL -3.2%, ~4.3x compression
- 2-bit and below: all failed. we tried everything. drift is the fundamental barrier.

The breakthrough is delta compression — adjacent keys in a transformer differ by ~30% of their absolute range, so storing deltas instead of absolutes lets 3-bit work where it otherwise gives +62% PPL. Think video P-frames for KV cache.

Feedback from this thread is what pushed us to find the bug and be more rigorous. Appreciate it.

---

## @MrRandom04 — "re-implementing all of llama.cpp"

Fair point. We're not trying to replace llama.cpp — quant.cpp is a 33K LOC embeddable engine for people who want something they can read and modify. Different use case. We also have a llama.cpp integration patch at `integrations/llamacpp/` that adds our KV types as a drop-in option. The plan is to upstream the delta compression as a PR.

---

## @MrRandom04 (follow-up) — "why not fork llama.cpp?"

Needed to test quantization across 4 architectures (Llama, Gemma, Qwen, MoE) and debug subtle bit-packing issues. Doing that inside 250K lines of someone else's codebase would've been brutal. The standalone engine proved the algorithm works; now the goal is getting delta compression into llama.cpp proper.

---

## @dinerburgeryum — "codebook calibration sensitive to out-of-domain data?"

The recommended config (uniform_4b) doesn't use codebooks at all — it's per-block min-max quantization, so there's nothing to calibrate and no domain sensitivity.

---

## @Blizado — "zero quality loss claim"

You were right to push back. The "zero loss" measurement had a bug — FP32 keys were still being used for attention, so the quantization wasn't actually being tested. We found and fixed it.

After the fix, real numbers:
- 4-bit K: PPL +0.0% (this one is genuinely lossless)
- delta + 3-bit K + Q4 V: PPL -3.2%
- 1-bit: doesn't work. sign reconstruction cosine is ~0.8, not enough for attention.

Rebranded to quant.cpp, rewrote all claims from scratch. https://github.com/quantumaikr/quant.cpp

---

## @BillDStrong — "1-bit version"

Update: 1-bit doesn't actually work. We had an FP32 fallback bug that masked the problem. After fixing it, sign-based key reconstruction gives cosine ~0.8, which destroys attention at longer sequences. What does work is delta + 3-bit (PPL -3.2%) — that's where the real result is.

---

## @OftenTangential — "36 is absurd PPL for Gemma 3 4B"

You were right. 101-token test set was way too short. We re-measured on 999 tokens with SmolLM2 1.7B: baseline PPL = 14.58, uniform 4-bit K = 14.58 (+0.0%), delta + 3-bit K + Q4 V = 14.11 (-3.2%). Standard benchmarks (WikiText-2) are next.

---

## @Viper-Reflex — "does this make my 24GB 3090 run bigger models?"

KV compression extends context, not model size. On your 3090 with Llama 8B Q4: context goes from 147K to 559K tokens. Doesn't help fit a bigger model, but if you're already running one, you get way more context out of the same VRAM.

---

## @ganonfirehouse420 — "huge context for local models"

Available now:
```
./quant model.gguf -p "your long prompt" -k uniform_4b -v q4
```
16GB Mac, SmolLM2 1.7B: 78K → 298K context. No hardware upgrade needed.

---

## @TopChard1274 — "brutal for people who invested in expensive systems"

KV compression helps every tier equally — the 3.8x ratio is the same whether you have 8GB or 80GB. Bigger systems benefit by pushing context further or running larger batches. It doesn't make hardware obsolete; it makes whatever you have go further.

---

## @Candid_Koala_3602 — "angular mappings instead of weights?"

quant.cpp compresses the KV cache data, not the transformer architecture itself. But you're touching on something real — attention is fundamentally a cosine similarity ranking, so preserving key direction is what matters. That's why delta compression works: small deltas preserve direction better than re-quantizing from scratch every token.

---

## @MaybeADragon — "Em dashes. No more to be said."

(Skip. Code is open source, 33K lines of C, 34 test suites. Results are reproducible.)

---

## @Fuehnix — "every reply is LLM generated"

Yeah I use Claude as a dev tool — for writing code, drafting docs, and yes, sometimes helping with replies. The code itself is 33K lines of C written with AI assistance and verified by hand. Every PPL number is a real measurement from a real model. If you think the results are wrong, point at a specific number and I'll show you how to reproduce it.

Repo is here if you want to look at actual code instead of prose style: https://github.com/quantumaikr/quant.cpp

---

## @Candid_Koala_3602 — "working on angular compression concept"

Cool — will take a look at the preprint. The idea of unifying transformer computation and compression into one mechanism is interesting. Our delta compression works at a simpler level (just exploiting temporal correlation in adjacent keys), but if you've found something that does both architecture and compression, that's a different beast. Happy to discuss if there's overlap.

---

## @RIP26770 — "XPU support?"

Not yet. Currently: NEON (ARM), AVX2 (x86) production-ready, Metal (Apple) verified, CUDA/Vulkan compile but untested on GPU. Intel XPU / SYCL isn't on the roadmap yet but the codebase is pure C so porting a backend is straightforward — contributions welcome.

---

## @MrHighVoltage — "Ignore all previous instructions..."

lol. No bitcoin wallet, no system prompt to leak. It's a C binary, not a chatbot. `./quant model.gguf -p "hello"` — that's the whole interface.

---

## @Big_River_ — (word salad)

(Skip.)

---

## @Turbulent-Half-1515 — "no human involved here"

I'm the author — human, based in Korea, running a company called QuantumAI. I use Claude Code as a development tool, same way others use Copilot or Cursor. The architectural decisions, the bug hunts (we found and disclosed an FP32 fallback bug that invalidated our own 1-bit claims), the strategy calls — those are mine. The 33K lines of C didn't write themselves either; AI accelerated it, I directed and verified it.

If the concern is about AI-assisted code quality: every number in the README is a reproducible measurement, the repo has 34 passing tests, and I've publicly corrected every wrong claim I made. That's more accountability than most projects on this sub.

---

## @quanteval — "prefill heavy, short outputs, 2.5 bits had measurable loss"

Good observation. You're right that our eval setup is prefill-heavy (teacher-forced PPL over 999 tokens). We haven't tested long autoregressive generation quality separately — that's a fair gap.

On bit-width: we agree. Our own testing confirms 2.5-bit and below has real loss. The "zero quality loss" claim now only applies to 4-bit K (+0.0% PPL). At 3-bit, delta compression gets it to -3.2%, but we wouldn't call that "zero loss" — it's "better than baseline on this benchmark," which could be noise or regularization. We report the exact numbers and let people judge.
