# r/LocalLLM Comment Responses — "7x longer LLM context in pure C" (2026-04-05)

Thread: quant.cpp — 7x longer LLM context in pure C (Gemma 4 26B on 16GB Mac)

Copy-paste ready. Each section = one comment.

---

## @dsanft — "4bit K completely kills inference quality due to kurtosis. K needs 8 bits."

You're right that K tensors have high kurtosis — outliers make them much harder to quantize than V. Naive per-tensor quantization does destroy quality.

The difference is granularity. quant.cpp uses per-block min-max quantization with 128-element blocks, not per-tensor or per-channel. Each block gets its own min/max scale, so outliers only affect their local block.

WikiText-2 PPL (SmolLM2 1.7B):

- FP32 baseline: 14.63
- 4-bit K + Q4 V: 14.57 (+0.0%)
- Cross-model: Qwen3.5 0.8B (+0.9%), Qwen3.5 4B (+0.6%)

For comparison, llama.cpp Q4_0 KV gives PPL +10.6% — that's the significant quality drop you're describing, and it's real with coarser quantization.

That said, you're absolutely right for QK-normed models like Gemma 4. Those project keys onto the unit sphere with extremely sparse distributions (~56 of 256 dims active). 4-bit completely breaks there (cosine drops to 0.62). quant.cpp auto-detects this and keeps keys in FP32 while only compressing values.

Reproducible: `./quant model.gguf --ppl input.txt -k uniform_4b -v q4`

---

## @MrHighVoltage — "at least do a proper job copying"

(Already responded. Post formatting fixed — switched to Markdown editor.)

---

## @maschayana — "Lol its still not fixed"

You're right, sorry about that. Reddit editor was fighting the markdown tables. Switched to Markdown mode and it should render properly now.

---

## @smuckola — "Titans, TurboQuant, KV Cache management landscape"

Great question — and you actually nailed it. quant.cpp is a C implementation of the TurboQuant paper (ICLR 2026). So you already found the connection without realizing it!

The KV cache management landscape breaks down roughly like this:

- **Eviction** (StreamingLLM, H2O, Scissors) — drop tokens you "probably" don't need. Saves memory but loses information permanently.
- **Architecture changes** (Titans, MLA, GQA) — redesign the model itself to use less KV memory. Best results, but requires retraining from scratch.
- **Compression** (TurboQuant/quant.cpp, KIVI, KVQuant) — keep all tokens, store them in fewer bits. Works on existing models, no retraining.

quant.cpp sits in the compression category. The advantage is that it works on any existing GGUF model — download, run, get 7x more context. No fine-tuning, no architecture change.

Titans is a different and complementary approach — it redesigns the attention mechanism itself so the model learns what to remember. Very promising, but requires models trained with it. If a Titans-architecture model ships as GGUF someday, quant.cpp could still compress its KV cache on top.

And thanks for the kind words about the focus. "Torvaldsian side quest" — I'm framing that.

---

## @sinan_online — "replicability and compatibility — containers, standard APIs, plug-n-play"

Thanks for the concrete use case — these are fair concerns.

**Replicability**: quant.cpp reads standard GGUF files directly. No model conversion, no custom formats. Any GGUF you download from Hugging Face works as-is. KV compression happens at runtime — the model file is untouched, so you can swap models freely. Same binary, different GGUF, same flags.

**Containers**: The binary is statically linkable with zero external dependencies (libc + pthreads only). No Python, no PyTorch, no CUDA runtime to install. A minimal Docker image can be under 10MB. That said, we don't ship an official container image yet — that's a fair gap.

**Standard API**: This is the honest limitation. quant.cpp has a C API (`quant_load` / `quant_generate`), not an OpenAI-compatible HTTP server. If you need a drop-in replacement for an existing API pipeline, llama.cpp's `llama-server` or vLLM is the right tool today.

Where quant.cpp fits in your workflow: if you're already running llama.cpp in a container and hitting context limits, we have an integration patch at `integrations/llamacpp/` that adds our KV compression as a drop-in option. Same API, longer context. The goal is to upstream delta compression into llama.cpp as a PR.

---

## @MimosaTen — "chatgpt-20b-Q4 could be the best model I've tried"

Nice — gpt-oss-20b is a solid model. It uses a GPT-2-style architecture with RoPE and MoE (32 experts), which is close to what quant.cpp already supports but not there yet. We handle Llama, Qwen, and Gemma architectures today.

If you're on limited hardware, KV compression would help a lot with a 20B MoE model — the KV cache is usually what runs you out of memory before the weights do, especially with long conversations.

I'll look into adding gpt-oss support. The MoE + RoPE + GQA pieces are already implemented for Gemma 4, so the gap is mostly the GPT-2 layer structure. Thanks for the suggestion!
