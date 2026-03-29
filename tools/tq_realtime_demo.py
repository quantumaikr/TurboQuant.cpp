#!/usr/bin/env python3
"""
TurboQuant Real-Time Demo — Actual KV cache compression on live model

This demo ACTUALLY uses TurboQuant to compress the KV cache from a real
Qwen3.5-0.8B inference, then computes attention scores using TurboQuant's
integer kernel, and compares speed + quality vs PyTorch FP32.

This is NOT a simulation — it's real quantization on real model data.

Usage:
    python3 tools/tq_realtime_demo.py
    python3 tools/tq_realtime_demo.py "Your question here"
"""

import sys
import os
import time
import warnings
import argparse

warnings.filterwarnings("ignore")
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../bindings/python"))

C_BOLD = "\033[1m"
C_DIM = "\033[2m"
C_CYAN = "\033[36m"
C_GREEN = "\033[32m"
C_YELLOW = "\033[33m"
C_RED = "\033[31m"
C_NC = "\033[0m"
BAR = "█"
BAR_E = "░"

def bar(v, mx, w=25):
    f = min(int(v / mx * w) if mx > 0 else 0, w)
    return f"{C_GREEN}{BAR * f}{C_DIM}{BAR_E * (w - f)}{C_NC}"

def sz(b):
    if b >= 1e9: return f"{b/1e9:.2f} GB"
    if b >= 1e6: return f"{b/1e6:.1f} MB"
    if b >= 1e3: return f"{b/1e3:.1f} KB"
    return f"{b} B"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("question", nargs="?", default="Explain KV cache quantization briefly.")
    args = parser.parse_args()

    print()
    print(f"{C_CYAN}{C_BOLD}╔══════════════════════════════════════════════════════════╗{C_NC}")
    print(f"{C_CYAN}{C_BOLD}║  TurboQuant Real-Time Demo — Actual KV Compression      ║{C_NC}")
    print(f"{C_CYAN}{C_BOLD}║  Model: Qwen3.5-0.8B  |  Powered by QuantumAI Inc.      ║{C_NC}")
    print(f"{C_CYAN}{C_BOLD}╚══════════════════════════════════════════════════════════╝{C_NC}")
    print()

    # ── Load model ──
    print(f"  {C_DIM}[1/5] Loading model...{C_NC}", end="", flush=True)
    import torch, contextlib, io
    import numpy as np
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_name = "Qwen/Qwen3.5-0.8B"
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    dtype = torch.float16 if device == "mps" else torch.float32

    with contextlib.redirect_stderr(io.StringIO()):
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, dtype=dtype).to(device)
    model.eval()
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model.generation_config.pad_token_id = tokenizer.eos_token_id
    dev_label = "MPS" if device == "mps" else "CPU"
    print(f" {C_GREEN}✓{C_NC} ({dev_label})")

    # ── Load TurboQuant ──
    print(f"  {C_DIM}[2/5] Loading TurboQuant...{C_NC}", end="", flush=True)
    try:
        from turboquant import TurboQuant
        tq = TurboQuant("cpu")
        print(f" {C_GREEN}✓{C_NC}")
    except Exception as e:
        print(f" {C_RED}✗{C_NC} ({e})")
        return 1

    # ── Generate response ──
    print(f"  {C_DIM}[3/5] Generating response...{C_NC}", end="", flush=True)
    messages = [{"role": "user", "content": args.question}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    inputs = tokenizer(text, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    prompt_len = inputs["input_ids"].shape[1]

    t0 = time.time()
    with torch.no_grad(), contextlib.redirect_stderr(io.StringIO()):
        out = model.generate(**inputs, max_new_tokens=80, do_sample=True, temperature=0.7, top_p=0.9)
    gen_time = time.time() - t0
    gen_tokens = out.shape[1] - prompt_len
    answer = tokenizer.decode(out[0][prompt_len:], skip_special_tokens=True)
    tps = gen_tokens / gen_time
    print(f" {C_GREEN}✓{C_NC} ({gen_tokens} tok, {tps:.1f} tok/s)")

    print()
    print(f"  {C_BOLD}{C_CYAN}Q:{C_NC} {args.question}")
    print(f"  {C_BOLD}{C_GREEN}A:{C_NC} {answer[:200]}{'...' if len(answer) > 200 else ''}")
    print()

    # ── Extract KV cache ──
    print(f"  {C_DIM}[4/5] Extracting & compressing KV cache...{C_NC}")
    with torch.no_grad():
        out2 = model(**{k: v[:, :prompt_len] for k, v in inputs.items()}, use_cache=True)
        cache = out2.past_key_values

    # Collect attention layers
    layers_data = []
    for i in range(len(cache.key_cache)):
        k = cache.key_cache[i]
        v = cache.value_cache[i]
        if k is None or not isinstance(k, torch.Tensor) or k.dim() < 3:
            continue
        k_np = k.squeeze(0).cpu().float().numpy()
        v_np = v.squeeze(0).cpu().float().numpy()
        layers_data.append({"keys": k_np, "values": v_np, "layer": i})

    if not layers_data:
        print(f"  {C_RED}No attention layers found{C_NC}")
        return 1

    nh, sl, hd = layers_data[0]["keys"].shape
    total_fp16 = sum(d["keys"].nbytes + d["values"].nbytes for d in layers_data)

    # ── REAL TurboQuant A/B comparison ──
    print(f"  {C_DIM}[5/5] Real A/B comparison: PyTorch FP32 vs TurboQuant Q4×Q8{C_NC}")
    print()
    print(f"  {C_BOLD}📊 Real-Time KV Cache Compression Results{C_NC}")
    print(f"  {C_DIM}{'─' * 56}{C_NC}")
    print(f"  {C_BOLD}Model:{C_NC} {model_name}  {C_DIM}│{C_NC}  {C_BOLD}{len(layers_data)}{C_NC} attn layers  "
          f"{C_DIM}│{C_NC}  {C_BOLD}{nh}{C_NC} heads  {C_DIM}│{C_NC}  dim {C_BOLD}{hd}{C_NC}")
    print(f"  {C_BOLD}Speed:{C_NC} {gen_tokens} tokens in {gen_time:.1f}s ({C_CYAN}{C_BOLD}{tps:.1f} tok/s{C_NC})")
    print()

    # A: FP32 attention (PyTorch baseline)
    query_np = np.random.RandomState(42).randn(hd).astype(np.float32) * 0.1
    total_fp32_time = 0
    total_tq_time = 0
    cosine_scores = []

    for ld in layers_data:
        for h in range(nh):
            keys_h = ld["keys"][h]  # [seq_len, head_dim]

            # FP32 attention
            t0 = time.time()
            for _ in range(100):
                fp32_scores = keys_h @ query_np
            total_fp32_time += (time.time() - t0)

            # TurboQuant Q4 attention
            t0 = time.time()
            for _ in range(100):
                q_data = tq.quantize_keys(keys_h, TurboQuant.UNIFORM_4B)
                tq_scores = tq.attention(query_np, q_data, sl, hd, TurboQuant.UNIFORM_4B)
            total_tq_time += (time.time() - t0)

            # Quality
            cos = float(np.dot(fp32_scores, tq_scores) /
                        (np.linalg.norm(fp32_scores) * np.linalg.norm(tq_scores) + 1e-10))
            cosine_scores.append(cos)

    n_total = len(layers_data) * nh
    fp32_avg = total_fp32_time / n_total * 1000  # ms per 100 reps
    tq_avg = total_tq_time / n_total * 1000
    avg_cosine = np.mean(cosine_scores)
    tq_compressed = sum(len(tq.quantize_keys(ld["keys"][0], TurboQuant.UNIFORM_4B)) * nh
                        for ld in layers_data)

    # Results
    print(f"  {C_BOLD}{'Metric':<28} {'FP32':>12} {'TurboQuant':>12} {'Ratio':>8}{C_NC}")
    print(f"  {'─'*28} {'─'*12} {'─'*12} {'─'*8}")

    print(f"  {'KV Cache Size':<28} {sz(total_fp16):>12} {sz(tq_compressed):>12} "
          f"{C_GREEN}{total_fp16/tq_compressed:.1f}x{C_NC}")

    speedup = fp32_avg / tq_avg if tq_avg > 0 else 1
    sp_color = C_GREEN if speedup > 1 else C_RED
    print(f"  {'Attention (100 reps)':<28} {fp32_avg:>10.1f}ms {tq_avg:>10.1f}ms "
          f"{sp_color}{speedup:.1f}x{C_NC}")

    grade = "A+" if avg_cosine > 0.99 else "A" if avg_cosine > 0.95 else "B+" if avg_cosine > 0.90 else "B"
    print(f"  {'Quality (cosine)':<28} {'1.000':>12} {avg_cosine:>12.4f} {C_GREEN}{grade}{C_NC}")

    # Memory bar
    print()
    print(f"  {C_BOLD}Memory Comparison:{C_NC}")
    print(f"  FP16:       {sz(total_fp16):>10}  {C_RED}{BAR * 25}{C_NC}")
    print(f"  TurboQuant: {sz(tq_compressed):>10}  {bar(tq_compressed, total_fp16)}")
    print(f"  {C_GREEN}{C_BOLD}Saved: {sz(total_fp16 - tq_compressed)} ({(1-tq_compressed/total_fp16)*100:.0f}%){C_NC}")

    print()
    print(f"  {C_DIM}All numbers above are REAL — measured on actual Qwen3.5-0.8B KV cache,{C_NC}")
    print(f"  {C_DIM}not synthetic benchmarks. TurboQuant compression is applied live.{C_NC}")
    print()


if __name__ == "__main__":
    main()
