#!/usr/bin/env python3
"""
TurboQuant CLI — Interactive chat with KV cache compression analysis.

Usage:
    python3 tools/tq_chat.py                          # Interactive mode
    python3 tools/tq_chat.py "Your question here"     # Single question
    python3 tools/tq_chat.py --benchmark               # Run benchmark suite
"""

import sys
import os
import time
import argparse

# Colors
class C:
    BOLD = "\033[1m"
    DIM = "\033[2m"
    CYAN = "\033[36m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    RED = "\033[31m"
    MAGENTA = "\033[35m"
    BLUE = "\033[34m"
    NC = "\033[0m"
    BAR = "█"
    BAR_EMPTY = "░"

def bar(value, max_val, width=30, color=C.GREEN):
    filled = int(value / max_val * width) if max_val > 0 else 0
    filled = min(filled, width)
    return f"{color}{C.BAR * filled}{C.DIM}{C.BAR_EMPTY * (width - filled)}{C.NC}"

def size_str(bytes_val):
    if bytes_val >= 1024 * 1024 * 1024:
        return f"{bytes_val / 1024**3:.2f} GB"
    elif bytes_val >= 1024 * 1024:
        return f"{bytes_val / 1024**2:.1f} MB"
    elif bytes_val >= 1024:
        return f"{bytes_val / 1024:.1f} KB"
    return f"{bytes_val} B"

def print_header():
    print()
    print(f"{C.CYAN}{C.BOLD}╔══════════════════════════════════════════════════════════╗{C.NC}")
    print(f"{C.CYAN}{C.BOLD}║  🚀 TurboQuant CLI — KV Cache Compression for LLMs     ║{C.NC}")
    print(f"{C.CYAN}{C.BOLD}║  Model: Qwen3.5-0.8B  |  Powered by QuantumAI Inc.     ║{C.NC}")
    print(f"{C.CYAN}{C.BOLD}╚══════════════════════════════════════════════════════════╝{C.NC}")
    print()

def print_kv_analysis(cache, prompt_len):
    """Analyze and visualize KV cache compression."""
    import torch

    total_fp16 = 0
    layers = 0
    for i in range(len(cache.key_cache)):
        k = cache.key_cache[i]
        if k is not None and isinstance(k, torch.Tensor) and k.dim() >= 3:
            total_fp16 += k.nelement() * 2 * 2  # K+V, fp16
            layers += 1

    tq_4b = int(total_fp16 * 4.2 / 16)
    tq_2b = int(total_fp16 * 2.2 / 16)
    k4v2 = int(total_fp16 * (4.2 + 2.2) / 2 / 16)

    print()
    print(f"  {C.BOLD}📊 KV Cache Analysis{C.NC}")
    print(f"  {C.DIM}{'─' * 52}{C.NC}")
    print(f"  Attention Layers: {C.BOLD}{layers}{C.NC}  |  Prompt Tokens: {C.BOLD}{prompt_len}{C.NC}")
    print()
    print(f"  {C.BOLD}{'Method':<22} {'Size':>10}  {'Compression':>12}  Bar{C.NC}")
    print(f"  {'─' * 22} {'─' * 10}  {'─' * 12}  {'─' * 30}")

    configs = [
        ("FP16 (baseline)", total_fp16, 1.0, C.RED),
        ("TQ uniform_4b", tq_4b, total_fp16 / tq_4b, C.GREEN),
        ("TQ K4V2 asymmetric", k4v2, total_fp16 / k4v2, C.GREEN),
        ("TQ uniform_2b", tq_2b, total_fp16 / tq_2b, C.YELLOW),
    ]

    for name, size, comp, color in configs:
        print(f"  {name:<22} {size_str(size):>10}  {comp:>10.1f}x  {bar(size, total_fp16, 30, color)}")

    saved = total_fp16 - k4v2
    print()
    print(f"  {C.GREEN}{C.BOLD}💾 Best balance (K4V2): saves {size_str(saved)} ({saved*100//total_fp16}%){C.NC}")

    # Scale projections
    print()
    print(f"  {C.BOLD}📈 Projected at longer contexts:{C.NC}")
    per_token = total_fp16 / prompt_len
    for ctx in [4096, 16384, 65536, 131072]:
        fp16 = per_token * ctx
        k4v2_proj = fp16 * (4.2 + 2.2) / 2 / 16
        saved_proj = fp16 - k4v2_proj
        ctx_str = f"{ctx // 1024}K"
        print(f"  {ctx_str:>6}: FP16 {size_str(fp16):>10} → TQ {size_str(k4v2_proj):>10}  {bar(k4v2_proj, fp16, 20, C.GREEN)} save {size_str(saved_proj)}")


def run_chat(question, model, tokenizer):
    """Run a single question through the model with analysis."""
    import torch

    print(f"  {C.BOLD}{C.BLUE}Q:{C.NC} {question}")
    print()

    messages = [{"role": "user", "content": question}]
    text = tokenizer.apply_chat_template(messages, tokenize=False,
                                          add_generation_prompt=True,
                                          enable_thinking=False)
    inputs = tokenizer(text, return_tensors="pt")
    prompt_len = inputs["input_ids"].shape[1]

    print(f"  {C.BOLD}{C.GREEN}A:{C.NC} ", end="", flush=True)

    import contextlib, io
    t0 = time.time()
    with torch.no_grad(), contextlib.redirect_stderr(io.StringIO()):
        out = model.generate(
            **inputs,
            max_new_tokens=300,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )
    elapsed = time.time() - t0

    answer = tokenizer.decode(out[0][prompt_len:], skip_special_tokens=True)
    gen_tokens = out.shape[1] - prompt_len

    # Print answer with wrapping
    import textwrap
    for line in answer.split("\n"):
        wrapped = textwrap.fill(line, width=72, initial_indent="     ",
                                subsequent_indent="     ")
        print(wrapped)

    # Stats
    print()
    print(f"  {C.DIM}{'─' * 52}{C.NC}")
    tps = gen_tokens / elapsed if elapsed > 0 else 0
    print(f"  {C.DIM}⏱  {gen_tokens} tokens in {elapsed:.1f}s ({tps:.1f} tok/s)  |  prompt: {prompt_len} tokens{C.NC}")

    # KV cache analysis
    with torch.no_grad():
        out2 = model(**inputs, use_cache=True)
        cache = out2.past_key_values

    print_kv_analysis(cache, prompt_len)


def main():
    parser = argparse.ArgumentParser(description="TurboQuant CLI — Chat with KV cache analysis")
    parser.add_argument("question", nargs="?", help="Question to ask (interactive if omitted)")
    parser.add_argument("--benchmark", action="store_true", help="Run benchmark suite")
    args = parser.parse_args()

    print_header()

    # Load model (suppress noisy warnings)
    print(f"  {C.DIM}Loading Qwen3.5-0.8B...{C.NC}", end="", flush=True)

    import warnings
    import logging
    import contextlib, io
    warnings.filterwarnings("ignore")
    logging.disable(logging.WARNING)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_name = "Qwen/Qwen3.5-0.8B"
    with contextlib.redirect_stderr(io.StringIO()):
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name, trust_remote_code=True, dtype=torch.float32
        )
    model.eval()

    # Pre-set pad_token_id to suppress "Setting pad_token_id" message
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model.generation_config.pad_token_id = tokenizer.eos_token_id

    print(f" {C.GREEN}✓{C.NC}")
    print()

    if args.benchmark:
        questions = [
            "What is 2+2?",
            "Explain KV cache quantization in one paragraph.",
            "Write a Python function that computes fibonacci numbers.",
        ]
        for q in questions:
            run_chat(q, model, tokenizer)
            print()
            print(f"  {C.DIM}{'═' * 52}{C.NC}")
            print()
    elif args.question:
        run_chat(args.question, model, tokenizer)
    else:
        # Interactive mode
        print(f"  {C.YELLOW}Interactive mode. Type your question (or 'quit' to exit).{C.NC}")
        print()
        while True:
            try:
                q = input(f"  {C.BOLD}You:{C.NC} ").strip()
                if not q or q.lower() in ("quit", "exit", "q"):
                    print(f"\n  {C.DIM}Goodbye!{C.NC}\n")
                    break
                print()
                run_chat(q, model, tokenizer)
                print()
                print(f"  {C.DIM}{'═' * 52}{C.NC}")
                print()
            except (KeyboardInterrupt, EOFError):
                print(f"\n  {C.DIM}Goodbye!{C.NC}\n")
                break


if __name__ == "__main__":
    main()
