"""
quantcpp CLI — chat with a local LLM in your terminal.

Ollama-style commands:
    quantcpp pull MODEL       Download a model from HuggingFace
    quantcpp list             List cached and available models
    quantcpp run MODEL [Q]    Chat with a model (auto-pulls if needed)
    quantcpp serve MODEL      Start OpenAI-compatible HTTP server

Backwards-compatible shortcut:
    quantcpp                  Auto-downloads Llama-3.2-1B, starts chat
    quantcpp "What is X?"     One-shot question with default model
    quantcpp --model NAME     Use a specific model
"""

import sys
import os
import json


# Ollama-style short aliases → canonical _MODEL_REGISTRY keys
MODEL_ALIASES = {
    "smollm2":      "SmolLM2-135M",
    "smollm2:135m": "SmolLM2-135M",
    "qwen3.5":      "Qwen3.5-0.8B",
    "qwen3.5:0.8b": "Qwen3.5-0.8B",
    "llama3.2":     "Llama-3.2-1B",
    "llama3.2:1b":  "Llama-3.2-1B",
}


def _resolve_name(name):
    """Resolve user input to canonical registry key or local path."""
    if name is None:
        return None
    if os.path.exists(name) and name.endswith(".gguf"):
        return name
    return MODEL_ALIASES.get(name.lower(), name)


def _registry():
    from quantcpp import _MODEL_REGISTRY, _CACHE_DIR
    return _MODEL_REGISTRY, _CACHE_DIR


def cmd_pull(args):
    """Download a model by alias or canonical name."""
    import quantcpp
    name = _resolve_name(args.model)

    if os.path.exists(name) and name.endswith(".gguf"):
        print(f"already local: {name}")
        return 0

    if name not in quantcpp._MODEL_REGISTRY:
        avail = ", ".join(sorted(quantcpp._MODEL_REGISTRY.keys()))
        aliases = ", ".join(sorted(MODEL_ALIASES.keys()))
        print(f"unknown model: {args.model!r}", file=sys.stderr)
        print(f"  registry:  {avail}", file=sys.stderr)
        print(f"  aliases:   {aliases}", file=sys.stderr)
        return 1

    print(f"pulling {name}...", file=sys.stderr)
    try:
        path = quantcpp.download(name)
        size_mb = os.path.getsize(path) / (1024 * 1024)
        print(f"\u2713 {name} \u2192 {path} ({size_mb:.0f} MB)", file=sys.stderr)
        return 0
    except Exception as e:
        print(f"download failed: {e}", file=sys.stderr)
        return 1


def cmd_list(args):
    """List cached and available models."""
    registry, cache_dir = _registry()

    rows = []
    for name, (repo, filename, approx_mb) in sorted(registry.items()):
        path = cache_dir / filename
        if path.exists():
            size_mb = path.stat().st_size / (1024 * 1024)
            status = "cached"
            display_path = str(path)
        else:
            size_mb = approx_mb
            status = "remote"
            display_path = f"~{approx_mb} MB"
        alias = next((a for a, n in MODEL_ALIASES.items() if n == name and ":" in a), "")
        rows.append((status, name, alias, size_mb, display_path))

    if args.json_output:
        print(json.dumps([
            {"status": s, "name": n, "alias": a, "size_mb": round(sz, 1), "path": p}
            for (s, n, a, sz, p) in rows
        ], indent=2))
        return 0

    print(f"\n  Models  cache: {cache_dir}\n")
    print(f"  {'STATUS':<8} {'NAME':<16} {'ALIAS':<14} {'SIZE':>8}")
    print(f"  {'-'*8} {'-'*16} {'-'*14} {'-'*8}")
    for status, name, alias, size_mb, _ in rows:
        size_str = f"{size_mb:.0f} MB"
        print(f"  {status:<8} {name:<16} {alias:<14} {size_str:>8}")
    print()
    return 0


def _resolve_to_path(name_or_path):
    """Resolve alias/name to a local .gguf path, downloading if needed."""
    import quantcpp
    name = _resolve_name(name_or_path)

    if os.path.exists(name) and name.endswith(".gguf"):
        return name

    if name not in quantcpp._MODEL_REGISTRY:
        avail = ", ".join(sorted(quantcpp._MODEL_REGISTRY.keys()))
        raise ValueError(
            f"unknown model: {name_or_path!r}. Available: {avail}"
        )

    repo, filename, _ = quantcpp._MODEL_REGISTRY[name]
    cached = quantcpp._CACHE_DIR / filename
    if cached.exists():
        return str(cached)

    print(f"model not cached \u2014 pulling {name}...", file=sys.stderr)
    return quantcpp.download(name)


def cmd_run(args):
    """Chat with a model (auto-pull if needed)."""
    try:
        model_path = _resolve_to_path(args.model)
    except ValueError as e:
        print(str(e), file=sys.stderr)
        return 1
    except Exception as e:
        print(f"pull failed: {e}", file=sys.stderr)
        return 1

    from quantcpp import Model
    print(f"loading {os.path.basename(model_path)}...", file=sys.stderr)
    m = Model(model_path, max_tokens=args.max_tokens, temperature=args.temperature,
              n_threads=args.threads)

    if args.prompt:
        question = " ".join(args.prompt) if isinstance(args.prompt, list) else args.prompt
        for tok in m.generate(question):
            print(tok, end="", flush=True)
        print()
    else:
        print("quantcpp \u2014 type your message, Ctrl+C to exit", file=sys.stderr)
        try:
            while True:
                question = input("\nYou: ")
                if not question.strip():
                    continue
                print("AI: ", end="", flush=True)
                for tok in m.generate(question):
                    print(tok, end="", flush=True)
                print()
        except (KeyboardInterrupt, EOFError):
            print("\nBye!", file=sys.stderr)

    m.close()
    return 0


def cmd_serve(args):
    """Start OpenAI-compatible HTTP server (requires quant-server binary)."""
    import shutil
    import subprocess

    try:
        model_path = _resolve_to_path(args.model)
    except Exception as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    binary = shutil.which("quant-server")
    if not binary:
        # Look in common build dirs relative to repo
        for guess in ("./build/quant-server", "./build_metal/quant-server"):
            if os.path.isfile(guess) and os.access(guess, os.X_OK):
                binary = guess
                break

    if not binary:
        print("quant-server binary not found.", file=sys.stderr)
        print("  Build with: cmake -B build -DTQ_BUILD_SERVER=ON && cmake --build build",
              file=sys.stderr)
        print("  Or install via your package manager.", file=sys.stderr)
        return 2

    cmd = [binary, model_path, "-p", str(args.port), "-j", str(args.threads)]
    print(f"quant serve {os.path.basename(model_path)} on :{args.port}", file=sys.stderr)
    os.execvp(cmd[0], cmd)


def cmd_chat_default(args):
    """Backwards-compatible default: auto-download Llama-3.2-1B and chat."""
    args.model = args.model or "Llama-3.2-1B"
    args.threads = getattr(args, "threads", 4)
    args.max_tokens = getattr(args, "max_tokens", 256)
    args.temperature = getattr(args, "temperature", 0.7)
    args.prompt = args.prompt or None
    return cmd_run(args)


def main():
    import argparse

    parser = argparse.ArgumentParser(
        prog="quantcpp",
        description="Chat with a local LLM. No API key, no GPU, no server.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
commands:
  pull MODEL            Download a model (e.g. llama3.2:1b)
  list                  List cached and available models
  run MODEL [PROMPT]    Chat with a model (auto-pulls if needed)
  serve MODEL           Start OpenAI-compatible HTTP server

examples:
  quantcpp pull llama3.2:1b
  quantcpp list
  quantcpp run llama3.2:1b
  quantcpp run llama3.2:1b "What is gravity?"
  quantcpp serve llama3.2:1b --port 8080

backwards-compat (no subcommand):
  quantcpp                          # default chat with Llama-3.2-1B
  quantcpp "What is gravity?"       # one-shot
  quantcpp --model SmolLM2-135M     # different model
""",
    )

    sub = parser.add_subparsers(dest="command")

    # pull
    p_pull = sub.add_parser("pull", help="Download a model from HuggingFace")
    p_pull.add_argument("model", help="Model name or alias (e.g. llama3.2:1b)")

    # list
    p_list = sub.add_parser("list", help="List cached and available models")
    p_list.add_argument("--json", dest="json_output", action="store_true")

    # run
    p_run = sub.add_parser("run", help="Chat with a model (auto-pulls if needed)")
    p_run.add_argument("model", help="Model name, alias, or .gguf path")
    p_run.add_argument("prompt", nargs="*", default=None, help="Optional prompt")
    p_run.add_argument("-j", "--threads", type=int, default=4)
    p_run.add_argument("-n", "--max-tokens", type=int, default=256)
    p_run.add_argument("-t", "--temperature", type=float, default=0.7)

    # serve
    p_serve = sub.add_parser("serve", help="Start OpenAI-compatible HTTP server")
    p_serve.add_argument("model", help="Model name, alias, or .gguf path")
    p_serve.add_argument("-p", "--port", type=int, default=8080)
    p_serve.add_argument("-j", "--threads", type=int, default=4)

    # Backwards-compat: top-level args for direct chat
    parser.add_argument("prompt", nargs="*", default=None,
                        help="(default mode) question to ask")
    parser.add_argument("--model", "-m", default=None,
                        help="(default mode) model name or .gguf path")
    parser.add_argument("--max-tokens", "-n", type=int, default=256)
    parser.add_argument("--temperature", "-t", type=float, default=0.7)
    parser.add_argument("--threads", "-j", type=int, default=4)

    args = parser.parse_args()

    if args.command == "pull":
        return cmd_pull(args)
    if args.command == "list":
        return cmd_list(args)
    if args.command == "run":
        return cmd_run(args)
    if args.command == "serve":
        return cmd_serve(args)

    # No subcommand → backwards-compat default chat
    return cmd_chat_default(args)


if __name__ == "__main__":
    sys.exit(main())
