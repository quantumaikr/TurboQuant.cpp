"""
quantcpp CLI — chat with a local LLM in your terminal.

Usage:
    quantcpp                          # auto-downloads Llama-3.2-1B, starts chat
    quantcpp "What is gravity?"       # one-shot question
    quantcpp --model SmolLM2-135M     # use a smaller model (faster download)
    quantcpp --model path/to/file.gguf  # use your own GGUF file
"""

import sys
import os


def main():
    import argparse
    parser = argparse.ArgumentParser(
        prog="quantcpp",
        description="Chat with a local LLM. No API key, no GPU, no server.",
    )
    parser.add_argument("prompt", nargs="*", help="Question to ask (omit for interactive chat)")
    parser.add_argument("--model", "-m", default="Llama-3.2-1B",
                        help="Model name or path to .gguf file (default: Llama-3.2-1B)")
    parser.add_argument("--max-tokens", "-n", type=int, default=256)
    parser.add_argument("--temperature", "-t", type=float, default=0.7)
    args = parser.parse_args()

    from quantcpp import Model

    # Load model
    model_path = args.model
    if os.path.isfile(model_path):
        print(f"Loading {model_path}...", file=sys.stderr)
        m = Model(model_path, max_tokens=args.max_tokens, temperature=args.temperature)
    else:
        print(f"Downloading {model_path}...", file=sys.stderr)
        m = Model.from_pretrained(model_path, max_tokens=args.max_tokens,
                                   temperature=args.temperature)

    # One-shot or interactive
    if args.prompt:
        question = " ".join(args.prompt)
        for tok in m.generate(question):
            print(tok, end="", flush=True)
        print()
    else:
        print("quantcpp — type your message, Ctrl+C to exit", file=sys.stderr)
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


if __name__ == "__main__":
    main()
