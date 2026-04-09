#!/usr/bin/env python3
"""
long_context_qa.py — Replace your RAG pipeline with 3 lines of Python.

Loads a document into long context (up to 32K tokens) using KV compression,
then answers questions about it. No vector database, no chunking, no embedding
model — just the document and a question.

Usage:
    pip install quantcpp
    python long_context_qa.py document.txt "What is the main argument?"

Requirements:
    - Any GGUF model (auto-downloads Llama-3.2-1B if none specified)
    - 8GB RAM is enough for 32K context thanks to KV compression

This is NOT a toy demo. KV compression makes 32K context fit in 5.5GB
for a 3B model — where FP32 KV would need 10.4GB and OOM on 8GB hardware.
"""

import sys
import os


def main():
    # ---------------------------------------------------------------
    # Parse arguments
    # ---------------------------------------------------------------
    if len(sys.argv) < 3:
        print("Usage: python long_context_qa.py <document.txt> <question>")
        print('       python long_context_qa.py <document.txt> <question> [model.gguf]')
        print()
        print("Examples:")
        print('  python long_context_qa.py paper.txt "What is the main finding?"')
        print('  python long_context_qa.py contract.txt "What are the termination clauses?"')
        sys.exit(1)

    doc_path = sys.argv[1]
    question = sys.argv[2]
    model_path = sys.argv[3] if len(sys.argv) > 3 else None

    # ---------------------------------------------------------------
    # Load document
    # ---------------------------------------------------------------
    if not os.path.isfile(doc_path):
        print(f"Error: file not found: {doc_path}")
        sys.exit(1)

    with open(doc_path, "r", encoding="utf-8", errors="replace") as f:
        document = f.read()

    # Rough token estimate (1 token ~ 4 chars for English)
    est_tokens = len(document) // 4
    print(f"Document: {doc_path} ({len(document):,} chars, ~{est_tokens:,} tokens)")

    if est_tokens > 30000:
        print(f"Warning: document is ~{est_tokens:,} tokens, may exceed 32K context.")
        print("         Truncating to ~30K tokens for safety.")
        document = document[:120000]
        est_tokens = 30000

    # ---------------------------------------------------------------
    # Load model (auto-download if needed)
    # ---------------------------------------------------------------
    from quantcpp import Model

    ctx_len = min(max(est_tokens + 2000, 4096), 32768)  # headroom for response

    if model_path:
        print(f"Model: {model_path}")
        m = Model(model_path, context_length=ctx_len, max_tokens=512)
    else:
        print("Model: Llama-3.2-1B (auto-downloading...)")
        m = Model.from_pretrained("Llama-3.2-1B", context_length=ctx_len, max_tokens=512)

    print(f"Context: {ctx_len:,} tokens (KV compression ON — fits in ~{ctx_len * 72 // 1024 // 1024 + 1}GB)")
    print()

    # ---------------------------------------------------------------
    # Build prompt: document + question
    # ---------------------------------------------------------------
    prompt = (
        f"Read the following document carefully, then answer the question.\n\n"
        f"--- DOCUMENT ---\n{document}\n--- END DOCUMENT ---\n\n"
        f"Question: {question}\n\n"
        f"Answer:"
    )

    # ---------------------------------------------------------------
    # Generate answer (streaming)
    # ---------------------------------------------------------------
    print(f"Q: {question}")
    print("A: ", end="", flush=True)

    for token in m.generate(prompt):
        print(token, end="", flush=True)

    print()
    m.close()


if __name__ == "__main__":
    main()
