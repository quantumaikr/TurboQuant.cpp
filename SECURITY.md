# Security Policy

## Supported Versions

| Version | Supported |
|---------|-----------|
| 0.5.x   | ✅        |
| < 0.5   | ❌        |

## Reporting a Vulnerability

If you discover a security vulnerability, please report it responsibly:

1. **Do NOT** open a public GitHub issue
2. Email **hi@quantumai.kr** with details
3. Include steps to reproduce if possible
4. We will respond within 48 hours

## Scope

quant.cpp processes untrusted model files (GGUF). Known attack surfaces:
- GGUF parser (src/engine/tq_gguf.c) — malformed headers, oversized tensors
- Tokenizer (src/engine/tq_tokenizer.c) — malformed vocab data
- mmap handling — file size validation

We take buffer overflows and memory corruption seriously.
