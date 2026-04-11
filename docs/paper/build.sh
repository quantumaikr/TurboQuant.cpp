#!/usr/bin/env bash
# Build the working-memory-cliff tech report into arXiv-ready PDF + TeX.
#
# Two paths:
#   1. pandoc (recommended)            — produces clean arXiv-quality PDF
#   2. md2tex.py + pdflatex (fallback) — pure Python + texlive, no pandoc
#
# The Python fallback only handles the markdown subset we actually use:
# headings, paragraphs, fenced code, tables, blockquotes, inline code,
# bold, italics, links. It is intentionally not a general md→tex
# converter — it just emits enough LaTeX to make the report compile.

set -e
cd "$(dirname "$0")"

if command -v pandoc >/dev/null 2>&1; then
  echo "==> Using pandoc"
  pandoc working-memory-cliff.md \
    --standalone \
    --pdf-engine=xelatex \
    --variable=geometry:margin=1in \
    --variable=fontsize=11pt \
    --variable=linkcolor:blue \
    --variable=documentclass:article \
    --highlight-style=tango \
    --metadata=title:"The Working Memory Cliff: Measuring When Quantized Edge LLMs Stop Following Instructions in Long Context" \
    --metadata=author:"quant.cpp maintainers" \
    --metadata=date:"$(date -u +%Y-%m-%d)" \
    -o working-memory-cliff.tex
  echo "==> Compiled to working-memory-cliff.tex"
  if command -v xelatex >/dev/null 2>&1; then
    xelatex -interaction=nonstopmode working-memory-cliff.tex >/dev/null
    xelatex -interaction=nonstopmode working-memory-cliff.tex >/dev/null
    echo "==> PDF: $(pwd)/working-memory-cliff.pdf"
  else
    echo "    (xelatex not installed — install MacTeX or texlive-xetex to build PDF)"
  fi
else
  echo "==> pandoc not found — using Python fallback"
  python3 md2tex.py working-memory-cliff.md > working-memory-cliff.tex
  echo "==> Compiled to working-memory-cliff.tex"
  if command -v pdflatex >/dev/null 2>&1; then
    pdflatex -interaction=nonstopmode working-memory-cliff.tex >/dev/null
    pdflatex -interaction=nonstopmode working-memory-cliff.tex >/dev/null
    echo "==> PDF: $(pwd)/working-memory-cliff.pdf"
  else
    echo "    (pdflatex not installed — install MacTeX or texlive to build PDF)"
    echo "    Hint:  brew install --cask mactex-no-gui   # or:  brew install pandoc basictex"
  fi
fi
