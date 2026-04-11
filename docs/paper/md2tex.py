#!/usr/bin/env python3
"""Minimal markdown→LaTeX converter for the working-memory-cliff tech report.

Handles only the markdown subset we actually use:
- Headings (# .. ######)
- Paragraphs and blockquotes
- Fenced code blocks (```)
- Pipe tables
- Inline: **bold**, *italic*, `code`, [text](url)
- Horizontal rules (---)
- Bullet lists (- and *)

Usage: python3 md2tex.py working-memory-cliff.md > working-memory-cliff.tex
"""
import re
import sys
from pathlib import Path


PREAMBLE = r"""\documentclass[11pt]{article}
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{lmodern}
\usepackage[margin=1in]{geometry}
\usepackage{hyperref}
\hypersetup{colorlinks=true,linkcolor=blue,urlcolor=blue,citecolor=blue}
\usepackage{booktabs}
\usepackage{array}
\usepackage{longtable}
\usepackage{listings}
\lstset{
  basicstyle=\ttfamily\small,
  breaklines=true,
  frame=single,
  numbers=none,
  showstringspaces=false,
  columns=fullflexible,
}
\usepackage{xcolor}
\usepackage{enumitem}
\setlist{nosep}
\usepackage{titlesec}
\titleformat*{\section}{\large\bfseries}
\titleformat*{\subsection}{\normalsize\bfseries}

\title{The Working Memory Cliff: Measuring When Quantized Edge LLMs Stop Following Instructions in Long Context}
\author{quant.cpp maintainers\thanks{Tech report v0.3, Apr 2026. Source, data and reproduction scripts: \url{https://github.com/quantumaikr/quant.cpp}}}
\date{\today}

\begin{document}
\maketitle

"""

POSTAMBLE = r"""
\end{document}
"""


def escape_tex(s: str) -> str:
    """Escape characters that LaTeX treats specially in text mode."""
    # Order matters: backslash first.
    s = s.replace("\\", r"\textbackslash{}")
    s = s.replace("&", r"\&")
    s = s.replace("%", r"\%")
    s = s.replace("$", r"\$")
    s = s.replace("#", r"\#")
    s = s.replace("_", r"\_")
    s = s.replace("{", r"\{")
    s = s.replace("}", r"\}")
    s = s.replace("~", r"\textasciitilde{}")
    s = s.replace("^", r"\textasciicircum{}")
    return s


def render_inline(s: str) -> str:
    """Convert inline markdown to LaTeX. Operates on already-text-escaped string fragments."""
    # `code`
    def code_repl(m):
        # the inline code content has been escaped already; reverse the escaping
        # for safer \texttt embedding by using \verb-like wrapping is hard, so
        # we keep escaped content but pass through \texttt
        return r"\texttt{" + m.group(1) + "}"

    # We need to handle inline code BEFORE bold/italic so that bold inside code is preserved.
    # But we want code content to be the literal escaped text. Strategy:
    # 1. Find inline code spans, replace with placeholders
    # 2. Apply bold/italic
    # 3. Restore code spans

    placeholders = []

    def stash_code(m):
        placeholders.append(m.group(1))
        return f"\x00CODE{len(placeholders) - 1}\x00"

    s = re.sub(r"`([^`]+)`", stash_code, s)

    # **bold**
    s = re.sub(r"\*\*([^*]+)\*\*", r"\\textbf{\1}", s)
    # *italic*
    s = re.sub(r"\*([^*]+)\*", r"\\textit{\1}", s)
    # [text](url)
    def link_repl(m):
        text = m.group(1)
        url = m.group(2).replace("%", r"\%").replace("#", r"\#").replace("_", r"\_")
        return r"\href{" + url + r"}{" + text + r"}"
    s = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", link_repl, s)

    # Restore code placeholders
    def restore_code(m):
        idx = int(m.group(1))
        return r"\texttt{" + placeholders[idx] + r"}"
    s = re.sub(r"\x00CODE(\d+)\x00", restore_code, s)

    return s


def convert(md: str) -> str:
    out = []
    lines = md.split("\n")
    i = 0
    in_code = False
    code_buf = []
    in_table = False
    table_buf = []

    def flush_table():
        nonlocal table_buf
        if not table_buf:
            return
        # Parse pipe table
        rows = []
        for raw in table_buf:
            cells = [c.strip() for c in raw.strip().strip("|").split("|")]
            rows.append(cells)
        # rows[0] = header, rows[1] = sep, rows[2:] = body
        if len(rows) < 2:
            table_buf = []
            return
        header = rows[0]
        body = rows[2:] if len(rows) >= 3 else []
        n = len(header)
        col_spec = "l" * n
        out.append(r"\begin{center}")
        out.append(r"\begin{tabular}{" + col_spec + r"}")
        out.append(r"\toprule")
        out.append(" & ".join(render_inline(escape_tex(c)) for c in header) + r" \\")
        out.append(r"\midrule")
        for row in body:
            # pad short rows
            row = row + [""] * (n - len(row))
            out.append(" & ".join(render_inline(escape_tex(c)) for c in row[:n]) + r" \\")
        out.append(r"\bottomrule")
        out.append(r"\end{tabular}")
        out.append(r"\end{center}")
        out.append("")
        table_buf = []

    while i < len(lines):
        line = lines[i]

        # Fenced code block
        if line.startswith("```"):
            if not in_code:
                flush_table()
                in_code = True
                code_buf = []
            else:
                in_code = False
                out.append(r"\begin{lstlisting}")
                out.extend(code_buf)
                out.append(r"\end{lstlisting}")
                out.append("")
                code_buf = []
            i += 1
            continue

        if in_code:
            code_buf.append(line)
            i += 1
            continue

        # Pipe table detection
        if "|" in line and line.lstrip().startswith("|"):
            in_table = True
            table_buf.append(line)
            i += 1
            continue
        elif in_table:
            in_table = False
            flush_table()

        # Headings
        m = re.match(r"^(#{1,6})\s+(.*)$", line)
        if m:
            level = len(m.group(1))
            text = render_inline(escape_tex(m.group(2)))
            if level == 1:
                # The title is set in the preamble; skip h1
                pass
            elif level == 2:
                out.append(r"\section*{" + text + "}")
            elif level == 3:
                out.append(r"\subsection*{" + text + "}")
            else:
                out.append(r"\subsubsection*{" + text + "}")
            out.append("")
            i += 1
            continue

        # Horizontal rule
        if line.strip() in ("---", "***", "___"):
            out.append(r"\medskip\hrule\medskip")
            out.append("")
            i += 1
            continue

        # Blockquote
        if line.startswith(">"):
            out.append(r"\begin{quote}")
            while i < len(lines) and lines[i].startswith(">"):
                stripped = lines[i].lstrip(">").strip()
                if stripped:
                    out.append(render_inline(escape_tex(stripped)))
                else:
                    out.append("")
                i += 1
            out.append(r"\end{quote}")
            out.append("")
            continue

        # Bullet list
        if re.match(r"^\s*[-*]\s+", line):
            out.append(r"\begin{itemize}")
            while i < len(lines) and re.match(r"^\s*[-*]\s+", lines[i]):
                item = re.sub(r"^\s*[-*]\s+", "", lines[i])
                out.append(r"\item " + render_inline(escape_tex(item)))
                i += 1
            out.append(r"\end{itemize}")
            out.append("")
            continue

        # Numbered list
        if re.match(r"^\s*\d+\.\s+", line):
            out.append(r"\begin{enumerate}")
            while i < len(lines) and re.match(r"^\s*\d+\.\s+", lines[i]):
                item = re.sub(r"^\s*\d+\.\s+", "", lines[i])
                out.append(r"\item " + render_inline(escape_tex(item)))
                i += 1
            out.append(r"\end{enumerate}")
            out.append("")
            continue

        # Plain paragraph (collect until blank line)
        if line.strip():
            para = [line]
            i += 1
            while i < len(lines) and lines[i].strip() and not lines[i].startswith(("#", ">", "-", "*", "|", "```")) and not re.match(r"^\s*\d+\.\s+", lines[i]):
                para.append(lines[i])
                i += 1
            text = " ".join(p.strip() for p in para)
            out.append(render_inline(escape_tex(text)))
            out.append("")
            continue

        # Blank line
        out.append("")
        i += 1

    if in_table:
        flush_table()

    return "\n".join(out)


def main():
    if len(sys.argv) < 2:
        print(__doc__, file=sys.stderr)
        sys.exit(1)
    md_path = Path(sys.argv[1])
    md = md_path.read_text(encoding="utf-8")
    body = convert(md)
    sys.stdout.write(PREAMBLE)
    sys.stdout.write(body)
    sys.stdout.write(POSTAMBLE)


if __name__ == "__main__":
    main()
