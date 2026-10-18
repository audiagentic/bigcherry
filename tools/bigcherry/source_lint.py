"""TR14 source lint: a string literal that contains a real line break.

Generated or hand-pasted edits keep writing ``"...\\n"`` as an actual line break inside the quotes. Python reports that as a
syntax error as soon as the file is parsed, but C++, HIP and the ``.rw`` rewrite blocks of the libr3 component are only
compiled on the GPU host, so the mistake used to surface a build later. This module finds it offline:

  * Python files (tools/, engines/, excluding generated and vendor trees): parsed with ``ast``; an unterminated string is
    TR14.STRING_LITERAL_NEWLINE, any other syntax error TR14.PYTHON_PARSE.
  * C, C++ and HIP sources and headers, and the code blocks of ``.rw`` files: a small lexer that understands line and block
    comments, character literals, raw strings and backslash continuation, and reports a ``"`` string that reaches the end of
    its line without closing.

Pure, read-only, no imports from the rest of the package.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

CPP_SUFFIXES = {".c", ".cc", ".cpp", ".cxx", ".h", ".hh", ".hpp", ".hxx", ".cu", ".cuh", ".hip", ".inc"}
_SKIP_PARTS = {"__pycache__", "node_modules", "vendor", "src", "work", "artifacts", ".git", "fixtures"}
_BLOCK_START = re.compile(r"^@@\s+(find|replace)\b")
_BLOCK_END = re.compile(r"^@@\s+end\b")


def cpp_string_newlines(text: str) -> list:
    """Line numbers (1-based) where a double-quoted string literal is still open at the end of its line."""
    bad = []
    n, i, line = len(text), 0, 1
    while i < n:
        c = text[i]
        two = text[i:i + 2]
        if c == "\n":
            line += 1
            i += 1
        elif two == "//":
            while i < n and text[i] != "\n":
                if text[i] == "\\" and i + 1 < n:       # a backslash continues a // comment onto the next line
                    i += 1
                    if text[i] == "\n":
                        line += 1
                i += 1
        elif two == "/*":
            end = text.find("*/", i + 2)
            end = n if end < 0 else end + 2
            line += text.count("\n", i, end)
            i = end
        elif c == "R" and text[i + 1:i + 2] == '"' and (i == 0 or not (text[i - 1].isalnum() or text[i - 1] == "_")):
            paren = text.find("(", i + 2)
            delim = text[i + 2:paren] if paren > 0 and paren - i < 20 else None
            if delim is None or "\n" in delim or " " in delim:
                i += 1
                continue
            close = text.find(")" + delim + '"', paren)
            end = n if close < 0 else close + len(delim) + 2
            line += text.count("\n", i, end)
            i = end
        elif c == '"':
            start = line
            i += 1
            while i < n:
                ch = text[i]
                if ch == "\\":
                    if i + 1 < n and text[i + 1] == "\n":
                        line += 1
                    i += 2
                    continue
                if ch == '"':
                    i += 1
                    break
                if ch == "\n":
                    bad.append(start)
                    break                                # leave the newline for the outer loop to count
                i += 1
        elif c == "'":
            prev = text[i - 1] if i else ""
            if prev.isalnum() and text[i + 1:i + 2].isalnum():
                i += 1                                    # a digit separator, as in 1'000'000
                continue
            i += 1
            while i < n and text[i] not in "'\n":
                i += 2 if text[i] == "\\" else 1
            if i < n and text[i] == "'":
                i += 1
        else:
            i += 1
    return bad


def rw_string_newlines(text: str) -> list:
    """The same check over the code blocks of a libr3 ``.rw`` file; its header and the text between blocks are prose."""
    bad, block, first = [], [], 0

    def flush():
        if first and block:
            bad.extend(first + b - 1 for b in cpp_string_newlines("\n".join(block)))

    for number, raw in enumerate(text.split("\n"), 1):
        if _BLOCK_START.match(raw):                  # "find" and "replace" are separate blocks
            flush()
            block, first = [], number + 1
        elif _BLOCK_END.match(raw):
            flush()
            block, first = [], 0
        elif first:
            block.append(raw)
    flush()
    return bad


def python_problem(text: str):
    """None, or (kind, line, message): kind is "string" for an unterminated literal and "syntax" for anything else."""
    try:
        ast.parse(text)
    except SyntaxError as exc:
        message = exc.msg or ""
        kind = "string" if "unterminated" in message or "EOL while scanning" in message else "syntax"
        return kind, exc.lineno or 0, message
    return None


def lint_file(path: Path) -> list:
    """[(kind, line, message)] for one file; empty when clean or not a file this lint covers."""
    name = path.name
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    if path.suffix == ".py":
        problem = python_problem(text)
        return [problem] if problem else []
    if name.endswith(".rw"):
        return [("string", n, "a string literal in a rewrite block ends at the line break") for n in rw_string_newlines(text)]
    if path.suffix in CPP_SUFFIXES:
        return [("string", n, "a string literal ends at the line break") for n in cpp_string_newlines(text)]
    return []


def source_files(root: Path):
    for top in ("tools", "engines"):
        base = root / top
        if not base.is_dir():
            continue
        for path in sorted(base.rglob("*")):
            if not path.is_file():
                continue
            parts = set(path.relative_to(root).parts[:-1])
            if parts & _SKIP_PARTS or any(p.startswith(".") for p in parts):
                continue
            if path.suffix == ".py" or path.suffix in CPP_SUFFIXES or path.name.endswith(".rw"):
                yield path
