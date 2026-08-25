"""Minimal C/C++ source scanning helpers.

The audit and the patcher both need to answer questions like "what values does
the switch inside *this* function use?". Parsing C++ properly is out of scope
and unnecessary: the constructs we care about are shallow and stylistically
consistent in llama.cpp. What matters is that we never match commented-out or
stringified code, and that we scope a search to one function body rather than
the whole file — a whole-file regex would happily pick up an unrelated switch
and report a false pass.
"""

from __future__ import annotations

import re
from pathlib import Path

_LINE_COMMENT = re.compile(r"//[^\n]*")
_BLOCK_COMMENT = re.compile(r"/\*.*?\*/", re.DOTALL)
_STRING = re.compile(r'"(?:[^"\\\n]|\\.)*"')
_CHAR = re.compile(r"'(?:[^'\\\n]|\\.)'")


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _blank_preserving_newlines(match: re.Match[str]) -> str:
    """Replace a span with spaces, keeping newlines so line numbers survive."""
    return "".join("\n" if ch == "\n" else " " for ch in match.group(0))


_CMAKE_COMMENT = re.compile(r"#[^\n]*")


def strip_noise(text: str, language: str = "c") -> str:
    """Blank out comments and literals, preserving offsets and line numbers.

    Offsets are preserved so a match found in the stripped text can be used to
    slice the original.

    CMake is handled separately and deliberately keeps its string literals:
    in CMake the interesting content — source paths, glob patterns, target
    names — *is* inside quotes, so blanking them would erase exactly what an
    anchor needs to match.
    """
    if language == "cmake":
        return _CMAKE_COMMENT.sub(_blank_preserving_newlines, text)
    if language == "none":
        return text
    text = _BLOCK_COMMENT.sub(_blank_preserving_newlines, text)
    text = _LINE_COMMENT.sub(_blank_preserving_newlines, text)
    text = _blank_strings_outside_includes(text)
    text = _CHAR.sub(_blank_preserving_newlines, text)
    return text


_INCLUDE_PATH = re.compile(r'^[ \t]*#[ \t]*include[ \t]*"[^"\n]*"', re.MULTILINE)


def _blank_strings_outside_includes(text: str) -> str:
    """Blank string literals, but keep ``#include "path"`` intact.

    An include path is spelled with quotes but is not a string literal in any
    sense an anchor cares about — it is the name of the file, and naming a file
    is exactly how a patch says where it wants to attach. Blanking it would
    make ``#include "ggml-cuda/mmvq.cuh"`` unmatchable.
    """
    protected = [m.span() for m in _INCLUDE_PATH.finditer(text)]

    def blank(match: re.Match[str]) -> str:
        start, end = match.span()
        for lo, hi in protected:
            if lo <= start and end <= hi:
                return match.group(0)
        return _blank_preserving_newlines(match)

    return _STRING.sub(blank, text)


def language_for(path: str) -> str:
    """Guess the noise-stripping dialect from a file name."""
    name = path.replace("\\", "/").rsplit("/", 1)[-1]
    if name == "CMakeLists.txt" or name.endswith(".cmake"):
        return "cmake"
    if name.endswith((".py", ".sh", ".yml", ".yaml")):
        return "none"
    return "c"


def find_braced_block(text: str, from_index: int) -> tuple[int, int] | None:
    """Return (start, end) of the brace-balanced block at/after ``from_index``.

    ``start`` indexes the opening brace, ``end`` is one past the closing brace.
    ``text`` must already be noise-stripped, otherwise a brace inside a comment
    or string will unbalance the scan.
    """
    open_at = text.find("{", from_index)
    if open_at < 0:
        return None
    depth = 0
    for i in range(open_at, len(text)):
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return open_at, i + 1
    return None


def function_body(text: str, name: str, *, occurrence: int = 0) -> str | None:
    """Return the body of the definition of ``name`` (braces included).

    Declarations (``;`` before any ``{``) are skipped, so a forward declaration
    ahead of the definition does not shadow it. ``text`` is noise-stripped here,
    so callers may pass raw source.
    """
    stripped = strip_noise(text)
    pattern = re.compile(r"\b" + re.escape(name) + r"\s*\(")
    seen = 0
    for match in pattern.finditer(stripped):
        after_params = find_braced_block(stripped, match.end())
        # Reject declarations: a ';' between the name and the next '{'.
        semi = stripped.find(";", match.end())
        brace = stripped.find("{", match.end())
        if brace < 0 or (0 <= semi < brace):
            continue
        if after_params is None:
            continue
        if seen != occurrence:
            seen += 1
            continue
        start, end = after_params
        return stripped[start:end]
    return None


def switch_body(text: str, subject: str) -> str | None:
    """Return the body of ``switch (subject) { ... }`` within ``text``."""
    stripped = strip_noise(text)
    pattern = re.compile(r"\bswitch\s*\(\s*" + re.escape(subject) + r"\s*\)")
    match = pattern.search(stripped)
    if not match:
        return None
    block = find_braced_block(stripped, match.end())
    if block is None:
        return None
    start, end = block
    return stripped[start:end]


def int_captures(text: str, pattern: str) -> list[int]:
    """All integers captured by ``pattern`` in ``text``, in source order."""
    return [int(m.group(1)) for m in re.finditer(pattern, text)]
