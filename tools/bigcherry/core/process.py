"""Decode a subprocess return code into something a human can act on.

A bare integer exit code from a native binary is nearly useless on its own
-- especially on Windows, where a crashed process reports its NTSTATUS
value as an unsigned 32-bit number (``subprocess``/``os`` surface it signed
on some paths, unsigned on others, so a caller has to normalize before
comparing against the documented hex constants). ``describe_returncode``
does that normalization once and, for the handful of NTSTATUS values that
actually show up in a native-binary build/smoke context, adds the
symbolic name and (for the ones where it matters) an actionable hint.

gpt-dev-agent review, 2026-08-31: found live during the b10705 bump --
runtime-smoke reported "runtime smoke exited 3221225781" with no
indication that's ``0xC0000135``/``STATUS_DLL_NOT_FOUND``, i.e. very likely
a PATH/runtime-library problem rather than the binary itself being broken
(which is exactly what it was: this session's shell had a stale PATH
missing the ROCm bin directory). Shared here rather than duplicated
per-callsite because build.py's own subprocess failures (configure/build,
not just runtime-smoke) can hit the same class of Windows loader error.
"""

from __future__ import annotations

# NTSTATUS values worth naming in a native-binary build/smoke context.
# Not an exhaustive NTSTATUS table -- just the ones plausible here: loader
# failures (DLL_NOT_FOUND, ENTRYPOINT_NOT_FOUND, DLL_INIT_FAILED, INVALID_
# IMAGE_FORMAT), and common native crash shapes (ACCESS_VIOLATION, STACK_
# OVERFLOW, STACK_BUFFER_OVERRUN, HEAP_CORRUPTION, ILLEGAL_INSTRUCTION).
_NTSTATUS_NAMES: dict[int, str] = {
    0xC0000005: "STATUS_ACCESS_VIOLATION",
    0xC000007B: "STATUS_INVALID_IMAGE_FORMAT",
    0xC000001D: "STATUS_ILLEGAL_INSTRUCTION",
    0xC00000FD: "STATUS_STACK_OVERFLOW",
    0xC0000135: "STATUS_DLL_NOT_FOUND",
    0xC0000139: "STATUS_ENTRYPOINT_NOT_FOUND",
    0xC0000142: "STATUS_DLL_INIT_FAILED",
    0xC0000374: "STATUS_HEAP_CORRUPTION",
    0xC0000409: "STATUS_STACK_BUFFER_OVERRUN",
}

# Status-specific hints toward the LIKELY cause -- not a diagnosis, a
# starting point. Only for statuses where the hint is genuinely narrowing;
# a generic crash (ACCESS_VIOLATION, HEAP_CORRUPTION, ...) gets no hint
# because "the binary has a bug" isn't actionable enough to print.
_NTSTATUS_HINTS: dict[int, str] = {
    0xC0000135: (
        "a required DLL was not found -- check PATH/runtime-library "
        "availability (e.g. a ROCm/CUDA bin directory) before assuming "
        "the binary itself is broken"
    ),
    0xC0000139: (
        "a required DLL was found but is missing an expected export -- "
        "usually a version mismatch between the binary and an on-PATH "
        "runtime library, not a binary defect"
    ),
    0xC0000142: (
        "a required DLL's initializer failed -- often a runtime library "
        "version mismatch or a missing dependency of that DLL itself"
    ),
}


def describe_returncode(code: int) -> str:
    """Render ``code`` as decimal + hex, with the NTSTATUS symbolic name
    and an actionable hint when recognized. Always returns a usable
    string, including for a plain Unix-style exit code (which just gets
    the decimal form, unchanged) or an unrecognized NTSTATUS (decimal +
    hex only, no name/hint fabricated).
    """
    normalized = code & 0xFFFFFFFF
    name = _NTSTATUS_NAMES.get(normalized)
    if name is None:
        # Not a recognized NTSTATUS-shaped value (or a normal small exit
        # code) -- don't imply Windows-specific meaning for e.g. exit 1.
        if 0 <= code <= 255:
            return str(code)
        return f"{code} (0x{normalized:08X})"

    rendered = f"{code} (0x{normalized:08X}, {name})"
    hint = _NTSTATUS_HINTS.get(normalized)
    if hint:
        rendered += f" -- {hint}"
    return rendered
