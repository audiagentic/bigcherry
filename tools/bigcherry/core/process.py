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

import json
import re
import socket
import subprocess
import sys
import time
from contextvars import ContextVar
from pathlib import Path

# The build boundary registers sinks; captured child output stays off console.
output_logs: ContextVar[tuple] = ContextVar("process_output_logs", default=())


def run_logged(*args, **kwargs):
    """Run a child and persist captured output as well as inherited output."""
    try:
        result = subprocess.run(*args, **kwargs)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        for sink in output_logs.get():
            sink.write(exc.stdout or b"")
            sink.write(exc.stderr or b"")
        raise
    for sink in output_logs.get():
        sink.write(result.stdout or b"")
        sink.write(result.stderr or b"")
    return result


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


# A compiler that dies from a signal (a random clang/gcc crash, about 1 in 160 identical compiles on this host with
# plenty of memory free) prints one of these. A real compile error, a user interrupt and a hangup are not crashes.
_SIGNALS = {
    "Segmentation fault": "SIGSEGV", "Aborted": "SIGABRT", "Bus error": "SIGBUS", "Illegal instruction": "SIGILL",
    "Floating point exception": "SIGFPE", "Killed": "SIGKILL",
}
_CRASH_LINE = re.compile(
    r"unable to execute command: (?P<a>" + "|".join(map(re.escape, _SIGNALS)) + r")"
    r"|internal compiler error: (?P<b>" + "|".join(map(re.escape, _SIGNALS)) + r")"
    r"|fatal error: (?P<c>Killed) signal terminated program")
_FAILED_UNIT = re.compile(r"^FAILED: (?:\[[^\]]*\] )?(?P<unit>\S+)")
RETRY_RECEIPT = "bigcherry-compile-retries.jsonl"
HOST_RETRY_RECEIPT = "build-compile-retries.jsonl"


def _is_build(argv: list[str]) -> bool:
    return len(argv) >= 3 and Path(argv[0]).stem == "cmake" and argv[1] == "--build"


def _receipt(build_dir: Path, record: dict) -> None:
    """One JSON line beside the build and one in the work root (the flake rate over time). Never fails the build."""
    line = json.dumps(record, sort_keys=True) + "\n"
    try:
        from .context import resolve_work_root
        targets = [build_dir / RETRY_RECEIPT, resolve_work_root() / HOST_RETRY_RECEIPT]
    except Exception:  # noqa: BLE001 - a receipt must never break a build
        targets = [build_dir / RETRY_RECEIPT]
    for target in targets:
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            with open(target, "a", encoding="utf-8") as handle:
                handle.write(line)
        except OSError:
            pass


def _stream(argv: list[str], cwd: Path) -> tuple[int, str, str]:
    """Run argv, passing its output through unchanged while looking for a compiler signal crash.
    Returns (exit code, signal name or "", the failed unit)."""
    crash, unit, current = "", "", ""
    process = subprocess.Popen(argv, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    assert process.stdout is not None
    try:
        for raw in process.stdout:
            text = raw.decode("utf-8", errors="replace")
            sys.stdout.write(text)
            sys.stdout.flush()
            failed = _FAILED_UNIT.match(text)
            if failed:
                current = failed.group("unit")
            found = _CRASH_LINE.search(text)
            if found and not crash:
                crash, unit = _SIGNALS[found.group("a") or found.group("b") or found.group("c")], current
        return process.wait(), crash, unit
    except BaseException:
        process.kill()  # as subprocess.run does when the parent is interrupted: no orphaned build
        process.wait()
        raise
    finally:
        process.stdout.close()


def run_checked(argv: list[str], cwd: Path) -> None:
    """Run a checked build subprocess without changing its arguments or cwd; its output reaches the build logs.

    A `cmake --build` whose compiler died from a signal is run once more (the build is incremental, so only what did
    not finish is redone) and both attempts are recorded; a real compile error, an interrupt or any other command
    is never retried."""
    if not _is_build(argv):
        run_logged(argv, cwd=cwd, check=True)
        return
    build_dir = Path(argv[2])
    code, signal_name, unit = _stream(argv, cwd)
    if code == 0:
        return
    if not signal_name:
        raise subprocess.CalledProcessError(code, argv)
    stamp = lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())  # noqa: E731
    base = {"build_dir": str(build_dir), "signal": signal_name, "unit": unit, "host": socket.gethostname()}
    _receipt(build_dir, {**base, "event": "crash", "attempt": 1, "at": stamp()})
    sys.stdout.write(f"bigcherry: the compiler died from {signal_name} in {unit or 'a unit'} (a random toolchain "
                     "crash, not a compile error); retrying cmake --build once (attempt 2 of 2)\n")
    sys.stdout.flush()
    code, again, _unit = _stream(argv, cwd)
    _receipt(build_dir, {**base, "event": "retry", "attempt": 2, "at": stamp(), "result": "ok" if code == 0 else "failed",
                         "crashed_again": bool(again)})
    if code != 0:
        raise subprocess.CalledProcessError(code, argv)
