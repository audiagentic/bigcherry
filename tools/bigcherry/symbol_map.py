"""Kernel symbol demangling, shared by HI35 (kernel-fraction) and HI09b.

rocprofv3 normally emits a demangled ``Kernel_Name``, but not always, and
HI09b's ``-Rpass-analysis=kernel-resource-usage`` remarks are always mangled.
One module, one table: best-effort demangle with the mangled name as its own
fallback.

The fallback is not a degraded path: the family patterns used by the consumers
are C++ identifiers, and Itanium mangling embeds them verbatim. So a mangled
name still classifies correctly; demangling only improves the report.
"""

from __future__ import annotations

import shutil
import subprocess

# Tools to try, in order. llvm-cxxfilt ships with ROCm and is on PATH wherever
# a build happened; c++filt (binutils) is the common fallback.
_DEMANGLE_TOOLS = ("llvm-cxxfilt", "c++filt")

_resolved: list[str] = []
_symbol_cache: dict[str, str] = {}


def _available_tools() -> list[str]:
    """Locate the demangling tools once per process."""
    global _resolved
    if not _resolved:
        _resolved = [exe for tool in _DEMANGLE_TOOLS if (exe := shutil.which(tool))]
    return _resolved


def demangle(symbol: str) -> str:
    """Best-effort demangle; returns the input unchanged when nothing works.

    Raises nothing: a missing tool, a timeout, or an empty result all fall
    through to the mangled name, which still carries the C++ identifiers the
    family patterns match on.
    """
    if not symbol:
        return ""
    cached = _symbol_cache.get(symbol)
    if cached is not None:
        return cached
    for exe in _available_tools():
        try:
            proc = subprocess.run(
                [exe],
                input=symbol.encode("utf-8"),
                capture_output=True,
                timeout=10,
            )
        except (OSError, subprocess.SubprocessError):
            continue
        if proc.returncode != 0:
            continue
        text = proc.stdout.decode("utf-8", errors="replace").strip()
        if text:
            _symbol_cache[symbol] = text
            return text
    _symbol_cache[symbol] = symbol
    return symbol


def demangle_many(symbols: list[str]) -> dict[str, str]:
    """Demangle a batch, keyed by the original symbol. Identity-preserving."""
    return {symbol: demangle(symbol) for symbol in symbols}
