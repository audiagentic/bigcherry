"""Compatibility wrapper for the HI24 slice-A lab verifier."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_IMPL = Path(__file__).resolve().parent / "lab" / "hi24-slice-a" / "verify_slice_a.py"
_SPEC = importlib.util.spec_from_file_location("bigcherry_hi24_verify_slice_a", _IMPL)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"cannot load slice-A verifier: {_IMPL}")
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)
globals().update(
    {name: value for name, value in vars(_MODULE).items() if not name.startswith("__")}
)

if __name__ == "__main__":
    raise SystemExit(_MODULE.main())
