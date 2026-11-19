"""The serving component's callback-boundary source tripwire, run as a platform test (the engine component is C++ only)."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import serving_callback_guard as guard


def test_scanner_recognises_its_own_patterns():
    guard.self_test()


def test_serving_sources_have_no_unguarded_callbacks():
    assert guard.check(guard.ROOT) == []
