"""1337_moe_expert_caching is superseded: the pinned revision carries upstream #29887 itself."""

from __future__ import annotations

import sys
import tomllib
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from bigcherry.core import paths  # noqa: E402
from bigcherry.patch.pinned_source import read_pinned  # noqa: E402

_REPO = Path(__file__).resolve().parents[3]
_PKG = _REPO / "engines/llamacpp/patches/1337_moe_expert_caching"
_V = paths.llama_root()


class Patch1337Superseded(unittest.TestCase):
    def test_state_is_superseded_in_both_places(self):
        meta = tomllib.loads((_PKG / "patch.toml").read_text(encoding="utf-8"))
        self.assertEqual(meta["state"], "superseded")
        self.assertIn('STATE = "superseded"', (_PKG / "patch.py").read_text(encoding="utf-8"))

    @unittest.skipUnless(_V.is_dir(), "pinned vendor checkout not present")
    def test_pin_has_the_upstream_cache(self):
        # the files the backport created exist at the pin, with upstream's option
        self.assertIn("llama_moe_cache", read_pinned(_V / "src/llama-moe-cache.cpp"))
        self.assertIn("llama_moe_cache", read_pinned(_V / "src/llama-moe-cache.h"))
        self.assertIn('"--moe-cache-mib"', read_pinned(_V / "common/arg.cpp"))


if __name__ == "__main__":
    unittest.main()
