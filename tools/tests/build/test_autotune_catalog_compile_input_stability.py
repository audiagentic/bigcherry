"""HI82 item-9 follow-up: autotune_catalog's generated compile inputs must
be byte-stable across repeated `bigcherry generate` calls with unchanged
content, or Ninja treats every downstream HIP object/DLL as perpetually
dirty (found via a real end-to-end campaign rerun on gfx1100 producing a
different effective_build_id/runtime_bundle_hash on every invocation;
diagnosed with GPT, req_cc5af49494fe457a)."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bigcherry.tuning.catalog import _write_compile_input_if_changed  # noqa: E402


class WriteCompileInputIfChangedTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.path = Path(self._tmp.name) / "generated.inc"

    def tearDown(self):
        self._tmp.cleanup()

    def test_creates_file_when_absent(self):
        _write_compile_input_if_changed(self.path, "content-v1\n")
        self.assertEqual(self.path.read_text(encoding="utf-8"), "content-v1\n")

    def test_preserves_mtime_when_content_unchanged(self):
        _write_compile_input_if_changed(self.path, "content-v1\n")
        before = self.path.stat().st_mtime_ns

        # A real second `bigcherry generate` invocation against the exact
        # same manifest -- content is byte-identical, only the call
        # happened again.
        _write_compile_input_if_changed(self.path, "content-v1\n")
        after = self.path.stat().st_mtime_ns

        self.assertEqual(before, after)
        self.assertEqual(self.path.read_text(encoding="utf-8"), "content-v1\n")

    def test_rewrites_when_content_actually_changes(self):
        _write_compile_input_if_changed(self.path, "content-v1\n")
        _write_compile_input_if_changed(self.path, "content-v2\n")
        self.assertEqual(self.path.read_text(encoding="utf-8"), "content-v2\n")


if __name__ == "__main__":
    unittest.main()
