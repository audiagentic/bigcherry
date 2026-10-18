"""Mechanics test for rad0005_mp4_ffmpeg6_writer against the radiance 1.8.0 fixture (v1.8.0, 7a7707a)."""

from __future__ import annotations

import importlib.util
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "tools"))

from bigcherry.patcher import apply_all  # noqa: E402

PACKAGE = REPO / "engines" / "radiance" / "patches" / "rad0005_mp4_ffmpeg6_writer" / "patch.py"
FIXTURE = REPO / "engines" / "radiance" / "tests" / "fixtures" / "7a7707a"
TARGET = "core/mm/mp4.cpp"


def _load():
    spec = importlib.util.spec_from_file_location("radiance_rad0005", PACKAGE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return [module.PATCH]


class TestRad0005(unittest.TestCase):
    def _tree(self):
        temp = tempfile.TemporaryDirectory()
        root = Path(temp.name)
        shutil.copytree(FIXTURE, root, dirs_exist_ok=True)
        return temp, root

    def test_applies_once_and_is_idempotent(self):
        temp, root = self._tree()
        try:
            first = apply_all(_load(), root)
            self.assertTrue(all(r.ok for r in first), first)
            self.assertTrue(all(e.status == "applied" for r in first for e in r.results), first)
            text = (root / TARGET).read_text(encoding="utf-8")
            self.assertIn("#if LIBAVFORMAT_VERSION_MAJOR >= 61", text)
            self.assertIn("int mem_write(void* op, uint8_t* buf, int n) {", text)
            self.assertEqual(text.count("int mem_write("), 2)
            after_first = text
            second = apply_all(_load(), root)
            self.assertTrue(all(e.status == "already-applied" for r in second for e in r.results), second)
            self.assertEqual((root / TARGET).read_text(encoding="utf-8"), after_first)
        finally:
            temp.cleanup()

    def test_fails_closed_when_the_callback_moves(self):
        temp, root = self._tree()
        try:
            path = root / TARGET
            path.write_text(path.read_text(encoding="utf-8").replace("int mem_write(", "int mem_write2("), encoding="utf-8")
            before = path.read_bytes()
            results = apply_all(_load(), root)
            self.assertTrue(any(not r.ok for r in results), results)
            self.assertEqual(path.read_bytes(), before)
        finally:
            temp.cleanup()


if __name__ == "__main__":
    unittest.main()
