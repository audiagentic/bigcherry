"""Card-free adaptive batch and sealed-predecessor tests."""
from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

TOOLS = Path(__file__).resolve().parents[2] / "lab/radiance"
sys.path.insert(0, str(TOOLS))
PATH = TOOLS / "iq6r_v4_adaptive.py"
spec = importlib.util.spec_from_file_location("iq6r_v4_adaptive", PATH)
assert spec and spec.loader
adaptive = importlib.util.module_from_spec(spec)
spec.loader.exec_module(adaptive)


class AdaptiveTests(unittest.TestCase):
    def test_batch_selector(self):
        self.assertEqual(adaptive.choose_batch([]), 1)
        self.assertEqual(adaptive.choose_batch([{"elapsed_s": 950}]), 1)
        self.assertEqual(adaptive.choose_batch([{"elapsed_s": 800}]), 2)
        self.assertEqual(adaptive.choose_batch([{"elapsed_s": 700}, {"elapsed_s": 500}]), 2)
        self.assertEqual(adaptive.choose_batch([{"elapsed_s": 580}, {"elapsed_s": 570}]), 4)
        self.assertEqual(adaptive.choose_batch([{"elapsed_s": 550}, {"elapsed_s": 1801}]), 1)

    def test_exact_count_recipe_map(self):
        sources = adaptive.source_recipes()
        self.assertEqual(len(sources), 289)
        for after in (4, 5, 6, 8, 292):
            self.assertEqual(sources[after - 4].count(b" gptq "), after)
        self.assertEqual(sources[-1], (
            adaptive.PINNED / "data/recipes/q38-27b-iq6r-v4.recipe").read_bytes())

    def test_first_job_inherits_stage_one(self):
        sources = [b"test"] * 289
        prior = {"sha256": "stage-one", "gptq_cumulative": 4}
        with (
            patch.object(adaptive, "micro_checkpoint", return_value=(Path("/tmp/s1.rad"), prior)),
            patch.object(adaptive, "adaptive_history", return_value=[]),
        ):
            result = adaptive.next_job(sources)
        self.assertEqual(result["gptq_after"], 5)
        self.assertEqual(result["batch"], 1)

    def test_adaptive_history_fails_closed_on_gap(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "adaptive-0002.json").write_text("{}")
            with (
                patch.object(adaptive, "DIR", root),
                self.assertRaisesRegex(RuntimeError, "unexpected adaptive"),
            ):
                adaptive.adaptive_history([b"stage"] * 289, "prior", 4)

    def test_adaptive_history_fails_closed_on_tampered_recipe(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            record = {
                "stage": 1, "gptq_cumulative": 5,
                "predecessor_sha256": "prior",
                "recipe_sha256": "wrong", "sha256": "abc", "bytes": 123,
            }
            (root / "adaptive-0001.json").write_text(json.dumps(record))
            with (
                patch.object(adaptive, "DIR", root),
                self.assertRaisesRegex(RuntimeError, "identity mismatch"),
            ):
                adaptive.adaptive_history([b"a"] * 289, "prior", 4)

    def test_free_space_guard_refuses_insufficient_space(self):
        stat = type("Stat", (), {"f_bavail": 100, "f_frsize": 4096})()
        with (
            patch.object(adaptive.os, "statvfs", return_value=stat),
            self.assertRaisesRegex(RuntimeError, "disk free-space guard"),
        ):
            adaptive.free_space_guard(Path("/tmp"))


if __name__ == "__main__":
    unittest.main()
