from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bigcherry import compare_tunes  # noqa: E402


class CompareTunesTests(unittest.TestCase):
    def test_midranks_and_constant_rank(self):
        self.assertEqual(compare_tunes._midranks({"a": 1, "b": 1, "c": 3}),
                         {"a": 1.5, "b": 1.5, "c": 3.0})
        self.assertIsNone(compare_tunes.spearman({"a": 1, "b": 1}, {"a": 2, "b": 3}))

    def test_signature_join_and_weighted_change(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            before, after, record = root / "before", root / "after", root / "record"
            header = {"kind": "header", "source_revision": "s", "manifest_hash": "m", "hardware": "h"}
            def result(winner: str, a: float, b: float):
                return {"kind": "result", "signature": "x", "winner": winner,
                        "hardware": "h",
                        "candidates": [{"name": "a", "status": "ok", "effective_us": a},
                                       {"name": "b", "status": "ok", "effective_us": b}]}
            before.write_text("\n".join(map(json.dumps, [header, result("a", 10, 20)])) + "\n")
            after.write_text("\n".join(map(json.dumps, [header, result("b", 20, 9)])) + "\n")
            record.write_text(json.dumps({"kind": "observation", "signature": "x", "calls": 5}) + "\n")
            report = compare_tunes.compare(before, after, record=record)
            self.assertEqual(report["common_signatures"], 1)
            self.assertEqual(report["winner_agreement"], 0)
            self.assertEqual(report["winner_agreement_calls_pct"], 0.0)
            self.assertAlmostEqual(report["call_weighted_time_change_pct"], -10.0)

    def test_partial_weighted_comparison_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            before, after, record = root / "before", root / "after", root / "record"
            header = {"kind": "header", "source_revision": "s", "manifest_hash": "m", "hardware": "h"}
            good = {"kind": "result", "signature": "x", "winner": "a",
                    "hardware": "h",
                    "candidates": [{"name": "a", "status": "ok", "effective_us": 10}]}
            missing = {"kind": "result", "signature": "x", "winner": "a",
                       "hardware": "h", "candidates": []}
            before.write_text("\n".join(map(json.dumps, [header, good])) + "\n")
            after.write_text("\n".join(map(json.dumps, [header, missing])) + "\n")
            record.write_text(json.dumps({"kind": "observation", "signature": "x", "calls": 5}) + "\n")
            report = compare_tunes.compare(before, after, record=record)
            self.assertIsNone(report["call_weighted_time_change_pct"])
            self.assertEqual(report["call_coverage_pct"], 0.0)

    def test_cross_hardware_comparison_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            before, after = root / "before", root / "after"
            header = {"kind": "header", "source_revision": "s", "manifest_hash": "m"}
            def row(hardware: str):
                return {"kind": "result", "signature": "x", "hardware": hardware,
                        "winner": "a", "candidates": []}
            before.write_text("\n".join(map(json.dumps, [header, row("a")])) + "\n")
            after.write_text("\n".join(map(json.dumps, [header, row("b")])) + "\n")
            with self.assertRaises(compare_tunes.CompareError):
                compare_tunes.compare(before, after)
