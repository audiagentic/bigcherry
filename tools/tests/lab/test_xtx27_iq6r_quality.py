"""Card-free VTP06 sparse BF16-vs-IQ6R KLD gate regression cases."""

from __future__ import annotations

import importlib.util
import json
import math
import struct
import tempfile
import unittest
from collections import Counter
from pathlib import Path

LAB = Path(__file__).resolve().parents[2] / "lab" / "radiance"


def module(name):
    spec = importlib.util.spec_from_file_location(name, LAB / (name + ".py"))
    assert spec and spec.loader
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


corpus = module("xtx27_calib_corpus")
quality = module("xtx27_iq6r_quality")


class QualityGateTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.checkpoint = root / "checkpoint"
        self.checkpoint.mkdir()
        (self.checkpoint / "tokenizer.json").write_text('{"fake":"tok"}')
        (self.checkpoint / "tokenizer_config.json").write_text('{"fake":"tokcfg"}')
        source = root / "source.jsonl"
        categories = ["language", "code", "math", "multilingual", "long_context"]
        examples = []
        for split, n in (("calibration", 200), ("holdout", 50)):
            for idx in range(n):
                category = categories[idx % len(categories)]
                prompt = f"{split} synthetic {category} {idx}: " + "test text " * 25
                if category == "long_context" and idx == 4:
                    prompt += "long-context source text " * 500
                examples.append({
                    "id": f"{split}-{idx}", "source": f"source-{idx%3}",
                    "prompt": prompt, "license": "SYNTHETIC_TEST_FIXTURE",
                    "category": category, "split": split,
                })
        source.write_text("".join(json.dumps(row) + "\n" for row in examples))
        self.bundle = root / "bundle"
        corpus.prepare(source, self.checkpoint, self.bundle, "candidate")
        self.reference = root / "reference"
        self.reference.mkdir()
        holdout = self.bundle / "holdout.jsonl"
        sources = [json.loads(row)["source"] for row in holdout.read_text().splitlines()]
        counts = Counter(sources)
        self.ref_doc = {
            "format": "FAKE_TOP128_REFERENCE_TEST",
            "model": str(self.checkpoint),
            "top": 128,
            "corpus": str(holdout),
            "rows": 50000,
            "docs": [
                {"source": name, "tokens": 1001, "score_from": 0, "row0": i*1000}
                for i, name in enumerate(sources)
            ],
        }
        (self.reference / "kld.json").write_text(json.dumps(self.ref_doc))
        self.candidate = root / "candidate-iq6r.rad"
        self.candidate.write_bytes(b"synthetic fake binary")
        self.report_path = root / "report.json"
        self.all_stats = {
            "positions": 50000,
            "kld": {"mean": 0.001, "p99": 0.001, "p99_9": 0.001},
            "ppl": {"ratio": math.exp(0.001)},
            "top1_agreement": 1.0,
        }
        self.report = {
            "model": str(self.candidate), "reference": str(self.checkpoint),
            "ref_dir": str(self.reference), "ref_form": "top-128 + tail",
            "corpus": str(holdout), "docs": 50,
            "all": self.all_stats,
            "by_source": {
                name: {
                    "positions": n * 1000,
                    "kld": {"mean": 0.001},
                    "top1_agreement": 1.0,
                } for name, n in counts.items()
            },
        }
        self.report_path.write_text(json.dumps(self.report))
        (root / "report.json.rows").write_bytes(
            struct.pack("<ffff", 0.001, 2.001, 2.0, 1.0) * 50000
        )
        self.policy = LAB / "xtx27_iq6r_quality_policy_proposed.json"
        self.output = root / "gate.json"

    def gate(self):
        return quality.gate(
            self.bundle, self.checkpoint, self.reference, self.report_path,
            self.candidate, self.policy, self.output,
        )

    def test_pass_stays_provisional(self):
        result = self.gate()
        self.assertEqual(result["status"],
                         "PROVISIONAL_THRESHOLDS_PASS_NOT_PRODUCTION_APPROVAL")
        self.assertFalse(result["quality_approved"])
        self.assertFalse(result["production_approved"])
        self.assertEqual(result["metrics"]["positions"], 50000)
        self.assertEqual(result["failures"], [])
        with self.assertRaisesRegex(ValueError, "overwrite"):
            self.gate()

    def test_structural_pilot_is_rejected(self):
        doc = json.loads((self.bundle / "bundle.json").read_text())
        doc["overview"]["mode"] = "pilot"
        (self.bundle / "bundle.json").write_text(json.dumps(doc))
        with self.assertRaisesRegex(ValueError, "candidate, not pilot"):
            self.gate()

    def test_mismatched_reference_source_order_rejected(self):
        self.ref_doc["docs"][0]["source"] = "unknown-source"
        (self.reference / "kld.json").write_text(json.dumps(self.ref_doc))
        with self.assertRaisesRegex(ValueError, "source, or offsets drift"):
            self.gate()

    def test_wrong_reference_form_rejected(self):
        self.report["ref_form"] = "full vocabulary"
        self.report_path.write_text(json.dumps(self.report))
        with self.assertRaisesRegex(ValueError, "top-128"):
            self.gate()

    def test_mismatched_report_vs_rows_rejected(self):
        self.report["all"]["kld"]["mean"] = 0.02
        self.report_path.write_text(json.dumps(self.report))
        with self.assertRaisesRegex(ValueError, "does not match .rows"):
            self.gate()

    def test_numeric_threshold_rejection_does_not_claim_quality(self):
        self.report["all"]["kld"]["p99"] = 1.0
        self.report_path.write_text(json.dumps(self.report))
        result = self.gate()
        self.assertEqual(result["status"], "QUALITY_THRESHOLD_REJECTED")
        self.assertIn("p99 KL", result["failures"])
        self.assertFalse(result["quality_approved"])

    def test_wrong_holdout_or_source_census_rejected(self):
        self.report["by_source"].pop("source-0")
        self.report_path.write_text(json.dumps(self.report))
        with self.assertRaisesRegex(ValueError, "coverage missing"):
            self.gate()

    def test_wrong_source_token_count_rejected(self):
        self.report["by_source"]["source-0"]["positions"] += 1
        self.report_path.write_text(json.dumps(self.report))
        with self.assertRaisesRegex(ValueError, "per-source KLD scored positions"):
            self.gate()

    def test_nonfinite_rows_rejected(self):
        path = self.report_path.with_suffix(".json.rows")
        data = bytearray(path.read_bytes())
        data[:4] = struct.pack("<f", float("nan"))
        path.write_bytes(data)
        with self.assertRaisesRegex(ValueError, "nonfinite KLD row"):
            self.gate()

    def test_same_candidate_as_reference_rejected(self):
        self.report["model"] = str(self.checkpoint)
        self.report_path.write_text(json.dumps(self.report))
        with self.assertRaisesRegex(ValueError, "model/reference"):
            self.gate()


if __name__ == "__main__":
    unittest.main()
