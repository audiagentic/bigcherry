"""Offline negative and alignment tests for pinned public/golden quality data."""
from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

PATH = Path(__file__).resolve().parents[2] / "lab/radiance/iq6r_v4_prepare_quality.py"
spec = importlib.util.spec_from_file_location("iq6r_v4_prepare_quality", PATH)
assert spec and spec.loader
quality = importlib.util.module_from_spec(spec)
spec.loader.exec_module(quality)


def fixture() -> tuple[list[dict], dict]:
    corpus = [
        {"source": "a", "category": "code", "tokens": 3, "prompt": "Hello"},
        {"source": "b", "category": "math", "tokens": 4, "prompt": "World"},
    ] * 298
    entries = []
    row = 0
    for item in corpus:
        entries.append({"source": item["source"], "tokens": item["tokens"], "score_from": 0, "row0": row})
        row += item["tokens"] - 1
    return corpus, {"docs": entries, "top": 512, "n_vocab": 248320, "rows": row}


class QualityAuditTest(unittest.TestCase):
    def test_rejects_bad_corpus_count(self):
        corpus, golden = fixture()
        with self.assertRaisesRegex(ValueError, "corpus document count"):
            quality.validate_docs(corpus[:-1], golden)

    def test_rejects_changed_golden_score_offset(self):
        corpus, golden = fixture()
        golden["rows"] = quality.EXPECTED_ROWS
        golden["docs"][1]["row0"] = 0
        with self.assertRaisesRegex(ValueError, "row-offset mismatch"):
            quality.validate_docs(corpus, golden)

    def test_rejects_changed_document_source(self):
        corpus, golden = fixture()
        golden["rows"] = quality.EXPECTED_ROWS
        golden["docs"][0]["source"] = "wrong"
        with self.assertRaisesRegex(ValueError, "source/token mismatch"):
            quality.validate_docs(corpus, golden)

    def test_rejects_wrong_total_rows(self):
        corpus, golden = fixture()
        with self.assertRaisesRegex(ValueError, "scoring rows"):
            quality.validate_docs(corpus, golden)

    def test_no_false_bf16_pass_assumptions(self):
        self.assertEqual(quality.EXPECTED_ROWS, 1_694_025)
        self.assertEqual(quality.EXPECTED_TOP, 512)


if __name__ == "__main__":
    unittest.main()
