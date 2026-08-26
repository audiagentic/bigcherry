import json
import tempfile
import unittest
from pathlib import Path

from bigcherry.transform_records import (
    TransformRecordError,
    load_transform_records,
    validate_transform_record,
)


def _common(kind="transform-attempt"):
    return {
        "kind": kind,
        "source_signature": "a" * 32,
        "hardware_provenance": {"digest": "b" * 32, "architecture": "gfx1201"},
        "build_provenance": {
            "source_revision": "c" * 40,
            "manifest_hash": "d" * 64,
            "build_descriptor_hash": "e" * 64,
        },
        "evidence_references": ["run-001/stdout", "run-001/correctness.json"],
        "reason": "candidate was measured",
    }


class TestTransformRecords(unittest.TestCase):
    def test_valid_attempt_normalizes_identity_and_provenance(self):
        row = _common()
        row.update({
            "transformation": {"id": 1, "name": "transpose", "source": "PREDEFINED"},
            "outcome": "SUCCESS",
            "transformed_signature": "f" * 32,
        })
        result = validate_transform_record(row)
        self.assertEqual(result["outcome"], "success")
        self.assertEqual(result["transformation"]["source"], "predefined")
        self.assertEqual(result["build_provenance"]["source_revision"], "c" * 40)

    def test_valid_gap_requires_tried_transformations(self):
        row = _common("transform-gap")
        row.update({
            "transformations_tried": [
                {"id": 1, "name": "transpose", "reason": "shape mismatch"},
            ],
            "pattern": "large batched F32 matmul",
            "calls": 3,
            "est_bytes": 4096,
        })
        result = validate_transform_record(row)
        self.assertEqual(result["outcome"], "gap")
        self.assertEqual(result["calls"], 3)

    def test_missing_provenance_is_rejected(self):
        row = _common()
        del row["build_provenance"]
        row["transformation"] = {"id": 1, "name": "transpose", "source": "predefined"}
        row["outcome"] = "rejected"
        with self.assertRaisesRegex(TransformRecordError, "build_provenance"):
            validate_transform_record(row)

    def test_duplicate_evidence_is_rejected_case_insensitively(self):
        row = _common()
        row["evidence_references"] = ["Evidence/A", "evidence/a"]
        row["transformation"] = {"id": 1, "name": "transpose", "source": "predefined"}
        row["outcome"] = "rejected"
        with self.assertRaisesRegex(TransformRecordError, "unique"):
            validate_transform_record(row)

    def test_gap_duplicate_transform_ids_are_rejected(self):
        row = _common("transform-gap")
        row.update({
            "transformations_tried": [
                {"id": 1, "name": "a", "reason": "no"},
                {"id": 1, "name": "b", "reason": "no"},
            ],
            "pattern": "pattern",
        })
        with self.assertRaisesRegex(TransformRecordError, "duplicates"):
            validate_transform_record(row)

    def test_loader_requires_header_and_loads_both_kinds(self):
        attempt = _common()
        attempt.update({
            "transformation": {"id": 1, "name": "transpose", "source": "predefined"},
            "outcome": "rejected",
        })
        gap = _common("transform-gap")
        gap.update({
            "transformations_tried": [{"id": 1, "name": "transpose", "reason": "no"}],
            "pattern": "pattern",
        })
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "transforms.jsonl"
            lines = [
                {"kind": "header", "transform_schema_version": 1},
                attempt,
                gap,
            ]
            path.write_text("\n".join(json.dumps(line) for line in lines) + "\n", encoding="utf-8")
            loaded = load_transform_records(path)
        self.assertEqual([row["kind"] for row in loaded], ["transform-attempt", "transform-gap"])

    def test_loader_rejects_duplicate_header_and_bad_json(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.jsonl"
            path.write_text(
                '{"kind":"header","transform_schema_version":1}\n'
                '{"kind":"header","transform_schema_version":1}\n', encoding="utf-8")
            with self.assertRaisesRegex(TransformRecordError, "duplicate header"):
                load_transform_records(path)
            path.write_text('{"kind":"header","transform_schema_version":1}\n{', encoding="utf-8")
            with self.assertRaisesRegex(TransformRecordError, "invalid JSON"):
                load_transform_records(path)


if __name__ == "__main__":
    unittest.main()
