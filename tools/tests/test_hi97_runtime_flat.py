"""HI97: runtime-flat (HI29) .transforms.jsonl artifacts must ingest through
the offline stack without TransformRecordError.

The C++ tuner flush emits a flat schema (per-row hardware digest, per-file
build source_revision+manifest_hash); the nested evidence schema (HI33)
requires architecture, build_descriptor_hash, and evidence_references, which
the runtime does not emit.  The agreed boundary: the artifact loader
dispatches on the header; the strict nested validator stays strict; missing
identity is NEVER fabricated (None + provenance_complete=False), and SQLite
ingestion resolves it from the authoritative DB, failing closed on zero or
ambiguous builds.
"""

import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from bigcherry.analyze_gaps import GapAnalysisError, analyze_gap_file, analyze_gaps
from bigcherry.inventory import RecordError
from bigcherry.transform_loader import load_transforms
from bigcherry.transform_records import (
    TransformRecordError,
    load_artifact_records,
    load_runtime_transform_records,
    load_transform_records,
)

ROOT = Path(__file__).resolve().parents[2]
SCHEMA = ROOT / "sql" / "dispatch-db.sql"

REVISION = "a" * 40
MANIFEST = "b" * 64
DESCRIPTOR = "c" * 64
HW = "d" * 32
SIG = "e" * 32


def flat_header():
    # Mirrors the C++ fprintf header in hip-autotune-tuner.cu exactly.
    return {"kind": "header", "artifact_version": 1,
            "source_revision": REVISION, "manifest_hash": MANIFEST}


def flat_attempt(**overrides):
    # Mirrors the C++ attempt-row fprintf exactly.
    row = {
        "kind": "transform-attempt",
        "original_sig": SIG,
        "hardware": HW,
        "transformation_id": 1,
        "transformation_name": "mmvq_to_mmq",
        "source": "predefined",
        "original_native_family": "mmvq",
        "result": "rejected",
        "rejection_reason": "no faster route",
        "transformed_winner": "mmq:tuned:v2",
        "original_us": 123.456,
        "transformed_us": 100.0,
        "improvement_pct": 19.0,
        "nmse": 0.001,
        "max_abs_error": 0.002,
    }
    row.update(overrides)
    return row


def flat_gap(**overrides):
    # Mirrors the C++ gap-row fprintf exactly.
    row = {
        "kind": "transform-gap",
        "sig": SIG,
        "hardware": HW,
        "native_family": "mmvq",
        "est_bytes": 4096,
        "transformations_tried": [{"id": 1, "reason": "no faster route"}],
    }
    row.update(overrides)
    return row


def nested_gap():
    return {
        "kind": "transform-gap",
        "source_signature": SIG,
        "hardware_provenance": {"digest": HW, "architecture": "gfx1201"},
        "build_provenance": {
            "source_revision": REVISION,
            "manifest_hash": MANIFEST,
            "build_descriptor_hash": DESCRIPTOR,
        },
        "pattern": "q4_k-to-f32",
        "native_family": "mmvq",
        "reason": "gap reason",
        "evidence_references": ["evidence://transform/gap-1"],
        "transformations_tried": [{"id": 1, "name": "mmvq_to_mmq", "reason": "no"}],
    }


class RuntimeFlatLoaderTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="bigcherry_hi97_")
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def write(self, name, *records):
        path = self.root / name
        with path.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record) + "\n")
        return path

    def flat_artifact(self):
        return self.write(
            "transforms.jsonl",
            flat_header(),
            flat_attempt(),
            flat_attempt(result="success", rejection_reason="",
                         transformed_winner="mmq:tuned:v3", transformation_id=2,
                         transformation_name="batch_for_mmvf", source="discovered"),
            flat_gap(),
        )

    def test_flat_artifact_parses_with_explicit_incomplete_provenance(self):
        records = load_runtime_transform_records(self.flat_artifact())
        self.assertEqual([r["kind"] for r in records],
                         ["transform-attempt", "transform-attempt", "transform-gap"])
        rejected, success, gap = records
        for record in records:
            self.assertEqual(record["source_schema"], "runtime-flat-v1")
            self.assertFalse(record["provenance_complete"])
            self.assertEqual(record["hardware_provenance"],
                             {"digest": HW, "architecture": None})
            self.assertEqual(record["build_provenance"],
                             {"source_revision": REVISION, "manifest_hash": MANIFEST,
                              "build_descriptor_hash": None})
            self.assertEqual(record["evidence_references"], [])
        self.assertEqual(rejected["outcome"], "rejected")
        self.assertEqual(rejected["reason"], "no faster route")
        self.assertEqual(rejected["transformation"],
                         {"id": 1, "name": "mmvq_to_mmq", "source": "predefined"})
        self.assertIsNone(rejected["transformed_signature"])
        self.assertEqual(rejected["transformed_winner"], "mmq:tuned:v2")
        self.assertEqual(success["outcome"], "success")
        self.assertEqual(success["reason"], "")
        self.assertEqual(success["transformation"]["source"], "discovered")
        self.assertEqual(gap["outcome"], "gap")
        self.assertEqual(gap["pattern"], "unclassified-runtime-gap")
        self.assertEqual(gap["transformation"]["tried"],
                         [{"id": 1, "name": None, "reason": "no faster route"}])
        self.assertEqual(gap["est_bytes"], 4096)

    def test_rejected_attempt_without_reason_fails_closed(self):
        path = self.write("bad.jsonl", flat_header(),
                          flat_attempt(result="rejected", rejection_reason=""))
        with self.assertRaisesRegex(TransformRecordError, "rejection_reason"):
            load_runtime_transform_records(path)

    def test_nested_rejected_attempt_still_requires_reason(self):
        # The strict validator is untouched: a nested rejected attempt with an
        # empty reason still fails exactly as before this change.
        attempt = {
            "kind": "transform-attempt",
            "source_signature": SIG,
            "hardware_provenance": {"digest": HW, "architecture": "gfx1201"},
            "build_provenance": {"source_revision": REVISION, "manifest_hash": MANIFEST,
                                 "build_descriptor_hash": DESCRIPTOR},
            "transformation": {"id": 1, "name": "x", "source": "predefined"},
            "outcome": "rejected",
            "reason": "",
            "evidence_references": ["evidence://1"],
        }
        with self.assertRaises(TransformRecordError):
            from bigcherry.transform_records import validate_transform_record
            validate_transform_record(attempt)

    def test_artifact_loader_dispatches_by_header(self):
        flat = self.flat_artifact()
        nested = self.write("nested.jsonl",
                            {"kind": "header", "transform_schema_version": 1},
                            nested_gap())
        self.assertEqual(load_artifact_records(flat), load_runtime_transform_records(flat))
        self.assertEqual(load_artifact_records(nested), load_transform_records(nested))
        # A nested artifact is NOT a runtime-flat artifact, and vice versa.
        with self.assertRaisesRegex(TransformRecordError, "not a runtime-flat"):
            load_runtime_transform_records(nested)


class RuntimeFlatSqliteIngestionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="bigcherry_hi97_sql_")
        self.root = Path(self.tmp.name)
        self.db = self.root / "dispatch.sqlite"
        conn = sqlite3.connect(self.db)
        try:
            conn.executescript(SCHEMA.read_text(encoding="utf-8"))
            conn.execute(
                "INSERT INTO build(source_revision, manifest_hash, signature_schema, "
                "hardware_schema, variant_set, build_descriptor_hash) VALUES (?,?,?,?,?,?)",
                (REVISION, MANIFEST, 1, 1, "inventory", DESCRIPTOR),
            )
            conn.execute(
                "INSERT INTO hardware(hardware_digest, architecture, architecture_code, "
                "wave_size, compute_units, feature_flags, canonical_json) "
                "VALUES (?,?,?,?,?,?,?)",
                (bytes.fromhex(HW), "gfx1201", 1201, 64, 80, 0, "{}"),
            )
            conn.execute(
                "INSERT INTO signature(signature_digest, base_digest, schema_version, op, "
                "src0_type, src1_type, dst_type, m, n, k, canonical_json) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (bytes.fromhex(SIG), bytes.fromhex(SIG), 1, "MUL_MAT",
                 "q4_k", "f32", "f32", 1, 8, 16, "{}"),
            )
            conn.execute(
                "INSERT INTO observation(build_id, hardware_id, signature_id, "
                "native_stable_name, calls, est_bytes) VALUES (1,1,1,?,?,?)",
                ("blas:native:v1", 17, 4096),
            )
            conn.commit()
        finally:
            conn.close()

    def tearDown(self):
        self.tmp.cleanup()

    def write(self, name, *records):
        path = self.root / name
        with path.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record) + "\n")
        return path

    def flat_artifact(self):
        return self.write(
            "transforms.jsonl",
            {"kind": "header", "artifact_version": 1,
             "source_revision": REVISION, "manifest_hash": MANIFEST},
            {"kind": "transform-attempt", "original_sig": SIG, "hardware": HW,
             "transformation_id": 1, "transformation_name": "mmvq_to_mmq",
             "source": "predefined", "original_native_family": "mmvq",
             "result": "rejected", "rejection_reason": "no faster route",
             "transformed_winner": "mmq:tuned:v2", "original_us": 123.456,
             "transformed_us": 100.0, "improvement_pct": 19.0,
             "nmse": 0.001, "max_abs_error": 0.002},
            {"kind": "transform-attempt", "original_sig": SIG, "hardware": HW,
             "transformation_id": 2, "transformation_name": "batch_for_mmvf",
             "source": "discovered", "original_native_family": "mmvq",
             "result": "success", "rejection_reason": "",
             "transformed_winner": "mmq:tuned:v3", "original_us": 100.0,
             "transformed_us": 90.0,
             "improvement_pct": 10.0, "nmse": 0.001, "max_abs_error": 0.002},
            {"kind": "transform-gap", "sig": SIG, "hardware": HW,
             "native_family": "mmvq", "est_bytes": 4096,
             "transformations_tried": [{"id": 1, "reason": "no faster route"}]},
        )

    def test_flat_artifact_ingests_end_to_end(self):
        # HI97's acceptance: a real-shape runtime (flat) transforms.jsonl
        # ingests into SQLite via load_transforms() with no TransformRecordError.
        result = load_transforms(self.flat_artifact(), self.db, SCHEMA)
        self.assertEqual(result, {"attempts": 2, "gaps": 1})
        conn = sqlite3.connect(self.db)
        try:
            attempts = conn.execute(
                "SELECT transformation_name, source, result, reason FROM transform_attempt "
                "ORDER BY transformation_id"
            ).fetchall()
            self.assertEqual(attempts,
                             [("mmvq_to_mmq", "predefined", "rejected", "no faster route"),
                              ("batch_for_mmvf", "discovered", "success", "")])
            gap = conn.execute(
                "SELECT pattern_description, native_family, calls, est_bytes "
                "FROM transform_gap"
            ).fetchone()
            self.assertEqual(gap, ("unclassified-runtime-gap", "mmvq", 17, 4096))
            # The build bound is the real DB build row (identity resolved, not
            # fabricated), and the gap observation join survived.
            self.assertEqual(conn.execute(
                "SELECT COUNT(*) FROM transform_attempt a JOIN build b "
                "ON a.build_id=b.build_id WHERE b.source_revision=?", (REVISION,)
            ).fetchone()[0], 2)
        finally:
            conn.close()

    def test_ambiguous_build_fails_closed(self):
        # Two builds sharing source_revision+manifest_hash with different
        # build identities: a flat record cannot bind one, and must not.
        conn = sqlite3.connect(self.db)
        conn.execute(
            "INSERT INTO build(source_revision, manifest_hash, signature_schema, "
            "hardware_schema, variant_set, build_descriptor_hash) VALUES (?,?,?,?,?,?)",
            (REVISION, MANIFEST, 1, 1, "workload-max", "f" * 64),
        )
        conn.commit()
        conn.close()
        with self.assertRaisesRegex(RecordError, "ambiguous build identity"):
            load_transforms(self.flat_artifact(), self.db, SCHEMA)

    def test_single_build_binds_despite_other_build_observation(self):
        # A DIFFERENT build (other source_revision) that observed the same
        # hardware/signature must not participate in build resolution.
        conn = sqlite3.connect(self.db)
        conn.execute(
            "INSERT INTO build(source_revision, manifest_hash, signature_schema, "
            "hardware_schema, variant_set, build_descriptor_hash) VALUES (?,?,?,?,?,?)",
            ("f" * 40, MANIFEST, 1, 1, "workload-max", "e" * 64),
        )
        conn.execute(
            "INSERT INTO observation(build_id, hardware_id, signature_id, "
            "native_stable_name, calls, est_bytes) VALUES (2,1,1,?,?,?)",
            ("blas:native:v1", 5, 256),
        )
        conn.commit()
        conn.close()
        result = load_transforms(self.flat_artifact(), self.db, SCHEMA)
        self.assertEqual(result, {"attempts": 2, "gaps": 1})

    def test_zero_matching_builds_fails(self):
        path = self.write(
            "orphan.jsonl",
            {"kind": "header", "artifact_version": 1,
             "source_revision": "9" * 40, "manifest_hash": MANIFEST},
            flat_gap(),
        )
        with self.assertRaisesRegex(RecordError, "does not match an existing build"):
            load_transforms(path, self.db, SCHEMA)


class RuntimeFlatAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="bigcherry_hi97_gap_")
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def write(self, name, *records):
        path = self.root / name
        with path.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record) + "\n")
        return path

    def test_analyze_gap_file_on_flat_artifact(self):
        # Same flush, same device, two signatures: one provenance namespace.
        path = self.write(
            "gaps.jsonl",
            {"kind": "header", "artifact_version": 1,
             "source_revision": REVISION, "manifest_hash": MANIFEST},
            flat_gap(),
            flat_gap(sig="1" * 32, est_bytes=4096),
        )
        report = analyze_gap_file(path)
        self.assertEqual(report["record_count"], 2)
        self.assertEqual(report["groups"], [
            {"pattern": "unclassified-runtime-gap", "native_family": "mmvq",
             "count": 2, "calls": 0, "estimated_bytes": 8192,
             "source_signatures": ["1" * 32, SIG]},
        ])
        # Absent provenance dimensions are None in the report, not fabricated.
        self.assertIsNone(report["provenance"]["hardware"]["architecture"])
        self.assertIsNone(report["provenance"]["build"]["build_descriptor_hash"])
        self.assertEqual(report["provenance"]["build"]["source_revision"], REVISION)

    def test_mixed_flat_gaps_fail_closed(self):
        # The one-provenance-namespace contract holds for flat records too:
        # gaps with a different hardware digest (a different device flush
        # spliced in) cannot share a report.
        path = self.write(
            "mixed.jsonl",
            {"kind": "header", "artifact_version": 1,
             "source_revision": REVISION, "manifest_hash": MANIFEST},
            flat_gap(),
            flat_gap(hardware="f" * 32),
        )
        with self.assertRaisesRegex(GapAnalysisError, "mixed hardware/build provenance"):
            analyze_gap_file(path)

    def test_direct_analyze_gaps_remains_strict(self):
        # Feeding a raw flat-shaped dict to the direct API must still fail:
        # the strict nested contract was not weakened, and source_schema is
        # not a bypass flag.
        with self.assertRaises(GapAnalysisError):
            analyze_gaps([flat_gap()])


if __name__ == "__main__":
    unittest.main()
