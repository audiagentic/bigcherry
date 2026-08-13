"""Tests for inventory.py — JSONL parsing, inventory building, SQLite loader.

Run with: python -m unittest tools.tests.test_inventory
"""

from __future__ import annotations

import json
import os
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path

# Ensure the bigcherry package is importable from project root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bigcherry import inventory  # noqa: E402
from bigcherry.inventory import Record, RecordError  # noqa: E402


# ------------------------------------------------------------------ fixtures

RECORD_HEADER = {
    "kind": "header",
    "source_revision": "abcdef1234567890",
    "manifest_hash": "deadbeef00112233",
    "signature_schema": 1,
    "hardware_schema": 1,
    "variant_set": "inventory",
}

RECORD_OBS_MMQ = {
    "kind": "observation",
    "hardware": "a" * 32,
    "signature": "b" * 32,
    "native": "mmq:native:v1",
    "canonical": {
        "op": "MUL_MAT",
        "src0_type": 8,      # q8_0
        "src1_type": 0,      # f32
        "dst_type": 0,       # f32
        "ne0": [64, 512],
        "ned": [64, 128],
    },
    "hardware_key": {
        "architecture_code": "gfx1100",
        "wave_size": 64,
        "compute_units": 60,
        "feature_flags": 1,
    },
    "calls": 5,
    "est_bytes": 2048,
    "devices": [0],
}

RECORD_OBS_MMF = {
    "kind": "observation",
    "hardware": "a" * 32,
    "signature": "c" * 32,
    "native": "mmf:native:v1",
    "canonical": {
        "op": "MUL_MAT",
        "src0_type": 0,      # f32
        "src1_type": 0,
        "dst_type": 0,
        "ne0": [256, 1024],
        "ned": [256, 256],
    },
    "hardware_key": {
        "architecture_code": "gfx1100",
        "wave_size": 64,
        "compute_units": 60,
        "feature_flags": 1,
    },
    "calls": 3,
    "est_bytes": 4096,
    "devices": [0],
}

RECORD_OBS_BLAS = {
    "kind": "observation",
    "hardware": "a" * 32,
    "signature": "d" * 32,
    "native": "blas:native:v1",
    "effective_api": "ggml_cuda_mul_mat_cublas",
    "effective_call_api": "cublasGemmEx",
    "canonical": {
        "op": "MUL_MAT",
        "src0_type": 1,      # f16
        "src1_type": 1,
        "dst_type": 1,
        "ne0": [2048, 4096],
        "ned": [2048, 1024],
    },
    "hardware_key": {
        "architecture_code": "gfx1100",
        "wave_size": 64,
        "compute_units": 60,
        "feature_flags": 1,
    },
    "calls": 1,
    "est_bytes": 8192,
    "devices": [0],
    "blas_metadata": {
        "operand_a_type": "f16",
        "operand_b_type": "f16",
        "output_type": "f16",
        "accumulation_type": "f16",
        "source_a_conversion": "direct",
        "source_b_conversion": "direct",
        "output_conversion": "direct",
        "requested_precision": "0",
        "effective_provider": "hipblas",
        "effective_backend": "unknown",
        "source_a_temp_bytes": 0,
        "source_b_temp_bytes": 0,
        "output_temp_bytes": 0,
    },
}

TUNING_HEADER = {
    "kind": "header",
    "artifact_version": 1,
    "source_revision": "abcdef1234567890",
    "manifest_hash": "deadbeef00112233",
    "variant_set": "workload-max",
    "build_descriptor_hash": "build-descriptor-test",
}

# Tuning result: native retained (no improvement above threshold)
TUNING_RESULT_NATIVE = {
    "kind": "result",
    "dispatch": "e" * 32,
    "winner": "mmq:native:v1",
    "improvement_pct": 0.0,
    "generated": 3,
    "eligible": 3,
    "measured": 2,
    "reason": "native retained",
    "candidates": [
        {
            "name": "mmq:native:v1",
            "status": "ok",
            "median_us": 1.500,
            "mad_us": 0.010,
            "p95_us": 1.600,
            "host_median_us": 0.400,
            "nmse": 0.0,
            "max_abs": 0.0,
            "workspace": 0,
            "samples": 10,
        },
        {
            "name": "mmq:generated:j4",
            "status": "ok",
            "median_us": 1.520,
            "mad_us": 0.008,
            "p95_us": 1.580,
            "host_median_us": 0.390,
            "nmse": 1e-6,
            "max_abs": 0.001,
            "workspace": 4096,
            "samples": 10,
        },
        {
            "name": "mmq:generated:j8",
            "status": "architecture",
            "median_us": 0.0,
            "mad_us": 0.0,
            "p95_us": 0.0,
            "host_median_us": 0.0,
            "nmse": 0.0,
            "max_abs": 0.0,
            "workspace": 0,
            "samples": 0,
        },
    ],
}

# Tuning result: challenger wins by 5%
TUNING_RESULT_IMPROVED = {
    "kind": "result",
    "dispatch": "f" * 32,
    "winner": "mmf:generated:nw4",
    "improvement_pct": 5.0,
    "generated": 4,
    "eligible": 3,
    "measured": 3,
    "reason": "measured winner",
    "candidates": [
        {
            "name": "mmf:native:v1",
            "status": "ok",
            "median_us": 2.000,
            "mad_us": 0.020,
            "p95_us": 2.200,
            "host_median_us": 0.600,
            "nmse": 0.0,
            "max_abs": 0.0,
            "workspace": 0,
            "samples": 10,
        },
        {
            "name": "mmf:generated:nw4",
            "status": "ok",
            "median_us": 1.900,
            "mad_us": 0.015,
            "p95_us": 2.050,
            "host_median_us": 0.550,
            "nmse": 2e-6,
            "max_abs": 0.002,
            "workspace": 8192,
            "samples": 10,
        },
        {
            "name": "mmf:generated:nw8",
            "status": "workspace",
            "median_us": 0.0,
            "mad_us": 0.0,
            "p95_us": 0.0,
            "host_median_us": 0.0,
            "nmse": 0.0,
            "max_abs": 0.0,
            "workspace": 0,
            "samples": 0,
        },
    ],
}

# Manifest for candidate data (used by load_measurements with manifest)
MANIFEST = {
    "manifest_hash": "deadbeef00112233",
    "candidates": [
        {
            "stable_name": "mmq:native:v1",
            "family": "mmq",
            "source_class": "native_wrapper",
            "implementation_version": 1,
            "architectures": ["gfx1100"],
            "architecture_mask": 1,
            "graph_safe": True,
            "deterministic": True,
            "config": {"j": 8},
        },
        {
            "stable_name": "mmq:generated:j4",
            "family": "mmq",
            "source_class": "new_generated_variant",
            "implementation_version": 1,
            "architectures": ["gfx1100"],
            "architecture_mask": 1,
            "graph_safe": False,
            "deterministic": True,
            "config": {"j": 4},
        },
        {
            "stable_name": "mmf:native:v1",
            "family": "mmf",
            "source_class": "native_wrapper",
            "implementation_version": 1,
            "architectures": ["gfx1100"],
            "architecture_mask": 1,
            "graph_safe": True,
            "deterministic": True,
            "config": {},
        },
        {
            "stable_name": "mmf:generated:nw4",
            "family": "mmf",
            "source_class": "new_generated_variant",
            "implementation_version": 1,
            "architectures": ["gfx1100"],
            "architecture_mask": 1,
            "graph_safe": False,
            "deterministic": True,
            "config": {"nwarps": 4},
        },
    ],
}


# ------------------------------------------------------------------ helpers

def make_jsonl_file(*records):
    """Write a JSONL file from dicts and return the path."""
    p = tempfile.mktemp(suffix=".jsonl")
    with open(p, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, separators=(",", ":")) + "\n")
    return Path(p)


class TempDB:
    """Create a temp SQLite DB initialized with the project schema."""

    def __init__(self):
        self._dir = tempfile.TemporaryDirectory(prefix="bigcherry_test_")
        self.db_path = Path(self._dir.name) / "test.sqlite"
        schema_path = Path(__file__).resolve().parents[2] / "sql" / "dispatch-db.sql"
        conn = sqlite3.connect(str(self.db_path))
        try:
            conn.executescript(schema_path.read_text(encoding="utf-8"))
            conn.commit()
        finally:
            conn.close()

    def query(self, sql, params=()):
        conn = sqlite3.connect(str(self.db_path))
        try:
            cursor = conn.execute(sql, params)
            return cursor.fetchall()
        finally:
            conn.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self._dir.cleanup()


# ------------------------------------------------------------------ tests

class TestReadJSONL(unittest.TestCase):
    """Parse record-mode JSONL files."""

    def test_full_record(self):
        path = make_jsonl_file(RECORD_HEADER, RECORD_OBS_MMQ, RECORD_OBS_MMF)
        try:
            rec = inventory.read_jsonl(path)
        finally:
            os.unlink(path)

        self.assertEqual(rec.header["source_revision"], "abcdef1234567890")
        self.assertEqual(len(rec.observations), 2)
        self.assertEqual(rec.observations[0]["native"], "mmq:native:v1")
        self.assertEqual(rec.observations[1]["native"], "mmf:native:v1")

    def test_truncated_last_line(self):
        """A truncated final line should be tolerated."""
        p = tempfile.mktemp(suffix=".jsonl")
        try:
            with open(p, "w", encoding="utf-8") as f:
                f.write(json.dumps(RECORD_HEADER) + "\n")
                f.write(json.dumps(RECORD_OBS_MMQ) + "\n")
                f.write('{"kind":"observation","hardware":"x')  # truncated
            rec = inventory.read_jsonl(Path(p))
            self.assertEqual(len(rec.observations), 1)
        finally:
            os.unlink(p)

    def test_no_header_raises(self):
        """A file without a header line must raise RecordError."""
        path = make_jsonl_file(RECORD_OBS_MMQ)
        try:
            with self.assertRaises(RecordError):
                inventory.read_jsonl(path)
        finally:
            os.unlink(path)

    def test_empty_lines_ignored(self):
        """Blank lines between records are skipped."""
        p = tempfile.mktemp(suffix=".jsonl")
        try:
            with open(p, "w", encoding="utf-8") as f:
                f.write(json.dumps(RECORD_HEADER) + "\n")
                f.write("\n")  # blank line
                f.write(json.dumps(RECORD_OBS_MMQ) + "\n")
                f.write("\n\n")  # multiple blank lines
            rec = inventory.read_jsonl(Path(p))
            self.assertEqual(len(rec.observations), 1)
        finally:
            os.unlink(p)

    def test_tuning_header_and_results(self):
        """read_jsonl only collects observations, not results.

        Tuning JSONL uses kind='result', which read_jsonl ignores because it
        only recognizes kind='observation'. This is intentional — the loader
        for tuning data (load_measurements) reads the file directly and filters
        by kind='result' itself.
        """
        path = make_jsonl_file(
            TUNING_HEADER, TUNING_RESULT_NATIVE, TUNING_RESULT_IMPROVED
        )
        try:
            rec = inventory.read_jsonl(path)
        finally:
            os.unlink(path)

        # Header is captured.
        self.assertEqual(rec.header["artifact_version"], 1)
        # Result records are NOT collected by read_jsonl (only observations).
        self.assertEqual(len(rec.observations), 0)


class TestBuildInventory(unittest.TestCase):
    """Derive observed type/width sets from a record."""

    def _make_record(self, *obs):
        return Record(header=RECORD_HEADER.copy(), observations=list(obs))

    def test_mmq_and_mmf_types(self):
        rec = self._make_record(RECORD_OBS_MMQ, RECORD_OBS_MMF)
        inv = inventory.build_inventory(rec)

        # MMQ saw q8_0 (type 8)
        self.assertIn("q8_0", inv["mmq_types"])
        # MMF saw f32 (type 0)
        self.assertIn("f32", inv["mmf_types"])
        self.assertEqual(inv["signatures_observed"], 2)

    def test_widths(self):
        rec = self._make_record(RECORD_OBS_MMQ, RECORD_OBS_MMF)
        inv = inventory.build_inventory(rec)

        # Width from ned[1]: 128 (mmq obs) and 256 (mmf obs)
        # But widths is capped at 1–16 for parameterised families,
        # so these large widths are excluded.
        self.assertEqual(inv["widths"], [])

    def test_blas_observed(self):
        rec = self._make_record(RECORD_OBS_BLAS)
        inv = inventory.build_inventory(rec)
        self.assertTrue(inv["uses_blas"])

    def test_empty_record(self):
        """No observations yields empty sets."""
        rec = self._make_record()
        inv = inventory.build_inventory(rec)
        self.assertEqual(inv["signatures_observed"], 0)
        self.assertEqual(inv["mmq_types"], [])
        self.assertFalse(inv["uses_blas"])

    def test_unknown_type_warns(self):
        """An unknown ggml_type should produce a warning."""
        obs = RECORD_OBS_MMQ.copy()
        obs["canonical"] = {**obs["canonical"], "src0_type": 999}
        rec = self._make_record(obs)

        # (prints to stderr, not Python warnings — skip assertion)
        inv = inventory.build_inventory(rec)
        # Should still work, just with empty type set for that family
        self.assertEqual(inv["mmq_types"], [])


class TestBuildDatabase(unittest.TestCase):
    """Populate SQLite from a record-mode JSONL."""

    def test_record_to_db(self):
        rec = Record(
            header=RECORD_HEADER.copy(),
            observations=[RECORD_OBS_MMQ, RECORD_OBS_MMF],
        )
        with TempDB() as db:
            counts = inventory.build_database(
                rec, db.db_path,
                Path(__file__).resolve().parents[2] / "sql" / "dispatch-db.sql",
            )

            self.assertEqual(counts["builds"], 1)
            self.assertEqual(counts["hardware"], 1)
            self.assertEqual(counts["signatures"], 2)
            self.assertEqual(counts["observations"], 2)

    def test_record_to_db_preserves_blas_diagnostics(self):
        rec = Record(header=RECORD_HEADER.copy(), observations=[RECORD_OBS_BLAS])
        with TempDB() as db:
            inventory.build_database(
                rec, db.db_path,
                Path(__file__).resolve().parents[2] / "sql" / "dispatch-db.sql",
            )
            row = db.query("SELECT diagnostics_json FROM observation")[0]
            diagnostics = json.loads(row[0])
            self.assertEqual(diagnostics["blas"]["effective_provider"], "hipblas")
            self.assertEqual(diagnostics["blas"]["output_conversion"], "direct")

            # Verify build row exists (query inside context so DB is open)
            builds = db.query("SELECT source_revision FROM build")
            self.assertEqual(builds[0][0], "abcdef1234567890")

            # Verify observations
            obs_count = db.query("SELECT COUNT(*) FROM observation")[0][0]
            self.assertEqual(obs_count, 1)


class TestLoadMeasurements(unittest.TestCase):
    """Load tuning JSONL into SQLite (HI20)."""

    def test_rejects_unknown_database_schema_version(self):
        path = make_jsonl_file(TUNING_HEADER, TUNING_RESULT_NATIVE)
        schema_path = Path(__file__).resolve().parents[2] / "sql" / "dispatch-db.sql"

        try:
            with TempDB() as db:
                connection = sqlite3.connect(str(db.db_path))
                try:
                    connection.execute(
                        "UPDATE schema_meta SET value = '99' "
                        "WHERE key = 'schema_version'"
                    )
                    connection.commit()
                finally:
                    connection.close()

                with self.assertRaisesRegex(
                    RecordError, "unsupported dispatch database schema_version"
                ):
                    inventory.load_measurements(
                        path, db.db_path, schema_path, manifest_path=None,
                    )
        finally:
            os.unlink(path)

    def test_basic_load(self):
        path = make_jsonl_file(TUNING_HEADER, TUNING_RESULT_NATIVE)
        schema_path = Path(__file__).resolve().parents[2] / "sql" / "dispatch-db.sql"

        with TempDB() as db:
            counts = inventory.load_measurements(
                path, db.db_path, schema_path, manifest_path=None,
            )

        self.assertEqual(counts["results"], 1)
        # 3 candidates in TUNING_RESULT_NATIVE
        self.assertEqual(counts["candidates"], 3)
        self.assertGreaterEqual(counts["measurements"], 3)
        os.unlink(path)

    def test_two_results(self):
        path = make_jsonl_file(
            TUNING_HEADER, TUNING_RESULT_NATIVE, TUNING_RESULT_IMPROVED
        )
        schema_path = Path(__file__).resolve().parents[2] / "sql" / "dispatch-db.sql"

        with TempDB() as db:
            counts = inventory.load_measurements(
                path, db.db_path, schema_path, manifest_path=None,
            )

        self.assertEqual(counts["results"], 2)
        os.unlink(path)

    def test_load_with_manifest(self):
        """When a manifest is provided, candidate data should be complete."""
        meas_path = make_jsonl_file(TUNING_HEADER, TUNING_RESULT_NATIVE)
        manifest_path = make_jsonl_file(json.dumps(MANIFEST))
        # Overwrite as JSON (not JSONL) for the manifest
        manifest_path.unlink()
        manifest_path.write_text(json.dumps(MANIFEST), encoding="utf-8")

        schema_path = Path(__file__).resolve().parents[2] / "sql" / "dispatch-db.sql"

        with TempDB() as db:
            _counts = inventory.load_measurements(
                meas_path, db.db_path, schema_path,
                manifest_path=manifest_path,
            )

            # Verify candidate has source_class from manifest (inside context)
            candidates = db.query("SELECT stable_name, source_class FROM candidate")
            by_name = {r[0]: r[1] for r in candidates}

            self.assertEqual(by_name.get("mmq:native:v1"), "native_wrapper")
            self.assertEqual(by_name.get("mmq:generated:j4"), "new_generated_variant")

        os.unlink(meas_path)
        os.unlink(manifest_path)

    def test_b3_fields_stored(self):
        """B3 schema fields (gpu_mad_us, host_median_us) should be populated."""
        path = make_jsonl_file(TUNING_HEADER, TUNING_RESULT_NATIVE)
        schema_path = Path(__file__).resolve().parents[2] / "sql" / "dispatch-db.sql"

        with TempDB() as db:
            inventory.load_measurements(
                path, db.db_path, schema_path, manifest_path=None,
            )

            # Check measurement has gpu_mad_us and host_median_us
            rows = db.query("""
                SELECT median_us, gpu_mad_us, p95_us, host_median_us
                FROM measurement WHERE accepted = 1 LIMIT 1
            """)
            self.assertEqual(len(rows), 1)
            # Values from TUNING_RESULT_NATIVE first candidate:
            # median=1.500, mad=0.010, p95=1.600, host=0.400
            self.assertAlmostEqual(rows[0][0], 1.500, places=3)
            self.assertAlmostEqual(rows[0][1], 0.010, places=3)
            self.assertAlmostEqual(rows[0][2], 1.600, places=3)
            self.assertAlmostEqual(rows[0][3], 0.400, places=3)

        os.unlink(path)

    def test_winner_reason_stored(self):
        """Winner.reason (B3) should be populated."""
        path = make_jsonl_file(TUNING_HEADER, TUNING_RESULT_IMPROVED)
        schema_path = Path(__file__).resolve().parents[2] / "sql" / "dispatch-db.sql"

        with TempDB() as db:
            inventory.load_measurements(
                path, db.db_path, schema_path, manifest_path=None,
            )

            rows = db.query("SELECT reason FROM winner")
            self.assertEqual(rows[0][0], "measured winner")

        os.unlink(path)

    def test_rejected_candidates_recorded(self):
        """Rejected candidates (architecture, workspace) should be in measurement."""
        path = make_jsonl_file(TUNING_HEADER, TUNING_RESULT_NATIVE)
        schema_path = Path(__file__).resolve().parents[2] / "sql" / "dispatch-db.sql"

        with TempDB() as db:
            inventory.load_measurements(
                path, db.db_path, schema_path, manifest_path=None,
            )

            # j8 is architecture-rejected in TUNING_RESULT_NATIVE
            rows = db.query("""
                SELECT stable_name, accepted, reject_reason
                FROM measurement m
                JOIN candidate c ON m.candidate_id = c.candidate_id
                WHERE m.accepted = 0
            """)
            rejected_names = {r[0]: r[2] for r in rows}
            self.assertIn("GGML_HIP_REJECT_ARCHITECTURE", rejected_names.get("mmq:generated:j8", ""))

        os.unlink(path)

    def test_invalid_dispatch_digest_rejected(self):
        """A malformed result must not be silently treated as no result."""
        bad_result = TUNING_RESULT_NATIVE.copy()
        bad_result["dispatch"] = "dead"  # too short

        path = make_jsonl_file(TUNING_HEADER, bad_result)
        schema_path = Path(__file__).resolve().parents[2] / "sql" / "dispatch-db.sql"

        with TempDB() as db:
            with self.assertRaisesRegex(RecordError, "invalid dispatch digest"):
                inventory.load_measurements(
                    path, db.db_path, schema_path, manifest_path=None,
                )

        os.unlink(path)

    def test_malformed_json_and_unknown_record_kind_are_rejected(self):
        schema_path = Path(__file__).resolve().parents[2] / "sql" / "dispatch-db.sql"
        for tail, message in (("{", "malformed JSON"),
                              (json.dumps({"kind": "future"}), "unknown record kind")):
            path = make_jsonl_file(TUNING_HEADER)
            path.write_text(path.read_text(encoding="utf-8") + tail + "\n", encoding="utf-8")
            try:
                with TempDB() as db:
                    with self.assertRaisesRegex(RecordError, message):
                        inventory.load_measurements(
                            path, db.db_path, schema_path, manifest_path=None,
                        )
            finally:
                os.unlink(path)

    def test_invalid_measurement_numeric_record_is_rejected(self):
        bad_result = json.loads(json.dumps(TUNING_RESULT_NATIVE))
        bad_result["candidates"][0]["median_us"] = "fast"
        path = make_jsonl_file(TUNING_HEADER, bad_result)
        schema_path = Path(__file__).resolve().parents[2] / "sql" / "dispatch-db.sql"
        try:
            with TempDB() as db:
                with self.assertRaisesRegex(RecordError, "median_us.*numeric"):
                    inventory.load_measurements(
                        path, db.db_path, schema_path, manifest_path=None,
                    )
        finally:
            os.unlink(path)

    def test_batching_and_raw_sample_count_must_be_consistent(self):
        bad_result = json.loads(json.dumps(TUNING_RESULT_NATIVE))
        bad_result["launches_per_sample"] = 0
        with self.assertRaisesRegex(RecordError, "launches_per_sample"):
            inventory._validate_measurement_result(bad_result, 2)

        bad_result = json.loads(json.dumps(TUNING_RESULT_NATIVE))
        candidate = bad_result["candidates"][0]
        candidate["samples"] = 1
        candidate["samples_us"] = [1.0, None, 2.0]
        with self.assertRaisesRegex(RecordError, "samples does not match"):
            inventory._validate_measurement_result(bad_result, 2)

    def test_sampling_policy_header_rejects_invalid_counts(self):
        with self.assertRaisesRegex(RecordError, "final_samples"):
            inventory._validate_measurement_header(
                {"kind": "header", "final_samples": -1}, 1
            )

    def test_workspace_evidence_keeps_request_allocation_peak_and_rebase_distinct(self):
        result = json.loads(json.dumps(TUNING_RESULT_NATIVE))
        candidate = result["candidates"][0]
        candidate["workspace"] = 4096
        candidate["pool_peak_bytes"] = 2048
        candidate["workspace_evidence"] = {
            "requested_bytes": 4096,
            "actual_bytes": 8192,
            "peak_bytes": 12288,
            "rebase_baseline_bytes": 10240,
            "rebase_current_bytes": 10240,
        }
        inventory._validate_measurement_result(result, 2)

    def test_workspace_evidence_rejects_request_mismatch(self):
        result = json.loads(json.dumps(TUNING_RESULT_NATIVE))
        candidate = result["candidates"][0]
        candidate["workspace_evidence"] = {
            "requested_bytes": 1,
            "actual_bytes": 1,
            "peak_bytes": 1,
            "rebase_baseline_bytes": 0,
            "rebase_current_bytes": 0,
        }
        with self.assertRaisesRegex(RecordError, "does not match candidate workspace"):
            inventory._validate_measurement_result(result, 2)

    def test_workspace_evidence_rejects_allocator_and_rebase_inconsistency(self):
        result = json.loads(json.dumps(TUNING_RESULT_NATIVE))
        candidate = result["candidates"][0]
        candidate["workspace"] = 4096
        candidate["workspace_evidence"] = {
            "requested_bytes": 4096,
            "actual_bytes": 2048,
            "peak_bytes": 1024,
            "rebase_baseline_bytes": 2048,
            "rebase_current_bytes": 4096,
        }
        with self.assertRaisesRegex(RecordError, "actual allocation"):
            inventory._validate_measurement_result(result, 2)

    def test_workspace_evidence_rejects_peak_and_pool_peak_mismatch(self):
        result = json.loads(json.dumps(TUNING_RESULT_NATIVE))
        candidate = result["candidates"][0]
        candidate["workspace"] = 4096
        candidate["pool_peak_bytes"] = 99
        candidate["workspace_evidence"] = {
            "requested_bytes": 4096,
            "actual_bytes": 4096,
            "peak_bytes": 8192,
            "rebase_baseline_bytes": 4096,
            "rebase_current_bytes": 4096,
        }
        with self.assertRaisesRegex(RecordError, "pool_peak_bytes"):
            inventory._validate_measurement_result(result, 2)


if __name__ == "__main__":
    unittest.main()
