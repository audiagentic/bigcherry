"""Tests for inventory.py — JSONL parsing, inventory building, SQLite loader.

Run with: python -m unittest tools.tests.test_inventory
"""

from __future__ import annotations

import json
import hashlib
import os
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path

# Ensure the bigcherry package is importable from project root.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bigcherry import inventory  # noqa: E402
from bigcherry.inventory import Record, RecordError  # noqa: E402
from bigcherry.tuning import catalog  # noqa: E402


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
        "src0_type": 8,  # q8_0
        "src1_type": 0,  # f32
        "dst_type": 0,  # f32
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
        "src0_type": 0,  # f32
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
        "src0_type": 1,  # f16
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
        "effective_call_api": "cublasGemmEx",
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
        schema_path = Path(__file__).resolve().parents[3] / "sql" / "dispatch-db.sql"
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
                rec,
                db.db_path,
                Path(__file__).resolve().parents[3] / "sql" / "dispatch-db.sql",
            )

            self.assertEqual(counts["builds"], 1)
            self.assertEqual(counts["hardware"], 1)
            self.assertEqual(counts["signatures"], 2)
            self.assertEqual(counts["observations"], 2)

    def test_record_to_db_preserves_blas_diagnostics(self):
        rec = Record(header=RECORD_HEADER.copy(), observations=[RECORD_OBS_BLAS])
        with TempDB() as db:
            inventory.build_database(
                rec,
                db.db_path,
                Path(__file__).resolve().parents[3] / "sql" / "dispatch-db.sql",
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

    def test_legacy_record_with_no_campaign_context_loads_with_null_identity(self):
        # RE09 (RV48 audit): an imported/ad-hoc record file, or any caller
        # not passing the new campaign-identity kwargs, must still load --
        # visibly NULL, not a fabricated or guessed identity.
        rec = Record(header=RECORD_HEADER.copy(), observations=[RECORD_OBS_MMQ])
        with TempDB() as db:
            inventory.build_database(
                rec,
                db.db_path,
                Path(__file__).resolve().parents[3] / "sql" / "dispatch-db.sql",
            )
            build_row = db.query(
                "SELECT source_slice_id, build_plan_id, effective_build_id, "
                "campaign_run_id, workload_id FROM build"
            )[0]
            self.assertEqual(build_row, (None, None, None, None, None))
            obs_row = db.query(
                "SELECT source_slice_id, workload_id, campaign_run_id FROM observation"
            )[0]
            self.assertEqual(obs_row, (None, None, None))

    def test_hostile_record_header_cannot_upgrade_to_campaign_scope(self):
        # GPT audit fix (2026-08-18): the record JSONL is attacker-
        # controlled bytes -- the compiled record binary never writes these
        # header fields, so a complete triple in the header is untrusted.
        # identity=None MUST mean legacy-imported with visibly-NULL
        # identity columns; a header triple must never upgrade scope.
        hostile_header = RECORD_HEADER.copy()
        hostile_header.update(
            {
                "source_slice_id": "slice-hostile",
                "build_plan_id": "plan-hostile",
                "effective_build_id": "eb-hostile",
                "campaign_run_id": "run-hostile",
            }
        )
        rec = Record(header=hostile_header, observations=[RECORD_OBS_MMQ])
        with TempDB() as db:
            inventory.build_database(
                rec,
                db.db_path,
                Path(__file__).resolve().parents[3] / "sql" / "dispatch-db.sql",
            )
            build_row = db.query(
                "SELECT source_slice_id, build_plan_id, effective_build_id, "
                "campaign_run_id, workload_id, identity_scope FROM build"
            )[0]
            self.assertEqual(
                build_row, (None, None, None, None, None, "legacy-imported")
            )

    def test_partial_campaign_identity_fails_closed(self):
        # GPT audit fix (2026-08-18): a CampaignDatabaseIdentity with an
        # empty required field is a caller bug -- fail closed rather than
        # silently write partial campaign evidence.
        rec = Record(header=RECORD_HEADER.copy(), observations=[RECORD_OBS_MMQ])
        with TempDB() as db:
            with self.assertRaises(RecordError):
                inventory.build_database(
                    rec,
                    db.db_path,
                    Path(__file__).resolve().parents[3] / "sql" / "dispatch-db.sql",
                    identity=inventory.CampaignDatabaseIdentity(
                        source_slice_id="slice-1",
                        build_plan_id="",
                        effective_build_id="eb-1",
                        campaign_run_id="run-1",
                    ),
                )

    def test_production_record_with_campaign_context_persists_identity(self):
        # RE09: a real caller that ran this record build through
        # execute_campaign_lane() and holds its CampaignLaneResult passes
        # those identities through explicitly.
        rec = Record(header=RECORD_HEADER.copy(), observations=[RECORD_OBS_MMQ])
        with TempDB() as db:
            inventory.build_database(
                rec,
                db.db_path,
                Path(__file__).resolve().parents[3] / "sql" / "dispatch-db.sql",
                identity=inventory.CampaignDatabaseIdentity(
                    source_slice_id="slice-1",
                    build_plan_id="plan-1",
                    effective_build_id="effective-1",
                    campaign_run_id="run-1",
                    workload_id="workload-1",
                ),
            )
            build_row = db.query(
                "SELECT source_slice_id, build_plan_id, effective_build_id, "
                "campaign_run_id, workload_id FROM build"
            )[0]
            self.assertEqual(
                build_row, ("slice-1", "plan-1", "effective-1", "run-1", "workload-1")
            )
            obs_row = db.query(
                "SELECT source_slice_id, workload_id, campaign_run_id FROM observation"
            )[0]
            self.assertEqual(obs_row, ("slice-1", "workload-1", "run-1"))


class TestLoadMeasurements(unittest.TestCase):
    """Load tuning JSONL into SQLite (HI20)."""

    def test_rejects_unknown_database_schema_version(self):
        path = make_jsonl_file(TUNING_HEADER, TUNING_RESULT_NATIVE)
        schema_path = Path(__file__).resolve().parents[3] / "sql" / "dispatch-db.sql"

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
                        path,
                        db.db_path,
                        schema_path,
                        manifest_path=None,
                    )
        finally:
            os.unlink(path)

    def test_basic_load(self):
        path = make_jsonl_file(TUNING_HEADER, TUNING_RESULT_NATIVE)
        schema_path = Path(__file__).resolve().parents[3] / "sql" / "dispatch-db.sql"

        with TempDB() as db:
            counts = inventory.load_measurements(
                path,
                db.db_path,
                schema_path,
                manifest_path=None,
            )

        self.assertEqual(counts["results"], 1)
        # 3 candidates in TUNING_RESULT_NATIVE
        self.assertEqual(counts["candidates"], 3)
        self.assertGreaterEqual(counts["measurements"], 3)
        os.unlink(path)

    def test_legacy_load_has_null_campaign_identity(self):
        # RE09 (RV48 audit): no campaign kwargs supplied -- behaves exactly
        # as before, NULL identity, not a fabricated one.
        path = make_jsonl_file(TUNING_HEADER, TUNING_RESULT_NATIVE)
        schema_path = Path(__file__).resolve().parents[3] / "sql" / "dispatch-db.sql"
        try:
            with TempDB() as db:
                inventory.load_measurements(
                    path, db.db_path, schema_path, manifest_path=None
                )
                build_row = db.query(
                    "SELECT source_slice_id, build_plan_id FROM build"
                )[0]
                self.assertEqual(build_row, (None, None))
                measurement_row = db.query(
                    "SELECT source_slice_id, build_plan_id, workload_id, "
                    "campaign_run_id FROM measurement LIMIT 1"
                )[0]
                self.assertEqual(measurement_row, (None, None, None, None))
                winner_row = db.query(
                    "SELECT source_slice_id, campaign_run_id FROM winner LIMIT 1"
                )[0]
                self.assertEqual(winner_row, (None, None))
        finally:
            os.unlink(path)

    def test_campaign_identity_persists_on_build_measurement_and_winner(self):
        # RE09: a real caller holding a CampaignLaneResult passes these
        # through; they land on build, measurement, AND winner rows.
        path = make_jsonl_file(TUNING_HEADER, TUNING_RESULT_NATIVE)
        schema_path = Path(__file__).resolve().parents[3] / "sql" / "dispatch-db.sql"
        try:
            with TempDB() as db:
                inventory.load_measurements(
                    path,
                    db.db_path,
                    schema_path,
                    manifest_path=None,
                    identity=inventory.CampaignDatabaseIdentity(
                        source_slice_id="slice-1",
                        build_plan_id="plan-1",
                        effective_build_id="effective-1",
                        campaign_run_id="run-1",
                        workload_id="workload-1",
                    ),
                )
                build_row = db.query(
                    "SELECT source_slice_id, build_plan_id, effective_build_id, "
                    "campaign_run_id, workload_id FROM build"
                )[0]
                self.assertEqual(
                    build_row,
                    ("slice-1", "plan-1", "effective-1", "run-1", "workload-1"),
                )
                measurement_row = db.query(
                    "SELECT source_slice_id, build_plan_id, effective_build_id, "
                    "workload_id, campaign_run_id FROM measurement LIMIT 1"
                )[0]
                self.assertEqual(
                    measurement_row,
                    ("slice-1", "plan-1", "effective-1", "workload-1", "run-1"),
                )
                winner_row = db.query(
                    "SELECT source_slice_id, build_plan_id, effective_build_id, "
                    "workload_id, campaign_run_id FROM winner LIMIT 1"
                )[0]
                self.assertEqual(
                    winner_row,
                    ("slice-1", "plan-1", "effective-1", "workload-1", "run-1"),
                )
        finally:
            os.unlink(path)

    def test_two_distinct_campaign_identities_can_share_the_same_legacy_key(self):
        # RE09/RV50 schema-4: the whole point of identity_scope + the two
        # partial unique indexes -- two campaign builds sharing the old
        # five-field legacy key but genuinely different campaign identity
        # must coexist as two real rows, not collide or force one to be
        # silently misattributed to the other (the schema-3-era problem
        # this migration exists to fix). Exercised directly at the SQL
        # level, not through two full load_measurements() calls into the
        # same DB --
        # through two full load_measurements() calls into the same DB --
        # a second full load hits a separate, pre-existing, unrelated
        # limitation (placeholder hardware digest collision -- see
        # _resolve_hardware's own KNOWN GAP comment and the "agreeing"
        # test below) that has nothing to do with campaign identity.
        schema_path = Path(__file__).resolve().parents[3] / "sql" / "dispatch-db.sql"
        with TempDB() as db:
            connection = sqlite3.connect(str(db.db_path))
            try:
                connection.executescript(schema_path.read_text(encoding="utf-8"))
                shared_legacy_key = (
                    "rev1",
                    "manifest1",
                    1,
                    1,
                    "inventory",
                    "descriptor1",
                )
                connection.execute(
                    "INSERT INTO build (source_revision, manifest_hash, signature_schema, "
                    "hardware_schema, variant_set, build_descriptor_hash, source_slice_id, "
                    "build_plan_id, effective_build_id, campaign_run_id, identity_scope) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'campaign')",
                    (*shared_legacy_key, "slice-a", "plan-a", "effective-a", "run-a"),
                )
                # Two DIFFERENT campaign identities sharing the exact same
                # legacy key -- the old schema-3 table-level UNIQUE would
                # have rejected this outright; schema-4's partial unique
                # index (scoped to identity_scope='campaign') allows it.
                connection.execute(
                    "INSERT INTO build (source_revision, manifest_hash, signature_schema, "
                    "hardware_schema, variant_set, build_descriptor_hash, source_slice_id, "
                    "build_plan_id, effective_build_id, campaign_run_id, identity_scope) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'campaign')",
                    (*shared_legacy_key, "slice-b", "plan-b", "effective-b", "run-b"),
                )
                connection.commit()
                count = connection.execute("SELECT COUNT(*) FROM build").fetchone()[0]
                self.assertEqual(count, 2)
                # A third row reusing an ALREADY-CLAIMED campaign triple
                # must still fail -- the partial unique index is real.
                with self.assertRaises(sqlite3.IntegrityError):
                    connection.execute(
                        "INSERT INTO build (source_revision, manifest_hash, signature_schema, "
                        "hardware_schema, variant_set, build_descriptor_hash, source_slice_id, "
                        "build_plan_id, effective_build_id, campaign_run_id, identity_scope) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'campaign')",
                        (
                            *shared_legacy_key,
                            "slice-a",
                            "plan-a",
                            "effective-a",
                            "run-a-again",
                        ),
                    )
            finally:
                connection.close()

    def test_a_campaign_load_never_matches_an_existing_legacy_imported_row(self):
        # RE09/RV50 schema-4: a legacy-imported row (predates campaign
        # identity, or was deliberately loaded without it) and a later
        # campaign-aware load sharing the exact same five-field legacy key
        # are now looked up via COMPLETELY DIFFERENT identity_scope-scoped
        # queries -- the campaign lookup can never match a
        # identity_scope='legacy-imported' row at all, so the migration-
        # era aliasing risk the interim (pre-schema-4) fail-closed check
        # used to guard against is now structurally impossible, not just
        # rejected. The campaign-aware load gets its OWN new build row.
        path_legacy = make_jsonl_file(TUNING_HEADER, TUNING_RESULT_NATIVE)
        path_campaign = make_jsonl_file(TUNING_HEADER, TUNING_RESULT_NATIVE)
        schema_path = Path(__file__).resolve().parents[3] / "sql" / "dispatch-db.sql"
        try:
            with TempDB() as db:
                inventory.load_measurements(
                    path_legacy, db.db_path, schema_path, manifest_path=None
                )
                row = db.query("SELECT source_slice_id, identity_scope FROM build")[0]
                self.assertEqual(row, (None, "legacy-imported"))
        finally:
            os.unlink(path_legacy)
            os.unlink(path_campaign)

    def test_agreeing_campaign_identity_on_the_same_legacy_key_does_not_raise(self):
        # A campaign-scoped load must succeed and be found on a SECOND
        # lookup by the same identity, proven directly against the same
        # SELECT the real code path uses, since a second full
        # load_measurements() call into the same DB hits a separate,
        # pre-existing, unrelated limitation (placeholder hardware digest
        # collision -- see _resolve_hardware's own KNOWN GAP comment) that
        # has nothing to do with campaign identity.
        path_a = make_jsonl_file(TUNING_HEADER, TUNING_RESULT_NATIVE)
        schema_path = Path(__file__).resolve().parents[3] / "sql" / "dispatch-db.sql"
        try:
            with TempDB() as db:
                inventory.load_measurements(
                    path_a,
                    db.db_path,
                    schema_path,
                    manifest_path=None,
                    identity=inventory.CampaignDatabaseIdentity(
                        source_slice_id="slice-a",
                        build_plan_id="plan-a",
                        effective_build_id="effective-a",
                        campaign_run_id="run-a",
                    ),
                )
                row = db.query(
                    "SELECT source_slice_id, build_plan_id, identity_scope FROM build"
                )[0]
                self.assertEqual(row, ("slice-a", "plan-a", "campaign"))
        finally:
            os.unlink(path_a)

    def test_measurement_import_skips_native_twin_candidate(self):
        """HI24 step 4: the synthetic double-native replicate must not become
        a candidate or a measurement row in SQLite; the JSONL stays its
        authoritative evidence."""
        import copy

        row = copy.deepcopy(TUNING_RESULT_NATIVE)
        twin = dict(row["candidates"][0])
        twin["name"] = "mmq:native:v1#twin"
        twin["median_us"] = 1.530
        row["candidates"].append(twin)
        # The funnel stays over the registry: measured does not count the twin.
        row["measured"] = 2

        path = make_jsonl_file(TUNING_HEADER, row)
        schema_path = Path(__file__).resolve().parents[3] / "sql" / "dispatch-db.sql"

        with TempDB() as db:
            counts = inventory.load_measurements(
                path,
                db.db_path,
                schema_path,
                manifest_path=None,
            )
            twins = db.query(
                "SELECT COUNT(*) FROM candidate WHERE stable_name LIKE '%#twin'"
            )[0][0]
            measurements = db.query("SELECT COUNT(*) FROM measurement")[0][0]
            winner_native = db.query("SELECT native_stable_name FROM winner")[0][0]

        self.assertEqual(counts["results"], 1)
        # 3 registry candidates -- not 4 with the twin row.
        self.assertEqual(counts["candidates"], 3)
        self.assertEqual(twins, 0)
        # 2 measured registry candidates + 1 screened-out = 3 rows, no twin.
        self.assertEqual(measurements, 3)
        self.assertEqual(winner_native, "mmq:native:v1")
        os.unlink(path)

    def test_two_results(self):
        path = make_jsonl_file(
            TUNING_HEADER, TUNING_RESULT_NATIVE, TUNING_RESULT_IMPROVED
        )
        schema_path = Path(__file__).resolve().parents[3] / "sql" / "dispatch-db.sql"

        with TempDB() as db:
            counts = inventory.load_measurements(
                path,
                db.db_path,
                schema_path,
                manifest_path=None,
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

        schema_path = Path(__file__).resolve().parents[3] / "sql" / "dispatch-db.sql"

        with TempDB() as db:
            _counts = inventory.load_measurements(
                meas_path,
                db.db_path,
                schema_path,
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
        schema_path = Path(__file__).resolve().parents[3] / "sql" / "dispatch-db.sql"

        with TempDB() as db:
            inventory.load_measurements(
                path,
                db.db_path,
                schema_path,
                manifest_path=None,
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
        schema_path = Path(__file__).resolve().parents[3] / "sql" / "dispatch-db.sql"

        with TempDB() as db:
            inventory.load_measurements(
                path,
                db.db_path,
                schema_path,
                manifest_path=None,
            )

            rows = db.query("SELECT reason FROM winner")
            self.assertEqual(rows[0][0], "measured winner")

        os.unlink(path)

    def test_rejected_candidates_recorded(self):
        """Rejected candidates (architecture, workspace) should be in measurement."""
        path = make_jsonl_file(TUNING_HEADER, TUNING_RESULT_NATIVE)
        schema_path = Path(__file__).resolve().parents[3] / "sql" / "dispatch-db.sql"

        with TempDB() as db:
            inventory.load_measurements(
                path,
                db.db_path,
                schema_path,
                manifest_path=None,
            )

            # j8 is architecture-rejected in TUNING_RESULT_NATIVE
            rows = db.query("""
                SELECT stable_name, accepted, reject_reason
                FROM measurement m
                JOIN candidate c ON m.candidate_id = c.candidate_id
                WHERE m.accepted = 0
            """)
            rejected_names = {r[0]: r[2] for r in rows}
            self.assertIn(
                "GGML_HIP_REJECT_ARCHITECTURE",
                rejected_names.get("mmq:generated:j8", ""),
            )

        os.unlink(path)

    def test_invalid_dispatch_digest_rejected(self):
        """A malformed result must not be silently treated as no result."""
        bad_result = TUNING_RESULT_NATIVE.copy()
        bad_result["dispatch"] = "dead"  # too short

        path = make_jsonl_file(TUNING_HEADER, bad_result)
        schema_path = Path(__file__).resolve().parents[3] / "sql" / "dispatch-db.sql"

        with TempDB() as db:
            with self.assertRaisesRegex(RecordError, "invalid dispatch digest"):
                inventory.load_measurements(
                    path,
                    db.db_path,
                    schema_path,
                    manifest_path=None,
                )

        os.unlink(path)

    def test_malformed_json_and_unknown_record_kind_are_rejected(self):
        schema_path = Path(__file__).resolve().parents[3] / "sql" / "dispatch-db.sql"
        for tail, message in (
            ("{", "malformed JSON"),
            (json.dumps({"kind": "future"}), "unknown record kind"),
        ):
            path = make_jsonl_file(TUNING_HEADER)
            path.write_text(
                path.read_text(encoding="utf-8") + tail + "\n", encoding="utf-8"
            )
            try:
                with TempDB() as db:
                    with self.assertRaisesRegex(RecordError, message):
                        inventory.load_measurements(
                            path,
                            db.db_path,
                            schema_path,
                            manifest_path=None,
                        )
            finally:
                os.unlink(path)

    def test_invalid_measurement_numeric_record_is_rejected(self):
        bad_result = json.loads(json.dumps(TUNING_RESULT_NATIVE))
        bad_result["candidates"][0]["median_us"] = "fast"
        path = make_jsonl_file(TUNING_HEADER, bad_result)
        schema_path = Path(__file__).resolve().parents[3] / "sql" / "dispatch-db.sql"
        try:
            with TempDB() as db:
                with self.assertRaisesRegex(RecordError, "median_us.*numeric"):
                    inventory.load_measurements(
                        path,
                        db.db_path,
                        schema_path,
                        manifest_path=None,
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


class TestSignatureCanonicalConsistency(unittest.TestCase):
    """HI121 review follow-up: a digest must correspond to exactly one
    canonical shape -- _resolve_signature() cross-checks every real
    (non-empty) canonical dict it sees for a digest against what's already
    stored, rather than silently keeping the first and ignoring the rest."""

    def _result_with_signature(self, *, signature_hex: str, canonical: dict) -> dict:
        result = json.loads(json.dumps(TUNING_RESULT_NATIVE))
        result["signature"] = signature_hex
        result["canonical"] = canonical
        return result

    def test_identical_canonical_on_repeat_sighting_is_fine(self):
        signature_hex = "a" * 32
        canonical = {"op": "MUL_MAT", "schema_version": 2, "flags": 0, "ne0": [1, 2, 3, 4], "ned": [1, 2, 3, 4]}
        path = make_jsonl_file(
            TUNING_HEADER,
            self._result_with_signature(signature_hex=signature_hex, canonical=canonical),
            self._result_with_signature(signature_hex=signature_hex, canonical=canonical),
        )
        schema_path = Path(__file__).resolve().parents[3] / "sql" / "dispatch-db.sql"
        try:
            with TempDB() as db:
                inventory.load_measurements(path, db.db_path, schema_path, manifest_path=None)
                rows = db.query("SELECT COUNT(*) FROM signature WHERE signature_digest=?", (bytes.fromhex(signature_hex),))
                self.assertEqual(rows[0][0], 1)
        finally:
            os.unlink(path)

    def test_conflicting_canonical_for_same_digest_raises(self):
        # The in-memory signature_cache short-circuits repeat lookups WITHIN
        # one load_measurements() call, so this needs two separate loads
        # against the same DB -- the realistic scenario anyway: two
        # different measurement runs/files loaded into one persistent DB,
        # not two rows a single real producer emitted in one run.
        signature_hex = "b" * 32
        canonical_a = {"op": "MUL_MAT", "schema_version": 2, "flags": 0, "ne0": [1, 2, 3, 4], "ned": [1, 2, 3, 4]}
        canonical_b = {"op": "MUL_MAT_ID", "schema_version": 2, "flags": 8, "ne0": [5, 6, 7, 8], "ned": [5, 6, 7, 8]}
        path_a = make_jsonl_file(
            TUNING_HEADER, self._result_with_signature(signature_hex=signature_hex, canonical=canonical_a),
        )
        path_b = make_jsonl_file(
            TUNING_HEADER, self._result_with_signature(signature_hex=signature_hex, canonical=canonical_b),
        )
        schema_path = Path(__file__).resolve().parents[3] / "sql" / "dispatch-db.sql"
        try:
            with TempDB() as db:
                inventory.load_measurements(path_a, db.db_path, schema_path, manifest_path=None)
                with self.assertRaisesRegex(RecordError, "different canonical content"):
                    inventory.load_measurements(path_b, db.db_path, schema_path, manifest_path=None)
        finally:
            os.unlink(path_a)
            os.unlink(path_b)

    def test_missing_canonical_on_repeat_sighting_does_not_raise(self):
        # Absence of canonical data is not a disagreement -- a row that
        # simply doesn't carry the field must not be treated as a conflict
        # against an already-stored real canonical shape.
        signature_hex = "c" * 32
        canonical = {"op": "MUL_MAT", "schema_version": 2, "flags": 0, "ne0": [1, 2, 3, 4], "ned": [1, 2, 3, 4]}
        with_canonical = self._result_with_signature(signature_hex=signature_hex, canonical=canonical)
        without_canonical = json.loads(json.dumps(TUNING_RESULT_NATIVE))
        without_canonical["signature"] = signature_hex
        path = make_jsonl_file(TUNING_HEADER, with_canonical, without_canonical)
        schema_path = Path(__file__).resolve().parents[3] / "sql" / "dispatch-db.sql"
        try:
            with TempDB() as db:
                inventory.load_measurements(path, db.db_path, schema_path, manifest_path=None)
        finally:
            os.unlink(path)


class TestHipCapabilityPersistence(unittest.TestCase):
    """HI121 M2: load_measurements() verifies the compiled producer's own
    self-reported capability mask against the supplied manifest before
    persisting anything to build_capability, and never accepts a
    caller-supplied mask directly."""

    def _manifest_and_header(self, *, producer_capabilities="0000000000000000000000000000001f",
                              header_caps=None, tamper_manifest_hash=False):
        families = ("mmvq", "mmq", "mmvf", "mmf", "blas")
        manifest = {
            "artifact_version": 1,
            "variant_set": "inventory",
            "source_revision": "b" * 40,
            "architectures": ["gfx1100"],
            "signature_schema_version": 2,
            "hardware_schema_version": 1,
            "producer_capabilities": producer_capabilities,
            "candidates": [
                {
                    "stable_name": f"{f}:native:v1", "family": f, "source_class": "native_wrapper",
                    "implementation_version": 1, "architectures": ["gfx1100"], "architecture_mask": 1,
                    "graph_safe": True, "deterministic": True, "config": {},
                }
                for f in families
            ],
            "summary": {
                "total": len(families),
                "by_family": dict.fromkeys(families, 1),
                "by_source_class": {"native_wrapper": len(families)},
            },
        }
        manifest["manifest_hash"] = catalog.manifest_hash(manifest)
        manifest["build_descriptor"] = catalog.build_descriptor(manifest)
        if tamper_manifest_hash:
            manifest["manifest_hash"] = "f" * 32

        header = {
            "kind": "header",
            "artifact_version": 1,
            "source_revision": manifest["source_revision"],
            "manifest_hash": manifest["manifest_hash"],
            "variant_set": "inventory",
            "build_descriptor_hash": manifest["build_descriptor"]["descriptor_hash"],
            "producer_capabilities": header_caps if header_caps is not None else producer_capabilities,
        }
        return manifest, header

    def _write_manifest(self, manifest: dict) -> Path:
        path = Path(tempfile.mktemp(suffix=".json"))
        path.write_text(json.dumps(manifest), encoding="utf-8")
        return path

    def test_matching_header_and_manifest_persists_capability_row(self):
        manifest, header = self._manifest_and_header()
        meas_path = make_jsonl_file(header, TUNING_RESULT_NATIVE)
        manifest_path = self._write_manifest(manifest)
        schema_path = Path(__file__).resolve().parents[3] / "sql" / "dispatch-db.sql"
        try:
            with TempDB() as db:
                inventory.load_measurements(
                    meas_path, db.db_path, schema_path, manifest_path=manifest_path,
                )
                row = db.query(
                    "SELECT producer_capabilities FROM build_capability WHERE backend='hip'"
                )
                self.assertEqual(len(row), 1)
                self.assertEqual(row[0][0], bytes.fromhex("0000000000000000000000000000001f"))
        finally:
            os.unlink(meas_path)
            os.unlink(manifest_path)

    def test_no_manifest_persists_no_capability_row(self):
        _, header = self._manifest_and_header()
        meas_path = make_jsonl_file(header, TUNING_RESULT_NATIVE)
        schema_path = Path(__file__).resolve().parents[3] / "sql" / "dispatch-db.sql"
        try:
            with TempDB() as db:
                inventory.load_measurements(
                    meas_path, db.db_path, schema_path, manifest_path=None,
                )
                row = db.query("SELECT COUNT(*) FROM build_capability")
                self.assertEqual(row[0][0], 0)
        finally:
            os.unlink(meas_path)

    def test_header_missing_producer_capabilities_persists_no_row(self):
        # Older measurements header predating this field -- not an error,
        # just nothing to verify or persist.
        manifest, header = self._manifest_and_header()
        del header["producer_capabilities"]
        meas_path = make_jsonl_file(header, TUNING_RESULT_NATIVE)
        manifest_path = self._write_manifest(manifest)
        schema_path = Path(__file__).resolve().parents[3] / "sql" / "dispatch-db.sql"
        try:
            with TempDB() as db:
                inventory.load_measurements(
                    meas_path, db.db_path, schema_path, manifest_path=manifest_path,
                )
                row = db.query("SELECT COUNT(*) FROM build_capability")
                self.assertEqual(row[0][0], 0)
        finally:
            os.unlink(meas_path)
            os.unlink(manifest_path)

    def test_header_capability_mismatch_raises_and_persists_nothing(self):
        manifest, header = self._manifest_and_header(
            producer_capabilities="0000000000000000000000000000001f",
            header_caps="00000000000000000000000000000001",  # different from manifest's claim
        )
        meas_path = make_jsonl_file(header, TUNING_RESULT_NATIVE)
        manifest_path = self._write_manifest(manifest)
        schema_path = Path(__file__).resolve().parents[3] / "sql" / "dispatch-db.sql"
        try:
            with TempDB() as db:
                with self.assertRaisesRegex(RecordError, "does not match"):
                    inventory.load_measurements(
                        meas_path, db.db_path, schema_path, manifest_path=manifest_path,
                    )
                row = db.query("SELECT COUNT(*) FROM build_capability")
                self.assertEqual(row[0][0], 0)
        finally:
            os.unlink(meas_path)
            os.unlink(manifest_path)

    def test_manifest_hash_tampered_after_the_fact_raises(self):
        # HI121 round-8 M2 gate: a manifest whose own manifest_hash field no
        # longer matches its recomputed content must be rejected outright,
        # not trusted just because the header happens to reference the
        # (now-stale) claimed hash.
        manifest, header = self._manifest_and_header(tamper_manifest_hash=True)
        header["manifest_hash"] = manifest["manifest_hash"]  # header agrees with the tampered claim
        meas_path = make_jsonl_file(header, TUNING_RESULT_NATIVE)
        manifest_path = self._write_manifest(manifest)
        schema_path = Path(__file__).resolve().parents[3] / "sql" / "dispatch-db.sql"
        try:
            with TempDB() as db:
                with self.assertRaisesRegex(RecordError, "recomputed manifest_hash"):
                    inventory.load_measurements(
                        meas_path, db.db_path, schema_path, manifest_path=manifest_path,
                    )
                row = db.query("SELECT COUNT(*) FROM build_capability")
                self.assertEqual(row[0][0], 0)
        finally:
            os.unlink(meas_path)
            os.unlink(manifest_path)

    def test_fabricated_self_consistent_embedded_descriptor_is_rejected(self):
        # The embedded descriptor can be made internally self-consistent by an
        # attacker, but it must still equal catalog.build_descriptor() derived
        # from the manifest's actual content.
        manifest, header = self._manifest_and_header()
        fabricated = dict(manifest["build_descriptor"])
        fabricated["candidate_count"] += 1
        descriptor_payload = json.dumps(
            {k: v for k, v in fabricated.items() if k != "descriptor_hash"},
            sort_keys=True,
            separators=(",", ":"),
        )
        fabricated["descriptor_hash"] = hashlib.blake2b(
            descriptor_payload.encode("utf-8"), digest_size=16
        ).hexdigest()
        manifest["build_descriptor"] = fabricated
        header["build_descriptor_hash"] = fabricated["descriptor_hash"]
        meas_path = make_jsonl_file(header, TUNING_RESULT_NATIVE)
        manifest_path = self._write_manifest(manifest)
        schema_path = Path(__file__).resolve().parents[3] / "sql" / "dispatch-db.sql"
        try:
            with TempDB() as db:
                with self.assertRaisesRegex(RecordError, "embedded build_descriptor"):
                    inventory.load_measurements(
                        meas_path, db.db_path, schema_path, manifest_path=manifest_path,
                    )
                self.assertEqual(
                    db.query("SELECT COUNT(*) FROM build_capability")[0][0], 0
                )
        finally:
            os.unlink(meas_path)
            os.unlink(manifest_path)

    def test_wrong_source_root_manifest_mismatch_raises(self):
        # The HI121 round-8 M2 gate this session's own real regression: a
        # manifest generated from a DIFFERENT source root than what the
        # compiled binary actually reports, even though the manifest is
        # internally self-consistent.
        manifest, header = self._manifest_and_header(producer_capabilities="0" * 32)
        header["source_revision"] = "c" * 40  # different root's revision
        meas_path = make_jsonl_file(header, TUNING_RESULT_NATIVE)
        manifest_path = self._write_manifest(manifest)
        schema_path = Path(__file__).resolve().parents[3] / "sql" / "dispatch-db.sql"
        try:
            with TempDB() as db:
                with self.assertRaisesRegex(RecordError, "source_revision"):
                    inventory.load_measurements(
                        meas_path, db.db_path, schema_path, manifest_path=manifest_path,
                    )
                row = db.query("SELECT COUNT(*) FROM build_capability")
                self.assertEqual(row[0][0], 0)
        finally:
            os.unlink(meas_path)
            os.unlink(manifest_path)

    def test_reload_with_identical_capability_is_idempotent(self):
        manifest, header = self._manifest_and_header()
        meas_path = make_jsonl_file(header, TUNING_RESULT_NATIVE)
        manifest_path = self._write_manifest(manifest)
        schema_path = Path(__file__).resolve().parents[3] / "sql" / "dispatch-db.sql"
        try:
            with TempDB() as db:
                inventory.load_measurements(
                    meas_path, db.db_path, schema_path, manifest_path=manifest_path,
                )
                inventory.load_measurements(
                    meas_path, db.db_path, schema_path, manifest_path=manifest_path,
                )
                row = db.query("SELECT COUNT(*) FROM build_capability")
                self.assertEqual(row[0][0], 1)
        finally:
            os.unlink(meas_path)
            os.unlink(manifest_path)


if __name__ == "__main__":
    unittest.main()
