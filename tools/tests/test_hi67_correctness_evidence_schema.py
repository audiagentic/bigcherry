"""HI67 slices 2/3 (RV49 contract, RV77 GPT-adjudicated schema design):
correctness_evidence / correctness_evidence_seed apply cleanly, migrate a
real schema-5 database in place, and enforce the identity/aggregation rules
RV77 required (existing build/hardware/signature/candidate FKs, per-seed
rows, seed_count >= 3)."""

from __future__ import annotations

import sqlite3
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

SCHEMA_SQL = Path(__file__).resolve().parents[2] / "sql" / "dispatch-db.sql"


def _seed_build_hardware_signature_candidates(conn: sqlite3.Connection) -> tuple[int, int, int, int, int]:
    conn.execute(
        "INSERT INTO build (source_revision, manifest_hash, signature_schema, "
        "hardware_schema, variant_set) VALUES ('deadbeefdeadbeefdead', 'aa', 1, 1, 'inventory')"
    )
    build_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
    conn.execute(
        "INSERT INTO hardware (hardware_digest, architecture, architecture_code, "
        "wave_size, compute_units, feature_flags, canonical_json) VALUES "
        "(x'00', 'gfx1100', 1, 32, 96, 0, '{}')"
    )
    hardware_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
    conn.execute(
        "INSERT INTO signature (signature_digest, base_digest, schema_version, op, "
        "src0_type, src1_type, dst_type, m, n, k, canonical_json) VALUES "
        "(x'01', x'02', 1, 'MUL_MAT', 'q8_0', 'f32', 'f32', 1, 1, 1, '{}')"
    )
    signature_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
    conn.execute(
        "INSERT INTO candidate (build_id, stable_name, family, source_class, "
        "implementation_version, architectures, architecture_mask, graph_safe, "
        "deterministic, config_json) VALUES (?, 'native', 'mmq', 'native_wrapper', "
        "1, '[]', 0, 1, 1, '{}')",
        (build_id,),
    )
    native_candidate_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
    conn.execute(
        "INSERT INTO candidate (build_id, stable_name, family, source_class, "
        "implementation_version, architectures, architecture_mask, graph_safe, "
        "deterministic, config_json) VALUES (?, 'mmq:fb1', 'mmq', 'existing_alternative', "
        "1, '[]', 0, 1, 1, '{}')",
        (build_id,),
    )
    candidate_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
    return build_id, hardware_id, signature_id, native_candidate_id, candidate_id


class CorrectnessEvidenceSchemaTests(unittest.TestCase):
    def setUp(self):
        self.conn = sqlite3.connect(":memory:")
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.conn.executescript(SCHEMA_SQL.read_text(encoding="utf-8"))

    def tearDown(self):
        self.conn.close()

    def test_tables_exist(self):
        tables = {
            row[0] for row in self.conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        self.assertIn("correctness_evidence", tables)
        self.assertIn("correctness_evidence_seed", tables)

    def test_schema_version_is_6(self):
        row = self.conn.execute(
            "SELECT value FROM schema_meta WHERE key = 'schema_version'"
        ).fetchone()
        self.assertEqual(row[0], "6")

    def test_real_schema_5_database_migrates_to_6_in_place(self):
        # The scenario the migration exists for: a real pre-existing
        # database at schema 5 (post-RE30, with vk_* tables already present
        # and real HIP data) but no correctness_evidence tables yet.
        # Re-applying the current dispatch-db.sql must move it to '6'
        # without touching or losing existing data.
        legacy = sqlite3.connect(":memory:")
        try:
            # Build a real schema-5 database by applying the current script
            # then rolling schema_meta back and dropping the schema-6-only
            # tables -- simpler and less drift-prone than hand-maintaining a
            # second copy of the schema-5 DDL inline.
            legacy.executescript(SCHEMA_SQL.read_text(encoding="utf-8"))
            legacy.execute(
                "INSERT INTO build (source_revision, manifest_hash, signature_schema, "
                "hardware_schema, variant_set) VALUES "
                "('realhistoricalrevision0001', 'realhash', 1, 1, 'inventory')"
            )
            legacy.execute(
                "UPDATE schema_meta SET value = '5' WHERE key = 'schema_version'"
            )
            legacy.execute("DROP TABLE correctness_evidence_seed")
            legacy.execute("DROP TABLE correctness_evidence")
            legacy.commit()

            version = legacy.execute(
                "SELECT value FROM schema_meta WHERE key = 'schema_version'"
            ).fetchone()[0]
            self.assertEqual(version, "5")

            # Re-apply the real, current schema file -- exactly what every
            # lifecycle.py DB-touching call does on every real run.
            legacy.executescript(SCHEMA_SQL.read_text(encoding="utf-8"))

            version = legacy.execute(
                "SELECT value FROM schema_meta WHERE key = 'schema_version'"
            ).fetchone()[0]
            self.assertEqual(version, "6")

            preserved = legacy.execute(
                "SELECT source_revision, manifest_hash FROM build"
            ).fetchone()
            self.assertEqual(preserved, ("realhistoricalrevision0001", "realhash"))

            tables = {
                row[0] for row in legacy.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                )
            }
            self.assertIn("correctness_evidence", tables)
            self.assertIn("correctness_evidence_seed", tables)
        finally:
            legacy.close()

    def test_migration_is_idempotent_once_already_at_6(self):
        self.conn.executescript(SCHEMA_SQL.read_text(encoding="utf-8"))
        row = self.conn.execute(
            "SELECT value FROM schema_meta WHERE key = 'schema_version'"
        ).fetchone()
        self.assertEqual(row[0], "6")

    def test_insert_with_three_seeds_succeeds(self):
        build_id, hardware_id, signature_id, native_id, candidate_id = (
            _seed_build_hardware_signature_candidates(self.conn)
        )
        self.conn.execute(
            "INSERT INTO correctness_evidence (build_id, hardware_id, signature_id, "
            "candidate_id, native_candidate_id, contract_version, threshold_t, "
            "headroom_fraction, e_n_nmse, e_c_nmse, max_abs_native, max_abs_candidate, "
            "seed_count, tool_version) VALUES (?, ?, ?, ?, ?, 'hi67-rv49-v1', 5e-4, 0.5, "
            "1e-5, 2e-5, 0.01, 0.012, 3, 'v1')",
            (build_id, hardware_id, signature_id, candidate_id, native_id),
        )
        evidence_id = self.conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        for seed in (1, 2, 3):
            self.conn.execute(
                "INSERT INTO correctness_evidence_seed (correctness_evidence_id, seed, "
                "reference_digest, e_n_nmse, e_c_nmse, max_abs_native, max_abs_candidate, "
                "native_execution_status, candidate_execution_status) VALUES "
                "(?, ?, ?, 1e-5, 2e-5, 0.01, 0.012, 'ok', 'ok')",
                (evidence_id, seed, f"digest{seed}"),
            )
        self.conn.commit()
        rows = self.conn.execute(
            "SELECT seed FROM correctness_evidence_seed WHERE correctness_evidence_id = ? ORDER BY seed",
            (evidence_id,),
        ).fetchall()
        self.assertEqual([r[0] for r in rows], [1, 2, 3])

    def test_seed_count_below_three_is_rejected(self):
        build_id, hardware_id, signature_id, native_id, candidate_id = (
            _seed_build_hardware_signature_candidates(self.conn)
        )
        with self.assertRaises(sqlite3.IntegrityError):
            self.conn.execute(
                "INSERT INTO correctness_evidence (build_id, hardware_id, signature_id, "
                "candidate_id, native_candidate_id, contract_version, threshold_t, "
                "headroom_fraction, e_n_nmse, e_c_nmse, max_abs_native, max_abs_candidate, "
                "seed_count, tool_version) VALUES (?, ?, ?, ?, ?, 'hi67-rv49-v1', 5e-4, 0.5, "
                "1e-5, 2e-5, 0.01, 0.012, 2, 'v1')",
                (build_id, hardware_id, signature_id, candidate_id, native_id),
            )

    def test_candidate_fk_is_enforced(self):
        build_id, hardware_id, signature_id, native_id, _candidate_id = (
            _seed_build_hardware_signature_candidates(self.conn)
        )
        with self.assertRaises(sqlite3.IntegrityError):
            self.conn.execute(
                "INSERT INTO correctness_evidence (build_id, hardware_id, signature_id, "
                "candidate_id, native_candidate_id, contract_version, threshold_t, "
                "headroom_fraction, e_n_nmse, e_c_nmse, max_abs_native, max_abs_candidate, "
                "seed_count, tool_version) VALUES (?, ?, ?, 999999, ?, 'hi67-rv49-v1', 5e-4, "
                "0.5, 1e-5, 2e-5, 0.01, 0.012, 3, 'v1')",
                (build_id, hardware_id, signature_id, native_id),
            )

    def test_duplicate_contract_version_for_same_identity_is_rejected(self):
        # UNIQUE (build_id, hardware_id, signature_id, candidate_id,
        # contract_version) -- a second evidence-generation run for the
        # identical identity+contract must not silently create a duplicate
        # row a promotion query could pick either of (cherry-picking risk,
        # RV77 Q2 change 6).
        build_id, hardware_id, signature_id, native_id, candidate_id = (
            _seed_build_hardware_signature_candidates(self.conn)
        )
        args = (build_id, hardware_id, signature_id, candidate_id, native_id)
        sql = (
            "INSERT INTO correctness_evidence (build_id, hardware_id, signature_id, "
            "candidate_id, native_candidate_id, contract_version, threshold_t, "
            "headroom_fraction, e_n_nmse, e_c_nmse, max_abs_native, max_abs_candidate, "
            "seed_count, tool_version) VALUES (?, ?, ?, ?, ?, 'hi67-rv49-v1', 5e-4, 0.5, "
            "1e-5, 2e-5, 0.01, 0.012, 3, 'v1')"
        )
        self.conn.execute(sql, args)
        with self.assertRaises(sqlite3.IntegrityError):
            self.conn.execute(sql, args)

    def test_seed_execution_status_vocabulary_is_enforced(self):
        build_id, hardware_id, signature_id, native_id, candidate_id = (
            _seed_build_hardware_signature_candidates(self.conn)
        )
        self.conn.execute(
            "INSERT INTO correctness_evidence (build_id, hardware_id, signature_id, "
            "candidate_id, native_candidate_id, contract_version, threshold_t, "
            "headroom_fraction, e_n_nmse, e_c_nmse, max_abs_native, max_abs_candidate, "
            "seed_count, tool_version) VALUES (?, ?, ?, ?, ?, 'hi67-rv49-v1', 5e-4, 0.5, "
            "1e-5, 2e-5, 0.01, 0.012, 3, 'v1')",
            (build_id, hardware_id, signature_id, candidate_id, native_id),
        )
        evidence_id = self.conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        with self.assertRaises(sqlite3.IntegrityError):
            self.conn.execute(
                "INSERT INTO correctness_evidence_seed (correctness_evidence_id, seed, "
                "reference_digest, e_n_nmse, e_c_nmse, max_abs_native, max_abs_candidate, "
                "native_execution_status, candidate_execution_status) VALUES "
                "(?, 1, 'digest1', 1e-5, 2e-5, 0.01, 0.012, 'maybe', 'ok')",
                (evidence_id,),
            )

    def test_hip_and_vk_tables_still_coexist_unaffected(self):
        tables = {
            row[0] for row in self.conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        for name in ("build", "hardware", "signature", "candidate", "measurement",
                      "winner", "vk_hardware", "vk_winner",
                      "correctness_evidence", "correctness_evidence_seed"):
            self.assertIn(name, tables)


if __name__ == "__main__":
    unittest.main()
