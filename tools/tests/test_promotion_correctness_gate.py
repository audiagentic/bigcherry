"""HI67 slice 3: promotion_correctness_gate.py's read-only identity
resolution and hard-AND correctness gating, against a real schema-6 sqlite
database (no upserts -- this module only ever SELECTs rows a real
inventory.load_measurements() ingestion would already have written)."""

from __future__ import annotations

import sqlite3
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bigcherry import correctness_evidence as ce  # noqa: E402
from bigcherry import promotion_correctness_gate as gate  # noqa: E402

SCHEMA_SQL = Path(__file__).resolve().parents[2] / "sql" / "dispatch-db.sql"

DISPATCH_HEX = "11" * 16
SIGNATURE_HEX = "22" * 16
HARDWARE_HEX = "33" * 16


class _Base(unittest.TestCase):
    def setUp(self):
        self.conn = sqlite3.connect(":memory:")
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.conn.executescript(SCHEMA_SQL.read_text(encoding="utf-8"))
        self.conn.execute(
            "INSERT INTO build (source_revision, manifest_hash, signature_schema, "
            "hardware_schema, variant_set) VALUES ('deadbeefdeadbeefdead', 'aa', 1, 1, 'inventory')"
        )
        self.build_id = self.conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        self.conn.execute(
            "INSERT INTO hardware (hardware_digest, architecture, architecture_code, "
            "wave_size, compute_units, feature_flags, canonical_json) VALUES "
            "(?, 'gfx1100', 1, 32, 96, 0, '{}')",
            (bytes.fromhex(HARDWARE_HEX),),
        )
        self.hardware_id = self.conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        self.conn.execute(
            "INSERT INTO signature (signature_digest, base_digest, schema_version, op, "
            "src0_type, src1_type, dst_type, m, n, k, canonical_json) VALUES "
            "(?, x'02', 1, 'MUL_MAT', 'q8_0', 'f32', 'f32', 1, 1, 1, '{}')",
            (bytes.fromhex(SIGNATURE_HEX),),
        )
        self.signature_id = self.conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        self.conn.execute(
            "INSERT INTO candidate (build_id, stable_name, family, source_class, "
            "implementation_version, architectures, architecture_mask, graph_safe, "
            "deterministic, config_json) VALUES (?, 'native', 'mmq', 'native_wrapper', "
            "1, '[]', 0, 1, 1, '{}')",
            (self.build_id,),
        )
        self.native_candidate_id = self.conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        self.conn.execute(
            "INSERT INTO candidate (build_id, stable_name, family, source_class, "
            "implementation_version, architectures, architecture_mask, graph_safe, "
            "deterministic, config_json) VALUES (?, 'mmq:fb1', 'mmq', 'existing_alternative', "
            "1, '[]', 0, 1, 1, '{}')",
            (self.build_id,),
        )
        self.candidate_id = self.conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        self.conn.execute(
            "INSERT INTO measurement (build_id, hardware_id, signature_id, dispatch_digest, "
            "candidate_id, objective, stage, accepted) VALUES (?, ?, ?, ?, ?, 'latency', "
            "'final', 1)",
            (self.build_id, self.hardware_id, self.signature_id,
             bytes.fromhex(DISPATCH_HEX), self.candidate_id),
        )
        self.conn.commit()

    def _write_evidence(self, *, seeds=3, headroom_fraction=0.5,
                         contract_version=ce.CONTRACT_VERSION,
                         native_candidate_id=None, e_c_nmse=2e-05,
                         max_abs_candidate=0.0009):
        rows = [
            ce.SeedEvidence(
                seed=i, reference_digest=f"d{i}", e_n_nmse=1e-05, e_c_nmse=e_c_nmse,
                max_abs_native=0.001, max_abs_candidate=max_abs_candidate,
                native_execution_status="ok", candidate_execution_status="ok",
                threshold_t=5e-4,
            )
            for i in range(1, seeds + 1)
        ]
        aggregate = ce.aggregate_seed_evidence(
            rows, headroom_fraction=headroom_fraction, contract_version=contract_version,
        ) if seeds >= 3 else None
        if aggregate is None:
            # Deliberately write an under-seeded row directly (bypassing
            # aggregate_seed_evidence's own >=3 guard) to test the gate's
            # OWN defense against a parent row claiming too few seeds.
            self.conn.execute(
                "INSERT INTO correctness_evidence (build_id, hardware_id, signature_id, "
                "candidate_id, native_candidate_id, contract_version, threshold_t, "
                "headroom_fraction, e_n_nmse, e_c_nmse, max_abs_native, max_abs_candidate, "
                "seed_count, tool_version) VALUES (?, ?, ?, ?, ?, ?, 5e-4, ?, 1e-05, ?, 0.001, "
                "?, ?, 'v1')",
                (self.build_id, self.hardware_id, self.signature_id, self.candidate_id,
                 native_candidate_id or self.native_candidate_id, contract_version,
                 headroom_fraction, e_c_nmse, max_abs_candidate, seeds),
            )
            evidence_id = self.conn.execute("SELECT last_insert_rowid()").fetchone()[0]
            for i in range(1, seeds + 1):
                self.conn.execute(
                    "INSERT INTO correctness_evidence_seed (correctness_evidence_id, seed, "
                    "reference_digest, e_n_nmse, e_c_nmse, max_abs_native, max_abs_candidate, "
                    "native_execution_status, candidate_execution_status, threshold_t) VALUES "
                    "(?, ?, ?, 1e-05, ?, 0.001, ?, 'ok', 'ok', 5e-4)",
                    (evidence_id, i, f"d{i}", e_c_nmse, max_abs_candidate),
                )
            self.conn.commit()
            return
        self.conn.execute(
            "INSERT INTO correctness_evidence (build_id, hardware_id, signature_id, "
            "candidate_id, native_candidate_id, contract_version, threshold_t, "
            "headroom_fraction, e_n_nmse, e_c_nmse, max_abs_native, max_abs_candidate, "
            "seed_count, tool_version) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'v1')",
            (self.build_id, self.hardware_id, self.signature_id, self.candidate_id,
             native_candidate_id or self.native_candidate_id, aggregate.contract_version,
             aggregate.threshold_t, aggregate.headroom_fraction, aggregate.e_n_nmse,
             aggregate.e_c_nmse, aggregate.max_abs_native, aggregate.max_abs_candidate,
             len(aggregate.seed_rows)),
        )
        evidence_id = self.conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        for row in aggregate.seed_rows:
            self.conn.execute(
                "INSERT INTO correctness_evidence_seed (correctness_evidence_id, seed, "
                "reference_digest, e_n_nmse, e_c_nmse, max_abs_native, max_abs_candidate, "
                "native_execution_status, candidate_execution_status, threshold_t) VALUES "
                "(?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (evidence_id, row.seed, row.reference_digest, row.e_n_nmse, row.e_c_nmse,
                 row.max_abs_native, row.max_abs_candidate, row.native_execution_status,
                 row.candidate_execution_status, row.threshold_t),
            )
        self.conn.commit()

    def tearDown(self):
        self.conn.close()


class ResolveIdentityTests(_Base):
    def test_resolves_real_identity_from_ingested_rows(self):
        identity = gate.resolve_promotion_identity(
            self.conn, dispatch_hex=DISPATCH_HEX, signature_hex=SIGNATURE_HEX,
            hardware_hex=HARDWARE_HEX, native_name="native", candidate_name="mmq:fb1",
        )
        self.assertEqual(identity.build_id, self.build_id)
        self.assertEqual(identity.hardware_id, self.hardware_id)
        self.assertEqual(identity.signature_id, self.signature_id)
        self.assertEqual(identity.native_candidate_id, self.native_candidate_id)
        self.assertEqual(identity.candidate_id, self.candidate_id)

    def test_unknown_signature_digest_fails_closed(self):
        with self.assertRaises(gate.CorrectnessGateError):
            gate.resolve_promotion_identity(
                self.conn, dispatch_hex=DISPATCH_HEX, signature_hex="ff" * 16,
                hardware_hex=HARDWARE_HEX, native_name="native", candidate_name="mmq:fb1",
            )

    def test_unknown_candidate_name_fails_closed(self):
        with self.assertRaises(gate.CorrectnessGateError):
            gate.resolve_promotion_identity(
                self.conn, dispatch_hex=DISPATCH_HEX, signature_hex=SIGNATURE_HEX,
                hardware_hex=HARDWARE_HEX, native_name="native", candidate_name="no_such_candidate",
            )

    def test_no_matching_measurement_row_fails_closed(self):
        with self.assertRaises(gate.CorrectnessGateError):
            gate.resolve_promotion_identity(
                self.conn, dispatch_hex="99" * 16, signature_hex=SIGNATURE_HEX,
                hardware_hex=HARDWARE_HEX, native_name="native", candidate_name="mmq:fb1",
            )


class EvaluateGateTests(_Base):
    def _identity(self):
        return gate.resolve_promotion_identity(
            self.conn, dispatch_hex=DISPATCH_HEX, signature_hex=SIGNATURE_HEX,
            hardware_hex=HARDWARE_HEX, native_name="native", candidate_name="mmq:fb1",
        )

    def test_missing_evidence_fails_closed(self):
        passed, status = gate.evaluate_correctness_gate(self.conn, self._identity())
        self.assertFalse(passed)
        self.assertEqual(status, "rejected_no_correctness_evidence")

    def test_valid_evidence_within_headroom_passes(self):
        self._write_evidence()
        passed, status = gate.evaluate_correctness_gate(self.conn, self._identity())
        self.assertTrue(passed)
        self.assertEqual(status, "")

    def test_wrong_contract_version_fails_closed(self):
        self._write_evidence(contract_version="some-other-contract-v9")
        passed, status = gate.evaluate_correctness_gate(self.conn, self._identity())
        self.assertFalse(passed)
        self.assertEqual(status, "rejected_correctness_contract")

    def test_wrong_headroom_fraction_fails_closed(self):
        # A permissive fraction (1.0) recorded by a buggy/malicious producer
        # must not self-authorize -- promotion requires ITS OWN expected
        # fraction to match, not whatever the row claims.
        self._write_evidence(headroom_fraction=1.0)
        passed, status = gate.evaluate_correctness_gate(self.conn, self._identity())
        self.assertFalse(passed)
        self.assertEqual(status, "rejected_correctness_contract")

    def test_evidence_for_a_different_native_baseline_fails_closed(self):
        other_native_id = self.conn.execute(
            "INSERT INTO candidate (build_id, stable_name, family, source_class, "
            "implementation_version, architectures, architecture_mask, graph_safe, "
            "deterministic, config_json) VALUES (?, 'other_native', 'mmq', "
            "'native_wrapper', 1, '[]', 0, 1, 1, '{}')",
            (self.build_id,),
        ).lastrowid
        self._write_evidence(native_candidate_id=other_native_id)
        passed, status = gate.evaluate_correctness_gate(self.conn, self._identity())
        self.assertFalse(passed)
        self.assertEqual(status, "rejected_correctness_contract")

    def test_under_seeded_evidence_is_rejected_at_the_schema_layer(self):
        # sql/dispatch-db.sql's own CHECK (seed_count >= 3) already makes an
        # under-seeded correctness_evidence row impossible to insert at
        # all -- the strongest possible fail-closed guarantee, enforced
        # before this module's own logic ever runs.
        with self.assertRaises(sqlite3.IntegrityError):
            self._write_evidence(seeds=2)

    def test_candidate_exceeding_headroom_fails_the_rv49_predicate(self):
        # Native uses almost none of its budget; candidate blows past the
        # 50%-remaining-headroom rule even though it is still < T.
        self._write_evidence(e_c_nmse=4.9e-04)
        passed, status = gate.evaluate_correctness_gate(self.conn, self._identity())
        self.assertFalse(passed)
        self.assertEqual(status, "rejected_correctness")

    def test_tampered_parent_aggregate_disagreeing_with_seeds_fails_closed(self):
        self._write_evidence()
        # Simulate a corrupted/tampered parent row: its claimed e_c_nmse no
        # longer matches what its own seed children recompute to.
        self.conn.execute(
            "UPDATE correctness_evidence SET e_c_nmse = 9.0 "
            "WHERE build_id = ? AND candidate_id = ?",
            (self.build_id, self.candidate_id),
        )
        self.conn.commit()
        passed, status = gate.evaluate_correctness_gate(self.conn, self._identity())
        self.assertFalse(passed)
        self.assertEqual(status, "rejected_correctness_contract")


class CorrectnessBindingTests(_Base):
    """HI67 follow-on: CorrectnessBinding/resolve_correctness_binding/
    require_correctness_binding are a purely additive, named-wrapper layer
    around resolve_promotion_identity()/evaluate_correctness_gate() --
    intended for a future caller (the replay-cache exporter) that needs
    "prove this exact binding is production-safe or stop" as one call.
    Not yet wired into any production path in this commit."""

    def _binding(self, *, candidate_name="mmq:fb1"):
        return gate.CorrectnessBinding(
            dispatch_hex=DISPATCH_HEX, signature_hex=SIGNATURE_HEX,
            hardware_hex=HARDWARE_HEX, native_name="native", candidate_name=candidate_name,
        )

    def test_resolve_correctness_binding_matches_resolve_promotion_identity(self):
        via_binding = gate.resolve_correctness_binding(self.conn, self._binding())
        via_kwargs = gate.resolve_promotion_identity(
            self.conn, dispatch_hex=DISPATCH_HEX, signature_hex=SIGNATURE_HEX,
            hardware_hex=HARDWARE_HEX, native_name="native", candidate_name="mmq:fb1",
        )
        self.assertEqual(via_binding, via_kwargs)

    def test_resolve_correctness_binding_fails_closed_on_unknown_candidate(self):
        with self.assertRaises(gate.CorrectnessGateError):
            gate.resolve_correctness_binding(self.conn, self._binding(candidate_name="nope"))

    def test_require_correctness_binding_passes_with_valid_evidence(self):
        self._write_evidence()
        identity = gate.require_correctness_binding(self.conn, self._binding())
        self.assertEqual(identity.candidate_id, self.candidate_id)

    def test_require_correctness_binding_raises_with_no_evidence(self):
        with self.assertRaises(gate.CorrectnessGateError) as ctx:
            gate.require_correctness_binding(self.conn, self._binding())
        self.assertIn("rejected_no_correctness_evidence", str(ctx.exception))

    def test_require_correctness_binding_raises_when_gate_fails(self):
        self._write_evidence(e_c_nmse=4.9e-04)  # blows past the headroom rule
        with self.assertRaises(gate.CorrectnessGateError) as ctx:
            gate.require_correctness_binding(self.conn, self._binding())
        self.assertIn("rejected_correctness", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
