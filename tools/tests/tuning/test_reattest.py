"""HI121 close-out step 7 (HI128): positive re-attestation tests.

Reuses ProjectMeasurementsTests' own fixture (not subclassed, to avoid
re-running its whole inherited suite here) -- it already builds a real
schema-8 DB with a verified build_capability row, two winner rows, and a
matching measurements/manifest pair. These tests strip the fixture's
default attestation (added for replay_projection's own tests) since
reattest.py exists specifically to (re)create it.
"""

from __future__ import annotations

import json
import sqlite3
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from bigcherry.tuning import correctness_evidence as ce  # noqa: E402
from bigcherry.tuning import reattest as ra  # noqa: E402
from bigcherry.tuning import signature_capabilities as sc  # noqa: E402
from bigcherry.tuning import verification_state  # noqa: E402
import test_replay_projection as trp  # noqa: E402




class ReattestWinnersTests(unittest.TestCase):
    def setUp(self):
        trp.ProjectMeasurementsTests.setUp(self)
        self.conn.execute("DELETE FROM winner_verification")
        self.conn.commit()

    def tearDown(self):
        trp.ProjectMeasurementsTests.tearDown(self)

    def _set_source_capabilities(self, mask_hex: str) -> None:
        trp.ProjectMeasurementsTests._set_source_capabilities(self, mask_hex)

    def _rewrite_result(self, signature_hex: str, **updates) -> None:
        trp.ProjectMeasurementsTests._rewrite_result(self, signature_hex, **updates)

    def _db_canonical(self, signature_hex: str) -> dict:
        row = self.conn.execute(
            "SELECT canonical_json FROM signature WHERE signature_digest = ?",
            (bytes.fromhex(signature_hex),),
        ).fetchone()
        return json.loads(row[0])

    def test_missing_inline_canonical_is_reported_not_attested(self):
        # The base fixture's result rows carry no inline "canonical" field
        # and no --signature-source is supplied -- both rows should be
        # reported as missing_canonical, nothing attested.
        self._set_source_capabilities("0000000000000000000000000000001f")
        report = ra.reattest_winners(
            self.dispatch_db, source_build_id=self.build_id,
            measurements_path=self.measurements_path, manifest_path=self.manifest_path,
            signature_digest_verifier=lambda _c: "0" * 32,
        )
        self.assertEqual(report.attested, 0)
        statuses = {o.dispatch: o.status for o in report.outcomes}
        self.assertEqual(statuses[self.s1_result["dispatch"]], "missing_canonical")

    def test_matching_canonical_and_verifier_attests_row(self):
        self._set_source_capabilities("0000000000000000000000000000001f")
        canonical = self._db_canonical(self.s1_hex)
        self._rewrite_result(self.s1_hex, canonical=canonical)

        report = ra.reattest_winners(
            self.dispatch_db, source_build_id=self.build_id,
            measurements_path=self.measurements_path, manifest_path=self.manifest_path,
            signature_digest_verifier=lambda c: self.s1_hex if c == canonical else "f" * 32,
        )
        self.assertEqual(report.attested, 1)
        statuses = {o.dispatch: o.status for o in report.outcomes}
        self.assertEqual(statuses[self.s1_result["dispatch"]], "attested")

        winner_id = self.conn.execute(
            "SELECT winner_id FROM winner WHERE signature_id = "
            "(SELECT signature_id FROM signature WHERE signature_digest = ?)",
            (bytes.fromhex(self.s1_hex),),
        ).fetchone()[0]
        self.assertTrue(verification_state.is_winner_verified(self.conn, winner_id=winner_id))

    def test_canonical_disagreeing_with_db_is_rejected(self):
        self._set_source_capabilities("0000000000000000000000000000001f")
        tampered = dict(self._db_canonical(self.s1_hex))
        tampered["op"] = 999
        self._rewrite_result(self.s1_hex, canonical=tampered)

        report = ra.reattest_winners(
            self.dispatch_db, source_build_id=self.build_id,
            measurements_path=self.measurements_path, manifest_path=self.manifest_path,
            signature_digest_verifier=lambda _c: self.s1_hex,
        )
        self.assertEqual(report.attested, 0)
        statuses = {o.dispatch: o.status for o in report.outcomes}
        self.assertEqual(statuses[self.s1_result["dispatch"]], "canonical_disagrees_with_db")

    def test_verifier_digest_mismatch_is_rejected(self):
        self._set_source_capabilities("0000000000000000000000000000001f")
        canonical = self._db_canonical(self.s1_hex)
        self._rewrite_result(self.s1_hex, canonical=canonical)

        report = ra.reattest_winners(
            self.dispatch_db, source_build_id=self.build_id,
            measurements_path=self.measurements_path, manifest_path=self.manifest_path,
            signature_digest_verifier=lambda _c: "f" * 32,  # wrong digest entirely
        )
        self.assertEqual(report.attested, 0)
        statuses = {o.dispatch: o.status for o in report.outcomes}
        self.assertEqual(statuses[self.s1_result["dispatch"]], "signature_verification_failed")

    def test_dry_run_verifies_but_writes_nothing(self):
        self._set_source_capabilities("0000000000000000000000000000001f")
        canonical = self._db_canonical(self.s1_hex)
        self._rewrite_result(self.s1_hex, canonical=canonical)

        report = ra.reattest_winners(
            self.dispatch_db, source_build_id=self.build_id,
            measurements_path=self.measurements_path, manifest_path=self.manifest_path,
            signature_digest_verifier=lambda c: self.s1_hex if c == canonical else "f" * 32,
            dry_run=True,
        )
        self.assertEqual(report.attested, 0)
        statuses = {o.dispatch: o.status for o in report.outcomes}
        self.assertEqual(statuses[self.s1_result["dispatch"]], "would_attest")
        self.assertEqual(
            self.conn.execute("SELECT COUNT(*) FROM winner_verification").fetchone()[0], 0,
        )

    def test_already_attested_row_is_skipped(self):
        self._set_source_capabilities("0000000000000000000000000000001f")
        winner_id = self.conn.execute(
            "SELECT winner_id FROM winner WHERE signature_id = "
            "(SELECT signature_id FROM signature WHERE signature_digest = ?)",
            (bytes.fromhex(self.s1_hex),),
        ).fetchone()[0]
        verification_state.record_winner_verification(self.conn, winner_id=winner_id)
        self.conn.commit()

        report = ra.reattest_winners(
            self.dispatch_db, source_build_id=self.build_id,
            measurements_path=self.measurements_path, manifest_path=self.manifest_path,
            signature_digest_verifier=lambda _c: self.s1_hex,
        )
        self.assertEqual(report.attested, 0)
        statuses = {o.dispatch: o.status for o in report.outcomes}
        self.assertEqual(statuses[self.s1_result["dispatch"]], "already_attested")

    def test_projected_artifact_is_rejected_outright(self):
        self._set_source_capabilities("0000000000000000000000000000001f")
        lines = self.measurements_path.read_text(encoding="utf-8").splitlines()
        header = json.loads(lines[0])
        header["hi121_source_provenance"] = {"source_build_id": self.build_id}
        lines[0] = json.dumps(header)
        self.measurements_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

        with self.assertRaises(ra.ReattestationError):
            ra.reattest_winners(
                self.dispatch_db, source_build_id=self.build_id,
                measurements_path=self.measurements_path, manifest_path=self.manifest_path,
                signature_digest_verifier=lambda _c: self.s1_hex,
            )

    def test_forged_row_is_reported_not_attested(self):
        self._set_source_capabilities("0000000000000000000000000000001f")
        canonical = self._db_canonical(self.s1_hex)
        self._rewrite_result(self.s1_hex, canonical=canonical, winner="not-a-real-candidate")

        report = ra.reattest_winners(
            self.dispatch_db, source_build_id=self.build_id,
            measurements_path=self.measurements_path, manifest_path=self.manifest_path,
            signature_digest_verifier=lambda _c: self.s1_hex,
        )
        self.assertEqual(report.attested, 0)
        statuses = {o.dispatch: o.status for o in report.outcomes}
        self.assertEqual(statuses[self.s1_result["dispatch"]], "row_no_longer_matches_db")

    # -- HI128 review round 2 (dev-gpt-agent): historical-build backfill --

    def test_historical_build_missing_capability_backfills_and_attests(self):
        # No _set_source_capabilities() call at all -- this build has NO
        # build_capability row, exactly the pre-HI121 UNKNOWN state HI128
        # exists to remediate. The manifest/header still carry a real
        # producer_capabilities claim, so Phase B should backfill it.
        canonical = self._db_canonical(self.s1_hex)
        self._rewrite_result(self.s1_hex, canonical=canonical)

        report = ra.reattest_winners(
            self.dispatch_db, source_build_id=self.build_id,
            measurements_path=self.measurements_path, manifest_path=self.manifest_path,
            signature_digest_verifier=lambda _c: self.s1_hex,
        )
        self.assertEqual(report.attested, 1)
        self.assertTrue(report.backfilled_build_capability)
        row = self.conn.execute(
            "SELECT producer_capabilities FROM build_capability WHERE build_id = ? AND backend = 'hip'",
            (self.build_id,),
        ).fetchone()
        self.assertEqual(row[0], bytes.fromhex("0000000000000000000000000000001f"))

    def test_null_build_descriptor_hash_backfills(self):
        self.conn.execute(
            "UPDATE build SET build_descriptor_hash = NULL WHERE build_id = ?", (self.build_id,),
        )
        self.conn.commit()
        canonical = self._db_canonical(self.s1_hex)
        self._rewrite_result(self.s1_hex, canonical=canonical)

        report = ra.reattest_winners(
            self.dispatch_db, source_build_id=self.build_id,
            measurements_path=self.measurements_path, manifest_path=self.manifest_path,
            signature_digest_verifier=lambda _c: self.s1_hex,
        )
        self.assertEqual(report.attested, 1)
        self.assertTrue(report.backfilled_build_descriptor)
        descriptor_hash = self.conn.execute(
            "SELECT build_descriptor_hash FROM build WHERE build_id = ?", (self.build_id,),
        ).fetchone()[0]
        self.assertEqual(descriptor_hash, self.manifest["build_descriptor"]["descriptor_hash"])

    def test_conflicting_existing_capability_aborts_with_zero_attestations(self):
        self._set_source_capabilities("00000000000000000000000000000001")  # wrong mask
        canonical = self._db_canonical(self.s1_hex)
        self._rewrite_result(self.s1_hex, canonical=canonical)

        with self.assertRaises(ra.ReattestationError):
            ra.reattest_winners(
                self.dispatch_db, source_build_id=self.build_id,
                measurements_path=self.measurements_path, manifest_path=self.manifest_path,
                signature_digest_verifier=lambda _c: self.s1_hex,
            )
        self.assertEqual(
            self.conn.execute("SELECT COUNT(*) FROM winner_verification").fetchone()[0], 0,
        )

    def test_conflicting_existing_descriptor_aborts_with_zero_attestations(self):
        self.conn.execute(
            "UPDATE build SET build_descriptor_hash = 'not-the-real-descriptor' WHERE build_id = ?",
            (self.build_id,),
        )
        self.conn.commit()
        canonical = self._db_canonical(self.s1_hex)
        self._rewrite_result(self.s1_hex, canonical=canonical)

        with self.assertRaises(ra.ReattestationError):
            ra.reattest_winners(
                self.dispatch_db, source_build_id=self.build_id,
                measurements_path=self.measurements_path, manifest_path=self.manifest_path,
                signature_digest_verifier=lambda _c: self.s1_hex,
            )
        self.assertEqual(
            self.conn.execute("SELECT COUNT(*) FROM winner_verification").fetchone()[0], 0,
        )

    def test_case_insensitive_verifier_digest_is_accepted(self):
        self._set_source_capabilities("0000000000000000000000000000001f")
        canonical = self._db_canonical(self.s1_hex)
        self._rewrite_result(self.s1_hex, canonical=canonical)

        report = ra.reattest_winners(
            self.dispatch_db, source_build_id=self.build_id,
            measurements_path=self.measurements_path, manifest_path=self.manifest_path,
            signature_digest_verifier=lambda _c: self.s1_hex.upper(),
        )
        self.assertEqual(report.attested, 1)

    def test_unsupported_signature_domain_is_not_mislabeled_as_verification_failure(self):
        self._set_source_capabilities("0000000000000000000000000000001f")
        canonical = self._db_canonical(self.s1_hex)
        self._rewrite_result(self.s1_hex, canonical=canonical)

        def raising_verifier(_c):
            raise sc.UnsupportedSignatureDomain("outside HI121's audited domain")

        report = ra.reattest_winners(
            self.dispatch_db, source_build_id=self.build_id,
            measurements_path=self.measurements_path, manifest_path=self.manifest_path,
            signature_digest_verifier=raising_verifier,
        )
        statuses = {o.dispatch: o.status for o in report.outcomes}
        self.assertEqual(statuses[self.s1_result["dispatch"]], "unsupported_signature_domain")

    def test_evidence_error_is_reported_as_verification_failure_not_unsupported(self):
        self._set_source_capabilities("0000000000000000000000000000001f")
        canonical = self._db_canonical(self.s1_hex)
        self._rewrite_result(self.s1_hex, canonical=canonical)

        def raising_verifier(_c):
            raise ce.EvidenceError("observed canonical does not match supplied canonical")

        report = ra.reattest_winners(
            self.dispatch_db, source_build_id=self.build_id,
            measurements_path=self.measurements_path, manifest_path=self.manifest_path,
            signature_digest_verifier=raising_verifier,
        )
        statuses = {o.dispatch: o.status for o in report.outcomes}
        self.assertEqual(statuses[self.s1_result["dispatch"]], "signature_verification_failed")

    def test_build_capability_mutated_during_verification_aborts_phase_b(self):
        # TOCTOU: simulate another process altering build_capability while
        # THIS verifier call (Phase A's real GPU subprocess, in production)
        # is in flight, by mutating the DB through a second connection as
        # a side effect of the verifier itself. Phase B's build-provenance
        # recheck must catch this and abort the WHOLE pass, attesting
        # nothing, rather than trusting the now-stale Phase-A proof.
        self._set_source_capabilities("0000000000000000000000000000001f")
        canonical = self._db_canonical(self.s1_hex)
        self._rewrite_result(self.s1_hex, canonical=canonical)

        def mutating_verifier(_c):
            extra = sqlite3.connect(str(self.dispatch_db))
            try:
                extra.execute(
                    "UPDATE build_capability SET producer_capabilities = ? "
                    "WHERE build_id = ? AND backend = 'hip'",
                    (bytes.fromhex("00000000000000000000000000000001"), self.build_id),
                )
                extra.commit()
            finally:
                extra.close()
            return self.s1_hex

        with self.assertRaises(ra.ReattestationError):
            ra.reattest_winners(
                self.dispatch_db, source_build_id=self.build_id,
                measurements_path=self.measurements_path, manifest_path=self.manifest_path,
                signature_digest_verifier=mutating_verifier,
            )
        self.assertEqual(
            self.conn.execute("SELECT COUNT(*) FROM winner_verification").fetchone()[0], 0,
        )

    def test_dry_run_detects_row_changed_during_final_check(self):
        # The dry-run's second, final-state read-only pass must catch a row
        # whose identity changed after Phase A's own check -- simulated
        # here by having the verifier itself replace the winner row (via a
        # second connection) with a different candidate name, breaking the
        # row-binding the final check re-verifies.
        self._set_source_capabilities("0000000000000000000000000000001f")
        canonical = self._db_canonical(self.s1_hex)
        self._rewrite_result(self.s1_hex, canonical=canonical)

        def mutating_verifier(_c):
            extra = sqlite3.connect(str(self.dispatch_db))
            try:
                extra.execute(
                    "UPDATE winner SET stable_name = 'mmq:generated:other' "
                    "WHERE build_id = ? AND signature_id = "
                    "(SELECT signature_id FROM signature WHERE signature_digest = ?)",
                    (self.build_id, bytes.fromhex(self.s1_hex)),
                )
                extra.commit()
            finally:
                extra.close()
            return self.s1_hex

        report = ra.reattest_winners(
            self.dispatch_db, source_build_id=self.build_id,
            measurements_path=self.measurements_path, manifest_path=self.manifest_path,
            signature_digest_verifier=mutating_verifier,
            dry_run=True,
        )
        statuses = {o.dispatch: o.status for o in report.outcomes}
        self.assertEqual(statuses[self.s1_result["dispatch"]], "changed_during_remediation")
        self.assertEqual(
            self.conn.execute("SELECT COUNT(*) FROM winner_verification").fetchone()[0], 0,
        )


if __name__ == "__main__":
    unittest.main()
