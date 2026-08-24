"""HI67 slice 2c/3: correctness_evidence.py's parsing, fail-closed digest
matching, worst-of-seeds aggregation, and DB write, exercised against a
faked subprocess.run -- no real HIP hardware or compiled test-backend-ops
binary needed for this layer's own correctness (the subprocess argv/env
construction and the on-disk schema are each verified separately)."""

from __future__ import annotations

import sqlite3
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bigcherry import correctness_evidence as ce  # noqa: E402

SCHEMA_SQL = Path(__file__).resolve().parents[2] / "sql" / "dispatch-db.sql"


def _completed(returncode: int, stderr: str):
    result = MagicMock()
    result.returncode = returncode
    result.stderr = stderr
    result.stdout = ""
    return result


def _metric_line(*, tensor="dst", err, max_abs, threshold=5e-4, digests=None):
    line = (
        f"BIGCHERRY_CORRECTNESS_METRIC op=MUL_MAT tensor={tensor} "
        f"backend1=HIP0 backend2=CPU err={err} max_abs={max_abs} "
        f"threshold={threshold} n=1024"
    )
    if digests is not None:
        backend1_digest, backend2_digest = digests
        line += f" backend1_digest={backend1_digest} backend2_digest={backend2_digest}"
    return line + "\n"


def _digest_line(*, name="dst", call_index=0, digest="deadbeef00000000", nels=1024):
    return f"BIGCHERRY_REF_DIGEST name={name} call_index={call_index} digest={digest} nels={nels}\n"


class ParsingTests(unittest.TestCase):
    def test_parse_ref_digests(self):
        text = _digest_line(name="a", digest="0011") + _digest_line(name="b", digest="2233")
        digests = ce.parse_ref_digests(text)
        self.assertEqual([d.name for d in digests], ["a", "b"])
        self.assertEqual([d.digest for d in digests], ["0011", "2233"])

    def test_parse_correctness_metrics(self):
        text = _metric_line(err="1e-05", max_abs="0.001")
        metrics = ce.parse_correctness_metrics(text)
        self.assertEqual(len(metrics), 1)
        self.assertAlmostEqual(metrics[0].err, 1e-05)
        self.assertAlmostEqual(metrics[0].max_abs, 0.001)
        self.assertAlmostEqual(metrics[0].threshold, 5e-4)

    def test_parse_correctness_metrics_without_digests_defaults_to_none(self):
        text = _metric_line(err="1e-05", max_abs="0.001")
        metrics = ce.parse_correctness_metrics(text)
        self.assertEqual(len(metrics), 1)
        self.assertIsNone(metrics[0].backend1_digest)
        self.assertIsNone(metrics[0].backend2_digest)

    def test_parse_correctness_metrics_with_digests(self):
        text = _metric_line(
            err="1e-05", max_abs="0.001",
            digests=("DEADBEEF00000001", "00000000CAFEF00D"),
        )
        metrics = ce.parse_correctness_metrics(text)
        self.assertEqual(len(metrics), 1)
        self.assertEqual(metrics[0].backend1_digest, "deadbeef00000001")
        self.assertEqual(metrics[0].backend2_digest, "00000000cafef00d")

    def test_parsing_ignores_unrelated_lines(self):
        text = "some other stderr noise\n" + _digest_line() + "more noise\n"
        self.assertEqual(len(ce.parse_ref_digests(text)), 1)


class RunEnvironmentTests(unittest.TestCase):
    """Confirms the source-verified dispatch-mode mechanism: native uses
    GGML_HIP_DISPATCH_MODE=native with no force var; a candidate uses
    GGML_HIP_DISPATCH_MODE=replay + GGML_HIP_FORCE_CANDIDATE -- NOT
    GGML_HIP_FORCE_CANDIDATE=native, which does not work (see
    correctness_evidence.py's module docstring for the source citations)."""

    def test_native_run_sets_native_mode_and_no_force_candidate(self):
        captured = {}

        def fake_runner(argv, **kwargs):
            captured["argv"] = argv
            captured["env"] = kwargs["env"]
            return _completed(0, "")

        ce.run_test_backend_ops(
            Path("/bin/test-backend-ops"), op_filter="m=1,n=1,k=1", seed=7,
            dispatch_mode="native", forced_candidate=None, runner=fake_runner,
        )
        self.assertEqual(captured["env"]["GGML_HIP_DISPATCH_MODE"], "native")
        self.assertNotIn("GGML_HIP_FORCE_CANDIDATE", captured["env"])
        self.assertNotIn("GGML_HIP_FORCE_CANDIDATE_STRICT", captured["env"])
        self.assertEqual(captured["env"]["BIGCHERRY_TEST_DETERMINISTIC_SEED"], "7")
        self.assertEqual(
            captured["argv"],
            [str(Path("/bin/test-backend-ops")), "test", "-o", "MUL_MAT", "-p", "m=1,n=1,k=1"],
        )

    def test_candidate_run_sets_replay_mode_and_force_candidate(self):
        captured = {}

        def fake_runner(argv, **kwargs):
            captured["env"] = kwargs["env"]
            return _completed(0, "")

        ce.run_test_backend_ops(
            Path("/bin/test-backend-ops"), op_filter="m=1,n=1,k=1", seed=7,
            dispatch_mode="replay", forced_candidate="mmq:fb1", runner=fake_runner,
        )
        self.assertEqual(captured["env"]["GGML_HIP_DISPATCH_MODE"], "replay")
        self.assertEqual(captured["env"]["GGML_HIP_FORCE_CANDIDATE"], "mmq:fb1")
        self.assertEqual(
            captured["env"]["GGML_HIP_FORCE_CANDIDATE_STRICT"], "1",
            "HI105: a candidate run must fail closed (not silently fall back to "
            "ordinary resolution) if the named candidate is unregistered/ineligible -- "
            "dispatch.cu's own comment names this producer as the reason STRICT exists",
        )

    def test_zero_seed_is_rejected(self):
        with self.assertRaises(ce.EvidenceError):
            ce.run_test_backend_ops(
                Path("/bin/x"), op_filter="", seed=0, dispatch_mode="native",
                forced_candidate=None, runner=lambda *a, **k: _completed(0, ""),
            )


class CollectSeedEvidenceTests(unittest.TestCase):
    def _runner_pair(self, native_stderr: str, candidate_stderr: str, *, native_rc=0, candidate_rc=0):
        calls = []

        def runner(argv, **kwargs):
            env = kwargs["env"]
            calls.append(env)
            if env["GGML_HIP_DISPATCH_MODE"] == "native":
                return _completed(native_rc, native_stderr)
            return _completed(candidate_rc, candidate_stderr)

        return runner, calls

    def test_matched_digests_and_metrics_produce_a_seed_row(self):
        native_stderr = _digest_line(digest="abc123") + _metric_line(err="1e-05", max_abs="0.001")
        candidate_stderr = _digest_line(digest="abc123") + _metric_line(err="2e-05", max_abs="0.0012")
        runner, _ = self._runner_pair(native_stderr, candidate_stderr)

        row = ce.collect_seed_evidence(
            Path("/bin/x"), op_filter="m=1,n=1,k=1", target_tensor="dst",
            candidate_stable_name="mmq:fb1", seed=3, runner=runner,
        )
        self.assertEqual(row.seed, 3)
        self.assertEqual(row.reference_digest, "abc123")
        self.assertAlmostEqual(row.e_n_nmse, 1e-05)
        self.assertAlmostEqual(row.e_c_nmse, 2e-05)
        self.assertEqual(row.native_execution_status, "ok")
        self.assertEqual(row.candidate_execution_status, "ok")

    def test_digest_mismatch_fails_closed(self):
        native_stderr = _digest_line(digest="aaaa") + _metric_line(err="1e-05", max_abs="0.001")
        candidate_stderr = _digest_line(digest="bbbb") + _metric_line(err="2e-05", max_abs="0.0012")
        runner, _ = self._runner_pair(native_stderr, candidate_stderr)

        with self.assertRaises(ce.EvidenceError) as ctx:
            ce.collect_seed_evidence(
                Path("/bin/x"), op_filter="m=1,n=1,k=1", target_tensor="dst",
                candidate_stable_name="mmq:fb1", seed=3, runner=runner,
            )
        self.assertIn("DIFFERENT CPU-reference", str(ctx.exception))

    def test_threshold_mismatch_between_native_and_candidate_fails_closed(self):
        native_stderr = _digest_line(digest="abc123") + _metric_line(err="1e-05", max_abs="0.001", threshold=5e-4)
        candidate_stderr = _digest_line(digest="abc123") + _metric_line(err="2e-05", max_abs="0.0012", threshold=1e-3)
        runner, _ = self._runner_pair(native_stderr, candidate_stderr)

        with self.assertRaises(ce.EvidenceError) as ctx:
            ce.collect_seed_evidence(
                Path("/bin/x"), op_filter="m=1,n=1,k=1", target_tensor="dst",
                candidate_stable_name="mmq:fb1", seed=3, runner=runner,
            )
        self.assertIn("DIFFERENT upstream correctness thresholds", str(ctx.exception))

    def test_seed_row_carries_threshold_from_the_metric_line(self):
        native_stderr = _digest_line(digest="abc123") + _metric_line(err="1e-05", max_abs="0.001", threshold=1e-3)
        candidate_stderr = _digest_line(digest="abc123") + _metric_line(err="2e-05", max_abs="0.0012", threshold=1e-3)
        runner, _ = self._runner_pair(native_stderr, candidate_stderr)

        row = ce.collect_seed_evidence(
            Path("/bin/x"), op_filter="m=1,n=1,k=1", target_tensor="dst",
            candidate_stable_name="mmq:fb1", seed=3, runner=runner,
        )
        self.assertAlmostEqual(row.threshold_t, 1e-3)

    def test_missing_digest_fails_closed(self):
        native_stderr = _metric_line(err="1e-05", max_abs="0.001")  # no digest line
        candidate_stderr = _digest_line(digest="abc123") + _metric_line(err="2e-05", max_abs="0.0012")
        runner, _ = self._runner_pair(native_stderr, candidate_stderr)

        with self.assertRaises(ce.EvidenceError) as ctx:
            ce.collect_seed_evidence(
                Path("/bin/x"), op_filter="m=1,n=1,k=1", target_tensor="dst",
                candidate_stable_name="mmq:fb1", seed=3, runner=runner,
            )
        self.assertIn("missing BIGCHERRY_REF_DIGEST", str(ctx.exception))

    def test_digest_tensor_can_differ_from_target_tensor(self):
        # HI80 (2026-08-23 real-hardware finding): a --test-file/
        # test_generic_op run's destination tensor is never pre-filled, so
        # it never gets a BIGCHERRY_REF_DIGEST line, even though its
        # BIGCHERRY_CORRECTNESS_METRIC fires correctly -- digest_tensor
        # lets the caller point the reproducibility check at a real leaf
        # tensor ("leaf_0") instead, independent of which tensor the
        # correctness metric is read from ("out").
        native_stderr = _digest_line(name="leaf_0", digest="abc123") + _metric_line(
            tensor="out", err="1e-05", max_abs="0.001"
        )
        candidate_stderr = _digest_line(name="leaf_0", digest="abc123") + _metric_line(
            tensor="out", err="2e-05", max_abs="0.0012"
        )
        runner, _ = self._runner_pair(native_stderr, candidate_stderr)

        row = ce.collect_seed_evidence(
            Path("/bin/x"), test_file=Path("/tmp/x.txt"), target_tensor="out",
            digest_tensor="leaf_0", candidate_stable_name="mmvf:native:v1", seed=3,
            runner=runner,
        )
        self.assertEqual(row.reference_digest, "abc123")
        self.assertAlmostEqual(row.e_c_nmse, 2e-05)

    def test_digest_tensor_defaults_to_target_tensor(self):
        # Preserves the original op_filter-path behavior exactly when
        # digest_tensor is not given.
        native_stderr = _digest_line(digest="abc123") + _metric_line(err="1e-05", max_abs="0.001")
        candidate_stderr = _digest_line(digest="abc123") + _metric_line(err="2e-05", max_abs="0.0012")
        runner, _ = self._runner_pair(native_stderr, candidate_stderr)

        row = ce.collect_seed_evidence(
            Path("/bin/x"), op_filter="m=1,n=1,k=1", target_tensor="dst",
            candidate_stable_name="mmq:fb1", seed=3, runner=runner,
        )
        self.assertEqual(row.reference_digest, "abc123")

    def test_nonzero_exit_is_recorded_not_raised(self):
        # A crash is real evidence (a failed seed), not a hard error at this
        # layer -- aggregate_seed_evidence is where "any failed seed" fails
        # closed, so a caller inspecting individual seed rows still sees it.
        native_stderr = _digest_line(digest="abc123") + _metric_line(err="1e-05", max_abs="0.001")
        candidate_stderr = _digest_line(digest="abc123")  # crashed before printing a metric
        runner, _ = self._runner_pair(native_stderr, candidate_stderr, candidate_rc=1)

        row = ce.collect_seed_evidence(
            Path("/bin/x"), op_filter="m=1,n=1,k=1", target_tensor="dst",
            candidate_stable_name="mmq:fb1", seed=3, runner=runner,
        )
        self.assertEqual(row.candidate_execution_status, "failed")


class AggregateSeedEvidenceTests(unittest.TestCase):
    def _row(self, seed, *, e_n=1e-05, e_c=2e-05, max_abs_n=0.001, max_abs_c=0.0012, native_status="ok", candidate_status="ok", threshold_t=5e-4):
        return ce.SeedEvidence(
            seed=seed, reference_digest=f"digest{seed}", e_n_nmse=e_n, e_c_nmse=e_c,
            max_abs_native=max_abs_n, max_abs_candidate=max_abs_c,
            native_execution_status=native_status, candidate_execution_status=candidate_status,
            threshold_t=threshold_t,
        )

    def test_worst_of_seeds_not_average(self):
        rows = [
            self._row(1, e_n=1e-05, e_c=2e-05),
            self._row(2, e_n=9e-05, e_c=1e-04),  # the worst
            self._row(3, e_n=2e-05, e_c=3e-05),
        ]
        agg = ce.aggregate_seed_evidence(rows)
        self.assertAlmostEqual(agg.e_n_nmse, 9e-05)
        self.assertAlmostEqual(agg.e_c_nmse, 1e-04)

    def test_threshold_derived_from_seed_rows_not_a_parameter(self):
        rows = [self._row(i, threshold_t=1e-3) for i in (1, 2, 3)]
        agg = ce.aggregate_seed_evidence(rows)
        self.assertAlmostEqual(agg.threshold_t, 1e-3)

    def test_seeds_disagreeing_on_threshold_fails_closed(self):
        # HI67 threshold-authority fix: T comes from test-backend-ops' own
        # emitted value per seed, never a caller-supplied float -- seeds that
        # disagree are not comparable evidence for the same operation.
        rows = [
            self._row(1, threshold_t=5e-4),
            self._row(2, threshold_t=5e-4),
            self._row(3, threshold_t=1e-3),
        ]
        with self.assertRaises(ce.EvidenceError) as ctx:
            ce.aggregate_seed_evidence(rows)
        self.assertIn("disagree", str(ctx.exception))

    def test_fewer_than_three_seeds_rejected(self):
        rows = [self._row(1), self._row(2)]
        with self.assertRaises(ce.EvidenceError):
            ce.aggregate_seed_evidence(rows)

    def test_any_failed_seed_fails_closed(self):
        rows = [self._row(1), self._row(2, candidate_status="failed"), self._row(3)]
        with self.assertRaises(ce.EvidenceError):
            ce.aggregate_seed_evidence(rows)

    def test_dispatchable_true_when_within_headroom(self):
        rows = [self._row(i, e_n=1e-05, e_c=1.2e-05, max_abs_n=0.001, max_abs_c=0.001) for i in (1, 2, 3)]
        agg = ce.aggregate_seed_evidence(rows)
        self.assertTrue(agg.native_passes)
        self.assertTrue(agg.candidate_passes_headroom)
        self.assertTrue(agg.candidate_max_abs_ok)
        self.assertTrue(agg.dispatchable)

    def test_dispatchable_false_when_candidate_exceeds_headroom(self):
        # native uses almost none of its budget; candidate blows way past
        # the 50%-remaining-headroom rule even though it's still < T.
        rows = [self._row(i, e_n=1e-06, e_c=4.9e-04) for i in (1, 2, 3)]
        agg = ce.aggregate_seed_evidence(rows)
        self.assertTrue(agg.native_passes)
        self.assertFalse(agg.candidate_passes_headroom)
        self.assertFalse(agg.dispatchable)

    def test_dispatchable_false_when_native_itself_fails(self):
        rows = [self._row(i, e_n=9e-04, e_c=1e-06) for i in (1, 2, 3)]
        agg = ce.aggregate_seed_evidence(rows)
        self.assertFalse(agg.native_passes)
        self.assertFalse(agg.dispatchable)

    def test_dispatchable_false_when_candidate_max_abs_worse_than_native(self):
        rows = [self._row(i, e_n=1e-05, e_c=1.1e-05, max_abs_n=0.001, max_abs_c=0.002) for i in (1, 2, 3)]
        agg = ce.aggregate_seed_evidence(rows)
        self.assertFalse(agg.candidate_max_abs_ok)
        self.assertFalse(agg.dispatchable)


class WriteCorrectnessEvidenceTests(unittest.TestCase):
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
            "(x'00', 'gfx1100', 1, 32, 96, 0, '{}')"
        )
        self.hardware_id = self.conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        self.conn.execute(
            "INSERT INTO signature (signature_digest, base_digest, schema_version, op, "
            "src0_type, src1_type, dst_type, m, n, k, canonical_json) VALUES "
            "(x'01', x'02', 1, 'MUL_MAT', 'q8_0', 'f32', 'f32', 1, 1, 1, '{}')"
        )
        self.signature_id = self.conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        self.conn.execute(
            "INSERT INTO candidate (build_id, stable_name, family, source_class, "
            "implementation_version, architectures, architecture_mask, graph_safe, "
            "deterministic, config_json) VALUES (?, 'native_wrapper', 'mmq', "
            "'native_wrapper', 1, '[]', 0, 1, 1, '{}')",
            (self.build_id,),
        )
        self.native_candidate_id = self.conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        self.conn.execute(
            "INSERT INTO candidate (build_id, stable_name, family, source_class, "
            "implementation_version, architectures, architecture_mask, graph_safe, "
            "deterministic, config_json) VALUES (?, 'mmq:fb1', 'mmq', "
            "'existing_alternative', 1, '[]', 0, 1, 1, '{}')",
            (self.build_id,),
        )
        self.candidate_id = self.conn.execute("SELECT last_insert_rowid()").fetchone()[0]

    def tearDown(self):
        self.conn.close()

    def test_write_inserts_parent_and_seed_rows(self):
        rows = [
            ce.SeedEvidence(seed=i, reference_digest=f"d{i}", e_n_nmse=1e-05, e_c_nmse=2e-05,
                             max_abs_native=0.001, max_abs_candidate=0.0012,
                             native_execution_status="ok", candidate_execution_status="ok", threshold_t=5e-4)
            for i in (1, 2, 3)
        ]
        agg = ce.aggregate_seed_evidence(rows)
        evidence_id = ce.write_correctness_evidence(
            self.conn, build_id=self.build_id, hardware_id=self.hardware_id,
            signature_id=self.signature_id, candidate_id=self.candidate_id,
            native_candidate_id=self.native_candidate_id, aggregate=agg, tool_version="v1",
        )
        parent = self.conn.execute(
            "SELECT seed_count, e_n_nmse, e_c_nmse FROM correctness_evidence WHERE correctness_evidence_id = ?",
            (evidence_id,),
        ).fetchone()
        self.assertEqual(parent[0], 3)
        seed_rows = self.conn.execute(
            "SELECT seed FROM correctness_evidence_seed WHERE correctness_evidence_id = ? ORDER BY seed",
            (evidence_id,),
        ).fetchall()
        self.assertEqual([r[0] for r in seed_rows], [1, 2, 3])

    def test_duplicate_identity_and_contract_is_rejected(self):
        rows = [
            ce.SeedEvidence(seed=i, reference_digest=f"d{i}", e_n_nmse=1e-05, e_c_nmse=2e-05,
                             max_abs_native=0.001, max_abs_candidate=0.0012,
                             native_execution_status="ok", candidate_execution_status="ok", threshold_t=5e-4)
            for i in (1, 2, 3)
        ]
        agg = ce.aggregate_seed_evidence(rows)
        ce.write_correctness_evidence(
            self.conn, build_id=self.build_id, hardware_id=self.hardware_id,
            signature_id=self.signature_id, candidate_id=self.candidate_id,
            native_candidate_id=self.native_candidate_id, aggregate=agg, tool_version="v1",
        )
        with self.assertRaises(sqlite3.IntegrityError):
            ce.write_correctness_evidence(
                self.conn, build_id=self.build_id, hardware_id=self.hardware_id,
                signature_id=self.signature_id, candidate_id=self.candidate_id,
                native_candidate_id=self.native_candidate_id, aggregate=agg, tool_version="v1",
            )


if __name__ == "__main__":
    unittest.main()
