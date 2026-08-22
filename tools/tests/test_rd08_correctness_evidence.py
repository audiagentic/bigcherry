"""HI83: rd08_correctness_evidence.py's control-transform and shape/seed
comparison logic, exercised against a faked subprocess.run and a real temp
directory for the checked-replace control edits -- no real HIP hardware, a
compiled test-backend-ops binary, or a git worktree needed for this layer's
own correctness (materialize_rd08_variants' worktree mechanics are
patch_source_isolation.materialize_source_variant()'s responsibility, and
were verified this session against a real isolated worktree)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bigcherry import rd08_correctness_evidence as rd08  # noqa: E402


def _completed(returncode: int, stderr: str):
    result = MagicMock()
    result.returncode = returncode
    result.stderr = stderr
    result.stdout = ""
    return result


def _digest_line(*, name="dst", digest="abc123", call_index=0, nels=1024):
    return f"BIGCHERRY_REF_DIGEST name={name} call_index={call_index} digest={digest} nels={nels}\n"


def _metric_line(*, tensor="dst", err="1e-05", max_abs="0.001", backend1_digest, backend2_digest):
    return (
        f"BIGCHERRY_CORRECTNESS_METRIC op=MUL_MAT tensor={tensor} "
        f"backend1=HIP0 backend2=CPU err={err} max_abs={max_abs} "
        f"threshold=5e-4 n=1024 "
        f"backend1_digest={backend1_digest} backend2_digest={backend2_digest}\n"
    )


class ApplyVdr1ControlTests(unittest.TestCase):
    def _make_tree(self, tmp: Path) -> None:
        vecdotq = tmp / "ggml" / "src" / "ggml-cuda" / "vecdotq.cuh"
        mmvq = tmp / "ggml" / "src" / "ggml-cuda" / "mmvq.cu"
        vecdotq.parent.mkdir(parents=True, exist_ok=True)
        vecdotq.write_text(
            "#define VDR_Q6_K_Q8_1_MMVQ 2\n#define VDR_Q6_K_Q8_1_MMQ  8\n",
            encoding="utf-8",
        )
        mmvq.write_text(
            "        case GGML_TYPE_Q6_K:    return vec_dot_q6_K_q8_1_vdr2;\n",
            encoding="utf-8",
        )

    def test_reverts_exactly_the_two_semantic_lines(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            self._make_tree(tmp)
            rd08.apply_vdr1_control(tmp)
            vecdotq_text = (tmp / "ggml" / "src" / "ggml-cuda" / "vecdotq.cuh").read_text(encoding="utf-8")
            mmvq_text = (tmp / "ggml" / "src" / "ggml-cuda" / "mmvq.cu").read_text(encoding="utf-8")
            self.assertIn("#define VDR_Q6_K_Q8_1_MMVQ 1", vecdotq_text)
            self.assertNotIn("#define VDR_Q6_K_Q8_1_MMVQ 2", vecdotq_text)
            self.assertIn("        case GGML_TYPE_Q6_K:    return vec_dot_q6_K_q8_1;\n", mmvq_text)
            self.assertNotIn("vec_dot_q6_K_q8_1_vdr2", mmvq_text)

    def test_fails_closed_when_anchor_missing(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            self._make_tree(tmp)
            (tmp / "ggml" / "src" / "ggml-cuda" / "vecdotq.cuh").write_text(
                "#define VDR_Q6_K_Q8_1_MMVQ 3\n", encoding="utf-8",
            )
            with self.assertRaises(rd08.Rd08CorrectnessError):
                rd08.apply_vdr1_control(tmp)

    def test_fails_closed_when_anchor_duplicated(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            self._make_tree(tmp)
            (tmp / "ggml" / "src" / "ggml-cuda" / "vecdotq.cuh").write_text(
                "#define VDR_Q6_K_Q8_1_MMVQ 2\n#define VDR_Q6_K_Q8_1_MMVQ 2\n", encoding="utf-8",
            )
            with self.assertRaises(rd08.Rd08CorrectnessError):
                rd08.apply_vdr1_control(tmp)


class ControlVariantDigestTests(unittest.TestCase):
    def test_stable_and_deterministic(self):
        self.assertEqual(rd08._control_variant_digest(), rd08._control_variant_digest())

    def test_matches_hand_computed_value(self):
        import hashlib
        import json
        payload = [
            {"path": str(p), "old": old, "new": new}
            for p, old, new in rd08._CONTROL_EDITS
        ]
        expected = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        self.assertEqual(rd08._control_variant_digest(), expected)


class CompareOneShapeSeedTests(unittest.TestCase):
    def _runner_pair(self, subject_stderr: str, control_stderr: str, *, subject_rc=0, control_rc=0):
        calls = {"n": 0}

        def runner(argv, **kwargs):
            calls["n"] += 1
            binary = argv[0]
            if binary == "subject-binary":
                return _completed(subject_rc, subject_stderr)
            return _completed(control_rc, control_stderr)

        return runner

    def test_matching_digests_are_ok(self):
        subject_stderr = _digest_line(digest="abc") + _metric_line(
            backend1_digest="deadbeef", backend2_digest="cafef00d"
        )
        control_stderr = _digest_line(digest="abc") + _metric_line(
            backend1_digest="deadbeef", backend2_digest="00000000"
        )
        runner = self._runner_pair(subject_stderr, control_stderr)
        row = rd08.compare_one_shape_seed(
            subject_binary=Path("subject-binary"), control_binary=Path("control-binary"),
            shape=rd08.RD08_SHAPES[0], seed=1, runner=runner,
        )
        self.assertTrue(row.ok)

    def test_mismatched_output_digest_is_not_ok(self):
        subject_stderr = _digest_line(digest="abc") + _metric_line(
            backend1_digest="deadbeef", backend2_digest="cafef00d"
        )
        control_stderr = _digest_line(digest="abc") + _metric_line(
            backend1_digest="00000000", backend2_digest="cafef00d"
        )
        runner = self._runner_pair(subject_stderr, control_stderr)
        row = rd08.compare_one_shape_seed(
            subject_binary=Path("subject-binary"), control_binary=Path("control-binary"),
            shape=rd08.RD08_SHAPES[0], seed=1, runner=runner,
        )
        self.assertFalse(row.ok)

    def test_mismatched_input_digest_is_not_ok(self):
        subject_stderr = _digest_line(digest="abc") + _metric_line(
            backend1_digest="deadbeef", backend2_digest="cafef00d"
        )
        control_stderr = _digest_line(digest="xyz") + _metric_line(
            backend1_digest="deadbeef", backend2_digest="cafef00d"
        )
        runner = self._runner_pair(subject_stderr, control_stderr)
        row = rd08.compare_one_shape_seed(
            subject_binary=Path("subject-binary"), control_binary=Path("control-binary"),
            shape=rd08.RD08_SHAPES[0], seed=1, runner=runner,
        )
        self.assertFalse(row.ok)

    def test_missing_digest_field_is_not_ok(self):
        # Simulates a binary predating the 1223 digest extension: no
        # backend1_digest/backend2_digest in the metric line at all.
        subject_stderr = (
            _digest_line(digest="abc")
            + "BIGCHERRY_CORRECTNESS_METRIC op=MUL_MAT tensor=dst backend1=HIP0 "
              "backend2=CPU err=1e-05 max_abs=0.001 threshold=5e-4 n=1024\n"
        )
        control_stderr = _digest_line(digest="abc") + _metric_line(
            backend1_digest="deadbeef", backend2_digest="cafef00d"
        )
        runner = self._runner_pair(subject_stderr, control_stderr)
        row = rd08.compare_one_shape_seed(
            subject_binary=Path("subject-binary"), control_binary=Path("control-binary"),
            shape=rd08.RD08_SHAPES[0], seed=1, runner=runner,
        )
        self.assertFalse(row.ok)

    def test_nonzero_exit_is_not_ok(self):
        subject_stderr = _digest_line(digest="abc") + _metric_line(
            backend1_digest="deadbeef", backend2_digest="cafef00d"
        )
        control_stderr = _digest_line(digest="abc") + _metric_line(
            backend1_digest="deadbeef", backend2_digest="cafef00d"
        )
        runner = self._runner_pair(subject_stderr, control_stderr, subject_rc=1)
        row = rd08.compare_one_shape_seed(
            subject_binary=Path("subject-binary"), control_binary=Path("control-binary"),
            shape=rd08.RD08_SHAPES[0], seed=1, runner=runner,
        )
        self.assertFalse(row.ok)


class RequireRd08CorrectnessEvidenceTests(unittest.TestCase):
    def test_all_passing_returns_all_rows(self):
        good_stderr_subject = _digest_line(digest="abc") + _metric_line(
            backend1_digest="deadbeef", backend2_digest="cafef00d"
        )
        good_stderr_control = _digest_line(digest="abc") + _metric_line(
            backend1_digest="deadbeef", backend2_digest="00000000"
        )

        def runner(argv, **kwargs):
            binary = argv[0]
            if binary == "subject-binary":
                return _completed(0, good_stderr_subject)
            return _completed(0, good_stderr_control)

        rows = rd08.require_rd08_correctness_evidence(
            subject_binary=Path("subject-binary"), control_binary=Path("control-binary"),
            shapes=(rd08.RD08_SHAPES[0],), seeds=(1, 2), runner=runner,
        )
        self.assertEqual(len(rows), 2)
        self.assertTrue(all(row.ok for row in rows))

    def test_raises_with_specific_reason_on_first_failure(self):
        subject_stderr = _digest_line(digest="abc") + _metric_line(
            backend1_digest="deadbeef", backend2_digest="cafef00d"
        )
        control_stderr = _digest_line(digest="abc") + _metric_line(
            backend1_digest="00000000", backend2_digest="cafef00d"
        )

        def runner(argv, **kwargs):
            binary = argv[0]
            if binary == "subject-binary":
                return _completed(0, subject_stderr)
            return _completed(0, control_stderr)

        with self.assertRaises(rd08.Rd08CorrectnessError) as ctx:
            rd08.require_rd08_correctness_evidence(
                subject_binary=Path("subject-binary"), control_binary=Path("control-binary"),
                shapes=(rd08.RD08_SHAPES[0],), seeds=(1,), runner=runner,
            )
        message = str(ctx.exception)
        self.assertIn("ffn", message)
        self.assertIn("seed=1", message)


if __name__ == "__main__":
    unittest.main()
