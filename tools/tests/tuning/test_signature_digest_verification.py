"""HI121 close-out step 4 (RV84 P0-4): tests for signature_digest_verification.py.

No real HIP hardware here -- these exercise the routing/memoization/fail-
closed logic with a fake test-backend-ops runner, exactly like
hi80_generate_correctness_evidence.py's own tests do. Real-hardware
validation (a real compiled test-backend-ops reproducing the exact stored
signature hex for MUL_MAT/MUL_MAT_ID/simple routed GLU) is tracked
separately as this item's Brutus acceptance gate, not reproducible offline.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bigcherry.tuning import correctness_evidence as ce  # noqa: E402
from bigcherry.tuning import dispatch_abi  # noqa: E402
from bigcherry.tuning import signature_capabilities as sc  # noqa: E402
from bigcherry.tuning import signature_digest_verification as sdv  # noqa: E402
from bigcherry.tuning import signature_mapping as scm  # noqa: E402

EPOCH = dispatch_abi.SIGNATURE_IDENTITY_EPOCH

MUL_MAT_CANONICAL = {
    "schema_version": EPOCH, "op": 2, "flags": 0, "fusion": 0, "glu_op": 0,
    "src0_type": 0, "src1_type": 0, "dst_type": 0,
    "ne0": [256, 16, 1, 1], "ne1": [256, 1, 1, 1], "ned": [16, 1, 1, 1],
}

GLU_CANONICAL = {
    "op": 4, "src0_type": 8, "src1_type": 0, "dst_type": 0,
    "fusion": 2, "glu_op": 2, "flags": 31,
    "n_expert": 256, "n_expert_used": 8,
    "ne0": [2048, 256, 256, 1], "ne1": [2048, 1, 1, 1], "ned": [256, 8, 1, 1],
    "schema_version": EPOCH,
}

UNSUPPORTED_OP_CANONICAL = {
    "schema_version": EPOCH, "op": 1, "flags": 0, "fusion": 0, "glu_op": 0,
}


def _write_fixture_vendor(tmp_path: Path) -> Path:
    vendor = tmp_path / "vendor" / "llama.cpp"
    (vendor / "ggml" / "include").mkdir(parents=True, exist_ok=True)
    (vendor / "ggml" / "src").mkdir(parents=True, exist_ok=True)
    (vendor / "ggml" / "include" / "ggml.h").write_text(
        "enum ggml_type {\n    GGML_TYPE_F32  = 0,\n    GGML_TYPE_Q8_0 = 8,\n};\n"
        "enum ggml_op {\n    GGML_OP_NONE,\n    GGML_OP_ADD,\n    GGML_OP_MUL_MAT,\n"
        "    GGML_OP_MUL_MAT_ID,\n    GGML_OP_GLU,\n    GGML_OP_COUNT,\n};\n",
        encoding="utf-8",
    )
    (vendor / "ggml" / "src" / "ggml.c").write_text(
        "static const struct ggml_type_traits type_traits[GGML_TYPE_COUNT] = {\n"
        '    [GGML_TYPE_F32] = {\n        .type_name = "f32",\n    },\n'
        '    [GGML_TYPE_Q8_0] = {\n        .type_name = "q8_0",\n    },\n'
        "};\n"
        'static const char * GGML_OP_NAME[GGML_OP_COUNT] = {\n'
        '    "NONE",\n    "ADD",\n    "MUL_MAT",\n    "MUL_MAT_ID",\n    "GLU",\n'
        "};\n",
        encoding="utf-8",
    )
    return vendor


def _record_runner_factory(*, observed_hex="c" * 32, observed_canonical=None, calls: list | None = None):
    def runner(argv, capture_output, text, env):
        if calls is not None:
            calls.append(argv)
        db_path = Path(env["GGML_HIP_DISPATCH_DB"])
        db_path.write_text(
            json.dumps({
                "kind": "observation", "signature": observed_hex,
                "canonical": observed_canonical if observed_canonical is not None else {},
            }) + "\n",
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")
    return runner


class ObservedTestBackendOpsSignatureHexTests(unittest.TestCase):
    def test_single_observation_returns_lowercased_hex(self):
        with tempfile.TemporaryDirectory() as tmp:
            test_file = Path(tmp) / "case.txt"
            test_file.write_text("dummy\n", encoding="utf-8")
            result, canonical = sdv.observed_test_backend_ops_signature_hex(
                Path("test-backend-ops"), test_file=test_file,
                runner=_record_runner_factory(observed_hex="AB" * 16),
            )
            self.assertEqual(result, "ab" * 16)
            self.assertEqual(canonical, {})

    def test_zero_observations_fails_closed(self):
        def runner(argv, capture_output, text, env):
            Path(env["GGML_HIP_DISPATCH_DB"]).write_text("", encoding="utf-8")
            return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")
        with self.assertRaisesRegex(ce.EvidenceError, "0 distinct"):
            sdv.observed_test_backend_ops_signature_hex(
                Path("test-backend-ops"), test_file=Path("x"), runner=runner,
            )

    def test_two_distinct_observations_fails_closed(self):
        def runner(argv, capture_output, text, env):
            db_path = Path(env["GGML_HIP_DISPATCH_DB"])
            db_path.write_text(
                json.dumps({"kind": "observation", "signature": "a" * 32, "canonical": {}}) + "\n"
                + json.dumps({"kind": "observation", "signature": "b" * 32, "canonical": {}}) + "\n",
                encoding="utf-8",
            )
            return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")
        with self.assertRaisesRegex(ce.EvidenceError, "2 distinct"):
            sdv.observed_test_backend_ops_signature_hex(
                Path("test-backend-ops"), test_file=Path("x"), runner=runner,
            )

    def test_repeated_identical_observations_are_fine(self):
        def runner(argv, capture_output, text, env):
            db_path = Path(env["GGML_HIP_DISPATCH_DB"])
            row = json.dumps({"kind": "observation", "signature": "c" * 32, "canonical": {}}) + "\n"
            db_path.write_text(row + row, encoding="utf-8")
            return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")
        result, canonical = sdv.observed_test_backend_ops_signature_hex(
            Path("test-backend-ops"), test_file=Path("x"), runner=runner,
        )
        self.assertEqual(result, "c" * 32)
        self.assertEqual(canonical, {})

    def test_nonzero_returncode_fails_closed(self):
        def runner(argv, capture_output, text, env):
            return subprocess.CompletedProcess(argv, 1, stdout="", stderr="boom")
        with self.assertRaises(ce.EvidenceError):
            sdv.observed_test_backend_ops_signature_hex(
                Path("test-backend-ops"), test_file=Path("x"), runner=runner,
            )


class ObservedSignatureDigestHexTests(unittest.TestCase):
    def test_mul_mat_routes_through_test_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            vendor = _write_fixture_vendor(Path(tmp))
            calls: list = []
            result = sdv.observed_signature_digest_hex(
                MUL_MAT_CANONICAL, binary=Path("test-backend-ops"), vendor_root=vendor,
                runner=_record_runner_factory(
                    observed_hex="d" * 32, observed_canonical=MUL_MAT_CANONICAL, calls=calls,
                ),
            )
            self.assertEqual(result, "d" * 32)
            self.assertEqual(len(calls), 1)
            self.assertIn("--test-file", calls[0])

    def test_glu_routes_through_moe_glu_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            vendor = _write_fixture_vendor(Path(tmp))
            calls: list = []
            result = sdv.observed_signature_digest_hex(
                GLU_CANONICAL, binary=Path("test-backend-ops"), vendor_root=vendor,
                runner=_record_runner_factory(
                    observed_hex="e" * 32, observed_canonical=GLU_CANONICAL, calls=calls,
                ),
            )
            self.assertEqual(result, "e" * 32)
            self.assertEqual(len(calls), 1)
            self.assertIn("--moe-glu-file", calls[0])

    def test_unsupported_domain_never_invokes_runner(self):
        with tempfile.TemporaryDirectory() as tmp:
            vendor = _write_fixture_vendor(Path(tmp))
            calls: list = []

            def runner(argv, capture_output, text, env):
                calls.append(argv)
                raise AssertionError("runner must not be invoked for an unsupported domain")

            with self.assertRaises(sc.UnsupportedSignatureDomain):
                sdv.observed_signature_digest_hex(
                    UNSUPPORTED_OP_CANONICAL, binary=Path("test-backend-ops"),
                    vendor_root=vendor, runner=runner,
                )
            self.assertEqual(calls, [])

    def test_glu_gate_bias_fails_closed_not_silently_substituted(self):
        """RV84 P0-4 follow-up (dev-gpt-agent design review, 2026-08-27):
        GATE_BIAS GLU is inside HI121's audited capability domain but the
        existing moe-glu mapper only supports simple GATE -- must raise
        SignatureMappingError, never silently run the simple-GATE case as
        if it verified the biased one."""
        with tempfile.TemporaryDirectory() as tmp:
            vendor = _write_fixture_vendor(Path(tmp))
            gate_bias = dict(GLU_CANONICAL, fusion=3, flags=31 | (1 << 8))  # GATE_BIAS + GATE_BIAS content flag
            calls: list = []

            def runner(argv, capture_output, text, env):
                calls.append(argv)
                raise AssertionError("runner must not run for an unrepresentable case")

            with self.assertRaises(scm.SignatureMappingError):
                sdv.observed_signature_digest_hex(
                    gate_bias, binary=Path("test-backend-ops"), vendor_root=vendor,
                    runner=runner,
                )
            self.assertEqual(calls, [])


def _echoing_runner_factory(*, calls: list | None = None):
    """A record-mode fake that echoes back whichever canonical the request
    was actually for (MUL_MAT_CANONICAL via --test-file, GLU_CANONICAL via
    --moe-glu-file) -- needed so _require_canonical_match() sees a
    consistent, correct pairing regardless of which routing branch a given
    call took, e.g. across two DIFFERENT canonicals sharing one verifier
    in a memoization test."""
    def runner(argv, capture_output, text, env):
        if calls is not None:
            calls.append(argv)
        db_path = Path(env["GGML_HIP_DISPATCH_DB"])
        if "--moe-glu-file" in argv:
            canonical, digest = GLU_CANONICAL, "e" * 32
        else:
            canonical, digest = MUL_MAT_CANONICAL, "d" * 32
        db_path.write_text(
            json.dumps({"kind": "observation", "signature": digest, "canonical": canonical}) + "\n",
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")
    return runner


class MakeSignatureDigestVerifierTests(unittest.TestCase):
    def test_memoizes_per_unique_canonical(self):
        with tempfile.TemporaryDirectory() as tmp:
            vendor = _write_fixture_vendor(Path(tmp))
            calls: list = []
            verifier = sdv.make_signature_digest_verifier(
                binary=Path("test-backend-ops"), vendor_root=vendor,
                runner=_echoing_runner_factory(calls=calls),
            )
            first = verifier(MUL_MAT_CANONICAL)
            second = verifier(dict(MUL_MAT_CANONICAL))  # structurally identical, new dict object
            self.assertEqual(first, "d" * 32)
            self.assertEqual(second, "d" * 32)
            self.assertEqual(len(calls), 1)

    def test_distinct_canonicals_each_invoke_the_runner(self):
        with tempfile.TemporaryDirectory() as tmp:
            vendor = _write_fixture_vendor(Path(tmp))
            calls: list = []
            verifier = sdv.make_signature_digest_verifier(
                binary=Path("test-backend-ops"), vendor_root=vendor,
                runner=_echoing_runner_factory(calls=calls),
            )
            verifier(MUL_MAT_CANONICAL)
            verifier(GLU_CANONICAL)
            self.assertEqual(len(calls), 2)

    def test_poisoned_canonical_with_correct_digest_is_rejected(self):
        """Adversarial-review follow-up (2026-08-27): a canonical tampered
        in a field the mapper ignores (nb0, here) but producing the same
        real digest must NOT be accepted -- digest equality alone is not
        proof of canonical authenticity."""
        with tempfile.TemporaryDirectory() as tmp:
            vendor = _write_fixture_vendor(Path(tmp))
            poisoned = dict(MUL_MAT_CANONICAL, nb0=[999, 999, 999, 999])
            verifier = sdv.make_signature_digest_verifier(
                binary=Path("test-backend-ops"), vendor_root=vendor,
                runner=_echoing_runner_factory(),
            )
            with self.assertRaisesRegex(ce.EvidenceError, "does not match the supplied"):
                verifier(poisoned)


if __name__ == "__main__":
    unittest.main()
