"""HI82 item 8 follow-up (GPT review, req_6b1466ee8369406c): e2e_smoke_report.py
had zero test coverage, and its bench.json v2 validator originally accepted a
schema_version==2 document that omitted the whole build_role/build_identity
provenance addition -- this closes that hole and covers it."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bigcherry.e2e_smoke_report import _validate_bench  # noqa: E402


def _build_identity() -> dict:
    return {
        "effective_build_id": "eff", "compile_verification_id": "cv",
        "compile_commands_digest": "ccd", "hip_compile_commands_digest": "hccd",
        "runtime_bundle_hash": "rbh", "runtime_artifacts": {"llama-server": "aaa"},
    }


def _metric() -> dict:
    return {"avg_ts": 100.0, "stddev_ts": 1.0}


def _config(build_role: str) -> dict:
    return {
        "build_role": build_role, "build_identity": _build_identity(),
        "metrics": {"pp": _metric(), "tg": _metric()},
    }


def _valid_bench() -> dict:
    return {
        "schema_version": 2,
        "params": {"n_prompt": 512, "n_gen": 128, "repetitions": 5},
        "configs": {
            "stock": _config("stock"), "native": _config("tune"), "replay": _config("replay"),
        },
    }


class ValidateBenchTests(unittest.TestCase):
    def test_accepts_well_formed_v2_document(self):
        bench = _validate_bench(_valid_bench(), source=Path("bench.json"))
        self.assertEqual(bench["schema_version"], 2)

    def test_rejects_missing_build_role(self):
        payload = _valid_bench()
        del payload["configs"]["native"]["build_role"]
        with self.assertRaisesRegex(RuntimeError, "build_role must be 'tune'"):
            _validate_bench(payload, source=Path("bench.json"))

    def test_rejects_wrong_build_role(self):
        payload = _valid_bench()
        payload["configs"]["native"]["build_role"] = "native"  # not "tune"
        with self.assertRaisesRegex(RuntimeError, "build_role must be 'tune'"):
            _validate_bench(payload, source=Path("bench.json"))

    def test_rejects_missing_build_identity(self):
        payload = _valid_bench()
        del payload["configs"]["stock"]["build_identity"]
        with self.assertRaisesRegex(RuntimeError, "build_identity"):
            _validate_bench(payload, source=Path("bench.json"))

    def test_rejects_incomplete_build_identity(self):
        # A document that claims schema_version 2 but omits the actual
        # provenance fields -- the exact hole GPT's review found.
        payload = _valid_bench()
        payload["configs"]["replay"]["build_identity"] = {"effective_build_id": "eff"}
        with self.assertRaisesRegex(RuntimeError, "missing required field"):
            _validate_bench(payload, source=Path("bench.json"))


if __name__ == "__main__":
    unittest.main()
