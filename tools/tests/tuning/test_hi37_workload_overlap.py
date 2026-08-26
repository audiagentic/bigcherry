"""HI37 Part 2: tests for inventory.workload_digest()/workload_overlap() and
the `bigcherry inventory workload-check` CLI.

workload_digest() must mirror the C++ compute_workload_digest() byte for
byte (see test_hi37_workload_digest.py for the C++-side source-contract
half); workload_overlap() is the call-weighted coverage number that
actually matters for deciding whether a tuned cache was measured for a
workload resembling the one being checked (advisory only -- it never gates
a cache load).
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bigcherry import inventory as inv  # noqa: E402


def _python_executable() -> str:
    """Return a launchable interpreter even when sys.executable is an alias."""
    if Path(sys.executable).is_file():
        return sys.executable
    return shutil.which("python") or sys.executable

SIG_A = "a" * 32
SIG_B = "b" * 32
SIG_C = "c" * 32


class WorkloadDigestTests(unittest.TestCase):
    def test_matches_reference_blake2b_computation(self):
        blob = bytes.fromhex(SIG_A) + bytes.fromhex(SIG_B)
        expected = hashlib.blake2b(blob, digest_size=16, person=b"llama-workload").hexdigest()
        self.assertEqual(inv.workload_digest([SIG_B, SIG_A]), expected)

    def test_order_independent_presence_not_frequency(self):
        # Sorted set: order of the input iterable and duplicate entries
        # (repeated calls to the same signature) must not change the digest.
        self.assertEqual(
            inv.workload_digest([SIG_A, SIG_B]),
            inv.workload_digest([SIG_B, SIG_A, SIG_A, SIG_B]),
        )

    def test_adding_one_signature_changes_the_digest(self):
        self.assertNotEqual(
            inv.workload_digest([SIG_A, SIG_B]),
            inv.workload_digest([SIG_A, SIG_B, SIG_C]),
        )


class WorkloadOverlapTests(unittest.TestCase):
    def _record(self, observations):
        return inv.Record(header={"kind": "header"}, observations=observations)

    def test_full_overlap_reports_100_percent(self):
        record = self._record(
            [
                {"signature": SIG_A, "calls": 100},
                {"signature": SIG_B, "calls": 50},
            ]
        )
        overlap = inv.workload_overlap(record, {SIG_A, SIG_B})
        self.assertEqual(overlap["covered_share_pct"], 100.0)
        self.assertEqual(overlap["signatures_covered"], 2)

    def test_partial_overlap_is_call_weighted_not_signature_count(self):
        # SIG_A dominates calls (990 of 1000); only SIG_A is tuned. Coverage
        # by call share must be ~99%, not 50% (1 of 2 signatures).
        record = self._record(
            [
                {"signature": SIG_A, "calls": 990},
                {"signature": SIG_B, "calls": 10},
            ]
        )
        overlap = inv.workload_overlap(record, {SIG_A})
        self.assertAlmostEqual(overlap["covered_share_pct"], 99.0)
        self.assertEqual(overlap["signatures_covered"], 1)
        self.assertEqual(overlap["signatures_observed"], 2)

    def test_the_gemma_e4b_case_reports_low_coverage(self):
        # 21 tuned hits against 100 misses, unweighted -- but the weighted
        # figure could be far better or worse. Construct the case where the
        # 21 tuned signatures are the COLD ones (1 call each) against 100
        # untuned hot ones (1000 calls each): the honest coverage is tiny
        # even though "21 of 121" sounds almost a fifth.
        observations = [{"signature": f"{i:032x}", "calls": 1} for i in range(21)]
        observations += [
            {"signature": f"{i + 1000:032x}", "calls": 1000} for i in range(100)
        ]
        record = self._record(observations)
        tuned = {o["signature"] for o in observations[:21]}
        overlap = inv.workload_overlap(record, tuned)
        self.assertLess(overlap["covered_share_pct"], 1.0)
        self.assertEqual(overlap["signatures_covered"], 21)

    def test_observation_without_signature_is_excluded_from_denominator(self):
        record = self._record(
            [
                {"calls": 500},  # malformed: no signature
                {"signature": SIG_A, "calls": 10},
            ]
        )
        overlap = inv.workload_overlap(record, {SIG_A})
        self.assertEqual(overlap["signatures_observed"], 1)
        self.assertEqual(overlap["calls_observed"], 10)


class WorkloadCheckCliTests(unittest.TestCase):
    def test_cli_runs_and_reports_advisory_language(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            record_path = tmp_path / "record.jsonl"
            record_path.write_text(
                "\n".join(
                    [
                        json.dumps({"kind": "header"}),
                        json.dumps({"kind": "observation", "signature": SIG_A, "calls": 10}),
                        json.dumps({"kind": "observation", "signature": SIG_B, "calls": 5}),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            measurements_path = tmp_path / "tune.measurements.jsonl"
            measurements_path.write_text(
                "\n".join(
                    [
                        json.dumps({"kind": "header"}),
                        json.dumps({"kind": "result", "signature": SIG_A}),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            result = subprocess.run(
                [
                    _python_executable(), "-m", "bigcherry", "inventory", "workload-check",
                    str(record_path), "--measurements", str(measurements_path),
                ],
                cwd=str(Path(__file__).resolve().parents[2]),
                capture_output=True, text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("Advisory only", result.stdout)
            self.assertIn("coverage", result.stdout)


if __name__ == "__main__":
    unittest.main()
