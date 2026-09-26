"""Contract [measurement] section: batching declared per series."""

from __future__ import annotations

import unittest
from pathlib import Path

from bigcherry.experiment import contract as ec
from bigcherry.patch.campaign import benchmark


def _doc(measurement: dict | None = None) -> dict:
    doc = {
        "title": "t",
        "source": {"source_id": "s", "commits": ["c"], "atomic_part": "a"},
        "hypothesis": {"expected_effect": "performance", "rationale": "r"},
        "target": {"kind": "orchestration"},
        "scope": {"backend": "hip", "architectures": ["gfx1100"]},
        "positive": {"models": ["m"], "workloads": ["decode"]},
        "controls": {"models": ["c"], "workloads": ["decode"]},
        "correctness": {"backend_reference": "required"},
        "acceptance": {"max_control_regression_pct": 1, "target_kernel_gain_pct": 0.0,
                       "effect_evidence_policy": "improvement_no_regression_v1", "min_sessions": 4,
                       "min_evidence_effect_pct": 0.0, "min_paired_rounds": 10},
    }
    if measurement is not None:
        doc["measurement"] = measurement
    return doc


class MeasurementSectionTests(unittest.TestCase):
    def test_absent_section_keeps_hash_and_defaults(self) -> None:
        plain = ec.parse_contract(_doc(), contract_id="X")
        self.assertIsNone(plain.measurement)
        declared = ec.parse_contract(_doc({"server_requests_per_start": 5}), contract_id="X")
        self.assertEqual(declared.measurement.server_requests_per_start, 5)
        self.assertEqual(declared.measurement.bench_invocation, "per-workload")
        self.assertNotEqual(plain.contract_hash, declared.contract_hash)

    def test_invalid_values_rejected(self) -> None:
        for bad in ({"bench_invocation": "both"}, {"server_requests_per_start": 0}, {"reps": 3}):
            with self.assertRaises(ec.ExperimentContractError):
                ec.parse_contract(_doc(bad), contract_id="X")


class CombinedCommandTests(unittest.TestCase):
    def test_one_command_covers_prefill_and_decode(self) -> None:
        command = benchmark._combined_llama_bench_command(
            Path("b/llama-bench"), Path("m.gguf"), ("prefill", "decode"), runtime_args=("-sm", "tensor"))
        self.assertEqual(command[command.index("-p") + 1], "512")
        self.assertEqual(command[command.index("-n") + 1], "128")
        self.assertEqual(command[-2:], ["-sm", "tensor"])


if __name__ == "__main__":
    unittest.main()
