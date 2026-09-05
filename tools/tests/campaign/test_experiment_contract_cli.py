"""Experiment Contract CLI tests (EC11/EC12)."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bigcherry.__main__ import build_parser  # noqa: E402

_TOML = """
[contract.SMOKE-TEST-001]
title = "smoke-test contract"

[contract.SMOKE-TEST-001.source]
source_id = "stew675-rdna-boosts"
commits = ["deadbeef"]
atomic_part = "smoke-test"

[contract.SMOKE-TEST-001.hypothesis]
family = "mmq"
expected_effect = "performance"
rationale = "CLI test"

[contract.SMOKE-TEST-001.scope]
backend = "hip"
architectures = ["gfx1100"]

[contract.SMOKE-TEST-001.positive]
models = ["m"]
workloads = ["decode"]

[contract.SMOKE-TEST-001.controls]
models = ["m"]
workloads = ["prefill"]

[contract.SMOKE-TEST-001.acceptance]
max_control_regression_pct = 1
"""


def _write(text: str, suffix: str) -> Path:
    handle = tempfile.NamedTemporaryFile("w", suffix=suffix, delete=False, encoding="utf-8")
    with handle:
        handle.write(text)
    return Path(handle.name)


class ExperimentContractCliTests(unittest.TestCase):
    def setUp(self):
        self.contracts_path = _write(_TOML, ".toml")
        self.parser = build_parser()

    def _run(self, argv: list[str]) -> int:
        args = self.parser.parse_args(argv)
        return args.func(args)

    def test_list_finds_the_registered_contract(self, ):
        rc = self._run(["experiment-contract", "list", "--contracts", str(self.contracts_path)])
        self.assertEqual(rc, 0)

    def test_validate_a_named_contract(self):
        rc = self._run([
            "experiment-contract", "validate", "SMOKE-TEST-001",
            "--contracts", str(self.contracts_path),
        ])
        self.assertEqual(rc, 0)

    def test_validate_unknown_contract_fails(self):
        rc = self._run([
            "experiment-contract", "validate", "NO-SUCH-CONTRACT",
            "--contracts", str(self.contracts_path),
        ])
        self.assertEqual(rc, 1)

    def test_plan_expands_lanes(self):
        rc = self._run([
            "experiment-contract", "plan", "SMOKE-TEST-001",
            "--contracts", str(self.contracts_path),
            "--source", "bigcherry-native", "--build", "control", "--platform", "linux-multi",
        ])
        self.assertEqual(rc, 0)

    def test_plan_unknown_source_fails_cleanly(self):
        rc = self._run([
            "experiment-contract", "plan", "SMOKE-TEST-001",
            "--contracts", str(self.contracts_path),
            "--source", "no-such-source", "--build", "control", "--platform", "linux-multi",
        ])
        self.assertEqual(rc, 1)

    def test_report_renders_and_exits_zero_on_promotion(self):
        evidence = _write(json.dumps({
            "correctness_gate": {"passed": True, "missing_checks": [], "failed_checks": []},
            "aggregated_effects": {"target_kernel_gain_pct": 6.0, "max_control_regression_pct": 0.1},
        }), ".json")
        rc = self._run([
            "experiment-contract", "report", "SMOKE-TEST-001",
            "--contracts", str(self.contracts_path), "--evidence-file", str(evidence),
        ])
        self.assertEqual(rc, 0)

    def test_report_exits_nonzero_when_promotion_gate_fails(self):
        evidence = _write(json.dumps({
            "correctness_gate": {"passed": False, "missing_checks": ["greedy_parity"], "failed_checks": []},
            "aggregated_effects": {"max_control_regression_pct": 0.1},
        }), ".json")
        rc = self._run([
            "experiment-contract", "report", "SMOKE-TEST-001",
            "--contracts", str(self.contracts_path), "--evidence-file", str(evidence),
        ])
        self.assertEqual(rc, 1)

    def test_experiment_and_experiment_contract_are_distinct_subcommands(self):
        # HI47's `experiment` bundle-runner CLI must not collide with EC11's
        # `experiment-contract` -- this was a real naming conflict caught
        # while building EC11 (argparse raised ArgumentError on registration).
        subparsers_actions = [
            action for action in self.parser._subparsers._group_actions
            if hasattr(action, "choices")
        ]
        names = set()
        for action in subparsers_actions:
            names.update(action.choices.keys())
        self.assertIn("experiment", names)
        self.assertIn("experiment-contract", names)


if __name__ == "__main__":
    unittest.main()
