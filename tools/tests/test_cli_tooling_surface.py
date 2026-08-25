"""TR09 characterization of the public CLI surface before decomposition."""

from __future__ import annotations

import unittest
from typing import Any, cast

from bigcherry import __main__ as cli


class CliSurfaceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.parser = cli.build_parser()
        subparsers = cast(Any, self.parser._subparsers)
        self.commands = cast(dict[str, Any], subparsers._group_actions[0].choices)

    def test_product_command_set_is_present(self) -> None:
        expected = {
            "pull",
            "audit",
            "check",
            "apply",
            "patches",
            "patch-status",
            "patch-explain",
            "patch-graph",
            "patch-verify-evidence",
            "patch-lint",
            "patch-validate",
            "sources",
            "repin",
            "pin-status",
            "replay-inspect",
            "build",
            "generate",
            "status",
            "experiment-contract",
            "doctor",
            "tune-journal",
            "tune-promote",
            "tune-null-fdr",
            "experiment",
            "campaign-build",
            "compare-tunes",
            "ab-benchmark",
            "probe-release",
            "validate-ref",
            "rank-replay",
            "resource-report",
            "candidate-binary-size",
            "report",
            "impact",
            "power",
            "kernel-fraction",
            "inventory",
        }
        self.assertTrue(expected <= set(self.commands))

    def test_repin_contract(self) -> None:
        args = self.parser.parse_args(["repin", "--ref", "b2"])
        self.assertEqual(args.ref, "b2")
        self.assertTrue(callable(args.func))

    def test_pin_status_contract(self) -> None:
        args = self.parser.parse_args(["pin-status", "--strict", "--json"])
        self.assertTrue(args.strict)
        self.assertFalse(args.complete)
        self.assertTrue(args.json)
        self.assertTrue(callable(args.func))


if __name__ == "__main__":
    unittest.main()
