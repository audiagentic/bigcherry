"""MEN10: the global --engine is settled once at the CLI boundary from cli/engine_scope.py's table."""

import argparse
import unittest

from bigcherry.cli import engine_scope
from bigcherry.cli.main import build_parser


def _resolve(argv):
    parser = build_parser()
    args = parser.parse_args(argv)
    engine_scope.resolve(args, parser)
    return args


class EngineScopeTests(unittest.TestCase):
    def test_every_command_has_a_scope(self):
        parser = build_parser()
        sub = next(a for a in parser._actions if isinstance(a, argparse._SubParsersAction))
        for command in sub.choices:
            self.assertIn(
                engine_scope.scope(command),
                {engine_scope.ENGINE, engine_scope.LANE, engine_scope.PLATFORM, engine_scope.SELF, "llamacpp"},
                command,
            )

    def test_llamacpp_tool_settles_its_engine_without_the_flag(self):
        self.assertEqual(_resolve(["patch-status"]).engine, "llamacpp")
        self.assertEqual(_resolve(["--engine", "llamacpp", "patch-status"]).engine, "llamacpp")

    def test_llamacpp_tool_refuses_another_engine(self):
        with self.assertRaises(SystemExit):
            _resolve(["--engine", "radiance", "patch-status"])

    def test_undeclared_engine_is_refused(self):
        with self.assertRaises(SystemExit):
            _resolve(["--engine", "vllm", "patch-status"])

    def test_platform_command_refuses_the_flag(self):
        self.assertIsNone(_resolve(["patch-lint"]).engine)
        with self.assertRaises(SystemExit):
            _resolve(["--engine", "radiance", "patch-lint"])

    def test_engine_bench_keeps_its_own_engine_argument(self):
        args = _resolve(["engine-bench", "--engine", "radiance", "--binary", "b", "--model", "m",
                         "--out", "o", "--depth", "1"])
        self.assertEqual(args.engine, "radiance")


if __name__ == "__main__":
    unittest.main()
