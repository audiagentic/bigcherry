"""MEN10: the global --engine is settled once at the CLI boundary from cli/engine_scope.py's table."""

import argparse
import contextlib
import io
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
        self.assertEqual(_resolve(["patch-rebase-check", "--all"]).engine, "llamacpp")
        self.assertEqual(_resolve(["--engine", "llamacpp", "patch-rebase-check", "--all"]).engine, "llamacpp")

    def test_llamacpp_tool_refuses_another_engine(self):
        with self.assertRaises(SystemExit):
            _resolve(["--engine", "radiance", "patch-rebase-check", "--all"])

    def test_undeclared_engine_is_refused(self):
        with self.assertRaises(SystemExit):
            _resolve(["--engine", "vllm", "patch-status"])

    def test_patch_commands_need_the_engine(self):
        for command in sorted(engine_scope._ENGINE_COMMANDS):
            with self.subTest(command=command), contextlib.redirect_stderr(io.StringIO()):
                argv = [command] if command in ("patches", "patch-status", "patch-graph") else None
                if argv is None:
                    argv = {
                        "patch-explain": ["patch-explain", "x"],
                        "patch-verify-evidence": ["patch-verify-evidence"],
                        "patch-gates": ["patch-gates", "x", "--intent", "validate"],
                        "patch-doc": ["patch-doc", "--all"],
                    }[command]
                with self.assertRaises(SystemExit):
                    _resolve(argv)
                self.assertEqual(_resolve(["--engine", "radiance", *argv]).engine, "radiance")

    def _run(self, argv):
        parser = build_parser()
        args = parser.parse_args(argv)
        engine_scope.resolve(args, parser)
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = args.func(args)
        return code, out.getvalue(), err.getvalue()

    def test_patches_lists_the_chosen_engines_patch_tree(self):
        code, out, _ = self._run(["--engine", "radiance", "patches"])
        self.assertEqual(code, 0)
        self.assertIn("rad0001_drain_mover_before_arena_free", out)
        self.assertNotIn("0100_cmake_options", out)

    def test_patches_on_llamacpp_lists_its_own_tree(self):
        code, out, _ = self._run(["--engine", "llamacpp", "patches"])
        self.assertEqual(code, 0)
        self.assertIn("0100_cmake_options", out)
        self.assertNotIn("rad0001_drain_mover_before_arena_free", out)

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
