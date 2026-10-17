"""MEN10: the global --engine is settled once at the CLI boundary from cli/engine_scope.py's table."""

import argparse
import contextlib
import io
import tempfile
import tomllib
import unittest
from pathlib import Path

from bigcherry.cli import engine_scope
from bigcherry.cli.main import build_parser
from bigcherry.core import engines, paths
from bigcherry.patch import lifecycle, rebase
from bigcherry.release import pin_bump


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
        self.assertEqual(_resolve(["status"]).engine, "llamacpp")
        self.assertEqual(_resolve(["--engine", "llamacpp", "status"]).engine, "llamacpp")

    def test_llamacpp_tool_refuses_another_engine(self):
        with self.assertRaises(SystemExit):
            _resolve(["--engine", "radiance", "status"])

    def test_undeclared_engine_is_refused(self):
        with self.assertRaises(SystemExit):
            _resolve(["--engine", "vllm", "patch-status"])

    def test_patch_commands_need_the_engine(self):
        for command in sorted(engine_scope._ENGINE_COMMANDS):
            with self.subTest(command=command), contextlib.redirect_stderr(io.StringIO()):
                argv = {
                    "patch-explain": ["patch-explain", "x"],
                    "patch-gates": ["patch-gates", "x", "--intent", "validate"],
                    "patch-doc": ["patch-doc", "--all"],
                    "patch-rebase-check": ["patch-rebase-check", "--all"],
                    "pin-bump": ["pin-bump", "b1"],
                    "pin-release": ["pin-release", "b1"],
                    "release-notes": ["release-notes", "b1"],
                    "sources": ["sources", "status"],
                    "smoke-cpu": ["smoke-cpu", "--build", "b"],
                }.get(command, [command])
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

    def test_dispositions_are_repository_wide_and_take_no_engine(self):
        # one store keyed by patch ID (engine-prefixed, so IDs never collide) and bound to a revision and digest
        argv = ["patch-disposition", "list"]
        self.assertIsNone(_resolve(argv).engine)
        with self.assertRaises(SystemExit):
            _resolve(["--engine", "llamacpp", *argv])

    def test_pin_status_reads_the_chosen_engines_recipes_and_pin(self):
        pinned = tomllib.loads(
            engines.load(paths.REPO_ROOT, "radiance").recipes_path(paths.REPO_ROOT).read_text(encoding="utf-8")
        )["pinned"]
        code, out, _ = self._run(["--engine", "radiance", "pin-status"])
        self.assertEqual(code, 0)
        self.assertIn(pinned, out)
        llamacpp_pin = tomllib.loads((paths.REPO_ROOT / "config" / "recipes.toml").read_text(encoding="utf-8"))["pinned"]
        _, llama_out, _ = self._run(["--engine", "llamacpp", "pin-status"])
        self.assertIn(llamacpp_pin, llama_out)
        self.assertNotIn(pinned, llama_out)

    def test_sources_status_uses_the_chosen_engines_registry_and_patch_tree(self):
        code, out, err = self._run(["--engine", "radiance", "sources", "status"])
        self.assertEqual(code, 0, err)
        self.assertIn("radiance: no external sources declared", out)
        self.assertIn("patch<->registry cross-check: OK", out)
        code, out, _ = self._run(["--engine", "llamacpp", "sources", "status"])
        self.assertEqual(code, 0)
        self.assertIn("source: ", out)

    def test_rebase_check_selects_from_the_chosen_engines_recipes(self):
        ids = rebase._selection_patch_ids("radiance", source_name="radiance-lab", all_patches=False)
        self.assertEqual(ids, ())  # rad0001 is superseded by radiance 1.7.0; the lab source has no evaluated fix left
        self.assertNotIn("0100_cmake_options", rebase._selection_patch_ids("radiance", all_patches=True))
        with self.assertRaises(rebase.RebaseCheckError):
            rebase._selection_patch_ids("radiance", source_name="bigcherry", all_patches=False)

    def test_commands_that_need_a_checkout_fail_cleanly_without_one(self):
        with tempfile.TemporaryDirectory() as empty:
            for argv in (
                ["--engine", "radiance", "--llama-root", empty, "patch-rebase-check", "--all"],
                ["--engine", "radiance", "--llama-root", empty, "audit"],
                ["--engine", "radiance", "--llama-root", empty, "patch-doc", "--all"],
            ):
                with self.subTest(argv=argv):
                    code, _, err = self._run(argv)
                    self.assertEqual(code, 2)
                    self.assertTrue(err.strip())
                    self.assertNotIn("Traceback", err)

    def test_radiance_release_commands_resolve_its_release_line(self):
        # the release line and the version scheme exist (MEN13): release-notes only stops on the missing release record
        code, _, err = self._run(["--engine", "radiance", "release-notes", "89cee7ce4144", "--stdout"])
        self.assertEqual(code, 1)
        self.assertIn("no release record releases/89cee7ce4144.json", err)

    def test_pin_bump_defaults_its_source_only_where_the_recipes_declare_it(self):
        self.assertEqual(pin_bump._resolve_fresh_selector("llamacpp", source_name=None), ("source", "bigcherry"))
        self.assertEqual(pin_bump._resolve_fresh_selector("radiance", source_name="radiance"), ("source", "radiance"))
        with self.assertRaises(pin_bump.PinBumpStop) as stopped:
            pin_bump._resolve_fresh_selector("radiance", source_name=None)
        self.assertEqual(stopped.exception.code, "SOURCE_REQUIRED")

    def test_each_engine_has_its_own_transition_marker_and_report_dir(self):
        from bigcherry import pin_transition

        self.assertNotEqual(pin_transition.marker_path("llamacpp"), pin_transition.marker_path("radiance"))
        self.assertNotEqual(pin_bump.default_report_dir("llamacpp", "b1"), pin_bump.default_report_dir("radiance", "b1"))

    def test_plan_items_are_the_chosen_engines_own(self):
        radiance = lifecycle.compute_all(engine="radiance")
        llamacpp = lifecycle.compute_all(engine="llamacpp")
        self.assertTrue(llamacpp)
        self.assertFalse(set(radiance) & set(llamacpp))
        for status in radiance.values():
            for patch_id in status.patch_ids:
                self.assertTrue(patch_id.startswith("rad"), patch_id)

    def test_engine_bench_keeps_its_own_engine_argument(self):
        args = _resolve(["engine-bench", "--engine", "radiance", "--binary", "b", "--model", "m",
                         "--out", "o", "--depth", "1"])
        self.assertEqual(args.engine, "radiance")


if __name__ == "__main__":
    unittest.main()
