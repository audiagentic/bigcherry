"""RE21/RE23: `build` is the multi-lane campaign engine -- fail-closed CLI
boundary tests.

Only the parser/dispatch boundary is exercised here (argparse wiring,
cmd_build_new's own validation, exit codes) -- run_campaign() and plan()'s
own correctness are already covered by test_campaign_planner.py/
test_campaign_lane.py. RE23 deleted the `legacy-build` compatibility command
and its recipe/tree-state mechanics entirely once the cutover's compatibility
gates were satisfied; the tests that used to prove `build` never routed to
it were retired along with it (see test_campaign_build_flip.py's
NoLegacyTreeStateMechanicsTests for the remaining static-reference guard).
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bigcherry import __main__ as cli  # noqa: E402


class BuildParserSplitTests(unittest.TestCase):
    def test_build_is_registered_and_legacy_build_is_gone(self):
        parser = cli.build_parser()
        subparsers_action = next(
            a for a in parser._actions if a.dest == "command" or hasattr(a, "choices") and a.choices)
        self.assertIn("build", subparsers_action.choices)
        self.assertNotIn("legacy-build", subparsers_action.choices)

    def test_new_build_rejects_recipe_flag_with_exit_2(self):
        parser = cli.build_parser()
        with self.assertRaises(SystemExit) as ctx:
            parser.parse_args(["build", "--recipe", "bigcherry"])
        self.assertEqual(ctx.exception.code, 2)

    def test_new_build_rejects_groups_flag_with_exit_2(self):
        parser = cli.build_parser()
        with self.assertRaises(SystemExit) as ctx:
            parser.parse_args(["build", "--groups", "core"])
        self.assertEqual(ctx.exception.code, 2)

    def test_new_build_rejects_states_flag_with_exit_2(self):
        parser = cli.build_parser()
        with self.assertRaises(SystemExit) as ctx:
            parser.parse_args(["build", "--states", "validated"])
        self.assertEqual(ctx.exception.code, 2)

    def test_new_build_rejects_variant_set_flag_with_exit_2(self):
        parser = cli.build_parser()
        with self.assertRaises(SystemExit) as ctx:
            parser.parse_args(["build", "--variant-set", "workload-max"])
        self.assertEqual(ctx.exception.code, 2)

    def test_new_build_rejects_force_flag_with_exit_2(self):
        parser = cli.build_parser()
        with self.assertRaises(SystemExit) as ctx:
            parser.parse_args(["build", "--force"])
        self.assertEqual(ctx.exception.code, 2)

    def test_new_build_rejects_target_flag_with_exit_2(self):
        parser = cli.build_parser()
        with self.assertRaises(SystemExit) as ctx:
            parser.parse_args(["build", "--target", "ggml-hip"])
        self.assertEqual(ctx.exception.code, 2)

    def test_new_build_rejects_dry_run_flag_with_exit_2(self):
        parser = cli.build_parser()
        with self.assertRaises(SystemExit) as ctx:
            parser.parse_args(["build", "--dry-run"])
        self.assertEqual(ctx.exception.code, 2)

    def test_new_build_dispatches_to_cmd_build_new(self):
        parser = cli.build_parser()
        args = parser.parse_args(["build", "--all"])
        self.assertIs(args.func, cli.cmd_build_new)


class CmdBuildNewRequestValidationTests(unittest.TestCase):
    """cmd_build_new's own pre-flight validation, never invoking
    run_campaign()/legacy cmd_build for a request it should reject outright.
    """

    def _args(self, **overrides):
        parser = cli.build_parser()
        argv = ["build"]
        for key, value in overrides.items():
            if value is None:
                continue
            flag = "--" + key.replace("_", "-")
            if value is True:
                argv.append(flag)
            elif isinstance(value, list):
                for item in value:
                    argv += [flag, item]
            else:
                argv += [flag, str(value)]
        return parser.parse_args(argv)

    def test_rejects_when_neither_profile_all_nor_lane_given(self):
        args = self._args()
        with patch("bigcherry.campaign_planner.run_campaign") as fake_run:
            code = cli.cmd_build_new(args)
        self.assertEqual(code, 2)
        fake_run.assert_not_called()

    def test_rejects_when_both_all_and_profile_given(self):
        args = self._args(all=True, profile="standard")
        with patch("bigcherry.campaign_planner.run_campaign") as fake_run:
            code = cli.cmd_build_new(args)
        self.assertEqual(code, 2)
        fake_run.assert_not_called()

    def test_rejects_when_both_profile_and_lane_given(self):
        args = self._args(profile="standard", lane=["a:b:c"])
        with patch("bigcherry.campaign_planner.run_campaign") as fake_run:
            code = cli.cmd_build_new(args)
        self.assertEqual(code, 2)
        fake_run.assert_not_called()

    def test_rejects_malformed_lane_selector(self):
        args = self._args(lane=["not-three-parts"])
        with patch("bigcherry.campaign_planner.run_campaign") as fake_run:
            code = cli.cmd_build_new(args)
        self.assertEqual(code, 2)
        fake_run.assert_not_called()

    def test_rejects_unknown_profile_via_planner_error(self):
        args = self._args(profile="does-not-exist")
        code = cli.cmd_build_new(args)
        self.assertEqual(code, 2)


class ExitCodeConventionTests(unittest.TestCase):
    """Exit 1 for a real lane execution failure, exit 0 only if every
    planned lane succeeds -- distinct from exit 2's syntax-rejection cases
    above.
    """

    def _args(self, **overrides):
        parser = cli.build_parser()
        argv = ["build"]
        for key, value in overrides.items():
            if value is None:
                continue
            flag = "--" + key.replace("_", "-")
            if value is True:
                argv.append(flag)
            else:
                argv += [flag, str(value)]
        return parser.parse_args(argv)

    def test_exit_1_when_a_planned_lane_fails(self):
        args = self._args(all=True)

        class FakeLane:
            def __init__(self, name):
                self.source_name, self.build_name, self.platform_name = name.split(":")

        def fake_plan(request, cfg):
            return (FakeLane("a:b:c"), FakeLane("d:e:f"))

        class FakeResult:
            build_plan_id = "bp1"
            workload_id = None

        def fake_run_campaign(lanes, *, cfg, context, store, run_id):
            return {"a:b:c": RuntimeError("boom"), "d:e:f": FakeResult()}

        with patch("bigcherry.campaign_planner.plan", side_effect=fake_plan), \
             patch("bigcherry.campaign_planner.run_campaign", side_effect=fake_run_campaign), \
             patch("bigcherry.config.load"):
            code = cli.cmd_build_new(args)
        self.assertEqual(code, 1)

    def test_exit_0_when_every_planned_lane_succeeds(self):
        args = self._args(all=True)

        class FakeLane:
            def __init__(self, name):
                self.source_name, self.build_name, self.platform_name = name.split(":")

        class FakeResult:
            build_plan_id = "bp1"
            workload_id = None

        def fake_plan(request, cfg):
            return (FakeLane("a:b:c"),)

        def fake_run_campaign(lanes, *, cfg, context, store, run_id):
            return {"a:b:c": FakeResult()}

        with patch("bigcherry.campaign_planner.plan", side_effect=fake_plan), \
             patch("bigcherry.campaign_planner.run_campaign", side_effect=fake_run_campaign), \
             patch("bigcherry.config.load"):
            code = cli.cmd_build_new(args)
        self.assertEqual(code, 0)


if __name__ == "__main__":
    unittest.main()
