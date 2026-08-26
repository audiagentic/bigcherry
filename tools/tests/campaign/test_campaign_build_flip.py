"""RE22: fail-closed cutover tests for RE21's `build` flip -- the actual
acceptance evidence for RE14 step 7 ("the default campaign path contains no
implicit fallback to legacy"), not just an assertion that it's true.

test_build_flip.py (RE21) already covers the broadened unsupported-flag
surface (--recipe/--groups/--states/--variant-set/--force/--target/
--dry-run all rejected with exit 2 on the new `build` parser) and the
exit-1-vs-exit-2 distinction -- not duplicated here. This file covers the
remaining checks from RE22's acceptance list: exact canonical lane-set
production, fault isolation at the CLI/result level (not just aggregate
counts), and the static/structural guarantee that the new path never
touches legacy tree-state mechanics.

RE23 note: this file used to also carry a `PoisonedLegacyTests` class that
poisoned ``bigcherry.__main__.cmd_build`` and proved `build` never called
it. RE23 deleted `cmd_build`/`legacy-build` entirely once the cutover's
compatibility gates were satisfied, so that check became structurally
impossible to write (there is no longer a legacy function to poison) --
the intent survives in NoLegacyTreeStateMechanicsTests below, which proves
the same thing at the level of "the symbol doesn't exist anywhere the new
path could reference it", a stronger guarantee than "it exists but wasn't
called this one time".
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bigcherry import __main__ as cli  # noqa: E402
from bigcherry.core import config as campaign_config # noqa: E402
from bigcherry.core import context as context_module # noqa: E402
from bigcherry.core import paths # noqa: E402


def _build_args(**overrides):
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


class ExactCanonicalLaneSetTests(unittest.TestCase):
    """Check 3: `build --all` produces exactly RE19's canonical lane set,
    no more, no fewer -- against the real recipes.toml, not a mock.
    """

    def test_build_all_produces_exactly_the_standard_profile_lanes(self):
        args = _build_args(all=True)
        cfg = campaign_config.load(paths.RECIPES)
        expected = {
            f"{sel.source}:{sel.build}:{sel.platform}"
            for sel in cfg.campaigns["standard"].lanes
        }
        self.assertTrue(expected, "the real standard profile must be non-empty")

        captured = {}

        def capturing_run_campaign(lanes, *, cfg, context, store, run_id):
            from bigcherry.campaign.planner import lane_id
            captured["ids"] = {lane_id(lane) for lane in lanes}
            return {lane_id(lane): _FakeOkResult() for lane in lanes}

        with patch("bigcherry.campaign.planner.run_campaign", side_effect=capturing_run_campaign):
            code = cli.cmd_build_new(args)

        self.assertEqual(code, 0)
        self.assertEqual(captured["ids"], expected)


class _FakeOkResult:
    build_plan_id = "bp1"
    workload_id = None


class FaultIsolationAtCliLevelTests(unittest.TestCase):
    """Check 4: one lane failing must not corrupt or misidentify a sibling
    lane's own result -- not just 'the aggregate failed count is right'.
    """

    def test_sibling_lane_result_identity_is_untouched_by_a_failing_lane(self):
        import io
        args = _build_args(all=True)

        class FakeLane:
            def __init__(self, name):
                self.source_name, self.build_name, self.platform_name = name.split(":")

        sibling_result = _FakeOkResult()
        sibling_result.build_plan_id = "sibling-bp"

        def fake_plan(request, cfg):
            return (FakeLane("a:b:c"), FakeLane("d:e:f"))

        def fake_run_campaign(lanes, *, cfg, context, store, run_id):
            return {"a:b:c": RuntimeError("injected fault"), "d:e:f": sibling_result}

        stdout, stderr = io.StringIO(), io.StringIO()
        with patch("bigcherry.campaign.planner.plan", side_effect=fake_plan), \
             patch("bigcherry.campaign.planner.run_campaign", side_effect=fake_run_campaign), \
             patch("bigcherry.core.config.load"), \
             patch("sys.stdout", stdout), patch("sys.stderr", stderr):
            code = cli.cmd_build_new(args)

        self.assertEqual(code, 1)
        # Correct per-lane attribution: the failed lane's id appears with
        # its failure on stderr, the sibling's id appears with ITS OWN
        # (unrelated) build_plan_id on stdout -- neither corrupted nor
        # swapped with the other.
        self.assertIn("a:b:c: FAILED", stderr.getvalue())
        self.assertIn("injected fault", stderr.getvalue())
        self.assertIn("d:e:f: ok build_plan_id=sibling-bp", stdout.getvalue())
        self.assertNotIn("d:e:f", stderr.getvalue())
        self.assertNotIn("a:b:c", stdout.getvalue())


class NoLegacyTreeStateMechanicsTests(unittest.TestCase):
    """Check 5: no lane's execution path references _ensure_tree_state,
    _reset_tree, or the legacy _build_dir identity function, and the new
    path's work_root structurally never overlaps the legacy checkout path.
    """

    _NEW_PATH_MODULES = (
        "campaign/planner", "campaign/lane", "campaign/execution",
        "campaign/workers", "campaign/plan", "campaign/build",
        "campaign/source", "build/builds", "core/artifacts", "core/context",
    )

    def test_new_path_modules_never_reference_legacy_tree_state_functions(self):
        import bigcherry
        root = Path(bigcherry.__file__).resolve().parent
        forbidden = ("_ensure_tree_state", "_reset_tree")
        for name in self._NEW_PATH_MODULES:
            source = (root / f"{name}.py").read_text(encoding="utf-8")
            for symbol in forbidden:
                self.assertNotIn(
                    symbol, source,
                    f"{name}.py must never reference legacy {symbol}")
            # The new path's own build_directory() (builds.py) is a distinct
            # identity function from legacy's _build_dir() (__main__.py) --
            # confirm the new modules never import or call the legacy one.
            self.assertNotIn(
                "_build_dir(", source,
                f"{name}.py must never call legacy __main__._build_dir")

    def test_legacy_mechanics_are_actually_gone_from_main(self):
        # RE23 acceptance itself, not just "the new path never referenced
        # them": the symbols must not exist anywhere in __main__.py at all.
        import bigcherry.__main__ as main_module
        for symbol in ("cmd_build", "_ensure_tree_state", "_reset_tree",
                       "_build_dir", "GENERATED_IN_TREE", "_verify_tree",
                       "_generate_for", "_cmake_configure_args",
                       "_build_one_recipe"):
            self.assertFalse(
                hasattr(main_module, symbol),
                f"RE23 deleted {symbol}; it must not exist on __main__ anymore")
        parser = cli.build_parser()
        subparsers_action = next(
            a for a in parser._actions
            if a.dest == "command" or hasattr(a, "choices") and a.choices)
        self.assertNotIn("legacy-build", subparsers_action.choices)

    def test_new_campaign_work_root_never_overlaps_legacy_checkout_root(self):
        # Legacy mutates paths.llama_root()'s default (REPO_ROOT/vendor/
        # llama.cpp) in place. The new path only ever writes under
        # ProjectContext.resolve()'s work_root, which by construction (see
        # context.py) defaults outside the repo tree entirely (LOCALAPPDATA/
        # BigCherry/work or ~/.cache/bigcherry) -- never under vendor/.
        legacy_root = paths.llama_root().resolve()
        context = context_module.ProjectContext.resolve()
        new_work_root = context.work_root.resolve()
        self.assertNotEqual(legacy_root, new_work_root)
        self.assertFalse(
            str(new_work_root).startswith(str(legacy_root)),
            "new campaign work_root must not live inside the legacy checkout")
        self.assertFalse(
            str(legacy_root).startswith(str(new_work_root)),
            "legacy checkout must not live inside the new campaign work_root")


if __name__ == "__main__":
    unittest.main()
