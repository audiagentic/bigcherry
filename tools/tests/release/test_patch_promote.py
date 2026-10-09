from __future__ import annotations

import shutil
import sys
import tempfile
import tomllib
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bigcherry.release import patch_promote as pp  # noqa: E402


_PATCH_TOML = """\
schema = 1
id = "1348_demo"
order = 1348
state = "untested"
kind = "enhancement"
origin = "local"
backend = "hip"
plan-item = "QFP99"
plan-ids = ["QFP99"]
requires = []
conflicts = []
requires-options = []
forbids-options = []
subsystems = []
hardware = ["amd"]
validation-architectures = ["gfx1100"]
backends = ["hip"]
tags = ["optimization"]
"""

_PATCH_PY = '''\
from bigcherry.patcher import EnvDoc
STATE = "untested"

_GATE = r"""
static bool bc_demo() {
    const char * s = getenv("BIGCHERRY_DEMO");
    return s != nullptr && atoi(s) != 0;
}
"""

ENV_DOCS = (
    EnvDoc("BIGCHERRY_DEMO", "0|1", "0 (off)", "demo"),
)
'''

_RECIPES = """\
version = 2
pinned = "b11474"

[patch-set.validated-enhancements]
patches = ["1000_base"]
required-state = "validated"

[experiment.demo]
patches = ["1348_demo"]

[experiment.combo]
patches = ["1000_base", "1348_demo"]

[source.bigcherry]
ref = "pinned"
overlay = true
patch-sets = ["validated-enhancements"]
"""


class PatchPromoteTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="patch-promote-")
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        patch = self.root / "engines" / "llamacpp" / "patches" / "1348_demo"
        patch.mkdir(parents=True)
        (self.root / "config").mkdir()
        (patch / "patch.toml").write_text(_PATCH_TOML, encoding="utf-8")
        (patch / "patch.py").write_text(_PATCH_PY, encoding="utf-8")
        (patch / "SUMMARY.md").write_text(
            "# demo\n\n**Status:** untested\n\n## What it does\n\nDemo.\n",
            encoding="utf-8",
        )
        (patch / "README.md").write_text("# demo\n", encoding="utf-8")
        (self.root / "config" / "recipes.toml").write_text(
            _RECIPES, encoding="utf-8"
        )
        self.evidence = self.root / "evidence.md"
        self.evidence.write_text(
            "Model: model-a\nModel: model-b\nBuild: test\nA/B: +3%\n",
            encoding="utf-8",
        )

    def test_state_recipe_and_readme_are_promoted_together(self):
        info = pp._load_patch(self.root, "1348_demo")
        pp._set_state(info)
        evidence = self.evidence.read_text(encoding="utf-8")
        pp._write_promotion_record(info, evidence)
        release_evidence = pp._write_release_evidence(self.root, info, evidence)
        pp._update_recipes(self.root, ("1348_demo",))

        patch_toml = tomllib.loads(
            (info.root / "patch.toml").read_text(encoding="utf-8")
        )
        self.assertEqual(patch_toml["state"], "validated")
        self.assertIn('STATE = "validated"', (info.root / "patch.py").read_text())
        self.assertIn("**Status:** validated", (info.root / "SUMMARY.md").read_text())
        self.assertIn("## Promotion record", (info.root / "README.md").read_text())
        self.assertIn("A/B: +3%", (info.root / "README.md").read_text())
        self.assertEqual(
            release_evidence,
            self.root / "releases" / "evidence" / "1348_demo-promotion.md",
        )
        self.assertIn("A/B: +3%", release_evidence.read_text(encoding="utf-8"))

        recipes = tomllib.loads(
            (self.root / "config" / "recipes.toml").read_text(encoding="utf-8")
        )
        self.assertEqual(
            recipes["patch-set"]["validated-enhancements"]["patches"],
            ["1000_base", "1348_demo"],
        )
        self.assertEqual(recipes["experiment"]["demo"]["patches"], [])
        self.assertEqual(recipes["experiment"]["combo"]["patches"], ["1000_base"])



    def test_state_update_tolerates_patch_without_python_state(self):
        info = pp._load_patch(self.root, "1348_demo")
        (info.root / "patch.py").unlink()
        changed = pp._set_state(info)
        self.assertNotIn(info.root / "patch.py", changed)
        self.assertIn('state = "validated"', (info.root / "patch.toml").read_text())
        self.assertIn("**Status:** validated", (info.root / "SUMMARY.md").read_text())

    def test_promotion_requires_second_model_unless_profile_only(self):
        single = self.root / "single.md"
        single.write_text(
            "Model: model-a\nBuild: test\nnative llama.cpp baseline: matched\n",
            encoding="utf-8",
        )

        def fake_git(root, *args, check=True):
            if args[:2] == ("status", "--porcelain"):
                return ""
            if args[:3] == ("rev-parse", "--abbrev-ref", "HEAD"):
                return "main"
            if args[:2] == ("rev-parse", "HEAD"):
                return "same"
            if args[:2] == ("rev-parse", "origin/main"):
                return "same"
            if args[:2] == ("ls-remote", "--heads"):
                return ""
            return ""

        with self.assertRaisesRegex(pp.PatchPromoteError, "at least two distinct"):
            pp.promote(
                self.root,
                ("1348_demo",),
                "@" + str(single),
            )

        with mock.patch.object(pp, "_git", side_effect=fake_git), \
                mock.patch.object(pp, "_run", side_effect=pp._run), \
                mock.patch.object(pp, "_preflight", side_effect=pp.PatchPromoteError("stop after gate")):
            with self.assertRaisesRegex(pp.PatchPromoteError, "stop after gate"):
                pp.promote(
                    self.root,
                    ("1348_demo",),
                    "@" + str(single),
                    profile_only=True,
                )

    def test_default_on_requires_two_named_models_and_flips_gate(self):
        info = pp._load_patch(self.root, "1348_demo")
        with self.assertRaises(pp.PatchPromoteError):
            pp._promote_default_on(info, "Model: only-one\n")

        pp._promote_default_on(
            info, self.evidence.read_text(encoding="utf-8")
        )
        patch_py = (info.root / "patch.py").read_text(encoding="utf-8")
        self.assertIn("return s == nullptr || atoi(s) != 0;", patch_py)
        self.assertIn('"1 (on)"', patch_py)
        self.assertIn(
            "Default-on promotion", (info.root / "SUMMARY.md").read_text()
        )

    def test_check_set_includes_evidence_governance_lint_and_production_rebase(self):
        info = pp._load_patch(self.root, "1348_demo")
        calls = []
        with mock.patch.object(
            pp, "_check_command", side_effect=lambda root, args, label: calls.append((args, label))
        ):
            pp._run_checks(self.root, (info,))

        flat = [" ".join(args) for args, _ in calls]
        self.assertTrue(any("patch-verify-evidence 1348_demo" in x for x in flat))
        self.assertTrue(any("test_patch_catalog.py" in x for x in flat))
        self.assertTrue(any("test_patch_governance.py" in x for x in flat))
        self.assertTrue(any("test_recipes.py" in x for x in flat))
        self.assertTrue(any("patch-lint" in x for x in flat))
        self.assertTrue(
            any("patch-rebase-check --source bigcherry" in x for x in flat)
        )

    def test_failed_check_leaves_primary_untouched_and_discards_slice(self):
        originals = {
            path.relative_to(self.root): path.read_bytes()
            for path in self.root.rglob("*")
            if path.is_file()
        }
        worktree = self.root.parent / f"{self.root.name}-slice"
        shutil.copytree(self.root, worktree)
        self.addCleanup(lambda: shutil.rmtree(worktree, ignore_errors=True))

        def fake_git(root, *args, check=True):
            if args[:2] == ("status", "--porcelain"):
                return ""
            if args[:3] == ("rev-parse", "--abbrev-ref", "HEAD"):
                return "main"
            if args[:2] == ("rev-parse", "HEAD"):
                return "same"
            if args[:2] == ("rev-parse", "origin/main"):
                return "same"
            return ""

        with mock.patch.object(pp, "_git", side_effect=fake_git), \
                mock.patch.object(pp, "_preflight"), \
                mock.patch("bigcherry.cli.slice.start_slice", return_value=worktree), \
                mock.patch.object(pp, "_discard_promotion_slice") as discard, \
                mock.patch.object(
                    pp, "_run_checks", side_effect=pp.PatchPromoteError("boom")
                ):
            with self.assertRaises(pp.PatchPromoteError):
                pp.promote(
                    self.root,
                    ("1348_demo",),
                    "@" + str(self.evidence),
                )

        discard.assert_called_once()
        after = {
            path.relative_to(self.root): path.read_bytes()
            for path in self.root.rglob("*")
            if path.is_file()
        }
        self.assertEqual(after, originals)

    def test_release_versions_follow_feature_then_fix_policy(self):
        self.assertEqual(
            pp.pin_release.next_version(["11474.0.0"], 11474, "minor"),
            "11474.1.0",
        )
        self.assertEqual(
            pp.pin_release.next_version(
                ["11474.0.0", "11474.1.0"], 11474, "patch"
            ),
            "11474.1.1",
        )

    def test_ledger_payload_is_machine_readable_and_plan_bound(self):
        info = pp._load_patch(self.root, "1348_demo")
        payload = pp._ledger_payload(
            (info,),
            ["engines/llamacpp/patches/1348_demo/patch.toml", "config/recipes.toml"],
            "feat(patch): promote 1348_demo",
        )
        self.assertEqual(payload["change_class"], "feature")
        self.assertEqual(payload["plan_item_ids"], ["QFP99"])
        self.assertIn("config/recipes.toml", payload["files"])


if __name__ == "__main__":
    unittest.main()
