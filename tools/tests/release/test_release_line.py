"""The release line follows the pin by itself: a bump declares `release-as` when the pin's major moves, a CI check
catches a miss or a stale key, and the key is cleared once the release exists."""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from bigcherry.release import pin_bump, release_line  # noqa: E402

_REPO = Path(__file__).resolve().parents[3]

_CONFIG = """{
  "packages": {
    "engines/radiance": {
      "release-type": "simple",
      "component": "bc-radiance"
    },
    "engines/llamacpp": {
      "release-type": "simple",
      "component": "bc-llamacpp"
    }
  }
}
"""


class ReleaseLineTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix="release-line-"))
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)
        (self.root / "release-please-config.json").write_text(_CONFIG, encoding="utf-8")
        self.manifest({"engines/radiance": "1170.38.0", "engines/llamacpp": "11552.6.0"})

    def manifest(self, versions):
        (self.root / ".release-please-manifest.json").write_text(json.dumps(versions), encoding="utf-8")

    def release_as(self, package="engines/radiance"):
        return json.loads((self.root / "release-please-config.json").read_text(encoding="utf-8"))["packages"][package].get("release-as")

    def pin(self, package, release_as):
        text = (self.root / "release-please-config.json").read_text(encoding="utf-8")
        (self.root / "release-please-config.json").write_text(
            text.replace(f'"{package}": {{', f'"{package}": {{\n      "release-as": "{release_as}",', 1), encoding="utf-8")

    # declare: what a pin bump does
    def test_a_new_major_sets_release_as_and_keeps_the_formatting(self):
        changed = release_line.declare(self.root, "engines/radiance", 2020)
        self.assertTrue(changed)
        self.assertEqual(self.release_as(), "2020.0.0")
        self.assertIn('      "release-as": "2020.0.0",\n      "release-type"', (self.root / "release-please-config.json").read_text())

    def test_the_same_major_sets_nothing(self):
        self.assertFalse(release_line.declare(self.root, "engines/radiance", 1170))
        self.assertIsNone(self.release_as())

    def test_the_other_line_is_untouched(self):
        release_line.declare(self.root, "engines/radiance", 2020)
        self.assertIsNone(self.release_as("engines/llamacpp"))

    def test_the_llama_line_moves_with_its_build(self):
        self.assertTrue(release_line.declare(self.root, "engines/llamacpp", 11600))
        self.assertEqual(self.release_as("engines/llamacpp"), "11600.0.0")

    def test_declaring_twice_is_a_no_op_and_a_different_pending_value_is_refused(self):
        release_line.declare(self.root, "engines/radiance", 2020)
        self.assertFalse(release_line.declare(self.root, "engines/radiance", 2020))
        with self.assertRaisesRegex(release_line.ReleaseLineError, "already"):
            release_line.declare(self.root, "engines/radiance", 2030)

    def test_returning_to_the_manifest_major_drops_a_pending_key_that_was_never_released(self):
        release_line.declare(self.root, "engines/radiance", 2020)
        self.assertTrue(release_line.declare(self.root, "engines/radiance", 1170))
        self.assertIsNone(self.release_as())

    # check: what CI enforces
    def test_a_pin_ahead_of_the_manifest_without_release_as_fails_with_the_fix(self):
        found = release_line.problems(self.root, "engines/radiance", 2020)
        self.assertEqual(len(found), 1)
        self.assertIn("2020", found[0])
        self.assertIn("1170.38.0", found[0])
        self.assertIn("release-as", found[0])

    def test_the_declared_key_passes_before_during_and_just_after_the_release(self):
        self.pin("engines/radiance", "2020.0.0")
        self.assertEqual(release_line.problems(self.root, "engines/radiance", 2020), [])
        self.manifest({"engines/radiance": "2020.0.0", "engines/llamacpp": "11552.6.0"})  # the release PR's own state
        self.assertEqual(release_line.problems(self.root, "engines/radiance", 2020), [])

    def test_a_key_older_than_the_manifest_is_stale(self):
        self.pin("engines/radiance", "2020.0.0")
        self.manifest({"engines/radiance": "2020.1.0", "engines/llamacpp": "11552.6.0"})
        found = release_line.problems(self.root, "engines/radiance", 2020)
        self.assertEqual(len(found), 1)
        self.assertIn("stale", found[0])

    def test_a_key_that_does_not_name_the_pin_major_fails(self):
        self.pin("engines/radiance", "2030.0.0")
        found = release_line.problems(self.root, "engines/radiance", 2020)
        self.assertEqual(len(found), 1)
        self.assertIn("2020.0.0", found[0])

    def test_a_matching_major_without_a_key_passes(self):
        self.assertEqual(release_line.problems(self.root, "engines/radiance", 1170), [])

    # clearing once the release exists
    def test_a_released_key_is_cleared(self):
        self.pin("engines/radiance", "2020.0.0")
        self.manifest({"engines/radiance": "2020.0.0", "engines/llamacpp": "11552.6.0"})
        self.assertEqual(release_line.clear_released(self.root), ["engines/radiance"])
        self.assertIsNone(self.release_as())
        self.assertEqual(json.loads((self.root / "release-please-config.json").read_text())["packages"]["engines/llamacpp"]["component"],
                         "bc-llamacpp")

    def test_a_pending_key_is_kept(self):
        self.pin("engines/radiance", "2020.0.0")
        self.assertEqual(release_line.clear_released(self.root), [])
        self.assertEqual(self.release_as(), "2020.0.0")

    def test_every_released_package_is_cleared_in_one_pass(self):
        self.pin("engines/radiance", "2020.0.0")
        self.pin("engines/llamacpp", "11600.0.0")
        self.manifest({"engines/radiance": "2020.3.0", "engines/llamacpp": "11600.0.0"})
        self.assertEqual(sorted(release_line.clear_released(self.root)), ["engines/llamacpp", "engines/radiance"])

    # the declare phase of pin-bump
    def test_pin_bump_declare_commits_release_as_with_the_pin(self):
        commits = []

        def git(root, *args, **kw):
            commits.append(args)
            return "a" * 40

        engine = mock.Mock(version_source="pinned-version")
        engine.release_package.return_value = "engines/radiance"
        with (mock.patch.object(pin_bump, "recipes_path", return_value=self.root / "recipes.toml"),
              mock.patch.object(pin_bump.recipes_module, "repin", return_value="v2.1.0"),
              mock.patch.object(pin_bump.recipes_module, "set_pinned_version"),
              mock.patch.object(pin_bump.recipes_module, "version_of_ref", return_value="2.2.0"),
              mock.patch.object(pin_bump.engines, "load", return_value=engine),
              mock.patch.object(pin_bump.components_module, "repin_components", return_value=[]),
              mock.patch.object(pin_bump.release_line, "pin_major", return_value=2020),
              mock.patch.object(pin_bump, "resolve_pin_sha", return_value="b" * 40),
              mock.patch.object(pin_bump.pin_transition, "write"),
              mock.patch.object(pin_bump.pin_transition, "marker_path", return_value=self.root / "marker.json"),
              mock.patch.object(pin_bump, "_git", side_effect=git)):
            (self.root / "marker.json").write_text("{}")
            pin_bump.run_phase_declare(repo_root=self.root, engine="radiance", vendor_root=self.root, target_ref="v2.2.0")
        self.assertEqual(self.release_as(), "2020.0.0")
        added = next(args for args in commits if args and args[0] == "add")
        self.assertIn("release-please-config.json", added)

    # the repository itself
    def test_this_repository_declares_its_release_line(self):
        self.assertEqual(release_line.check_repository(_REPO), [])


if __name__ == "__main__":
    unittest.main()
