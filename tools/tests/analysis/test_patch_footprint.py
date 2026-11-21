"""Offline proof of RUM02's declared-composition footprint generator."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from bigcherry.analysis import patch_footprint
from bigcherry.core import paths
from bigcherry.patch.apply import Edit, FilePatch
from bigcherry.patch.patchset import PatchModule


class PatchFootprintTest(unittest.TestCase):
    def _example(self):
        original = FilePatch(
            path="core/x.cpp",
            edits=(
                Edit(id="one", anchor=r"foo\(\)", text="é\n"),
                Edit(id="two", anchor="bar", text="new", expect_matches=3, mode="replace_all"),
            ),
        )
        other = FilePatch(
            path="core/x.cpp",
            edits=(Edit(id="three", anchor="baz", text="more"),),
        )
        created = FilePatch(
            path="core/owned.h",
            create=True,
            edits=(Edit(id="create", anchor=r"\A", text="payload"),),
        )
        return patch_footprint.inventory(
            engine="test", pin="abc123", pin_version="2.2.0",
            compositions={
                "production": ("alpha", "beta"),
                "diagnostic": ("alpha", "beta"),
            },
            definitions={"alpha": [original, created], "beta": [other]},
        )

    def test_lexical_bytes_and_upstream_created_separation(self):
        result = self._example()
        self.assertEqual(result["totals"], {
            "composed_patches": 2, "edits": 4, "anchors": 3,
            "upstream_anchors": 3, "created_target_anchors": 0,
            "anchor_sites": 5, "anchor_bytes": len(br"foo\(\)") + 3 + 3,
            "inserted_bytes": len("é\nnewmorepayload".encode()), "created_edits": 1,
            "upstream_files": 1,
        })
        self.assertEqual(result["patches"][0]["patch"], "alpha")
        self.assertEqual(result["patches"][0]["created_files"], ["core/owned.h"])
        self.assertEqual(result["patches"][0]["sources"], ["diagnostic", "production"])
        self.assertEqual(result["upstream_files"][0]["patch_ids"], ["alpha", "beta"])
        self.assertEqual(result["upstream_files"][0]["patch_count"], 2)
        self.assertEqual(result["upstream_files"][0]["anchors"], 3)
        self.assertNotIn("core/owned.h", [row["path"] for row in result["upstream_files"]])

    def test_followup_edits_on_created_files_are_not_upstream(self):
        creation = FilePatch("owned/new.h", (Edit("create", r"\A", "body"),), create=True)
        followup = FilePatch("owned/new.h", (Edit("later", "body", "changed"),))
        result = patch_footprint.inventory(
            engine="test", pin="a", pin_version="2",
            compositions={"prod": ("alpha", "beta")},
            definitions={"alpha": [creation], "beta": [followup]},
        )
        self.assertEqual(result["totals"]["anchors"], 1)
        self.assertEqual(result["totals"]["upstream_anchors"], 0)
        self.assertEqual(result["totals"]["created_target_anchors"], 1)
        self.assertEqual(result["upstream_files"], [])
        self.assertEqual(result["patches"][0]["upstream_files"], [])

    def test_check_detects_both_missing_and_stale_without_writing(self):
        result = self._example()
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp) / "footprint"
            self.assertEqual(len(patch_footprint.publish(result, directory, check=True)), 2)
            self.assertFalse(directory.exists())
            self.assertEqual(len(patch_footprint.publish(result, directory, check=False)), 2)
            self.assertEqual(patch_footprint.publish(result, directory, check=True), [])
            (directory / "footprint.md").write_text("stale", encoding="utf-8")
            self.assertEqual(
                patch_footprint.publish(result, directory, check=True),
                [str(directory / "footprint.md")],
            )
            self.assertEqual((directory / "footprint.md").read_text(), "stale")

    def test_composition_fails_closed_for_missing_set_and_requires(self):
        alpha = PatchModule(
            engine="radiance", patch_id="rad9000_alpha", path=Path("a.py"),
            order=1, state="validated", upstream=None, content_hash="0",
            requires=("rad9001_beta",),
        )
        beta = PatchModule(
            engine="radiance", patch_id="rad9001_beta", path=Path("b.py"),
            order=2, state="validated", upstream=None, content_hash="1",
        )
        modules = {module.patch_id: module for module in (alpha, beta)}
        recipes = {
            "version": 2, "engine": "radiance", "pinned": "a1b2c3",
            "patch-set": {"primary": {"patches": ["rad9000_alpha"]}},
            "source": {"one": {"patch-sets": ["primary"]}},
        }
        with self.assertRaisesRegex(ValueError, "requires|missing"):
            patch_footprint.composed_members(recipes, modules, engine="radiance")
        recipes["patch-set"]["primary"]["patches"].append("rad9001_beta")
        members = patch_footprint.composed_members(recipes, modules, engine="radiance")
        self.assertEqual(members["one"], ("rad9001_beta", "rad9000_alpha"))
        recipes["source"]["one"]["patch-sets"].append("unknown")
        with self.assertRaisesRegex(ValueError, "unknown patch-set"):
            patch_footprint.composed_members(recipes, modules, engine="radiance")

    def test_committed_radiance_footprint_is_current(self):
        report = patch_footprint.collect("radiance")
        self.assertEqual(report["engine"], "radiance")
        self.assertEqual(
            {row["patch"] for row in report["patches"]},
            set().union(*(set(ids) for ids in report["compositions"].values())),
        )
        directory = paths.REPO_ROOT / "docs" / "evidence" / "radiance-patch-footprint"
        self.assertEqual(patch_footprint.publish(report, directory, check=True), [])


if __name__ == "__main__":
    unittest.main()
