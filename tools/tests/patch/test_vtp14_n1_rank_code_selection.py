"""N1: native Radiance per-rank code eligibility source-patch contract tests.

Card-free; tests exact source anchored patch behaviour and production
isolation, leaving device discovery/startup deliberately unchanged.
"""
from __future__ import annotations

import importlib.util
import re
import sys
import tempfile
import tomllib
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "tools"))
from bigcherry.patcher import apply_all

PACKAGE = (
    REPO / "engines/radiance/patches/rad0016_vtp14_rank_code_selection"
)


def patch_files():
    spec = importlib.util.spec_from_file_location("vtp14_n1_patch", PACKAGE / "patch.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return list(module.PATCHES)


def original_literal(edit):
    # The patch tool anchors exactly the upstream source (re.escape).
    return re.sub(r"\\(.)", r"\1", edit.anchor, flags=re.DOTALL)


class NativeRankCodeEligibilityTests(unittest.TestCase):
    def fixture(self, root: Path):
        for f in patch_files():
            dest = root / f.path
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text("\n".join(original_literal(e) for e in f.edits) + "\n")

    def test_patch_is_isolated_and_production_unchanged(self):
        with (PACKAGE / "patch.toml").open("rb") as f:
            meta = tomllib.load(f)
        with (REPO / "engines/radiance/recipes.toml").open("rb") as f:
            recipe = tomllib.load(f)
        self.assertEqual(meta["state"], "untested")
        self.assertEqual(meta["order"], 16)
        self.assertEqual(meta["plan-ids"], ["VTP14"])
        self.assertEqual(meta["tags"], ["tensor-parallel", "dispatch"])
        name = "vtp14-rank-code-selection"
        self.assertEqual(recipe["patch-set"][name]["patches"],
                         ["rad0016_vtp14_rank_code_selection"])
        self.assertIn(name,
                      recipe["source"]["radiance-vtp14-rank-code-selection"]["patch-sets"])
        self.assertNotIn(name, recipe["source"]["radiance"]["patch-sets"])
        self.assertNotIn(name, recipe["source"]["radiance-native"]["patch-sets"])
        self.assertTrue((REPO / "docs/planning/active/run-dense-27b-virtual-tp/VTP14.md").exists())

    def test_exact_source_is_idempotent_with_rank_code_contract(self):
        edits = patch_files()
        self.assertEqual(len(edits), 7)
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            self.fixture(root)
            first = apply_all(edits, root)
            self.assertTrue(all(f.ok for f in first), first)
            self.assertTrue(all(e.status == "applied" for f in first for e in f.results))
            before = {f.path: (root / f.path).read_bytes() for f in edits}
            again = apply_all(edits, root)
            self.assertTrue(all(f.ok for f in again), again)
            self.assertTrue(all(e.status == "already-applied" for f in again for e in f.results))
            self.assertEqual(before, {f.path: (root / f.path).read_bytes() for f in edits})
            loader = (root / "core/plugin/loader.cpp").read_text()
            select = (root / "core/plugin/select.cpp").read_text()
            registry = (root / "core/plugin/registry.h").read_text()
            builder = (root / "core/build/rad_builder.cpp").read_text()
            self.assertIn("int Registry::set_code_devices", loader)
            self.assertIn("std::find(devices.begin(), devices.begin() + i", loader)
            self.assertIn("rad::dev_code_runs(s.code_from, s.code_to, code_devices_[i]", loader)
            self.assertIn("row.code_eligible_by_rank = code_mask", loader)
            self.assertIn("if (rank < 0 || (size_t)rank >= code_devices_.size()", select)
            self.assertIn("has no compatible device code", select)
            self.assertIn("set_code_device(int device) { code_devices_.clear();", registry)
            self.assertIn("reg_.build_bands", builder)
            self.assertIn("oi.weights.empty() ? nullptr : &accept, rank_", builder)
            self.assertNotIn("BC_VTP_NATIVE_TP3", "".join(
                (root / f.path).read_text() for f in edits
            ))
            guards = [e.guard for f in edits for e in f.edits]
            self.assertEqual(len(guards), len(set(guards)))

    def test_source_drift_fails_without_rewriting_unrelated_files(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            self.fixture(root)
            files = patch_files()
            target = files[2]
            path = root / target.path
            old = original_literal(target.edits[0])
            self.assertIn(old, path.read_text())
            path.write_text(path.read_text().replace(old, "upstream changed; no matching anchor"))
            snapshot = {f.path: (root / f.path).read_bytes() for f in files}
            result = apply_all(files, root)
            self.assertTrue(any(not f.ok for f in result))
            # Failure is atomic per file. Other files may be staged by the
            # outer collection, so test the failing file's immutability.
            self.assertEqual(path.read_bytes(), snapshot[target.path])


if __name__ == "__main__":
    unittest.main()
