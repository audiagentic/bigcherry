"""VTP14 N1 (rad0016): per-rank kernel code eligibility and --rank-devices, as a managed source patch.

Card-free. The contract tests run everywhere. The pinned-source test applies the patch to real radiance v1.17.0 when
a radiance checkout carrying that tag is reachable (RADIANCE_SRC, or the workspace's vendor/radiance) and is skipped
otherwise; it is what catches anchor drift against upstream, which a fixture built from the patch's own anchors cannot.
The C++ behaviour (refusals, per-rank selection, legacy path unchanged) is tested by tests/bc_rank_devices_test.cpp,
which the patch adds to radiance's own suite.
"""
from __future__ import annotations

import importlib.util
import os
import re
import subprocess
import sys
import tempfile
import tomllib
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "tools"))
from bigcherry.patcher import apply_all

PACKAGE = REPO / "engines/radiance/patches/rad0016_vtp14_rank_code_selection"
PIN = "v1.17.0"
PIN_COMMIT = "19d4e8fe0e18071b9b6300f15fc4ccf38dc4fa3d"


def patch_files():
    spec = importlib.util.spec_from_file_location("vtp14_n1_patch", PACKAGE / "patch.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return list(module.PATCHES)


def original_literal(edit):
    # Every upstream edit anchors exact upstream text (re.escape).
    return re.sub(r"\\(.)", r"\1", edit.anchor, flags=re.DOTALL)


def radiance_checkout() -> Path | None:
    candidates = [os.environ.get("RADIANCE_SRC")]
    for parent in REPO.parents:
        candidates.append(str(parent / "vendor" / "radiance"))
    for c in candidates:
        if not c or not (Path(c) / ".git").exists():
            continue
        r = subprocess.run(["git", "-C", c, "rev-parse", "--verify", "-q", PIN + "^{commit}"],
                           capture_output=True, text=True)
        if r.returncode == 0 and r.stdout.strip() == PIN_COMMIT:
            return Path(c)
    return None


class NativeRankCodeEligibilityTests(unittest.TestCase):
    def anchored_fixture(self, root: Path):
        for f in patch_files():
            if f.create:
                continue
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
        # one patch set in the main build, inert until its options are given; the as-released source stays unpatched
        self.assertEqual(recipe["patch-set"]["vtp14-mixed-tp"]["patches"],
                         ["rad0016_vtp14_rank_code_selection", "rad0017_vtp14_n3_shared_rendezvous", "rad0018_vtp14_n2_tp_ownership",
                          "rad0019_vtp14_n2_tp_ownership_fp8", "rad0020_vtp14_tp3_pipelined_prefill",
                          "rad0021_vtp14_weighted_twoshot_3rank", "rad0022_tune_pin", "rad0023_ar_timeline",
                          "rad0024_ts3_decode", "rad0025_multiseq_pipelined_prefill",
                          "rad0026_mixed_pipelined_prefill", "rad0027_mixed_one_pipelined_prefill",
                          "rad0028_rank_consistent_block_drafts"])
        self.assertEqual(recipe["source"]["radiance"]["patch-sets"][-1], "vtp14-mixed-tp")
        self.assertNotIn("vtp14-mixed-tp", recipe["source"]["radiance-native"]["patch-sets"])
        self.assertNotIn("radiance-vtp14-rank-code-selection", recipe["source"])
        self.assertTrue((REPO / "docs/planning/active/run-dense-27b-virtual-tp/VTP14.md").exists())

    def test_shape_guards_unique_and_created_files_from_src(self):
        files = patch_files()
        paths = [f.path for f in files]
        self.assertEqual(len(paths), len(set(paths)))
        created = {f.path for f in files if f.create}
        self.assertEqual(created, {"core/plugin/bc_rank_devices.h", "core/plugin/bc_rank_devices.cpp",
                                   "tests/bc_rank_devices_test.cpp"})
        guards = [e.guard for f in files for e in f.edits]
        self.assertEqual(len(guards), len(set(guards)))
        for f in files:
            if f.create:
                src = (PACKAGE / "src" / Path(f.path).name).read_text()
                self.assertEqual(f.edits[0].text, src)
                self.assertTrue(re.search(f.edits[0].guard, src, re.MULTILINE), f.path)
            else:
                for e in f.edits:
                    # A guard must recognise its own inserted text (idempotence) and not the upstream anchor.
                    self.assertTrue(re.search(e.guard, e.text, re.MULTILINE), e.id)
                    self.assertFalse(re.search(e.guard, original_literal(e), re.MULTILINE), e.id)

    def test_anchored_fixture_applies_idempotently(self):
        files = patch_files()
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            self.anchored_fixture(root)
            first = apply_all(files, root)
            self.assertTrue(all(f.ok for f in first), first)
            self.assertTrue(all(e.status == "applied" for f in first for e in f.results))
            before = {f.path: (root / f.path).read_bytes() for f in files}
            again = apply_all(files, root)
            self.assertTrue(all(f.ok for f in again), again)
            self.assertTrue(all(e.status == "already-applied" for f in again for e in f.results))
            self.assertEqual(before, {f.path: (root / f.path).read_bytes() for f in files})

    def test_source_drift_fails_closed(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            self.anchored_fixture(root)
            files = patch_files()
            target = next(f for f in files if f.path == "core/plugin/loader.cpp")
            path = root / target.path
            old = original_literal(next(e for e in target.edits if e.id == "n1-loader-ranks"))
            self.assertIn(old, path.read_text())
            path.write_text(path.read_text().replace(old, "upstream changed; no matching anchor"))
            snapshot = path.read_bytes()
            result = apply_all(files, root)
            self.assertTrue(any(not f.ok for f in result))
            self.assertEqual(path.read_bytes(), snapshot)

    def test_applies_to_pinned_radiance_source(self):
        checkout = radiance_checkout()
        if checkout is None:
            self.skipTest(f"no radiance checkout with {PIN} ({PIN_COMMIT[:8]}); set RADIANCE_SRC")
        files = patch_files()
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            for f in files:
                if f.create:
                    continue
                blob = subprocess.run(["git", "-C", str(checkout), "show", f"{PIN_COMMIT}:{f.path}"],
                                      capture_output=True, check=True).stdout
                dest = root / f.path
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(blob)
            first = apply_all(files, root)
            self.assertTrue(all(f.ok for f in first), first)
            self.assertTrue(all(e.status == "applied" for f in first for e in f.results))
            again = apply_all(files, root)
            self.assertTrue(all(e.status == "already-applied" for f in again for e in f.results))
            engine = (root / "core/engine_bringup.cpp").read_text()
            loader = (root / "core/plugin/loader.cpp").read_text()
            select = (root / "core/plugin/select.cpp").read_text()
            # Inert without the flag: the upstream calls survive on the legacy branch.
            self.assertIn("rad_tools_registry().set_code_device(devs[0]);", engine)
            self.assertIn("rad_gate_reference_kernels(ranks_[0]->program", engine)
            self.assertIn("rad::dev_code_runs(s.code_from, s.code_to, code_device_, &why) == 0", loader)
            self.assertIn("rad::dev_code_runs(s.code_from, s.code_to, code_devices_[i], &why_i)", loader)
            self.assertIn("if (need && (r->code_ranks & need) != need)", select)
            self.assertIn("bc_rank_collectives_agree(colls, &why)", engine)


if __name__ == "__main__":
    unittest.main()
