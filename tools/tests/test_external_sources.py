"""Contract tests for the external-source registry and patch PROVENANCE.

These pin the provenance machinery that future agents depend on: every
external backport patch must be linked to its source, snapshot, and tracked
commit in external-sources.toml, and the two must agree.
"""

from __future__ import annotations

import sys
import tomllib
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ROOT = Path(__file__).resolve().parents[2]
from bigcherry import sources as src  # noqa: E402


class TestRegistryStructure(unittest.TestCase):
    def setUp(self):
        self.registry = src.load_registry()

    def test_version_and_sources_present(self):
        self.assertEqual(self.registry["version"], 1)
        self.assertGreaterEqual(len(self.registry["sources"]), 1)

    def test_rdna_source_shape(self):
        rdna = next(s for s in self.registry["sources"] if s["id"] == "stew675-rdna-boosts")
        self.assertTrue(rdna["repo"].startswith("https://github.com/"))
        self.assertTrue(rdna["locator"])
        self.assertTrue(rdna["upstream"].startswith("https://github.com/ggml-org/llama.cpp"))
        active = [s for s in rdna["snapshots"] if s.get("active")]
        self.assertEqual(len(active), 1, "exactly one active snapshot")
        # The rebase history must be preserved: v1 stays as a historical
        # snapshot, it is not silently rewritten away.
        labels = {s.get("label") for s in rdna["snapshots"]}
        self.assertGreaterEqual(len(labels), 2, "the rebased v1 snapshot must remain historical")

    def test_every_tracked_commit_is_unique_40hex(self):
        seen = set()
        for source in self.registry["sources"]:
            for entry in source.get("tracked", []):
                self.assertRegex(entry["commit"], r"^[0-9a-f]{40}$")
                self.assertNotIn(entry["commit"], seen, f"duplicate {entry['commit']}")
                seen.add(entry["commit"])
                self.assertIn(entry["status"], src.TRACKED_STATUSES)

    def test_superseded_ssmentry_points_at_successor(self):
        rdna = next(s for s in self.registry["sources"] if s["id"] == "stew675-rdna-boosts")
        by_item = {}
        for entry in rdna["tracked"]:
            by_item.setdefault(entry.get("plan-item"), []).append(entry)
        # The SSM pre-scan chain supersedes both RD14 and RD16.
        rd24 = by_item.get("RD24", [])
        self.assertEqual(len(rd24), 1)
        superseded = {e["plan-item"] for e in rdna["tracked"] if e["status"] == "superseded"}
        self.assertEqual(superseded, {"RD14", "RD16"})
        # The excluded MTP feature commits are declared, not silently dropped.
        excluded = [e for e in rdna["tracked"] if e["status"] == "excluded"]
        self.assertGreaterEqual(len(excluded), 2)


class TestPatchProvenanceCrossCheck(unittest.TestCase):
    def test_cross_check_is_clean(self):
        problems = src.cross_check_patches()
        self.assertEqual(problems, [], f"patch<->registry problems: {problems}")

    def test_rd19_rd20_patches_carry_provenance(self):
        for stem, item in (("1200_rd19_single_gpu_meta_bypass", "RD19"),
                           ("1201_rd20_attn_gate_tp_split", "RD20")):
            pfile = ROOT / "patches" / f"{stem}.py"
            self.assertTrue(pfile.is_file(), f"missing {pfile}")
            prov = src._patch_provenance(pfile)
            self.assertIsNotNone(prov, f"{stem}: no PROVENANCE dict")
            self.assertEqual(prov["source-id"], "stew675-rdna-boosts")
            self.assertEqual(prov["plan-item"], item)
            self.assertRegex(prov["fork-commit"], r"^[0-9a-f]{40}$")
            self.assertRegex(prov["original-commit"], r"^[0-9a-f]{40}$")
            self.assertNotEqual(prov["fork-commit"], prov["original-commit"],
                                f"{stem}: fork and original commit must differ (rebase)")

    def test_rdna_patches_are_untested_and_in_their_own_group(self):
        """The first-sweep isolation contract: rdna-boosts patches must NOT be
        pullable by the production groups, so a native build cannot pick them
        up accidentally."""
        from bigcherry import patchset
        for info in patchset.describe():
            if info.group == "rdna-boosts":
                self.assertEqual(info.state, "untested", f"{info.name}: expected untested")

    def test_rdna_patches_not_in_production_patch_sets(self):
        """recipes.toml's framework and validated-enhancements sets are exact
        lists; the new patches must not appear in them until promoted."""
        recipes = tomllib.loads((ROOT / "recipes.toml").read_text(encoding="utf-8"))
        production = set()
        for patch_set in recipes.get("patch-set", {}).values():
            production.update(patch_set.get("patches", []))
        for patch_id in ("1200_rd19_single_gpu_meta_bypass",
                         "1201_rd20_attn_gate_tp_split"):
            self.assertNotIn(patch_id, production,
                             f"{patch_id} must not be in a production patch-set "
                             f"before isolated validation")

    def test_cross_check_detects_provenance_mismatch(self):
        """A corrupted PROVENANCE must be caught, not silently accepted."""
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            (tmp_path / "1200_rd19_single_gpu_meta_bypass.py").write_text(
                (ROOT / "patches" / "1200_rd19_single_gpu_meta_bypass.py")
                .read_text(encoding="utf-8")
                .replace('"plan-item": "RD19"', '"plan-item": "RD99"'),
                encoding="utf-8",
            )
            problems = src.cross_check_patches(src.load_registry(), tmp_path)
            self.assertTrue(any("plan-item" in p for p in problems),
                            f"expected a plan-item mismatch problem, got {problems}")

    def test_sources_status_runs_clean(self):
        """`sources status` (the offline path) returns 0 and no stderr
        problems on the committed registry."""
        import io
        from contextlib import redirect_stdout
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = src._status()
        self.assertEqual(rc, 0)
        self.assertIn("stew675-rdna-boosts", buf.getvalue())
        self.assertIn("cross-check: OK", buf.getvalue())


if __name__ == "__main__":
    unittest.main()
