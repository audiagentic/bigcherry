"""Static contract tests for the nasone-rdna NRO/PNRO draft package group.

These tests deliberately do not claim GPU qualification. They pin the
repository-level invariants of the initial NRO landing: namespace uniqueness,
package/document completeness, dependency decomposition, fail-closed draft
selectors, and non-inclusion in production patch sets.
"""

from __future__ import annotations

import ast
import re
import tomllib
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
PATCHES = ROOT / "patches"
PLANS = ROOT / "docs" / "planning" / "active" / "patching-nasone-rdna-optimizations"

EXPECTED = {
    "1250_nro01_allreduce_q8_wire": ("NRO01", ["1001_hip_internal_allreduce"]),
    "1251_nro02_allreduce_fused_residual": ("NRO02", ["1250_nro01_allreduce_q8_wire"]),
    "1252_nro03_allreduce_p2p_provider": ("NRO03", ["1001_hip_internal_allreduce"]),
    "1253_nro04_gfx1100_bf16_chunked_gdn": ("NRO04", ["1221_rd50_gdn_chunked_recurrence"]),
    "1254_nro05_gdn_mtp_prefix_tail": ("NRO05", ["1253_nro04_gfx1100_bf16_chunked_gdn"]),
    "1255_nro06_adaptive_mtp_depth": ("NRO06", []),
    "1256_nro07_topk_hybrid": ("NRO07", []),
    "1257_nro08_topk_wave32": ("NRO08", ["1256_nro07_topk_hybrid"]),
}

# The capability rebaseline gave the active successors a dedicated PNRO
# namespace. Patch manifests retain their historical NRO plan IDs for
# provenance, so the test keeps both identities explicit instead of silently
# treating a renamed document as a missing package.
EXPECTED_PLAN_DOCS = {f"PNRO{i:02d}" for i in range(1, 16)}


def _manifest(patch_id: str) -> dict:
    return tomllib.loads((PATCHES / patch_id / "patch.toml").read_text(encoding="utf-8"))


def _patch_source(patch_id: str) -> str:
    return (PATCHES / patch_id / "patch.py").read_text(encoding="utf-8")


class NroPackageShapeTests(unittest.TestCase):
    def test_prefix_is_dedicated_and_plan_items_are_complete(self):
        for i in range(1, 16):
            item = PLANS / f"PNRO{i:02d}.md"
            self.assertTrue(item.is_file(), item)
            text = item.read_text(encoding="utf-8")
            self.assertRegex(text, rf"(?m)^id: PNRO{i:02d}$")
            self.assertRegex(text, r"(?m)^plan: patching-nasone-rdna-optimizations$")
            for heading in (
                "## Description", "## Steps", "## Detailed Solution & Technical Design",
                "## Code Samples & Guidance", "## Files", "## Validation",
                "## Effort & Risk", "## Standards", "## Acceptance Criteria",
                "## Notes", "## Change Log", "## Ledger-events",
            ):
                self.assertIn(heading, text, f"{item}: missing {heading}")
        self.assertEqual(
            {path.stem for path in PLANS.glob("PNRO*.md")},
            EXPECTED_PLAN_DOCS,
        )

    def test_priority_packages_have_full_draft_docs(self):
        for patch_id in EXPECTED:
            package = PATCHES / patch_id
            for name in ("patch.toml", "patch.py", "SUMMARY.md", "README.md", "TESTING.md"):
                self.assertTrue((package / name).is_file(), f"{patch_id}: missing {name}")
            # Validation adapters are intentionally deferred until a resolvable
            # Experiment Contract and executable correctness producer exist.
            self.assertFalse((package / "validation.toml").exists())

    def test_manifests_pin_group_state_plan_and_dependencies(self):
        for patch_id, (plan_id, requires) in EXPECTED.items():
            manifest = _manifest(patch_id)
            self.assertEqual(manifest["id"], patch_id)
            self.assertEqual(manifest["order"], int(patch_id.split("_", 1)[0]))
            self.assertEqual(manifest["group"], "nasone-rdna")
            self.assertEqual(manifest["state"], "untested")
            self.assertEqual(manifest["plan-ids"], [plan_id])
            self.assertEqual(manifest["requires"], requires)

    def test_summary_headers_match_manifest(self):
        for patch_id, (plan_id, _) in EXPECTED.items():
            text = (PATCHES / patch_id / "SUMMARY.md").read_text(encoding="utf-8")
            self.assertTrue(text.startswith(f"# {patch_id}\n\n"))
            self.assertIn("**Status:** untested", text)
            self.assertIn("**Group:** nasone-rdna", text)
            self.assertIn(f"**Plan item:** {plan_id}", text)

    def test_patch_modules_are_valid_python_and_export_patches(self):
        for patch_id in EXPECTED:
            source = _patch_source(patch_id)
            tree = ast.parse(source, filename=str(PATCHES / patch_id / "patch.py"))
            names = {
                target.id
                for node in tree.body if isinstance(node, ast.Assign)
                for target in node.targets if isinstance(target, ast.Name)
            }
            self.assertIn("PATCHES", names, patch_id)

    def test_no_nro_priority_patch_is_in_production_patch_sets(self):
        recipes = tomllib.loads((ROOT / "config" / "recipes.toml").read_text(encoding="utf-8"))
        production = set()
        for patch_set in recipes.get("patch-set", {}).values():
            production.update(patch_set.get("patches", []))
        leaked = production.intersection(EXPECTED)
        self.assertFalse(leaked, f"NRO drafts leaked into production: {leaked}")


class NroSafetyInvariantTests(unittest.TestCase):
    def test_q8_wire_is_disabled_by_default_and_unwired(self):
        source = _patch_source("1250_nro01_allreduce_q8_wire")
        self.assertIn("BIGCHERRY_NRO01_Q8_THRESHOLD_DEFAULT = 0", source)
        self.assertIn("not dispatched until NRO01", source)
        self.assertIn("i < ne ? src[i] : 0.0f", source)

    def test_residual_child_is_kernel_only(self):
        source = _patch_source("1251_nro02_allreduce_fused_residual")
        self.assertIn("bigcherry_nro02_add_residual_kernel", source)
        self.assertIn("bigcherry_nro02_q8_0_add_residual_kernel", source)
        self.assertNotIn("skip_node", source)

    def test_p2p_draft_encodes_source_current_push(self):
        source = _patch_source("1252_nro03_allreduce_p2p_provider")
        set_pos = source.index("ggml_cuda_set_device(src_device)")
        copy_pos = source.index("cudaMemcpyPeerAsync", set_pos)
        self.assertLess(set_pos, copy_pos)
        self.assertNotIn("p2p_issuer", source)
        self.assertIn("GGML_CUDA_AR_P2P", source)
        self.assertIn(", 0) != 0", source)

    def test_gfx1100_bf16_gdn_cannot_activate_in_initial_draft(self):
        source = _patch_source("1253_nro04_gfx1100_bf16_chunked_gdn")
        self.assertIn("GGML_CUDA_CC_IS_RDNA3(cc)", source)
        self.assertIn("S_v == 128", source)
        self.assertRegex(
            source,
            re.compile(r"bigcherry_nro04_gfx1100_bf16_ready\(\).*?return false;", re.S),
        )

    def test_mtp_prefix_is_exactly_n_tokens_minus_k(self):
        source = _patch_source("1254_nro05_gdn_mtp_prefix_tail")
        self.assertIn("*n_prefix = n_tokens - K", source)
        self.assertIn("n_seqs != 1", source)
        self.assertIn("K <= 1", source)

    def test_adaptive_controller_constants_are_frozen_for_first_sweep(self):
        source = _patch_source("1255_nro06_adaptive_mtp_depth")
        for fragment in (
            "case 1: return 2", "case 2: return 4", "case 3: return 10",
            "case 4: return 6", "case 5: return 3", "std::max(depth * 5, 20)",
        ):
            self.assertIn(fragment, source)
        self.assertIn("Pure/testable controller only", source)

    def test_topk_hybrid_and_wave32_are_separate_and_unwired(self):
        n7 = _patch_source("1256_nro07_topk_hybrid")
        n8 = _patch_source("1257_nro08_topk_wave32")
        self.assertIn("bigcherry_nro07_top_k_float_to_ordered", n7)
        self.assertIn("Selection kernels/dispatch intentionally await", n7)
        self.assertIn("static_assert(BLOCK_SIZE % 32 == 0", n8)
        self.assertIn("Runtime NRO07 dispatch is intentionally unchanged", n8)


if __name__ == "__main__":
    unittest.main()
