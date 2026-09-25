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
    # e06dcf63 is one atomic commit: NRO02's fusion ships inside 1250.
    "1250_nro01_allreduce_q8_wire": (["NRO01", "NRO02"], ["1252_nro03_allreduce_p2p_provider"]),
    "1252_nro03_allreduce_p2p_provider": ("NRO03", []),
    "1253_nro04_gfx1100_bf16_chunked_gdn": ("NRO04", []),
    "1254_nro05_gdn_mtp_prefix_tail": ("NRO05", ["1253_nro04_gfx1100_bf16_chunked_gdn"]),
    "1255_nro06_adaptive_mtp_depth": ("NRO06", []),
    "1256_nro07_topk_hybrid": ("NRO07", []),
    "1257_nro08_topk_wave32": ("NRO08", ["1256_nro07_topk_hybrid"]),
}

# The capability rebaseline gave the active successors a dedicated PNRO
# namespace. Patch manifests retain their historical NRO plan IDs for
# provenance, so the test keeps both identities explicit instead of silently
# treating a renamed document as a missing package.
#
# PNRO17 (2026-09-11) was a real-hardware finding filed from PVPS02's
# merge-gate run, root-caused and fixed the same day, then marked
# completed -- it now lives under docs/planning/completed/ rather than
# this PLANS (active) directory, so it is correctly absent from this glob.
EXPECTED_PLAN_DOCS = {f"PNRO{i:02d}" for i in range(1, 18)}


def _manifest(patch_id: str) -> dict:
    return tomllib.loads((PATCHES / patch_id / "patch.toml").read_text(encoding="utf-8"))


def _patch_source(patch_id: str) -> str:
    return (PATCHES / patch_id / "patch.py").read_text(encoding="utf-8")



def _load_patch_module(patch_id):
    import importlib.util

    path = PATCHES / patch_id / "patch.py"
    spec = importlib.util.spec_from_file_location(f"nro_{patch_id}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class NroPackageShapeTests(unittest.TestCase):
    def test_prefix_is_dedicated_and_plan_items_are_complete(self):
        for i in range(1, 18):
            item = PLANS / f"PNRO{i:02d}.md"
            if not item.is_file():
                item = Path("docs/planning/completed/patching-nasone-rdna-optimizations") / f"PNRO{i:02d}.md"
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
            {path.stem for path in PLANS.glob("PNRO*.md")}
            | {path.stem for path in Path("docs/planning/completed/patching-nasone-rdna-optimizations").glob("PNRO*.md")},
            EXPECTED_PLAN_DOCS,
        )

    def test_priority_packages_have_full_draft_docs(self):
        for patch_id in EXPECTED:
            package = PATCHES / patch_id
            for name in ("patch.toml", "patch.py", "SUMMARY.md", "README.md", "TESTING.md"):
                self.assertTrue((package / name).is_file(), f"{patch_id}: missing {name}")
            # A validation adapter exists only once a contract and an
            # executable producer do (1253 first, PNRO04).
            if (package / "validation.toml").exists():
                self.assertTrue((package / "validation" / "producer.py").is_file(), patch_id)

    def test_manifests_pin_state_plan_and_dependencies(self):
        for patch_id, (plan_id, requires) in EXPECTED.items():
            manifest = _manifest(patch_id)
            self.assertEqual(manifest["id"], patch_id)
            self.assertEqual(manifest["order"], int(patch_id.split("_", 1)[0]))
            self.assertEqual(manifest["state"], "untested")
            self.assertEqual(manifest["plan-ids"], plan_id if isinstance(plan_id, list) else [plan_id])
            self.assertEqual(manifest["requires"], requires)

    def test_summary_headers_match_manifest(self):
        for patch_id, (plan_id, _) in EXPECTED.items():
            text = (PATCHES / patch_id / "SUMMARY.md").read_text(encoding="utf-8")
            self.assertTrue(text.startswith(f"# {patch_id}\n\n"))
            self.assertIn("**Status:** untested", text)
            self.assertIn(f"**Plan item:** {'/'.join(plan_id) if isinstance(plan_id, list) else plan_id}", text)

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
    def test_allreduce_modes_are_opt_in_and_traced(self):
        source = _patch_source("1250_nro01_allreduce_q8_wire")
        self.assertIn("return ggml_cuda_ar_wire_mode::legacy", source)
        self.assertIn('getenv("GGML_CUDA_AR_FUSED_RESIDUAL") != nullptr', source)
        self.assertIn("BIGCHERRY_PATCH_HIT patch=1250_nro01", source)
        self.assertIn("BIGCHERRY_PATCH_HIT patch=1250_nro02", source)
        self.assertNotIn("p2p_issuer", source)

    def test_p2p_port_pushes_source_current_after_a_content_probe(self):
        source = _patch_source("1252_nro03_allreduce_p2p_provider")
        self.assertNotIn("p2p_issuer", source.split("PROVENANCE")[1])
        self.assertIn("ggml_cuda_set_device(p->devices[direction]);  // source-current push", source)
        self.assertIn("p->p2p_enabled = ggml_cuda_ar_p2p_probe(p);", source)
        self.assertIn("memcmp(readback.data(), pattern.data(), bytes) == 0", source)
        self.assertIn('ggml_cuda_ar_env_u64("GGML_CUDA_AR_P2P", 0) != 0', source)

    def test_bf16_gdn_port_is_rdna_s128_with_sequential_fallback(self):
        """PNRO04: exact port of nasone block 02's BF16 route; the fp32 chunked
        kernel is not ported, so non-BF16 shapes stay sequential."""
        source = _patch_source("1253_nro04_gfx1100_bf16_chunked_gdn")
        self.assertIn("create=True", source)
        self.assertIn("GGML_CUDA_CC_IS_RDNA3(cc_)", source)
        self.assertIn("S_v == 128", source)
        self.assertIn("patch=1253_nro04 path=gdn_chunked_bf16", source)
        self.assertNotIn("if (ggml_cuda_op_gated_delta_net_chunked(ctx, dst, state_d_ext))", source)

    def test_mtp_prefix_is_exactly_n_tokens_minus_k(self):
        source = _patch_source("1254_nro05_gdn_mtp_prefix_tail")
        self.assertIn("n_prefix = n_tokens - K", source)
        self.assertIn("n_seqs == 1", source)
        self.assertIn("K > 1", source)

    def test_adaptive_controller_constants_are_frozen_for_first_sweep(self):
        source = _patch_source("1255_nro06_adaptive_mtp_depth")
        for fragment in (
            "case 1: return 2", "case 2: return 4", "case 3: return 10",
            "case 4: return 6", "case 5: return 3", "std::max(depth * 5, 20)",
        ):
            self.assertIn(fragment, source)
        self.assertIn("Pure/testable controller only", source)

    def test_topk_ports_reproduce_the_fork_and_stay_separate(self):
        """PNRO06/07: 1256 and 1257 are port_diff-generated from the fork's
        file states (byte-exact, verified at generation); 1256 carries the
        hybrid kernels and wave64 flag, 1257 the wave32 kernels and removes
        the flag, each with its own activation markers."""
        n7 = _load_patch_module("1256_nro07_topk_hybrid")
        n8 = _load_patch_module("1257_nro08_topk_wave32")
        paths = ["ggml/src/ggml-cuda/top-k.cu", "ggml/src/ggml-hip/CMakeLists.txt"]
        self.assertEqual([p.path for p in n7.PATCHES], paths)
        self.assertEqual([p.path for p in n8.PATCHES], paths)
        hybrid = "".join(e.text for p in n7.PATCHES for e in p.edits)
        wave32 = "".join(e.text for p in n8.PATCHES for e in p.edits)
        self.assertIn("top_k_parallel_radix_cuda", hybrid)
        self.assertIn("BIGCHERRY_PATCH_HIT patch=1256_nro07 path=topk_", hybrid)
        self.assertIn("-mwavefrontsize64", hybrid)
        self.assertIn("BIGCHERRY_PATCH_HIT patch=1257_nro08 path=topk_", wave32)
        self.assertNotIn("-mwavefrontsize64", "".join(e.text for e in n8.PATCHES[1].edits))


if __name__ == "__main__":
    unittest.main()
