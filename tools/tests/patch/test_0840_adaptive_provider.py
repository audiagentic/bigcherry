"""Hardware-free mechanics tests for 0840 adaptive dispatch and 1272 composition."""

from __future__ import annotations

import sys
import tempfile
import tomllib
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from bigcherry.patcher import apply_all  # noqa: E402
from bigcherry.patch import patchset  # noqa: E402
from bigcherry.patch import registry as patch_registry  # noqa: E402

_REPO = Path(__file__).resolve().parents[3]

_CUDA_SOURCE = """static bool provider_available(const std::string & p) {
    if (p == "adaptive" || p == "p2p" || p == "root3") {
        return false;
    }
    return true;
}

static void comm_init(ggml_backend_cuda_comm * ret, const std::string & provider) {
    if (provider == "ccl") {
        ggml_backend_cuda_comm_init_nccl(ret);
    } else if (provider == "host") {
        ggml_backend_cuda_comm_init_internal(ret);
    } else if (provider == "butterfly") {
        ggml_backend_cuda_comm_init_none(ret);
    } else {
        GGML_ABORT("unknown provider");
    }
}
"""


class Patch0840AdaptiveProvider(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        registry = patch_registry.load_registry(_REPO / "patches")
        descriptor = registry.get("0840_hybrid_allreduce_dispatch")
        patches = patch_registry.load_implementation(descriptor, root=_REPO / "patches")
        cls.descriptor = descriptor
        cls.patches = patches
        cls.provider_patches = tuple(
            p for p in patches
            if p.path == "ggml/src/ggml-cuda/ggml-cuda.cu"
            and any(e.id.startswith("adaptive-provider") or e.id == "adaptive-auto-default" for e in p.edits)
        )
        cls.dispatch_patch = next(
            p for p in patches
            if any(e.id == "hybrid-try-allreduce" for e in p.edits)
        )
        if len(cls.provider_patches) != 1:
            raise AssertionError(f"expected one provider FilePatch, got {len(cls.provider_patches)}")

    def _tree(self, source=_CUDA_SOURCE):
        td = tempfile.TemporaryDirectory()
        root = Path(td.name)
        path = root / "ggml/src/ggml-cuda/ggml-cuda.cu"
        path.parent.mkdir(parents=True)
        path.write_text(source, encoding="utf-8")
        return td, root, path

    def test_requires_cli_and_rccl_guard_not_qualification_telemetry(self):
        self.assertEqual(
            set(self.descriptor.requires),
            {
                "0860_allreduce_provider_cli",
                "1225_hi85_nccl_heterogeneous_arch_guard",
            },
        )
        self.assertNotIn("0830_split_reduce_telemetry", self.descriptor.requires)

    def test_owns_minimal_provider_name_seam(self):
        edit = next(e for e in self.dispatch_patch.edits if e.id == "hybrid-provider-context-field")
        self.assertIn("provider_name", edit.text)
        self.assertNotIn(
            "gp03-fix-explicit-rccl-plan-telemetry",
            {e.id for p in self.patches for e in p.edits},
        )

    def test_auto_default_is_scoped_to_dual_physical_gfx1100_hip(self):
        edit = next(e for e in self.provider_patches[0].edits if e.id == "adaptive-auto-default")
        self.assertIn("#ifdef GGML_USE_HIP", edit.text)
        self.assertIn("ret->dev_ids.size() == 2", edit.text)
        self.assertIn("ret->dev_ids[0] != ret->dev_ids[1]", edit.text)
        self.assertIn("info.device_count == info.physical_device_count", edit.text)
        self.assertEqual(edit.text.count("== GGML_CUDA_CC_RDNA3"), 2)
        self.assertIn('provider = "adaptive";', edit.text)
        self.assertIn('provider = "ccl";', edit.text)
        self.assertIn('provider = "host";', edit.text)

    def test_apply_registers_adaptive_and_is_idempotent(self):
        source = _CUDA_SOURCE.replace(
            "static void comm_init",
            "static void provider_default() {\n"
            "    std::string provider = \"auto\";\n"
            "    if (provider == \"auto\") {\n"
            "#if defined(__linux__)\n"
            "        provider = \"ccl\";\n"
            "#else\n"
            "        provider = \"host\";\n"
            "#endif\n"
            "    }\n"
            "}\n\n"
            "static void comm_init",
            1,
        )
        td, root, path = self._tree(source)
        with td:
            first = apply_all(self.provider_patches, root)
            self.assertTrue(all(r.ok for r in first), [e.detail for r in first for e in r.failed])
            text = path.read_text(encoding="utf-8")
            self.assertNotIn('p == "adaptive"', text)
            self.assertIn('if (p == "p2p" || p == "root3")', text)
            self.assertIn('provider == "adaptive"', text)
            self.assertIn("ggml_backend_cuda_comm_init_hybrid(ret);", text)
            self.assertIn("use_adaptive_default", text)
            second = apply_all(self.provider_patches, root)
            self.assertTrue(all(r.ok for r in second), [e.detail for r in second for e in r.failed])
            self.assertEqual(text, path.read_text(encoding="utf-8"))

    def test_switch_uses_0860_cli_value_not_internal_env_threshold(self):
        edit = next(e for e in self.dispatch_patch.edits if e.id == "hybrid-try-allreduce")
        self.assertIn("g_ggml_backend_cuda_comm_config.switch_bytes", edit.text)
        self.assertIn("reduction_bytes < switch_bytes", edit.text)
        self.assertNotIn("ggml_cuda_ar_pipeline_copy_threshold", edit.text)
        self.assertNotIn("GGML_CUDA_AR_COPY_THRESHOLD", edit.text)

    def test_adaptive_host_preserves_1272_wire_policy(self):
        edit = next(e for e in self.dispatch_patch.edits if e.id == "hybrid-init")
        self.assertNotIn("force_exact_f32", edit.text)
        self.assertNotIn("bf16_threshold =", edit.text)
        self.assertIn("GGML_CUDA_AR_WIRE", edit.text)
        self.assertTrue(all(p.path != "ggml/src/ggml-cuda/allreduce.cu" for p in self.patches))
        self.assertTrue(all(p.path != "ggml/src/ggml-cuda/allreduce.cuh" for p in self.patches))

    def test_adaptive_wire_recipe_is_exact_and_resolves(self):
        data = tomllib.loads((_REPO / "config/recipes.toml").read_text(encoding="utf-8"))
        expected = [
            "0830_split_reduce_telemetry",
            "0860_allreduce_provider_cli",
            "1225_hi85_nccl_heterogeneous_arch_guard",
            "0840_hybrid_allreduce_dispatch",
            "1272_ar_host_compressed_wire",
        ]
        self.assertEqual(data["experiment"]["allreduce-adaptive-wire"]["patches"], expected)
        resolved = patchset.resolve_exact(expected, directory=_REPO / "patches")
        self.assertEqual([m.patch_id for m in resolved.modules], expected)

    def test_missing_unavailable_anchor_fails_closed(self):
        source = _CUDA_SOURCE.replace('p == "root3"', 'p == "other"', 1)
        td, root, path = self._tree(source)
        with td:
            results = apply_all(self.provider_patches, root)
            self.assertFalse(all(r.ok for r in results))

    def test_missing_butterfly_branch_fails_closed(self):
        source = _CUDA_SOURCE.replace("butterfly", "bfly")
        td, root, path = self._tree(source)
        with td:
            results = apply_all(self.provider_patches, root)
            self.assertFalse(all(r.ok for r in results))


if __name__ == "__main__":
    unittest.main()
