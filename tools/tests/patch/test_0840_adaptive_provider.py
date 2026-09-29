"""Hardware-free mechanics tests for 0840's adaptive provider registration on the 0860 seam."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from bigcherry.patcher import apply_all  # noqa: E402
from bigcherry.patch import registry as patch_registry  # noqa: E402

_REPO = Path(__file__).resolve().parents[3]

_CUDA_SOURCE = '''static bool provider_available(const std::string & p) {
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
'''


class Patch0840AdaptiveProvider(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        registry = patch_registry.load_registry(_REPO / "patches")
        descriptor = registry.get("0840_hybrid_allreduce_dispatch")
        patches = patch_registry.load_implementation(descriptor, root=_REPO / "patches")
        cls.descriptor = descriptor
        cls.provider_patches = tuple(
            p for p in patches
            if p.path == "ggml/src/ggml-cuda/ggml-cuda.cu"
            and any(e.id.startswith("adaptive-provider") for e in p.edits)
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

    def test_requires_0860(self):
        self.assertIn("0860_allreduce_provider_cli", self.descriptor.requires)

    def test_apply_registers_adaptive_and_is_idempotent(self):
        td, root, path = self._tree()
        with td:
            first = apply_all(self.provider_patches, root)
            self.assertTrue(all(r.ok for r in first), [e.detail for r in first for e in r.failed])
            text = path.read_text(encoding="utf-8")
            self.assertNotIn('p == "adaptive"', text)
            self.assertIn('if (p == "p2p" || p == "root3")', text)
            self.assertIn('provider == "adaptive"', text)
            self.assertIn("ggml_backend_cuda_comm_init_hybrid(ret);", text)
            second = apply_all(self.provider_patches, root)
            self.assertTrue(all(r.ok for r in second), [e.detail for r in second for e in r.failed])
            self.assertEqual(text, path.read_text(encoding="utf-8"))

    def test_missing_unavailable_anchor_fails_closed(self):
        td, root, path = self._tree(_CUDA_SOURCE.replace('p == "root3"', 'p == "other"', 1))
        with td:
            results = apply_all(self.provider_patches, root)
            self.assertFalse(all(r.ok for r in results))

    def test_missing_butterfly_branch_fails_closed(self):
        td, root, path = self._tree(_CUDA_SOURCE.replace("butterfly", "bfly"))
        with td:
            results = apply_all(self.provider_patches, root)
            self.assertFalse(all(r.ok for r in results))


if __name__ == "__main__":
    unittest.main()
