"""Mechanics tests for 1225_hi85_nccl_heterogeneous_arch_guard."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from bigcherry.patch import registry as patch_registry  # noqa: E402

_REPO = Path(__file__).resolve().parents[3]


class Patch1225Mechanics(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        registry = patch_registry.load_registry(_REPO / "patches")
        descriptor = registry.get("1225_hi85_nccl_heterogeneous_arch_guard")
        patches = patch_registry.load_implementation(descriptor, root=_REPO / "patches")
        cls.cuda_patch = next(p for p in patches if p.path == "ggml/src/ggml-cuda/ggml-cuda.cu")

    def test_admission_helper_is_compile_safe_outside_hip(self):
        edit = next(e for e in self.cuda_patch.edits if e.id == "gp02-admission-function")
        text = edit.text
        self.assertIn("#ifdef GGML_USE_HIP", text)
        self.assertIn("hipDeviceAttributeHostNativeAtomicSupported", text)
        self.assertIn("#else", text)
        self.assertIn("(void) dev_ids;", text)
        self.assertIn("(void) n_devices;", text)
        self.assertIn("return true;\n#endif", text)


if __name__ == "__main__":
    unittest.main()
