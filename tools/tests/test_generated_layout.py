"""Out-of-tree catalog generation contract (BC06)."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bigcherry import autotune_catalog as catalog  # noqa: E402
from bigcherry import paths  # noqa: E402


class GeneratedLayoutTests(unittest.TestCase):
    def test_generated_root_receives_compile_inputs_not_source_tree(self):
        source = paths.llama_root()
        generated_source = paths.cuda_dir(source) / "hip-autotune-registry.inc"
        before = generated_source.read_bytes() if generated_source.exists() else None
        manifest = catalog.build_manifest(
            source,
            variant_set="inventory",
            architectures=["gfx1100"],
            inventory=None,
            source_revision="bc06-test",
        )
        with tempfile.TemporaryDirectory() as directory:
            generated = Path(directory) / "generated"
            artifacts = Path(directory) / "artifacts"
            result = catalog.emit(
                manifest, source, artifacts, generated_root=generated
            )
            self.assertTrue(result.registry_path.is_relative_to(generated))
            self.assertTrue((generated / "hip-autotune-build-hash.h").is_file())
            self.assertTrue((generated / "template-instances").is_dir())
        after = generated_source.read_bytes() if generated_source.exists() else None
        self.assertEqual(after, before)


if __name__ == "__main__":
    unittest.main()
