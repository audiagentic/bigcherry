"""Card-free guard and idempotence test for rad0062."""
import runpy
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "tools"))
from bigcherry.patcher import apply_patch

PKG = ROOT / "engines/radiance/patches/rad0062_iq6r_converter_gpu_arch"
PATCH = runpy.run_path(str(PKG / "patch.py"))["PATCHES"][0]
BASE = """if(LQ_HIP_SRC)
  target_compile_definitions(libquant PRIVATE LQ_CARD=1)
endif()
"""

class QuantizerArchitecturePatchTest(unittest.TestCase):
    def test_idempotence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / PATCH.path
            target.parent.mkdir(parents=True)
            target.write_text(BASE)
            result = apply_patch(PATCH, root)
            self.assertTrue(result.ok)
            self.assertTrue(result.changed)
            first = target.read_text()
            self.assertIn("RAD_QUANT_GPU_TARGETS", first)
            self.assertIn("HIP_ARCHITECTURES", first)
            result = apply_patch(PATCH, root)
            self.assertTrue(result.ok)
            self.assertFalse(result.changed)
            self.assertEqual(target.read_text(), first)

    def test_missing_anchor(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / PATCH.path
            target.parent.mkdir(parents=True)
            target.write_text("missing anchor\n")
            self.assertFalse(apply_patch(PATCH, root).ok)

if __name__ == "__main__":
    unittest.main()
