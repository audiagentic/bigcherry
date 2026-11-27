"""Card-free tests that protect the IQ6R GPU architecture selection."""
from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

PATH = Path(__file__).resolve().parents[2] / "lab/radiance/iq6r_v4_offload.py"
spec = importlib.util.spec_from_file_location("iq6r_v4_offload", PATH)
assert spec and spec.loader
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

BUNDLES = (
    "Extracting offload bundle: libquant.so.0.host-x86_64-unknown-linux-gnu-\n"
    "Extracting offload bundle: libquant.so.0.hipv4-amdgcn-amd-amdhsa--gfx1201\n"
)


class OffloadGateTest(unittest.TestCase):
    def test_parser_recovers_available_gpu_code_only(self):
        self.assertEqual(module.parse_offload_arches(BUNDLES), ["gfx1201"])
        self.assertEqual(module.parse_offload_arches("host only"), [])

    @patch.object(module.subprocess, "run")
    def test_rejects_xtx_when_quantizer_only_targets_r9700(self, run):
        run.return_value = SimpleNamespace(stdout=BUNDLES)
        with self.assertRaisesRegex(RuntimeError, "cannot run on required gfx1100"):
            module.quantizer_offload_arches(Path("libquant.so"), "gfx1100")

    @patch.object(module.subprocess, "run")
    def test_allows_matching_r9700(self, run):
        run.return_value = SimpleNamespace(stdout=BUNDLES)
        self.assertEqual(
            module.quantizer_offload_arches(Path("libquant.so"), "gfx1201"),
            ["gfx1201"],
        )


if __name__ == "__main__":
    unittest.main()
