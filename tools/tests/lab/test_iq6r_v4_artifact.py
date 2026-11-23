"""Offline tests of IQ6R artifact admission; none require GPUs or RAD weights."""
from __future__ import annotations

import importlib.util
import unittest
from collections import Counter
from pathlib import Path

PATH = Path(__file__).resolve().parents[2] / "lab/radiance/iq6r_v4_audit_artifact.py"
spec = importlib.util.spec_from_file_location("iq6r_v4_audit_artifact", PATH)
assert spec and spec.loader
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)

MINIMAL_INFO = """
  format        RAD2, 21.02 GiB, little-endian
  architecture  qwen35
  source quant  w8a8_fwht128
  n_layers                                      64
weight directory (1032 entries, grouped by layer and movement unit)
  by encoding
      bf16                                        739      3.37 GiB
      trellis:codes=u6[1x1],scale=f32[1x*],trellis=i8{65536x1},trellis.mad=u32{2x1},scale.col=f32[*x128]/fwht128       145     5.70 GiB
      trellis:codes=u7[1x1],scale=f32[1x*],trellis=i8{65536x1},trellis.mad=u32{2x1},scale.col=f32[*x128]/fwht128       145     5.65 GiB
      trellis:codes=u5[1x1],scale=f32[1x*],trellis=i8{65536x1},trellis.mad=u32{2x1},scale.col=f32[*x128]/fwht128       1       8.3 MiB
      trellis:codes=u4[1x1],scale=f32[1x*],trellis=i8{65536x1},trellis.mad=u32{2x1},scale.col=f32[*x128]/fwht128       1       4.0 MiB
  by what made it
      gptq block=1x*,calib=/tmp/test,codes=u6       145      5.70 GiB
"""


class ArtifactAuditTests(unittest.TestCase):
    def test_parses_counts(self):
        parsed = audit.parse_radinfo(MINIMAL_INFO)
        self.assertEqual(parsed["directory_entries"], 1032)
        self.assertEqual(
            Counter(parsed["trellis_codes"]),
            Counter({"u6": 145, "u7": 145, "u5": 1, "u4": 1}),
        )

    def test_rejects_no_trellis(self):
        with self.assertRaisesRegex(ValueError, "no trellis"):
            audit.parse_radinfo(
                MINIMAL_INFO.replace("trellis:codes=", "invalid:codes=")
            )

    def test_rejects_wrong_source_quant(self):
        with self.assertRaisesRegex(ValueError, "metadata"):
            audit.parse_radinfo(MINIMAL_INFO.replace("w8a8_fwht128", "mxfp4"))

    def test_rejects_wrong_layer_count(self):
        with self.assertRaisesRegex(ValueError, "layer count"):
            audit.parse_radinfo(MINIMAL_INFO.replace("n_layers                                      64", "n_layers                                      63"))

    def test_recipe_code_counter(self):
        fake = ("# comment\n"
                "blk.0.ffn_down.weight gptq codes=u6 tcq=16\n" * 145
                + "blk.1.ffn_down.weight gptq codes=u7 tcq=16\n" * 145
                + "output.k5.weight gptq codes=u5 tcq=16\n"
                + "output.k4.weight gptq codes=u4 tcq=16\n"
                + "token_embd.weight rtn codes=u6\n")
        self.assertEqual(sum(audit.recipe_codes(fake).values()), 292)


if __name__ == "__main__":
    unittest.main()
