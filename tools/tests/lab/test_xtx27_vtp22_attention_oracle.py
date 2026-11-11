"""Card-free original BF16 full-attention CPU oracle entry and provenance tests."""
from __future__ import annotations

import io
import json
import pathlib
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

LAB = pathlib.Path(__file__).resolve().parents[2] / "lab" / "radiance"
sys.path.insert(0, str(LAB))

import xtx27_vtp22_attention_oracle as oracle
import xtx27_vtp22_attention_pack as pack


class AttentionReferenceTests(unittest.TestCase):
    def test_attn_full_checkpoint_cpu_math_source(self):
        cpp = oracle.SOURCE.read_text()
        for mark in (
            "constexpr std::array<int,3> Q{6,6,12}",
            "constexpr std::array<int,3> KV{1,1,2}",
            "constexpr int HIDDEN=5120,HEAD=256,ROPE=64",
            "normalize_rotate(r.q_gate,rows,r.qh,qnorm,HEAD*2,0)",
            "normalize_rotate(r.k,rows,r.kh,knorm,HEAD,0)",
            "const int kv=h/6",
            "for(int prev=0;prev<=t;++prev)",
            "scores[prev]=dot/16.0",
            "attended+=scores[prev]/sum*fbf16(r.v[index])",
            "sigmoid(gate)",
            "gemv(qw,input,rows,r.qh*HEAD*2,HIDDEN)",
            "r.partial[static_cast<size_t>(t)*HIDDEN+neuron]",
            "sum+=ranks[1].partial[i]",
            "sum+=ranks[2].partial[i]",
            "final[i]=bfrn(fbf16(bfrn(sum))+fbf16(residual[i]))",
        ):
            self.assertIn(mark, cpp)
        self.assertNotIn("hipMalloc", cpp)

    def test_provenance_rejects_mismatched_checkpoint_payloads(self):
        fake = {
            "layer": 3, "original_tensor_sha256": {"q": "expected"},
            "sources": {
                "q": {"path": "ignored-file", "offset": 4, "bytes": 6},
            },
        }
        with (patch.object(oracle.pack, "validate", return_value=fake),
              patch.object(oracle.projection, "sha", return_value="different"),
              self.assertRaisesRegex(ValueError, "original BF16 checkpoint payload changed")):
            oracle.validate(pathlib.Path("not-used"))

    def test_attn_reference_is_immutable_and_fail_closed(self):
        with tempfile.TemporaryDirectory() as d:
            base = pathlib.Path(d)
            with (patch.object(oracle, "validate", return_value={"rows": 1, "layer": 3}),
                  patch.object(oracle, "compile_reference",
                               side_effect=AssertionError("should refuse before compile"))):
                existing = base / "existing"
                existing.mkdir()
                with self.assertRaises(ValueError):
                    oracle.run(base / "pack", existing, base)

    def test_plan_is_card_free_and_run_is_opt_in(self):
        with tempfile.TemporaryDirectory() as tmp:
            packet = pathlib.Path(tmp) / "summary.json"
            packet.write_text("{}")
            with (redirect_stdout(io.StringIO()) as output,
                  patch.object(oracle, "validate", return_value={"rows": 4, "layer": 3}),
                  patch.object(oracle, "run", side_effect=AssertionError("must not run"))):
                self.assertEqual(
                    oracle.main(["--packet-receipt", str(packet)]), 0
                )
            data = json.loads(output.getvalue())
            self.assertFalse(data["gpu_required"])
            self.assertTrue(data["not_model_generation"])
            with patch("sys.stderr", io.StringIO()), self.assertRaises(SystemExit):
                oracle.main(["--packet-receipt", str(packet), "--phase", "cpu"])

    def test_cpu_oracle_still_requires_original_byte_exact_pack(self):
        self.assertIs(oracle.pack, pack)
        self.assertEqual(oracle.SCHEMA,
                         "vtp22-qwen38-original-bf16-full-attention-cpu-reference-v1")
        self.assertTrue(oracle.SOURCE.is_file())


if __name__ == "__main__":
    unittest.main()
