"""VTP23 card-free original BF16 full-attention TP3 GPU acceptance tests."""
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

import xtx27_vtp23_attention_gpu as gpu


class OriginalAttentionTP3GpuTests(unittest.TestCase):
    def test_canonical_original_attention_geometry(self):
        self.assertTrue(gpu.SOURCE.is_file())
        text = gpu.SOURCE.read_text()
        for token in (
            "constexpr std::array<int,3> Q{6,6,12}",
            "constexpr std::array<int,3> KV{1,1,2}",
            "constexpr int N=5120,HD=256,ROT=64",
            "head/6",
            "original_bf16_gemv<<<",
            "zero_centered_rms<<<",
            "text_rope64<<<",
            "original_causal_gated_gqa<<<",
            "original_output_col_gemv<<<",
            "sigmoid(gate)",
            "for(int prev=0;prev<=token;++prev)",
            "products[d]=bf16float(q[qb+d])*bf16float(k[kb+d])",
            "gated[(static_cast<size_t>(token)*qheads+head)*HD+d]",
            "for(auto& r:rank)launch(r);",
            "sum+=rank[1].pinned_partial[i]",
            "sum+=rank[2].pinned_partial[i]",
            "pinned_final[i]=hostrn(hostbf(hostrn(sum))+hostbf(residual[i]))",
            "hipHostMallocPortable",
            "VTP23 full attention BF16 residual exceeds CPU oracle",
        ):
            self.assertIn(token, text)
        self.assertNotIn("hipMemcpyPeer", text)
        self.assertNotIn("hipDeviceEnablePeerAccess", text)

    def test_cardfree_default_does_not_run_tools(self):
        with (patch.object(gpu.subprocess, "run",
                           side_effect=AssertionError("no system commands")),
              redirect_stdout(io.StringIO()) as output):
            self.assertEqual(gpu.main([]), 0)
        data = json.loads(output.getvalue())
        self.assertEqual(data["schema"], gpu.SCHEMA)
        self.assertEqual(data["phase"], "card_free")
        self.assertFalse(data["full_model_generation"])
        self.assertFalse(data["quality_qualified"])
        self.assertIsNone(data["source_receipts"])

    def test_opt_in_and_checkpoint_receipt_requirement(self):
        for flags in (
            ["--phase", "gpu"],
            ["--phase", "compile"],
            ["--phase", "gpu", "--run"],
            ["--iterations", "0"],
            ["--iterations", "31"],
        ):
            with patch("sys.stderr", io.StringIO()), self.assertRaises(SystemExit):
                gpu.main(flags)

    def test_invalid_cpu_oracle_rejected_before_hardware(self):
        with tempfile.TemporaryDirectory() as temp:
            receipt = pathlib.Path(temp) / "incomplete.json"
            receipt.write_text(json.dumps({"schema": gpu.oracle.SCHEMA, "status": "INCOMPLETE"}))
            with (patch.object(gpu.oracle, "validate",
                               return_value={"rows": 1, "layer": 3}),
                  self.assertRaises(ValueError)):
                gpu.validate(receipt, receipt)

    def test_gpu_source_digests_are_distinct_and_input_is_original(self):
        self.assertEqual(set(gpu.SOURCES), {"original_attention"})
        self.assertTrue(gpu.RECEIPT_TYPE.startswith("VTP23_"))
        self.assertTrue(gpu.PURPOSE.startswith("VTP23:"))
        self.assertIsNot(gpu.SOURCE, gpu.oracle.SOURCE)
        self.assertIn("exact original Qwen3.8-27B checkpoint", gpu.__doc__)
        self.assertIn("No full model inference", gpu.__doc__)


if __name__ == "__main__":
    unittest.main()
