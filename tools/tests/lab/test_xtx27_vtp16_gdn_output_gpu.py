"""Card-free VTP16 original GDN output GPU staged-collective tests."""
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

import xtx27_vtp16_gdn_output_gpu as gpu


class OriginalOutputGpuAdmissionTests(unittest.TestCase):
    def test_plan_is_card_free_and_marks_nonintegrated_path(self):
        with (patch.object(gpu.subprocess, "run",
                           side_effect=AssertionError("must not start a process")),
              redirect_stdout(io.StringIO()) as output):
            self.assertEqual(gpu.main([]), 0)
        doc = json.loads(output.getvalue())
        self.assertEqual(doc["phase"], "card_free")
        self.assertTrue(doc["not_full_model"])
        self.assertTrue(doc["not_integrated_with_gdn_core"])
        self.assertIsNone(doc["sources"])

    def test_phase_cannot_start_without_explicit_permission_or_reference(self):
        for args in (
            ["--phase", "gpu"],
            ["--phase", "gpu", "--run"],
            ["--phase", "compile"],
            ["--iterations", "31"],
        ):
            with patch("sys.stderr", io.StringIO()), self.assertRaises(SystemExit):
                gpu.main(args)

    def test_reference_refusal_precedes_gpu_admission(self):
        with tempfile.TemporaryDirectory() as temp:
            malformed = pathlib.Path(temp) / "invalid.json"
            malformed.write_text(json.dumps({"schema": "wrong", "status": "not_complete"}))
            with self.assertRaises(ValueError):
                gpu.validate(malformed, malformed)

    def test_three_physical_output_collective_and_broadcast_in_source(self):
        source = gpu.SOURCE.read_text()
        for token in (
            "constexpr std::array<int,3> WIDTHS{1536,1536,3072}",
            "const uint16_t* x,const uint16_t* weight",
            "output_col_gemv<<<",
            "for(int k=lane;k<width;k+=wave)",
            "for(auto& r:ranks)launch(r);",
            "hipMemcpyDeviceToHost",
            "v+=ranks[1].cpu_partial[j]",
            "v+=ranks[2].cpu_partial[j]",
            "roundbf16(f32(roundbf16(v))+f32(resid[j]))",
            "hipMemcpyHostToDevice",
            "rank did not receive identical BF16 host broadcast",
            "original BF16 physical output shard numeric oracle failed",
        ):
            self.assertIn(token, source)
        self.assertNotIn("hipMemcpyPeer", source)
        self.assertNotIn("hipDeviceEnablePeerAccess", source)
        self.assertIn("if(prop.warpSize!=32)", source)

    def test_code_reflects_distinct_stage_not_complete_model(self):
        self.assertTrue(gpu.SOURCE.is_file())
        self.assertEqual(len(gpu.SOURCES), 1)
        self.assertNotEqual(gpu.LEASE_PURPOSE, "")
        self.assertIn("no-P2P", gpu.__doc__)
        self.assertIn("NOT yet joined", gpu.__doc__)
        self.assertTrue(gpu.RECEIPT_TYPE.startswith("VTP16_"))


if __name__ == "__main__":
    unittest.main()
