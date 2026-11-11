"""Card-free VTP17 pinned async host transport and numerical contract tests."""
from __future__ import annotations

import io
import json
import pathlib
import sys
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

LAB = pathlib.Path(__file__).resolve().parents[2] / "lab" / "radiance"
sys.path.insert(0, str(LAB))

import xtx27_vtp16_gdn_output_gpu as vtp16
import xtx27_vtp17_gdn_output_pinned_gpu as vtp17


class PinnedTransportOptimisationTests(unittest.TestCase):
    def test_same_input_reference_and_model_dimensions_as_baseline(self):
        self.assertIs(vtp17.oracle, vtp16.oracle)
        self.assertEqual(vtp17.validate.__code__.co_code, vtp16.validate.__code__.co_code)
        self.assertIn("pinned", vtp17.SCRATCH.as_posix())
        self.assertNotEqual(vtp17.SOURCE, vtp16.SOURCE)
        self.assertNotEqual(vtp17.LEASE_PURPOSE, vtp16.LEASE_PURPOSE)
        self.assertNotEqual(vtp17.RECEIPT_TYPE, vtp16.RECEIPT_TYPE)

    def test_no_gpu_execution_by_default(self):
        with (redirect_stdout(io.StringIO()) as out,
              patch.object(vtp17.subprocess, "run",
                           side_effect=AssertionError("would execute external process"))):
            self.assertEqual(vtp17.main([]), 0)
        doc = json.loads(out.getvalue())
        self.assertEqual(doc["phase"], "card_free")
        self.assertTrue(doc["not_full_model"])
        self.assertTrue(doc["not_integrated_with_gdn_core"])
        self.assertIsNone(doc["sources"])

    def test_no_unleased_run_or_missing_original_weight_oracles(self):
        for flags in (
            ["--phase", "gpu"], ["--phase", "compile"],
            ["--phase", "gpu", "--run"], ["--iterations", "31"],
        ):
            with patch("sys.stderr", io.StringIO()), self.assertRaises(SystemExit):
                vtp17.main(flags)

    def test_pinned_async_buffers_and_transfer_batching(self):
        s = vtp17.SOURCE.read_text()
        self.assertIn("hipHostMallocPortable", s)
        self.assertIn("hipHostMalloc(reinterpret_cast<void**>(&r.pinned_partial)", s)
        self.assertIn("hipHostMalloc(reinterpret_cast<void**>(&host_final)", s)
        self.assertIn("hipHostFree(r.pinned_partial)", s)
        self.assertIn("hipHostFree(host_final)", s)
        self.assertIn("hipMemcpyAsync(r.pinned_partial,r.partial", s)
        self.assertIn("hipMemcpyDeviceToHost,r.stream", s)
        self.assertIn("hipMemcpyHostToDevice,r.stream", s)
        self.assertIn("for(auto& r:ranks)launch(r);", s)
        self.assertIn("v+=ranks[1].pinned_partial[j]", s)
        self.assertIn("v+=ranks[2].pinned_partial[j]", s)
        self.assertIn("host_final[j]=roundbf16(f32(roundbf16(v))+f32(resid[j]))", s)
        self.assertIn("rank did not receive identical BF16 host broadcast", s)
        self.assertIn("max_partial_error,max_reduce_error,max_final_error", s)
        self.assertIn("host_reduce_us_p50", s)
        self.assertIn("d2h_stage_us_p50", s)
        self.assertIn("h2d_stage_us_p50", s)
        self.assertNotIn("hipMemcpyPeer", s)
        self.assertNotIn("hipDeviceEnablePeerAccess", s)
        self.assertIn("WIDTHS{1536,1536,3072}", s)

    def test_source_order_is_kernel_then_async_d2h_then_host_reduce(self):
        s = vtp17.SOURCE.read_text()
        a = s.index("for(auto& r:ranks)launch(r);")
        b = s.index("HC(hipMemcpyAsync(r.pinned_partial,r.partial", a)
        c = s.index("const auto after_d2h", b)
        d = s.index("float v=ranks[0].pinned_partial[j]", c)
        e = s.index("HC(hipMemcpyAsync(r.broadcast,host_final", d)
        f = s.index("HC(hipStreamSynchronize(r.stream))", e)
        self.assertLess(a, b)
        self.assertLess(b, c)
        self.assertLess(c, d)
        self.assertLess(d, e)
        self.assertLess(e, f)


if __name__ == "__main__":
    unittest.main()
