"""Card-free VTP18 BF16 output GEMV row-fusion admission and A/B contracts."""
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

import xtx27_vtp17_gdn_output_pinned_gpu as vtp17
import xtx27_vtp18_gdn_output_rowreuse_gpu as vtp18


class RowReuseOptimisationTests(unittest.TestCase):
    def test_same_source_receipt_schema_and_oracle_as_pinned_baseline(self):
        self.assertIs(vtp18.oracle, vtp17.oracle)
        self.assertEqual(vtp18.validate.__code__.co_code, vtp17.validate.__code__.co_code)
        self.assertNotEqual(vtp18.SOURCE, vtp17.SOURCE)
        self.assertNotEqual(vtp18.LEASE_PURPOSE, vtp17.LEASE_PURPOSE)
        self.assertNotEqual(vtp18.RECEIPT_TYPE, vtp17.RECEIPT_TYPE)

    def test_default_plan_never_claims_execution(self):
        with (redirect_stdout(io.StringIO()) as out,
              patch.object(vtp18.subprocess, "run",
                           side_effect=AssertionError("must not execute"))):
            self.assertEqual(vtp18.main([]), 0)
        p = json.loads(out.getvalue())
        self.assertEqual(p["phase"], "card_free")
        self.assertTrue(p["not_full_model"])
        self.assertTrue(p["not_integrated_with_gdn_core"])
        self.assertIsNone(p["sources"])

    def test_gpu_phase_requires_explicit_passing_receipts_and_lease(self):
        for args in (
            ["--phase", "gpu"], ["--phase", "gpu", "--run"],
            ["--phase", "compile"], ["--iterations", "0"],
            ["--iterations", "31"],
        ):
            with patch("sys.stderr", io.StringIO()), self.assertRaises(SystemExit):
                vtp18.main(args)

    def test_statically_specialised_batch_ownership_and_exact_fallback(self):
        src = vtp18.SOURCE.read_text()
        for token in (
            "template<int M>",
            "__global__ void output_col_gemv_reuse",
            "float acc[M]{}",
            "const float value=bf16value(w[k])",
            "for(int row=0;row<M;++row)",
            "acc[row]+=value*bf16value(x[static_cast<size_t>(row)*width+k])",
            "output_col_gemv_reuse<2><<<",
            "output_col_gemv_reuse<4><<<",
            "output_col_gemv_reuse<8><<<",
            "else",
            "output_col_gemv<<<",
        ):
            self.assertIn(token, src)
        self.assertLess(src.index("if(r.rows==2)"), src.index("else if(r.rows==4)"))
        self.assertLess(src.index("else if(r.rows==4)"), src.index("else if(r.rows==8)"))
        self.assertNotIn("hipMemcpyPeer", src)
        self.assertNotIn("hipDeviceEnablePeerAccess", src)
        self.assertIn("WIDTHS{1536,1536,3072}", src)

    def test_pinned_transport_and_output_numeric_guard_remain_unchanged(self):
        src = vtp18.SOURCE.read_text()
        for token in (
            "hipHostMallocPortable",
            "hipHostFree(r.pinned_partial)",
            "hipHostFree(host_final)",
            "hipMemcpyAsync(r.pinned_partial,r.partial",
            "v+=ranks[1].pinned_partial[j]",
            "v+=ranks[2].pinned_partial[j]",
            "host_final[j]=roundbf16(f32(roundbf16(v))+f32(resid[j]))",
            "rank did not receive identical BF16 host broadcast",
            "original BF16 physical output shard numeric oracle failed",
            "max_partial_error,max_reduce_error,max_final_error",
        ):
            self.assertIn(token, src)
        self.assertIn("original_BF16_out_proj_row_reuse_pinned_async_three_GPU", src)


if __name__ == "__main__":
    unittest.main()
