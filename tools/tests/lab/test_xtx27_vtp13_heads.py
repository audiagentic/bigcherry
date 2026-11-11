"""No-GPU admission and operator contract tests for VTP13 head-local proof."""
from __future__ import annotations

import io
import json
import os
import pathlib
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from types import SimpleNamespace
from unittest.mock import patch

LAB = pathlib.Path(__file__).resolve().parents[2] / "lab" / "radiance"
sys.path.insert(0, str(LAB))
import xtx27_vtp13_heads as heads


class HeadLocalTests(unittest.TestCase):
    def test_default_plan_is_card_free_and_explicitly_gdn_limited(self):
        with (patch.object(heads.subprocess, "run", side_effect=AssertionError("external call")),
              redirect_stdout(io.StringIO()) as output):
            self.assertEqual(heads.main([]), 0)
        data = json.loads(output.getvalue())
        self.assertTrue(data["gdn_is_simplified"])
        self.assertFalse(data["checkpoint_payloads"])
        self.assertEqual(len(data["cases"]), 4)
        self.assertEqual(len([c for c in data["cases"] if c["label"].startswith("attention")]), 2)
        self.assertEqual(len([c for c in data["cases"] if c["label"].startswith("gdn")]), 2)

    def test_shapes_reversals_boundaries(self):
        specs = heads.cases(("attention", "gdn"), (1, 4, 8, 64), 3, 256, reverse=True)
        self.assertEqual(len(specs), 8)
        self.assertTrue(all("--reverse" in x["args"] for x in specs))
        self.assertTrue(all(x["kind"] == "head_local" for x in specs))
        self.assertTrue(any("12Q/2KV" in c["data_definition"] for c in specs))
        self.assertTrue(any("8K/24V" in c["data_definition"] for c in specs))
        for op, rows, iters, context in [
            (("bad",), (1,), 3, 256),
            (("attention",), (65,), 3, 256),
            (("gdn",), (1,), 0, 256),
            (("gdn",), (1,), 3, 2049),
        ]:
            with self.assertRaises(ValueError):
                heads.cases(op, rows, iters, context)

    def test_gpu_and_compile_need_opt_in(self):
        for phase in ("compile", "gpu"):
            with patch("sys.stderr", io.StringIO()), self.assertRaises(SystemExit):
                heads.main(["--phase", phase])
        with patch("sys.stderr", io.StringIO()), self.assertRaises(SystemExit):
            heads.main(["--phase", "compile", "--run", "--scratch", str(heads.ROOT)])

    def test_lease_rejects_wrong_owner_before_launch(self):
        with (patch.dict(os.environ, {"VTP04_LEASED": "1"}),
              patch.object(heads.subprocess, "run", return_value=SimpleNamespace(
                  stdout='gpu held by someone for "wrong owner"')) as calls,
              tempfile.TemporaryDirectory() as temp):
            with self.assertRaisesRegex(RuntimeError, "not owned"):
                heads.micro.run_leased(
                    scratch=pathlib.Path(temp), binaries={}, cases=[],
                    digest="1234", purpose=heads.LEASE_PURPOSE,
                    receipt_type=heads.RECEIPT_TYPE,
                )
            self.assertEqual(calls.call_count, 1)

    def test_hip_source_has_real_head_concurrency_and_oracles(self):
        source = heads.SOURCE.read_text()
        for symbol in (
            "constexpr std::array<int,3> QHEADS{6,6,12}",
            "constexpr std::array<int,3> KVHEADS{1,1,2}",
            "constexpr std::array<int,3> GDN_KEYS{4,4,8}",
            "constexpr std::array<int,3> GDN_VALUES{12,12,24}",
            "full_attention<<<dim3(r.q_heads,rows),ATTN_DIM",
            "gdn_update<<<r.gdn_values,GDN_DIM",
            "const int kv=head/6",
            "const int key=h/3",
            "hipMemsetAsync(r.state,0",
            "cpu_attention(r,rows,context)",
            "cpu_gdn(r,rows)",
            "R9700 two-head attention/GDN independent CPU oracle mismatch",
            "gfx1100", "gfx1201",
        ):
            self.assertIn(symbol, source)
        self.assertNotIn("hipDeviceEnablePeerAccess", source)
        self.assertNotIn("hipMemcpyPeer", source)
        self.assertNotIn("full_qwen_gdn=YES", source)


if __name__ == "__main__":
    unittest.main()
