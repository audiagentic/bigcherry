"""Card-free contract tests for the VTP13 concurrent FFN / TP3 proof."""
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
import xtx27_vtp13_chain as chain


class ConcurrentInt8FfnTests(unittest.TestCase):
    def test_plan_requires_no_external_process(self):
        with (patch.object(chain.subprocess, "run", side_effect=AssertionError("subprocess")),
              redirect_stdout(io.StringIO()) as out):
            self.assertEqual(chain.main([]), 0)
        manifest = json.loads(out.getvalue())
        self.assertEqual(manifest["schema"], "vtp13-concurrent-ffn-proof-v1")
        self.assertEqual([x["args"][1] for x in manifest["cases"]], ["1", "4"])
        self.assertFalse(manifest["checkpoint_payload"])
        self.assertTrue(manifest["not_native_radiance_gemm"])
        self.assertEqual(manifest["precision"], "int8-int8-int32-f32, uniform three-card")

    def test_one_case_per_shape_and_safe_reverse(self):
        specs = chain.cases((1, 8, 64), 5, reverse=True)
        self.assertEqual(len(specs), 3)
        self.assertEqual(len({s["label"] for s in specs}), 3)
        self.assertTrue(all(s["args"][-1] == "--reverse" for s in specs))
        self.assertTrue(all(s["kind"] == "ffn_chain" for s in specs))
        self.assertTrue(all("three" not in s["label"] for s in specs))
        self.assertIn("5120/4864/7424", specs[0]["data_definition"])
        for iters in (0, 31):
            with self.assertRaises(ValueError):
                chain.cases((1,), iters)
        with self.assertRaises(ValueError):
            chain.cases((65,), 5)

    def test_fail_closed_without_explicit_gpu_or_compile_permission(self):
        for phase in ("compile", "gpu"):
            with (patch("sys.stderr", io.StringIO()),
                  patch.object(chain.subprocess, "run", side_effect=AssertionError("process"))):
                with self.assertRaises(SystemExit) as caught:
                    chain.main(["--phase", phase])
                self.assertEqual(caught.exception.code, 2)
        for bad in ("0", "65", "4,4", "1,", "1;2"):
            with patch("sys.stderr", io.StringIO()), self.assertRaises(SystemExit):
                chain.main(["--rows", bad])
        with patch("sys.stderr", io.StringIO()), self.assertRaises(SystemExit):
            chain.main(["--phase", "compile", "--run", "--scratch", str(chain.ROOT)])

    def test_digest_isolated_from_existing_vtp04_binary(self):
        before = chain.micro.source_digest()
        with tempfile.TemporaryDirectory() as tmp:
            original = chain.SOURCE.read_bytes()
            p = pathlib.Path(tmp) / "ffn.hip"
            p.write_bytes(original)
            self.assertEqual(chain.micro.source_digest({"ffn_chain": p}),
                             chain.micro.source_digest(chain.SOURCES))
            p.write_bytes(original + b"\n//new")
            self.assertNotEqual(chain.micro.source_digest({"ffn_chain": p}),
                                chain.micro.source_digest(chain.SOURCES))
        self.assertEqual(chain.micro.source_digest(), before)

    def test_checkpoint_original_bf16_view_is_explicit(self):
        checkpoint = {
            "layer": 0, "tensors": {
                name: {"file": f"/checkpoint/{name}.safetensors", "offset": j * 1000}
                for j, name in enumerate(("gate", "up", "down"))
            },
        }
        cases = chain.cases((1, 4), 3, reverse=True, checkpoint=checkpoint)
        self.assertEqual(len(cases), 2)
        self.assertTrue(all("checkpoint-bf16-layer0" in case["label"] for case in cases))
        self.assertTrue(all(case["args"][-1] == "--reverse" for case in cases))
        self.assertEqual(cases[0]["args"].count("--gate-file"), 1)
        self.assertEqual(cases[0]["args"].count("--down-byte"), 1)
        self.assertIn("BF16-weight reference", cases[0]["data_definition"])

    def test_checkpoint_kernel_has_original_bf16_and_per_channel_scales(self):
        source = chain.SOURCE.read_text()
        for token in (
            "read_checkpoint_args(argc,argv)",
            "load_checkpoint_weights(d,checkpoint)",
            "load_bf16_projection(",
            "bf16_to_float(",
            "dev.hgate_scales[j]", "dev.hup_scales[j]", "dev.hdown_scales[j]",
            "w_scales[output]*x_scales[row]",
            "checkpoint_bf16_weights_int8_tp3",
        ):
            self.assertIn(token, source)

    def test_gpu_runner_owns_distinct_gpu_lease_and_refuses_other_owner(self):
        with (patch.dict(os.environ, {"VTP04_LEASED": "1"}),
              patch.object(chain.subprocess, "run", return_value=SimpleNamespace(
                  stdout='gpu: held by someone for "unrelated"')) as calls,
              tempfile.TemporaryDirectory() as scratch):
            with self.assertRaisesRegex(RuntimeError, "not owned"):
                chain.micro.run_leased(
                    scratch=pathlib.Path(scratch), binaries={},
                    cases=[], digest="abcd", purpose=chain.LEASE_PURPOSE,
                    receipt_type=chain.RECEIPT_TYPE,
                )
            self.assertEqual(calls.call_count, 1)
            self.assertFalse((pathlib.Path(scratch) / "runs").exists())

    def test_actual_int8_chain_and_independent_three_device_oracle_are_present(self):
        source = chain.SOURCE.read_text()
        for token in (
            "constexpr std::array<int,3> WIDTHS{5120,4864,7424}",
            "bf16_source_weight",
            "bf16_source_input",
            "acc+=static_cast<int32_t>(wr[col])*static_cast<int32_t>(xr[col])",
            "silu_mul<<<", "quant_mid<<<", "dot_i8<<<",
            "launch(dev[reverse?2-t:t],rows)",
            "hipEventSynchronize(d.ready)",
            "hipMemcpyDeviceToHost",
            "hipMemcpyHostToDevice",
            "cpu_reference(dev,input,rows,true)",
            "cpu_reference(dev,input,rows,false)",
            "BF16-derived independent INT8 FFN CPU oracle mismatch",
            "rank-divergent physical TP3 broadcast",
            "std::ceil(p*v.size())",
            "gfx1100", "gfx1201",
        ):
            self.assertIn(token, source)
        self.assertNotIn("hipDeviceEnablePeerAccess", source)
        self.assertNotIn("hipMemcpyPeer", source)


if __name__ == "__main__":
    unittest.main()
