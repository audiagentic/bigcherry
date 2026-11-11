"""VTP15 integrated original-BF16 GDN core, card-free correctness contracts."""
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
import xtx27_vtp15_gdn_bridge as bridge
import xtx27_vtp15_gdn_integrated_gpu as integrated


class OriginalGdnIntegratedTests(unittest.TestCase):
    def test_global_qkv_a_b_z_reconstruction_is_complete(self):
        local = {}
        rows = 2
        for role, nk, nv in zip(bridge.ROLES, (4, 4, 8), (12, 12, 24), strict=True):
            n = (2 * nk + nv) * 128
            local[role] = {
                "qkv": [[(i + row) / 128 for i in range(n)] for row in range(rows)],
                "a": [[(i + row) / 64 for i in range(nv)] for row in range(rows)],
                "b": [[(i + row) / 128 for i in range(nv)] for row in range(rows)],
                "z": [[(i + row) / 256 for i in range(nv * 128)] for row in range(rows)],
            }
        qkv, a, b, z = bridge._global_from_rank(local, rows)
        self.assertEqual([len(x) for x in (qkv[0], a[0], b[0], z[0])],
                         [10240, 48, 48, 6144])
        self.assertEqual(qkv[0][0], local["xtx0"]["qkv"][0][0])
        self.assertEqual(qkv[0][2048], local["xtx0"]["qkv"][0][512])
        self.assertEqual(qkv[0][1024], local["r9700"]["qkv"][0][0])
        self.assertEqual(qkv[0][3072], local["r9700"]["qkv"][0][1024])
        self.assertEqual(qkv[0][7168], local["r9700"]["qkv"][0][2048])
        self.assertEqual(a[1][24], local["r9700"]["a"][1][0])
        self.assertEqual(b[1][24], local["r9700"]["b"][1][0])
        self.assertEqual(z[1][3072], local["r9700"]["z"][1][0])
        local["r9700"]["qkv"][0].pop()
        with self.assertRaises(ValueError):
            bridge._global_from_rank(local, rows)

    def test_gpu_is_opt_in_and_requires_all_validated_sources(self):
        with (patch.object(integrated.subprocess, "run",
                           side_effect=AssertionError("external process")),
              redirect_stdout(io.StringIO()) as out):
            self.assertEqual(integrated.main([]), 0)
        doc = json.loads(out.getvalue())
        self.assertTrue(doc["not_full_model"])
        self.assertTrue(doc["not_optimized_gemm"])
        self.assertIsNone(doc["matched_sources"])
        for args in (
            ["--phase", "gpu"],
            ["--phase", "gpu", "--run"],
            ["--phase", "compile"],
            ["--iterations", "31"],
        ):
            with patch("sys.stderr", io.StringIO()), self.assertRaises(SystemExit):
                integrated.main(args)

    def test_integrated_hip_chain_is_simultaneous_and_owns_true_gdn(self):
        s = integrated.SOURCE.read_text()
        for token in (
            "constexpr std::array<int,3> KEYS{4,4,8}",
            "constexpr std::array<int,3> VALUES{12,12,24}",
            "project_original_bf16<<<",
            "r.projections[0].output,r.w,r.conv",
            "delta_qwen<<<r.values,H",
            "r.projections[1].output,r.projections[2].output",
            "r.projections[3].output,r.coef",
            "conv4<<<", "qk_norm<<<",
            "std::array<Projection,4>",
            "state[stbase+d]*dec",
            "beta*(v-predicted)",
            "original BF16 Qwen projection-to-GDN GPU core oracle mismatch",
            "for(int i=0;i<3;++i)launch(ranks[i]);",
        ):
            self.assertIn(token, s)
        self.assertNotIn("hipMemcpyPeer", s)
        self.assertNotIn("hipDeviceEnablePeerAccess", s)
        self.assertIn("output_projection=NO", s)

    def test_integrated_reference_rejects_malformed_chain(self):
        with tempfile.TemporaryDirectory() as temp:
            d = pathlib.Path(temp)
            w, r, core, learned = [d / f"input-{i}.json" for i in range(4)]
            for f in (w, r, core, learned):
                f.write_text("{}")
            with self.assertRaises((ValueError, KeyError)):
                integrated.validate(w, r, core, learned)

    def test_cpu_bridge_is_no_gpu_and_refuses_implicit_writes(self):
        with redirect_stdout(io.StringIO()) as out:
            self.assertEqual(bridge.main([
                "--weights-receipt", "/not-called/weights.json",
                "--projected-receipt", "/not-called/references.json",
                "--qkv-pack-receipt", "/not-called/pack.json",
            ]), 0)
        doc = json.loads(out.getvalue())
        self.assertEqual(doc["status"], "CARD_FREE_PLAN")
        self.assertFalse(doc["gpu_required"])
        with patch("sys.stderr", io.StringIO()), self.assertRaises(SystemExit):
            bridge.main([
                "--phase", "verify",
                "--weights-receipt", "/not-called/weights.json",
                "--projected-receipt", "/not-called/references.json",
                "--qkv-pack-receipt", "/not-called/pack.json",
            ])


if __name__ == "__main__":
    unittest.main()
