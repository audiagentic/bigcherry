"""Card-free VTP19 single-lease A/B/C GPU timing and provenance tests."""
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

import xtx27_vtp19_output_abccba as ab


class BalancedOriginalOutputTimingTests(unittest.TestCase):
    def fixture(self, m=4):
        return {
            "rows": m,
            "packet_directory": "/not-used/packet",
            "reference_directory": "/not-used/reference",
            "residual_file": "/not-used/residual",
        }

    def test_palindrome_has_two_arms_per_variant_and_unique_labels(self):
        self.assertEqual(ab.ORDER, (
            "pageable", "pinned", "rowreuse", "rowreuse", "pinned", "pageable",
        ))
        cases = ab.plan([self.fixture(1), self.fixture(4)], 9)
        self.assertEqual(len(cases), 12)
        self.assertEqual(len({c["label"] for c in cases}), 12)
        self.assertEqual([c["kind"] for c in cases[:6]], list(ab.ORDER))
        self.assertTrue(all(c["args"][3] == "9" for c in cases))
        self.assertTrue(all(c["data_definition"].startswith("VTP19") for c in cases))
        with self.assertRaises(ValueError):
            ab.plan([self.fixture()], 0)
        with self.assertRaises(ValueError):
            ab.plan([self.fixture()], 31)

    def test_abccba_ingests_canonical_oracles_and_computes_median(self):
        cases = ab.plan([self.fixture(4)], 7)
        medians = {
            "pageable": (300., 306.),
            "pinned": (250., 252.),
            "rowreuse": (200., 208.),
        }
        with tempfile.TemporaryDirectory() as temp:
            base = pathlib.Path(temp)
            receipt = {"cases": []}
            seen = {"pageable": 0, "pinned": 0, "rowreuse": 0}
            for case in cases:
                variant = case["kind"]
                index = seen[variant]
                seen[variant] += 1
                wall = medians[variant][index]
                content = (
                    "mode,rows,wall_us_p50,wall_us_p95,xtx0_kernel_us_p50,"
                    "xtx1_kernel_us_p50,r9700_kernel_us_p50,max_partial_abs,"
                    "max_reduce_abs,max_final_bf16_abs,final_bf16_exact,total,oracle\n"
                    f"original_BF16_out_proj,4,{wall},{wall+9},155,154,102,"
                    "0.000001,0.000001,0,20480,20480,PASS\n"
                )
                name = case["label"] + ".stdout"
                (base / name).write_text(content)
                receipt["cases"].append({
                    "label": case["label"], "kind": variant, "rc": 0,
                    "oracle_pass": True, "stdout": name,
                })
            summary = ab.extract_rows(receipt, base, cases)
            self.assertEqual(set(summary), {"4"})
            stats = summary["4"]["variants"]
            self.assertEqual(stats["pageable"]["wall_p50_median_of_arms_us"], 303.)
            self.assertEqual(stats["pinned"]["wall_p50_median_of_arms_us"], 251.)
            self.assertEqual(stats["rowreuse"]["wall_p50_median_of_arms_us"], 204.)
            self.assertAlmostEqual(stats["pinned"]["indicative_reduction_percent_vs_pageable"],
                                   17.16, places=2)
            self.assertAlmostEqual(stats["rowreuse"]["indicative_reduction_percent_vs_pinned"],
                                   18.73, places=2)
            receipt["cases"][3]["oracle_pass"] = False
            with self.assertRaises(ValueError):
                ab.extract_rows(receipt, base, cases)
            receipt["cases"][3]["oracle_pass"] = True
            target = base / receipt["cases"][3]["stdout"]
            target.write_text(target.read_text().replace("20480,20480,PASS", "20479,20480,PASS"))
            with self.assertRaises(ValueError):
                ab.extract_rows(receipt, base, cases)

    def test_read_only_gpu_telemetry_is_observational_and_fail_safe(self):
        from types import SimpleNamespace

        with patch.object(ab.subprocess, "run", return_value=SimpleNamespace(
            returncode=0, stdout="GPU[0]: sclk=0", stderr=""
        )) as invoke:
            r = ab.readonly_gpu_telemetry()
        self.assertEqual(r["returncode"], 0)
        self.assertIn("sclk", r["stdout"])
        self.assertIn("captured_utc", r)
        cmd = invoke.call_args.args[0]
        self.assertEqual(cmd[0], "rocm-smi")
        self.assertEqual(cmd[1:], ["--showclocks", "--showpower", "--showtemp"])
        with patch.object(ab.subprocess, "run", side_effect=OSError("missing")):
            absent = ab.readonly_gpu_telemetry()
        self.assertIsNone(absent["returncode"])
        self.assertIn("missing", absent["error"])

    def test_no_gpu_calls_in_default_plan(self):
        with (redirect_stdout(io.StringIO()) as out,
              patch.object(ab.subprocess, "run",
                           side_effect=AssertionError("would run a process"))):
            self.assertEqual(ab.main([]), 0)
        plan = json.loads(out.getvalue())
        self.assertEqual(plan["schema"], ab.SCHEMA)
        self.assertTrue(plan["numerical_oracle_required_on_every_arm"])
        self.assertTrue(plan["not_full_model"])
        self.assertTrue(plan["power_and_temperature_not_enforced"])
        self.assertEqual(plan["schedule"], [])

    def test_fail_closed_phase_and_input_pair_admission(self):
        self.assertEqual(ab.inputs([], []), [])
        with self.assertRaises(ValueError):
            ab.inputs([pathlib.Path("not-touched")], [])
        for args in (
            ["--phase", "gpu"], ["--phase", "gpu", "--run"],
            ["--phase", "compile"], ["--iterations", "0"],
            ["--iterations", "31"],
        ):
            with patch("sys.stderr", io.StringIO()), self.assertRaises(SystemExit):
                ab.main(args)

    def test_three_binaries_and_source_provenance_are_distinct(self):
        self.assertEqual(set(ab.VARIANTS), {"pageable", "pinned", "rowreuse"})
        self.assertEqual(len({str(p) for p in ab.VARIANTS.values()}), 3)
        for src in ab.VARIANTS.values():
            self.assertTrue(src.is_file())
        self.assertIn("output_col_gemv_reuse<4><<<", ab.VARIANTS["rowreuse"].read_text())
        self.assertIn("hipHostMallocPortable", ab.VARIANTS["pinned"].read_text())
        self.assertNotIn("hipHostMallocPortable", ab.VARIANTS["pageable"].read_text())


if __name__ == "__main__":
    unittest.main()
