"""VTP05 all-card campaign tests. No HIP, GPU, build or model execution."""
from __future__ import annotations

import importlib.util
import io
import json
import pathlib
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from unittest.mock import Mock, patch

DIRECTORY = pathlib.Path(__file__).resolve().parents[2] / "lab" / "radiance"
sys.path.insert(0, str(DIRECTORY))
SPEC = importlib.util.spec_from_file_location("xtx27_quant_campaign", DIRECTORY / "xtx27_quant_campaign.py")
assert SPEC and SPEC.loader
campaign = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(campaign)


def fake_target_cache(build: pathlib.Path) -> None:
    (build / "CMakeCache.txt").write_text("RAD_GPU_TARGETS:UNINITIALIZED=gfx1201\n")
    r3 = build / "components/libr3/CMakeCache.txt"
    r3.parent.mkdir(parents=True, exist_ok=True)
    r3.write_text("R3_TARGETS:STRING=gfx1100\nRADIANCE_SRC:PATH=/synthetic/pin\n")


class CampaignTests(unittest.TestCase):
    def test_tier_counts_are_unique_and_legal(self):
        expected = {"pilot": 12, "decode": 24, "placement": 72,
                    "prefill": 48, "r9700-local": 48}
        for tier, count in expected.items():
            with self.subTest(tier=tier):
                cases = campaign.entries(tier)
                self.assertEqual(len(cases), count)
                self.assertEqual(len({c["case_id"] for c in cases}), count)
                self.assertEqual({c["quant"] for c in cases}, {"iq6r", "w8a8"})
                self.assertTrue(all(c["cold_mib"] == 0 for c in cases
                                    if c["quant"] == "w8a8"))
                self.assertTrue(all(c["cold_mib"] == 256 for c in cases
                                    if c["quant"] == "iq6r"))
                self.assertTrue(all(c["geometry"]["N"] % 256 == 0 and
                                    c["geometry"]["K"] % 256 == 0 for c in cases))

    def test_default_plan_is_card_free(self):
        buf = io.StringIO()
        with (
            patch.object(campaign.subprocess, "run", side_effect=AssertionError("ran command")),
            redirect_stdout(buf),
        ):
            self.assertEqual(campaign.main([]), 0)
        report = json.loads(buf.getvalue())
        self.assertEqual(report["count"], 12)
        self.assertFalse(report["gpu_used"])

    def test_build_preflight_rejects_missing_kernel_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            build = pathlib.Path(tmp)
            exe = build / "bin/r4d_selftest"
            exe.parent.mkdir(parents=True)
            exe.write_bytes(b"--perfiqr --perfw8")
            fake_target_cache(build)
            for name in ("libr3.so", "libr4d.so"):
                plugin = build / "radiance_home/kernels" / name
                plugin.parent.mkdir(parents=True, exist_ok=True)
                plugin.write_bytes(
                    b"gemm_iqr_nt_m64 gemm_iqr_tiled "
                    b"gemm_i8a8_nt_m16 gemm_i8a8_tiled"
                )
            self.assertEqual(len(campaign.preflight(build)["objects"]), 6)
            (build / "radiance_home/kernels/libr3.so").write_bytes(b"bad")
            with self.assertRaises(ValueError):
                campaign.preflight(build)

    def test_frozen_case_receipt_resumption_and_integrity(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = pathlib.Path(tmp)
            cases = campaign.entries("pilot")
            doc = {
                "schema": 1, "tier": "pilot", "cases": cases,
                "provenance": {"build_dir": "/test", "objects": {
                    key: {"binary_sha256": "A", "plugin_sha256": "B"}
                    for key in ("iq6r:xtx0", "iq6r:xtx1", "iq6r:r9700",
                                "w8a8:xtx0", "w8a8:xtx1", "w8a8:r9700")
                }},
            }
            (directory / "campaign.json").write_text(json.dumps(doc))
            self.assertEqual(campaign.manifest_at(directory)["cases"], cases)
            done, pending = campaign.progress(directory, doc)
            self.assertEqual(len(done), 0)
            self.assertEqual(len(pending), 12)
            one = cases[0]
            receipt = {
                "status": "TIMING_OBTAINED_NUMERICAL_UNCHECKED",
                "geometry": one["geometry"], "m": one["m"],
                "quant": one["quant"], "cold_mib": one["cold_mib"],
                "selected_kernel": one["selected_kernel"],
                "binary_sha256": "A", "plugin_sha256": "B",
                "device_us": 13.2,
                "numerical_oracle": "NOT_TESTED_SELFTEST_PERF_MODE_TIMING_ONLY",
            }
            saved = directory / "receipts" / (one["case_id"] + ".json")
            saved.parent.mkdir()
            saved.write_text(json.dumps(receipt))
            done, pending = campaign.progress(directory, doc)
            self.assertEqual(len(done), 1)
            self.assertEqual(len(pending), 11)
            receipt["plugin_sha256"] = "WRONG ABI"
            saved.write_text(json.dumps(receipt))
            with self.assertRaises(ValueError):
                campaign.progress(directory, doc)

    def test_ffn_barrier_lower_bound_requires_all_physical_rows(self):
        doc = {"cases": campaign.entries("pilot")}
        us = {"xtx0": {"gate_up": 10.0, "down": 5.0},
              "xtx1": {"gate_up": 15.0, "down": 5.0},
              "r9700": {"gate_up": 20.0, "down": 7.0}}
        done = [{"case_id": item["case_id"],
                 "device_us": us[item["gpu"]][item["projection"]]}
                for item in doc["cases"] if item["quant"] == "iq6r"]
        self.assertEqual(campaign.ffn_lower_bounds(doc, done[:-1]), [])
        summary = campaign.ffn_lower_bounds(doc, done)
        self.assertEqual(len(summary), 1)
        self.assertEqual(summary[0]["ideal_two_projection_barrier_us"], 27.0)
        self.assertEqual(summary[0]["minimum_card_occupancy_us"], 27.0)
        self.assertEqual(summary[0]["cache_mode"], "IQR_COLD_FLUSH")

    def test_command_and_lease_intent(self):
        case = campaign.entries("pilot")[0]
        doc = {"provenance": {"build_dir": "/scratch/matched"}}
        argv = campaign.one_command(doc, case, pathlib.Path("/scratch/test"))
        self.assertIn("--run", argv)
        self.assertEqual(argv[argv.index("--gpu") + 1], "xtx0")
        self.assertEqual(argv[argv.index("--cold-mib") + 1], "256")
        self.assertIn("--build-dir", argv)
        self.assertEqual(argv[argv.index("--timeout-s") + 1], "30")
        # One case per invocation, never a hidden whole campaign.
        self.assertEqual(argv.count("--quant"), 1)

    def test_prepare_manifest_is_fixed_and_does_not_run_gpu(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            build = root / "build"
            exe = build / "bin/r4d_selftest"
            exe.parent.mkdir(parents=True)
            exe.write_bytes(b"--perfiqr --perfw8")
            fake_target_cache(build)
            for name in ("libr3.so", "libr4d.so"):
                p = build / "radiance_home/kernels" / name
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_bytes(b"gemm_iqr_nt_m64 gemm_iqr_tiled "
                              b"gemm_i8a8_nt_m16 gemm_i8a8_tiled")
            manifest_dir = root / "campaign"
            with (
                patch.object(campaign.subprocess, "run",
                             return_value=Mock(stdout="abcd1234\n")) as command_mock,
                redirect_stdout(io.StringIO()),
            ):
                self.assertEqual(campaign.main([
                    "--phase", "prepare", "--tier", "pilot",
                    "--build-dir", str(build), "--campaign-dir", str(manifest_dir),
                ]), 0)
                self.assertEqual(command_mock.call_count, 1)
            doc = campaign.manifest_at(manifest_dir)
            campaign.validate_frozen(doc)
            self.assertEqual(doc["checkout"], "abcd1234")
            self.assertEqual(len(doc["cases"]), 12)
            with redirect_stdout(io.StringIO()), self.assertRaises(SystemExit):
                campaign.main(["--phase", "prepare", "--build-dir", str(build),
                               "--campaign-dir", str(manifest_dir)])


if __name__ == "__main__":
    unittest.main()
