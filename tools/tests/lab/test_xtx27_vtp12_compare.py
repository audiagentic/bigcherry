"""Card-free adversarial tests for VTP12 historical FP8/INT8 comparison."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "lab" / "radiance"))
import xtx27_vtp12_compare as c


def fixture(*, gpu: str = "r9700", proj: str = "gate_up", m: int = 4,
            n: int = 14336, k: int = 5120, i8: float = 70.0,
            fp8: float = 60.0) -> tuple[dict, str]:
    row = "gemm_i8a8_nt_m16" if m <= 16 else "gemm_i8a8_tiled"
    times = {
        "gemm_fp8a8_tiled": fp8 + 10,
        "gemm_i8a8_tiled": i8 + (10 if m <= 16 else 0),
    }
    if m <= 64:
        times.update({
            "gemm_fp8a8_nt_m16": fp8,
            "gemm_i8a8_nt_m16": i8,
        })
    text = f"\n8-bit trunk GEMMs  N={n} K={k} M={m}  9 iters\n"
    text += "".join(f"  {key:<22s} {value:9.1f} us  1.0 TFLOP/s\n"
                    for key, value in times.items())
    receipt = {
        "status": "TIMING_OBTAINED_NUMERICAL_UNCHECKED",
        "kind": "VTP05_RADIANCE_KERNEL_TIMING_ONLY_NOT_MODEL_QUALITY",
        "quant": "w8a8", "m": m, "iters": 9, "cold_mib": 0,
        "cache_mode": "warm_or_resident",
        "geometry": {"candidate": "capacity", "m": m, "gpu": gpu,
                     "arch": c.ARCH[gpu], "part": "physical",
                     "projection": proj, "N": n, "K": k},
        "observed_nkm": [n, k, m], "returncode": 0,
        "binary_sha256": "a" * 64, "plugin_sha256": "b" * 64,
        "selected_kernel": row,
        "device_us": float(times[row]),
        "numerical_oracle": "NOT_TESTED_SELFTEST_PERF_MODE_TIMING_ONLY",
        "argv": ["/build/bin/r4d_selftest",
                 "/build/kernel/" + ("libr4d.so" if gpu == "r9700" else "libr3.so"),
                 "--perfw8", str(n), str(k), str(m), "9"],
    }
    return receipt, text


def write_case(root: Path, tag: str, receipt: dict, output: str) -> None:
    folder = root / tag
    folder.mkdir()
    (folder / "receipt.json").write_text(json.dumps(receipt))
    (folder / "stdout.log").write_text(output)


class ParseTest(unittest.TestCase):
    def test_realistic_fp8_and_int8_rows(self) -> None:
        _, text = fixture()
        got = c.parse_perf(text, 14336, 5120, 4, 9)
        self.assertEqual(got["gemm_fp8a8_nt_m16"], 60.0)
        self.assertEqual(got["gemm_i8a8_nt_m16"], 70.0)

    def test_missing_row_rejected(self) -> None:
        _, text = fixture()
        with self.assertRaisesRegex(ValueError, "missing/invalid"):
            c.parse_perf(text.replace("gemm_fp8a8_nt_m16", "missing_kernel____"), 14336, 5120, 4, 9)

    def test_mismatch_and_duplicate_rejected(self) -> None:
        _, text = fixture()
        with self.assertRaisesRegex(ValueError, "geometry mismatch"):
            c.parse_perf(text, 14336, 5120, 8, 9)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            c.parse_perf(text + "  gemm_i8a8_tiled  10.0 us  0.1 TFLOP/s\n", 14336, 5120, 4, 9)

    def test_m64_and_large_m_rows(self) -> None:
        _, text = fixture(m=64)
        self.assertEqual(c.parse_perf(text, 14336, 5120, 64, 9)["gemm_fp8a8_nt_m16"], 60)
        _, large = fixture(m=128)
        rows = c.parse_perf(large, 14336, 5120, 128, 9)
        self.assertNotIn("gemm_fp8a8_nt_m16", rows)


class ReceiptTest(unittest.TestCase):
    def test_valid_prior_case_and_integrity_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            receipt, output = fixture()
            write_case(d, "case", receipt, output)
            self.assertEqual(c.read_pair(d / "case")["fp8_preferred_us"], 60)
            receipt["device_us"] = 100.0
            (d / "case" / "receipt.json").write_text(json.dumps(receipt))
            with self.assertRaisesRegex(ValueError, "does not match"):
                c.read_pair(d / "case")

    def test_wrong_plugin_or_command_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            receipt, output = fixture()
            receipt["argv"][1] = "/build/kernels/libr3.so"
            write_case(d, "case", receipt, output)
            with self.assertRaisesRegex(ValueError, "mismatch"):
                c.read_pair(d / "case")

    def test_partial_group_is_not_estimated(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            receipt, output = fixture()
            write_case(root, "case", receipt, output)
            result = c.analyze(root)
            self.assertEqual(result["source_receipt_count"], 1)
            self.assertEqual(result["summaries"], [])

    def test_full_group_xtx_barrier_and_r9700_noncritical(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for gpu in c.ARCH:
                for proj in ("gate_up", "down"):
                    slow_x = gpu != "r9700"
                    i8 = 100.0 if slow_x else 70.0
                    fp8 = 80.0 if slow_x else 60.0
                    receipt, output = fixture(gpu=gpu, proj=proj, i8=i8, fp8=fp8)
                    write_case(root, f"{gpu}-{proj}", receipt, output)
            result = c.analyze(root)
            self.assertFalse(result["simultaneous_three_gpu_execution"])
            summary = result["summaries"][0]
            self.assertEqual(summary["ideal_stage_barrier_saving_percent"], 0)
            self.assertEqual(summary["uniform_i8_selected"]["sum_stage_barriers_us"], 200.0)
            self.assertEqual(summary["mixed_xtx_i8_r9700_fp8_provisional"]["sum_stage_barriers_us"], 200.0)

    def test_r9700_critical_path_can_shrink_ideal_barrier(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for gpu in c.ARCH:
                for proj in ("gate_up", "down"):
                    receipt, output = fixture(gpu=gpu, proj=proj,
                                              i8=140.0 if gpu == "r9700" else 100.0,
                                              fp8=90.0 if gpu == "r9700" else 80.0)
                    write_case(root, f"{gpu}-{proj}", receipt, output)
            summary = c.analyze(root)["summaries"][0]
            self.assertGreater(summary["ideal_stage_barrier_saving_percent"], 0)
            self.assertEqual(summary["uniform_i8_selected"]["sum_stage_barriers_us"], 280.0)
            self.assertEqual(summary["mixed_xtx_i8_r9700_fp8_provisional"]["sum_stage_barriers_us"], 200.0)

    def test_duplicate_case_aborts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            receipt, output = fixture()
            write_case(root, "same-1", receipt, output)
            write_case(root, "same-2", receipt, output)
            with self.assertRaisesRegex(ValueError, "duplicate source"):
                c.analyze(root)


    def test_cross_case_plugin_drift_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            receipt, output = fixture()
            write_case(root, "first", receipt, output)
            receipt2, output2 = fixture(m=8)
            receipt2["plugin_sha256"] = "c" * 64
            write_case(root, "second", receipt2, output2)
            with self.assertRaisesRegex(ValueError, "different plugin"):
                c.analyze(root)


if __name__ == "__main__":
    unittest.main()
