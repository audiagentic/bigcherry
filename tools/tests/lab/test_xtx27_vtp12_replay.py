"""Card-free frozen VTP12 replay tests including M64 dispatch correction."""
from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "lab" / "radiance"))
import xtx27_vtp12_pilot as p
import xtx27_vtp12_replay as r


def fixture(directory: Path, *, m: int = 64, challenger: str = "r9700-45") -> None:
    cases = p.schedule(challenger, m)
    manifest = {
        "schema": p.SCHEMA, "challenger": challenger, "m": m, "iters": 9,
        "cases": cases,
        "provenance": {"build_dir": "/offline/build",
                       "plugins": {gpu: {"path": ("/offline/" + ("libr4d.so" if
                                    gpu == "r9700" else "libr3.so"))} for gpu in p.ROLES}},
    }
    (directory / "campaign.json").write_text(json.dumps(manifest))
    (directory / "summary.json").write_text('{"original":"unmodified"}\n')
    for ordinal, case in enumerate(cases):
        x = case["geometry"]
        gpu = case["gpu"]
        int8_narrow = 110.0 if gpu == "r9700" else 245.0
        int8_tiled = 225.0 if gpu == "r9700" else 180.0
        fp8_narrow = 101.0 if gpu == "r9700" else 900.0
        fp8_tiled = 160.0 if gpu == "r9700" else 700.0
        rows = {
            "gemm_i8a8_nt_m16": int8_narrow,
            "gemm_i8a8_tiled": int8_tiled,
            "gemm_fp8a8_nt_m16": fp8_narrow,
            "gemm_fp8a8_tiled": fp8_tiled,
        }
        output = f"\n8-bit trunk GEMMs  N={x['N']} K={x['K']} M={m}  9 iters\n"
        output += "".join(f"  {key:<22s} {value:9.1f} us  1.0 TFLOP/s\n"
                          for key, value in rows.items())
        frow = "gemm_fp8a8_nt_m16"
        raw = {
            **case, "ordinal": ordinal,
            "status": "SYNTHETIC_WARM_ONLY_NUMERICAL_UNCHECKED",
            "returncode": 0,
            "argv": p.q.command(Path("/offline/build/bin/r4d_selftest"),
                                Path(manifest["provenance"]["plugins"][gpu]["path"]),
                                quant="w8a8", geo=x, m=m, iters=9, cold_mib=0),
            "i8_kernel": "gemm_i8a8_nt_m16", "i8_us": int8_narrow,
            "fp8_kernel": frow, "fp8_us": fp8_narrow,
            "all_rows_us": rows,
            "stdout_sha256": hashlib.sha256(output.encode()).hexdigest(),
        }
        (directory / f"case-{ordinal:02d}.json").write_text(json.dumps(raw))
        (directory / f"case-{ordinal:02d}.stdout").write_text(output)


class ReplayTests(unittest.TestCase):
    def test_m64_corrects_source_i8_narrow_vs_tiled(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture(root)
            result = r.replay(root)
            self.assertEqual(result["cases_validated"], 24)
            self.assertIn("M=64", result["original_summary_warning"])
            policy = result["policies"]["vtp05_declared_i8_dispatch"]
            self.assertEqual(policy["per_card"]["xtx0"]["gate_up"]["control_i8_us"], 180)
            self.assertEqual(policy["per_card"]["r9700"]["gate_up"]["control_i8_us"], 225)
            optimistic = result["policies"]["optimistic_best_observed_i8"]
            self.assertEqual(optimistic["per_card"]["r9700"]["gate_up"]["control_i8_us"], 110)
            self.assertEqual(policy["stage_estimates"]["control_i8_us"]["ideal_ffn_stage_barriers_us"], 450)
            self.assertEqual(optimistic["stage_estimates"]["control_i8_us"]["ideal_ffn_stage_barriers_us"], 360)

    def test_43_percent_128_tile_candidate_replays(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture(root, m=4, challenger="r9700-43")
            data = r.replay(root)
            self.assertEqual(data["challenger"], "r9700-43")
            self.assertEqual(data["cases_validated"], 24)
            self.assertEqual(data["policies"]["vtp05_declared_i8_dispatch"]
                             ["stage_estimates"]["control_i8_us"]
                             ["ideal_ffn_stage_barriers_us"], 490)

    def test_decode_m4_preserves_narrow_row(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture(root, m=4, challenger="r9700-50")
            result = r.replay(root)
            self.assertEqual(result["policies"]["vtp05_declared_i8_dispatch"]
                             ["per_card"]["xtx0"]["gate_up"]["control_i8_us"], 245)
            self.assertIn("M<=16", result["original_summary_warning"])

    def test_modified_output_detected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture(root)
            file = root / "case-02.stdout"
            file.write_text(file.read_text().replace("245.0", "100.0"))
            with self.assertRaisesRegex(ValueError, "inconsistent"):
                r.replay(root)

    def test_modified_shapes_detected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture(root)
            file = root / "case-00.json"
            item = json.loads(file.read_text())
            item["geometry"]["N"] = 16
            file.write_text(json.dumps(item))
            with self.assertRaisesRegex(ValueError, "inconsistent"):
                r.replay(root)

    def test_missing_case_detected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture(root)
            (root / "case-17.json").unlink()
            with self.assertRaisesRegex(ValueError, "missing"):
                r.replay(root)

    def test_modified_kernel_row_detected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture(root)
            file = root / "case-00.json"
            item = json.loads(file.read_text())
            item["all_rows_us"]["gemm_i8a8_tiled"] = 1.0
            file.write_text(json.dumps(item))
            with self.assertRaisesRegex(ValueError, "kernel timing mismatch"):
                r.replay(root)

    def test_extra_case_detected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture(root)
            (root / "case-24.json").write_text("{}")
            with self.assertRaisesRegex(ValueError, "extra"):
                r.replay(root)

    def test_source_summary_is_not_mutated(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture(root)
            original = (root / "summary.json").read_bytes()
            a = r.replay(root)
            self.assertEqual(original, (root / "summary.json").read_bytes())
            self.assertEqual(a["source_original_summary_sha256"],
                             hashlib.sha256(original).hexdigest())


if __name__ == "__main__":
    unittest.main()
