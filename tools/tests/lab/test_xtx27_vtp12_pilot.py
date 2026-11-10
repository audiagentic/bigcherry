"""Card-free tests for the VTP12 bounded ABBA hardware experiment."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "lab" / "radiance"))
import xtx27_vtp12_pilot as v


class GeometryTests(unittest.TestCase):
    def test_total_exact_and_256_aligned_all_candidates(self) -> None:
        for candidate in v.CANDIDATES:
            shapes = [v.geometry(candidate, role, "down") for role in v.ROLES]
            self.assertEqual(sum(x["effective_intermediate"] for x in shapes), 17408)
            self.assertTrue(all(x["effective_intermediate"] % 256 == 0 for x in shapes))
            self.assertEqual([x["effective_intermediate"] for x in shapes],
                             {"control": [5120, 5120, 7168],
                              "r9700-45": [4864, 4864, 7680],
                              "r9700-50": [4352, 4352, 8704]}[candidate])

    def test_ffn_gate_down_geometry(self) -> None:
        for candidate in v.CANDIDATES:
            for role in v.ROLES:
                gate = v.geometry(candidate, role, "gate_up")
                down = v.geometry(candidate, role, "down")
                self.assertEqual(gate["N"], 2 * down["K"])
                self.assertEqual(gate["K"], 5120)
                self.assertEqual(down["N"], 5120)
                self.assertEqual(gate["arch"], v.q.ARCH[role])

    def test_invalid_candidate_or_projection_rejected(self) -> None:
        for candidate, gpu, proj in [
            ("unknown", "xtx0", "down"), ("control", "unknown", "down"),
            ("control", "xtx0", "qkv"),
        ]:
            with self.assertRaisesRegex(ValueError, "unsupported"):
                v.geometry(candidate, gpu, proj)

    def test_abba_schedule_complete_and_deterministic(self) -> None:
        for challenger in ("r9700-45", "r9700-50"):
            schedule = v.schedule(challenger, 4)
            self.assertEqual(len(schedule), 24)
            for role in v.ROLES:
                for projection in v.PROJECTIONS:
                    cases = [x for x in schedule if x["gpu"] == role
                             and x["projection"] == projection]
                    self.assertEqual([x["candidate"] for x in cases],
                                     ["control", challenger, challenger, "control"])

    def test_bad_token_rows_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "pilot limited"):
            v.schedule("r9700-45", 2048)


class BarrierTests(unittest.TestCase):
    @staticmethod
    def fixture(*, challenger: str, x: float, r_i8: float,
                r_fp8: float) -> list[dict]:
        result = []
        for entry in v.schedule(challenger, 4):
            gpu = entry["gpu"]
            is_r = gpu == "r9700"
            base = r_i8 if is_r else x
            i8 = base * (1.0 if entry["candidate"] == "control" else 1.1)
            f8 = r_fp8 if is_r else x * 1.5
            result.append({**entry, "i8_us": i8, "fp8_us": f8})
        return result

    def test_xtx_critical_path_no_fp8_barrier_win(self) -> None:
        entries = self.fixture(challenger="r9700-45", x=120, r_i8=65, r_fp8=50)
        summary = v.to_summary(entries, challenger="r9700-45")
        self.assertEqual(summary["stage_estimates"]["control_i8_us"]["ideal_ffn_stage_barriers_us"], 240)
        self.assertEqual(summary["stage_estimates"]["challenger_fp8_us"]["ideal_ffn_stage_barriers_us"], 264)
        self.assertFalse("three_gpu" in summary["status"])

    def test_r9700_critical_path_has_mixed_speedup(self) -> None:
        entries = self.fixture(challenger="r9700-50", x=100, r_i8=145, r_fp8=105)
        summary = v.to_summary(entries, challenger="r9700-50")
        self.assertEqual(summary["stage_estimates"]["challenger_i8_us"]["ideal_ffn_stage_barriers_us"], 319.0)
        self.assertEqual(summary["stage_estimates"]["challenger_fp8_us"]["ideal_ffn_stage_barriers_us"], 220.0)

    def test_missing_or_reordered_abba_rejected(self) -> None:
        cases = self.fixture(challenger="r9700-45", x=100, r_i8=90, r_fp8=80)
        with self.assertRaisesRegex(ValueError, "incomplete"):
            v.to_summary(cases[:-1], challenger="r9700-45")
        cases[0]["candidate"] = "r9700-45"
        with self.assertRaisesRegex(ValueError, "order"):
            v.to_summary(cases, challenger="r9700-45")


class LeaseTests(unittest.TestCase):
    def test_no_lease_marker_never_touches_hardware(self) -> None:
        with (
            patch.dict(v.os.environ, {"VTP12_LEASED": "0"}),
            patch.object(v.subprocess, "run") as hardware,
        ):
            with self.assertRaisesRegex(RuntimeError, "lease"):
                v.validate_context()
            hardware.assert_not_called()

    def test_inherited_device_mask_refused_without_hardware(self) -> None:
        with (
            patch.dict(v.os.environ, {"VTP12_LEASED": "1", "HIP_VISIBLE_DEVICES": "2"}),
            patch.object(v.subprocess, "run") as hardware,
        ):
            with self.assertRaisesRegex(RuntimeError, "filter"):
                v.validate_context()
            hardware.assert_not_called()


if __name__ == "__main__":
    unittest.main()
