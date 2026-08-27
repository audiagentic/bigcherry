"""PROF01/HI132: offline tests for report normalization/rendering."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bigcherry.profiling.report import render_markdown, report_to_dict  # noqa: E402
from bigcherry.profiling.schema import (  # noqa: E402
    ControlBlock, GpuProfilePass, KernelStat,
    ProfileReceipt, ProfileReport,
)


def _receipt(**overrides) -> ProfileReceipt:
    base = dict(
        schema_version=1, campaign_run_id="run1", model_path="/m.gguf",
        platform_name="linux-multi", devices="0,1",
        runtime_profile_name="production-dual-xtx", workload_label="default",
        lane_source="bigcherry-native", lane_build="control",
        build_plan_id="bp1", source_slice_id="ss1", binary_path="/bin/llama-server",
        rocprofv3_version="1.0.0", control_reps=10, profile_passes=2,
        expected_gpu_count=2, started_at="t0", finished_at="t1",
        environment_stable=True, environment_note="",
    )
    base.update(overrides)
    return ProfileReceipt(**base)


class ReportRoundTripTests(unittest.TestCase):
    def test_report_to_dict_is_json_shaped(self):
        report = ProfileReport(receipt=_receipt())
        d = report_to_dict(report)
        self.assertEqual(d["receipt"]["campaign_run_id"], "run1")
        self.assertEqual(d["controls"], ())
        self.assertEqual(d["gpu_passes"], ())
        self.assertFalse(d["cpu"]["available"])

    def test_markdown_renders_unstable_environment_note(self):
        report = ProfileReport(
            receipt=_receipt(environment_stable=False, environment_note="drift 12%"),
        )
        md = render_markdown(report)
        self.assertIn("environment stable: **False**", md)
        self.assertIn("drift 12%", md)

    def test_markdown_renders_controls_and_gpu_passes(self):
        controls = (ControlBlock(
            label="A", reps=10, tg_tps_values=(100.0, 101.0),
            tg_tps_mean=100.5, tg_tps_stddev=0.7,
        ),)
        gpu_passes = (GpuProfilePass(
            label="1", output_dir="/tmp/gpu-1",
            kernels=(KernelStat(
                name="mul_mat_vec_q", calls=10, total_us=100.0, mean_us=10.0,
                p95_us=12.0, vgpr_count=24, sgpr_count=128, scratch_size=0,
                agent_ids=("GPU0", "GPU1"),
            ),),
            agent_ids_seen=("GPU0", "GPU1"), rccl_activity_seen=True,
            expected_gpu_count=2, capture_status="complete",
        ),)
        report = ProfileReport(receipt=_receipt(), controls=controls, gpu_passes=gpu_passes)
        md = render_markdown(report)
        self.assertIn("| A | 10 | 100.50 | 0.70 |", md)
        self.assertIn("mul_mat_vec_q", md)
        self.assertIn("capture status: **complete**", md)

    def test_markdown_reports_cpu_unavailable_by_default(self):
        report = ProfileReport(receipt=_receipt())
        md = render_markdown(report)
        self.assertIn("unavailable", md)


if __name__ == "__main__":
    unittest.main()
