"""PROF01/HI132: offline tests for rocprofv3 CSV parsing -- no GPU
required. Fixture columns match a real rocprofv3 --kernel-trace --stats
run captured on Brutus this session (RD33's kernel-resource diagnostic)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bigcherry.profiling import rocprof  # noqa: E402

_HEADER = (
    "Kind,Agent_Id,Queue_Id,Stream_Id,Thread_Id,Dispatch_Id,Kernel_Id,"
    "Kernel_Name,Correlation_Id,Start_Timestamp,End_Timestamp,LDS_Block_Size,"
    "Scratch_Size,VGPR_Count,Accum_VGPR_Count,SGPR_Count,Workgroup_Size_X,"
    "Workgroup_Size_Y,Workgroup_Size_Z,Grid_Size_X,Grid_Size_Y,Grid_Size_Z\n"
)


def _row(agent, name, start, end, vgpr=24, sgpr=128, scratch=0):
    return (
        f'"KERNEL_DISPATCH","{agent}","1","1","1","1","1","{name}","1",'
        f'"{start}","{end}","0","{scratch}","{vgpr}","0","{sgpr}","32","8","1",'
        f'"512","8","1"\n'
    )


class ParseKernelTraceTests(unittest.TestCase):
    def test_expected_reduction_provider_matches_runtime_selector(self):
        self.assertEqual(
            rocprof.expected_reduction_provider({"GGML_HIP_REDUCE_PLAN": "meta"}),
            "meta",
        )
        self.assertEqual(
            rocprof.expected_reduction_provider({"GGML_HIP_REDUCE_PLAN": "rccl"}),
            "rccl",
        )
        self.assertEqual(rocprof.expected_reduction_provider({}), "auto")
        self.assertEqual(
            rocprof.expected_reduction_provider({"GGML_HIP_REDUCE_PLAN": "invalid"}),
            "auto",
        )

    def test_aggregates_multiple_dispatches_of_the_same_kernel(self):
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.csv"
            path.write_text(
                _HEADER
                + _row("GPU0", "void mul_mat_vec_q<...>", 1000, 8000)
                + _row("GPU0", "void mul_mat_vec_q<...>", 2000, 6500),
                encoding="utf-8",
            )
            stats = rocprof.parse_kernel_trace(path)
            self.assertEqual(len(stats), 1)
            k = stats[0]
            self.assertEqual(k.calls, 2)
            self.assertAlmostEqual(k.total_us, 7.0 + 4.5)
            self.assertEqual(k.vgpr_count, 24)
            self.assertEqual(k.sgpr_count, 128)
            self.assertEqual(k.scratch_size, 0)
            self.assertEqual(k.agent_ids, ("GPU0",))

    def test_distinct_kernel_names_produce_distinct_rows(self):
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.csv"
            path.write_text(
                _HEADER
                + _row("GPU0", "kernel_a", 0, 1000)
                + _row("GPU0", "kernel_b", 0, 2000),
                encoding="utf-8",
            )
            stats = rocprof.parse_kernel_trace(path)
            names = {k.name for k in stats}
            self.assertEqual(names, {"kernel_a", "kernel_b"})

    def test_missing_file_raises(self):
        with self.assertRaises(rocprof.RocprofError):
            rocprof.parse_kernel_trace(Path("/nonexistent/trace.csv"))


class BuildGpuProfilePassTests(unittest.TestCase):
    def _write_two_gpu_kernel_trace(self, out_dir):
        (out_dir / "run_kernel_trace.csv").write_text(
            _HEADER
            + _row("GPU0", "kernel_a", 0, 1000)
            + _row("GPU1", "kernel_a", 0, 1000),
            encoding="utf-8",
        )

    def test_single_gpu_capture_is_complete(self):
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            out_dir = Path(directory)
            (out_dir / "run_kernel_trace.csv").write_text(
                _HEADER + _row("GPU0", "kernel_a", 0, 1000), encoding="utf-8",
            )
            gp = rocprof.build_gpu_profile_pass(
                label="1", output_dir=out_dir, expected_gpu_count=1,
            )
            self.assertEqual(gp.capture_status, "complete")
            self.assertEqual(gp.agent_ids_seen, ("GPU0",))

    def test_multi_gpu_capture_missing_second_agent_is_incomplete(self):
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            out_dir = Path(directory)
            (out_dir / "run_kernel_trace.csv").write_text(
                _HEADER + _row("GPU0", "kernel_a", 0, 1000), encoding="utf-8",
            )
            gp = rocprof.build_gpu_profile_pass(
                label="1", output_dir=out_dir, expected_gpu_count=2,
                expected_reduction_provider="meta",
            )
            self.assertEqual(gp.capture_status, "incomplete_multi_gpu_capture")

    def test_multi_gpu_capture_missing_rccl_activity_is_incomplete(self):
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            out_dir = Path(directory)
            (out_dir / "run_kernel_trace.csv").write_text(
                _HEADER
                + _row("GPU0", "kernel_a", 0, 1000)
                + _row("GPU1", "kernel_a", 0, 1000),
                encoding="utf-8",
            )
            # no *rccl*trace.csv written -- both agents present but no RCCL
            gp = rocprof.build_gpu_profile_pass(
                label="1", output_dir=out_dir, expected_gpu_count=2,
                expected_reduction_provider="rccl",
            )
            self.assertEqual(gp.capture_status, "incomplete_multi_gpu_capture")

    def test_meta_expected_multi_gpu_capture_without_rccl_is_complete(self):
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            out_dir = Path(directory)
            self._write_two_gpu_kernel_trace(out_dir)
            gp = rocprof.build_gpu_profile_pass(
                label="1", output_dir=out_dir, expected_gpu_count=2,
                expected_reduction_provider="meta",
            )
            self.assertEqual(gp.capture_status, "complete")
            self.assertEqual(gp.expected_reduction_provider, "meta")
            self.assertFalse(gp.rccl_activity_seen)

    def test_multi_gpu_capture_with_agents_and_rccl_is_complete(self):
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            out_dir = Path(directory)
            (out_dir / "run_kernel_trace.csv").write_text(
                _HEADER
                + _row("GPU0", "kernel_a", 0, 1000)
                + _row("GPU1", "kernel_a", 0, 1000),
                encoding="utf-8",
            )
            (out_dir / "run_rccl_api_trace.csv").write_text(
                "Kind,Name\nRCCL_API,ncclAllReduce\n", encoding="utf-8",
            )
            gp = rocprof.build_gpu_profile_pass(
                label="1", output_dir=out_dir, expected_gpu_count=2,
            )
            self.assertEqual(gp.capture_status, "complete")
            self.assertTrue(gp.rccl_activity_seen)

    def test_no_kernel_trace_file_raises(self):
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(rocprof.RocprofError):
                rocprof.build_gpu_profile_pass(
                    label="1", output_dir=Path(directory), expected_gpu_count=1,
                )


if __name__ == "__main__":
    unittest.main()
