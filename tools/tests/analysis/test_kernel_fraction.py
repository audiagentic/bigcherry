"""kernel_fraction parser: column-name drift and agent-id formats (HI35).

rocprofv3 has moved both its CSV column names and its Agent_Id format
between point releases (bare number vs 'Agent N'). A silently mis-parsed
trace produces a plausible percentage, so the parser must be pinned by
tests over both formats.
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bigcherry.analysis import kernel_fraction  # noqa: E402

_NEW_HEADER = (
    '"Kind","Agent_Id","Queue_Id","Stream_Id","Thread_Id","Dispatch_Id",'
    '"Kernel_Id","Kernel_Name","Correlation_Id","Start_Timestamp",'
    '"End_Timestamp","LDS_Block_Size","Scratch_Size","VGPR_Count",'
    '"Accum_VGPR_Count","SGPR_Count","Workgroup_Size_X",'
    '"Workgroup_Size_Y","Workgroup_Size_Z","Grid_Size_X","Grid_Size_Y",'
    '"Grid_Size_Z"'
)


def _row(name: str, start: int, end: int, agent: str) -> str:
    return (
        f'"KERNEL_DISPATCH","{agent}",1,1,1,1,1,"{name}",1,'
        f"{start},{end},0,0,16,0,128,512,1,1,1,1,1,1"
    )


def _write(path: Path, agent: str) -> Path:
    path.write_text(
        "\n".join(
            [
                _NEW_HEADER,
                _row("mul_mat_vec_q_a", 0, 1000, agent),
                _row("rms_norm_b", 1100, 1200, agent),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return path


class AgentFormatTests(unittest.TestCase):
    def test_bare_number_agent(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _write(Path(tmp) / "t.csv", "2")
            trace = kernel_fraction.parse_kernel_trace([path])
        agents = {row["agent"] for row in trace["rows"]}
        self.assertEqual(agents, {2})

    def test_agent_n_agent(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _write(Path(tmp) / "t.csv", "Agent 3")
            trace = kernel_fraction.parse_kernel_trace([path])
        agents = {row["agent"] for row in trace["rows"]}
        self.assertEqual(agents, {3})

    def test_agent_formats_agree(self):
        with tempfile.TemporaryDirectory() as tmp:
            bare = _write(Path(tmp) / "a.csv", "3")
            named = _write(Path(tmp) / "b.csv", "Agent 3")
            a = kernel_fraction.parse_kernel_trace([bare])
            b = kernel_fraction.parse_kernel_trace([named])
        self.assertEqual(
            [(r["kernel"], r["family"], r["dur_ns"], r["agent"]) for r in a["rows"]],
            [(r["kernel"], r["family"], r["dur_ns"], r["agent"]) for r in b["rows"]],
        )

    def test_no_agent_column(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "t.csv"
            path.write_text(
                "\n".join(
                    [
                        '"Kind","Kernel_Name","Start_Timestamp","End_Timestamp"',
                        '"KERNEL_DISPATCH","mul_mat_vec_q_a",0,1000',
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            trace = kernel_fraction.parse_kernel_trace([path])
        self.assertIsNone(trace["rows"][0]["agent"])
        self.assertEqual(trace["columns"]["agent"], None)

    def test_family_split_and_wall_union(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _write(Path(tmp) / "t.csv", "Agent 3")
            trace = kernel_fraction.parse_kernel_trace([path])
        families = {row["kernel"]: row["family"] for row in trace["rows"]}
        self.assertEqual(families["mul_mat_vec_q_a"], "mmvq")
        self.assertEqual(families["rms_norm_b"], "norm/rope/act")
        # union of [0,1000] and [1100,1200] is 1200, not 1100
        self.assertEqual(trace["wall_ns"], 1200)


class TimingFailClosedTests(unittest.TestCase):
    """gpt-dev-agent review, 2026-08-31: a blank timing cell used to
    silently become 0ns instead of failing, and a short/truncated row was
    silently skipped instead of raising."""

    def test_blank_duration_cell_raises_instead_of_becoming_zero(self):
        header = '"Kind","Kernel_Name","Start_Timestamp","End_Timestamp","Duration"'
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "t.csv"
            path.write_text(
                "\n".join(
                    [
                        header,
                        '"KERNEL_DISPATCH","mul_mat_vec_q_a",0,1000,1000',
                        # blank Duration on an otherwise-valid row -- must
                        # not silently contribute 0ns to the totals.
                        '"KERNEL_DISPATCH","mul_mat_vec_q_b",2000,3000,',
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaises(kernel_fraction.KernelFractionError):
                kernel_fraction.parse_kernel_trace([path])

    def test_truncated_row_raises_instead_of_being_silently_dropped(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "t.csv"
            path.write_text(
                "\n".join(
                    [
                        '"Kind","Kernel_Name","Start_Timestamp","End_Timestamp"',
                        '"KERNEL_DISPATCH","mul_mat_vec_q_a",0,1000',
                        '"KERNEL_DISPATCH","mul_mat_vec_q_b",2000',  # short row
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaises(kernel_fraction.KernelFractionError):
                kernel_fraction.parse_kernel_trace([path])

    def test_blank_start_cell_raises_instead_of_becoming_zero(self):
        header = '"Kind","Kernel_Name","Start_Timestamp","End_Timestamp"'
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "t.csv"
            path.write_text(
                "\n".join([header, '"KERNEL_DISPATCH","mul_mat_vec_q_a",,1000']) + "\n",
                encoding="utf-8",
            )
            with self.assertRaises(kernel_fraction.KernelFractionError):
                kernel_fraction.parse_kernel_trace([path])

    def test_zero_is_a_valid_value_not_confused_with_blank(self):
        header = '"Kind","Kernel_Name","Start_Timestamp","End_Timestamp"'
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "t.csv"
            path.write_text(
                "\n".join([header, '"KERNEL_DISPATCH","mul_mat_vec_q_a",0,0']) + "\n",
                encoding="utf-8",
            )
            trace = kernel_fraction.parse_kernel_trace([path])  # must not raise
        self.assertEqual(trace["rows"][0]["dur_ns"], 0)


class MixedFileAndDurationConsistencyTests(unittest.TestCase):
    """gpt-dev-agent review round 3, 2026-08-31: mixed timing schemas
    across multiple input files could produce a false matmul_wall_pct
    (silently omitting the duration-only file's matmul rows from the
    union), and Duration disagreeing with End-Start silently used the
    wrong span endpoint instead of failing."""

    def test_mixing_a_timestamped_and_duration_only_file_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            timestamped = Path(tmp) / "a.csv"
            timestamped.write_text(
                '"Kind","Kernel_Name","Start_Timestamp","End_Timestamp"\n'
                '"KERNEL_DISPATCH","mul_mat_vec_q_a",0,1000\n',
                encoding="utf-8",
            )
            duration_only = Path(tmp) / "b.csv"
            duration_only.write_text(
                '"Kind","Kernel_Name","Duration"\n'
                '"KERNEL_DISPATCH","mul_mat_vec_q_b",500\n',
                encoding="utf-8",
            )
            with self.assertRaises(kernel_fraction.KernelFractionError):
                kernel_fraction.parse_kernel_trace([timestamped, duration_only])

    def test_two_timestamped_files_are_fine(self):
        with tempfile.TemporaryDirectory() as tmp:
            a = _write(Path(tmp) / "a.csv", "Agent 1")
            b = _write(Path(tmp) / "b.csv", "Agent 2")
            trace = kernel_fraction.parse_kernel_trace([a, b])  # must not raise
        self.assertEqual(len(trace["rows"]), 4)

    def test_duration_disagreeing_with_end_minus_start_raises(self):
        header = '"Kind","Kernel_Name","Start_Timestamp","End_Timestamp","Duration"'
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "t.csv"
            path.write_text(
                "\n".join(
                    [header, '"KERNEL_DISPATCH","mul_mat_vec_q_a",0,1000,999']  # 999 != 1000-0
                )
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaises(kernel_fraction.KernelFractionError):
                kernel_fraction.parse_kernel_trace([path])

    def test_duration_agreeing_with_end_minus_start_is_fine(self):
        header = '"Kind","Kernel_Name","Start_Timestamp","End_Timestamp","Duration"'
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "t.csv"
            path.write_text(
                "\n".join([header, '"KERNEL_DISPATCH","mul_mat_vec_q_a",0,1000,1000'])
                + "\n",
                encoding="utf-8",
            )
            trace = kernel_fraction.parse_kernel_trace([path])  # must not raise
        self.assertEqual(trace["rows"][0]["dur_ns"], 1000)


if __name__ == "__main__":
    unittest.main()
