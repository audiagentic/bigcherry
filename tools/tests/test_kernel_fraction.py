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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bigcherry import kernel_fraction  # noqa: E402

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


if __name__ == "__main__":
    unittest.main()
