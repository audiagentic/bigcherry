"""GP08: tests for the AllReduce provider-arm end-to-end comparison tool.

No real GPU/RCCL work here -- fake child programs stand in for real
provider binaries, matching test_ab_benchmark.py's own pattern for the
module this reuses."""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bigcherry.campaign import benchmark as ab_benchmark  # noqa: E402
from bigcherry.campaign import collective_benchmark as cb  # noqa: E402


def _python_executable() -> str:
    if Path(sys.executable).is_file():
        return sys.executable
    return shutil.which("python") or sys.executable


def _py(code: str) -> str:
    return textwrap.dedent(code).strip()


class LoadArmsConfigTests(unittest.TestCase):
    def test_loads_valid_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "arms.json"
            path.write_text(json.dumps([
                {"name": "a", "binary": "/bin/a", "env": {"X": "1"}, "requires_rccl": True},
                {"name": "b", "binary": "/bin/b"},
            ]))
            arms = cb.load_arms_config(path)
        self.assertEqual(len(arms), 2)
        self.assertEqual(arms[0]["name"], "a")
        self.assertTrue(arms[0]["requires_rccl"])
        self.assertEqual(arms[1]["env"], {})
        self.assertFalse(arms[1]["requires_rccl"])

    def test_rejects_duplicate_names(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "arms.json"
            path.write_text(json.dumps([
                {"name": "a", "binary": "/bin/a"},
                {"name": "a", "binary": "/bin/other"},
            ]))
            with self.assertRaises(ValueError):
                cb.load_arms_config(path)

    def test_rejects_missing_binary_field(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "arms.json"
            path.write_text(json.dumps([{"name": "a"}]))
            with self.assertRaises(ValueError):
                cb.load_arms_config(path)

    def test_rejects_empty_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "arms.json"
            path.write_text(json.dumps([]))
            with self.assertRaises(ValueError):
                cb.load_arms_config(path)


class CheckQualifiedTests(unittest.TestCase):
    def test_finds_matching_pass_row(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cases.jsonl"
            path.write_text(
                json.dumps({"topology_id": "xtx_xtx", "compatibility_revision_id": "rev1", "classification": "pass"}) + "\n"
                + json.dumps({"topology_id": "xtx_xtx", "compatibility_revision_id": "rev2", "classification": "pass"}) + "\n"
            )
            self.assertTrue(cb.check_qualified(path, topology_id="xtx_xtx", revision_id="rev1"))

    def test_rejects_when_only_non_pass_rows_match(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cases.jsonl"
            path.write_text(
                json.dumps({"topology_id": "xtx_xtx", "compatibility_revision_id": "rev1", "classification": "timeout"}) + "\n"
            )
            self.assertFalse(cb.check_qualified(path, topology_id="xtx_xtx", revision_id="rev1"))

    def test_rejects_wrong_topology(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cases.jsonl"
            path.write_text(
                json.dumps({"topology_id": "xtx_r9700", "compatibility_revision_id": "rev1", "classification": "pass"}) + "\n"
            )
            self.assertFalse(cb.check_qualified(path, topology_id="xtx_xtx", revision_id="rev1"))

    def test_rejects_wrong_revision(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cases.jsonl"
            path.write_text(
                json.dumps({"topology_id": "xtx_xtx", "compatibility_revision_id": "old-rev", "classification": "pass"}) + "\n"
            )
            self.assertFalse(cb.check_qualified(path, topology_id="xtx_xtx", revision_id="new-rev"))

    def test_missing_file_is_unqualified_not_an_error(self):
        self.assertFalse(cb.check_qualified(Path("/nonexistent/cases.jsonl"), topology_id="x", revision_id="y"))


class BuildArmScheduleTests(unittest.TestCase):
    def test_covers_every_candidate_every_round(self):
        schedule = cb.build_arm_schedule(3, "baseline", ["a", "b", "c"], seed=1)
        self.assertEqual(len(schedule), 9)  # 3 rounds x 3 candidates
        for round_index in range(3):
            round_pairs = [(f, s) for r, f, s in schedule if r == round_index]
            candidates_seen = {c for pair in round_pairs for c in pair if c != "baseline"}
            self.assertEqual(candidates_seen, {"a", "b", "c"})

    def test_order_alternates_by_round(self):
        schedule = cb.build_arm_schedule(2, "baseline", ["a"], seed=1)
        round0 = [(f, s) for r, f, s in schedule if r == 0][0]
        round1 = [(f, s) for r, f, s in schedule if r == 1][0]
        self.assertEqual(round0, ("baseline", "a"))
        self.assertEqual(round1, ("a", "baseline"))

    def test_deterministic_given_seed(self):
        a = cb.build_arm_schedule(4, "baseline", ["a", "b"], seed=5)
        b = cb.build_arm_schedule(4, "baseline", ["a", "b"], seed=5)
        self.assertEqual(a, b)


class ScheduleNamedArmsTests(unittest.TestCase):
    def test_covers_full_permutation_block(self):
        import itertools
        orders = ab_benchmark.schedule_named_arms(6, ["x", "y", "z"], seed=3)
        self.assertEqual(set(orders), set(itertools.permutations(("x", "y", "z"))))

    def test_rejects_fewer_than_two_arms(self):
        with self.assertRaises(ValueError):
            ab_benchmark.schedule_named_arms(3, ["only-one"])

    def test_rejects_duplicate_arm_names(self):
        with self.assertRaises(ValueError):
            ab_benchmark.schedule_named_arms(3, ["a", "a"])


class MainIntegrationTests(unittest.TestCase):
    """Fake child programs stand in for real provider binaries -- no GPU
    or RCCL work, matching test_ab_benchmark.py's own pattern."""

    def _make_fake_binary(self, tmp_path: Path, rate: float) -> str:
        fake = tmp_path / f"fake_{rate}.py"
        fake.write_text(_py(f"""
            import sys
            print("rate: {rate}")
        """))
        wrapper = tmp_path / f"wrapper_{rate}.sh"
        if sys.platform.startswith("win"):
            wrapper = tmp_path / f"wrapper_{rate}.bat"
            wrapper.write_text(f'@"{_python_executable()}" "{fake}" %*\r\n')
        else:
            wrapper.write_text(f'#!/bin/sh\nexec "{_python_executable()}" "{fake}" "$@"\n')
            wrapper.chmod(0o755)
        return str(wrapper)

    def test_refuses_rccl_arm_without_qualification_args(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            baseline_bin = self._make_fake_binary(tmp_path, 100.0)
            arms_config = tmp_path / "arms.json"
            arms_config.write_text(json.dumps([
                {"name": "baseline", "binary": baseline_bin},
                {"name": "rccl-candidate", "binary": baseline_bin, "requires_rccl": True},
            ]))
            rc = cb.main([
                "--arms-config", str(arms_config), "--baseline", "baseline",
                "--output", str(tmp_path / "out"), "--rounds", "1",
                "--metric", "rate=rate:\\s*([0-9.]+)", "--",
                "bench",
            ])
            self.assertEqual(rc, 2)
            self.assertFalse((tmp_path / "out").exists())

    def test_refuses_rccl_arm_when_not_qualified(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            baseline_bin = self._make_fake_binary(tmp_path, 100.0)
            arms_config = tmp_path / "arms.json"
            arms_config.write_text(json.dumps([
                {"name": "baseline", "binary": baseline_bin},
                {"name": "rccl-candidate", "binary": baseline_bin, "requires_rccl": True},
            ]))
            qualification = tmp_path / "cases.jsonl"
            qualification.write_text(json.dumps({
                "topology_id": "other_topology", "compatibility_revision_id": "rev1",
                "classification": "pass",
            }) + "\n")
            rc = cb.main([
                "--arms-config", str(arms_config), "--baseline", "baseline",
                "--output", str(tmp_path / "out"), "--rounds", "1",
                "--metric", "rate=rate:\\s*([0-9.]+)",
                "--qualification-jsonl", str(qualification),
                "--topology-id", "xtx_xtx", "--rccl-version", "2.27.7",
                "--rccl-source-revision", "abc123",
                "--", "bench",
            ])
            self.assertEqual(rc, 1)
            self.assertFalse((tmp_path / "out").exists())

    def test_runs_and_compares_arms_without_rccl(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            baseline_bin = self._make_fake_binary(tmp_path, 100.0)
            candidate_bin = self._make_fake_binary(tmp_path, 110.0)
            arms_config = tmp_path / "arms.json"
            arms_config.write_text(json.dumps([
                {"name": "baseline", "binary": baseline_bin},
                {"name": "candidate", "binary": candidate_bin},
            ]))
            out_dir = tmp_path / "out"
            rc = cb.main([
                "--arms-config", str(arms_config), "--baseline", "baseline",
                "--output", str(out_dir), "--rounds", "6", "--settle-seconds", "0",
                "--metric", "rate=rate:\\s*([0-9.]+)", "--",
                "bench",
            ])
            self.assertEqual(rc, 0)
            summary = json.loads((out_dir / "run.json").read_text(encoding="utf-8"))
            self.assertEqual(len(summary["runs"]), 12)  # 6 rounds x 2 arms
            self.assertIn("candidate", summary["comparisons"])
            effect = summary["comparisons"]["candidate"]["rate"]
            self.assertAlmostEqual(effect["geometric_effect_pct"], 10.0, delta=0.5)

    def test_runs_and_compares_qualified_rccl_arm(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            baseline_bin = self._make_fake_binary(tmp_path, 100.0)
            candidate_bin = self._make_fake_binary(tmp_path, 105.0)
            arms_config = tmp_path / "arms.json"
            arms_config.write_text(json.dumps([
                {"name": "baseline", "binary": baseline_bin},
                {"name": "rccl-candidate", "binary": candidate_bin, "requires_rccl": True},
            ]))
            qualification = tmp_path / "cases.jsonl"
            qualification.write_text(json.dumps({
                "topology_id": "xtx_xtx", "compatibility_revision_id": "abc123",
                "classification": "pass",
            }) + "\n")
            out_dir = tmp_path / "out"
            rc = cb.main([
                "--arms-config", str(arms_config), "--baseline", "baseline",
                "--output", str(out_dir), "--rounds", "2", "--settle-seconds", "0",
                "--metric", "rate=rate:\\s*([0-9.]+)",
                "--qualification-jsonl", str(qualification),
                "--topology-id", "xtx_xtx", "--rccl-version", "2.27.7",
                "--rccl-source-revision", "abc123",
                "--", "bench",
            ])
            self.assertEqual(rc, 0)
            summary = json.loads((out_dir / "run.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["qualification"]["compatibility_revision_id"], "abc123")

    def test_fails_loudly_on_arm_command_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            baseline_bin = self._make_fake_binary(tmp_path, 100.0)
            broken = tmp_path / "broken.py"
            broken.write_text("import sys; sys.exit(3)")
            broken_wrapper = tmp_path / ("broken.bat" if sys.platform.startswith("win") else "broken.sh")
            if sys.platform.startswith("win"):
                broken_wrapper.write_text(f'@"{_python_executable()}" "{broken}" %*\r\n')
            else:
                broken_wrapper.write_text(f'#!/bin/sh\nexec "{_python_executable()}" "{broken}" "$@"\n')
                broken_wrapper.chmod(0o755)
            arms_config = tmp_path / "arms.json"
            arms_config.write_text(json.dumps([
                {"name": "baseline", "binary": baseline_bin},
                {"name": "broken", "binary": str(broken_wrapper)},
            ]))
            out_dir = tmp_path / "out"
            rc = cb.main([
                "--arms-config", str(arms_config), "--baseline", "baseline",
                "--output", str(out_dir), "--rounds", "1", "--settle-seconds", "0",
                "--metric", "rate=rate:\\s*([0-9.]+)", "--",
                "bench",
            ])
            self.assertEqual(rc, 1)
            summary = json.loads((out_dir / "run.json").read_text(encoding="utf-8"))
            self.assertIn("error", summary)


if __name__ == "__main__":
    unittest.main()
